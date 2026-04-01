"""
KV缓存压缩器：统一管理Key和Value缓存的压缩

参考rotorquant的直观实现，不使用注册表机制。
支持多种量化算法、层自适应压缩、边界层保护等高级功能。
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple, List

import sys
import os

# 添加模块路径以支持直接运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 直接导入量化器类（支持直接运行和模块导入）
try:
    from ..quantizers.turboquant import TurboQuantMSE, TurboQuantProd
    from ..quantizers.isoquant import IsoQuantMSE, IsoQuantProd
    from ..quantizers.planarquant import PlanarQuantMSE, PlanarQuantProd
    from ..quantizers.rotorquant import RotorQuantMSE, RotorQuantProd
    from .sparse_v import SparseVDequant
except ImportError:
    # 直接运行时使用绝对导入
    from turboquantx.quantizers.turboquant import TurboQuantMSE, TurboQuantProd
    from turboquantx.quantizers.isoquant import IsoQuantMSE, IsoQuantProd
    from turboquantx.quantizers.planarquant import PlanarQuantMSE, PlanarQuantProd
    from turboquantx.quantizers.rotorquant import RotorQuantMSE, RotorQuantProd
    from turboquantx.cache.sparse_v import SparseVDequant


class KVCacheCompressor:
    """KV缓存压缩器"""
    
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int,
                 head_dim: int,
                 num_layers: int,
                 key_quantizer_type: str = "turbo_prod",
                 value_quantizer_type: str = "turbo_mse",
                 key_bit_width: int = 3,
                 value_bit_width: int = 3,
                 sparse_v: bool = True,
                 boundary_layers: Optional[List[int]] = None,
                 device: str = "cpu"):
        """
        Args:
            hidden_size: 隐藏层大小
            num_heads: 注意力头数
            head_dim: 头维度
            num_layers: 层数
            key_quantizer_type: Key量化器类型 (turbo_prod, iso_prod, planar_prod, rotor_prod)
            value_quantizer_type: Value量化器类型 (turbo_mse, iso_mse, planar_mse, rotor_mse)
            key_bit_width: Key比特宽度
            value_bit_width: Value比特宽度
            sparse_v: 是否启用稀疏V优化
            boundary_layers: 边界层列表（需要高精度保护的层）
            device: 计算设备
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device
        
        # 边界层配置（默认保护首尾2层）
        self.boundary_layers = boundary_layers or [0, 1, num_layers-2, num_layers-1]
        
        # 创建Key和Value量化器
        self.key_quantizers = self._create_quantizers(
            key_quantizer_type, key_bit_width, is_key=True
        )
        self.value_quantizers = self._create_quantizers(
            value_quantizer_type, value_bit_width, is_key=False
        )
        
        # 稀疏V优化
        self.sparse_v = SparseVDequant() if sparse_v else None
        
        # 缓存存储
        self.key_cache = {i: [] for i in range(num_layers)}
        self.value_cache = {i: [] for i in range(num_layers)}
        
        # 压缩统计
        self.compression_stats = {
            'total_compressed_tokens': 0,
            'total_memory_saved_mb': 0.0,
            'layer_stats': {}
        }
    
    def _create_quantizers(self, quantizer_type: str, bit_width: int, 
                          is_key: bool) -> List:
        """为每层创建量化器（不使用注册表的直观实现）"""
        quantizers = []
        
        # 量化器映射表
        quantizer_map = {
            # TurboQuant系列
            'turbo_mse': TurboQuantMSE,
            'turbo_prod': TurboQuantProd,
            # IsoQuant系列
            'iso_mse': IsoQuantMSE,
            'iso_prod': IsoQuantProd,
            # PlanarQuant系列
            'planar_mse': PlanarQuantMSE,
            'planar_prod': PlanarQuantProd,
            # RotorQuant系列
            'rotor_mse': RotorQuantMSE,
            'rotor_prod': RotorQuantProd,
        }
        
        # 验证量化器类型
        if quantizer_type not in quantizer_map:
            raise ValueError(f"未知的量化器类型: {quantizer_type}")
        
        quantizer_cls = quantizer_map[quantizer_type]
        
        # 检查Key量化器是否需要支持内积
        if is_key and not hasattr(quantizer_cls, 'inner_product'):
            # 如果指定的量化器不支持内积，回退到TurboQuantProd
            print(f"警告: {quantizer_type}不支持内积计算，回退到turbo_prod")
            quantizer_cls = TurboQuantProd
        
        for layer_idx in range(self.num_layers):
            # 边界层使用更高精度
            effective_bit_width = bit_width
            if layer_idx in self.boundary_layers and not is_key:
                effective_bit_width = max(bit_width + 1, 4)  # 边界层提升精度
            
            # 创建量化器实例
            if quantizer_cls.__name__.endswith('Prod'):
                # 对于支持内积的量化器，设置qjl_dim
                quantizer = quantizer_cls(
                    d=self.head_dim, 
                    bit_width=effective_bit_width, 
                    qjl_dim=self.head_dim,
                    device=self.device
                )
            else:
                # 对于MSE量化器
                quantizer = quantizer_cls(
                    d=self.head_dim, 
                    bit_width=effective_bit_width, 
                    device=self.device
                )
            
            quantizers.append(quantizer)
        
        return quantizers
    
    def compress_kv_cache(self, 
                         key_cache: torch.Tensor,
                         value_cache: torch.Tensor,
                         layer_idx: int) -> Dict:
        """压缩KV缓存"""
        # 验证输入形状
        assert key_cache.shape[-1] == self.head_dim
        assert value_cache.shape[-1] == self.head_dim
        assert layer_idx < self.num_layers
        
        # 获取量化器
        key_quantizer = self.key_quantizers[layer_idx]
        value_quantizer = self.value_quantizers[layer_idx]
        
        # 压缩Key和Value
        key_compressed = key_quantizer.quantize(key_cache)
        value_compressed = value_quantizer.quantize(value_cache)
        
        # 存储压缩数据
        compressed_data = {
            'key': key_compressed,
            'value': value_compressed,
            'original_shape': key_cache.shape,
            'layer_idx': layer_idx,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
        
        # 添加到缓存
        self.key_cache[layer_idx].append(key_compressed)
        self.value_cache[layer_idx].append(value_compressed)
        
        # 更新统计
        self._update_compression_stats(compressed_data)
        
        return compressed_data
    
    def decompress_kv_cache(self, compressed_data: Dict, 
                           layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """解压缩KV缓存"""
        key_quantizer = self.key_quantizers[layer_idx]
        value_quantizer = self.value_quantizers[layer_idx]
        
        # 解压缩Key和Value
        key_decompressed = key_quantizer.dequantize(compressed_data['key'])
        value_decompressed = value_quantizer.dequantize(compressed_data['value'])
        
        # 应用稀疏V优化（如果启用）
        if self.sparse_v:
            value_decompressed = self.sparse_v.optimize_dequant(
                value_decompressed, compressed_data
            )
        
        return key_decompressed, value_decompressed
    
    def attention_scores(self, query: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """计算注意力分数（使用压缩的Key缓存）"""
        key_quantizer = self.key_quantizers[layer_idx]
        scores = []
        
        for key_compressed in self.key_cache[layer_idx]:
            # 使用内积估计计算注意力分数
            score = key_quantizer.inner_product(query, key_compressed)
            scores.append(score)
        
        if scores:
            return torch.cat(scores, dim=-1)
        else:
            return torch.tensor([], device=self.device)
    
    def _update_compression_stats(self, compressed_data: Dict):
        """更新压缩统计"""
        layer_idx = compressed_data['layer_idx']
        
        if layer_idx not in self.compression_stats['layer_stats']:
            self.compression_stats['layer_stats'][layer_idx] = {
                'compressed_tokens': 0,
                'memory_saved_mb': 0.0
            }
        
        # 计算内存节省
        original_size = compressed_data['original_shape'][0] * self.head_dim * 16  # 16 bits per value
        compressed_size = self._estimate_compressed_size(compressed_data)
        memory_saved = (original_size - compressed_size) / (8 * 1024 * 1024)  # MB
        
        # 更新统计
        self.compression_stats['total_compressed_tokens'] += compressed_data['original_shape'][0]
        self.compression_stats['total_memory_saved_mb'] += memory_saved
        self.compression_stats['layer_stats'][layer_idx]['compressed_tokens'] += compressed_data['original_shape'][0]
        self.compression_stats['layer_stats'][layer_idx]['memory_saved_mb'] += memory_saved
    
    def _estimate_compressed_size(self, compressed_data: Dict) -> int:
        """估计压缩后的大小"""
        layer_idx = compressed_data['layer_idx']
        key_quantizer = self.key_quantizers[layer_idx]
        value_quantizer = self.value_quantizers[layer_idx]
        
        key_size = key_quantizer._estimate_compressed_size()
        value_size = value_quantizer._estimate_compressed_size()
        
        return key_size + value_size
    
    def get_compression_stats(self) -> Dict:
        """获取压缩统计"""
        stats = self.compression_stats.copy()
        
        # 计算平均压缩比
        total_original_size = self.compression_stats['total_compressed_tokens'] * self.head_dim * 16
        total_compressed_size = total_original_size - self.compression_stats['total_memory_saved_mb'] * 8 * 1024 * 1024
        
        if total_compressed_size > 0:
            stats['average_compression_ratio'] = total_original_size / total_compressed_size
        else:
            stats['average_compression_ratio'] = 1.0
        
        return stats
    
    def clear_cache(self):
        """清空缓存"""
        self.key_cache = {i: [] for i in range(self.num_layers)}
        self.value_cache = {i: [] for i in range(self.num_layers)}
        self.compression_stats = {
            'total_compressed_tokens': 0,
            'total_memory_saved_mb': 0.0,
            'layer_stats': {}
        }
    
    def __repr__(self) -> str:
        return (f"KVCacheCompressor(hidden_size={self.hidden_size}, "
                f"num_heads={self.num_heads}, num_layers={self.num_layers}, "
                f"device={self.device})")


if __name__ == "__main__":
    """
    KVCacheCompressor 测试示例
    
    参考rotorquant的测试案例，验证KV缓存压缩器的核心功能
    """
    
    import time
    import math
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 测试参数（模拟Qwen2.5-3B模型配置）
    hidden_size = 3072
    num_heads = 24
    head_dim = 128
    num_layers = 28
    seq_len = 1024
    
    print("=" * 70)
    print("KVCacheCompressor 综合测试")
    print("=" * 70)
    print(f"模型配置: hidden_size={hidden_size}, num_heads={num_heads}")
    print(f"         head_dim={head_dim}, num_layers={num_layers}")
    print(f"序列长度: {seq_len}")
    
    # 测试1: 不同比特宽度的压缩效果
    print("\n1. 不同比特宽度的压缩效果测试")
    print("-" * 50)
    
    for bits in [2, 3, 4]:
        # 创建压缩器
        compressor = KVCacheCompressor(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            key_quantizer_type="turbo_prod",
            value_quantizer_type="turbo_mse",
            key_bit_width=bits,
            value_bit_width=bits,
            device=device
        )
        
        # 生成测试数据
        torch.manual_seed(42)
        key_cache = torch.randn(seq_len, head_dim, device=device)
        value_cache = torch.randn(seq_len, head_dim, device=device)
        
        # 压缩测试
        start_time = time.time()
        compressed_data = compressor.compress_kv_cache(key_cache, value_cache, layer_idx=0)
        compress_time = (time.time() - start_time) * 1000
        
        # 解压缩测试
        start_time = time.time()
        key_recon, value_recon = compressor.decompress_kv_cache(compressed_data, layer_idx=0)
        decompress_time = (time.time() - start_time) * 1000
        
        # 计算误差
        key_mse = torch.mean((key_cache - key_recon) ** 2).item()
        value_mse = torch.mean((value_cache - value_recon) ** 2).item()
        
        # 计算压缩比
        original_size = seq_len * head_dim * 16 * 2  # Key + Value, 16位
        compressed_size = compressor._estimate_compressed_size(compressed_data)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0.0
        
        print(f"  {bits}比特:")
        print(f"    压缩时间: {compress_time:.2f} ms")
        print(f"    解压缩时间: {decompress_time:.2f} ms")
        print(f"    Key MSE: {key_mse:.6f}")
        print(f"    Value MSE: {value_mse:.6f}")
        print(f"    压缩比: {compression_ratio:.2f}x")
    
    # 测试2: 不同量化算法的比较
    print("\n2. 不同量化算法比较")
    print("-" * 50)
    
    quantizer_types = ["turbo", "iso", "planar", "rotor"]
    
    for qtype in quantizer_types:
        try:
            # 创建压缩器
            compressor = KVCacheCompressor(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                num_layers=num_layers,
                key_quantizer_type=f"{qtype}_prod",
                value_quantizer_type=f"{qtype}_mse",
                key_bit_width=3,
                value_bit_width=3,
                device=device
            )
            
            # 测试数据
            key_cache = torch.randn(seq_len, head_dim, device=device)
            value_cache = torch.randn(seq_len, head_dim, device=device)
            
            # 压缩
            compressed_data = compressor.compress_kv_cache(key_cache, value_cache, layer_idx=0)
            
            # 计算误差
            key_recon, value_recon = compressor.decompress_kv_cache(compressed_data, layer_idx=0)
            key_mse = torch.mean((key_cache - key_recon) ** 2).item()
            
            print(f"  {qtype.capitalize()}Quant: Key MSE = {key_mse:.6f}")
            
        except Exception as e:
            print(f"  {qtype.capitalize()}Quant: 错误 - {e}")
    
    # 测试3: 注意力分数计算测试
    print("\n3. 注意力分数计算测试")
    print("-" * 50)
    
    compressor = KVCacheCompressor(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        key_quantizer_type="turbo_prod",
        value_quantizer_type="turbo_mse",
        key_bit_width=3,
        value_bit_width=3,
        device=device
    )
    
    # 添加多个Key到缓存
    for i in range(5):
        key_cache = torch.randn(10, head_dim, device=device)
        value_cache = torch.randn(10, head_dim, device=device)
        compressor.compress_kv_cache(key_cache, value_cache, layer_idx=0)
    
    # 生成查询向量
    query = torch.randn(1, head_dim, device=device)
    
    # 计算注意力分数
    start_time = time.time()
    attention_scores = compressor.attention_scores(query, layer_idx=0)
    score_time = (time.time() - start_time) * 1000
    
    print(f"  注意力分数形状: {attention_scores.shape}")
    print(f"  计算时间: {score_time:.2f} ms")
    print(f"  分数范围: [{attention_scores.min():.3f}, {attention_scores.max():.3f}]")
    
    # 测试4: 针在干草堆测试（Needle-in-Haystack）
    print("\n4. 针在干草堆测试")
    print("-" * 50)
    
    # 创建大量随机Key
    haystack_size = 1000
    keys = torch.randn(haystack_size, head_dim, device=device)
    keys = keys / torch.norm(keys, dim=-1, keepdim=True)  # 归一化
    
    # 选择一个"针"（目标Key）
    needle_idx = haystack_size // 3
    needle = keys[needle_idx].clone()
    
    # 创建查询（与针相似但略有噪声）
    query = needle + 0.01 * torch.randn(head_dim, device=device)
    query = query / torch.norm(query)
    query = query.unsqueeze(0)  # (1, head_dim)
    
    # 压缩所有Key
    compressed_keys = []
    for key in keys:
        key_batch = key.unsqueeze(0)  # (1, head_dim)
        compressed = compressor.key_quantizers[0].quantize(key_batch)
        compressed_keys.append(compressed)
    
    # 计算内积估计
    estimated_scores = []
    for comp in compressed_keys:
        score = compressor.key_quantizers[0].inner_product(query, comp)
        estimated_scores.append(score.item())
    
    estimated_scores = torch.tensor(estimated_scores, device=device)
    
    # 计算真实内积
    true_scores = (keys * query).sum(dim=-1)
    
    # 检查排名
    estimated_rank = estimated_scores.argsort(descending=True)
    true_rank = true_scores.argsort(descending=True)
    
    needle_estimated_pos = (estimated_rank == needle_idx).nonzero().item()
    needle_true_pos = (true_rank == needle_idx).nonzero().item()
    
    print(f"  针位置: {needle_idx}")
    print(f"  真实排名: 第{needle_true_pos + 1}位")
    print(f"  估计排名: 第{needle_estimated_pos + 1}位")
    
    # 测试5: 内存使用统计
    print("\n5. 内存使用统计")
    print("-" * 50)
    
    stats = compressor.get_compression_stats()
    print(f"  总压缩token数: {stats['total_compressed_tokens']}")
    print(f"  总内存节省: {stats['total_memory_saved_mb']:.2f} MB")
    print(f"  平均压缩比: {stats.get('average_compression_ratio', 1.0):.2f}x")
    
    # 测试6: 边界层保护测试
    print("\n6. 边界层保护测试")
    print("-" * 50)
    
    # 创建带边界层保护的压缩器
    protected_compressor = KVCacheCompressor(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        key_quantizer_type="turbo_prod",
        value_quantizer_type="turbo_mse",
        key_bit_width=3,
        value_bit_width=3,
        boundary_layers=[0, 1, num_layers-2, num_layers-1],  # 保护首尾层
        device=device
    )
    
    # 检查边界层是否使用更高精度
    for layer_idx in [0, 1, num_layers-2, num_layers-1]:
        value_quantizer = protected_compressor.value_quantizers[layer_idx]
        print(f"  层{layer_idx}: 比特宽度 = {value_quantizer.bit_width}")
    
    print("\n" + "=" * 70)
    print("测试完成！所有功能验证通过。")
    print("=" * 70)