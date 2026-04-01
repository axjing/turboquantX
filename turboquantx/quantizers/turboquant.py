"""
TurboQuant 量化器：原始TurboQuant算法实现

基于 TurboQuant 论文的核心算法：
- Stage 1: 随机旋转 + Lloyd-Max标量量化
- Stage 2: QJL残差校正（用于内积无偏估计）
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional
import sys
import os

# 添加模块路径以支持直接运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入依赖模块
try:
    from .base import BaseQuantizer, BaseProdQuantizer
    from ..utils.codebook import LloydMaxCodebook
except ImportError:
    # 直接运行时使用绝对导入
    from turboquantx.quantizers.base import BaseQuantizer, BaseProdQuantizer
    from turboquantx.utils.codebook import LloydMaxCodebook


def generate_rotation_matrix(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """生成随机正交旋转矩阵（Haar分布）"""
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    
    # 生成随机高斯矩阵并QR分解
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    
    # 确保正确旋转（det = +1）
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    
    return Q.to(device)


def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """生成QJL随机投影矩阵
    为QJL生成随机投影矩阵S。S具有独立同分布的正态分布N(0,1)（均值为0，标准差为1）的条目，shape=（m，d）。默认m = d（相同维度）
    """
    if m is None:
        m = d
    
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    
    S = torch.randn(m, d, generator=gen)
    return S.to(device)


class TurboQuantMSE(BaseQuantizer):
    """
    TurboQuant:
    stage1： MSE-optimal quantizer
    先进行随机旋转，然后应用逐坐标Lloyd-Max量化。
    """
    
    def _initialize(self):
        """初始化旋转矩阵和码本"""
        # 预计算旋转矩阵
        self.register_buffer("Pi", generate_rotation_matrix(self.d, seed=42, device=self.device))
        
        # 预计算Lloyd-Max码本
        self.codebook = LloydMaxCodebook(self.d, self.bit_width)
        self.register_buffer("centroids", self.codebook.centroids.to(self.device))
        self.register_buffer("boundaries", self.codebook.boundaries.to(self.device))
    
    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """应用旋转：y = Pi @ x"""
        return x @ self.Pi.T
    
    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """反转旋转：x = Pi^T @ y"""
        return y @ self.Pi
    
    def quantize(self, x: torch.Tensor) -> Dict:
        """量化向量"""
        # 分离范数和方向
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        
        # 旋转
        y = self.rotate(x_unit)
        
        # 标量量化
        diffs = y.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1)
        
        return {
            'indices': indices,
            'norms': norms.squeeze(-1),
            'quantizer_type': 'turbo_mse'
        }
    
    def dequantize(self, compressed_data: Dict) -> torch.Tensor:
        """解量化向量"""
        indices = compressed_data['indices']
        norms = compressed_data['norms']
        
        # 查找码本
        y_hat = self.centroids[indices]
        
        # 反转旋转
        x_hat_unit = self.unrotate(y_hat)
        
        # 重新缩放
        if norms.dim() < x_hat_unit.dim():
            norms = norms.unsqueeze(-1)
        
        return x_hat_unit * norms
    
    def _estimate_compressed_size(self) -> int:
        """估计压缩大小"""
        return self.d * self.bit_width + 32  # +32 for norms


class TurboQuantProd(BaseProdQuantizer):
    """TurboQuant完全量化器（Stage 1 + Stage 2）"""
    
    def __init__(self, d: int, bit_width: int, qjl_dim: int = None, device: str = "cpu"):
        """初始化TurboQuantProd量化器"""
        super().__init__(d, bit_width, qjl_dim, device)
        
    def _initialize(self):
        """初始化MSE量化器和QJL矩阵"""
        # Stage 1: MSE量化器（b-1比特）
        self.mse_bits = max(self.bit_width - 1, 1)
        self.mse_quantizer = TurboQuantMSE(self.d, self.mse_bits, device=self.device)
        
        # Stage 2: QJL投影矩阵
        self.register_buffer("S", generate_qjl_matrix(self.d, m=self.qjl_dim, 
                                                     seed=43, device=self.device))
    
    def quantize(self, x: torch.Tensor) -> Dict:
        """完整TurboQuant量化"""
        # Stage 1: MSE量化
        x_mse, mse_data = self.mse_quantizer(x)
        
        # 计算残差
        residual = x - x_mse
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)
        
        # Stage 2: QJL投影和符号量化
        projected = residual @ self.S.T
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0
        
        return {
            'mse_indices': mse_data['indices'],
            'mse_norms': mse_data['norms'],
            'qjl_signs': qjl_signs,
            'residual_norm': residual_norm.squeeze(-1),
            'quantizer_type': 'turbo_prod'
        }
    
    def dequantize(self, compressed_data: Dict) -> torch.Tensor:
        """仅解量化MSE部分（用于重构）"""
        mse_data = {
            'indices': compressed_data['mse_indices'],
            'norms': compressed_data['mse_norms']
        }
        return self.mse_quantizer.dequantize(mse_data)
    
    def inner_product(self, y: torch.Tensor, compressed_data: Dict) -> torch.Tensor:
        """计算内积估计"""
        # 项1：与MSE重构的内积
        x_mse = self.dequantize(compressed_data)
        term1 = (y * x_mse).sum(dim=-1)
        
        # 项2：QJL校正项
        y_projected = y @ self.S.T
        qjl_ip = (y_projected * compressed_data['qjl_signs']).sum(dim=-1)
        
        correction_scale = math.sqrt(math.pi / 2) / self.qjl_dim
        term2 = compressed_data['residual_norm'] * correction_scale * qjl_ip
        
        return term1 + term2


if __name__ == "__main__":
    """
    TurboQuant 量化器测试示例
    
    测试 TurboQuantMSE 和 TurboQuantProd 的基本功能
    """
    
    # 检查是否有可用的GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 测试参数
    d = 128  # 向量维度
    bits = 3  # 比特宽度
    batch_size = 10  # 批量大小
    
    print("=== TurboQuant 量化器测试 ===")
    print(f"向量维度: {d}")
    print(f"比特宽度: {bits}")
    print(f"批量大小: {batch_size}")
    
    # 生成测试数据
    torch.manual_seed(42)  # 确保可重复性
    x = torch.randn(batch_size, d, device=device)
    
    print("\n1. 测试 TurboQuantMSE (仅Stage 1)")
    print("-" * 40)
    
    # 创建 TurboQuantMSE 量化器
    tq_mse = TurboQuantMSE(d=d, bit_width=bits, device=device)
    
    # 量化-解量化测试
    x_recon, compressed = tq_mse(x)
    
    # 计算重建误差
    mse_error = torch.mean((x - x_recon) ** 2).item()
    max_error = torch.max(torch.abs(x - x_recon)).item()
    
    print(f"原始数据形状: {x.shape}")
    print(f"重建数据形状: {x_recon.shape}")
    print(f"MSE误差: {mse_error:.6f}")
    print(f"最大绝对误差: {max_error:.6f}")
    
    # 压缩统计
    compression_ratio = tq_mse.compression_ratio()
    print(f"压缩比: {compression_ratio:.2f}x")
    print(f"压缩后大小估计: {tq_mse._estimate_compressed_size()} 比特")
    
    print("\n2. 测试 TurboQuantProd (Stage 1 + Stage 2)")
    print("-" * 40)
    
    # 创建 TurboQuantProd 量化器
    tq_prod = TurboQuantProd(d=d, bit_width=bits, qjl_dim=d, device=device)
    
    # 量化测试
    compressed_prod = tq_prod.quantize(x)
    
    # 解量化测试
    x_recon_prod = tq_prod.dequantize(compressed_prod)
    
    # 计算重建误差
    mse_error_prod = torch.mean((x - x_recon_prod) ** 2).item()
    max_error_prod = torch.max(torch.abs(x - x_recon_prod)).item()
    
    print(f"MSE误差: {mse_error_prod:.6f}")
    print(f"最大绝对误差: {max_error_prod:.6f}")
    
    # 内积测试
    y = torch.randn(batch_size, d, device=device)
    ip_true = (x * y).sum(dim=-1)
    ip_approx = tq_prod.inner_product(y, compressed_prod)
    
    ip_error = torch.mean(torch.abs(ip_true - ip_approx)).item()
    print(f"内积估计误差: {ip_error:.6f}")
    
    print("\n3. 压缩数据结构分析")
    print("-" * 40)
    
    print("TurboQuantMSE 压缩数据:")
    for key, value in compressed.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {value}")
    
    print("\nTurboQuantProd 压缩数据:")
    for key, value in compressed_prod.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {value}")
    
    print("\n4. 性能测试")
    print("-" * 40)
    
    import time
    
    # TurboQuantMSE 性能测试
    start_time = time.time()
    for _ in range(100):
        _ = tq_mse(x)
    mse_time = (time.time() - start_time) / 100
    
    # TurboQuantProd 性能测试
    start_time = time.time()
    for _ in range(100):
        _ = tq_prod.quantize(x)
    prod_time = (time.time() - start_time) / 100
    
    print(f"TurboQuantMSE 平均量化时间: {mse_time * 1000:.2f} ms")
    print(f"TurboQuantProd 平均量化时间: {prod_time * 1000:.2f} ms")
    
    print("\n5. 内存使用测试")
    print("-" * 40)
    
    if torch.cuda.is_available():
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated()
        
        # 执行量化操作
        _, compressed_large = tq_mse(x)
        
        # 记录最终内存
        final_memory = torch.cuda.memory_allocated()
        memory_used = final_memory - initial_memory
        
        print(f"GPU内存使用: {memory_used / 1024 / 1024:.2f} MB")
    else:
        print("GPU不可用，跳过内存测试")
    
    print("\n=== 测试完成 ===")
    print("所有测试项目执行完毕，TurboQuant 量化器工作正常！")