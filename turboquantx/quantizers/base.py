"""
量化器基类：定义统一的量化器接口

所有量化器必须继承此基类并实现统一API，确保模块化设计。
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class BaseQuantizer(nn.Module, ABC):
    """量化器基类，定义统一的量化器接口。"""
    
    def __init__(self, d: int, bit_width: int, device: str = "cpu"):
        """
        初始化量化器
        
        Args:
            d: 向量维度
            bit_width: 每维比特数
            device: 计算设备
        """
        super().__init__()
        self.d = d
        self.bit_width = bit_width
        self.device = device
        
        # 量化器配置
        self._config = {
            'd': d,
            'bit_width': bit_width,
            'device': device,
            'quantizer_type': self.__class__.__name__
        }
        
        # 初始化量化器
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """子类必须实现的初始化方法"""
        pass
    
    @abstractmethod
    def quantize(self, x: torch.Tensor) -> Dict:
        """
        量化向量
        
        Args:
            x: 输入向量 (batch, d) 或 (d,)
            
        Returns:
            压缩表示字典
        """
        pass
    
    @abstractmethod
    def dequantize(self, compressed_data: Dict) -> torch.Tensor:
        """
        解量化向量
        
        Args:
            compressed_data: 压缩数据字典
            
        Returns:
            重构向量
        """
        pass
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        完整的量化-解量化循环
        
        Args:
            x: 输入向量
            
        Returns:
            (重构向量, 压缩数据)
        """
        compressed = self.quantize(x)
        reconstructed = self.dequantize(compressed)
        return reconstructed, compressed
    
    def compression_ratio(self, original_bits: int = 16) -> float:
        """
        计算压缩比
        
        Args:
            original_bits: 原始精度（默认16位）
            
        Returns:
            压缩比（原始大小/压缩大小）
        """
        original_size = self.d * original_bits
        compressed_size = self._estimate_compressed_size()
        return original_size / compressed_size if compressed_size > 0 else 0.0
    
    def _estimate_compressed_size(self) -> int:
        """估计压缩后的大小（比特数）"""
        # 基础实现：每维bit_width比特
        return self.d * self.bit_width
    
    def get_config(self) -> Dict:
        """获取量化器配置"""
        return self._config.copy()
    
    def to_device(self, device: str):
        """移动量化器到指定设备"""
        self.device = device
        self.to(device)
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(d={self.d}, bits={self.bit_width}, device={self.device})"


class BaseProdQuantizer(BaseQuantizer):
    """支持内积计算的量化器基类（用于Key缓存）"""
    
    def __init__(self, d: int, bit_width: int, qjl_dim: Optional[int] = None, 
                 device: str = "cpu"):
        # 先设置qjl_dim属性，再调用父类初始化
        self.qjl_dim = qjl_dim or d
        super().__init__(d, bit_width, device)
        
    @abstractmethod
    def inner_product(self, y: torch.Tensor, compressed_data: Dict) -> torch.Tensor:
        """
        计算内积（用于注意力计算）
        
        Args:
            y: 查询向量 (batch, d) 或 (d,)
            compressed_data: 压缩的Key数据
            
        Returns:
            内积估计值
        """
        pass
    
    def _estimate_compressed_size(self) -> int:
        """估计包含QJL的压缩大小"""
        # MSE部分 + QJL部分 + 残差范数
        mse_bits = (self.bit_width - 1) * self.d if self.bit_width > 1 else self.d
        qjl_bits = self.qjl_dim  # 1 bit per dimension
        norm_bits = 32  # float32 for residual norm
        return mse_bits + qjl_bits + norm_bits