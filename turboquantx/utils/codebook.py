"""
码本工具：Lloyd-Max最优码本生成

实现Lloyd-Max算法，用于生成最优量化码本。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class LloydMaxCodebook:
    """Lloyd-Max最优码本生成器"""
    
    def __init__(self, d: int, bits: int, num_samples: int = 100000, 
                 device: str = "cpu"):
        """
        给定维度和位宽的预计算Lloyd-Max最优码本
        Args:
            d: 向量维度
            bits: 比特数
            num_samples: 训练样本数
            device: 计算设备
        """
        self.d = d
        self.bits = bits
        self.num_levels = 2**bits # 1 << bits  # 2^bits
        self.device = device
        
        # 生成训练数据（高斯分布）
        self._generate_training_data(num_samples)
        
        # 计算最优码本
        self.centroids, self.boundaries = self._compute_optimal_codebook()
    
    def _generate_training_data(self, num_samples: int):
        """生成训练数据（标准高斯分布）"""
        # 假设旋转后坐标服从标准高斯分布
        self.samples = torch.randn(num_samples, self.d, device=self.device)
    
    def _compute_optimal_codebook(self) -> tuple:
        """计算Lloyd-Max最优码本"""
        # 初始化码本（均匀分布）
        min_val = self.samples.min()
        max_val = self.samples.max()
        
        # 均匀初始化码本
        centroids = torch.linspace(min_val, max_val, self.num_levels, 
                                  device=self.device)
        
        # Lloyd-Max迭代
        max_iter = 100
        tolerance = 1e-6
        
        for iter in range(max_iter):
            # 计算边界点
            boundaries = (centroids[:-1] + centroids[1:]) / 2
            
            # 分配样本到最近的码本点
            diffs = self.samples.unsqueeze(-1) - centroids.unsqueeze(0).unsqueeze(0)
            distances = diffs.abs()
            assignments = distances.argmin(dim=-1)
            
            # 更新码本点
            new_centroids = []
            for i in range(self.num_levels):
                mask = (assignments == i)
                if mask.any():
                    # 计算该区间内样本的均值
                    cluster_samples = self.samples[mask]
                    new_centroid = cluster_samples.mean()
                else:
                    # 如果没有样本，保持原值
                    new_centroid = centroids[i]
                new_centroids.append(new_centroid)
            
            new_centroids = torch.stack(new_centroids)
            
            # 检查收敛
            if torch.allclose(centroids, new_centroids, atol=tolerance):
                centroids = new_centroids
                break
            
            centroids = new_centroids
        
        # 最终边界点
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        
        return centroids, boundaries
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """量化向量"""
        diffs = x.unsqueeze(-1) - self.centroids.unsqueeze(0).unsqueeze(0)
        indices = diffs.abs().argmin(dim=-1)
        return indices
    
    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """解量化索引"""
        return self.centroids[indices]
    
    def __repr__(self) -> str:
        return f"LloydMaxCodebook(d={self.d}, bits={self.bits}, levels={self.num_levels})"