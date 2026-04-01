"""
PlanarQuant 量化器：2D平面旋转（Givens旋转）量化

最简单的旋转量化器，使用2D平面旋转，计算效率最高。
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional

from .base import BaseQuantizer, BaseProdQuantizer
from ..utils.codebook import LloydMaxCodebook


# ── 2D旋转数学运算 ──────────────────────────────────────────────

def make_random_rotations(n_groups: int, device='cpu', seed=None) -> torch.Tensor:
    """生成随机2D旋转参数 (cos θ, sin θ)"""
    gen = torch.Generator(device='cpu')
    if seed is not None:
        gen.manual_seed(seed)
    
    # 随机角度 [0, 2π)
    angles = torch.rand(n_groups, generator=gen) * (2 * math.pi)
    angles = angles.to(device)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def rot2_apply(cs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """应用2D旋转到向量对"""
    c = cs[..., 0]
    s = cs[..., 1]
    v0 = v[..., 0]
    v1 = v[..., 1]
    return torch.stack([c * v0 - s * v1, s * v0 + c * v1], dim=-1)


def rot2_inverse(cs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """应用反向2D旋转"""
    c = cs[..., 0]
    s = cs[..., 1]
    v0 = v[..., 0]
    v1 = v[..., 1]
    return torch.stack([c * v0 + s * v1, -s * v0 + c * v1], dim=-1)


def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: Optional[int] = None,
                       device: str = "cpu") -> torch.Tensor:
    """生成QJL随机投影矩阵"""
    if m is None:
        m = d
    
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    
    S = torch.randn(m, d, generator=gen)
    return S.to(device)


class PlanarQuantMSE(BaseQuantizer):
    """PlanarQuant MSE量化器（2D旋转 + 标量量化）"""
    
    def _initialize(self):
        """初始化旋转参数和码本"""
        # 2D块 - 对应对
        self.n_groups = (self.d + 1) // 2  # ceil(d/2)
        self.d_padded = self.n_groups * 2
        
        # Lloyd-Max码本
        cb = LloydMaxCodebook(self.d, self.bit_width)
        self.register_buffer('centroids', cb.centroids.to(self.device))
        
        # 随机2D旋转 (cos θ, sin θ)
        rot = make_random_rotations(self.n_groups, device=self.device, seed=42)
        self.register_buffer('rot2', rot)  # (n_groups, 2)
    
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """嵌入到2D对"""
        pad = self.d_padded - self.d
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:-1], self.n_groups, 2)
    
    def _extract(self, v: torch.Tensor) -> torch.Tensor:
        """从2D对提取向量"""
        flat = v.reshape(*v.shape[:-2], -1)
        return flat[..., :self.d]
    
    def quantize(self, x: torch.Tensor) -> Dict:
        """量化向量"""
        # 分离范数和方向
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        
        # 嵌入到2D对
        v = self._embed(x_unit)
        
        # 旋转每个对
        v_rot = rot2_apply(self.rot2, v)
        
        # 标量量化
        flat = v_rot.reshape(*v_rot.shape[:-2], -1)
        diffs = flat.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1)
        
        return {
            'indices': indices,
            'norms': norms.squeeze(-1),
            'quantizer_type': 'planar_mse'
        }
    
    def dequantize(self, compressed_data: Dict) -> torch.Tensor:
        """解量化向量"""
        indices = compressed_data['indices']
        norms = compressed_data['norms']
        
        # 查找码本
        values = self.centroids[indices]
        
        # 重塑为2D对
        v_q = values.reshape(*values.shape[:-1], self.n_groups, 2)
        
        # 反向旋转
        v_recon = rot2_inverse(self.rot2, v_q)
        
        # 提取并重新缩放
        x_hat = self._extract(v_recon)
        
        if norms.dim() < x_hat.dim():
            norms = norms.unsqueeze(-1)
        
        return x_hat * norms


class PlanarQuantProd(BaseProdQuantizer):
    """PlanarQuant完整量化器（MSE + QJL）"""
    
    def _initialize(self):
        """初始化MSE量化器和QJL矩阵"""
        # Stage 1: PlanarQuant MSE量化器（b-1比特）
        self.mse_bits = max(self.bit_width - 1, 1)
        self.mse_quantizer = PlanarQuantMSE(self.d, self.mse_bits, device=self.device)
        
        # Stage 2: QJL投影矩阵
        self.register_buffer("S", generate_qjl_matrix(self.d, m=self.qjl_dim,
                                                     seed=43, device=self.device))
    
    def quantize(self, x: torch.Tensor) -> Dict:
        """完整量化过程"""
        # Stage 1: MSE量化
        x_hat, mse_data = self.mse_quantizer(x)
        
        # 计算残差
        residual = x - x_hat
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
            'quantizer_type': 'planar_prod'
        }
    
    def dequantize(self, compressed_data: Dict) -> torch.Tensor:
        """解量化MSE部分"""
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