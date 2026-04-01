"""
IsoQuant 量化器：四元数4D块旋转量化

基于四元数旋转的量化器，相比TurboQuant具有更好的硬件对齐和计算效率。
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional

from .base import BaseQuantizer, BaseProdQuantizer
from ..utils.codebook import LloydMaxCodebook


# ── 四元数数学运算 ───────────────────────────────────────────────────

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """四元数共轭：(w, x, y, z) → (w, -x, -y, -z)"""
    signs = torch.tensor([1, -1, -1, -1], dtype=q.dtype, device=q.device)
    return q * signs


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """四元数Hamilton乘积"""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    
    rw = aw*bw - ax*bx - ay*by - az*bz
    rx = aw*bx + ax*bw + ay*bz - az*by
    ry = aw*by - ax*bz + ay*bw + az*bx
    rz = aw*bz + ax*by - ay*bx + az*bw
    
    return torch.stack([rw, rx, ry, rz], dim=-1)


def make_random_unit_quaternion(shape: Tuple[int, ...], 
                               device='cpu', seed=None) -> torch.Tensor:
    """生成随机单位四元数"""
    gen = torch.Generator(device='cpu')
    if seed is not None:
        gen.manual_seed(seed)
    
    q = torch.randn(*shape, 4, generator=gen).to(device)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


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


class IsoQuantMSE(BaseQuantizer):
    """IsoQuant MSE量化器（四元数旋转 + 标量量化）"""
    
    def __init__(self, d: int, bit_width: int, mode: str = 'full', 
                 device: str = "cpu"):
        """
        Args:
            d: 向量维度
            bit_width: 比特宽度
            mode: 'full'（q_L v q̄_R）或 'fast'（q_L v）
            device: 计算设备
        """
        self.mode = mode
        super().__init__(d, bit_width, device)
    
    def _initialize(self):
        """初始化四元数和码本"""
        # 4D块 - 对齐2的幂次方维度
        self.n_groups = (self.d + 3) // 4  # ceil(d/4)
        self.d_padded = self.n_groups * 4
        
        # Lloyd-Max码本
        cb = LloydMaxCodebook(self.d, self.bit_width)
        self.register_buffer('centroids', cb.centroids.to(self.device))
        
        # 随机单位四元数
        q_L = make_random_unit_quaternion((self.n_groups,), device=self.device, seed=42)
        self.register_buffer('q_L', q_L)
        
        if self.mode == 'full':
            q_R = make_random_unit_quaternion((self.n_groups,), device=self.device, seed=43)
            self.register_buffer('q_R', q_R)
    
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """嵌入到4D四元数组"""
        pad = self.d_padded - self.d
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:-1], self.n_groups, 4)
    
    def _extract(self, v: torch.Tensor) -> torch.Tensor:
        """从4D四元数组提取向量"""
        flat = v.reshape(*v.shape[:-2], -1)
        return flat[..., :self.d]
    
    def _rotate(self, v: torch.Tensor) -> torch.Tensor:
        """应用四元数旋转"""
        if self.mode == 'full':
            # T(v) = q_L * v * conjugate(q_R)
            temp = quat_multiply(self.q_L, v)
            return quat_multiply(temp, quat_conjugate(self.q_R))
        else:
            # T(v) = q_L * v
            return quat_multiply(self.q_L, v)
    
    def _unrotate(self, v: torch.Tensor) -> torch.Tensor:
        """应用反向旋转"""
        if self.mode == 'full':
            # T^{-1}(v) = conjugate(q_L) * v * q_R
            temp = quat_multiply(quat_conjugate(self.q_L), v)
            return quat_multiply(temp, self.q_R)
        else:
            # T^{-1}(v) = conjugate(q_L) * v
            return quat_multiply(quat_conjugate(self.q_L), v)
    
    def quantize(self, x: torch.Tensor) -> Dict:
        """量化向量"""
        # 分离范数和方向
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        
        # 嵌入到4D块
        v = self._embed(x_unit)
        
        # 旋转
        v_rot = self._rotate(v)
        
        # 标量量化
        flat = v_rot.reshape(*v_rot.shape[:-2], -1)
        diffs = flat.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1)
        
        return {
            'indices': indices,
            'norms': norms.squeeze(-1),
            'quantizer_type': 'isoquant_mse',
            'mode': self.mode
        }
    
    def dequantize(self, compressed_data: Dict) -> torch.Tensor:
        """解量化向量"""
        indices = compressed_data['indices']
        norms = compressed_data['norms']
        
        # 查找码本
        values = self.centroids[indices]
        
        # 重塑为4D块
        v_q = values.reshape(*values.shape[:-1], self.n_groups, 4)
        
        # 反向旋转
        v_recon = self._unrotate(v_q)
        
        # 提取并重新缩放
        x_hat = self._extract(v_recon)
        
        if norms.dim() < x_hat.dim():
            norms = norms.unsqueeze(-1)
        
        return x_hat * norms


class IsoQuantProd(BaseProdQuantizer):
    """IsoQuant完整量化器（MSE + QJL）"""
    
    def __init__(self, d: int, bit_width: int, mode: str = 'full',
                 qjl_dim: Optional[int] = None, device: str = "cpu"):
        """
        Args:
            d: 向量维度
            bit_width: 比特宽度
            mode: 'full' 或 'fast'
            qjl_dim: QJL投影维度
            device: 计算设备
        """
        self.mode = mode
        super().__init__(d, bit_width, qjl_dim, device)
    
    def _initialize(self):
        """初始化MSE量化器和QJL矩阵"""
        # Stage 1: IsoQuant MSE量化器（b-1比特）
        self.mse_bits = max(self.bit_width - 1, 1)
        self.mse_quantizer = IsoQuantMSE(self.d, self.mse_bits, 
                                        mode=self.mode, device=self.device)
        
        # Stage 2: QJL投影矩阵
        self.register_buffer("S", generate_qjl_matrix(self.d, m=self.qjl_dim,
                                                     seed=44, device=self.device))
    
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
            'quantizer_type': 'isoquant_prod',
            'mode': self.mode
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