"""
RotorQuant 量化器：Clifford代数旋转量化

基于Clifford代数的旋转量化器，提供高质量的旋转变换。
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional

from .base import BaseQuantizer, BaseProdQuantizer
from ..utils.codebook import LloydMaxCodebook


# ── Clifford代数运算 ─────────────────────────────────────────────────

def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Clifford代数几何积"""
    # 简化实现：3D向量
    # a, b: (..., 3) 作为向量
    # 返回：标量 + 双向量 + 伪标量
    a0, a1, a2 = a.unbind(-1)
    b0, b1, b2 = b.unbind(-1)
    
    # 标量部分
    scalar = a0*b0 + a1*b1 + a2*b2
    
    # 双向量部分
    bivector_01 = a0*b1 - a1*b0
    bivector_02 = a0*b2 - a2*b0
    bivector_12 = a1*b2 - a2*b1
    
    # 伪标量部分
    pseudoscalar = a0*b1*b2 + a1*b2*b0 + a2*b0*b1 - a0*b2*b1 - a1*b0*b2 - a2*b1*b0
    
    return torch.stack([scalar, bivector_01, bivector_02, bivector_12, pseudoscalar], dim=-1)


def make_random_rotor(shape: tuple, device='cpu', seed=None) -> torch.Tensor:
    """生成随机rotor（单位四元数）"""
    gen = torch.Generator(device='cpu')
    if seed is not None:
        gen.manual_seed(seed)
    
    # 生成随机四元数并归一化
    q = torch.randn(*shape, 4, generator=gen).to(device)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def rotor_sandwich(rotor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotor三明治积：R v R^†"""
    # 简化实现：使用四元数运算
    # rotor: (..., 4) as [w, x, y, z]
    # vector: (..., 3) as [v0, v1, v2]
    
    # 将向量转换为纯四元数
    w = torch.zeros_like(vector[..., 0])
    v_quat = torch.stack([w, vector[..., 0], vector[..., 1], vector[..., 2]], dim=-1)
    
    # R v R^†
    # 先计算 R v
    temp = quat_multiply(rotor, v_quat)
    # 再计算 (R v) R^†
    rotor_conj = torch.stack([rotor[..., 0], -rotor[..., 1], -rotor[..., 2], -rotor[..., 3]], dim=-1)
    result = quat_multiply(temp, rotor_conj)
    
    # 提取向量部分
    return result[..., 1:]


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """四元数乘法（简化实现）"""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    
    rw = aw*bw - ax*bx - ay*by - az*bz
    rx = aw*bx + ax*bw + ay*bz - az*by
    ry = aw*by - ax*bz + ay*bw + az*bx
    rz = aw*bz + ax*by - ay*bx + az*bw
    
    return torch.stack([rw, rx, ry, rz], dim=-1)


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


class RotorQuantMSE(BaseQuantizer):
    """RotorQuant MSE量化器（Clifford旋转 + 标量量化）"""
    
    def _initialize(self):
        """初始化rotor和码本"""
        # 3D块 - 对齐3的倍数维度
        self.n_groups = (self.d + 2) // 3  # ceil(d/3)
        self.d_padded = self.n_groups * 3
        
        # Lloyd-Max码本
        cb = LloydMaxCodebook(self.d, self.bit_width)
        self.register_buffer('centroids', cb.centroids.to(self.device))
        
        # 随机rotor（单位四元数）
        rotor = make_random_rotor((self.n_groups,), device=self.device, seed=42)
        self.register_buffer('rotor', rotor)
    
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """嵌入到3D块"""
        pad = self.d_padded - self.d
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:-1], self.n_groups, 3)
    
    def _extract(self, v: torch.Tensor) -> torch.Tensor:
        """从3D块提取向量"""
        flat = v.reshape(*v.shape[:-2], -1)
        return flat[..., :self.d]
    
    def quantize(self, x: torch.Tensor) -> Dict:
        """量化向量"""
        # 分离范数和方向
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        
        # 嵌入到3D块
        v = self._embed(x_unit)
        
        # 应用rotor旋转
        v_rot = rotor_sandwich(self.rotor, v)
        
        # 标量量化
        flat = v_rot.reshape(*v_rot.shape[:-2], -1)
        diffs = flat.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1)
        
        return {
            'indices': indices,
            'norms': norms.squeeze(-1),
            'quantizer_type': 'rotor_mse'
        }
    
    def dequantize(self, compressed_data: Dict) -> torch.Tensor:
        """解量化向量"""
        indices = compressed_data['indices']
        norms = compressed_data['norms']
        
        # 查找码本
        values = self.centroids[indices]
        
        # 重塑为3D块
        v_q = values.reshape(*values.shape[:-1], self.n_groups, 3)
        
        # 应用反向旋转
        rotor_conj = torch.stack([self.rotor[..., 0], -self.rotor[..., 1], 
                                -self.rotor[..., 2], -self.rotor[..., 3]], dim=-1)
        v_recon = rotor_sandwich(rotor_conj, v_q)
        
        # 提取并重新缩放
        x_hat = self._extract(v_recon)
        
        if norms.dim() < x_hat.dim():
            norms = norms.unsqueeze(-1)
        
        return x_hat * norms


class RotorQuantProd(BaseProdQuantizer):
    """RotorQuant完整量化器（MSE + QJL）"""
    
    def _initialize(self):
        """初始化MSE量化器和QJL矩阵"""
        # Stage 1: RotorQuant MSE量化器（b-1比特）
        self.mse_bits = max(self.bit_width - 1, 1)
        self.mse_quantizer = RotorQuantMSE(self.d, self.mse_bits, device=self.device)
        
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
            'quantizer_type': 'rotor_prod'
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