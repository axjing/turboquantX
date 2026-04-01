"""
后端优化器：硬件感知优化，自动选择最优后端

支持 Triton/CUDA/Metal 三类底层加速技术，自动检测硬件并选择最优后端。
"""

import torch
from enum import Enum
from typing import Optional, Dict


class BackendType(Enum):
    """后端类型枚举"""
    CPU = "cpu"
    CUDA = "cuda"
    TRITON = "triton"
    METAL = "metal"
    AUTO = "auto"


class BackendOptimizer:
    """后端优化器"""
    
    def __init__(self):
        self.available_backends = self._detect_available_backends()
        self.optimal_backend = self._select_optimal_backend()
    
    def _detect_available_backends(self) -> Dict[BackendType, bool]:
        """检测可用后端"""
        backends = {
            BackendType.CPU: True,  # CPU总是可用
            BackendType.CUDA: torch.cuda.is_available(),
            BackendType.TRITON: self._check_triton_available(),
            BackendType.METAL: self._check_metal_available(),
        }
        return backends
    
    def _check_triton_available(self) -> bool:
        """检查Triton是否可用"""
        try:
            import triton
            return True
        except ImportError:
            return False
    
    def _check_metal_available(self) -> bool:
        """检查Metal是否可用（Apple Silicon）"""
        # 简化检测：检查是否为macOS和M系列芯片
        import platform
        if platform.system() != "Darwin":
            return False
        
        # 检查是否有M系列芯片
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            cpu_info = result.stdout.lower()
            return 'm1' in cpu_info or 'm2' in cpu_info or 'm3' in cpu_info or 'm4' in cpu_info
        except:
            return False
    
    def _select_optimal_backend(self) -> BackendType:
        """选择最优后端"""
        # 优先级：Triton > CUDA > Metal > CPU
        if self.available_backends[BackendType.TRITON]:
            return BackendType.TRITON
        elif self.available_backends[BackendType.CUDA]:
            return BackendType.CUDA
        elif self.available_backends[BackendType.METAL]:
            return BackendType.METAL
        else:
            return BackendType.CPU
    
    def get_optimal_backend(self) -> BackendType:
        """获取最优后端"""
        return self.optimal_backend
    
    def get_backend_info(self) -> Dict:
        """获取后端信息"""
        info = {
            'optimal_backend': self.optimal_backend.value,
            'available_backends': {k.value: v for k, v in self.available_backends.items()},
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        # 添加设备信息
        if torch.cuda.is_available():
            info['cuda_devices'] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
        
        return info
    
    def optimize_for_backend(self, 
                           quantizer_config: Dict,
                           backend: Optional[BackendType] = None) -> Dict:
        """根据后端优化量化器配置"""
        if backend is None:
            backend = self.optimal_backend
        
        optimized_config = quantizer_config.copy()
        
        # 根据后端类型优化配置
        if backend == BackendType.TRITON:
            # Triton优化：使用更小的块大小和向量化
            optimized_config.update({
                'block_size': 32,
                'vectorized': True,
                'fused_operations': True
            })
        elif backend == BackendType.CUDA:
            # CUDA优化：平衡内存和计算
            optimized_config.update({
                'block_size': 64,
                'use_shared_memory': True,
                'optimize_memory_access': True
            })
        elif backend == BackendType.METAL:
            # Metal优化：针对Apple Silicon优化
            optimized_config.update({
                'block_size': 16,
                'use_simd': True,
                'metal_specific_optimizations': True
            })
        else:  # CPU
            # CPU优化：最大化缓存利用率
            optimized_config.update({
                'block_size': 128,
                'cache_aware': True,
                'multithreaded': True
            })
        
        return optimized_config


# 全局后端优化器实例
_backend_optimizer = BackendOptimizer()


def get_optimal_backend() -> BackendType:
    """获取最优后端"""
    return _backend_optimizer.get_optimal_backend()


def get_backend_info() -> Dict:
    """获取后端信息"""
    return _backend_optimizer.get_backend_info()


def optimize_quantizer_config(quantizer_config: Dict, 
                            backend: Optional[BackendType] = None) -> Dict:
    """优化量化器配置"""
    return _backend_optimizer.optimize_for_backend(quantizer_config, backend)