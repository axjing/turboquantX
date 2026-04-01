"""
turboquantx: Unified KV Cache Compression Framework for LLMs

整合 TurboQuant 和 RotorQuant 核心优势的统一 KV 缓存压缩框架。
"""

# 延迟导入以避免循环依赖
from .quantizers.base import BaseQuantizer
from .cache.compressor import KVCacheCompressor
from .cache.sparse_v import SparseVDequant
from .backends.optimizer import get_optimal_backend

# 默认量化器推荐
QuantMSE = None
QuantProd = None

__all__ = [
    # 基类
    "BaseQuantizer",
    # 缓存管理
    "KVCacheCompressor", "SparseVDequant",
    # 后端优化
    "get_optimal_backend",
    # 默认推荐
    "QuantMSE", "QuantProd",
]

# 延迟导入函数
def get_turboquant():
    """获取TurboQuant量化器"""
    from .quantizers.turboquant import TurboQuantMSE, TurboQuantProd
    return TurboQuantMSE, TurboQuantProd

def get_isoquant():
    """获取IsoQuant量化器"""
    from .quantizers.isoquant import IsoQuantMSE, IsoQuantProd
    return IsoQuantMSE, IsoQuantProd

def get_planarquant():
    """获取PlanarQuant量化器"""
    from .quantizers.planarquant import PlanarQuantMSE, PlanarQuantProd
    return PlanarQuantMSE, PlanarQuantProd

def get_rotorquant():
    """获取RotorQuant量化器"""
    from .quantizers.rotorquant import RotorQuantMSE, RotorQuantProd
    return RotorQuantMSE, RotorQuantProd

def get_quantizer_registry():
    """获取量化器注册表"""
    from .utils.registry import QuantizerRegistry
    return QuantizerRegistry

# 初始化默认量化器
def _initialize_defaults():
    global QuantMSE, QuantProd
    IsoQuantMSE, IsoQuantProd = get_isoquant()
    QuantMSE = IsoQuantMSE
    QuantProd = IsoQuantProd

# 自动初始化
_initialize_defaults()

__version__ = "1.0.0"