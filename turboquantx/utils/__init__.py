"""
工具模块：提供通用工具函数和辅助类
"""

# 延迟导入以避免循环依赖
from .codebook import LloydMaxCodebook
from .profiler import QuantizationProfiler
from .bit_packing import BitPacker

# 不在这里导入QuantizerRegistry，以避免循环导入

__all__ = [
    "LloydMaxCodebook",
    "QuantizationProfiler",
    "BitPacker"
]

# 提供延迟导入函数
def get_quantizer_registry():
    """延迟获取QuantizerRegistry实例"""
    from .registry import QuantizerRegistry
    return QuantizerRegistry