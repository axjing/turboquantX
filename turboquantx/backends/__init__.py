"""
后端支持模块：硬件感知优化和多后端支持
"""

from .optimizer import get_optimal_backend, BackendType

__all__ = ["get_optimal_backend", "BackendType"]