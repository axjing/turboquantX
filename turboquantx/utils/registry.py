"""
量化器注册表：支持插件化架构，轻松扩展新量化算法
"""

from typing import Dict, Type, Optional, Any
import importlib


class QuantizerRegistry:
    """量化器注册表，支持动态注册和实例化"""
    
    _registry: Dict[str, str] = {}  # 存储量化器类名和模块路径
    _default_configs: Dict[str, Dict[str, Any]] = {}
    _initialized = False
    
    @classmethod
    def _initialize(cls):
        """延迟初始化，避免循环导入"""
        if cls._initialized:
            return
        
        # 注册内置量化器（使用字符串路径）
        cls._registry = {
            "turbo_mse": "turboquantx.quantizers.turboquant.TurboQuantMSE",
            "turbo_prod": "turboquantx.quantizers.turboquant.TurboQuantProd",
            "iso_mse": "turboquantx.quantizers.isoquant.IsoQuantMSE", 
            "iso_fast": "turboquantx.quantizers.isoquant.IsoQuantMSE",
            "iso_prod": "turboquantx.quantizers.isoquant.IsoQuantProd",
            "planar_mse": "turboquantx.quantizers.planarquant.PlanarQuantMSE",
            "planar_prod": "turboquantx.quantizers.planarquant.PlanarQuantProd",
            "rotor_mse": "turboquantx.quantizers.rotorquant.RotorQuantMSE",
            "rotor_prod": "turboquantx.quantizers.rotorquant.RotorQuantProd",
        }
        
        cls._default_configs = {
            "iso_mse": {"mode": "full"},
            "iso_fast": {"mode": "fast"},
            "iso_prod": {"mode": "full"},
        }
        
        cls._initialized = True
    
    @classmethod
    def register(cls, name: str, quantizer_class_path: str, 
                default_config: Optional[Dict[str, Any]] = None):
        """注册新的量化器"""
        cls._initialize()
        cls._registry[name] = quantizer_class_path
        if default_config:
            cls._default_configs[name] = default_config
        else:
            cls._default_configs[name] = {}
    
    @classmethod
    def _import_quantizer_class(cls, class_path: str):
        """动态导入量化器类"""
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    @classmethod
    def create(cls, name: str, d: int, bit_width: int, **kwargs) -> Any:
        """创建量化器实例"""
        cls._initialize()
        if name not in cls._registry:
            raise ValueError(f"量化器 '{name}' 未注册。可用量化器: {list(cls._registry.keys())}")
        
        # 动态导入量化器类
        quantizer_class = cls._import_quantizer_class(cls._registry[name])
        
        # 合并默认配置和用户配置
        config = cls._default_configs.get(name, {}).copy()
        config.update(kwargs)
        
        return quantizer_class(d, bit_width, **config)
    
    @classmethod
    def list_quantizers(cls) -> Dict[str, str]:
        """列出所有已注册的量化器"""
        cls._initialize()
        return cls._registry.copy()
    
    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """获取量化器信息"""
        cls._initialize()
        if name not in cls._registry:
            raise ValueError(f"量化器 '{name}' 未注册")
        
        quantizer_class = cls._import_quantizer_class(cls._registry[name])
        return {
            'class': quantizer_class,
            'name': name,
            'default_config': cls._default_configs.get(name, {}),
            'description': quantizer_class.__doc__ or "无描述"
        }