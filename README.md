# turboquantX

**turboquantX** 是一个统一的高性能 KV 缓存压缩框架，专门为大语言模型（LLM）推理优化设计。

## 🚀 核心特性

### 多量化算法支持

- **TurboQuant**: 原始 TurboQuant 算法（极坐标量化 + QJL）
- **IsoQuant**: 四元数 4D 块旋转量化（推荐默认）
- **PlanarQuant**: 2D Givens 旋转量化（最快）
- **RotorQuant**: Clifford 代数旋转量化

### 优化策略

1. **批量处理优化**: 向量化操作，减少Python循环
2. **缓存机制**: 旋转矩阵、码本只初始化一次
3. **内存布局**: 连续内存访问，减少缓存失效
4. **算法优化**: 合并重复计算，提前退出条件

### 预留接口

- ⏳ Triton加速: 架构支持，待实现
- ⏳ CUDA内核: 架构支持，待实现
- ⏳ llama.cpp: 集成层预留

### 设计原则

- ✅ **第一性原理**: 聚焦核心功能，剔除冗余
- ✅ **简单易用**: 配置驱动，预设模板
- ✅ **高效**: 算法+内存+缓存三重优化
- ✅ **模块化**: 单一职责，接口隔离
- ✅ **可拓展**: 插件化架构，预留接口

## 🏗️ 项目结构

```
turboquantX/
├── turboquantx/           # 核心库
│   ├── quantizers/         # 量化算法
│   │   ├── base.py         # 基础量化器类
│   │   ├── turboquant.py   # TurboQuant 算法
│   │   ├── isoquant.py     # IsoQuant 算法
│   │   ├── planarquant.py  # PlanarQuant 算法
│   │   └── rotorquant.py   # RotorQuant 算法
│   ├── cache/              # 缓存管理
│   │   ├── compressor.py   # KV 缓存压缩器
│   │   └── sparse_v.py     # 稀疏 V 优化
│   ├── utils/              # 工具函数
│   │   ├── codebook.py     # 码本生成
│   │   ├── bit_packing.py  # 位打包工具
│   │   ├── registry.py     # 量化器注册表
│   │   └── profiler.py     # 性能分析
│   └── backends/           # 硬件后端
│       └── optimizer.py     # 后端优化
├── benchmarks/             # 基准测试套件
│   ├── benchmark_quantizers.py    # 算法比较
│   ├── benchmark_perplexity.py    # 困惑度测试
│   ├── benchmark_vram.py          # VRAM 节省
│   ├── benchmark_speed.py         # 速度性能
│   └── benchmark_google_parity.py # Google 对等性测试
├── tests/                  # 单元测试
└── examples/              # 使用示例
```

## TodoList

**P0:**

- [ ] 实现Triton内核加速
- [ ] 完成CUDA后端集成
- [ ]添加llama.py集成示例

**P1**

- [ ] 动态位宽调整
- [ ] MoE-aware压缩
- [ ] 长上下文优化（64K+）

**P2**

- [ ] 自动配置搜索
- [ ] 多节点分布式缓存
- [ ] 硬件感知优化

## 📦 安装

### 基础安装

```bash
pip install turboquantX
```

### 完整安装

```bash
pip install turboquantX[full]
```

### 开发安装

```bash
pip install turboquantX[dev]
```

## 🎯 快速开始

### 基础用法

```python
import turboquantx as tqx

# 1行配置
config = tqx.TurboConfig(bits=3, use_sparse_v=True)

# 1行创建缓存
cache = tqx.kv_cache.TurboKVCache(d_key=128, d_value=128, config=config)

# 1行压缩
cache.append(keys, values)

# 1行查询
scores = cache.attention_scores(queries)
```

### 高级用法

```python
from turboquantx.core.config import TURBO4_CONFIG
from turboquantx.quantizers import TurboQuant

# 使用预设配置
config = TURBO4_CONFIG.copy()
config.rotation_type = "quaternion"

# 直接使用量化器
tq = TurboQuant(d=256, bit_width=4, rotation_type="quaternion")
compressed = tq.quantize(x)
ip = tq.inner_product(y, compressed)  # 无偏内积
```

## 📊 性能基准测试

项目包含完整的性能测试套件：

### 综合算法比较

```bash
python -m benchmarks.benchmark_quantizers --bits 2 3 4
```

### 困惑度测试

```bash
python -m benchmarks.benchmark_perplexity --model Qwen/Qwen2.5-3B-Instruct --bits 3 4
```

### VRAM 节省测试

```bash
python -m benchmarks.benchmark_vram --model Qwen/Qwen2.5-3B-Instruct --contexts 1024 2048 4096
```

### 速度性能测试

```bash
python -m benchmarks.benchmark_speed
```

### Google TurboQuant 对等性测试

```bash
python -m benchmarks.benchmark_google_parity --bits 3 4
```

### 性能分析

```python
from turboquantx.utils.profiler import QuantizationProfiler

profiler = QuantizationProfiler()
with profiler.record("quantization"):
    x_hat, indices = quantizer(x)

print("性能统计:")
print(profiler.get_stats())
```

## 🧪 测试

运行完整测试套件：

```bash
python -m pytest tests/ -v
```

运行特定测试类别：

```bash
# 测试量化算法
python -m pytest tests/test_quantizers.py -v

# 测试缓存压缩
python -m pytest tests/test_cache.py -v

# 测试工具函数
python -m pytest tests/test_utils.py -v
```

## 🤝 贡献

我们欢迎社区贡献！请查看我们的贡献指南获取详细信息。

### 开发设置

```bash
git clone https://github.com/axjing/turboquantX.git
cd turboquantx
pip install -e .[dev]
```

### 代码风格

我们使用 `black` 进行代码格式化，`isort` 进行导入排序：

```bash
black turboquantx/
isort turboquantx/
```

## 📄 引用

如果您在研究中使用了 turboquantx，请引用：

```bibtex
@software{turboquantx,
  title = {turboquantx: Unified KV Cache Compression Framework},
  author = {axjing},
  year = {2026},
  url = {https://github.com/axjing/turboquantX}
}
```

本项目参考了 [turboquant_plus](https://github.com/TheTom/turboquant_plus.git) | [rotorquant](https://github.com/scrya-com/rotorquant.git)

- <https://github.com/TheTom/turboquant_plus.git>
- <https://github.com/scrya-com/rotorquant.git>

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

**turboquantX** - 让 LLM 推理更高效！🚀
