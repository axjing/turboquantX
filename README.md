# TurboQuantX: High-Performance KV Cache Compression

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/turboquantx/turboquantx)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)

**TurboQuantX** is a unified, high-performance KV cache compression library that combines the best features of [TurboQuant+](https://github.com/TheTom/turboquant_plus) and [RotorQuant](https://github.com/scrya-com/rotorquant), delivering superior performance, flexibility, and ease of use.

## 🚀 Key Features

- **Multiple Rotation Types**: Hadamard, Quaternion, Clifford, and Planar rotations
- **PolarQuant + QJL**: Unbiased inner product quantization for accurate attention
- **MSE-Only Mode**: Optimized value compression
- **Sparse V Optimization**: Up to 50% faster dequantization at long contexts
- **Unified API**: Consistent interface with both NumPy and PyTorch support
- **Production Ready**: Comprehensive testing, documentation, and examples

## 📊 Performance

| Config | Bits | Compression | PPL (wikitext-2) | Speed vs FP16 |
|--------|------|-------------|------------------|---------------|
| **turbo2** | 2.5 | 6.4x | +6.48% | 0.85x |
| **turbo3** | 3.5 | 4.6x | +1.06% | 0.90x |
| **turbo4** | 4.25 | 3.8x | +0.23% | 0.93x |

*Results on M5 Max 128GB with Qwen3.5-35B-A3B*

## 🔧 Installation

### Quick Install

```bash
# Basic installation (NumPy only)
pip install turboquantx

# With PyTorch support
pip install turboquantx[torch]

# Full installation with all features
pip install turboquantx[all]

# Development install
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- NumPy 1.24+
- SciPy 1.10+
- Optional: PyTorch 2.0+, Triton 2.1+

## 🎯 Quick Start

### Basic Usage

```python
import numpy as np
import turboquantx as tqx
from turboquantx.core.config import TurboConfig

# Create configuration
config = TurboConfig(
    bits=3,  # 3-bit quantization
    rotation_type="hadamard",  # Use Walsh-Hadamard rotation
    use_sparse_v=True  # Enable Sparse V optimization
)

# Create KV cache
cache = tqx.kv_cache.TurboKVCache(
    d_key=128,
    d_value=128,
    config=config
)

# Simulate model inference
batch_size, seq_len, d = 2, 10, 128
keys = np.random.randn(batch_size, seq_len, d).astype(np.float32)
values = np.random.randn(batch_size, seq_len, d).astype(np.float32)

# Compress and cache
keys_flat = keys.reshape(-1, d)
values_flat = values.reshape(-1, d)
cache.append(keys_flat, values_flat)

# Query the cache
queries = np.random.randn(batch_size, d).astype(np.float32)
attention_scores = cache.attention_scores(queries)

# Retrieve values with Sparse V optimization
values_compressed = cache.get_values(attention_weights=attention_scores)

print(f"Compression ratio: {config.compression_ratio:.2f}x")
print(f"Values shape: {values_compressed.shape}")
```

### Advanced Usage

```python
import torch
from turboquantx.core.config import TurboConfig, TURBO4_CONFIG

# Use predefined high-quality config
config = TURBO4_CONFIG.copy()
config.update(
    rotation_type="quaternion",  # Use quaternion rotation
    use_sparse_v=True,
    backend="torch"
)

# Create quantizer directly
quantizer = tqx.quantizers.TurboQuant(
    d=256,
    bit_width=4,
    rotation_type="quaternion"
)

# Quantize data
x = torch.randn(100, 256).cuda()
compressed = quantizer.quantize(x)

# Dequantize
x_hat = quantizer.dequantize(compressed)

# Compute unbiased inner products
y = torch.randn(100, 256).cuda()
ip = quantizer.inner_product(y, compressed)
```

## 📚 API Reference

### Core Components

#### TurboConfig

Central configuration class for all components:

```python
from turboquantx.core.config import TurboConfig

config = TurboConfig(
    bits=3,                    # Bits per coordinate (2, 3, 4)
    rotation_type="hadamard",  # Rotation type
    use_sparse_v=True,         # Enable Sparse V
    sparse_v_threshold=1e-6,   # Attention threshold
    backend="torch",          # Backend: torch, numpy
    device="cuda"             # Device: cpu, cuda, mps
)
```

#### Quantizers

- **PolarQuant**: MSE-optimal quantizer with rotation
- **TurboQuant**: Full quantizer with PolarQuant + QJL for unbiased inner products
- **TurboQuantMSE**: MSE-only mode for values

```python
from turboquantx.quantizers import PolarQuant, TurboQuant, TurboQuantMSE

# MSE-only
mse_quant = PolarQuant(d=128, bits=3, seed=42)
mse_quant.initialize()

# Full TurboQuant (unbiased inner products)
turbo = TurboQuant(d=128, bit_width=3, seed=42)
compressed = turbo.quantize(x)
x_hat = turbo.dequantize(compressed)
ip = turbo.inner_product(y, compressed)  # Unbiased!
```

#### KV Cache

```python
from turboquantx.kv_cache import TurboKVCache

cache = TurboKVCache(
    d_key=128,
    d_value=128,
    config=config
)

# Append new tokens
cache.append(keys, values)

# Compute attention scores
scores = cache.attention_scores(queries)

# Retrieve with Sparse V optimization
values = cache.get_values(attention_weights=scores)
```

#### Rotations

```python
from turboquantx.rotations import (
    HadamardRotation,
    QuaternionRotation,
    PlanarRotation
)

# Walsh-Hadamard (fast, O(n log n))
rot = HadamardRotation(d=128, seed=42)
rot.initialize()
y = rot.rotate(x)

# Quaternion (SO(4), better for geometric data)
quat = QuaternionRotation(d=128, seed=42, mode="full")
quat.initialize()
y = quat.rotate(x)
```

### Predefined Configurations

```python
from turboquantx.core.config import (
    TURBO2_CONFIG,  # 2-bit, 6.4x compression
    TURBO3_CONFIG,  # 3-bit, 4.6x compression
    TURBO4_CONFIG,  # 4-bit, 3.8x compression
    ISOQUANT_CONFIG,  # Quaternion rotation
    PLANARQUANT_CONFIG,  # Planar rotation
)

# Use turbo3 with custom settings
config = TURBO3_CONFIG.copy()
config.update(rotation_type="quaternion", use_sparse_v=True)
```

## 🔬 Advanced Features

### Sparse V Optimization

Sparse V skips dequantization of value vectors with low attention weights:

```python
config = TurboConfig(
    bits=3,
    use_sparse_v=True,           # Enable
    sparse_v_threshold=1e-6     # Threshold
)

# Automatically applied during get_values()
values = cache.get_values(attention_weights=scores)
```

### Multiple Rotation Types

Choose the best rotation for your use case:

- **hadamard**: Fast Walsh-Hadamard Transform, O(n log n), good for most cases
- **quaternion**: SO(4) isoclinic rotation, better geometric properties
- **planar**: 2D Givens rotations, lightweight
- **clifford**: Full Clifford algebra (advanced)

### Backend Support

```python
# NumPy (default)
config = TurboConfig(backend="numpy")

# PyTorch with CUDA
config = TurboConfig(backend="torch", device="cuda")

# MPS (Apple Silicon)
config = TurboConfig(backend="torch", device="mps")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=turboquantx --cov-report=html

# Run benchmarks
pytest tests/ --benchmark-only

# Run specific test file
pytest tests/test_polar_quant.py -v
```

## 📈 Benchmarks

Run the built-in benchmarks:

```bash
# Basic compression benchmark
python -m turboquantx.benchmarks.compression

# Speed benchmark
python -m turboquantx.benchmarks.speed

# Memory benchmark
python -m turboquantx.benchmarks.memory
```

## 🏗️ Architecture

```
turboquantx/
├── core/                      # Core abstractions
│   ├── base.py               # Base classes
│   ├── config.py             # Configuration
│   └── utils.py              # Utilities
├── quantizers/                # Quantization algorithms
│   ├── polar.py              # PolarQuant (MSE)
│   ├── turbo.py              # TurboQuant (PolarQuant + QJL)
│   └── codebook.py           # Lloyd-Max centroids
├── kv_cache/                  # KV cache implementations
│   ├── compressor.py         # TurboKVCache
│   └── sparse_v.py           # Sparse V optimization
├── rotations/                 # Rotation operations
│   ├── hadamard.py           # Walsh-Hadamard Transform
│   ├── quaternion.py         # SO(4) rotation
│   └── planar.py             # 2D Givens rotation
└── integrations/              # External integrations
    └── llama_cpp.py          # llama.cpp support
```

## 🔬 Research Foundation

TurboQuantX implements algorithms from these papers:

- **TurboQuant**: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) - Base algorithm
- **PolarQuant**: [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) - Optimal quantization
- **QJL**: [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) - JL transform
- **IsoQuant**: Hardware-aligned SO(4) rotations
- **Sparse V**: Attention-gated dequantization

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **TheTom** for TurboQuant+ implementation and llama.cpp integration
- **scrya-com** for RotorQuant and quaternion rotations
- Google Research for TurboQuant paper and algorithm
- Community contributors and testers

## 📧 Contact

- GitHub Issues: [github.com/turboquantx/turboquantx/issues](https://github.com/turboquantx/turboquantx/issues)
- Discussions: [github.com/turboquantx/turboquantx/discussions](https://github.com/turboquantx/turboquantx/discussions)

---

**Made with ❤️ for the LLM community**
