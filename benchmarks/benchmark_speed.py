#!/usr/bin/env python3
"""
turboquantx 速度测评

基于rotorquant/benchmark_triton.py设计，比较四种量化算法的速度性能。

测试：
  A. 量化+解量化循环速度
  B. 不同向量数量的性能
  C. 不同嵌入维度的性能
  D. 不同比特宽度的性能

用法：
    python -m benchmarks.benchmark_speed
"""

import torch
import time
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquantx.quantizers.turboquant import TurboQuantMSE
from turboquantx.quantizers.isoquant import IsoQuantMSE
from turboquantx.quantizers.planarquant import PlanarQuantMSE
from turboquantx.quantizers.rotorquant import RotorQuantMSE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def time_fn(fn, n_warmup=10, n_iter=100, sync=True):
    """使用预热和平均计时函数。"""
    for _ in range(n_warmup):
        fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iter * 1000  # ms
    return elapsed


def verify_correctness():
    """验证四种量化算法的数值正确性。"""
    import torch.nn.functional as F
    
    print("=" * 70)
    print("正确性验证: 四种量化算法")
    print("=" * 70)

    d = 128
    n = 1024
    bits = 3
    device = DEVICE

    x = torch.randn(n, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    # 创建量化器
    tq = TurboQuantMSE(d, bits, device=device)
    iq = IsoQuantMSE(d, bits, device=device)
    pq = PlanarQuantMSE(d, bits, device=device)
    rq = RotorQuantMSE(d, bits, device=device)

    # 测试量化质量
    x_tq, _ = tq(x)
    x_iq, _ = iq(x)
    x_pq, _ = pq(x)
    x_rq, _ = rq(x)

    # 计算MSE
    mse_tq = ((x - x_tq)**2).sum(dim=-1).mean().item()
    mse_iq = ((x - x_iq)**2).sum(dim=-1).mean().item()
    mse_pq = ((x - x_pq)**2).sum(dim=-1).mean().item()
    mse_rq = ((x - x_rq)**2).sum(dim=-1).mean().item()

    # 计算余弦相似度
    cos_tq = F.cosine_similarity(x, x_tq, dim=-1).mean().item()
    cos_iq = F.cosine_similarity(x, x_iq, dim=-1).mean().item()
    cos_pq = F.cosine_similarity(x, x_pq, dim=-1).mean().item()
    cos_rq = F.cosine_similarity(x, x_rq, dim=-1).mean().item()

    print(f"\n  量化质量 (d={d}, n={n}, bits={bits}):")
    print(f"  {'方法':<12}  {'MSE':>12}  {'余弦相似度':>12}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}")
    print(f"  {'TurboQuant':<12}  {mse_tq:>12.6f}  {cos_tq:>12.6f}")
    print(f"  {'IsoQuant':<12}  {mse_iq:>12.6f}  {cos_iq:>12.6f}")
    print(f"  {'PlanarQuant':<12}  {mse_pq:>12.6f}  {cos_pq:>12.6f}")
    print(f"  {'RotorQuant':<12}  {mse_rq:>12.6f}  {cos_rq:>12.6f}")

    # 验证所有方法都产生合理的结果
    all_pass = all([cos > 0.9 for cos in [cos_tq, cos_iq, cos_pq, cos_rq]])
    print(f"\n  状态: {'✅ 所有通过' if all_pass else '⚠️ 部分失败'}")
    print()
    
    return all_pass


def benchmark_quantize_dequantize():
    """测评量化+解量化循环速度。"""
    print("=" * 70)
    print("测评 A: 量化+解量化循环速度")
    print("=" * 70)

    d = 128
    bits = 3
    device = DEVICE

    print(f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"  d={d}, bits={bits}\n")
    print(f"  {'n_vecs':>8s}  {'TurboQ':>10s}  {'IsoQ':>10s}  {'PlanarQ':>10s}  {'RotorQ':>10s}  {'最快':>8s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

    for n in [256, 1024, 4096, 16384, 65536]:
        x = torch.randn(n, d, device=device)
        x = x / x.norm(dim=-1, keepdim=True)

        # TurboQuant
        tq = TurboQuantMSE(d, bits, device=device)
        def tq_fn():
            return tq(x)
        t_tq = time_fn(tq_fn)

        # IsoQuant
        iq = IsoQuantMSE(d, bits, device=device)
        def iq_fn():
            return iq(x)
        t_iq = time_fn(iq_fn)

        # PlanarQuant
        pq = PlanarQuantMSE(d, bits, device=device)
        def pq_fn():
            return pq(x)
        t_pq = time_fn(pq_fn)

        # RotorQuant
        rq = RotorQuantMSE(d, bits, device=device)
        def rq_fn():
            return rq(x)
        t_rq = time_fn(rq_fn)

        # 找出最快的方法
        times = [t_tq, t_iq, t_pq, t_rq]
        min_time = min(times)
        fastest = ['TurboQ', 'IsoQ', 'PlanarQ', 'RotorQ'][times.index(min_time)]

        print(f"  {n:>8d}  {t_tq:>8.2f}ms  {t_iq:>8.2f}ms  {t_pq:>8.2f}ms  {t_rq:>8.2f}ms  {fastest:>8s}")

    print()


def benchmark_varying_dimensions():
    """测评不同嵌入维度的性能。"""
    print("=" * 70)
    print("测评 B: 不同嵌入维度的性能")
    print("=" * 70)

    n = 4096
    bits = 3
    device = DEVICE

    print(f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"  n_vectors={n}, bits={bits}\n")
    print(f"  {'dim':>6s}  {'TurboQ':>10s}  {'IsoQ':>10s}  {'PlanarQ':>10s}  {'RotorQ':>10s}  {'最快':>8s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

    for d in [64, 128, 256, 512]:
        x = torch.randn(n, d, device=device)
        x = x / x.norm(dim=-1, keepdim=True)

        # TurboQuant
        tq = TurboQuantMSE(d, bits, device=device)
        def tq_fn():
            return tq(x)
        t_tq = time_fn(tq_fn)

        # IsoQuant
        iq = IsoQuantMSE(d, bits, device=device)
        def iq_fn():
            return iq(x)
        t_iq = time_fn(iq_fn)

        # PlanarQuant
        pq = PlanarQuantMSE(d, bits, device=device)
        def pq_fn():
            return pq(x)
        t_pq = time_fn(pq_fn)

        # RotorQuant
        rq = RotorQuantMSE(d, bits, device=device)
        def rq_fn():
            return rq(x)
        t_rq = time_fn(rq_fn)

        # 找出最快的方法
        times = [t_tq, t_iq, t_pq, t_rq]
        min_time = min(times)
        fastest = ['TurboQ', 'IsoQ', 'PlanarQ', 'RotorQ'][times.index(min_time)]

        print(f"  {d:>6d}  {t_tq:>8.2f}ms  {t_iq:>8.2f}ms  {t_pq:>8.2f}ms  {t_rq:>8.2f}ms  {fastest:>8s}")

    print()


def benchmark_bitwidth_sweep():
    """测评不同比特宽度的性能。"""
    print("=" * 70)
    print("测评 C: 比特宽度扫描（质量 vs 速度）")
    print("=" * 70)

    d = 128
    n = 4096
    device = DEVICE

    print(f"  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"  d={d}, n={n}\n")
    print(f"  {'bits':>4s}  {'n_levels':>8s}  {'TurboQ':>10s}  {'IsoQ':>10s}  {'PlanarQ':>10s}  {'RotorQ':>10s}  {'最快':>8s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

    x = torch.randn(n, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    for bits in [1, 2, 3, 4]:
        # TurboQuant
        tq = TurboQuantMSE(d, bits, device=device)
        def tq_fn():
            return tq(x)
        t_tq = time_fn(tq_fn)

        # IsoQuant
        iq = IsoQuantMSE(d, bits, device=device)
        def iq_fn():
            return iq(x)
        t_iq = time_fn(iq_fn)

        # PlanarQuant
        pq = PlanarQuantMSE(d, bits, device=device)
        def pq_fn():
            return pq(x)
        t_pq = time_fn(pq_fn)

        # RotorQuant
        rq = RotorQuantMSE(d, bits, device=device)
        def rq_fn():
            return rq(x)
        t_rq = time_fn(rq_fn)

        # 找出最快的方法
        times = [t_tq, t_iq, t_pq, t_rq]
        min_time = min(times)
        fastest = ['TurboQ', 'IsoQ', 'PlanarQ', 'RotorQ'][times.index(min_time)]

        print(f"  {bits:>4d}  {2**bits:>8d}  {t_tq:>8.2f}ms  {t_iq:>8.2f}ms  {t_pq:>8.2f}ms  {t_rq:>8.2f}ms  {fastest:>8s}")

    print()


def benchmark_memory_efficiency():
    """测评内存效率。"""
    print("=" * 70)
    print("测评 D: 内存效率（量化器状态大小）")
    print("=" * 70)

    d = 128
    bits = 3
    device = 'cpu'  # 使用CPU测量内存使用

    print(f"  维度: d={d}, bits={bits}\n")
    print(f"  {'方法':<12}  {'旋转参数':>12}  {'码本大小':>12}  {'总大小':>12}  {'相对大小':>12}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")

    # TurboQuant
    tq = TurboQuantMSE(d, bits, device=device)
    tq_rot_params = d * d  # 旋转矩阵
    tq_codebook = 2**bits  # 码本
    tq_total = tq_rot_params + tq_codebook

    # IsoQuant
    iq = IsoQuantMSE(d, bits, device=device)
    iq_rot_params = iq.rotation_params
    iq_codebook = 2**bits
    iq_total = iq_rot_params + iq_codebook

    # PlanarQuant
    pq = PlanarQuantMSE(d, bits, device=device)
    pq_rot_params = pq.rotation_params
    pq_codebook = 2**bits
    pq_total = pq_rot_params + pq_codebook

    # RotorQuant
    rq = RotorQuantMSE(d, bits, device=device)
    rq_rot_params = rq.rotation_params
    rq_codebook = 2**bits
    rq_total = rq_rot_params + rq_codebook

    # 计算相对大小
    print(f"  {'TurboQuant':<12}  {tq_rot_params:>12,}  {tq_codebook:>12,}  {tq_total:>12,}  {'1.0x':>12}")
    print(f"  {'IsoQuant':<12}  {iq_rot_params:>12,}  {iq_codebook:>12,}  {iq_total:>12,}  {tq_total/iq_total:>11.1f}x")
    print(f"  {'PlanarQuant':<12}  {pq_rot_params:>12,}  {pq_codebook:>12,}  {pq_total:>12,}  {tq_total/pq_total:>11.1f}x")
    print(f"  {'RotorQuant':<12}  {rq_rot_params:>12,}  {rq_codebook:>12,}  {rq_total:>12,}  {tq_total/rq_total:>11.1f}x")

    print(f"\n  内存效率优势:")
    print(f"  - IsoQuant:     {tq_total/iq_total:.1f}x 更小的旋转参数")
    print(f"  - PlanarQuant:  {tq_total/pq_total:.1f}x 更小的旋转参数")
    print(f"  - RotorQuant:   {tq_total/rq_total:.1f}x 更小的旋转参数")

    print()


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                turboquantx 速度测评                               ║")
    print("║                比较四种量化算法的性能                               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    else:
        print("使用CPU进行测试")
    print(f"PyTorch: {torch.__version__}")
    print()

    # 步骤1: 验证正确性
    ok = verify_correctness()
    if not ok:
        print("正确性检查失败 — 中止测评")
        return

    # 步骤2: 运行测评
    benchmark_quantize_dequantize()
    benchmark_varying_dimensions()
    benchmark_bitwidth_sweep()
    benchmark_memory_efficiency()

    print("=" * 70)
    print("所有测评完成")
    print("=" * 70)


if __name__ == "__main__":
    main()