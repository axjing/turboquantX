#!/usr/bin/env python3
"""
turboquantx 量化算法综合测评

比较 TurboQuant、IsoQuant、PlanarQuant、RotorQuant 四种量化算法：
1. 合成MSE失真（单位向量，d=128）
2. 内积保持性（QJL两阶段）
3. 针在干草堆检索（NIAH）
4. 速度（量化+解量化延迟）
5. 参数效率

参考 rotorquant/benchmark_vs_reference.py 设计
"""

import sys
import os
import time
import math
import gc
import argparse
import torch

# 添加模块路径以支持直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入turboquantx量化器
from turboquantx.quantizers.turboquant import TurboQuantMSE, TurboQuantProd
from turboquantx.quantizers.isoquant import IsoQuantMSE, IsoQuantProd
from turboquantx.quantizers.planarquant import PlanarQuantMSE, PlanarQuantProd
from turboquantx.quantizers.rotorquant import RotorQuantMSE, RotorQuantProd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'


def gpu_mem_mb():
    """获取GPU内存使用（MB）"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def gpu_peak_mb():
    """获取GPU峰值内存使用（MB）"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: MSE失真测试
# ═══════════════════════════════════════════════════════════════════════════════

def test_mse(d=128, n=2000, bits_list=[2, 3, 4]):
    """MSE失真测试 - 四种量化算法比较"""
    print("\n" + "=" * 72)
    print("TEST 1: MSE失真 — turboquantx量化算法比较")
    print("=" * 72)
    print(f"  d={d}, n_vectors={n}, device={DEVICE}\n")

    # 生成随机单位向量
    torch.manual_seed(42)
    x = torch.randn(n, d, device=DEVICE)
    x = x / x.norm(dim=-1, keepdim=True)

    # 理论边界：sqrt(3)*pi/2 * 1/4^b（来自论文）
    def theory(b):
        return math.sqrt(3) * math.pi / 2 * (1 / 4**b)

    print(f"  {'bits':>4}  {'TurboQ MSE':>12}  {'IsoQ MSE':>12}  {'PlanarQ MSE':>12}  {'RotorQ MSE':>12}  {'theory':>12}")
    print(f"  {'────':>4}  {'────────────':>12}  {'────────────':>12}  {'────────────':>12}  {'────────────':>12}  {'────────────':>12}")

    for bits in bits_list:
        # TurboQuant
        tq = TurboQuantMSE(d, bits, device=DEVICE)
        x_tq, _ = tq(x)
        mse_tq = ((x - x_tq)**2).sum(dim=-1).mean().item()

        # IsoQuant
        iq = IsoQuantMSE(d, bits, device=DEVICE)
        x_iq, _ = iq(x)
        mse_iq = ((x - x_iq)**2).sum(dim=-1).mean().item()

        # PlanarQuant
        pq = PlanarQuantMSE(d, bits, device=DEVICE)
        x_pq, _ = pq(x)
        mse_pq = ((x - x_pq)**2).sum(dim=-1).mean().item()

        # RotorQuant
        rq = RotorQuantMSE(d, bits, device=DEVICE)
        x_rq, _ = rq(x)
        mse_rq = ((x - x_rq)**2).sum(dim=-1).mean().item()

        th = theory(bits)

        print(f"  {bits:>4}  {mse_tq:>12.6f}  {mse_iq:>12.6f}  {mse_pq:>12.6f}  {mse_rq:>12.6f}  {th:>12.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: 内积保持性测试
# ═══════════════════════════════════════════════════════════════════════════════

def test_inner_product(d=128, n_pairs=3000, bits_list=[2, 3, 4]):
    """内积保持性测试（两阶段QJL校正）"""
    print("\n" + "=" * 72)
    print("TEST 2: 内积无偏性（两阶段QJL校正）")
    print("=" * 72)
    print(f"  d={d}, n_pairs={n_pairs}\n")

    torch.manual_seed(42)
    x = torch.randn(n_pairs, d, device=DEVICE)
    y = torch.randn(n_pairs, d, device=DEVICE)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    true_ip = (x * y).sum(dim=-1)

    print(f"  {'bits':>4}  {'method':>10}  {'bias':>12}  {'RMSE':>12}  {'corr':>8}")
    print(f"  {'────':>4}  {'──────────':>10}  {'────────────':>12}  {'────────────':>12}  {'────────':>8}")

    for bits in bits_list:
        results = {}

        # TurboQuant Prod
        tqp = TurboQuantProd(d, bits, qjl_dim=d, device=DEVICE)
        cx_tq = tqp.quantize(x)
        est_ip_tq = tqp.inner_product(y, cx_tq)
        results['TurboQ'] = est_ip_tq

        # IsoQuant Prod
        iqp = IsoQuantProd(d, bits, qjl_dim=d, device=DEVICE)
        cx_iq = iqp.quantize(x)
        est_ip_iq = iqp.inner_product(y, cx_iq)
        results['IsoQ'] = est_ip_iq

        # PlanarQuant Prod
        pqp = PlanarQuantProd(d, bits, qjl_dim=d, device=DEVICE)
        cx_pq = pqp.quantize(x)
        est_ip_pq = pqp.inner_product(y, cx_pq)
        results['PlanarQ'] = est_ip_pq

        # RotorQuant Prod
        rqp = RotorQuantProd(d, bits, qjl_dim=d, device=DEVICE)
        cx_rq = rqp.quantize(x)
        est_ip_rq = rqp.inner_product(y, cx_rq)
        results['RotorQ'] = est_ip_rq

        for name, approx in results.items():
            diff = approx - true_ip
            bias = diff.mean().item()
            rmse = diff.pow(2).mean().sqrt().item()
            corr = torch.corrcoef(torch.stack([true_ip, approx]))[0, 1].item()
            print(f"  {bits:>4}  {name:>10}  {bias:>+12.6f}  {rmse:>12.6f}  {corr:>8.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: 针在干草堆测试（NIAH）
# ═══════════════════════════════════════════════════════════════════════════════

def test_niah(d=128, bits_list=[2, 3, 4], seq_lens=[512, 2048, 8192]):
    """针在干草堆检索测试"""
    print("\n" + "=" * 72)
    print("TEST 3: 针在干草堆检索（NIAH）")
    print("=" * 72)

    print(f"\n  {'bits':>4}  {'seq':>6}  {'TurboQ':>8}  {'IsoQ':>8}  {'PlanarQ':>8}  {'RotorQ':>8}")
    print(f"  {'────':>4}  {'──────':>6}  {'────────':>8}  {'────────':>8}  {'────────':>8}  {'────────':>8}")

    for bits in bits_list:
        for seq_len in seq_lens:
            torch.manual_seed(42)
            keys = torch.randn(seq_len, d, device=DEVICE)
            keys = keys / keys.norm(dim=-1, keepdim=True)
            needle_idx = seq_len // 3
            needle = keys[needle_idx].clone()
            query = needle + 0.01 * torch.randn(d, device=DEVICE)
            query = query / query.norm()

            results = {}

            # TurboQuant
            tqp = TurboQuantProd(d, bits, qjl_dim=d, device=DEVICE)
            cx_tq = tqp.quantize(keys)
            est_ip_tq = tqp.inner_product(query.expand(seq_len, -1), cx_tq)
            found_tq = est_ip_tq.argmax().item()
            results['TurboQ'] = 'EXACT' if found_tq == needle_idx else f'MISS({abs(found_tq-needle_idx)})'

            # IsoQuant
            iqp = IsoQuantProd(d, bits, qjl_dim=d, device=DEVICE)
            cx_iq = iqp.quantize(keys)
            est_ip_iq = iqp.inner_product(query.expand(seq_len, -1), cx_iq)
            found_iq = est_ip_iq.argmax().item()
            results['IsoQ'] = 'EXACT' if found_iq == needle_idx else f'MISS({abs(found_iq-needle_idx)})'

            # PlanarQuant
            pqp = PlanarQuantProd(d, bits, qjl_dim=d, device=DEVICE)
            cx_pq = pqp.quantize(keys)
            est_ip_pq = pqp.inner_product(query.expand(seq_len, -1), cx_pq)
            found_pq = est_ip_pq.argmax().item()
            results['PlanarQ'] = 'EXACT' if found_pq == needle_idx else f'MISS({abs(found_pq-needle_idx)})'

            # RotorQuant
            rqp = RotorQuantProd(d, bits, qjl_dim=d, device=DEVICE)
            cx_rq = rqp.quantize(keys)
            est_ip_rq = rqp.inner_product(query.expand(seq_len, -1), cx_rq)
            found_rq = est_ip_rq.argmax().item()
            results['RotorQ'] = 'EXACT' if found_rq == needle_idx else f'MISS({abs(found_rq-needle_idx)})'

            print(f"  {bits:>4}  {seq_len:>6}  {results['TurboQ']:>8}  {results['IsoQ']:>8}  {results['PlanarQ']:>8}  {results['RotorQ']:>8}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: 速度测试
# ═══════════════════════════════════════════════════════════════════════════════

def test_speed(d=128, bits=3, n_list=[1000, 5000, 10000]):
    """速度测试（量化+解量化循环）"""
    print("\n" + "=" * 72)
    print("TEST 4: 速度测试（量化+解量化循环）")
    print("=" * 72)
    print(f"  GPU: {GPU_NAME}")
    print(f"  d={d}, bits={bits}\n")

    for n in n_list:
        torch.manual_seed(42)
        x = torch.randn(n, d, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)

        # 预热
        for _ in range(3):
            tq = TurboQuantMSE(d, bits, device=DEVICE)
            tq(x); torch.cuda.synchronize()

        # TurboQuant
        t0 = time.perf_counter()
        for _ in range(10):
            tq = TurboQuantMSE(d, bits, device=DEVICE)
            tq(x); torch.cuda.synchronize()
        t_tq = (time.perf_counter() - t0) / 10 * 1000

        # IsoQuant
        iq = IsoQuantMSE(d, bits, device=DEVICE)
        for _ in range(3):
            iq(x); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            iq(x); torch.cuda.synchronize()
        t_iq = (time.perf_counter() - t0) / 10 * 1000

        # PlanarQuant
        pq = PlanarQuantMSE(d, bits, device=DEVICE)
        for _ in range(3):
            pq(x); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            pq(x); torch.cuda.synchronize()
        t_pq = (time.perf_counter() - t0) / 10 * 1000

        # RotorQuant
        rq = RotorQuantMSE(d, bits, device=DEVICE)
        for _ in range(3):
            rq(x); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            rq(x); torch.cuda.synchronize()
        t_rq = (time.perf_counter() - t0) / 10 * 1000

        print(f"  n={n:>5}: TQ={t_tq:>8.2f}ms  IQ={t_iq:>8.2f}ms  PQ={t_pq:>8.2f}ms  RQ={t_rq:>8.2f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: 参数效率测试
# ═══════════════════════════════════════════════════════════════════════════════

def test_params(d=128, bits=3):
    """参数/缓冲区效率测试"""
    print("\n" + "=" * 72)
    print("TEST 5: 参数/缓冲区效率")
    print("=" * 72)

    # TurboQuant
    tq = TurboQuantMSE(d, bits, device='cpu')
    tq_params = d * d + 2**bits  # 旋转矩阵 + 码本
    print(f"\n  TurboQuant:")
    print(f"    旋转矩阵: {d}x{d} = {d*d:,}")
    print(f"    码本: {2**bits} 质心")
    print(f"    总计: {tq_params:,}")

    # IsoQuant
    iq = IsoQuantMSE(d, bits, device='cpu')
    iq_params = iq.rotation_params + 2**bits
    print(f"\n  IsoQuant:")
    print(f"    旋转参数: {iq.rotation_params}")
    print(f"    码本: {2**bits} 质心")
    print(f"    总计: {iq_params}")
    print(f"    比率: {tq_params/iq_params:.1f}x 更小")

    # PlanarQuant
    pq = PlanarQuantMSE(d, bits, device='cpu')
    pq_params = pq.rotation_params + 2**bits
    print(f"\n  PlanarQuant:")
    print(f"    旋转参数: {pq.rotation_params}")
    print(f"    码本: {2**bits} 质心")
    print(f"    总计: {pq_params}")
    print(f"    比率: {tq_params/pq_params:.1f}x 更小")

    # RotorQuant
    rq = RotorQuantMSE(d, bits, device='cpu')
    rq_params = rq.rotation_params + 2**bits
    print(f"\n  RotorQuant:")
    print(f"    旋转参数: {rq.rotation_params}")
    print(f"    码本: {2**bits} 质心")
    print(f"    总计: {rq_params}")
    print(f"    比率: {tq_params/rq_params:.1f}x 更小")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='turboquantx量化算法综合测评')
    parser.add_argument('--bits', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--skip-synthetic', action='store_true', help='跳过合成测试')
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║               turboquantx量化算法综合测评                        ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"  GPU: {GPU_NAME}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  比特: {args.bits}")

    if not args.skip_synthetic:
        test_mse(bits_list=args.bits)
        test_inner_product(bits_list=args.bits)
        test_niah(bits_list=args.bits)
        test_speed()
        test_params()

    print("\n" + "=" * 72)
    print("所有测试完成")
    print("=" * 72)