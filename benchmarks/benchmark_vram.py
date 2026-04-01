#!/usr/bin/env python3
"""
turboquantx VRAM节省测评

基于rotorquant/benchmark_vram.py设计，测量压缩存储（索引+范数）的实际GPU内存使用情况。

用法：
    python -m benchmarks.benchmark_vram
    python -m benchmarks.benchmark_vram --model Qwen/Qwen2.5-7B-Instruct --contexts 460 1860 4096
"""

import sys, os, gc, math, time, argparse
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

# 导入turboquantx实现
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turboquantx.quantizers.turboquant import TurboQuantMSE
from turboquantx.quantizers.isoquant import IsoQuantMSE
from turboquantx.quantizers.planarquant import PlanarQuantMSE
from turboquantx.quantizers.rotorquant import RotorQuantMSE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def flush():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# 压缩KV缓存层
# ═══════════════════════════════════════════════════════════════════════════════

from transformers.cache_utils import DynamicCache, DynamicLayer


class TurboQuantLayer(DynamicLayer):
    """缓存层存储TurboQuant压缩的索引+范数。"""

    def __init__(self, bits: int = 3, residual_len: int = 128):
        super().__init__()
        self.bits = bits
        self.residual_len = residual_len
        self._quantizers = {}  # head_dim -> TurboQuantMSE
        self._key_indices = None   # 索引张量
        self._key_norms = None     # 范数张量
        self._value_indices = None
        self._value_norms = None
        self._residual_keys = None
        self._residual_values = None
        self._total_len = 0
        self._head_dim = None

    def _get_quantizer(self, head_dim, device):
        key = (head_dim, str(device))
        if key not in self._quantizers:
            self._quantizers[key] = TurboQuantMSE(head_dim, self.bits, device=str(device))
        return self._quantizers[key]

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self._head_dim = key_states.shape[-1]
        self._residual_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self._residual_values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._residual_keys = torch.cat([self._residual_keys, key_states], dim=-2)
        self._residual_values = torch.cat([self._residual_values, value_states], dim=-2)
        self._total_len += key_states.shape[-2]

        if self._residual_keys.shape[-2] > self.residual_len:
            overflow = self._residual_keys.shape[-2] - self.residual_len
            to_q_k = self._residual_keys[..., :overflow, :]
            to_q_v = self._residual_values[..., :overflow, :]

            tq = self._get_quantizer(self._head_dim, self.device)

            # 量化键 - 存储索引（uint8）+ 范数（float32）
            k_flat = to_q_k.reshape(-1, self._head_dim).float()
            x_hat, indices = tq(k_flat)
            k_vec_idx = indices.to(torch.uint8)  # 压缩
            k_norms = k_flat.norm(dim=-1)  # 范数

            v_flat = to_q_v.reshape(-1, self._head_dim).float()
            x_hat, indices = tq(v_flat)
            v_vec_idx = indices.to(torch.uint8)
            v_norms = v_flat.norm(dim=-1)

            if self._key_indices is None:
                self._key_indices = k_vec_idx
                self._key_norms = k_norms
                self._value_indices = v_vec_idx
                self._value_norms = v_norms
            else:
                self._key_indices = torch.cat([self._key_indices, k_vec_idx], dim=0)
                self._key_norms = torch.cat([self._key_norms, k_norms], dim=0)
                self._value_indices = torch.cat([self._value_indices, v_vec_idx], dim=0)
                self._value_norms = torch.cat([self._value_norms, v_norms], dim=0)

            self._residual_keys = self._residual_keys[..., overflow:, :]
            self._residual_values = self._residual_values[..., overflow:, :]

        # 为注意力计算解量化
        if self._key_indices is not None and self._key_indices.numel() > 0:
            tq = self._get_quantizer(self._head_dim, self.device)
            # 重建索引进行解量化
            k_deq = tq.dequantize(self._key_indices.long(), self._key_norms).to(self.dtype)
            k_deq = k_deq.reshape(self._residual_keys.shape[0], self._residual_keys.shape[1],
                                  -1, self._head_dim)
            v_deq = tq.dequantize(self._value_indices.long(), self._value_norms).to(self.dtype)
            v_deq = v_deq.reshape(self._residual_values.shape[0], self._residual_values.shape[1],
                                  -1, self._head_dim)
            self.keys = torch.cat([k_deq, self._residual_keys], dim=-2)
            self.values = torch.cat([v_deq, self._residual_values], dim=-2)
        else:
            self.keys = self._residual_keys
            self.values = self._residual_values

        return self.keys, self.values

    def get_seq_length(self):
        return self._total_len


class IsoQuantLayer(DynamicLayer):
    """缓存层存储IsoQuant压缩的索引+范数。"""

    def __init__(self, bits: int = 3, residual_len: int = 128):
        super().__init__()
        self.bits = bits
        self.residual_len = residual_len
        self._quantizers = {}
        self._key_indices = None
        self._key_norms = None
        self._value_indices = None
        self._value_norms = None
        self._residual_keys = None
        self._residual_values = None
        self._total_len = 0
        self._head_dim = None

    def _get_quantizer(self, head_dim, device):
        key = (head_dim, str(device))
        if key not in self._quantizers:
            self._quantizers[key] = IsoQuantMSE(head_dim, self.bits, device=str(device))
        return self._quantizers[key]

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self._head_dim = key_states.shape[-1]
        self._residual_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self._residual_values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._residual_keys = torch.cat([self._residual_keys, key_states], dim=-2)
        self._residual_values = torch.cat([self._residual_values, value_states], dim=-2)
        self._total_len += key_states.shape[-2]

        if self._residual_keys.shape[-2] > self.residual_len:
            overflow = self._residual_keys.shape[-2] - self.residual_len
            to_q_k = self._residual_keys[..., :overflow, :]
            to_q_v = self._residual_values[..., :overflow, :]

            iq = self._get_quantizer(self._head_dim, self.device)

            k_flat = to_q_k.reshape(-1, self._head_dim).float()
            x_hat, indices = iq(k_flat)
            k_vec_idx = indices.to(torch.uint8)
            k_norms = k_flat.norm(dim=-1)

            v_flat = to_q_v.reshape(-1, self._head_dim).float()
            x_hat, indices = iq(v_flat)
            v_vec_idx = indices.to(torch.uint8)
            v_norms = v_flat.norm(dim=-1)

            if self._key_indices is None:
                self._key_indices = k_vec_idx
                self._key_norms = k_norms
                self._value_indices = v_vec_idx
                self._value_norms = v_norms
            else:
                self._key_indices = torch.cat([self._key_indices, k_vec_idx], dim=0)
                self._key_norms = torch.cat([self._key_norms, k_norms], dim=0)
                self._value_indices = torch.cat([self._value_indices, v_vec_idx], dim=0)
                self._value_norms = torch.cat([self._value_norms, v_norms], dim=0)

            self._residual_keys = self._residual_keys[..., overflow:, :]
            self._residual_values = self._residual_values[..., overflow:, :]

        # 解量化
        if self._key_indices is not None and self._key_indices.numel() > 0:
            iq = self._get_quantizer(self._head_dim, self.device)
            k_deq = iq.dequantize(self._key_indices.long(), self._key_norms).to(self.dtype)
            k_deq = k_deq.reshape(self._residual_keys.shape[0], self._residual_keys.shape[1],
                                  -1, self._head_dim)
            v_deq = iq.dequantize(self._value_indices.long(), self._value_norms).to(self.dtype)
            v_deq = v_deq.reshape(self._residual_values.shape[0], self._residual_values.shape[1],
                                  -1, self._head_dim)
            self.keys = torch.cat([k_deq, self._residual_keys], dim=-2)
            self.values = torch.cat([v_deq, self._residual_values], dim=-2)
        else:
            self.keys = self._residual_keys
            self.values = self._residual_values

        return self.keys, self.values

    def get_seq_length(self):
        return self._total_len


class PlanarQuantLayer(DynamicLayer):
    """缓存层存储PlanarQuant压缩的索引+范数。"""

    def __init__(self, bits: int = 3, residual_len: int = 128):
        super().__init__()
        self.bits = bits
        self.residual_len = residual_len
        self._quantizers = {}
        self._key_indices = None
        self._key_norms = None
        self._value_indices = None
        self._value_norms = None
        self._residual_keys = None
        self._residual_values = None
        self._total_len = 0
        self._head_dim = None

    def _get_quantizer(self, head_dim, device):
        key = (head_dim, str(device))
        if key not in self._quantizers:
            self._quantizers[key] = PlanarQuantMSE(head_dim, self.bits, device=str(device))
        return self._quantizers[key]

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self._head_dim = key_states.shape[-1]
        self._residual_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self._residual_values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._residual_keys = torch.cat([self._residual_keys, key_states], dim=-2)
        self._residual_values = torch.cat([self._residual_values, value_states], dim=-2)
        self._total_len += key_states.shape[-2]

        if self._residual_keys.shape[-2] > self.residual_len:
            overflow = self._residual_keys.shape[-2] - self.residual_len
            to_q_k = self._residual_keys[..., :overflow, :]
            to_q_v = self._residual_values[..., :overflow, :]

            pq = self._get_quantizer(self._head_dim, self.device)

            k_flat = to_q_k.reshape(-1, self._head_dim).float()
            x_hat, indices = pq(k_flat)
            k_vec_idx = indices.to(torch.uint8)
            k_norms = k_flat.norm(dim=-1)

            v_flat = to_q_v.reshape(-1, self._head_dim).float()
            x_hat, indices = pq(v_flat)
            v_vec_idx = indices.to(torch.uint8)
            v_norms = v_flat.norm(dim=-1)

            if self._key_indices is None:
                self._key_indices = k_vec_idx
                self._key_norms = k_norms
                self._value_indices = v_vec_idx
                self._value_norms = v_norms
            else:
                self._key_indices = torch.cat([self._key_indices, k_vec_idx], dim=0)
                self._key_norms = torch.cat([self._key_norms, k_norms], dim=0)
                self._value_indices = torch.cat([self._value_indices, v_vec_idx], dim=0)
                self._value_norms = torch.cat([self._value_norms, v_norms], dim=0)

            self._residual_keys = self._residual_keys[..., overflow:, :]
            self._residual_values = self._residual_values[..., overflow:, :]

        # 解量化
        if self._key_indices is not None and self._key_indices.numel() > 0:
            pq = self._get_quantizer(self._head_dim, self.device)
            k_deq = pq.dequantize(self._key_indices.long(), self._key_norms).to(self.dtype)
            k_deq = k_deq.reshape(self._residual_keys.shape[0], self._residual_keys.shape[1],
                                  -1, self._head_dim)
            v_deq = pq.dequantize(self._value_indices.long(), self._value_norms).to(self.dtype)
            v_deq = v_deq.reshape(self._residual_values.shape[0], self._residual_values.shape[1],
                                  -1, self._head_dim)
            self.keys = torch.cat([k_deq, self._residual_keys], dim=-2)
            self.values = torch.cat([v_deq, self._residual_values], dim=-2)
        else:
            self.keys = self._residual_keys
            self.values = self._residual_values

        return self.keys, self.values

    def get_seq_length(self):
        return self._total_len


class RotorQuantLayer(DynamicLayer):
    """缓存层存储RotorQuant压缩的索引+范数。"""

    def __init__(self, bits: int = 3, residual_len: int = 128):
        super().__init__()
        self.bits = bits
        self.residual_len = residual_len
        self._quantizers = {}
        self._key_indices = None
        self._key_norms = None
        self._value_indices = None
        self._value_norms = None
        self._residual_keys = None
        self._residual_values = None
        self._total_len = 0
        self._head_dim = None

    def _get_quantizer(self, head_dim, device):
        key = (head_dim, str(device))
        if key not in self._quantizers:
            self._quantizers[key] = RotorQuantMSE(head_dim, self.bits, device=str(device))
        return self._quantizers[key]

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self._head_dim = key_states.shape[-1]
        self._residual_keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self._residual_values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._residual_keys = torch.cat([self._residual_keys, key_states], dim=-2)
        self._residual_values = torch.cat([self._residual_values, value_states], dim=-2)
        self._total_len += key_states.shape[-2]

        if self._residual_keys.shape[-2] > self.residual_len:
            overflow = self._residual_keys.shape[-2] - self.residual_len
            to_q_k = self._residual_keys[..., :overflow, :]
            to_q_v = self._residual_values[..., :overflow, :]

            rq = self._get_quantizer(self._head_dim, self.device)

            k_flat = to_q_k.reshape(-1, self._head_dim).float()
            x_hat, indices = rq(k_flat)
            k_vec_idx = indices.to(torch.uint8)
            k_norms = k_flat.norm(dim=-1)

            v_flat = to_q_v.reshape(-1, self._head_dim).float()
            x_hat, indices = rq(v_flat)
            v_vec_idx = indices.to(torch.uint8)
            v_norms = v_flat.norm(dim=-1)

            if self._key_indices is None:
                self._key_indices = k_vec_idx
                self._key_norms = k_norms
                self._value_indices = v_vec_idx
                self._value_norms = v_norms
            else:
                self._key_indices = torch.cat([self._key_indices, k_vec_idx], dim=0)
                self._key_norms = torch.cat([self._key_norms, k_norms], dim=0)
                self._value_indices = torch.cat([self._value_indices, v_vec_idx], dim=0)
                self._value_norms = torch.cat([self._value_norms, v_norms], dim=0)

            self._residual_keys = self._residual_keys[..., overflow:, :]
            self._residual_values = self._residual_values[..., overflow:, :]

        # 解量化
        if self._key_indices is not None and self._key_indices.numel() > 0:
            rq = self._get_quantizer(self._head_dim, self.device)
            k_deq = rq.dequantize(self._key_indices.long(), self._key_norms).to(self.dtype)
            k_deq = k_deq.reshape(self._residual_keys.shape[0], self._residual_keys.shape[1],
                                  -1, self._head_dim)
            v_deq = rq.dequantize(self._value_indices.long(), self._value_norms).to(self.dtype)
            v_deq = v_deq.reshape(self._residual_values.shape[0], self._residual_values.shape[1],
                                  -1, self._head_dim)
            self.keys = torch.cat([k_deq, self._residual_keys], dim=-2)
            self.values = torch.cat([v_deq, self._residual_values], dim=-2)
        else:
            self.keys = self._residual_keys
            self.values = self._residual_values

        return self.keys, self.values

    def get_seq_length(self):
        return self._total_len


class TurboQuantCache(DynamicCache):
    def __init__(self, bits=3, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        while len(self.layers) <= layer_idx:
            self.layers.append(TurboQuantLayer(bits=self.bits))
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


class IsoQuantCache(DynamicCache):
    def __init__(self, bits=3, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        while len(self.layers) <= layer_idx:
            self.layers.append(IsoQuantLayer(bits=self.bits))
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


class PlanarQuantCache(DynamicCache):
    def __init__(self, bits=3, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        while len(self.layers) <= layer_idx:
            self.layers.append(PlanarQuantLayer(bits=self.bits))
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


class RotorQuantCache(DynamicCache):
    def __init__(self, bits=3, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        while len(self.layers) <= layer_idx:
            self.layers.append(RotorQuantLayer(bits=self.bits))
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# 测评
# ═══════════════════════════════════════════════════════════════════════════════

def run_test(model, tokenizer, input_ids, cache_factory, label):
    """使用给定缓存运行前向传播，测量VRAM和速度。"""
    flush()
    mem_before = gpu_mem_mb()

    cache = cache_factory()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=cache, use_cache=True, labels=input_ids)
        loss = outputs.loss
        ppl = math.exp(min(loss.item(), 20)) if loss is not None else float('nan')
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak = gpu_peak_mb()
    mem_after = gpu_mem_mb()
    tok_s = input_ids.shape[1] / elapsed

    return {
        'label': label,
        'ppl': ppl,
        'peak_mb': peak,
        'cache_mb': mem_after - mem_before,
        'tok_s': tok_s,
    }


def compressed_bytes(n_vectors, head_dim, bits):
    """压缩表示（索引+范数）的字节数。所有方法使用相同的公式。"""
    # 索引: n_vectors * head_dim * bits / 8 字节（位打包）
    # 范数: n_vectors * 4 字节（float32）
    idx_bytes = n_vectors * head_dim * bits / 8
    norm_bytes = n_vectors * 4
    return idx_bytes + norm_bytes


def fp16_bytes(n_vectors, head_dim):
    return n_vectors * head_dim * 2  # fp16每个2字节


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--bits', nargs='+', type=int, default=[3, 4])
    parser.add_argument('--contexts', nargs='+', type=int, default=[460, 1860, 4096, 8192, 16384, 32768])
    args = parser.parse_args()

    print("=" * 95)
    print("  turboquantx VRAM节省: FP16 vs 四种量化算法 (KV缓存)")
    print("=" * 95)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    flush()
    mem_start = gpu_mem_mb()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map='cuda:0'
    )
    model.eval()

    model_mb = gpu_mem_mb() - mem_start
    gpu_name = torch.cuda.get_device_name(0)
    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // model.config.num_attention_heads)
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads

    print(f"  模型: {args.model} ({model_mb:.0f} MB)")
    print(f"  GPU: {gpu_name}")
    print(f"  架构: {n_layers} 层, {n_kv_heads} KV头, head_dim={head_dim}")

    # ── 第1部分：可行上下文下的真实模型PPL ───────────────────────
    print(f"\n{'─' * 95}")
    print("  第1部分: 真实PPL + 测量峰值VRAM (Qwen2.5-3B, 单次前向传播)")
    print(f"{'─' * 95}")

    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100])
    except Exception:
        text = "The quick brown fox jumps over the lazy dog. " * 10000

    all_ids = tokenizer.encode(text, return_tensors='pt').to('cuda:0')

    test_contexts = [c for c in [1024, 2048, 4096] if c <= all_ids.shape[1]]

    for bits in args.bits:
        print(f"\n  {bits}-bit:")
        print(f"  {'上下文':>8}  {'方法':<12} {'峰值VRAM':>12} {'速度':>12} {'PPL':>8}")
        print(f"  {'────────':>8}  {'────────────':<12} {'────────────':>12} {'────────────':>12} {'────────':>8}")

        for ctx in test_contexts:
            input_ids = all_ids[:, :ctx]

            fp16 = run_test(model, tokenizer, input_ids, lambda: DynamicCache(), 'FP16')
            tq = run_test(model, tokenizer, input_ids,
                         lambda b=bits: TurboQuantCache(bits=b), f'TQ {bits}b')
            iq = run_test(model, tokenizer, input_ids,
                         lambda b=bits: IsoQuantCache(bits=b), f'IQ {bits}b')
            pq = run_test(model, tokenizer, input_ids,
                         lambda b=bits: PlanarQuantCache(bits=b), f'PQ {bits}b')
            rq = run_test(model, tokenizer, input_ids,
                         lambda b=bits: RotorQuantCache(bits=b), f'RQ {bits}b')

            for r in [fp16, tq, iq, pq, rq]:
                print(f"  {ctx:>8}  {r['label']:<12} {r['peak_mb']:>10.0f} MB "
                      f"{r['tok_s']:>10.1f} t/s {r['ppl']:>8.2f}")

    # ── 第2部分：分析VRAM节省（真实情况） ─────────────────
    print(f"\n{'─' * 95}")
    print(f"  第2部分: KV缓存VRAM — FP16 vs 压缩存储")
    print(f"  (所有量化方法使用相同的压缩格式: uint8索引 + float32范数)")
    print(f"  模型配置: {n_layers} 层 × {n_kv_heads} KV头 × head_dim={head_dim}")
    print(f"{'─' * 95}")

    # 每个上下文token的n_kv_vectors = 2 (K+V) * n_layers * n_kv_heads
    vectors_per_token = 2 * n_layers * n_kv_heads

    for bits in args.bits:
        print(f"\n  {bits}-bit 压缩:")
        print(f"  {'上下文':>8}  {'FP16 KV':>10}  {f'TQ/IQ/PQ/RQ {bits}b':>12}  {'节省':>10}  {'比率':>8}")
        print(f"  {'────────':>8}  {'──────────':>10}  {'────────────':>12}  {'──────────':>10}  {'────────':>8}")

        for ctx in args.contexts:
            n_vecs = ctx * vectors_per_token
            fp16_mb = fp16_bytes(n_vecs, head_dim) / 1024**2
            comp_mb = compressed_bytes(n_vecs, head_dim, bits) / 1024**2
            saved = fp16_mb - comp_mb
            ratio = fp16_mb / comp_mb

            print(f"  {ctx:>8}  {fp16_mb:>8.1f} MB  {comp_mb:>10.1f} MB  {saved:>8.1f} MB  {ratio:>7.1f}x")

    # ── 第3部分：量化器状态优势 ──────────────────────────
    print(f"\n{'─' * 95}")
    print(f"  第3部分: 量化器状态VRAM (存储一次，在所有token间共享)")
    print(f"{'─' * 95}")

    for bits in args.bits:
        # TurboQuant: d×d旋转矩阵（float32）+ 每层码本
        tq_per_layer = (head_dim * head_dim * 4 + 2**bits * 4)  # 字节
        tq_total = tq_per_layer * n_layers * n_kv_heads  # 每个层×头一个量化器

        # IsoQuant: 旋转参数 + 码本
        iq_obj = IsoQuantMSE(head_dim, bits, device='cpu')
        iq_per_layer = (iq_obj.rotation_params * 4 + 2**bits * 4)
        iq_total = iq_per_layer * n_layers * n_kv_heads

        # PlanarQuant: 旋转参数 + 码本
        pq_obj = PlanarQuantMSE(head_dim, bits, device='cpu')
        pq_per_layer = (pq_obj.rotation_params * 4 + 2**bits * 4)
        pq_total = pq_per_layer * n_layers * n_kv_heads

        # RotorQuant: 旋转参数 + 码本
        rq_obj = RotorQuantMSE(head_dim, bits, device='cpu')
        rq_per_layer = (rq_obj.rotation_params * 4 + 2**bits * 4)
        rq_total = rq_per_layer * n_layers * n_kv_heads

        print(f"\n  {bits}-bit 量化器状态 ({n_layers} 层 × {n_kv_heads} KV头):")
        print(f"    TurboQuant:   {tq_total/1024:.1f} KB  ({head_dim}×{head_dim} 旋转矩阵)")
        print(f"    IsoQuant:     {iq_total/1024:.1f} KB  — {tq_total/iq_total:.0f}x 更小")
        print(f"    PlanarQuant:  {pq_total/1024:.1f} KB  — {tq_total/pq_total:.0f}x 更小")
        print(f"    RotorQuant:   {rq_total/1024:.1f} KB  — {tq_total/rq_total:.0f}x 更小")

    print(f"\n{'=' * 95}")
    print("  测评完成")
    print(f"{'=' * 95}")


if __name__ == '__main__':
    main()