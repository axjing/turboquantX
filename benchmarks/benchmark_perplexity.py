#!/usr/bin/env python3
"""
turboquantx 困惑度测评

基于rotorquant/benchmark_perplexity.py设计，测量KV缓存量化对语言建模质量的影响。

测试方法：在wikitext-2测试集上运行模型前向传播，计算困惑度。
对于turboquantx：修补DynamicCache以在预填充后量化键（与poc_high_context.py相同策略）。

用法：
    python -m benchmarks.benchmark_perplexity
    python -m benchmarks.benchmark_perplexity --model Qwen/Qwen2.5-7B-Instruct --bits 2 3 4
"""

import torch
import torch.nn.functional as F
import math
import time
import gc
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_perplexity(model, tokenizer, dataset_text, max_length=2048, stride=512, device="cuda"):
    """使用滑动窗口计算文本的困惑度。
    
    使用Hugging Face困惑度文档中的标准方法：
    使用stride重叠滑动max_length个token的窗口。
    """
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    n_tokens = 0

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - begin - 1  # 我们评分的token

        chunk_ids = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(chunk_ids, use_cache=False)
            logits = outputs.logits

        # 仅评分非重叠部分（第一个窗口除外）
        shift = max(0, max_length - stride) if begin > 0 else 0
        shift_logits = logits[:, shift:-1, :].contiguous()
        shift_labels = chunk_ids[:, shift + 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        n_scored = shift_labels.numel()
        nlls.append(loss.item())
        n_tokens += n_scored

        if end >= seq_len:
            break

    ppl = math.exp(sum(nlls) / n_tokens)
    return ppl, n_tokens


def _make_compressor(backend, bits, device):
    """为给定后端创建键压缩函数。
    
    返回函数：compress(keys_tensor, layer_idx) -> quantized_keys
    """
    compressors = {}

    if backend == 'turboquant':
        from turboquantx.quantizers.turboquant import TurboQuantMSE
        
        def compress(ks, li):
            D = ks.shape[-1]
            if li not in compressors:
                compressors[li] = TurboQuantMSE(D, bits, device=device)
            tq = compressors[li]
            flat = ks.reshape(-1, D).float()
            x_hat, _ = tq(flat)
            return x_hat.to(ks.dtype).reshape(ks.shape)

    elif backend == 'isoquant':
        from turboquantx.quantizers.isoquant import IsoQuantMSE
        
        def compress(ks, li):
            D = ks.shape[-1]
            if li not in compressors:
                compressors[li] = IsoQuantMSE(D, bits, device=device)
            iq = compressors[li]
            flat = ks.reshape(-1, D).float()
            x_hat, _ = iq(flat)
            return x_hat.to(ks.dtype).reshape(ks.shape)

    elif backend == 'planarquant':
        from turboquantx.quantizers.planarquant import PlanarQuantMSE
        
        def compress(ks, li):
            D = ks.shape[-1]
            if li not in compressors:
                compressors[li] = PlanarQuantMSE(D, bits, device=device)
            pq = compressors[li]
            flat = ks.reshape(-1, D).float()
            x_hat, _ = pq(flat)
            return x_hat.to(ks.dtype).reshape(ks.shape)

    elif backend == 'rotorquant':
        from turboquantx.quantizers.rotorquant import RotorQuantMSE
        
        def compress(ks, li):
            D = ks.shape[-1]
            if li not in compressors:
                compressors[li] = RotorQuantMSE(D, bits, device=device)
            rq = compressors[li]
            flat = ks.reshape(-1, D).float()
            x_hat, _ = rq(flat)
            return x_hat.to(ks.dtype).reshape(ks.shape)

    else:
        raise ValueError(f"未知后端: {backend}")

    return compress


def compute_perplexity_with_quant(model, tokenizer, dataset_text, bits=3,
                                  max_length=2048, stride=512, device="cuda",
                                  backend="turboquant"):
    """使用量化KV缓存压缩计算困惑度。
    
    在前向传播期间量化键，以便注意力看到量化键。
    这测量了KV缓存量化的实际质量影响。
    
    backend: 'turboquant', 'isoquant', 'planarquant', 'rotorquant'
    """
    from transformers import DynamicCache

    compress = _make_compressor(backend, bits, device)

    _orig = DynamicCache.update

    def _patched(self, ks, vs, li, ck=None):
        # 在前向传播期间量化键 - 注意力看到量化键
        kq = compress(ks, li)
        return _orig(self, kq, vs, li, ck)

    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    n_tokens = 0

    DynamicCache.update = _patched

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        chunk_ids = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(chunk_ids, use_cache=True)
            logits = outputs.logits

        shift = max(0, max_length - stride) if begin > 0 else 0
        shift_logits = logits[:, shift:-1, :].contiguous()
        shift_labels = chunk_ids[:, shift + 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nlls.append(loss.item())
        n_tokens += shift_labels.numel()

        del outputs
        torch.cuda.empty_cache()

        if end >= seq_len:
            break

    DynamicCache.update = _orig

    ppl = math.exp(sum(nlls) / n_tokens)
    return ppl, n_tokens


def main():
    parser = argparse.ArgumentParser(description="turboquantx KV缓存量化困惑度测评")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--backends", type=str, nargs="+",
                        default=["turboquant", "isoquant", "planarquant", "rotorquant"],
                        help="要测评的后端")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="限制数据集token数（0=完整测试集）")
    args = parser.parse_args()

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    import logging; logging.disable(logging.WARNING)

    backend_names = {
        'turboquant': 'TurboQuant',
        'isoquant': 'IsoQuant',
        'planarquant': 'PlanarQuant',
        'rotorquant': 'RotorQuant',
    }

    print()
    print("=" * 75)
    print("  turboquantx KV缓存量化困惑度测评 (wikitext-2)")
    print(f"  模型: {args.model}")
    print(f"  比特: {args.bits}")
    print(f"  后端: {[backend_names.get(b, b) for b in args.backends]}")
    print(f"  窗口: {args.max_length}, 步长: {args.stride}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print("=" * 75)

    # 加载数据集
    print("\n加载wikitext-2...", flush=True)
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    # 加载模型
    print("加载模型...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()

    if args.max_tokens > 0:
        tokens = tokenizer(text, return_tensors="pt")
        text = tokenizer.decode(tokens.input_ids[0][:args.max_tokens])

    total_tokens = len(tokenizer(text).input_ids)
    print(f"数据集: {total_tokens:,} tokens")
    print()

    # FP16基线
    print("计算FP16基线困惑度...", flush=True)
    t0 = time.perf_counter()
    ppl_fp16, n_tok = compute_perplexity(
        model, tokenizer, text,
        max_length=args.max_length, stride=args.stride,
    )
    t_fp16 = time.perf_counter() - t0
    print(f"  FP16:     困惑度 = {ppl_fp16:.2f}  ({n_tok:,} tokens, {t_fp16:.1f}s)")
    print()

    # 所有后端 × 所有比特宽度
    print("  往返量化（在前向传播期间量化键）：")
    print(f"  {'方法':>20s}  {'困惑度':>8s}  {'差值':>8s}  {'%变化':>8s}  {'时间':>8s}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'FP16':>20s}  {ppl_fp16:>8.2f}  {'—':>8s}  {'—':>8s}  {t_fp16:>6.1f}s")

    for bits in args.bits:
        for backend in args.backends:
            torch.cuda.empty_cache()
            gc.collect()

            name = f"{backend_names.get(backend, backend)} {bits}b"

            try:
                t0 = time.perf_counter()
                ppl, n_tok = compute_perplexity_with_quant(
                    model, tokenizer, text, bits=bits,
                    max_length=args.max_length, stride=args.stride,
                    backend=backend,
                )
                t_elapsed = time.perf_counter() - t0

                delta = ppl - ppl_fp16
                pct = (ppl - ppl_fp16) / ppl_fp16 * 100

                print(f"  {name:>20s}  {ppl:>8.2f}  {delta:>+8.2f}  {pct:>+7.1f}%  {t_elapsed:>6.1f}s")
            except Exception as e:
                print(f"  {name:>20s}  {'错误':>8s}  {str(e)[:40]}")

    print()
    print("=" * 75)
    print("完成")
    print("=" * 75)


if __name__ == "__main__":
    main()