"""
LLM perplexity + memory benchmarks for KV cache compression.

Evaluates perplexity on WikiText-2 with and without TurboQuant compression.
GPU is required for full LLM inference.
"""

import math
import os
from typing import List

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from tqe.kv_cache.compressor import KVCacheCompressor


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    max_tokens: int = 2048,
    device: str = "cuda",
    stride: int = 512,
) -> float:
    """
    Compute perplexity on a text string using sliding-window evaluation.

    Args:
        model:      HuggingFace causal LM
        tokenizer:  corresponding tokenizer
        text:       raw text to evaluate
        max_tokens: maximum sequence length
        device:     "cuda" or "cpu"
        stride:     sliding window stride
    Returns:
        perplexity: float
    """
    encodings = tokenizer(text, return_tensors='pt')
    seq_len = encodings.input_ids.shape[1]
    seq_len = min(seq_len, max_tokens)
    input_ids = encodings.input_ids[:, :seq_len].to(device)

    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + stride, seq_len)
        target_len = end - prev_end
        chunk_ids = input_ids[:, begin:end]
        with torch.no_grad():
            outputs = model(chunk_ids, labels=chunk_ids)
        nll = outputs.loss * target_len
        nlls.append(nll)
        prev_end = end
        if end >= seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = math.exp(total_nll.item() / seq_len)
    return ppl


def run_kv_benchmark(
    model_name: str = "google/gemma-2-2b-it",
    bits_list: List[float] = [2.0, 3.0, 4.0, 16.0],
    max_tokens: int = 2048,
    device: str = "cuda",
) -> list:
    """
    Run perplexity + memory benchmark for multiple bit-widths.

    Returns list of result dicts with:
      method, bits_per_dim, perplexity, kv_memory_gb, compression_ratio
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers and datasets packages are required.")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    model.eval()

    # Load WikiText-2
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(dataset["text"])

    results = []

    for bits in bits_list:
        if bits == 16.0:
            # Baseline FP16
            ppl = compute_perplexity(model, tokenizer, text, max_tokens, device)
            if device == "cuda":
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
            else:
                mem_gb = 0.0
            results.append({
                'method': 'Baseline FP16',
                'bits_per_dim': 16.0,
                'perplexity': ppl,
                'kv_memory_gb': mem_gb,
                'compression_ratio': 1.0,
            })
        else:
            compressor = KVCacheCompressor(model, bits_per_dim=bits, device=device)
            compressor.patch_model()
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ppl = compute_perplexity(model, tokenizer, text, max_tokens, device)
            if device == "cuda":
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
            else:
                mem_gb = 0.0
            ratio = compressor.stats()['compression_ratio']
            compressor.unpatch_model()
            results.append({
                'method': f'TurboQuant {bits}bit',
                'bits_per_dim': bits,
                'perplexity': ppl,
                'kv_memory_gb': mem_gb,
                'compression_ratio': ratio,
            })
        print(f"bits={bits}: PPL={results[-1]['perplexity']:.2f}, "
              f"ratio={results[-1]['compression_ratio']:.1f}×")

    return results
