"""
GPU memory profiling utilities for TurboQuant benchmarking.
"""

import torch
from typing import Optional, Callable
import contextlib
import functools


def get_gpu_memory_info(device: int = 0) -> dict:
    """
    Get current GPU memory usage.

    Returns:
        dict with 'free_gb', 'total_gb', 'used_gb', 'peak_gb'
    """
    if not torch.cuda.is_available():
        return {'free_gb': 0.0, 'total_gb': 0.0, 'used_gb': 0.0, 'peak_gb': 0.0}

    free, total = torch.cuda.mem_get_info(device)
    used = total - free
    peak = torch.cuda.max_memory_allocated(device)
    return {
        'free_gb': free / 1e9,
        'total_gb': total / 1e9,
        'used_gb': used / 1e9,
        'peak_gb': peak / 1e9,
    }


@contextlib.contextmanager
def memory_tracker(device: int = 0, label: str = ""):
    """
    Context manager to track GPU memory usage during a block.

    Usage:
        with memory_tracker(label="encode") as mem:
            codes = quantizer.encode(v)
        print(mem['delta_gb'])
    """
    if not torch.cuda.is_available():
        yield {'before_gb': 0.0, 'after_gb': 0.0, 'delta_gb': 0.0, 'peak_gb': 0.0}
        return

    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    mem_dict = {}
    try:
        yield mem_dict
    finally:
        after = torch.cuda.memory_allocated(device)
        peak = torch.cuda.max_memory_allocated(device)
        mem_dict.update({
            'label': label,
            'before_gb': before / 1e9,
            'after_gb': after / 1e9,
            'delta_gb': (after - before) / 1e9,
            'peak_gb': peak / 1e9,
        })


def estimate_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # float16
) -> dict:
    """
    Estimate theoretical KV cache memory for an LLM.

    Args:
        num_layers:   number of attention layers
        num_kv_heads: number of key-value heads (GQA: can be < num_heads)
        head_dim:     dimension per head
        seq_len:      sequence length
        batch_size:   batch size
        dtype_bytes:  bytes per element (2=fp16, 4=fp32)
    Returns:
        dict with 'total_gb', 'per_layer_gb', 'breakdown'
    """
    # Keys + values each: (batch, kv_heads, seq_len, head_dim) * dtype_bytes
    per_kv_bytes = batch_size * num_kv_heads * seq_len * head_dim * dtype_bytes
    total_bytes = 2 * num_layers * per_kv_bytes  # factor 2 for K and V
    return {
        'total_gb': total_bytes / 1e9,
        'per_layer_gb': (2 * per_kv_bytes) / 1e9,
        'breakdown': {
            'num_layers': num_layers,
            'num_kv_heads': num_kv_heads,
            'head_dim': head_dim,
            'seq_len': seq_len,
            'batch_size': batch_size,
        }
    }
