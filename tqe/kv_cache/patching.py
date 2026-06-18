"""
Model patching utilities for transparent KV cache compression.
Provides helper functions to apply/remove TurboQuant patches to HuggingFace models.
"""

import warnings
from typing import Optional

import torch

from tqe.algorithms.turbo_quant import TurboQuantizer
from tqe.kv_cache.hooks import TurboQuantKVCache

SUPPORTED_ARCH = {"LlamaForCausalLM", "MistralForCausalLM", "Gemma2ForCausalLM"}


def get_head_dim(attn_module) -> int:
    """
    Infer head_dim from an attention module's k_proj linear layer.
    Handles both standard and grouped-query attention.
    """
    k_proj = getattr(attn_module, 'k_proj', None)
    if k_proj is None:
        raise ValueError(f"Cannot find k_proj in {type(attn_module).__name__}")
    num_kv_heads = getattr(attn_module, 'num_key_value_heads', None)
    if num_kv_heads is None:
        num_kv_heads = getattr(attn_module, 'num_heads', 1)
    return k_proj.out_features // num_kv_heads


def build_layer_quantizers(
    model,
    bits_per_dim: float,
    device: str,
) -> dict:
    """
    Walk model and build a TurboQuantizer per attention layer.

    Returns:
        dict mapping layer_idx (int) → TurboQuantizer
    """
    model_type = type(model).__name__
    if model_type not in SUPPORTED_ARCH:
        warnings.warn(
            f"Model {model_type} is not in tested architectures {SUPPORTED_ARCH}.",
            UserWarning,
        )

    quantizers = {}
    layer_idx = 0
    for name, module in model.named_modules():
        if hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
            try:
                head_dim = get_head_dim(module)
            except Exception:
                continue
            quantizers[layer_idx] = TurboQuantizer(
                input_dim=head_dim,
                total_bits_per_dim=bits_per_dim,
                device=device,
            )
            layer_idx += 1
    return quantizers


def create_compressed_cache(
    model,
    bits_per_dim: float = 4.0,
    compress_keys: bool = True,
    compress_values: bool = True,
    device: str = "cpu",
) -> TurboQuantKVCache:
    """
    Create a TurboQuantKVCache populated with per-layer quantizers.

    Usage:
        cache = create_compressed_cache(model, bits_per_dim=4.0)
        outputs = model.generate(..., past_key_values=cache)
    """
    quantizers = build_layer_quantizers(model, bits_per_dim, device)
    return TurboQuantKVCache(
        quantizers=quantizers,
        compress_keys=compress_keys,
        compress_values=compress_values,
    )
