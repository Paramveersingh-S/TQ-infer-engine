"""
HuggingFace-compatible attention hooks for transparent KV cache compression.

The hook intercepts the attention computation AFTER keys/values are projected
but BEFORE they are stored or used for attention score computation.

Architecture-specific notes:
  - Llama/Mistral: uses GroupedQueryAttention (GQA) — handle KV head count ≠ Q head count
  - Gemma: same as Llama
  - The hook receives (keys, values) as (..., kv_heads, seq_len, head_dim) tensors

Hook flow:
  Strategy: monkey-patch the forward method of each attention layer to:
    1. Run original forward to get k, v projections
    2. If it's a "new" token (cache_position matters): compress k, v and store compressed
    3. For attention score computation: decompress (or use estimate_inner_products directly)

Important: Use CUSTOM KV CACHE CLASS approach (cleaner than hooks):
  Subclass DynamicCache and override update() to compress on write, read to decompress.
"""

import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = object  # fallback

from tqe.algorithms.turbo_quant import TurboQuantizer


class TurboQuantKVCache(DynamicCache if DynamicCache is not object else object):
    """
    DynamicCache subclass that stores KV in TurboQuant compressed form.

    HuggingFace's DynamicCache interface:
      update(key_states, value_states, layer_idx, cache_kwargs) → (key_states, value_states)
      get_seq_length(layer_idx) → int

    Our override (Mode 1 - DECODE_RETURN):
      update():
        1. Compress key_states and value_states with layer-specific TurboQuantizer
        2. Store compressed codes in self.key_codes[layer_idx] and self.value_codes[layer_idx]
        3. Return DECOMPRESSED states (so the rest of attention works normally)
    """

    def __init__(
        self,
        quantizers: dict,
        compress_keys: bool = True,
        compress_values: bool = True,
    ):
        """
        Args:
            quantizers: dict mapping layer_idx → TurboQuantizer
            compress_keys: compress key states
            compress_values: compress value states
        """
        if DynamicCache is not object:
            super().__init__()
        self.quantizers = quantizers
        self.compress_keys = compress_keys
        self.compress_values = compress_values
        self.key_codes: dict = {}
        self.value_codes: dict = {}
        self._seq_lengths: dict = {}

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int, cache_kwargs=None):
        """
        Compress k/v on write, return decompressed states for attention computation.
        """
        quantizer = self.quantizers.get(layer_idx)

        if quantizer is not None and self.compress_keys:
            # key_states: (batch, kv_heads, seq_len, head_dim)
            orig_shape = key_states.shape
            k_flat = key_states.reshape(-1, orig_shape[-1]).to(quantizer.dtype)
            k_codes = quantizer.encode(k_flat)
            k_decoded = quantizer.decode(k_codes).reshape(orig_shape)
            # Store codes for possible later use
            self.key_codes[layer_idx] = k_codes
            key_states = k_decoded.to(key_states.dtype)

        if quantizer is not None and self.compress_values:
            orig_shape = value_states.shape
            v_flat = value_states.reshape(-1, orig_shape[-1]).to(quantizer.dtype)
            v_codes = quantizer.encode(v_flat)
            v_decoded = quantizer.decode(v_codes).reshape(orig_shape)
            self.value_codes[layer_idx] = v_codes
            value_states = v_decoded.to(value_states.dtype)

        self._seq_lengths[layer_idx] = key_states.shape[-2]
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_lengths.get(layer_idx, 0)
