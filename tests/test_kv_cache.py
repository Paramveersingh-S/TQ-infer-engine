"""
Integration tests for KV cache compression.
These tests use a tiny GPT-2 model (no GPU needed).
"""

import pytest
import torch

try:
    from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from tqe.algorithms.turbo_quant import TurboQuantizer

pytestmark = pytest.mark.skipif(
    not HAS_TRANSFORMERS,
    reason="transformers not installed"
)


@pytest.fixture
def small_model():
    """Tiny 2-layer GPT-2 for KV cache tests (no real weights needed)."""
    config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=1000)
    return GPT2LMHeadModel(config).eval()


def test_turboquant_kvcache_shapes():
    """
    Verify that TurboQuantizer encode/decode on realistic KV shapes works correctly.
    Shape: (batch=2, kv_heads=4, seq_len=32, head_dim=64)
    """
    batch, heads, seq, head_dim = 2, 4, 32, 64
    tq = TurboQuantizer(input_dim=head_dim, total_bits_per_dim=4.0)
    v = torch.randn(batch * heads * seq, head_dim)
    codes = tq.encode(v)
    v_hat = tq.decode(codes)
    assert v_hat.shape == v.shape


def test_kvcache_memory_reduced():
    """After quantization, TurboQuant memory < FP16 baseline."""
    head_dim = 64
    n_vectors = 1000
    tq = TurboQuantizer(input_dim=head_dim, total_bits_per_dim=4.0)
    compressed_mem = tq.memory_bytes(n_vectors)
    fp16_mem = n_vectors * head_dim * 2
    assert compressed_mem < fp16_mem, (
        f"Compressed memory {compressed_mem} ≥ FP16 {fp16_mem}"
    )


def test_kvcache_decode_roundtrip():
    """Store 100 random (k, v) pairs, retrieve and check MSE is finite."""
    torch.manual_seed(42)
    head_dim = 64
    n = 100
    tq = TurboQuantizer(input_dim=head_dim, total_bits_per_dim=4.0)
    keys = torch.randn(n, head_dim)
    codes = tq.encode(keys)
    keys_hat = tq.decode(codes)
    mse = ((keys - keys_hat) ** 2).mean()
    assert torch.isfinite(mse), "MSE should be finite"
    assert mse.item() < keys.var().item(), "MSE should be less than signal variance"


def test_kvcache_compression_ratio_target():
    """4-bit TurboQuant should give ≥ 3.5× compression vs FP16."""
    tq = TurboQuantizer(input_dim=128, total_bits_per_dim=4.0)
    ratio = tq.compression_ratio(1000, original_dtype_bytes=2)
    assert ratio >= 3.5, f"Compression ratio {ratio:.2f}× < 3.5×"


def test_kvcache_inner_products_finite():
    """estimate_inner_products should return finite values."""
    torch.manual_seed(5)
    head_dim = 128
    n = 200
    tq = TurboQuantizer(input_dim=head_dim, total_bits_per_dim=4.0)
    keys = torch.randn(n, head_dim)
    queries = torch.randn(n, head_dim)
    codes = tq.encode(keys)
    scores = tq.estimate_inner_products(queries, codes)
    assert torch.isfinite(scores).all(), "Inner product estimates contain NaN/Inf"
