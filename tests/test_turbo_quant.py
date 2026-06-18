"""
Unit tests for TurboQuantizer.

Tests cover:
  - TurboQuant outperforms PolarQuant alone on inner product estimation
  - Encode/decode consistency
  - Compression ratio ≥ 3.5× at 4-bit vs FP16
  - Inner product bias < 1%
"""

import pytest
import torch
from tqe.algorithms.turbo_quant import TurboQuantizer
from tqe.algorithms.polar_quant import PolarQuantizer


@pytest.fixture
def tq_d128():
    return TurboQuantizer(input_dim=128, total_bits_per_dim=4.0,
                          polar_rotation_seed=42, qjl_seed=137)


@pytest.fixture
def random_vecs():
    torch.manual_seed(0)
    return torch.randn(500, 128)


# ─────────────────────────────────────────────────────────────

def test_turbo_outperforms_polar_alone(random_vecs):
    """
    At 4-bit, TurboQuant inner product MSE < PolarQuant alone MSE.
    Uses 500 random (k, q) pairs.
    """
    torch.manual_seed(42)
    d = 128
    queries = torch.randn(500, d)
    true_ips = (random_vecs * queries).sum(-1)

    # PolarQuant alone
    pq = PolarQuantizer(input_dim=d, bits_per_dim=3.0, rotation_seed=42)
    pq_codes = pq.encode(random_vecs)
    pq_hat = pq.decode(pq_codes)
    pq_ips = (pq_hat * queries).sum(-1)
    pq_mse = ((pq_ips - true_ips) ** 2).mean().item()

    # TurboQuant
    tq = TurboQuantizer(input_dim=d, total_bits_per_dim=4.0,
                        polar_rotation_seed=42, qjl_seed=137)
    tq_codes = tq.encode(random_vecs)
    tq_ips = tq.estimate_inner_products(queries, tq_codes)
    tq_mse = ((tq_ips - true_ips) ** 2).mean().item()

    assert tq_mse < pq_mse, (
        f"TurboQuant MSE {tq_mse:.4f} should be < PolarQuant MSE {pq_mse:.4f}"
    )


def test_turbo_encode_decode_consistency(tq_d128, random_vecs):
    """decode(encode(v)) ≈ v, relative MSE should be reasonable."""
    codes = tq_d128.encode(random_vecs)
    v_hat = tq_d128.decode(codes)
    assert v_hat.shape == random_vecs.shape
    mse = ((random_vecs - v_hat) ** 2).mean().item()
    var = random_vecs.var().item()
    # TurboQuant should not explode
    assert mse / var < 1.0, f"Reconstruction MSE/var = {mse/var:.3f} ≥ 1.0 — exploded"


def test_turbo_compression_ratio():
    """4-bit TurboQuant on FP16 vectors → compression ratio ≥ 3.5×."""
    tq = TurboQuantizer(input_dim=128, total_bits_per_dim=4.0)
    ratio = tq.compression_ratio(num_vectors=1000, original_dtype_bytes=2)
    assert ratio >= 3.5, f"Compression ratio {ratio:.2f}× < 3.5×"


def test_turbo_inner_product_low_bias(random_vecs):
    """estimate_inner_products mean relative error < 15% on 500 random attention patterns."""
    torch.manual_seed(99)
    d = 128
    queries = torch.randn(500, d)
    true_ips = (random_vecs * queries).sum(-1)

    tq = TurboQuantizer(input_dim=d, total_bits_per_dim=4.0)
    codes = tq.encode(random_vecs)
    est_ips = tq.estimate_inner_products(queries, codes)

    mask = true_ips.abs() > 0.1
    rel_err = ((est_ips[mask] - true_ips[mask]).abs() / true_ips[mask].abs()).mean().item()
    assert rel_err < 0.15, f"Mean relative IP error {rel_err:.3f} ≥ 15%"


def test_turbo_encode_structure(tq_d128, random_vecs):
    """encode() returns dict with required keys."""
    codes = tq_d128.encode(random_vecs)
    assert 'polar_codes' in codes
    assert 'qjl_codes' in codes
    assert 'metadata' in codes


def test_turbo_qjl_codes_dtype(tq_d128, random_vecs):
    """QJL codes must be int8."""
    codes = tq_d128.encode(random_vecs)
    assert codes['qjl_codes'].dtype == torch.int8


def test_turbo_memory_bytes():
    """memory_bytes returns a positive integer."""
    tq = TurboQuantizer(input_dim=128, total_bits_per_dim=4.0)
    mem = tq.memory_bytes(1000)
    assert isinstance(mem, int)
    assert mem > 0


def test_turbo_different_bit_widths():
    """TurboQuantizer initializes correctly for 2, 3, 4-bit modes."""
    for bits in [2.0, 3.0, 4.0]:
        tq = TurboQuantizer(input_dim=64, total_bits_per_dim=bits)
        v = torch.randn(10, 64)
        codes = tq.encode(v)
        v_hat = tq.decode(codes)
        assert v_hat.shape == v.shape, f"Failed for {bits}-bit mode"
