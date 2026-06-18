"""
Unit tests for QJLQuantizer.

Tests cover:
  - Output shape and dtype
  - All codes are ±1 (no zeros)
  - Inner product unbiasedness
  - Zero per-vector overhead
  - Memory calculation
"""

import pytest
import torch
from tqe.algorithms.qjl import QJLQuantizer


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def qjl_d256():
    return QJLQuantizer(input_dim=256, proj_dim=256, seed=42)


@pytest.fixture
def random_vecs():
    torch.manual_seed(0)
    return torch.randn(100, 256)


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

def test_qjl_encode_shape(qjl_d256, random_vecs):
    """QJL(d=256, m=256).encode(randn(100, 256)) → shape (100, 256), dtype int8."""
    codes = qjl_d256.encode(random_vecs)
    assert codes.shape == (100, 256), f"Expected (100, 256), got {codes.shape}"
    assert codes.dtype == torch.int8, f"Expected int8, got {codes.dtype}"


def test_qjl_encode_values(qjl_d256, random_vecs):
    """All values must be exactly +1 or -1."""
    codes = qjl_d256.encode(random_vecs)
    unique = codes.unique()
    assert set(unique.tolist()) == {-1, 1}, (
        f"Codes should only contain ±1, got {unique.tolist()}"
    )


def test_qjl_inner_product_unbiased():
    """
    Generate 1000 random pairs (k, q) in ℝ^256.
    Estimate <k, q> with QJL, compute true <k, q>.
    Mean absolute relative error < 10%.
    """
    torch.manual_seed(42)
    d = 256
    n = 1000
    qjl = QJLQuantizer(input_dim=d, proj_dim=d, seed=42)

    keys = torch.randn(n, d)
    queries = torch.randn(n, d)
    true_ips = (keys * queries).sum(-1)

    key_codes = qjl.encode(keys)
    est_ips = qjl.estimate_inner_product(queries, key_codes)

    # Relative error (avoid division by near-zero)
    mask = true_ips.abs() > 0.1
    rel_err = ((est_ips[mask] - true_ips[mask]).abs() / true_ips[mask].abs()).mean()
    assert rel_err.item() < 0.10, f"Mean relative error {rel_err:.3f} ≥ 10%"


def test_qjl_zero_overhead(qjl_d256, random_vecs):
    """
    The quantizer stores NO per-vector constants.
    Codes returned by encode() should contain only int8 arrays, no floats.
    """
    codes = qjl_d256.encode(random_vecs)
    # codes is a single tensor (not a dict) — must be int8
    assert codes.dtype == torch.int8
    # No float data in the returned codes
    assert codes.is_floating_point() is False


def test_qjl_memory(qjl_d256):
    """memory_bytes(1000) for d=m=256 should be 256000 bytes."""
    assert qjl_d256.memory_bytes(1000) == 256 * 1000


def test_qjl_batched_shapes():
    """Works on higher-dimensional batched inputs."""
    torch.manual_seed(7)
    qjl = QJLQuantizer(input_dim=64, proj_dim=64, seed=1)
    v = torch.randn(4, 8, 16, 64)  # (batch, heads, seq, head_dim)
    codes = qjl.encode(v)
    assert codes.shape == (4, 8, 16, 64)
    assert codes.dtype == torch.int8


def test_qjl_decode_approximate_shape(qjl_d256, random_vecs):
    """decode_approximate returns shape (..., input_dim)."""
    codes = qjl_d256.encode(random_vecs)
    v_hat = qjl_d256.decode_approximate(codes)
    assert v_hat.shape == random_vecs.shape


def test_qjl_sign_zero_edge_case():
    """sign(0) must be mapped to +1, not left as 0."""
    qjl = QJLQuantizer(input_dim=4, proj_dim=4, seed=99)
    # Force a vector that, after projection, might produce zeros
    v = torch.zeros(1, 4)
    codes = qjl.encode(v)
    assert 0 not in codes.unique().tolist(), "sign(0) must be mapped to ±1"
