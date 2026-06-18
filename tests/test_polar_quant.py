"""
Unit tests for PolarQuantizer.

Tests cover:
  - Rotation matrix orthogonality
  - Encode/decode roundtrip quality
  - Per-vector (not per-dimension) scale overhead
  - Batched operation support
"""

import pytest
import torch
from tqe.algorithms.polar_quant import PolarQuantizer


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def pq_d128_4bit():
    return PolarQuantizer(input_dim=128, bits_per_dim=4.0, rotation_seed=42)


@pytest.fixture
def random_vecs():
    torch.manual_seed(0)
    return torch.randn(200, 128)


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

def test_polar_rotation_orthogonal(pq_d128_4bit):
    """R @ R.T must be identity within tolerance 1e-5."""
    R = pq_d128_4bit.R
    eye = torch.eye(R.shape[0], dtype=R.dtype)
    err = (R @ R.T - eye).abs().max().item()
    assert err < 1e-5, f"Rotation matrix not orthogonal, max error = {err:.2e}"


def test_polar_encode_decode_roundtrip():
    """At 8-bit: decode(encode(v)) ≈ v with relative MSE < 5%."""
    torch.manual_seed(1)
    pq = PolarQuantizer(input_dim=64, bits_per_dim=8.0, rotation_seed=42)
    v = torch.randn(100, 64)
    codes = pq.encode(v)
    v_hat = pq.decode(codes)
    mse = ((v - v_hat) ** 2).mean().item()
    var = v.var().item()
    rel_mse = mse / var
    assert rel_mse < 0.05, f"8-bit roundtrip relative MSE {rel_mse:.4f} ≥ 5%"


def test_polar_encode_decode_4bit(pq_d128_4bit, random_vecs):
    """At 4-bit: MSE(decode(encode(v)), v) / var(v) < 0.15."""
    codes = pq_d128_4bit.encode(random_vecs)
    v_hat = pq_d128_4bit.decode(codes)
    mse = ((random_vecs - v_hat) ** 2).mean().item()
    var = random_vecs.var().item()
    rel_mse = mse / var
    assert rel_mse < 0.15, f"4-bit roundtrip relative MSE {rel_mse:.4f} ≥ 15%"


def test_polar_no_per_dimension_constants(pq_d128_4bit, random_vecs):
    """
    codes dict should have only 1 float per vector (the scale), NOT per-dimension.
    Scale shape: (..., 1) meaning shape[-1] == 1.
    """
    codes = pq_d128_4bit.encode(random_vecs)
    scale = codes['scale']
    # scale should have shape (n_vectors, 1) — NOT (n_vectors, d) or (n_vectors, d//2)
    assert scale.shape[-1] == 1, (
        f"scale shape {scale.shape} — should have last dim=1 (1 float per vector)"
    )


def test_polar_batched():
    """Works on high-dimensional batched tensors like (batch, seq, heads, head_dim)."""
    torch.manual_seed(3)
    pq = PolarQuantizer(input_dim=64, bits_per_dim=4.0, rotation_seed=7)
    v = torch.randn(4, 8, 10, 64)  # typical (batch, heads, seq, head_dim)
    codes = pq.encode(v)
    v_hat = pq.decode(codes)
    assert v_hat.shape == v.shape


def test_polar_odd_input_dim():
    """Handles odd input_dim via zero-padding."""
    torch.manual_seed(5)
    pq = PolarQuantizer(input_dim=65, bits_per_dim=4.0, rotation_seed=42)
    v = torch.randn(10, 65)
    codes = pq.encode(v)
    v_hat = pq.decode(codes)
    assert v_hat.shape == (10, 65)


def test_polar_memory_bytes():
    """Memory calculation includes angle bytes + radius bytes + scale float."""
    pq = PolarQuantizer(input_dim=128, bits_per_dim=4.0)
    mem = pq.memory_bytes(1000)
    # Rough sanity: should be less than 128 * 1000 * 2 (FP16 baseline)
    assert mem < 128 * 1000 * 2, f"PolarQuant memory {mem} exceeds FP16 baseline"
    assert mem > 0


def test_polar_encode_output_structure(pq_d128_4bit, random_vecs):
    """encode() returns dict with required keys."""
    codes = pq_d128_4bit.encode(random_vecs)
    required_keys = {'q_angles', 'q_radii', 'scale', 'input_dim', 'bits_per_dim'}
    assert required_keys <= set(codes.keys()), (
        f"Missing keys: {required_keys - set(codes.keys())}"
    )


def test_polar_q_angles_range(pq_d128_4bit, random_vecs):
    """Quantized angles must be in [0, 2^B_angle - 1]."""
    codes = pq_d128_4bit.encode(random_vecs)
    B = pq_d128_4bit.B_angle
    n_levels = (2 ** B) - 1
    assert codes['q_angles'].min().item() >= 0
    assert codes['q_angles'].max().item() <= n_levels


def test_polar_q_radii_dtype(pq_d128_4bit, random_vecs):
    """q_radii must be uint8."""
    codes = pq_d128_4bit.encode(random_vecs)
    assert codes['q_radii'].dtype == torch.uint8


def test_polar_deterministic(pq_d128_4bit, random_vecs):
    """Same input → same output every time (deterministic)."""
    codes1 = pq_d128_4bit.encode(random_vecs)
    codes2 = pq_d128_4bit.encode(random_vecs)
    assert torch.equal(codes1['q_angles'], codes2['q_angles'])
    assert torch.equal(codes1['q_radii'], codes2['q_radii'])
