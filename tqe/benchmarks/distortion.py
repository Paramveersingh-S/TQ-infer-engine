"""
MSE / inner-product distortion benchmarks.

Theoretical lower bounds for comparison:
  - Rate-distortion theory: D(R) ≥ σ² * 2^{-2R}  (for Gaussian sources)
    where R = bits/dim, σ² = signal variance
  - At 4 bits/dim: D_theory = σ² * 2^{-8} ≈ 0.0039 σ²
  - TurboQuant achieves within 2.7× of this
"""

import math
import time
from typing import List

import numpy as np
import torch

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from tqe.algorithms.qjl import QJLQuantizer
from tqe.algorithms.polar_quant import PolarQuantizer
from tqe.algorithms.turbo_quant import TurboQuantizer


def _uniform_quantize(v: torch.Tensor, bits: int) -> torch.Tensor:
    """Naive uniform scalar quantization (baseline)."""
    vmin = v.min(dim=-1, keepdim=True).values
    vmax = v.max(dim=-1, keepdim=True).values
    scale = (vmax - vmin).clamp(min=1e-8)
    n_levels = (2 ** bits) - 1
    q = torch.round((v - vmin) / scale * n_levels).clamp(0, n_levels)
    return (q / n_levels) * scale + vmin


def theoretical_distortion(sigma_sq: float, bits_per_dim: float) -> float:
    """
    Shannon rate-distortion lower bound: D*(R) = σ² · 2^{-2R}
    """
    return sigma_sq * (2 ** (-2 * bits_per_dim))


def benchmark_all_quantizers(
    d: int = 256,
    n_vectors: int = 10_000,
    bits_range: List[float] = [2, 3, 4, 5, 6, 8],
) -> "pd.DataFrame | list":
    """
    Compare:
      1. TurboQuant (our method)
      2. PolarQuant (first stage only)
      3. QJL (1-bit only, as a baseline)
      4. Uniform scalar quantization (naive baseline)
      5. Theoretical lower bound D(R)

    Return DataFrame with columns:
      method, bits_per_dim, mse_distortion, inner_product_mse,
      memory_bytes_per_vector, encoding_time_ms
    """
    torch.manual_seed(42)
    v = torch.randn(n_vectors, d)
    q = torch.randn(n_vectors, d)
    true_ips = (v * q).sum(-1)  # true inner products
    sigma_sq = float(v.var())

    rows = []

    for bits in bits_range:
        # --- TurboQuant ---
        tq = TurboQuantizer(d, total_bits_per_dim=float(bits), device='cpu')
        t0 = time.perf_counter()
        codes = tq.encode(v)
        enc_ms = (time.perf_counter() - t0) * 1000.0
        v_hat = tq.decode(codes)
        mse = float(((v - v_hat) ** 2).mean())
        ip_est = tq.estimate_inner_products(q, codes)
        ip_mse = float(((ip_est - true_ips) ** 2).mean())
        mem = tq.memory_bytes(1)
        rows.append({
            'method': 'TurboQuant',
            'bits_per_dim': bits,
            'mse_distortion': mse,
            'inner_product_mse': ip_mse,
            'memory_bytes_per_vector': mem,
            'encoding_time_ms': enc_ms / n_vectors,
        })

        # --- PolarQuant ---
        pq = PolarQuantizer(d, bits_per_dim=float(bits), device='cpu')
        t0 = time.perf_counter()
        pq_codes = pq.encode(v)
        enc_ms_pq = (time.perf_counter() - t0) * 1000.0
        v_pq_hat = pq.decode(pq_codes)
        mse_pq = float(((v - v_pq_hat) ** 2).mean())
        ip_pq = (v_pq_hat * q).sum(-1)
        ip_mse_pq = float(((ip_pq - true_ips) ** 2).mean())
        rows.append({
            'method': 'PolarQuant',
            'bits_per_dim': bits,
            'mse_distortion': mse_pq,
            'inner_product_mse': ip_mse_pq,
            'memory_bytes_per_vector': pq.memory_bytes(1),
            'encoding_time_ms': enc_ms_pq / n_vectors,
        })

        # --- Uniform ---
        t0 = time.perf_counter()
        v_unif = _uniform_quantize(v, int(bits))
        enc_ms_unif = (time.perf_counter() - t0) * 1000.0
        mse_unif = float(((v - v_unif) ** 2).mean())
        ip_unif = (v_unif * q).sum(-1)
        ip_mse_unif = float(((ip_unif - true_ips) ** 2).mean())
        rows.append({
            'method': 'UniformScalar',
            'bits_per_dim': bits,
            'mse_distortion': mse_unif,
            'inner_product_mse': ip_mse_unif,
            'memory_bytes_per_vector': math.ceil(d * bits / 8),
            'encoding_time_ms': enc_ms_unif / n_vectors,
        })

        # --- Theoretical bound ---
        rows.append({
            'method': 'Theoretical',
            'bits_per_dim': bits,
            'mse_distortion': theoretical_distortion(sigma_sq, bits),
            'inner_product_mse': None,
            'memory_bytes_per_vector': math.ceil(d * bits / 8),
            'encoding_time_ms': 0.0,
        })

    # QJL (1-bit only, regardless of requested bits)
    qjl = QJLQuantizer(d, d, device='cpu')
    t0 = time.perf_counter()
    qjl_codes = qjl.encode(v)
    enc_ms_qjl = (time.perf_counter() - t0) * 1000.0
    v_qjl_hat = qjl.decode_approximate(qjl_codes)
    mse_qjl = float(((v - v_qjl_hat) ** 2).mean())
    ip_qjl = qjl.estimate_inner_product(q, qjl_codes)
    ip_mse_qjl = float(((ip_qjl - true_ips) ** 2).mean())
    rows.append({
        'method': 'QJL',
        'bits_per_dim': 1,
        'mse_distortion': mse_qjl,
        'inner_product_mse': ip_mse_qjl,
        'memory_bytes_per_vector': qjl.memory_bytes(1),
        'encoding_time_ms': enc_ms_qjl / n_vectors,
    })

    if HAS_PANDAS:
        import pandas as pd
        return pd.DataFrame(rows)
    return rows
