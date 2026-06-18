"""
Random rotations, Beta distribution tools, and other math utilities for TurboQuant.
"""

import math
import torch
import numpy as np


def haar_orthogonal_matrix(dim: int, seed: int = 42, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generate a Haar-distributed (uniformly random) orthogonal matrix via QR decomposition.

    Args:
        dim:  matrix dimension (d × d)
        seed: random seed for reproducibility
        dtype: tensor dtype
    Returns:
        Q: (dim, dim) orthogonal tensor
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    G = torch.randn(dim, dim, generator=gen, dtype=dtype)
    Q, R = torch.linalg.qr(G)
    # Sign-fix for Haar distribution
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)
    Q.requires_grad_(False)
    return Q


def block_diagonal_rotation(dim: int, block_size: int = 512, seed: int = 42,
                             dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Memory-efficient block-diagonal random rotation for large d (d > 8192).

    Instead of a single d×d rotation, applies independent orthogonal rotations in blocks.
    This avoids O(d²) memory overhead for very large dimensions.

    Args:
        dim:        total dimension
        block_size: size of each diagonal block
        seed:       random seed
        dtype:      tensor dtype
    Returns:
        Callable that applies the block-diagonal rotation to a tensor
    """
    blocks = []
    n_full = dim // block_size
    remainder = dim % block_size
    for i in range(n_full):
        blocks.append(haar_orthogonal_matrix(block_size, seed=seed + i, dtype=dtype))
    if remainder > 0:
        blocks.append(haar_orthogonal_matrix(remainder, seed=seed + n_full, dtype=dtype))
    return torch.block_diag(*blocks)


def theoretical_distortion(sigma_sq: float, bits_per_dim: float) -> float:
    """
    Shannon rate-distortion lower bound for Gaussian sources.
    D*(R) = σ² · 2^{-2R}

    Args:
        sigma_sq:     source variance
        bits_per_dim: R (bits per dimension)
    Returns:
        D*(R): theoretical minimum distortion
    """
    return sigma_sq * (2.0 ** (-2.0 * bits_per_dim))


def inner_product_error(
    v_keys: torch.Tensor,
    v_queries: torch.Tensor,
    v_keys_hat: torch.Tensor,
) -> dict:
    """
    Compute inner product estimation error statistics.

    Args:
        v_keys:     (N, d) true key vectors
        v_queries:  (N, d) query vectors
        v_keys_hat: (N, d) reconstructed key vectors
    Returns:
        dict with 'mean_abs_error', 'mean_rel_error', 'mse'
    """
    true_ips = (v_keys * v_queries).sum(-1)
    est_ips = (v_keys_hat * v_queries).sum(-1)
    abs_err = (est_ips - true_ips).abs()
    rel_err = abs_err / (true_ips.abs().clamp(min=1e-8))
    mse = ((est_ips - true_ips) ** 2).mean()
    return {
        'mean_abs_error': float(abs_err.mean()),
        'mean_rel_error': float(rel_err.mean()),
        'mse': float(mse),
    }
