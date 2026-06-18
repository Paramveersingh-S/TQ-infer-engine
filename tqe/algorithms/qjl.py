"""
QJL: Quantized Johnson-Lindenstrauss Transform
Reference: arXiv:2406.03482 — "Quantized Johnson-Lindenstrauss Transform for LLM KV Cache"

Mathematics:
  - Random matrix Φ ∈ ℝ^{m×d}, entries i.i.d. N(0, 1/m)
  - Encoding: q = sign(Φ @ v)   where v ∈ ℝ^d, q ∈ {-1, +1}^m
  - Inner product estimation: <v_k, v_q> ≈ (2/π) * (d/m) * <Φ @ v_q, q_k>
    (full precision query, 1-bit key)
  - Unbiasedness: E[estimator] = <v_k, v_q>  (proven in paper)
  - Overhead: ZERO — sign bits need no quantization constants
"""

import math
import torch


class QJLQuantizer:
    """
    Quantized Johnson-Lindenstrauss transform for efficient 1-bit inner product estimation.
    
    The random projection matrix Φ is generated once at init with a fixed seed for
    reproducibility across model layers.
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        seed: int = 42,
    ):
        """
        Args:
            input_dim: d — dimension of input vectors
            proj_dim:  m — dimension of projected space (m ≥ d for accuracy, m = d is typical)
            dtype:     precision for the random projection matrix
            device:    "cpu" or "cuda"
            seed:      fixed seed for reproducibility per layer
        """
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.dtype = dtype
        self.device = device
        self.seed = seed

        # Generate and STORE random projection matrix Φ at init time
        # Shape: (proj_dim, input_dim); entries i.i.d. N(0, 1/m)
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.phi = torch.randn(proj_dim, input_dim, generator=gen, dtype=dtype) / math.sqrt(proj_dim)
        self.phi = self.phi.to(device)
        self.phi.requires_grad_(False)

    def _check_device(self, tensor: torch.Tensor, name: str = "input") -> None:
        if tensor.device.type != self.device:
            raise ValueError(
                f"{name} is on {tensor.device} but quantizer is on {self.device}. "
                f"Move with: tensor.to('{self.device}')"
            )

    def encode(self, v: torch.Tensor) -> torch.Tensor:
        """
        Encode vector(s) to 1-bit sign representation.

        Args:
            v: (..., input_dim) tensor of vectors to quantize
        Returns:
            q: (..., proj_dim) tensor of dtype torch.int8, values in {-1, +1}
        """
        self._check_device(v, "v")
        assert v.shape[-1] == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got {v.shape[-1]}."
        )
        # projected: (..., proj_dim)
        projected = v.to(self.dtype) @ self.phi.T
        signs = projected.sign().to(torch.int8)
        # sign(0) = 0 — map 0 → +1
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return signs

    def estimate_inner_product(self, query: torch.Tensor, key_codes: torch.Tensor) -> torch.Tensor:
        """
        Estimate inner product <key, query> from stored 1-bit key codes.

        Args:
            query:      (..., input_dim)   full-precision query vector
            key_codes:  (..., proj_dim)    int8 sign codes for keys
        Returns:
            scores:     (...,)             estimated inner products
        """
        self._check_device(query, "query")
        # proj_query: (..., proj_dim)
        proj_query = query.to(self.dtype) @ self.phi.T
        scores = (2.0 / math.pi) * (self.input_dim / self.proj_dim) * (
            proj_query * key_codes.to(self.dtype)
        ).sum(-1)
        return scores

    def decode_approximate(self, q: torch.Tensor) -> torch.Tensor:
        """
        Approximate reconstruction of original vector from sign codes.
        Used for residual computation in TurboQuant.

        Args:
            q: (..., proj_dim) sign codes (int8)
        Returns:
            v_hat: (..., input_dim) approximate reconstruction
        """
        self._check_device(q, "q")
        # v_hat = (2/π) * Φ.T @ q  (pseudoinverse approximation via JL)
        v_hat = (2.0 / math.pi) * (q.to(self.dtype) @ self.phi)
        return v_hat

    def memory_bytes(self, num_vectors: int) -> int:
        """Return bytes used to store num_vectors encoded with QJL."""
        # proj_dim bits per vector, stored as int8 (1 byte per dim — not bit-packed)
        return num_vectors * self.proj_dim
