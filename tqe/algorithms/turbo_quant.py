"""
TurboQuant: Two-Stage Online Vector Quantizer
Reference: arXiv:2504.19874 — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
Presented at: ICLR 2026

Two-Stage Pipeline:
  Stage 1 (PolarQuant): Encode v → codes_pq using (bits_per_dim - 1) bits/dim
  Stage 2 (QJL):        Compute residual e = v - PolarQuant.decode(codes_pq)
                        Encode e → codes_qjl using 1 bit/dim via QJL
  
  Decode: v_hat = PolarQuant.decode(codes_pq) + QJL.decode_approximate(codes_qjl)

  Inner product estimation (the key application):
    <v_k, v_q> ≈ PolarQuant_ip(codes_pq_k, v_q) + QJL.estimate_inner_product(v_q, codes_qjl_k)
    where PolarQuant_ip is the inner product with the PolarQuant reconstruction
"""

import torch
from tqe.algorithms.qjl import QJLQuantizer
from tqe.algorithms.polar_quant import PolarQuantizer


class TurboQuantizer:
    """
    Two-stage vector quantizer combining PolarQuant (main) and QJL (residual corrector).
    Achieves near-optimal distortion within ~2.7× of the Shannon rate-distortion bound.
    """

    def __init__(
        self,
        input_dim: int,
        total_bits_per_dim: float = 4.0,
        qjl_proj_dim: int = None,
        polar_rotation_seed: int = 42,
        qjl_seed: int = 137,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        total_bits_per_dim=4.0 means:
          - PolarQuant gets 3 bits/dim  (captures main structure)
          - QJL gets 1 bit/dim          (corrects residual bias)

        total_bits_per_dim=3.5 means:
          - PolarQuant gets 2.5 bits/dim
          - QJL gets 1 bit/dim
        """
        self.input_dim = input_dim
        self.total_bits_per_dim = total_bits_per_dim
        self.device = device
        self.dtype = dtype

        # PolarQuant uses (total - 1) bits/dim
        polar_bits = max(2.0, total_bits_per_dim - 1.0)
        self.polar = PolarQuantizer(
            input_dim=input_dim,
            bits_per_dim=polar_bits,
            rotation_seed=polar_rotation_seed,
            device=device,
            dtype=dtype,
        )

        qjl_dim = qjl_proj_dim or input_dim
        self.qjl = QJLQuantizer(
            input_dim=input_dim,
            proj_dim=qjl_dim,
            dtype=dtype,
            device=device,
            seed=qjl_seed,
        )

    def _check_device(self, tensor: torch.Tensor, name: str = "input") -> None:
        if tensor.device.type != self.device:
            raise ValueError(
                f"{name} is on {tensor.device} but quantizer is on {self.device}. "
                f"Move with: tensor.to('{self.device}')"
            )

    def encode(self, v: torch.Tensor) -> dict:
        """
        Full TurboQuant encoding.

        Args:
            v: (..., input_dim)
        Returns:
            {
              'polar_codes': dict from PolarQuantizer.encode(),
              'qjl_codes':   torch.Tensor int8 (..., qjl_proj_dim),
              'metadata':    {'input_dim', 'total_bits_per_dim', 'device'}
            }
        """
        self._check_device(v, "v")
        assert v.shape[-1] == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got {v.shape[-1]}."
        )
        with torch.no_grad():
            # Stage 1: PolarQuant
            codes_pq = self.polar.encode(v)
            v_polar_hat = self.polar.decode(codes_pq)

            # Stage 2: Residual via QJL — CRITICAL: compute residual in original space
            residual = v.to(self.dtype) - v_polar_hat
            codes_qjl = self.qjl.encode(residual)

        return {
            'polar_codes': codes_pq,
            'qjl_codes': codes_qjl,
            'metadata': {
                'input_dim': self.input_dim,
                'total_bits_per_dim': self.total_bits_per_dim,
                'device': self.device,
            }
        }

    def decode(self, codes: dict) -> torch.Tensor:
        """
        Approximate reconstruction.
        Returns: v_hat = polar_reconstruct + qjl_approximate_reconstruct
        """
        with torch.no_grad():
            v_polar_hat = self.polar.decode(codes['polar_codes'])
            v_qjl_hat = self.qjl.decode_approximate(codes['qjl_codes'])
        return v_polar_hat + v_qjl_hat

    def estimate_inner_products(self, query: torch.Tensor, key_codes: dict) -> torch.Tensor:
        """
        THE CORE OPERATION — fast inner product estimation for attention.

        Args:
            query:      (..., input_dim)  full precision query vector
            key_codes:  dict from encode()  (batch of compressed keys)
        Returns:
            scores:     (...,)  estimated attention logits

        Steps:
          1. polar_contribution = (self.polar.decode(key_codes['polar_codes']) * query).sum(-1)
          2. qjl_contribution = self.qjl.estimate_inner_product(query, key_codes['qjl_codes'])
          3. return polar_contribution + qjl_contribution

        NOTE: The QJL contribution CORRECTS the bias from PolarQuant — mathematically proven.
        """
        self._check_device(query, "query")
        with torch.no_grad():
            polar_reconstruct = self.polar.decode(key_codes['polar_codes'])
            polar_contribution = (polar_reconstruct * query.to(self.dtype)).sum(-1)
            qjl_contribution = self.qjl.estimate_inner_product(query, key_codes['qjl_codes'])
        return polar_contribution + qjl_contribution

    def memory_bytes(self, num_vectors: int) -> int:
        """Total bytes for polar codes + qjl codes."""
        return self.polar.memory_bytes(num_vectors) + self.qjl.memory_bytes(num_vectors)

    def compression_ratio(self, num_vectors: int, original_dtype_bytes: int = 2) -> float:
        """
        Compression ratio vs. original storage.

        Args:
            num_vectors:          number of vectors
            original_dtype_bytes: 2 for float16, 4 for float32
        Returns: original_bytes / compressed_bytes
        """
        original = num_vectors * self.input_dim * original_dtype_bytes
        return original / self.memory_bytes(num_vectors)
