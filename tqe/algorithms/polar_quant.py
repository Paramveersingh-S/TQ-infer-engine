"""
PolarQuant: Polar Coordinate Quantizer
Reference: arXiv:2502.02617 — "PolarQuant: Leveraging Polar Decomposition for KV Cache Quantization"

Mathematics:
  Given vector v ∈ ℝ^d (after random rotation):
  1. Group pairs: (v_{2i}, v_{2i+1}) for i = 0, ..., d//2 - 1
  2. Compute radius: r_i = sqrt(v_{2i}^2 + v_{2i+1}^2)
  3. Compute angle:  θ_i = atan2(v_{2i+1}, v_{2i})  ∈ [-π, π]
  4. After random rotation, θ_i is near-uniform — quantize with B_angle bits using uniform quantizer
  5. Radii are sparse/concentrated — apply recursive polar transform, store final radius at B_radius bits
  6. Total bits per pair: B_angle + B_radius/pair

  Zero overhead: angle range [-π, π] is fixed, no per-block scaling needed.

Quantizer levels:
  - 2-bit: B_angle=1, B_radius=1
  - 3-bit: B_angle=2, B_radius=1 (per pair, = 1.5 bits/dim)
  - 4-bit: B_angle=3, B_radius=1 (per pair, = 2 bits/dim)  ← SWEET SPOT
  - 8-bit: B_angle=7, B_radius=1 (per pair, = 4 bits/dim)
"""

import math
import torch


class PolarQuantizer:
    """
    Polar coordinate quantizer with Haar random rotation for near-uniform angle distribution.
    """

    # Mapping: bits_per_dim → (B_angle, B_radius) per 2-element pair
    BITS_CONFIG = {
        2.0: (1, 1),
        3.0: (2, 1),
        4.0: (3, 1),
        8.0: (7, 1),
    }

    def __init__(
        self,
        input_dim: int,
        bits_per_dim: float = 4.0,
        rotation_seed: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        The random rotation matrix R ∈ ℝ^{d×d} is generated at init via Gram-Schmidt
        on a random Gaussian matrix (Haar distribution = uniformly random orthogonal matrix).
        Store R as a buffer (not a parameter).

        bits_per_dim controls B_angle + B_radius split:
          2.0  → B_angle=1, B_radius=1 per pair  (= 1 bit/dim)
          3.0  → B_angle=2, B_radius=1 per pair  (= 1.5 bits/dim)
          4.0  → B_angle=3, B_radius=1 per pair  (= 2 bits/dim)
          8.0  → B_angle=7, B_radius=1 per pair  (= 4 bits/dim)
        """
        self.input_dim = input_dim
        self.bits_per_dim = bits_per_dim
        self.device = device
        self.dtype = dtype
        self.rotation_seed = rotation_seed

        # Resolve closest supported bit config
        closest_key = min(self.BITS_CONFIG.keys(), key=lambda k: abs(k - bits_per_dim))
        self.B_angle, self.B_radius = self.BITS_CONFIG[closest_key]

        # Handle odd input_dim by working with padded dim
        self._padded_dim = input_dim if input_dim % 2 == 0 else input_dim + 1

        # Generate rotation matrix
        self.R = self._generate_rotation(self._padded_dim, rotation_seed).to(device)

    def _generate_rotation(self, dim: int, seed: int) -> torch.Tensor:
        """
        Generate a Haar-distributed random orthogonal matrix.
        Algorithm: QR decomposition of a random Gaussian matrix.
        Steps:
          1. G = torch.randn(dim, dim, generator=...)
          2. Q, R = torch.linalg.qr(G)
          3. Sign-fix: Q *= sign(diag(R))  ← ensures Haar distribution
          4. Return Q
        """
        gen = torch.Generator()
        gen.manual_seed(seed)
        G = torch.randn(dim, dim, generator=gen, dtype=self.dtype)
        Q, R_mat = torch.linalg.qr(G)
        # Sign-fix for Haar distribution
        signs = torch.sign(torch.diag(R_mat))
        Q = Q * signs.unsqueeze(0)
        Q.requires_grad_(False)
        return Q

    def _check_device(self, tensor: torch.Tensor, name: str = "input") -> None:
        if tensor.device.type != self.device:
            raise ValueError(
                f"{name} is on {tensor.device} but quantizer is on {self.device}. "
                f"Move with: tensor.to('{self.device}')"
            )

    def _angle_quantize(self, angles: torch.Tensor, n_bits: int) -> torch.Tensor:
        """
        Uniform quantization of angles ∈ [-π, π] to n_bits levels.
        n_levels = 2^n_bits
        Step size: Δ = 2π / n_levels
        Quantized: q_angle = round((angle + π) / Δ) clipped to [0, n_levels-1]
        Return as integer tensor.
        """
        n_levels = 2 ** n_bits
        delta = 2.0 * math.pi / n_levels
        q = torch.round((angles + math.pi) / delta).long()
        q = q.clamp(0, n_levels - 1)
        return q

    def _angle_dequantize(self, q_angles: torch.Tensor, n_bits: int) -> torch.Tensor:
        """Inverse of _angle_quantize. Return angles ∈ [-π, π]."""
        n_levels = 2 ** n_bits
        delta = 2.0 * math.pi / n_levels
        return q_angles.to(self.dtype) * delta - math.pi + delta / 2.0

    def _radius_quantize(self, radii: torch.Tensor, n_bits: int):
        """
        Quantize radii using per-vector max normalization.
        Steps:
          1. scale = radii.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
          2. normalized = radii / scale  (now in [0, 1])
          3. q_radii = round(normalized * (2^n_bits - 1)).clamp(0, 2^n_bits - 1).to(uint8)
        Return (q_radii, scale). Scale is the ONLY per-vector constant stored.
        NOTE: This is the unavoidable 1 overhead float per vector (not per dimension).
        """
        scale = radii.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        normalized = radii / scale
        levels = (2 ** n_bits) - 1
        q_radii = torch.round(normalized * levels).clamp(0, levels).to(torch.uint8)
        return q_radii, scale

    def _radius_dequantize(self, q_radii: torch.Tensor, scale: torch.Tensor, n_bits: int) -> torch.Tensor:
        """Inverse of _radius_quantize."""
        levels = (2 ** n_bits) - 1
        return (q_radii.to(self.dtype) / levels) * scale

    def encode(self, v: torch.Tensor) -> dict:
        """
        Full PolarQuant encode pipeline.

        Args:
            v: (..., input_dim) float tensor
        Returns:
            dict with keys:
              'q_angles':  (..., input_dim//2) int tensor  [B_angle bits used]
              'q_radii':   (..., input_dim//2) uint8 tensor
              'scale':     (..., 1) float32 tensor (per-vector normalization constant)
              'input_dim': int
              'bits_per_dim': float
        """
        self._check_device(v, "v")
        assert v.shape[-1] == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got {v.shape[-1]}."
        )
        orig_shape = v.shape
        v_f = v.to(self.dtype)

        # Zero-pad to even dimension if needed
        if self.input_dim % 2 != 0:
            pad = torch.zeros(*v_f.shape[:-1], 1, dtype=self.dtype, device=v_f.device)
            v_f = torch.cat([v_f, pad], dim=-1)

        # Step 1: random rotation
        rotated = v_f @ self.R.T  # (..., padded_dim)

        # Step 2: reshape to pairs (..., pairs, 2)
        pairs = rotated.reshape(*rotated.shape[:-1], self._padded_dim // 2, 2)

        # Step 3: compute radii and angles
        v1 = pairs[..., 0]
        v2 = pairs[..., 1]
        radii = torch.sqrt(v1 ** 2 + v2 ** 2).clamp(min=1e-8)  # numerical stability
        angles = torch.atan2(v2, v1)  # (..., pairs)

        # Step 4: quantize angles and radii
        q_angles = self._angle_quantize(angles, self.B_angle)
        q_radii, scale = self._radius_quantize(radii, self.B_radius)

        return {
            'q_angles': q_angles,
            'q_radii': q_radii,
            'scale': scale.to(torch.float32),
            'input_dim': self.input_dim,
            'bits_per_dim': self.bits_per_dim,
        }

    def decode(self, codes: dict) -> torch.Tensor:
        """
        Reconstruct approximate vector from polar codes.

        Steps:
          1. Dequantize angles → angles_hat
          2. Dequantize radii → radii_hat (using stored scale)
          3. Reconstruct pairs: v_{2i} = r_i * cos(θ_i), v_{2i+1} = r_i * sin(θ_i)
          4. Flatten pairs back to (..., input_dim)
          5. Apply INVERSE rotation: v_hat = reconstructed @ self.R  (R is orthogonal, so R^{-1} = R.T)
          6. Return v_hat
        """
        q_angles = codes['q_angles']
        q_radii = codes['q_radii']
        scale = codes['scale']

        # Dequantize
        angles_hat = self._angle_dequantize(q_angles, self.B_angle)
        radii_hat = self._radius_dequantize(q_radii, scale, self.B_radius)

        # Reconstruct pairs
        v1_hat = radii_hat * torch.cos(angles_hat)
        v2_hat = radii_hat * torch.sin(angles_hat)

        # Stack pairs (..., pairs, 2) then flatten → (..., padded_dim)
        reconstructed = torch.stack([v1_hat, v2_hat], dim=-1)
        reconstructed = reconstructed.reshape(*reconstructed.shape[:-2], self._padded_dim)

        # Inverse rotation: R^T is inverse since R is orthogonal
        v_hat = reconstructed @ self.R  # (..., padded_dim)

        # Strip padding if needed
        if self.input_dim % 2 != 0:
            v_hat = v_hat[..., :self.input_dim]

        return v_hat.to(self.dtype)

    def memory_bytes(self, num_vectors: int) -> int:
        """
        Compute total bytes for storing num_vectors with PolarQuant.
        Account for: angle bits + radius bits + scale floats.
        Use ceiling division for bit packing.
        """
        pairs = self._padded_dim // 2
        # angle bits per vector
        angle_bits = pairs * self.B_angle
        angle_bytes = math.ceil(angle_bits / 8)
        # radius bits per vector (stored as uint8 = 1 byte each)
        radius_bytes = pairs * 1  # 1 byte per uint8
        # scale: 1 float32 per vector
        scale_bytes = 4
        return num_vectors * (angle_bytes + radius_bytes + scale_bytes)
