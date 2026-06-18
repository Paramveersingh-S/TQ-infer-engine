"""
TurboQuantIndex: Compressed ANN index using TurboQuant.

API mirrors FAISS IndexFlatIP (inner product) for easy comparison.

Usage:
    index = TurboQuantIndex(dim=200, bits_per_dim=4.0, device="cuda")
    index.add(vectors)               # (N, 200) float32 tensor
    D, I = index.search(query, k=10) # returns top-k distances and indices
    
    # Compare with FAISS:
    faiss_index = faiss.IndexFlatIP(200)
    faiss_index.add(vectors.numpy())
    D_faiss, I_faiss = faiss_index.search(query.numpy(), k=10)

Architecture:
  - Store N vectors in TurboQuant compressed form
  - Search: for each query, estimate inner products with ALL stored vectors
  - Return top-k by estimated inner product
  
  Optimization: Use batched matrix operations for all N inner product estimates.
  Avoid Python loops over N.
"""

import math
import torch

from tqe.algorithms.turbo_quant import TurboQuantizer


class TurboQuantIndex:
    """
    Approximate nearest-neighbor index backed by TurboQuant compressed vectors.
    Zero preprocessing overhead compared to FAISS-PQ codebook training.
    """

    def __init__(
        self,
        dim: int,
        bits_per_dim: float = 4.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        batch_size: int = 100_000,
    ):
        self.dim = dim
        self.bits_per_dim = bits_per_dim
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size  # chunked add() to prevent OOM

        self.quantizer = TurboQuantizer(
            input_dim=dim,
            total_bits_per_dim=bits_per_dim,
            device=device,
            dtype=dtype,
        )

        # Storage for compressed codes (list of per-batch code dicts)
        self._stored_polar_q_angles = []
        self._stored_polar_q_radii = []
        self._stored_polar_scales = []
        self._stored_qjl_codes = []
        self._n_total = 0

    def _check_memory(self, vectors: torch.Tensor) -> None:
        """Guard against OOM on GPU."""
        if self.device == "cuda" and torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0]
            required = vectors.nbytes * 2
            if required > free_mem * 0.8:
                raise MemoryError(
                    f"Adding {len(vectors)} vectors ({vectors.nbytes/1e9:.2f} GB) "
                    f"may exceed available GPU memory ({free_mem/1e9:.2f} GB free). "
                    f"Use index.add() in batches."
                )

    def add(self, vectors: torch.Tensor):
        """
        Add vectors to index.

        Args:
            vectors: (N, dim) float tensor
        Stores compressed codes, NOT original vectors.
        Chunks in batches of self.batch_size to prevent OOM.
        """
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim, (
            f"Expected (N, {self.dim}) tensor, got {vectors.shape}"
        )
        vectors = vectors.to(self.device)
        self._check_memory(vectors)

        with torch.no_grad():
            for start in range(0, len(vectors), self.batch_size):
                chunk = vectors[start : start + self.batch_size]
                codes = self.quantizer.encode(chunk)
                self._stored_polar_q_angles.append(codes['polar_codes']['q_angles'])
                self._stored_polar_q_radii.append(codes['polar_codes']['q_radii'])
                self._stored_polar_scales.append(codes['polar_codes']['scale'])
                self._stored_qjl_codes.append(codes['qjl_codes'])
                self._n_total += len(chunk)

    def _get_all_codes(self) -> dict:
        """Concatenate all stored codes into a single dict."""
        return {
            'polar_codes': {
                'q_angles': torch.cat(self._stored_polar_q_angles, dim=0),
                'q_radii':  torch.cat(self._stored_polar_q_radii, dim=0),
                'scale':    torch.cat(self._stored_polar_scales, dim=0),
                'input_dim': self.dim,
                'bits_per_dim': self.bits_per_dim - 1.0,
            },
            'qjl_codes': torch.cat(self._stored_qjl_codes, dim=0),
        }

    def search(self, query: torch.Tensor, k: int = 10):
        """
        Approximate nearest neighbor search.

        Args:
            query: (q, dim) float tensor — q query vectors
        Returns:
            distances: (q, k) estimated inner product scores (descending)
            indices:   (q, k) indices into the index
        """
        assert query.ndim == 2 and query.shape[1] == self.dim
        assert self._n_total > 0, "Index is empty. Call add() first."
        k = min(k, self._n_total)
        query = query.to(self.device)

        all_codes = self._get_all_codes()

        with torch.no_grad():
            # Decode all stored vectors once
            all_reconstructed = self.quantizer.polar.decode(all_codes['polar_codes'])  # (N, dim)
            all_qjl = all_codes['qjl_codes']  # (N, proj_dim)

            # Batched inner product estimation for all q queries against N stored vectors
            # polar_contribution: (q, N)
            polar_contribution = query.to(self.dtype) @ all_reconstructed.T

            # QJL contribution: (q, N)
            # proj_query: (q, proj_dim); all_qjl: (N, proj_dim)
            proj_query = query.to(self.dtype) @ self.quantizer.qjl.phi.T
            import math
            scale = (2.0 / math.pi) * (self.dim / self.quantizer.qjl.proj_dim)
            qjl_contribution = scale * (proj_query @ all_qjl.to(self.dtype).T)

            scores = polar_contribution + qjl_contribution  # (q, N)

        topk_vals, topk_idx = scores.topk(k, dim=-1)
        return topk_vals, topk_idx

    def ntotal(self) -> int:
        """Number of vectors in index."""
        return self._n_total

    def memory_bytes(self) -> int:
        """Total bytes used by the index."""
        return self.quantizer.memory_bytes(self._n_total)

    def reset(self):
        """Clear the index."""
        self._stored_polar_q_angles.clear()
        self._stored_polar_q_radii.clear()
        self._stored_polar_scales.clear()
        self._stored_qjl_codes.clear()
        self._n_total = 0
