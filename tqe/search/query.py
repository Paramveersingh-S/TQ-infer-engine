"""
Recall evaluation and search benchmarking utilities.

Metrics:
  recall_at_k(true_indices, approx_indices, k):
    = |true_top_k ∩ approx_top_k| / k
    Averaged over all queries.
  
  This is the exact metric used in the TurboQuant paper (1@k recall on GloVe).
"""

import time
import numpy as np
import torch
from typing import Optional


def compute_recall_at_k(
    index_to_test,
    ground_truth_index,
    queries: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Compute recall@k for approximate search vs. exact ground truth.

    Args:
        index_to_test:      TurboQuantIndex or FAISS index
        ground_truth_index: FAISS IndexFlatIP (exact search) — must have .search() method
        queries:            (q, dim) float tensor
        k:                  number of neighbors to consider
    Returns:
        recall: float in [0, 1]
    """
    q_np = queries.cpu().numpy() if isinstance(queries, torch.Tensor) else queries
    q_torch = queries if isinstance(queries, torch.Tensor) else torch.from_numpy(queries)

    # Ground truth (exact search)
    _, gt_indices = ground_truth_index.search(q_np.astype(np.float32), k)

    # Approximate search
    if hasattr(index_to_test, 'search'):
        # Check if it's our TurboQuantIndex (returns tensors) or FAISS (returns numpy)
        result = index_to_test.search(q_torch, k)
        if isinstance(result[1], torch.Tensor):
            approx_indices = result[1].cpu().numpy()
        else:
            _, approx_indices = result
    else:
        raise ValueError(f"Unknown index type: {type(index_to_test)}")

    # Compute recall@k
    recalls = []
    for i in range(len(queries)):
        gt_set = set(gt_indices[i].tolist())
        approx_set = set(approx_indices[i].tolist())
        recall = len(gt_set & approx_set) / k
        recalls.append(recall)

    return float(np.mean(recalls))


def benchmark_search_speed(
    index,
    queries: torch.Tensor,
    k: int = 10,
    n_repeats: int = 100,
) -> dict:
    """
    Benchmark search speed for a given index.

    Args:
        index:      TurboQuantIndex
        queries:    (q, dim) float tensor
        k:          number of neighbors
        n_repeats:  number of timed repetitions
    Returns:
        dict: {
          'mean_latency_ms': float,
          'p99_latency_ms': float,
          'queries_per_second': float
        }
    """
    latencies = []

    # Warm-up
    for _ in range(5):
        index.search(queries, k)

    for _ in range(n_repeats):
        t0 = time.perf_counter()
        index.search(queries, k)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))
    p99_latency = float(np.percentile(latencies, 99))
    qps = (len(queries) * n_repeats) / (np.sum(latencies) / 1000.0)

    return {
        'mean_latency_ms': mean_latency,
        'p99_latency_ms': p99_latency,
        'queries_per_second': float(qps),
    }
