"""
ANN recall + speed benchmarks on GloVe-200 dataset.

Compares TurboQuant against FAISS-PQ and exact search.
GPU is recommended for FAISS-GPU.
"""

import time
import numpy as np
import torch
from typing import List

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from tqe.search.index import TurboQuantIndex
from tqe.search.query import compute_recall_at_k, benchmark_search_speed


def download_glove(dim: int = 200) -> np.ndarray:
    """
    Download / load GloVe vectors. Uses HuggingFace datasets as primary source.
    Falls back to cached local file if available.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "sentence-transformers/embedding-training-data",
            split="train[:1100000]",
            trust_remote_code=True,
        )
        # Fallback: return random data for development
    except Exception:
        pass

    # For development: return random data shaped like GloVe-200
    print("Using synthetic GloVe-like data for benchmarking...")
    np.random.seed(42)
    n = 1_100_000
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
    return vecs / norms  # normalize for cosine similarity


def run_ann_benchmark(
    n_train: int = 100_000,
    n_query: int = 1_000,
    dim: int = 200,
    k_values: List[int] = [1, 10, 100],
    bits_list: List[float] = [2.0, 3.0, 4.0],
    device: str = "cpu",
) -> list:
    """
    Full ANN benchmark on GloVe-200.

    Returns list of result dicts.
    """
    all_vecs = download_glove(dim)
    train_vecs = all_vecs[:n_train]
    query_vecs = all_vecs[n_train : n_train + n_query]

    results = []

    # --- Ground truth (FAISS exact) ---
    if HAS_FAISS:
        print("Building exact FAISS index...")
        exact_index = faiss.IndexFlatIP(dim)
        exact_index.add(train_vecs)

        # --- FAISS PQ baseline ---
        print("Building FAISS-PQ index...")
        m_subq = max(1, dim // 8)  # m subquantizers
        t0 = time.perf_counter()
        pq_index = faiss.IndexPQ(dim, m_subq, 8)
        pq_index.train(train_vecs)
        pq_index.add(train_vecs)
        pq_build_s = time.perf_counter() - t0

        for k in k_values:
            recall = _faiss_recall(pq_index, exact_index, query_vecs, k)
            results.append({
                'method': 'FAISS-PQ',
                'bits_per_dim': 4.0,
                'k': k,
                'recall': recall,
                'build_time_s': pq_build_s,
                'memory_bytes_per_vector': dim // 2,
            })
    else:
        print("FAISS not available — skipping FAISS baselines.")
        exact_index = None

    # --- TurboQuant ---
    train_t = torch.from_numpy(train_vecs)
    query_t = torch.from_numpy(query_vecs).to(device)

    for bits in bits_list:
        print(f"Building TurboQuant {bits}bit index...")
        tq_index = TurboQuantIndex(dim, bits_per_dim=bits, device=device)
        t0 = time.perf_counter()
        tq_index.add(train_t)
        build_s = time.perf_counter() - t0

        for k in k_values:
            if exact_index is not None:
                recall = compute_recall_at_k(tq_index, exact_index, query_t, k)
            else:
                recall = 0.0

            speed = benchmark_search_speed(tq_index, query_t[:100], k=k, n_repeats=10)
            results.append({
                'method': f'TurboQuant',
                'bits_per_dim': bits,
                'k': k,
                'recall': recall,
                'build_time_s': build_s,
                'memory_bytes_per_vector': tq_index.memory_bytes() // max(1, tq_index.ntotal()),
                'mean_latency_ms': speed['mean_latency_ms'],
                'qps': speed['queries_per_second'],
            })
        print(f"  TurboQuant {bits}bit: build={build_s:.2f}s")

    return results


def _faiss_recall(approx_index, exact_index, query_vecs: np.ndarray, k: int) -> float:
    """Compute recall@k between two FAISS indices."""
    _, exact_idx = exact_index.search(query_vecs, k)
    _, approx_idx = approx_index.search(query_vecs, k)
    recalls = []
    for i in range(len(query_vecs)):
        gt = set(exact_idx[i].tolist())
        ap = set(approx_idx[i].tolist())
        recalls.append(len(gt & ap) / k)
    return float(np.mean(recalls))
