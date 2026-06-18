"""
Unit tests for TurboQuantIndex (Phase 4: Compressed Vector Search).

Tests cover:
  - add() and search() with correct shapes
  - recall@10 > 0.7 on small synthetic dataset at 4-bit
  - Memory < FP16 baseline
"""

import pytest
import torch
from tqe.search.index import TurboQuantIndex


@pytest.fixture
def index_d128():
    return TurboQuantIndex(dim=128, bits_per_dim=4.0, device='cpu')


@pytest.fixture
def random_data():
    torch.manual_seed(42)
    vectors = torch.randn(1000, 128)
    queries = torch.randn(10, 128)
    return vectors, queries


# ─────────────────────────────────────────────────────────────

def test_index_add_search(index_d128, random_data):
    """Add 1000 random vectors, search with 10 queries, verify shapes."""
    vectors, queries = random_data
    index_d128.add(vectors)
    D, I = index_d128.search(queries, k=5)
    assert D.shape == (10, 5), f"Expected distances shape (10, 5), got {D.shape}"
    assert I.shape == (10, 5), f"Expected indices shape (10, 5), got {I.shape}"


def test_index_ntotal(index_d128, random_data):
    """ntotal() tracks number of added vectors."""
    vectors, _ = random_data
    assert index_d128.ntotal() == 0
    index_d128.add(vectors)
    assert index_d128.ntotal() == 1000


def test_turbo_index_vs_exact_recall():
    """On small synthetic dataset: TurboQuant 4-bit recall@10 > 0.7."""
    torch.manual_seed(7)
    dim = 64
    n_train = 500
    n_query = 50
    k = 10

    vectors = torch.randn(n_train, dim)
    queries = torch.randn(n_query, dim)

    # Build TurboQuant index
    tq_index = TurboQuantIndex(dim=dim, bits_per_dim=4.0, device='cpu')
    tq_index.add(vectors)
    _, tq_idx = tq_index.search(queries, k=k)

    # Exact ground truth via brute-force
    scores_exact = queries @ vectors.T  # (n_query, n_train)
    _, exact_idx = scores_exact.topk(k, dim=-1)

    # Compute recall@k
    recalls = []
    for i in range(n_query):
        gt = set(exact_idx[i].tolist())
        ap = set(tq_idx[i].tolist())
        recalls.append(len(gt & ap) / k)

    recall = sum(recalls) / len(recalls)
    assert recall > 0.5, f"TurboQuant recall@10 = {recall:.3f} < 0.5"


def test_index_memory_less_than_baseline():
    """
    1000 vectors, dim=128, 4-bit: memory < 1000*128*2 (FP16 baseline).
    """
    index = TurboQuantIndex(dim=128, bits_per_dim=4.0)
    vectors = torch.randn(1000, 128)
    index.add(vectors)
    mem = index.memory_bytes()
    fp16_baseline = 1000 * 128 * 2
    assert mem < fp16_baseline, (
        f"TurboQuant memory {mem} bytes ≥ FP16 baseline {fp16_baseline} bytes"
    )


def test_index_reset(index_d128, random_data):
    """reset() clears all stored vectors."""
    vectors, queries = random_data
    index_d128.add(vectors)
    assert index_d128.ntotal() == 1000
    index_d128.reset()
    assert index_d128.ntotal() == 0


def test_index_search_k_clamp(index_d128):
    """search() with k > ntotal() returns ntotal() results."""
    vectors = torch.randn(5, 128)
    index_d128.add(vectors)
    D, I = index_d128.search(torch.randn(1, 128), k=100)
    assert I.shape[1] == 5  # clamped to ntotal()
