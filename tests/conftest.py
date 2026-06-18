"""
tests/conftest.py — Shared pytest fixtures for TurboQuant test suite.

Provides:
  - device: "cuda" if available, else "cpu"
  - random_vectors: 1000 × 256 float tensor (seed=42)
  - small_model: tiny 2-layer GPT-2 for KV cache tests (no GPU needed)
  - set_seed: autouse fixture that seeds torch and numpy before each test
"""

import pytest
import torch
import numpy as np


@pytest.fixture(scope="session")
def device():
    """Return "cuda" if GPU is available, else "cpu"."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def random_vectors():
    """1000 random vectors in ℝ^256 (fixed seed=42)."""
    torch.manual_seed(42)
    return torch.randn(1000, 256)


@pytest.fixture
def small_model():
    """
    Tiny 2-layer GPT-2 model for KV cache tests.
    No real weights needed — randomly initialized.
    """
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=1000)
        return GPT2LMHeadModel(config).eval()
    except ImportError:
        pytest.skip("transformers not installed")


@pytest.fixture(autouse=True)
def set_seed():
    """Seed torch and numpy before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
