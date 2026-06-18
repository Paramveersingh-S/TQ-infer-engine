"""
Dataset download utilities.
Downloads GloVe-200 vectors for ANN benchmarks.
"""

import os
import argparse


def download_glove(output_dir: str = "data/glove"):
    """Download GloVe-200 embeddings."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import numpy as np
        print("Generating synthetic GloVe-like data for local development...")
        np.random.seed(42)
        n = 1_100_000
        dim = 200
        vecs = np.random.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
        vecs /= norms
        out_path = os.path.join(output_dir, "glove_200_synthetic.npy")
        np.save(out_path, vecs)
        print(f"Saved {n} synthetic vectors → {out_path}")
    except ImportError:
        print("numpy required. Install with: pip install numpy")


def download_wikitext(output_dir: str = "data/wikitext"):
    """Download WikiText-2 for perplexity benchmarks."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        ds.save_to_disk(output_dir)
        print(f"Saved WikiText-2 → {output_dir}")
    except ImportError:
        print("datasets required. Install with: pip install datasets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", action="store_true")
    parser.add_argument("--wikitext", action="store_true")
    args = parser.parse_args()

    if args.glove or not any([args.wikitext]):
        download_glove()
    if args.wikitext:
        download_wikitext()
