"""
ANN Benchmark on GloVe-200 dataset.
Compares TurboQuant against FAISS-PQ baselines.

Metrics:
  - Recall@1, Recall@10, Recall@100
  - Index build time (seconds)
  - Query time (ms per query, batch of 1000)
  - Memory (bytes per vector)

Expected TurboQuant results (from paper):
  - TurboQuant 4-bit recall@1 > FAISS-PQ 4-bit recall@1
  - TurboQuant index build: virtually zero preprocessing (key advantage)
  - FAISS-PQ requires ~hours of k-means training for codebook construction
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqe.benchmarks.ann_benchmark import run_ann_benchmark


def main():
    parser = argparse.ArgumentParser(description="ANN recall benchmark")
    parser.add_argument("--config", default="configs/ann_4bit.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-query", type=int, default=1_000)
    parser.add_argument("--output", default="results/ann_benchmark.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ann_cfg = cfg.get('benchmark', {}).get('ann', {})
    k_values = ann_cfg.get('k_values', [1, 10, 100])
    bits_list = ann_cfg.get('bits_to_test', [2.0, 3.0, 4.0])

    print(f"Running ANN benchmark: n_train={args.n_train}, device={args.device}")
    results = run_ann_benchmark(
        n_train=args.n_train,
        n_query=args.n_query,
        dim=200,
        k_values=k_values,
        bits_list=bits_list,
        device=args.device,
    )

    # Print table
    print("\n" + "=" * 75)
    print(f"{'Method':<18} {'Bits':>6} {'k':>4} {'Recall':>8} {'Build(s)':>10} {'Mem/vec(B)':>12}")
    print("-" * 75)
    for r in results:
        print(f"{r['method']:<18} {r['bits_per_dim']:>6.1f} {r['k']:>4} "
              f"{r['recall']:>8.4f} {r['build_time_s']:>10.2f} "
              f"{r.get('memory_bytes_per_vector', 0):>12}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    try:
        import pandas as pd
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")
    except ImportError:
        print("(pandas not available — results not saved to CSV)")

    try:
        from tqe.utils.visualization import plot_ann_pareto
        plot_ann_pareto(results, save_path="results/fig3_ann_pareto.png")
    except Exception as e:
        print(f"Skipping visualization: {e}")


if __name__ == "__main__":
    main()
