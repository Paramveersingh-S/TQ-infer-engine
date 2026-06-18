"""
Evaluate LLM perplexity with and without TurboQuant KV cache compression.

Models tested:
  - google/gemma-2-2b-it               (default, smaller for fast testing on T4)
  - meta-llama/Llama-3.1-8B-Instruct  (primary, needs A100)
  - mistralai/Mistral-7B-Instruct-v0.3 (secondary)

Dataset: wikitext-2-raw-v1 (test split)

Results table format:
  | Method       | Bits/dim | Perplexity | KV Memory (GB) | Compression Ratio |
  |--------------|----------|------------|----------------|-------------------|
  | Baseline FP16| 16       | X.XX       | X.XX           | 1.0×              |
  | TurboQuant   | 4        | X.XX       | X.XX           | ~4.0×             |
  | TurboQuant   | 3        | X.XX       | X.XX           | ~5.3×             |
  | TurboQuant   | 2        | X.XX       | X.XX           | ~8.0×             |
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqe.benchmarks.kv_benchmark import run_kv_benchmark


def main():
    parser = argparse.ArgumentParser(description="KV cache perplexity benchmark")
    parser.add_argument("--config", default="configs/kv_cache_4bit.yaml")
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/kv_cache_benchmark.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bits_list = cfg.get('benchmark', {}).get('kv', {}).get('bits_to_test', [2.0, 3.0, 4.0, 16.0])
    max_tokens = cfg.get('benchmark', {}).get('kv', {}).get('max_tokens', 2048)

    print(f"Running KV benchmark: model={args.model}, device={args.device}")
    results = run_kv_benchmark(
        model_name=args.model,
        bits_list=bits_list,
        max_tokens=max_tokens,
        device=args.device,
    )

    # Print results table
    print("\n" + "=" * 65)
    print(f"{'Method':<20} {'Bits':>6} {'Perplexity':>12} {'KV Memory (GB)':>16} {'Ratio':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r['method']:<20} {r['bits_per_dim']:>6.1f} {r['perplexity']:>12.2f} "
              f"{r['kv_memory_gb']:>16.3f} {r['compression_ratio']:>7.1f}×")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    try:
        import pandas as pd
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")
    except ImportError:
        print(f"\n(pandas not available — results not saved to CSV)")

    # Visualization
    try:
        from tqe.utils.visualization import plot_perplexity_vs_compression
        plot_perplexity_vs_compression(results, save_path="results/fig2_kv_perplexity.png")
    except Exception as e:
        print(f"Skipping visualization: {e}")


if __name__ == "__main__":
    main()
