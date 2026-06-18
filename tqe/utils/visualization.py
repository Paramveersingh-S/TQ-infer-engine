"""
Plotting utilities for TurboQuant benchmark results.

Generates all 5 required figures:
  Figure 1: MSE distortion vs bit-width for all methods + theoretical bound
  Figure 2: LLM perplexity vs compression ratio (scatter)
  Figure 3: ANN Recall@10 vs memory/vector (Pareto frontier)
  Figure 4: Attention logit computation speedup vs bit-width (bar chart)
  Figure 5: KV cache memory savings vs context length
"""

import os
from typing import List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as mstyle
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


_COLORS = {
    'TurboQuant': '#2E86AB',
    'PolarQuant': '#A23B72',
    'QJL': '#F18F01',
    'UniformScalar': '#C73E1D',
    'Theoretical': '#3B1F2B',
    'FAISS-PQ': '#44BBA4',
    'Baseline FP16': '#E94F37',
}


def _setup_style():
    if HAS_MPL:
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
        })


def plot_distortion_vs_bits(results, save_path: str = "results/fig1_distortion.png"):
    """
    Figure 1: MSE distortion vs bit-width for all methods + theoretical lower bound.
    """
    if not HAS_MPL:
        print("matplotlib not available, skipping plot.")
        return

    _setup_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        import pandas as pd
        df = pd.DataFrame(results) if not hasattr(results, 'groupby') else results
    except ImportError:
        print("pandas not available, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for method, grp in df.groupby('method'):
        grp_sorted = grp.sort_values('bits_per_dim')
        color = _COLORS.get(method, None)
        ls = '--' if method == 'Theoretical' else '-'
        ax.plot(grp_sorted['bits_per_dim'], grp_sorted['mse_distortion'],
                label=method, color=color, linestyle=ls, marker='o', linewidth=2)

    ax.set_xlabel("Bits per Dimension")
    ax.set_ylabel("MSE Distortion")
    ax.set_title("Figure 1 · MSE Distortion vs Bit-width")
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_perplexity_vs_compression(results, save_path: str = "results/fig2_kv_perplexity.png"):
    """
    Figure 2: LLM perplexity vs compression ratio scatter plot.
    """
    if not HAS_MPL:
        return
    _setup_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        color = _COLORS.get(r.get('method', ''), '#4C72B0')
        ax.scatter(r['compression_ratio'], r['perplexity'],
                   label=f"{r['method']} ({r['bits_per_dim']:.0f}bit)",
                   color=color, s=120, zorder=3)

    ax.set_xlabel("Compression Ratio (×FP16)")
    ax.set_ylabel("Perplexity (WikiText-2)")
    ax.set_title("Figure 2 · Perplexity vs Compression Ratio")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_ann_pareto(results, save_path: str = "results/fig3_ann_pareto.png"):
    """
    Figure 3: ANN Recall@10 vs memory per vector (Pareto frontier).
    """
    if not HAS_MPL:
        return
    _setup_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    k10 = [r for r in results if r.get('k', 10) == 10]
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in k10:
        color = _COLORS.get(r['method'], '#4C72B0')
        label = f"{r['method']} {r['bits_per_dim']:.0f}bit"
        ax.scatter(r['memory_bytes_per_vector'], r['recall'],
                   label=label, color=color, s=150, zorder=3)

    ax.set_xlabel("Memory per Vector (bytes)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Figure 3 · ANN Recall@10 vs Memory per Vector (Pareto)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_compression_ratio_table(quantizers_info: list,
                                  save_path: str = "results/fig4_compression.png"):
    """
    Figure 4: Compression ratio bar chart for multiple bit-widths.
    """
    if not HAS_MPL:
        return
    _setup_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    bits = [r['bits'] for r in quantizers_info]
    ratios = [r['ratio'] for r in quantizers_info]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f"{b}bit" for b in bits], ratios, color='#2E86AB', edgecolor='white', linewidth=1.2)
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{ratio:.1f}×", ha='center', fontsize=11, fontweight='bold')
    ax.set_xlabel("Bit-width")
    ax.set_ylabel("Compression Ratio (×FP16)")
    ax.set_title("Figure 4 · TurboQuant Compression Ratio vs Bit-width")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_kv_memory_vs_context(
    context_lengths: list,
    baseline_gb: list,
    compressed_gb: list,
    save_path: str = "results/fig5_kv_memory.png",
):
    """
    Figure 5: KV cache memory savings vs context length.
    """
    if not HAS_MPL:
        return
    _setup_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(context_lengths, baseline_gb, label='Baseline FP16', color='#E94F37',
            marker='o', linewidth=2)
    ax.plot(context_lengths, compressed_gb, label='TurboQuant 4-bit', color='#2E86AB',
            marker='s', linewidth=2)
    ax.fill_between(context_lengths, compressed_gb, baseline_gb,
                    alpha=0.15, color='#2E86AB', label='Memory saved')
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("KV Cache Memory (GB)")
    ax.set_title("Figure 5 · KV Cache Memory vs Context Length")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
