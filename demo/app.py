"""
Gradio Demo for TurboQuant Inference Engine.

Three tabs:
  Tab 1: KV Cache Compression Demo
  Tab 2: Semantic Search with Compressed Vectors
  Tab 3: Algorithm Visualizer (step-by-step, CPU-only)
"""

import argparse
import math
import sys
import numpy as np
import torch

import gradio as gr

from tqe.algorithms.turbo_quant import TurboQuantizer
from tqe.algorithms.polar_quant import PolarQuantizer
from tqe.algorithms.qjl import QJLQuantizer
from tqe.search.index import TurboQuantIndex

# ─────────────────────────────────────────────────────────────
# Pre-load instances (fast, CPU)
# ─────────────────────────────────────────────────────────────

DEMO_DIM = 200
_quantizers: dict[float, TurboQuantizer] = {}
_search_index: dict[float, TurboQuantIndex] = {}
_demo_vectors: torch.Tensor | None = None
_llm_model = None
_llm_tokenizer = None


def _ensure_quantizers():
    global _quantizers
    if not _quantizers:
        for bits in [2.0, 3.0, 4.0, 8.0]:
            _quantizers[bits] = TurboQuantizer(
                input_dim=DEMO_DIM, total_bits_per_dim=bits, device="cpu"
            )


def _ensure_search_index():
    global _search_index, _demo_vectors
    if _demo_vectors is None:
        torch.manual_seed(42)
        _demo_vectors = torch.randn(10_000, DEMO_DIM)
        _demo_vectors = _demo_vectors / _demo_vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    if not _search_index:
        _ensure_quantizers()
        for bits in [2.0, 3.0, 4.0, 8.0]:
            idx = TurboQuantIndex(dim=DEMO_DIM, bits_per_dim=bits, device="cpu")
            idx.add(_demo_vectors)
            _search_index[bits] = idx


# ─────────────────────────────────────────────────────────────
# Tab 1: KV Cache Compression Demo
# ─────────────────────────────────────────────────────────────

def run_kv_demo(prompt: str, bits: float, model_name: str):
    """
    Demo KV cache compression. Uses compression statistics without loading full LLM
    (loading full LLM requires GPU and HuggingFace token).
    """
    _ensure_quantizers()
    tq = _quantizers.get(bits, _quantizers[4.0])

    # Simulate KV cache stats for a realistic sequence
    sim_batch, sim_heads, sim_seq, sim_head_dim = 1, 32, 512, 128
    n_layers = 32

    # Baseline FP16 memory (per layer: 2 tensors K+V)
    baseline_bytes = 2 * n_layers * sim_batch * sim_heads * sim_seq * sim_head_dim * 2
    # Compressed memory
    compressed_bytes = tq.memory_bytes(sim_batch * sim_heads * sim_seq) * 2 * n_layers
    ratio = baseline_bytes / compressed_bytes
    saved_gb = (baseline_bytes - compressed_bytes) / 1e9

    output_text = (
        f"[Demo mode — full LLM inference requires GPU + model weights]\n\n"
        f"Prompt: {prompt[:100]}...\n\n"
        f"Simulating KV cache for {model_name} at {bits:.0f}-bit compression:\n"
        f"  • Layers: {n_layers}\n"
        f"  • KV heads: {sim_heads}\n"
        f"  • Sequence length: {sim_seq} tokens\n"
    )

    stats_text = (
        f"📊 KV Cache Statistics\n"
        f"{'─'*35}\n"
        f"Baseline FP16:   {baseline_bytes/1e9:.3f} GB\n"
        f"TurboQuant {bits:.0f}bit: {compressed_bytes/1e9:.3f} GB\n"
        f"Compression:     {ratio:.2f}×\n"
        f"Memory Saved:    {saved_gb:.3f} GB\n"
        f"Bits/dim:        {bits:.1f}\n"
    )

    return output_text, stats_text


# ─────────────────────────────────────────────────────────────
# Tab 2: Semantic Search
# ─────────────────────────────────────────────────────────────

def run_search_demo(query_str: str, bits: float, k: int = 5):
    """Run semantic search with compressed vectors."""
    _ensure_search_index()

    # Parse query vector from string or create random
    try:
        vals = [float(x.strip()) for x in query_str.strip().split(",") if x.strip()]
        if len(vals) == DEMO_DIM:
            query = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
        else:
            torch.manual_seed(hash(query_str) % 2**31)
            query = torch.randn(1, DEMO_DIM)
    except Exception:
        torch.manual_seed(hash(query_str) % 2**31)
        query = torch.randn(1, DEMO_DIM)

    query = query / query.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Search
    idx = _search_index.get(bits, _search_index[4.0])
    D, I = idx.search(query, k=k)

    # Exact brute-force for comparison
    exact_scores = query @ _demo_vectors.T
    _, exact_I = exact_scores.topk(k, dim=-1)

    # Recall
    approx_set = set(I[0].tolist())
    exact_set = set(exact_I[0].tolist())
    recall = len(approx_set & exact_set) / k

    result = f"🔍 Top-{k} Results (TurboQuant {bits:.0f}-bit)\n{'─'*40}\n"
    for rank, (idx_val, score) in enumerate(zip(I[0].tolist(), D[0].tolist()), 1):
        exact_mark = "✓" if idx_val in exact_set else " "
        result += f"  {rank}. Vector #{idx_val:5d}  score={score:+.4f}  {exact_mark}\n"

    result += f"\n📈 Recall@{k}: {recall:.3f}  ({bits:.0f}-bit vs exact)\n"
    result += f"🗜  Memory:   {idx.memory_bytes()/1e6:.2f} MB  "
    result += f"vs {_demo_vectors.nbytes/1e6:.2f} MB FP32 baseline\n"

    return result


# ─────────────────────────────────────────────────────────────
# Tab 3: Algorithm Visualizer
# ─────────────────────────────────────────────────────────────

def run_visualizer(vector_str: str, bits: float):
    """Step-by-step TurboQuant encoding visualization."""
    _ensure_quantizers()

    # Parse input vector
    try:
        vals = [float(x.strip()) for x in vector_str.strip().replace(";", ",").split(",")]
        if len(vals) < 4:
            vals = vals + [0.0] * (4 - len(vals))
        v_input = torch.tensor(vals[:64] if len(vals) >= 64 else vals, dtype=torch.float32)
    except Exception:
        v_input = torch.randn(8)

    d = v_input.shape[0]
    if d % 2 != 0:
        v_input = torch.cat([v_input, torch.zeros(1)])
        d = d + 1

    pq = PolarQuantizer(input_dim=d, bits_per_dim=max(2.0, bits - 1.0), rotation_seed=42)
    qjl = QJLQuantizer(input_dim=d, proj_dim=d, seed=137)

    v = v_input.unsqueeze(0)

    # Step 1: Random rotation
    rotated = v @ pq.R.T
    # Step 2: Polar pairs
    pairs = rotated.reshape(1, d // 2, 2)
    radii = torch.sqrt(pairs[..., 0]**2 + pairs[..., 1]**2)
    angles = torch.atan2(pairs[..., 1], pairs[..., 0])
    # Step 3: Quantize
    q_angles = pq._angle_quantize(angles, pq.B_angle)
    q_radii, scale = pq._radius_quantize(radii, pq.B_radius)
    # Step 4: Decode
    pq_codes = pq.encode(v)
    v_pq_hat = pq.decode(pq_codes)
    residual = v - v_pq_hat
    # Step 5: QJL on residual
    qjl_codes = qjl.encode(residual)
    v_qjl_hat = qjl.decode_approximate(qjl_codes)
    # Final
    v_final = v_pq_hat + v_qjl_hat

    mse_pq = ((v - v_pq_hat) ** 2).mean().item()
    mse_tq = ((v - v_final) ** 2).mean().item()

    lines = [
        f"🔬 TurboQuant Encoding Walkthrough ({bits:.0f}-bit, d={d})",
        "─" * 50,
        "",
        f"📥 Original vector (first 8 dims): {v[0,:8].tolist()}",
        "",
        f"🔄 After random rotation (first 8 dims): {rotated[0,:8].round(decimals=3).tolist()}",
        "",
        f"📐 Polar pairs (first 4 pairs):",
        f"   radii:  {radii[0,:4].round(decimals=3).tolist()}",
        f"   angles: {angles[0,:4].round(decimals=3).tolist()} (rad)",
        "",
        f"🗜  Stage 1 — PolarQuant ({bits-1:.0f}-bit):",
        f"   q_angles (first 4): {q_angles[0,:4].tolist()}",
        f"   q_radii  (first 4): {q_radii[0,:4].tolist()}",
        f"   scale:              {scale[0,0].item():.4f}",
        f"   Reconstruction MSE: {mse_pq:.6f}",
        "",
        f"🔑 Stage 2 — QJL residual correction (1-bit):",
        f"   residual norm:      {residual.norm().item():.4f}",
        f"   QJL codes (first 8): {qjl_codes[0,:8].tolist()}",
        "",
        f"✅ Final TurboQuant Reconstruction:",
        f"   MSE (PolarQuant only): {mse_pq:.6f}",
        f"   MSE (TurboQuant):      {mse_tq:.6f}",
        f"   Improvement:           {mse_pq/max(mse_tq,1e-12):.2f}× better",
        "",
        f"💾 Compression ratio vs FP32: {d*4 / (TurboQuantizer(d,bits).memory_bytes(1)):.2f}×",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Build Gradio Interface
# ─────────────────────────────────────────────────────────────

def build_demo():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(
        title="TurboQuant Inference Engine",
        theme=theme,
        css=".gradio-container { max-width: 1100px !important; }",
    ) as demo:

        gr.Markdown("""
# 🚀 TurboQuant Inference Engine
**Online, data-oblivious vector quantization achieving near-optimal distortion rates.**

*Based on: arXiv:2504.19874 (ICLR 2026) — Google Research*
        """)

        with gr.Tabs():

            # ── Tab 1: KV Cache ──────────────────────────────────────────
            with gr.Tab("🧠 KV Cache Compression"):
                gr.Markdown("Simulate TurboQuant KV cache compression for LLM inference.")
                with gr.Row():
                    with gr.Column(scale=2):
                        kv_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter any text prompt...",
                            value="Explain the mathematical foundation of transformer attention mechanisms.",
                            lines=4,
                        )
                        kv_bits = gr.Slider(
                            minimum=2, maximum=8, step=0.5, value=4.0,
                            label="Bits per Dimension",
                        )
                        kv_model = gr.Dropdown(
                            choices=[
                                "google/gemma-2-2b-it",
                                "meta-llama/Llama-3.1-8B-Instruct",
                                "mistralai/Mistral-7B-Instruct-v0.3",
                            ],
                            value="google/gemma-2-2b-it",
                            label="Model",
                        )
                        kv_btn = gr.Button("▶ Run Demo", variant="primary")
                    with gr.Column(scale=3):
                        kv_output = gr.Textbox(label="Output", lines=8, interactive=False)
                        kv_stats = gr.Textbox(label="📊 Memory Stats", lines=10, interactive=False)

                kv_btn.click(
                    fn=run_kv_demo,
                    inputs=[kv_prompt, kv_bits, kv_model],
                    outputs=[kv_output, kv_stats],
                )

            # ── Tab 2: Semantic Search ────────────────────────────────────
            with gr.Tab("🔍 Semantic Search"):
                gr.Markdown(
                    "Search 10K synthetic vectors using TurboQuant-compressed index. "
                    "Enter comma-separated floats or leave blank for random query."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        search_query = gr.Textbox(
                            label="Query Vector (comma-separated) or leave blank for random",
                            placeholder="0.1, -0.5, 0.3, ...",
                            lines=3,
                        )
                        search_bits = gr.Slider(
                            minimum=2, maximum=8, step=1.0, value=4.0,
                            label="Compression (bits/dim)",
                        )
                        search_k = gr.Slider(minimum=1, maximum=20, step=1, value=5,
                                             label="Top-k")
                        search_btn = gr.Button("🔎 Search", variant="primary")
                    with gr.Column(scale=3):
                        search_out = gr.Textbox(label="Results", lines=18, interactive=False)

                search_btn.click(
                    fn=run_search_demo,
                    inputs=[search_query, search_bits, search_k],
                    outputs=[search_out],
                )

            # ── Tab 3: Algorithm Visualizer ──────────────────────────────
            with gr.Tab("🔬 Algorithm Visualizer"):
                gr.Markdown(
                    "Step-by-step walkthrough of TurboQuant encoding. "
                    "Enter a comma-separated numeric vector."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        viz_vector = gr.Textbox(
                            label="Input Vector (comma-separated floats)",
                            placeholder="1.2, -0.5, 0.8, 2.1, -1.3, 0.4, -0.9, 0.7",
                            value="1.2, -0.5, 0.8, 2.1, -1.3, 0.4, -0.9, 0.7",
                            lines=3,
                        )
                        viz_bits = gr.Slider(
                            minimum=2, maximum=8, step=1.0, value=4.0,
                            label="Bits per Dimension",
                        )
                        viz_btn = gr.Button("🔍 Visualize", variant="primary")
                    with gr.Column(scale=3):
                        viz_out = gr.Textbox(label="Encoding Steps", lines=28, interactive=False)

                viz_btn.click(
                    fn=run_visualizer,
                    inputs=[viz_vector, viz_bits],
                    outputs=[viz_out],
                )

        gr.Markdown("""
---
**References**: [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) · 
[QJL (arXiv:2406.03482)](https://arxiv.org/abs/2406.03482) · 
[PolarQuant (arXiv:2502.02617)](https://arxiv.org/abs/2502.02617)
        """)

    return demo


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def _run_test_mode():
    """Self-test all tabs with synthetic data."""
    print("Running self-test...")
    _ensure_quantizers()
    _ensure_search_index()

    o, s = run_kv_demo("test prompt", 4.0, "google/gemma-2-2b-it")
    assert "Compression" in s, "KV demo stats missing"

    result = run_search_demo("", 4.0, 5)
    assert "Recall" in result, "Search result missing recall"

    viz = run_visualizer("1.0, -1.0, 0.5, 2.0, -0.5, 1.5, -2.0, 0.3", 4.0)
    assert "TurboQuant" in viz

    print("✅ All self-tests passed!")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true",
                        help="Self-test all tabs and exit")
    args = parser.parse_args()

    if args.test_mode:
        _run_test_mode()
    else:
        _ensure_quantizers()
        _ensure_search_index()
        demo = build_demo()
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
