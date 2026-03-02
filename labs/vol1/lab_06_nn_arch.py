import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 06: THE QUADRATIC WALL
#
# Chapter: nn_architectures.qmd — "Network Architectures"
# Core Invariant: Transformer self-attention is O(N²) in sequence length.
#   Doubling the context quadruples attention memory. This is the architectural
#   bottleneck of modern LLMs — not FLOPS, but attention memory.
#
# 2-Act Structure (35–40 min total):
#   Act I  — The Attention Explosion (12–15 min)
#             Prediction → quadratic chart → reveal → FlashAttention reflection
#   Act II — Depth vs Width Tradeoff (20–25 min)
#             Prediction → parameter/latency instruments → OOM failure state
#             → parallelism reflection
#
# Deployment Contexts: Cloud (H100, 80 GB HBM) vs Edge (Jetson Orin NX, 16 GB)
#
# Sources:
#   - Attention memory formula: sec-network-architectures-transformers-attentiononly-architecture-1b56
#     "for a 4,096-token sequence with 16-bit scores, the attention matrix alone
#      consumes 4096² × 2 ≈ 32 MB per layer per head"
#   - Quadratic scaling: TransformerScaling class in chapter — scaling_ratio = 4.0
#   - FlashAttention: "tiling to avoid materializing the full matrix"
#   - Depth sequential bottleneck: "O(T) sequential depth that prevents GPU
#     parallelization" (from RNN section; same physics applies to layer depth)
#   - H100 specs: H100 SXM5 HBM3e, NVIDIA spec (80 GB HBM, 3350 GB/s BW)
#   - Jetson Orin NX: 16 GB unified memory, 102 GB/s
#
# Design Ledger saves: chapter=6, context, seq_len_chosen,
#   quadratic_oom_triggered, depth_width_decision
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible) ─────────────────────────
@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    import numpy as np
    import plotly.graph_objects as go

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    ledger = DesignLedger()

    # ── Hardware constants (all plain floats, no pint) ──
    # Source: NVIDIA H100 SXM5 HBM3e spec
    H100_RAM_GB      = 80     # GB HBM3e
    H100_BW_GBS      = 3350   # GB/s
    H100_TFLOPS_FP16 = 1979   # TFLOPS dense FP16

    # Source: NVIDIA Jetson Orin NX 16GB spec
    ORIN_RAM_GB  = 16     # GB unified memory
    ORIN_BW_GBS  = 102    # GB/s
    ORIN_TOPS    = 100    # TOPS (INT8 equivalent)

    # ── Attention constants ──
    # Source: chapter LEGO cell — AttentionMemory class
    # "for a 4,096-token sequence with 16-bit scores, the attention matrix alone
    #  consumes 4096² × 2 ≈ 32 MB per layer per head"
    BYTES_FP16   = 2      # bytes per FP16 element
    BYTES_FP32   = 4      # bytes per FP32 element

    # GPT-2 XL default config (lighthouse from chapter)
    GPT2_NUM_HEADS   = 16    # multi-head attention heads
    GPT2_HEAD_DIM    = 64    # head dimension (d_model / num_heads = 1024 / 16)
    GPT2_NUM_LAYERS  = 48    # transformer blocks
    GPT2_D_MODEL     = 1024  # hidden dimension

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme, np, go,
        H100_RAM_GB, H100_BW_GBS, H100_TFLOPS_FP16,
        ORIN_RAM_GB, ORIN_BW_GBS, ORIN_TOPS,
        BYTES_FP16, BYTES_FP32,
        GPT2_NUM_HEADS, GPT2_HEAD_DIM, GPT2_NUM_LAYERS, GPT2_D_MODEL,
    )


# ─── CELL 1: HEADER ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _cloud_color = COLORS["Cloud"]
    _edge_color  = COLORS["Edge"]

    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 06
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Quadratic Wall
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 640px; line-height: 1.65;">
                Transformer self-attention is O(N&sup2;) in sequence length.
                Marketing wants 100K-token context. Engineering says memory will
                explode. This lab shows you the numbers — and why depth vs width
                determines whether your model runs at all on edge hardware.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: Attention Explosion &middot; 12&ndash;15 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Act II: Depth vs Width &middot; 20&ndash;25 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35&ndash;40 min total
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">N&sup2; Attention Memory</span>
                <span class="badge badge-info">FlashAttention</span>
                <span class="badge badge-info">Depth vs Width</span>
                <span class="badge badge-ok">&#9729; Cloud: H100 (80 GB HBM)</span>
                <span class="badge badge-fail">&#9881; Edge: Jetson Orin NX (16 GB)</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: RECOMMENDED READING ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete these sections before this lab:

    - **@sec-network-architectures-transformers-attentiononly-architecture-1b56** —
      Transformer architecture, self-attention, and the O(N²) memory scaling law
    - **The Quadratic Bottleneck** callout — 100K-token memory calculation
    - **@sec-network-architectures-multilayer-perceptrons-dense-pattern-processing-bc11** —
      MLP depth vs width parameter scaling
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={"☁️ Cloud (H100, 80 GB HBM)": "cloud", "⚙️ Edge (Jetson Orin NX, 16 GB)": "edge"},
        value="☁️ Cloud (H100, 80 GB HBM)",
        label="**Deployment context** (used in Act II):",
        inline=True,
    )
    context_toggle
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ACT I — THE ATTENTION EXPLOSION
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["Cloud"]
    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="margin: 8px 0 4px 0;">
            <div class="concept-header">
                <span style="background:{_color}; color:white; border-radius:50%;
                             width:22px; height:22px; display:inline-flex; align-items:center;
                             justify-content:center; font-size:0.78rem; font-weight:800;
                             flex-shrink:0;">I</span>
                <span>Act I of II</span>
                <span class="concept-header-line"></span>
            </div>
            <div style="font-size:1.5rem; font-weight:800; color:#0f172a; line-height:1.2;
                        margin-top: 4px;">The Attention Explosion</div>
            <div style="color:#94a3b8; font-size:0.88rem; margin-top:3px;">
                The O(N²) memory wall that limits every large language model
            </div>
        </div>
        """),
    ])
    return


# ─── ACT I STAKEHOLDER MESSAGE ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["Cloud"]
    mo.Html(f"""
    <div style="border-left:4px solid {_color}; background:{COLORS['BlueL']};
                border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
        <div style="font-size:0.72rem; font-weight:700; color:{_color};
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
            Incoming Message &middot; VP of Product
        </div>
        <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
            "Marketing needs us to support 100K token context for enterprise document analysis.
            Our competitor just announced it. Engineering is pushing back and says memory
            will 'explode' — but I don't understand why doubling the context window is
            such a big deal. Can you show me the actual numbers?"
        </div>
    </div>
    """)
    return


# ─── ACT I PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Your Prediction

    *Before touching the simulator, commit to your hypothesis.*

    The attention matrix for a single layer and a single head holds N × N score values,
    where N is the sequence length. The chapter states that attention memory scales O(N²).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A) 2× larger  — linear scaling, like doubling a list": "linear",
            "B) 4× larger  — quadratic scaling, one order of magnitude jump": "quadratic",
            "C) 8× larger  — cubic scaling, each token sees twice as many tokens": "cubic",
            "D) Same size  — it just depends on positional encoding, not raw length": "same",
        },
        label="**Prediction Lock** — Doubling context length from 4K to 8K tokens increases "
              "the attention matrix memory by:",
    )
    mo.vstack([
        act1_prediction,
        mo.callout(mo.md("Select your prediction to unlock the simulator."), kind="warn")
        if act1_prediction.value is None
        else mo.md(""),
    ])
    return (act1_prediction,)


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(
        act1_prediction.value is None,
        mo.callout(mo.md("**Select a prediction above to unlock Act I.**"), kind="warn"),
    )
    return


# ─── ACT I SIMULATOR ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The Simulator

    Adjust the sequence length and observe how attention memory scales relative to
    FFN (Feed-Forward Network) memory. The red line marks your device's HBM capacity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Sequence length slider — log scale via discrete steps
    # Source: chapter uses 512 base, 4096 as example, 100K as wall case
    act1_seq_slider = mo.ui.slider(
        start=512,
        stop=131072,
        value=4096,
        step=512,
        label="Sequence length (tokens)",
    )
    act1_num_heads_slider = mo.ui.slider(
        start=1,
        stop=32,
        value=12,
        step=1,
        label="Number of attention heads",
    )
    mo.hstack([act1_seq_slider, act1_num_heads_slider], justify="start", gap="2rem")
    return (act1_seq_slider, act1_num_heads_slider)


@app.cell(hide_code=True)
def _(
    mo, go, np, apply_plotly_theme,
    act1_seq_slider, act1_num_heads_slider,
    COLORS, BYTES_FP16,
    GPT2_D_MODEL, GPT2_NUM_LAYERS,
    H100_RAM_GB, ORIN_RAM_GB,
):
    # Use named (non-underscore) variables for those that cross cell boundaries
    act1_seq = act1_seq_slider.value
    act1_h   = act1_num_heads_slider.value

    # ── Physics: Attention matrix memory ──────────────────────────────────────
    # Source: chapter callout "The Quadratic Bottleneck"
    # Formula: attention_bytes = seq_len^2 * num_heads * BYTES_FP16
    # Per-layer, per-head: seq_len^2 * 2 bytes = N^2 * 2
    # Full model: * num_heads * num_layers

    # Single-layer, all-heads attention matrix
    act1_attn_single_gb = (act1_seq ** 2) * act1_h * BYTES_FP16 / 1e9

    # Full model attention (all layers — for training, all must be resident)
    act1_attn_full_gb = act1_attn_single_gb * GPT2_NUM_LAYERS

    # ── Physics: FFN memory (for comparison — linear in seq_len) ──────────────
    _ffn_activations_gb = (act1_seq * 4 * GPT2_D_MODEL * BYTES_FP16 * GPT2_NUM_LAYERS) / 1e9

    # ── Build scaling curve ────────────────────────────────────────────────────
    _seq_range = np.arange(512, 131073, 512)
    _attn_curve = (_seq_range ** 2) * act1_h * BYTES_FP16 * GPT2_NUM_LAYERS / 1e9
    _ffn_curve  = (_seq_range * 4 * GPT2_D_MODEL * BYTES_FP16 * GPT2_NUM_LAYERS) / 1e9

    # ── Quadratic ratio (the key insight) ─────────────────────────────────────
    _base_seq = 4096
    act1_ratio = (act1_seq / _base_seq) ** 2

    # ── Color state ───────────────────────────────────────────────────────────
    _attn_color = (
        COLORS["RedLine"] if act1_attn_single_gb > H100_RAM_GB
        else COLORS["OrangeLine"] if act1_attn_single_gb > H100_RAM_GB * 0.5
        else COLORS["BlueLine"]
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    _fig = go.Figure()

    _fig.add_trace(go.Scatter(
        x=_seq_range / 1000,
        y=_attn_curve,
        name="Attention matrix (all layers, full model)",
        line=dict(color=COLORS["BlueLine"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,99,149,0.08)",
    ))

    _fig.add_trace(go.Scatter(
        x=_seq_range / 1000,
        y=_ffn_curve,
        name="FFN activations (all layers — linear in N)",
        line=dict(color=COLORS["GreenLine"], width=2, dash="dot"),
    ))

    _fig.add_hline(
        y=H100_RAM_GB,
        line_dash="dash",
        line_color=COLORS["OrangeLine"],
        annotation_text=f"H100 HBM: {H100_RAM_GB} GB",
        annotation_position="top right",
    )

    _fig.add_hline(
        y=ORIN_RAM_GB,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        annotation_text=f"Orin NX RAM: {ORIN_RAM_GB} GB",
        annotation_position="top right",
    )

    _fig.add_trace(go.Scatter(
        x=[act1_seq / 1000],
        y=[act1_attn_full_gb],
        mode="markers",
        marker=dict(size=14, color=_attn_color, symbol="circle", line=dict(color="white", width=2)),
        name=f"Current: {act1_seq:,} tokens",
        showlegend=True,
    ))

    _fig.update_layout(
        xaxis_title="Sequence length (thousands of tokens)",
        yaxis_title="Memory (GB)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=400,
    )
    apply_plotly_theme(_fig)

    # ── Metric cards ──────────────────────────────────────────────────────────
    _card_attn_color = (
        COLORS["RedLine"] if act1_attn_single_gb > H100_RAM_GB
        else COLORS["OrangeLine"] if act1_attn_single_gb > 10
        else COLORS["BlueLine"]
    )
    _card_ratio_color = (
        COLORS["RedLine"] if act1_ratio > 100
        else COLORS["OrangeLine"] if act1_ratio > 10
        else COLORS["GreenLine"]
    )

    _ffn_idx = min((act1_seq // 512) - 1, len(_ffn_curve) - 1)

    mo.vstack([
        mo.md(f"""
        **Physics**

        ```
        Attention memory  = N² × H × 2 bytes × L  (full model, all layers)
                          = {act1_seq:,}² × {act1_h} heads × 2 B × {GPT2_NUM_LAYERS} layers
                          = {act1_attn_full_gb:.2f} GB

        FFN activations   = N × 4 × d_model × 2 bytes × L  (linear in N)
                          = {act1_seq:,} × 4 × {GPT2_D_MODEL} × 2 B × {GPT2_NUM_LAYERS} layers
                          = {_ffn_curve[_ffn_idx]:.2f} GB
        ```
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0;">
            <div style="padding: 18px 24px; border: 1.5px solid {COLORS['Border']};
                        border-radius: 12px; min-width: 200px; text-align: center;
                        background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <div style="color: #475569; font-size: 0.85rem; font-weight: 500;
                            margin-bottom: 4px;">Attention (full model)</div>
                <div style="font-size: 2rem; font-weight: 800;
                            color: {_card_attn_color};">
                    {act1_attn_full_gb:.1f} GB
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                    {GPT2_NUM_LAYERS} layers &times; {act1_h} heads
                </div>
            </div>
            <div style="padding: 18px 24px; border: 1.5px solid {COLORS['Border']};
                        border-radius: 12px; min-width: 200px; text-align: center;
                        background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <div style="color: #475569; font-size: 0.85rem; font-weight: 500;
                            margin-bottom: 4px;">Single-layer attention</div>
                <div style="font-size: 2rem; font-weight: 800;
                            color: {COLORS['BlueLine']};">
                    {act1_attn_single_gb:.2f} GB
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                    per transformer block
                </div>
            </div>
            <div style="padding: 18px 24px; border: 1.5px solid {COLORS['Border']};
                        border-radius: 12px; min-width: 200px; text-align: center;
                        background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <div style="color: #475569; font-size: 0.85rem; font-weight: 500;
                            margin-bottom: 4px;">Scale vs 4K baseline</div>
                <div style="font-size: 2rem; font-weight: 800;
                            color: {_card_ratio_color};">
                    {act1_ratio:.1f}&times;
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                    vs {_base_seq:,}-token baseline
                </div>
            </div>
        </div>
        """),
        mo.as_html(_fig),
    ])
    return (act1_seq, act1_h, act1_attn_single_gb, act1_attn_full_gb, act1_ratio)


# ─── ACT I OOM BANNER ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_attn_full_gb, H100_RAM_GB, ORIN_RAM_GB, act1_seq):
    if act1_attn_full_gb > H100_RAM_GB:
        mo.callout(mo.md(
            f"**OOM — Cloud infeasible without tiling.** "
            f"Attention alone requires **{act1_attn_full_gb:.0f} GB**. "
            f"H100 HBM capacity: **{H100_RAM_GB} GB**. "
            f"This is exactly why FlashAttention exists: it tiles the computation "
            f"to avoid materializing the full N&times;N matrix in HBM. "
            f"Pull the sequence length back below ~20K tokens to fit on a single H100."
        ), kind="danger")
    elif act1_attn_full_gb > ORIN_RAM_GB:
        mo.callout(mo.md(
            f"**OOM — Edge infeasible.** "
            f"Attention requires **{act1_attn_full_gb:.0f} GB**. "
            f"Jetson Orin NX has only **{ORIN_RAM_GB} GB**. "
            f"Cloud (H100) can still handle this sequence length, but the edge context "
            f"window is severely limited by the quadratic wall."
        ), kind="warn")
    elif act1_seq >= 32000:
        mo.callout(mo.md(
            f"**Approaching the wall.** At {act1_seq:,} tokens, attention alone consumes "
            f"**{act1_attn_full_gb:.1f} GB**. This is getting close to practical limits. "
            f"Notice how fast memory grows — try 64K and 128K to see why 100K context "
            f"is an engineering challenge, not a model quality decision."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**Feasible at {act1_seq:,} tokens.** "
            f"Attention uses {act1_attn_full_gb:.1f} GB. "
            f"Try increasing to 32K, 64K, and 100K to see the quadratic explosion."
        ), kind="success")
    return


# ─── ACT I REVEAL — PREDICTION vs REALITY ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_ratio):
    _pred_map = {
        "linear":    2.0,
        "quadratic": 4.0,
        "cubic":     8.0,
        "same":      1.0,
    }
    _predicted_ratio = _pred_map[act1_prediction.value]
    _actual_ratio    = 4.0  # quadratic: (8K/4K)^2 = 4
    _gap = abs(_actual_ratio - _predicted_ratio)

    if act1_prediction.value == "quadratic":
        mo.callout(mo.md(
            f"**Correct.** You predicted {_predicted_ratio:.0f}×. "
            f"The actual ratio when doubling from 4K to 8K tokens is exactly **{_actual_ratio:.0f}×**. "
            f"The N² scaling law is direct: (8000/4000)² = 4. "
            f"At 100K tokens, the memory is (100000/4000)² = **625× larger** than at 4K. "
            f"That is why marketing's request is not a feature — it is a physics problem."
        ), kind="success")
    elif act1_prediction.value == "linear":
        mo.callout(mo.md(
            f"**Off by {_actual_ratio / _predicted_ratio:.0f}×.** "
            f"You predicted {_predicted_ratio:.0f}× (linear). "
            f"The actual ratio is **{_actual_ratio:.0f}×** (quadratic). "
            f"Linear scaling would mean attention memory grows proportionally with N. "
            f"But the attention matrix stores N × N scores — every token attends to every "
            f"other token — so doubling N doubles *both* dimensions, quadrupling the matrix."
        ), kind="warn")
    elif act1_prediction.value == "cubic":
        mo.callout(mo.md(
            f"**Close, but too pessimistic by {_predicted_ratio / _actual_ratio:.0f}×.** "
            f"You predicted {_predicted_ratio:.0f}× (cubic). "
            f"The actual ratio is **{_actual_ratio:.0f}×** (quadratic). "
            f"The N × N attention matrix grows as N², not N³. "
            f"The head dimension is fixed — only sequence length scales. "
            f"Quadratic is already severe: 625× at 100K tokens."
        ), kind="warn")
    else:  # same
        mo.callout(mo.md(
            f"**Off by {_actual_ratio / _predicted_ratio:.0f}×.** "
            f"You predicted {_predicted_ratio:.0f}× (no change). "
            f"The actual ratio is **{_actual_ratio:.0f}×** when doubling context. "
            f"Positional encoding adds negligible memory. The bottleneck is the "
            f"N × N attention score matrix — quadratic in sequence length. "
            f"At 100K tokens, this is 625× larger than at 4K tokens."
        ), kind="warn")
    return


# ─── ACT I REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reflection

    *The chapter notes: "FlashAttention and sparse attention variants exist — they recompute
    rather than store the attention matrix to break this memory wall."*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A) Compressing the attention matrix using a learned low-rank factorization": "compress",
            "B) Computing attention in tiles without materializing the full N×N matrix in HBM": "tiling",
            "C) Approximating attention by using fewer heads in deeper layers": "fewer_heads",
            "D) Converting attention scores to INT8 precision before the softmax": "int8",
        },
        label="**FlashAttention reduces memory from O(N²) to O(N) by:**",
    )
    mo.vstack([
        act1_reflection,
        mo.callout(mo.md("Select your answer."), kind="warn")
        if act1_reflection.value is None
        else mo.md(""),
    ])
    return (act1_reflection,)


@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(
        act1_reflection.value is None,
        mo.callout(mo.md("Select your answer above."), kind="warn"),
    )

    if act1_reflection.value == "tiling":
        mo.callout(mo.md(
            "**Correct.** FlashAttention tiles the Q, K, V matrices to fit in SRAM (fast on-chip "
            "memory), computes attention incrementally using the online softmax algorithm, and "
            "accumulates the output without ever writing the full N×N matrix to HBM. "
            "Memory usage drops from O(N²) to O(N) — the N×N matrix simply never exists "
            "as a whole in device memory. Arithmetic (FLOPs) is identical; only data movement "
            "to HBM changes. The chapter states: *'FlashAttention — tiling to avoid "
            "materializing the full matrix.'*"
        ), kind="success")
    elif act1_reflection.value == "compress":
        mo.callout(mo.md(
            "**Not quite.** Low-rank factorization (as in Linformer) is a separate approach "
            "that *approximates* attention. FlashAttention achieves *exact* attention — the "
            "same mathematical result — by restructuring how the computation accesses memory. "
            "The key insight is tiling: splitting the sequence into blocks that fit in fast "
            "SRAM, so the full N×N matrix never needs to be written to slow HBM."
        ), kind="warn")
    elif act1_reflection.value == "fewer_heads":
        mo.callout(mo.md(
            "**Not quite.** FlashAttention does not reduce the number of heads or change the "
            "model architecture in any way. It is a pure IO-optimization: the same computation "
            "is performed, but memory access to HBM is restructured via tiling so the full "
            "N×N score matrix never needs to reside in HBM simultaneously."
        ), kind="warn")
    else:  # int8
        mo.callout(mo.md(
            "**Not quite.** INT8 quantization of attention scores is a separate technique that "
            "reduces precision but still stores the full N×N matrix (in INT8). The memory "
            "savings from INT8 vs FP16 is 2×, not the O(N) vs O(N²) reduction that FlashAttention "
            "achieves. FlashAttention tiles the computation so the full matrix is never "
            "materialized — the result is exact, not approximate."
        ), kind="warn")
    return


# ─── ACT I MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation: Attention memory": mo.md("""
        **Formula:**

        `Attention_memory = N² × H × d_head × dtype_bytes`

        Where:
        - **N** — sequence length (tokens)
        - **H** — number of attention heads
        - **d_head** — head dimension (= d_model / H)
        - **dtype_bytes** — 2 for FP16, 4 for FP32

        **Full model (all layers):**

        `Total_attention_GB = (N² × H × dtype_bytes × L) / 1e9`

        - **L** — number of transformer layers

        **Quadratic scaling proof:**

        If N₂ = k × N₁, then:

        `Attention(N₂) / Attention(N₁) = (k × N₁)² / N₁² = k²`

        Doubling N (k = 2) → 4× memory. Tripling N (k = 3) → 9× memory.

        At 100K tokens vs 4K baseline: k = 25, so **625× more attention memory**.

        **Source:** @sec-network-architectures-transformers-attentiononly-architecture-1b56 —
        "for a 4,096-token sequence with 16-bit scores, the attention matrix alone
        consumes 4096² × 2 ≈ 32 MB per layer per head"
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ACT II — DEPTH VS WIDTH TRADEOFF
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["Edge"]
    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="margin: 8px 0 4px 0;">
            <div class="concept-header">
                <span style="background:{_color}; color:white; border-radius:50%;
                             width:22px; height:22px; display:inline-flex; align-items:center;
                             justify-content:center; font-size:0.78rem; font-weight:800;
                             flex-shrink:0;">II</span>
                <span>Act II of II</span>
                <span class="concept-header-line"></span>
            </div>
            <div style="font-size:1.5rem; font-weight:800; color:#0f172a; line-height:1.2;
                        margin-top: 4px;">Depth vs Width Tradeoff</div>
            <div style="color:#94a3b8; font-size:0.88rem; margin-top:3px;">
                For a fixed parameter budget, does a deeper or wider network run faster on edge hardware?
            </div>
        </div>
        """),
    ])
    return


# ─── ACT II STAKEHOLDER MESSAGE ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, context_toggle):
    _ctx = context_toggle.value
    _color = COLORS["Cloud"] if _ctx == "cloud" else COLORS["Edge"]
    _device = "H100 (Cloud)" if _ctx == "cloud" else "Jetson Orin NX (Edge)"

    mo.Html(f"""
    <div style="border-left:4px solid {_color}; background:{COLORS['BlueL']};
                border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
        <div style="font-size:0.72rem; font-weight:700; color:{_color};
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
            Incoming Message &middot; Edge ML Lead ({_device})
        </div>
        <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
            "We have a 100M parameter budget for our on-device text classifier. I want to use
            a 64-layer deep network to get better representations. A colleague says I should
            use 4 layers with a wider hidden dimension instead — that it will run faster on
            {_device} even with the same parameter count. I don't understand why depth would
            matter if FLOPS are the same."
        </div>
    </div>
    """)
    return


# ─── ACT II PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Your Prediction

    *Consider the architecture: a deeper network has more sequential layer dependencies.
    Each layer must wait for the output of the layer before it.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_prediction = mo.ui.radio(
        options={
            "A) Deeper (better parallelism — more layers = more GPU threads)": "deeper",
            "B) Wider (fewer sequential dependencies — more parallel work per layer)": "wider",
            "C) Same — parameters determine FLOPS, FLOPS determine latency": "same_flops",
            "D) Depends only on batch size, not architecture shape": "batch_size",
        },
        label="**Prediction Lock** — For a fixed 100M parameter budget, a deeper network "
              "(64 layers, small width) vs a wider network (4 layers, large width): "
              "on edge hardware (Jetson Orin NX), which runs faster for inference?",
    )
    mo.vstack([
        act2_prediction,
        mo.callout(mo.md("Select your prediction to unlock the Design Challenge."), kind="warn")
        if act2_prediction.value is None
        else mo.md(""),
    ])
    return (act2_prediction,)


@app.cell(hide_code=True)
def _(mo, act2_prediction):
    mo.stop(
        act2_prediction.value is None,
        mo.callout(mo.md("**Select a prediction above to unlock Act II.**"), kind="warn"),
    )
    return


# ─── ACT II INSTRUMENTS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The Design Challenge

    Explore the depth/width tradeoff. Stay within the device's memory budget.
    The parameter target is shown — try to stay near 100M parameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Source: MLP parameter formula from chapter — parameters = layers × (width² + width × 4×width)
    # Transformer block: attention (4×d²) + FFN (8×d²) ≈ 12×d² parameters per layer
    act2_depth_slider = mo.ui.slider(
        start=2,
        stop=100,
        value=12,
        step=2,
        label="Number of layers (depth)",
    )
    act2_width_slider = mo.ui.slider(
        start=64,
        stop=4096,
        value=512,
        step=64,
        label="Hidden dimension (width)",
    )
    mo.hstack([act2_depth_slider, act2_width_slider], justify="start", gap="2rem")
    return (act2_depth_slider, act2_width_slider)


@app.cell(hide_code=True)
def _(
    mo, go, np, apply_plotly_theme,
    act2_depth_slider, act2_width_slider, context_toggle,
    COLORS, BYTES_FP16,
    H100_RAM_GB, H100_BW_GBS, H100_TFLOPS_FP16,
    ORIN_RAM_GB, ORIN_BW_GBS, ORIN_TOPS,
):
    # Named (non-underscore) variables used across cells
    act2_L = act2_depth_slider.value
    act2_W = act2_width_slider.value
    act2_ctx = context_toggle.value

    # ── Device selection ──────────────────────────────────────────────────────
    if act2_ctx == "cloud":
        act2_device_name = "H100 (Cloud)"
        act2_device_ram  = H100_RAM_GB
        _device_bw       = H100_BW_GBS
        _device_tops     = H100_TFLOPS_FP16 * 1000  # GOPS
        _par_factor      = 1.0
        _dev_color       = COLORS["Cloud"]
    else:
        act2_device_name = "Jetson Orin NX"
        act2_device_ram  = ORIN_RAM_GB
        _device_bw       = ORIN_BW_GBS
        _device_tops     = ORIN_TOPS * 1000
        _par_factor      = 0.15
        _dev_color       = COLORS["Edge"]

    # ── Parameter count ───────────────────────────────────────────────────────
    # Source: chapter MLP section — "parameters = layers × (width² + ...)"
    # For a transformer block: ~12 × d_model² parameters (4d² attention + 8d² FFN)
    _params_per_layer = 12 * act2_W ** 2
    _total_params     = act2_L * _params_per_layer
    act2_params_m     = _total_params / 1e6   # in millions; shared with other cells
    _target_params_m  = 100.0

    # ── Model memory ─────────────────────────────────────────────────────────
    act2_model_mem_gb = (_total_params * BYTES_FP16) / 1e9

    # ── Sequential depth penalty ─────────────────────────────────────────────
    # Source: chapter RNN analysis — "O(T) sequential depth prevents GPU
    #   parallelization"; same physics for transformer layer depth at inference.
    _flops_per_token     = 2 * 12 * act2_W ** 2 * act2_L
    _compute_ms_raw      = (_flops_per_token / (_device_tops * 1e9)) * 1000
    _depth_overhead_ms   = act2_L * 0.01 / (1 + _par_factor * 10)
    _latency_ms          = _compute_ms_raw + _depth_overhead_ms
    _bw_ms               = (act2_model_mem_gb * 1e9 / (_device_bw * 1e9)) * 1000
    act2_latency_ms      = max(_latency_ms, _bw_ms)
    _bottleneck          = "Memory-BW bound" if _bw_ms > _latency_ms else "Compute bound"

    # ── Parallelism utilization gauge ─────────────────────────────────────────
    _width_par   = min(1.0, act2_W / 1024)
    _depth_pen   = min(1.0, 24 / act2_L)
    act2_par_pct = _width_par * _depth_pen * 100

    # ── OOM + budget checks ───────────────────────────────────────────────────
    act2_oom         = act2_model_mem_gb > act2_device_ram
    act2_over_budget = act2_params_m > _target_params_m * 1.5

    # ── Build depth vs width comparison curves ────────────────────────────────
    _depths   = np.arange(2, 101, 2)
    _ws_tgt   = np.clip(np.sqrt(_target_params_m * 1e6 / (12 * _depths)), 64, 4096)
    _dl       = np.array([
        max(
            (2 * 12 * w**2 * d / (_device_tops * 1e9)) * 1000,
            ((12 * w**2 * d * BYTES_FP16) / 1e9 * 1e9 / (_device_bw * 1e9)) * 1000,
        ) + d * 0.01 / (1 + _par_factor * 10)
        for d, w in zip(_depths, _ws_tgt)
    ])

    _fig2 = go.Figure()
    _fig2.add_trace(go.Scatter(
        x=_depths, y=_dl,
        name="Latency at 100M param budget",
        line=dict(color=_dev_color, width=2.5),
    ))
    _fig2.add_vline(
        x=act2_L, line_dash="dash", line_color=COLORS["OrangeLine"],
        annotation_text=f"Current: L={act2_L}", annotation_position="top",
    )
    _opt_idx = int(np.argmin(_dl))
    _fig2.add_trace(go.Scatter(
        x=[_depths[_opt_idx]], y=[_dl[_opt_idx]],
        mode="markers+text",
        marker=dict(size=12, color=COLORS["GreenLine"], symbol="star"),
        text=[f"Optimal: L={_depths[_opt_idx]}"],
        textposition="top center",
        name="Optimal depth",
    ))
    _fig2.update_layout(
        xaxis_title="Number of layers (depth)",
        yaxis_title="Estimated inference latency (ms per token)",
        title=f"Depth vs Latency at Fixed 100M Param Budget — {act2_device_name}",
        height=350,
    )
    apply_plotly_theme(_fig2)

    # ── Colors ────────────────────────────────────────────────────────────────
    _gauge_color = (
        COLORS["GreenLine"] if act2_par_pct > 60
        else COLORS["OrangeLine"] if act2_par_pct > 30
        else COLORS["RedLine"]
    )
    _param_color = (
        COLORS["RedLine"] if act2_over_budget or act2_oom
        else COLORS["OrangeLine"] if abs(act2_params_m - _target_params_m) > 30
        else COLORS["GreenLine"]
    )

    mo.vstack([
        mo.md(f"""
        **Physics**

        ```
        params_per_layer  = 12 × width²  (attention + FFN per transformer block)
                          = 12 × {act2_W}² = {_params_per_layer:,}

        total_params      = depth × params_per_layer
                          = {act2_L} × {_params_per_layer:,} = {_total_params:,} ({act2_params_m:.1f} M)

        model_memory      = total_params × 2 bytes (FP16)
                          = {act2_model_mem_gb:.2f} GB

        bottleneck        = {_bottleneck}
        effective_latency ≈ {act2_latency_ms:.3f} ms  (per token, single request)
        ```
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0;">
            <div style="padding: 16px 22px; border: 1.5px solid {COLORS['Border']};
                        border-radius: 12px; min-width: 170px; text-align: center;
                        background: white;">
                <div style="color: #475569; font-size: 0.82rem; font-weight: 500;
                            margin-bottom: 4px;">Total parameters</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {_param_color};">
                    {act2_params_m:.0f} M
                </div>
                <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">
                    target: 100 M
                </div>
            </div>
            <div style="padding: 16px 22px; border: 1.5px solid {COLORS['Border']};
                        border-radius: 12px; min-width: 170px; text-align: center;
                        background: white;">
                <div style="color: #475569; font-size: 0.82rem; font-weight: 500;
                            margin-bottom: 4px;">Model memory (FP16)</div>
                <div style="font-size: 1.8rem; font-weight: 800;
                            color: {COLORS['RedLine'] if act2_oom else COLORS['BlueLine']};">
                    {act2_model_mem_gb:.2f} GB
                </div>
                <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">
                    budget: {act2_device_ram} GB ({act2_device_name})
                </div>
            </div>
            <div style="padding: 16px 22px; border: 1.5px solid {COLORS['Border']};
                        border-radius: 12px; min-width: 170px; text-align: center;
                        background: white;">
                <div style="color: #475569; font-size: 0.82rem; font-weight: 500;
                            margin-bottom: 4px;">Inference latency</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {_dev_color};">
                    {act2_latency_ms:.3f} ms
                </div>
                <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">
                    per token &middot; {_bottleneck}
                </div>
            </div>
            <div style="padding: 16px 22px; border: 1.5px solid {COLORS['Border']};
                        border-radius: 12px; min-width: 170px; text-align: center;
                        background: white;">
                <div style="color: #475569; font-size: 0.82rem; font-weight: 500;
                            margin-bottom: 4px;">Parallelism utilization</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {_gauge_color};">
                    {act2_par_pct:.0f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">
                    width util &times; depth penalty
                </div>
            </div>
        </div>
        """),
        mo.as_html(_fig2),
    ])
    return (
        act2_L, act2_W, act2_params_m, act2_model_mem_gb, act2_latency_ms,
        act2_oom, act2_over_budget, act2_par_pct, act2_device_name, act2_device_ram,
    )


# ─── ACT II FAILURE STATE ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_oom, act2_model_mem_gb, act2_device_ram, act2_device_name,
      act2_over_budget, act2_params_m):
    _under = act2_params_m < 50.0

    if act2_oom:
        mo.callout(mo.md(
            f"**OOM — Model infeasible on {act2_device_name}.** "
            f"Model requires **{act2_model_mem_gb:.1f} GB** RAM. "
            f"{act2_device_name} has **{act2_device_ram} GB**. "
            f"Reduce depth, reduce width, or switch to the Cloud context. "
            f"This is the architecture selection constraint: the contract with physics "
            f"(@sec-network-architectures) is violated before inference even begins."
        ), kind="danger")
    elif act2_over_budget:
        mo.callout(mo.md(
            f"**Over parameter budget.** "
            f"Current design has **{act2_params_m:.0f} M** parameters vs 100 M target. "
            f"Reduce depth or width to stay within the design envelope."
        ), kind="warn")
    elif _under:
        mo.callout(mo.md(
            f"**Under-parameterized.** "
            f"Current design has only **{act2_params_m:.0f} M** parameters. "
            f"The 100 M target gives sufficient capacity for most classification tasks. "
            f"Increase depth or width to reach the budget."
        ), kind="info")
    else:
        mo.callout(mo.md(
            f"**Feasible design.** {act2_params_m:.0f} M params fit on {act2_device_name} "
            f"({act2_model_mem_gb:.2f} GB of {act2_device_ram} GB used). "
            f"Now explore whether going deeper (more layers) or wider (larger hidden dim) "
            f"changes inference latency. The parameter count stays roughly constant "
            f"when you trade depth for width."
        ), kind="success")
    return


# ─── ACT II REVEAL ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction):
    if act2_prediction.value == "wider":
        mo.callout(mo.md(
            "**Correct.** On edge hardware, a wider network (fewer layers, larger hidden dim) "
            "runs faster than a deeper network with the same parameter count. "
            "Each layer is a sequential dependency: the GPU must complete layer L before "
            "it can start layer L+1. On a large H100 with thousands of SMs, this overhead is "
            "hidden by massive parallelism. On a Jetson Orin NX with far fewer compute units, "
            "sequential depth becomes the bottleneck — exactly the same physics as RNNs, "
            "where the chapter notes *'O(T) sequential depth prevents GPU parallelization.'* "
            "Wider layers provide more parallel work *within* a single layer, matching the "
            "hardware's available parallelism."
        ), kind="success")
    elif act2_prediction.value == "deeper":
        mo.callout(mo.md(
            "**Incorrect.** More layers do not mean more parallelism — they mean more *sequential* "
            "dependencies. Each layer must complete before the next begins. On the H100 with "
            "thousands of SMs, this overhead is mostly hidden. On the Jetson Orin NX with far "
            "fewer compute units, each additional layer adds a sequential barrier that the small "
            "device cannot overlap. The chapter's RNN analysis applies here: *'O(T) sequential "
            "depth prevents GPU parallelization.'* For inference on edge hardware, fewer deeper "
            "sequential steps with more parallel work per step wins."
        ), kind="warn")
    elif act2_prediction.value == "same_flops":
        mo.callout(mo.md(
            "**Partially correct, but wrong conclusion.** FLOPs determine the *compute* time, "
            "but inference latency is not purely compute-bound. Every layer boundary is a "
            "synchronization point and kernel launch. On edge hardware with limited compute units, "
            "sequential depth means the hardware cannot pipeline across layers. A 64-layer network "
            "has 64 sequential synchronization barriers; a 4-layer network has 4. With the same "
            "FLOP count, the 4-layer version exposes more parallelism within each wider layer — "
            "and parallelism is the scarce resource on edge hardware."
        ), kind="warn")
    else:  # batch_size
        mo.callout(mo.md(
            "**Incorrect.** Batch size affects throughput (samples/second) but not the "
            "fundamental sequential dependency structure of layers. Even at batch size = 1 "
            "(the common edge inference case), a deeper network has more sequential layers "
            "that must execute in order. Width affects how much parallel work is available "
            "per layer; depth affects how many sequential steps the hardware must take. "
            "The depth/width tradeoff is architectural, not just a batching concern."
        ), kind="warn")
    return


# ─── ACT II REFLECTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reflection

    *The chapter's RNN analysis established: "O(T) sequential depth prevents GPU
    parallelization across the time dimension." The same physics applies to transformer
    layer depth during inference.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A) Shallower models are less accurate and should be avoided": "accuracy",
            "B) Sequential layer depth limits parallelism on smaller accelerators, "
               "making wider shallow networks faster at inference": "sequential",
            "C) Edge devices have fewer total parameters, so depth doesn't matter": "params",
            "D) Wider networks are easier to quantize, reducing latency indirectly": "quantize",
        },
        label="**Why do edge devices prefer shallower, wider architectures for inference?**",
    )
    mo.vstack([
        act2_reflection,
        mo.callout(mo.md("Select your answer."), kind="warn")
        if act2_reflection.value is None
        else mo.md(""),
    ])
    return (act2_reflection,)


@app.cell(hide_code=True)
def _(mo, act2_reflection):
    mo.stop(
        act2_reflection.value is None,
        mo.callout(mo.md("Select your answer above."), kind="warn"),
    )

    if act2_reflection.value == "sequential":
        mo.callout(mo.md(
            "**Correct.** On the H100 with ~16,000 CUDA cores, sequential layer dependencies "
            "are largely hidden — the device has enough parallelism to keep compute units busy "
            "even while waiting for the previous layer's output. On the Jetson Orin NX with far "
            "fewer compute units, each layer boundary is a stall: the device finishes the layer, "
            "synchronizes, and starts the next one. Wider layers expose more parallelism *within* "
            "a single layer computation (larger matrix multiplications), matching the hardware's "
            "available SM count more efficiently."
        ), kind="success")
    elif act2_reflection.value == "accuracy":
        mo.callout(mo.md(
            "**Incorrect — this conflates model quality with systems performance.** "
            "This lab measures inference *latency*, not accuracy. A shallower wider model "
            "can achieve comparable accuracy to a deeper narrower one at the same parameter "
            "count (this is the MobileNet design philosophy). The reason to prefer it on edge "
            "is systems performance: sequential depth limits parallelism on smaller accelerators."
        ), kind="warn")
    elif act2_reflection.value == "params":
        mo.callout(mo.md(
            "**Incorrect.** This lab holds total parameters *fixed* across depth/width variations. "
            "The question is not about parameter count but about architectural shape: a 100M-param "
            "model with 4 layers (wide) has the same parameter count as a 100M-param model with "
            "64 layers (deep). They differ in sequential depth and thus in parallelism utilization "
            "on devices with limited compute units."
        ), kind="warn")
    else:  # quantize
        mo.callout(mo.md(
            "**Indirect at best, not the primary reason.** Quantization benefits do not depend "
            "on network width — both deep and wide networks can be quantized to INT8. The "
            "fundamental reason shallow wide networks are faster on edge hardware is the "
            "sequential depth bottleneck: fewer layer synchronization barriers, more parallel "
            "work per layer, better utilization of a small SM count."
        ), kind="warn")
    return


# ─── ACT II MATHPEEK ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations: Parameter scaling and sequential depth": mo.md("""
        **Parameter count for a transformer-style block:**

        `params_per_layer = 12 × width²`

        - Attention (4×d²) + FFN (8×d²) ≈ 12×d² per layer

        `total_params = depth × 12 × width²`

        **Depth/Width tradeoff at fixed parameter budget P:**

        `width = sqrt(P / (12 × depth))`

        Doubling depth → width shrinks by √2.
        Halving depth → width grows by √2.

        **Sequential depth bottleneck:**

        `inference_latency ≥ depth × t_layer`

        Where `t_layer` is the minimum time to execute a single layer.
        This lower bound is hard: no amount of hardware parallelism within a
        layer can reduce the sequential dependency across layers.

        On an H100 (large SM count): `t_layer` is small → depth overhead is hidden.
        On a Jetson Orin NX (small SM count): `t_layer` is larger → depth multiplies directly
        into latency.

        **Source:** @sec-network-architectures-multilayer-perceptrons-dense-pattern-processing-bc11 —
        parameter scaling formula; RNN sequential bottleneck analysis for depth physics.
        """),
    })
    return


# ─── LEDGER SAVE + HUD ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, ledger,
    context_toggle, act1_prediction, act1_reflection, act2_prediction, act2_reflection,
    act1_seq_slider,
    act2_oom, act2_params_m,
):
    _ctx = context_toggle.value
    _seq_chosen = act1_seq_slider.value

    # Save design decisions to ledger
    ledger.save(
        chapter=6,
        design={
            "context":               _ctx,
            "seq_len_chosen":        _seq_chosen,
            "act1_prediction":       act1_prediction.value or "unanswered",
            "act1_correct":          act1_prediction.value == "quadratic",
            "act1_reflection":       act1_reflection.value or "unanswered",
            "act2_prediction":       act2_prediction.value or "unanswered",
            "act2_reflection":       act2_reflection.value or "unanswered",
            "quadratic_oom_triggered": act2_oom,
            "depth_width_decision":  (act2_prediction.value or "unanswered"),
            "act2_result":           float(act2_params_m),
        },
    )

    _act1_done = act1_prediction.value is not None
    _act2_done = act2_prediction.value is not None

    mo.Html(f"""
    <div class="lab-hud">
        <span>
            <span class="hud-label">LAB</span>&nbsp;
            <span class="hud-value">06 · The Quadratic Wall</span>
        </span>
        <span>
            <span class="hud-label">CONTEXT</span>&nbsp;
            <span class="hud-value">{_ctx.upper()}</span>
        </span>
        <span>
            <span class="hud-label">SEQ LEN</span>&nbsp;
            <span class="hud-value">{_seq_chosen:,} tokens</span>
        </span>
        <span>
            <span class="hud-label">OOM HIT</span>&nbsp;
            <span class="{'hud-none' if act2_oom else 'hud-active'}">{str(act2_oom).upper()}</span>
        </span>
        <span>
            <span class="hud-label">ACT I</span>&nbsp;
            <span class="{'hud-active' if _act1_done else 'hud-none'}">
                {'COMPLETE' if _act1_done else 'PENDING'}
            </span>
        </span>
        <span>
            <span class="hud-label">ACT II</span>&nbsp;
            <span class="{'hud-active' if _act2_done else 'hud-none'}">
                {'COMPLETE' if _act2_done else 'PENDING'}
            </span>
        </span>
    </div>
    """)
    return


# ─── KEY TAKEAWAYS ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("---"),
        mo.callout(mo.md("""
        **Key Takeaways**

        1. **Attention memory is O(N²) — not a linear cost, a quadratic wall.**
           Doubling context from 4K to 8K tokens quadruples attention memory.
           At 100K tokens vs 4K, it is 625× larger. FlashAttention breaks this wall by
           computing attention in tiles without materializing the full N×N matrix in HBM —
           the math is identical; only HBM data movement changes.

        2. **For edge inference, depth is the enemy of latency.**
           Sequential layer dependencies cannot be parallelized away on hardware with limited
           compute units. At a fixed parameter budget, a shallower wider network runs faster
           on a Jetson Orin NX than a deeper narrower one — because it exposes more parallel
           work per layer. The architecture selection decision (depth vs width) is a
           systems contract, not a modeling preference.
        """), kind="info"),
    ])
    return


if __name__ == "__main__":
    app.run()
