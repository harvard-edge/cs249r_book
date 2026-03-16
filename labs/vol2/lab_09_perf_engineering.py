import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-09: THE OPTIMIZATION TRAP
#
# Volume II, Chapter: Performance Engineering (performance_engineering.qmd)
#
# Five Parts (~58 minutes):
#   Part A — The Roofline Diagnostic (10 min)
#             LLM decode at batch=1 is 295x below the H100 ridge point.
#             Prediction: is LLM decode compute-bound or memory-bound?
#             Instrument: interactive log-log Roofline plot with GPU + op selectors.
#
#   Part B — The Fusion Dividend (12 min)
#             Three-level fusion comparison: no fusion vs elementwise vs FlashAttention.
#             Prediction: what is the ratio of FlashAttention to elementwise savings?
#             Instrument: stacked bar chart of HBM traffic per fusion level.
#
#   Part C — FlashAttention: The Savings Curve (12 min)
#             Standard attention O(N^2) vs FlashAttention O(N) memory.
#             Prediction: memory savings ratio at 32K tokens.
#             Instrument: log-log savings curve vs sequence length.
#
#   Part D — Precision Engineering: Naive vs Outlier-Aware (12 min)
#             INT4 quantization shifts roofline, but outliers break naive quant.
#             Prediction: perplexity difference between naive and outlier-aware INT4.
#             Instrument: roofline shift + quality comparison panels.
#
#   Part E — The Optimization Playbook (12 min)
#             Wrong optimization yields zero improvement. Diagnose first.
#             Instrument: mystery workload diagnosis + optimization selector.
#
# Hardware constants:
#   H100 FP16: 989 TFLOPS, 3350 GB/s HBM3, 80 GB, ridge = 295 FLOP/byte
#   V100 FP16: 125 TFLOPS, 900 GB/s HBM2, ridge = 139 FLOP/byte
#   A100 FP16: 312 TFLOPS, 2039 GB/s HBM2e, ridge = 153 FLOP/byte
#   B200 FP16: 2250 TFLOPS, 8000 GB/s HBM3e, ridge = 281 FLOP/byte
#
# Design Ledger: chapter="v2_09"
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: SETUP + OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP ────────────────────────────────────────────────────────────
@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    # WASM bootstrap
    if sys.platform == "emscripten":
        import micropip
        await micropip.install("https://mlsysbook.ai/labs/wheels/mlsysim-0.1.0-py3-none-any.whl")
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog

    ledger = DesignLedger()

    # ── Hardware constants (NVIDIA datasheets) ────────────────────────────────
    H100_TFLOPS_FP16 = 989.0
    H100_BW_GBS      = 3350.0
    H100_RAM_GB      = 80.0
    H100_RIDGE       = H100_TFLOPS_FP16 * 1e12 / (H100_BW_GBS * 1e9)  # ~295

    V100_TFLOPS_FP16 = 125.0
    V100_BW_GBS      = 900.0
    V100_RIDGE       = V100_TFLOPS_FP16 * 1e12 / (V100_BW_GBS * 1e9)  # ~139

    A100_TFLOPS_FP16 = 312.0
    A100_BW_GBS      = 2039.0
    A100_RIDGE       = A100_TFLOPS_FP16 * 1e12 / (A100_BW_GBS * 1e9)  # ~153

    B200_TFLOPS_FP16 = 2250.0
    B200_BW_GBS      = 8000.0
    B200_RIDGE       = B200_TFLOPS_FP16 * 1e12 / (B200_BW_GBS * 1e9)  # ~281

    # ── Transformer workload constants (70B LLM) ─────────────────────────────
    HEAD_DIM     = 128
    NUM_HEADS    = 64
    NUM_LAYERS   = 80
    HIDDEN_DIM   = 8192
    BYTES_FP16   = 2

    # ── Fusion constants ─────────────────────────────────────────────────────
    # Source: @sec-performance-engineering, FlashAttention paper
    # Elementwise fusion savings: ~64 MB/layer (GELU+LN+Dropout intermediates)
    ELEM_FUSION_SAVE_MB = 64.0
    # FlashAttention savings scale with N^2 — at N=4096: ~4 GB/layer
    # Naive attention HBM = 2 * N^2 * bytes_per_elem * num_heads
    # Flash HBM = 4 * N * d * bytes_per_elem * num_heads

    # ── Quantization quality constants ───────────────────────────────────────
    # Source: @sec-performance-engineering, published quantization benchmarks
    # 70B model perplexity at various precision/method combos
    PPL_FP16           = 5.2    # baseline
    PPL_INT8_NAIVE     = 5.3    # INT8 naive — small degradation
    PPL_INT8_OUTLIER   = 5.2    # INT8 outlier-aware — matches FP16
    PPL_INT4_NAIVE     = 13.5   # INT4 naive — catastrophic for large models
    PPL_INT4_OUTLIER   = 5.7    # INT4 outlier-aware — within 0.5 of FP16

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math, DecisionLog,
        H100_TFLOPS_FP16, H100_BW_GBS, H100_RAM_GB, H100_RIDGE,
        V100_TFLOPS_FP16, V100_BW_GBS, V100_RIDGE,
        A100_TFLOPS_FP16, A100_BW_GBS, A100_RIDGE,
        B200_TFLOPS_FP16, B200_BW_GBS, B200_RIDGE,
        HEAD_DIM, NUM_HEADS, NUM_LAYERS, HIDDEN_DIM, BYTES_FP16,
        ELEM_FUSION_SAVE_MB,
        PPL_FP16, PPL_INT8_NAIVE, PPL_INT8_OUTLIER,
        PPL_INT4_NAIVE, PPL_INT4_OUTLIER,
    )


# ─── CELL 1: HEADER ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(LAB_CSS, mo):
    mo.vstack([
        LAB_CSS,
        mo.Html("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 09
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.2rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Optimization Trap
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.1rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Roofline &middot; Fusion &middot; FlashAttention &middot; Quantization
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 700px; line-height: 1.65;">
                A junior engineer profiles a Transformer layer and declares it
                "compute-bound because Transformers are compute-heavy." The roofline
                says otherwise. Most ML operations live far below the ridge point,
                and applying the wrong optimization yields zero improvement.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; align-items: center;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    5 Parts &middot; ~58 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter: Performance Engineering
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px;">
                <span class="badge badge-fail">H100 at 0.3% utilization</span>
                <span class="badge badge-warn">FlashAttention 256x savings at 32K</span>
                <span class="badge badge-info">Profile first, optimize second</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Diagnose bottlenecks with the Roofline Model</strong> &mdash;
                    determine whether LLM decode at batch=1 is compute-bound or memory-bound, and calculate
                    that it achieves only 0.34% of H100 peak utilization.</div>
                <div style="margin-bottom: 3px;">2. <strong>Quantify fusion savings</strong> &mdash; compare elementwise
                    fusion (~64 MB/layer) vs FlashAttention (~4 GB/layer) and discover the 60x ratio between them.</div>
                <div style="margin-bottom: 3px;">3. <strong>Match optimizations to bottlenecks</strong> &mdash; predict which
                    optimization helps a given workload and discover that applying the wrong one yields &lt;5% improvement.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    Roofline model and arithmetic intensity from @sec-performance-engineering
                    &middot; Iron Law of ML Performance &middot; Lab V2-08 (inference at scale)
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~58 min</strong><br/>
                    A: 10 &middot; B: 12 &middot; C: 12 &middot; D: 12 &middot; E: 12 min
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;Your H100 has 989 TFLOPS. LLM decode uses 0.3% of it. Three techniques
                &mdash; fusion, tiling, and quantization &mdash; close the gap. But which one
                helps which workload, and why does applying the wrong one yield zero improvement?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** &mdash; Complete the following before this lab:

    - **The Roofline Model** &mdash; Arithmetic intensity, ridge point formula,
      workload placement on the roofline (@sec-performance-engineering).
    - **FlashAttention: Tiled Attention as a System Primitive** &mdash; How tiling
      reduces HBM traffic from O(N^2) to O(N).
    - **Operator Fusion** &mdash; Elementwise fusion vs full attention block fusion
      and the HBM traffic reduction for each.
    - **Precision Engineering** &mdash; How INT4 quantization shifts the roofline
      and why outlier-aware methods preserve quality.
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: PART A — THE ROOFLINE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 4: PART A BANNER ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div id="part-a" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['BlueLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">A</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Part A &middot; 10 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            The Roofline Diagnostic
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            You expect a powerful GPU to be compute-bound. The roofline reveals that LLM
            decode at batch=1 sits 295x below the H100 ridge point &mdash; using only 0.34%
            of available compute. The GPU is not slow. It is starving for data.
        </div>
    </div>
    """)
    return


# ─── CELL 5: PART A STAKEHOLDER ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']}; background: {COLORS['BlueL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['BlueLine']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Junior ML Engineer, LLM Serving Team
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "I profiled our 70B model inference on the H100 cluster. Transformers are
            all matrix multiplications, so it should be compute-bound, right? I want to
            request faster GPUs to improve our token generation speed."
        </div>
    </div>
    """)
    return


# ─── CELL 6: PART A PREDICTION ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    pA_pred = mo.ui.radio(
        options={
            "A: Compute-bound -- H100 has 989 TFLOPS, compute must dominate": "A",
            "B: Slightly memory-bound -- close to the ridge point": "B",
            "C: Deeply memory-bound -- 100x+ below the ridge point": "C",
            "D: It depends on the model size": "D",
        },
        label=(
            "LLM decode (batch=1) on an H100. The model reads its full weight matrix "
            "to generate each token, performing ~1 FLOP per byte loaded (FP16). "
            "The H100 ridge point is 295 FLOP/byte. Is this workload compute-bound or memory-bound?"
        ),
    )
    mo.vstack([
        mo.md("### Your Prediction"),
        mo.md("*Commit before touching the simulator.*"),
        pA_pred,
    ])
    return (pA_pred,)


# ─── CELL 7: PART A GATE ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pA_pred):
    mo.stop(
        pA_pred.value is None,
        mo.callout(mo.md("Select your prediction above to unlock the Roofline simulator."), kind="warn"),
    )
    mo.callout(
        mo.md(f"**Prediction locked:** Option **{pA_pred.value}**. Now explore the simulator below."),
        kind="info",
    )
    return


# ─── CELL 8: PART A CONTROLS ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    pA_batch = mo.ui.dropdown(
        options={"Batch 1": 1, "Batch 4": 4, "Batch 16": 16,
                 "Batch 64": 64, "Batch 256": 256},
        value="Batch 1",
        label="Batch size",
    )
    pA_gpu = mo.ui.dropdown(
        options={
            "V100 FP16 (ridge = 139)": "v100",
            "A100 FP16 (ridge = 153)": "a100",
            "H100 FP16 (ridge = 295)": "h100",
            "B200 FP16 (ridge = 281)": "b200",
        },
        value="H100 FP16 (ridge = 295)",
        label="GPU generation",
    )
    pA_op = mo.ui.dropdown(
        options={
            "LLM Decode (AI = batch FLOP/byte)": "decode",
            "LayerNorm (AI ~ 1.5)": "layernorm",
            "Large GEMM 4096x4096 (AI ~ 1365)": "gemm",
            "Attention naive (AI ~ 64)": "attn_naive",
        },
        value="LLM Decode (AI = batch FLOP/byte)",
        label="ML operation",
    )
    mo.hstack([pA_batch, pA_gpu, pA_op], gap="1.5rem", justify="start")
    return (pA_batch, pA_gpu, pA_op)


# ─── CELL 9: PART A ROOFLINE CHART ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, H100_TFLOPS_FP16, H100_BW_GBS, H100_RIDGE,
    A100_TFLOPS_FP16, A100_BW_GBS, A100_RIDGE,
    V100_TFLOPS_FP16, V100_BW_GBS, V100_RIDGE,
    B200_TFLOPS_FP16, B200_BW_GBS, B200_RIDGE,
    apply_plotly_theme, go, math, mo, np,
    pA_batch, pA_gpu, pA_op,
):
    _GPU_SPECS = {
        "h100": ("H100", H100_TFLOPS_FP16, H100_BW_GBS, H100_RIDGE, COLORS["BlueLine"]),
        "a100": ("A100", A100_TFLOPS_FP16, A100_BW_GBS, A100_RIDGE, COLORS["OrangeLine"]),
        "v100": ("V100", V100_TFLOPS_FP16, V100_BW_GBS, V100_RIDGE, "#7c3aed"),
        "b200": ("B200", B200_TFLOPS_FP16, B200_BW_GBS, B200_RIDGE, COLORS["GreenLine"]),
    }
    _gpu_name, _peak, _bw, _ridge, _gpu_color = _GPU_SPECS[pA_gpu.value]
    _batch = pA_batch.value

    # Operation arithmetic intensity
    _OP_AI = {
        "decode": float(_batch),
        "layernorm": 1.5,
        "gemm": 1365.0,
        "attn_naive": 64.0,
    }
    _ai = _OP_AI[pA_op.value]

    # Roofline calculation
    _achievable = min(_peak, (_bw / 1e3) * _ai)
    _util_pct = (_achievable / _peak) * 100.0
    _is_mem_bound = _ai < _ridge

    # Build figure
    _ai_range = np.logspace(-1, 4, 400)
    _mem_ceil = (_bw / 1e3) * _ai_range
    _comp_ceil = np.full_like(_ai_range, _peak)
    _roofline = np.minimum(_mem_ceil, _comp_ceil)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_ai_range, y=_mem_ceil, mode="lines", name="Memory BW ceiling",
        line=dict(color=COLORS["RedLine"], width=2),
    ))
    _fig.add_trace(go.Scatter(
        x=_ai_range, y=_comp_ceil, mode="lines",
        name=f"Compute ceiling ({_peak:.0f} TFLOPS)",
        line=dict(color=COLORS["BlueLine"], width=2),
    ))
    _fig.add_trace(go.Scatter(
        x=_ai_range, y=_roofline, mode="lines", name="Roofline",
        line=dict(color="#334155", width=3), showlegend=False,
    ))
    _fig.add_vline(
        x=_ridge, line_dash="dash", line_color="#94a3b8", line_width=1.5,
        annotation_text=f"Ridge = {_ridge:.0f}", annotation_position="top right",
        annotation_font_size=10, annotation_font_color="#94a3b8",
    )
    _dot_color = COLORS["RedLine"] if _is_mem_bound else COLORS["GreenLine"]
    _fig.add_trace(go.Scatter(
        x=[_ai], y=[_achievable], mode="markers",
        name=f"{'Memory-bound' if _is_mem_bound else 'Compute-bound'}: AI={_ai:.1f}",
        marker=dict(size=16, color=_dot_color, line=dict(color="white", width=2)),
    ))
    _fig.add_annotation(
        x=math.log10(max(_ai, 0.1)), y=math.log10(max(_achievable, 0.01)),
        xref="x", yref="y",
        text=f"Utilization: {_util_pct:.2f}%",
        showarrow=True, arrowhead=2, arrowcolor=_dot_color,
        ax=60, ay=-40,
        font=dict(size=11, color=_dot_color, family="SF Mono, monospace"),
        bgcolor="white", bordercolor=_dot_color, borderpad=4,
    )
    _fig.update_layout(
        height=380,
        xaxis=dict(title="Arithmetic Intensity (FLOP/byte)", type="log", range=[-1, 4]),
        yaxis=dict(title="Achievable Performance (TFLOPS)", type="log", range=[-1, 4]),
        legend=dict(orientation="h", y=-0.25, x=0, font_size=11),
        margin=dict(l=60, r=20, t=30, b=110),
    )
    apply_plotly_theme(_fig)

    # Metric cards
    _u_color = (COLORS["GreenLine"] if _util_pct > 20 else
                COLORS["OrangeLine"] if _util_pct > 5 else COLORS["RedLine"])
    _regime = "Memory-Bound" if _is_mem_bound else "Compute-Bound"
    _r_color = COLORS["RedLine"] if _is_mem_bound else COLORS["GreenLine"]

    _cards = f"""
    <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
        <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:140px; text-align:center; background:white;
                    border-top:3px solid {_u_color}; flex:1;">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Utilization</div>
            <div style="font-size:1.7rem; font-weight:800; color:{_u_color};">{_util_pct:.2f}%</div>
        </div>
        <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:140px; text-align:center; background:white;
                    border-top:3px solid {_r_color}; flex:1;">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Regime</div>
            <div style="font-size:1.3rem; font-weight:800; color:{_r_color};">{_regime}</div>
        </div>
        <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:140px; text-align:center; background:white;
                    border-top:3px solid {COLORS['BlueLine']}; flex:1;">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">AI / Ridge</div>
            <div style="font-size:1.3rem; font-weight:800; color:{COLORS['BlueLine']};">
                {_ai:.1f} / {_ridge:.0f}</div>
        </div>
    </div>"""

    mo.vstack([
        mo.as_html(_fig),
        mo.Html(_cards),
        mo.md(f"""
**Roofline Calculation**

```
Arithmetic Intensity  = {_ai:.1f} FLOP/byte
Ridge Point ({_gpu_name})    = {_ridge:.0f} FLOP/byte
AI / Ridge            = {_ai:.1f} / {_ridge:.0f} = {_ai/_ridge:.4f}
Achievable TFLOPS     = min({_peak:.0f}, {_bw/1e3:.1f} x {_ai:.1f}) = {_achievable:.1f} TFLOPS
Utilization           = {_achievable:.1f} / {_peak:.0f} = {_util_pct:.2f}%
```
*Source: @sec-performance-engineering, roofline model*
"""),
    ])
    return (pA_batch, pA_gpu, pA_op)


# ─── CELL 10: PART A REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo, pA_pred):
    if pA_pred.value == "C":
        _pA_msg = (
            "**Correct.** LLM decode at batch=1 has arithmetic intensity of ~1 FLOP/byte. "
            "The H100 ridge point is 295. The workload is 295x below the ridge, achieving "
            "only 0.34% utilization. The fix is not faster math -- it is moving less data."
        )
        _pA_kind = "success"
    elif pA_pred.value == "A":
        _pA_msg = (
            "**The GPU is almost entirely idle.** You predicted compute-bound, but LLM decode "
            "at batch=1 has AI=1 FLOP/byte -- 295x below the H100 ridge point. Utilization is "
            "0.34%, not the 80%+ you expected. The 989 TFLOPS are irrelevant when the workload "
            "cannot feed data fast enough."
        )
        _pA_kind = "warn"
    elif pA_pred.value == "B":
        _pA_msg = (
            "**Much worse than 'slightly' memory-bound.** AI=1 is not 'close to' the ridge "
            "at 295 -- it is 295x below. This is not a marginal inefficiency. The GPU achieves "
            "0.34% utilization. Increasing batch size is the primary lever to improve this."
        )
        _pA_kind = "warn"
    else:
        _pA_msg = (
            "**Model size matters for total memory, not for the roofline regime.** At batch=1, "
            "ANY autoregressive LLM has AI~1 FLOP/byte regardless of size. The ridge point "
            "determines the regime, and at AI=1 on the H100 (ridge=295), every LLM decode is "
            "deeply memory-bound."
        )
        _pA_kind = "warn"

    mo.vstack([
        mo.md(f"**You predicted:** {pA_pred.value}"),
        mo.callout(mo.md(_pA_msg), kind=_pA_kind),
        mo.accordion({
            "Math Peek: Roofline Model": mo.md("""
**The Roofline Equation**

$$\\text{Achievable FLOPS} = \\min(\\pi, \\beta \\cdot I)$$

Where:
- $\\pi$ = peak compute (TFLOPS)
- $\\beta$ = memory bandwidth (TB/s)
- $I$ = arithmetic intensity (FLOP/byte)

The **ridge point** $I_{\\text{ridge}} = \\pi / \\beta$ divides memory-bound ($I < I_{\\text{ridge}}$)
from compute-bound ($I > I_{\\text{ridge}}$).

For LLM decode at batch=1: $I = 1$ FLOP/byte, $I_{\\text{ridge}} = 295$ on H100.
Utilization = $I / I_{\\text{ridge}} = 1/295 = 0.34\\%$.

*Source: @sec-performance-engineering*
"""),
        }),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: PART B — THE FUSION DIVIDEND
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 11: PART B BANNER ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div id="part-b" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['GreenLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">B</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Part B &middot; 12 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            The Fusion Dividend
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            LLM decode is memory-bound &mdash; the fix is moving less data. Fusion eliminates
            HBM round-trips by keeping intermediates in SRAM. But not all fusions are equal:
            FlashAttention saves 60x more than elementwise fusion.
        </div>
    </div>
    """)
    return


# ─── CELL 12: PART B PREDICTION ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    pB_pred = mo.ui.radio(
        options={
            "A: ~2x -- fusion is fusion, all savings are similar": "A",
            "B: ~10x -- attention is bigger but not dramatically": "B",
            "C: ~60x -- attention score matrix is quadratically larger": "C",
            "D: ~1000x -- FlashAttention is a completely different class": "D",
        },
        label=(
            "A Transformer layer has 50+ operations. Fusing three elementwise ops "
            "(GELU + LayerNorm + Dropout) saves ~64 MB/layer in HBM traffic. "
            "FlashAttention fuses the entire attention block. What is the ratio of "
            "FlashAttention savings to elementwise fusion savings?"
        ),
    )
    mo.vstack([
        mo.md("### Your Prediction"),
        mo.md("*What is the ratio of FlashAttention savings to elementwise fusion savings?*"),
        pB_pred,
    ])
    return (pB_pred,)


# ─── CELL 13: PART B GATE ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pB_pred):
    mo.stop(
        pB_pred.value is None,
        mo.callout(mo.md("Select your prediction above to unlock the Fusion comparison."), kind="warn"),
    )
    return


# ─── CELL 14: PART B CONTROLS + CHART ───────────────────────────────────────
@app.cell(hide_code=True)
def _(BYTES_FP16, COLORS, ELEM_FUSION_SAVE_MB, NUM_HEADS,
      apply_plotly_theme, go, mo):
    pB_seqlen = mo.ui.slider(
        start=512, stop=131072, value=4096, step=512,
        label="Sequence length (tokens)",
    )
    mo.vstack([pB_seqlen])
    return (pB_seqlen,)


@app.cell(hide_code=True)
def _(BYTES_FP16, COLORS, ELEM_FUSION_SAVE_MB, HEAD_DIM, NUM_HEADS,
      apply_plotly_theme, go, mo, pB_seqlen):
    _N = pB_seqlen.value

    # HBM traffic calculation per layer
    # No fusion: naive attention materializes N x N score matrix per head
    _naive_attn_mb = 2 * _N * _N * BYTES_FP16 * NUM_HEADS / (1024 * 1024)
    # Elementwise intermediates (GELU + LN + Dropout): ~64 MB fixed per layer
    _elem_traffic_mb = ELEM_FUSION_SAVE_MB
    # Total naive HBM traffic
    _no_fusion_mb = _naive_attn_mb + _elem_traffic_mb

    # Elementwise fusion: removes elementwise round-trips, attention still naive
    _elem_fused_mb = _naive_attn_mb  # only attention traffic remains

    # FlashAttention: O(N) memory -- reads Q,K,V,O only
    _flash_attn_mb = 4 * _N * HEAD_DIM * BYTES_FP16 * NUM_HEADS / (1024 * 1024)
    _full_fused_mb = _flash_attn_mb

    # Savings
    _elem_savings_mb = _no_fusion_mb - _elem_fused_mb
    _flash_savings_mb = _no_fusion_mb - _full_fused_mb
    _ratio = _flash_savings_mb / max(_elem_savings_mb, 0.01)

    # Build stacked bar chart
    _labels = ["No Fusion", "Elementwise Fusion", "FlashAttention"]
    _attn_vals = [_naive_attn_mb, _naive_attn_mb, _flash_attn_mb]
    _elem_vals = [_elem_traffic_mb, 0, 0]

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        name="Attention HBM Traffic", x=_labels, y=_attn_vals,
        marker_color=COLORS["RedLine"], opacity=0.85,
    ))
    _fig.add_trace(go.Bar(
        name="Elementwise HBM Traffic", x=_labels, y=_elem_vals,
        marker_color=COLORS["OrangeLine"], opacity=0.85,
    ))
    _fig.update_layout(
        barmode="stack", height=360,
        yaxis=dict(title="HBM Traffic per Layer (MB)", type="log"),
        margin=dict(l=60, r=20, t=30, b=40),
        legend=dict(orientation="h", y=-0.15, x=0),
    )
    apply_plotly_theme(_fig)

    # Cards
    _cards = f"""
    <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
        <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Elementwise Savings</div>
            <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">
                {_elem_savings_mb:.0f} MB</div>
        </div>
        <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {COLORS['GreenLine']}; flex:1;">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">FlashAttention Savings</div>
            <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">
                {_flash_savings_mb:,.0f} MB</div>
        </div>
        <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {COLORS['BlueLine']}; flex:1;">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Savings Ratio</div>
            <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">
                {_ratio:.0f}x</div>
        </div>
    </div>"""

    mo.vstack([
        mo.as_html(_fig),
        mo.Html(_cards),
        mo.md(f"""
**Fusion Physics** (sequence length = {_N:,} tokens)

```
Naive attention HBM    = 2 * N^2 * 2 bytes * {NUM_HEADS} heads
                       = 2 * {_N}^2 * 2 * {NUM_HEADS} = {_naive_attn_mb:,.0f} MB

FlashAttention HBM     = 4 * N * d * 2 bytes * {NUM_HEADS} heads
                       = 4 * {_N} * {HEAD_DIM} * 2 * {NUM_HEADS} = {_flash_attn_mb:.1f} MB

Elementwise savings    = {_elem_savings_mb:.0f} MB (fixed per layer)
FlashAttention savings = {_flash_savings_mb:,.0f} MB
Ratio                  = {_flash_savings_mb:,.0f} / {_elem_savings_mb:.0f} = {_ratio:.0f}x
```
"""),
    ])
    return


# ─── CELL 15: PART B REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pB_pred):
    if pB_pred.value == "C":
        _pB_msg = (
            "**Correct.** FlashAttention saves ~60x more HBM traffic than elementwise fusion "
            "because the attention score matrix grows quadratically with sequence length (N^2) "
            "while elementwise intermediates are fixed. At 4K tokens, this is already 60x. "
            "At 32K tokens, the ratio grows to ~4000x."
        )
        _pB_kind = "success"
    else:
        _pB_msg = (
            "**The ratio is ~60x at 4K tokens, and it grows quadratically.** "
            "Elementwise fusion saves ~64 MB/layer (fixed). FlashAttention eliminates the "
            "N x N score matrix, saving ~4 GB/layer at 4K tokens. The score matrix grows as "
            "O(N^2) while elementwise intermediates are O(N), so the ratio widens with "
            "sequence length. Increase the slider to see it grow."
        )
        _pB_kind = "warn"

    mo.vstack([
        mo.callout(mo.md(_pB_msg), kind=_pB_kind),
        mo.accordion({
            "Math Peek: Fusion Savings": mo.md("""
**HBM Traffic Reduction**

Naive attention per layer: $\\text{HBM}_{\\text{naive}} = 2 N^2 \\cdot b \\cdot H$ bytes

FlashAttention per layer: $\\text{HBM}_{\\text{flash}} = 4 N d \\cdot b \\cdot H$ bytes

Savings ratio: $\\frac{N^2}{2Nd} = \\frac{N}{2d}$

At $N = 4096$, $d = 128$: ratio = $4096 / 256 = 16$, but including the full traffic
accounting the actual ratio is ~60x because FlashAttention eliminates *all* intermediate
materializations, not just the score matrix.

*Source: @sec-performance-engineering, FlashAttention*
"""),
        }),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: PART C — FLASHATTENTION: THE SAVINGS CURVE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 16: PART C BANNER ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div id="part-c" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['OrangeLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">C</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Part C &middot; 12 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            FlashAttention: The Savings Curve
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            Standard attention materializes an N x N score matrix in HBM, costing O(N^2)
            memory. FlashAttention tiles the computation to SRAM, reducing memory to O(N).
            The savings ratio grows linearly with sequence length: 32x at 8K, 256x at 32K.
        </div>
    </div>
    """)
    return


# ─── CELL 17: PART C PREDICTION ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    pC_pred = mo.ui.number(
        start=1, stop=10000, value=None,
        label=(
            "Standard attention at 32K tokens uses ~32 GB of HBM for the score matrix "
            "(FP16, 64 heads). FlashAttention uses tiled SRAM computation. "
            "What are the memory savings? Enter a multiplier (e.g., 8 for 8x)."
        ),
    )
    mo.vstack([
        mo.md("### Your Prediction"),
        mo.md("*How much memory does FlashAttention save at 32K tokens?*"),
        pC_pred,
    ])
    return (pC_pred,)


# ─── CELL 18: PART C GATE ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pC_pred):
    mo.stop(
        pC_pred.value is None,
        mo.callout(mo.md("Enter your savings prediction above to unlock the savings curve."), kind="warn"),
    )
    return


# ─── CELL 19: PART C SAVINGS CURVE ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(BYTES_FP16, COLORS, HEAD_DIM, NUM_HEADS,
      apply_plotly_theme, go, mo, np, pC_pred):
    _d = HEAD_DIM
    _h = NUM_HEADS
    _b = BYTES_FP16

    _seq_lengths = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])

    # Naive: 2 * N^2 * bytes * heads (score matrix + softmax output)
    _naive_gb = 2 * _seq_lengths.astype(float)**2 * _b * _h / (1024**3)
    # Flash: 4 * N * d * bytes * heads (Q, K, V, O reads)
    _flash_gb = 4 * _seq_lengths.astype(float) * _d * _b * _h / (1024**3)
    # Savings ratio
    _ratios = _naive_gb / _flash_gb

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_seq_lengths, y=_naive_gb, mode="lines+markers",
        name="Naive Attention (O(N^2))",
        line=dict(color=COLORS["RedLine"], width=3),
        marker=dict(size=8),
    ))
    _fig.add_trace(go.Scatter(
        x=_seq_lengths, y=_flash_gb, mode="lines+markers",
        name="FlashAttention (O(N))",
        line=dict(color=COLORS["GreenLine"], width=3),
        marker=dict(size=8),
    ))

    # Annotate key points
    for _i, _n in enumerate(_seq_lengths):
        if _n in [8192, 32768, 131072]:
            _fig.add_annotation(
                x=_n, y=_naive_gb[_i],
                text=f"{_ratios[_i]:.0f}x savings",
                showarrow=True, arrowhead=2, ax=40, ay=-30,
                font=dict(size=10, color=COLORS["BlueLine"]),
            )

    _fig.update_layout(
        height=380,
        xaxis=dict(title="Sequence Length (tokens)", type="log"),
        yaxis=dict(title="HBM Memory (GB)", type="log"),
        legend=dict(orientation="h", y=-0.2, x=0),
        margin=dict(l=60, r=20, t=30, b=90),
    )
    apply_plotly_theme(_fig)

    # Find the 32K ratio
    _idx_32k = list(_seq_lengths).index(32768)
    _actual_ratio = _ratios[_idx_32k]
    _predicted = pC_pred.value if pC_pred.value else 1
    _gap = _actual_ratio / _predicted

    _cards = f"""
    <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
        <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {COLORS['RedLine']}; flex:1;">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Naive at 32K</div>
            <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">
                {_naive_gb[_idx_32k]:.1f} GB</div>
        </div>
        <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {COLORS['GreenLine']}; flex:1;">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Flash at 32K</div>
            <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">
                {_flash_gb[_idx_32k]*1024:.0f} MB</div>
        </div>
        <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {COLORS['BlueLine']}; flex:1;">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Savings Ratio</div>
            <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">
                {_actual_ratio:.0f}x</div>
        </div>
    </div>"""

    if _gap < 1.5 and _gap > 0.67:
        _reveal_kind = "success"
        _reveal_msg = f"**Excellent.** You predicted {_predicted}x. Actual: {_actual_ratio:.0f}x. You were within range."
    else:
        _reveal_kind = "warn"
        _reveal_msg = (
            f"**You predicted {_predicted}x. Actual: {_actual_ratio:.0f}x. "
            f"You were off by {_gap:.1f}x.** Most students predict 4-8x (thinking linearly). "
            f"The actual savings ratio is N/(2d) = {32768}/{2*_d} = {_actual_ratio:.0f}x, "
            f"growing linearly with sequence length."
        )

    mo.vstack([
        mo.as_html(_fig),
        mo.Html(_cards),
        mo.callout(mo.md(_reveal_msg), kind=_reveal_kind),
        mo.md(f"""
**Savings Formula**

```
Naive HBM  = 2 * N^2 * bytes * heads = 2 * 32768^2 * 2 * 64 = {_naive_gb[_idx_32k]:.1f} GB
Flash HBM  = 4 * N * d * bytes * heads = 4 * 32768 * 128 * 2 * 64 = {_flash_gb[_idx_32k]*1024:.0f} MB
Ratio      = N / (2d) = 32768 / 256 = {_actual_ratio:.0f}x
```
*Source: @sec-performance-engineering, FlashAttention tiling analysis*
"""),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE E: PART D — PRECISION ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: PART D BANNER ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div id="part-d" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: #7c3aed; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">D</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Part D &middot; 12 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            Precision Engineering: Naive vs Outlier-Aware
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            INT4 quantization quadruples effective memory bandwidth, shifting the workload
            rightward on the roofline. But transformer models have outlier features &mdash;
            a handful of channels with values 100x larger. Naive quantization clips these
            outliers, causing catastrophic accuracy loss.
        </div>
    </div>
    """)
    return


# ─── CELL 21: PART D PREDICTION ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    pD_pred = mo.ui.radio(
        options={
            "A: Both are similar -- 4 bits is 4 bits": "A",
            "B: Naive is 1-2 perplexity points worse": "B",
            "C: Naive is 5-10 points worse -- catastrophic for large models": "C",
            "D: Outlier-aware is worse due to overhead": "D",
        },
        label=(
            "You quantize a 70B LLM from FP16 to INT4. Naive (uniform) quantization vs "
            "outlier-aware quantization. Both use 4 bits per weight on average. "
            "What is the perplexity difference?"
        ),
    )
    mo.vstack([
        mo.md("### Your Prediction"),
        pD_pred,
    ])
    return (pD_pred,)


# ─── CELL 22: PART D GATE ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pD_pred):
    mo.stop(
        pD_pred.value is None,
        mo.callout(mo.md("Select your prediction to unlock the quantization comparison."), kind="warn"),
    )
    return


# ─── CELL 23: PART D INSTRUMENT ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, H100_BW_GBS, H100_RIDGE, H100_TFLOPS_FP16,
      PPL_FP16, PPL_INT4_NAIVE, PPL_INT4_OUTLIER,
      PPL_INT8_NAIVE, PPL_INT8_OUTLIER,
      apply_plotly_theme, go, mo, np):
    pD_precision = mo.ui.dropdown(
        options={"FP16": 16, "INT8": 8, "INT4": 4},
        value="INT4",
        label="Precision (bits)",
    )
    mo.vstack([pD_precision])
    return (pD_precision,)


@app.cell(hide_code=True)
def _(COLORS, H100_BW_GBS, H100_RIDGE, H100_TFLOPS_FP16,
      PPL_FP16, PPL_INT4_NAIVE, PPL_INT4_OUTLIER,
      PPL_INT8_NAIVE, PPL_INT8_OUTLIER,
      apply_plotly_theme, go, mo, np, pD_precision):
    _bits = pD_precision.value
    _bw_mult = 16 / _bits  # effective bandwidth multiplier
    _eff_bw = H100_BW_GBS * _bw_mult
    _eff_ridge = H100_TFLOPS_FP16 * 1e12 / (_eff_bw * 1e9)

    # Quality lookup
    _ppl_table = {
        16: {"naive": PPL_FP16, "outlier": PPL_FP16},
        8:  {"naive": PPL_INT8_NAIVE, "outlier": PPL_INT8_OUTLIER},
        4:  {"naive": PPL_INT4_NAIVE, "outlier": PPL_INT4_OUTLIER},
    }
    _ppl_naive = _ppl_table[_bits]["naive"]
    _ppl_outlier = _ppl_table[_bits]["outlier"]

    # Left panel: roofline shift
    _ai_range = np.logspace(-1, 4, 400)
    _mem_ceil_orig = (H100_BW_GBS / 1e3) * _ai_range
    _mem_ceil_quant = (_eff_bw / 1e3) * _ai_range
    _comp_ceil = np.full_like(_ai_range, H100_TFLOPS_FP16)

    _fig_roof = go.Figure()
    _fig_roof.add_trace(go.Scatter(
        x=_ai_range, y=_mem_ceil_orig, mode="lines",
        name="FP16 bandwidth ceiling",
        line=dict(color="#94a3b8", width=2, dash="dash"),
    ))
    _fig_roof.add_trace(go.Scatter(
        x=_ai_range, y=_mem_ceil_quant, mode="lines",
        name=f"INT{_bits} effective bandwidth ({_bw_mult:.0f}x)",
        line=dict(color=COLORS["GreenLine"], width=2),
    ))
    _fig_roof.add_trace(go.Scatter(
        x=_ai_range, y=_comp_ceil, mode="lines",
        name="Compute ceiling",
        line=dict(color=COLORS["BlueLine"], width=2),
    ))
    _fig_roof.add_vline(x=H100_RIDGE, line_dash="dot", line_color="#ccc")
    _fig_roof.add_vline(x=_eff_ridge, line_dash="dash", line_color=COLORS["GreenLine"],
                        annotation_text=f"New ridge: {_eff_ridge:.0f}",
                        annotation_font_size=10)
    _fig_roof.update_layout(
        height=320, title=dict(text="Roofline Shift with Quantization", font_size=13),
        xaxis=dict(title="Arithmetic Intensity (FLOP/byte)", type="log", range=[-1, 4]),
        yaxis=dict(title="TFLOPS", type="log", range=[-1, 4]),
        legend=dict(orientation="h", y=-0.25, font_size=10),
        margin=dict(l=50, r=20, t=40, b=100),
    )
    apply_plotly_theme(_fig_roof)

    # Right panel: quality comparison
    _fig_qual = go.Figure()
    _fig_qual.add_trace(go.Bar(
        x=["Naive", "Outlier-Aware", "FP16 Baseline"],
        y=[_ppl_naive, _ppl_outlier, PPL_FP16],
        marker_color=[COLORS["RedLine"], COLORS["GreenLine"], COLORS["BlueLine"]],
        text=[f"{_ppl_naive:.1f}", f"{_ppl_outlier:.1f}", f"{PPL_FP16:.1f}"],
        textposition="outside",
    ))
    _fig_qual.update_layout(
        height=320, title=dict(text=f"Perplexity at INT{_bits} (lower is better)", font_size=13),
        yaxis=dict(title="Perplexity", range=[0, max(_ppl_naive + 2, 8)]),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    apply_plotly_theme(_fig_qual)

    _ppl_diff = _ppl_naive - _ppl_outlier
    _d_color = COLORS["RedLine"] if _ppl_diff > 3 else COLORS["OrangeLine"] if _ppl_diff > 1 else COLORS["GreenLine"]

    mo.vstack([
        mo.hstack([mo.as_html(_fig_roof), mo.as_html(_fig_qual)]),
        mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">BW Multiplier</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_bw_mult:.0f}x</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_d_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Perplexity Gap (naive - outlier)</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_d_color};">{_ppl_diff:.1f}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">New Ridge Point</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_eff_ridge:.0f}</div>
            </div>
        </div>"""),
    ])
    return


# ─── CELL 24: PART D REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(PPL_INT4_NAIVE, PPL_INT4_OUTLIER, mo, pD_pred):
    _diff = PPL_INT4_NAIVE - PPL_INT4_OUTLIER
    if pD_pred.value == "C":
        _pD_msg = (
            f"**Correct.** Naive INT4 perplexity spikes by {_diff:.1f} points for a 70B model "
            "due to outlier clipping. Outlier-aware methods protect the salient 1% of weights "
            "at higher precision, staying within 0.5 points of FP16. The bandwidth gain is "
            "identical for both methods -- the quality cost is in the method, not the bit-width."
        )
        _pD_kind = "success"
    else:
        _pD_msg = (
            f"**Naive INT4 is catastrophic for large models.** The perplexity gap is {_diff:.1f} "
            "points -- not similar, not mild, but catastrophic. Transformer models have outlier "
            "features (channels with values 100x larger than the rest). Naive quantization clips "
            "these, destroying model quality. Outlier-aware methods protect the top 1% at full "
            "precision, preserving quality while capturing the same bandwidth gain."
        )
        _pD_kind = "warn"

    mo.callout(mo.md(_pD_msg), kind=_pD_kind)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE F: PART E — THE OPTIMIZATION PLAYBOOK
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 25: PART E BANNER ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div id="part-e" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['RedLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">E</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Part E &middot; 12 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            The Optimization Playbook
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            Applying the wrong optimization yields zero improvement. FlashAttention on a
            compute-bound prefill workload? Negligible gain. INT4 quantization on a
            memory-bound decode workload? Huge gain. The meta-skill: diagnose first, then treat.
        </div>
    </div>
    """)
    return


# ─── CELL 26: PART E PREDICTION ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    pE_pred = mo.ui.radio(
        options={
            "A: FlashAttention -- it is the most powerful optimization": "A",
            "B: INT4 quantization -- reduce data movement": "B",
            "C: It depends on whether the workload is compute-bound or memory-bound": "C",
            "D: All optimizations help equally": "D",
        },
        label=(
            "You have a mystery workload. Without profiling it first, which single "
            "optimization would you apply?"
        ),
    )
    mo.vstack([
        mo.md("### Your Prediction"),
        pE_pred,
    ])
    return (pE_pred,)


# ─── CELL 27: PART E GATE ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pE_pred):
    mo.stop(
        pE_pred.value is None,
        mo.callout(mo.md("Select your prediction to unlock the optimization playbook."), kind="warn"),
    )
    return


# ─── CELL 28: PART E INSTRUMENT ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, go, mo):
    pE_workload = mo.ui.dropdown(
        options={
            "LLM Decode (batch=1) -- AI=1, memory-bound": "decode",
            "LLM Prefill (batch=64) -- AI=512, compute-bound": "prefill",
            "Vision Inference (batch=32) -- AI=180, near ridge": "vision",
        },
        value="LLM Decode (batch=1) -- AI=1, memory-bound",
        label="Select workload to optimize",
    )
    pE_optim = mo.ui.dropdown(
        options={
            "FlashAttention (reduces HBM traffic)": "flash",
            "INT4 Quantization (4x effective bandwidth)": "int4",
            "Operator Fusion (fuse elementwise ops)": "fusion",
            "CUDA Graphs (reduce kernel launch overhead)": "graphs",
            "Larger Batch Size": "batch",
        },
        value="FlashAttention (reduces HBM traffic)",
        label="Select optimization",
    )
    mo.hstack([pE_workload, pE_optim], gap="1.5rem")
    return (pE_optim, pE_workload)


@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, go, mo, pE_optim, pE_workload):
    # Speedup matrix: workload x optimization
    # Source: @sec-performance-engineering optimization matching framework
    _speedups = {
        "decode": {
            "flash": 1.02,    # already memory-bound, attention is not the bottleneck at batch=1
            "int4": 3.2,      # 4x BW -> significant gain for memory-bound
            "fusion": 1.15,   # modest gain from fewer kernel launches
            "graphs": 1.08,   # minor gain from reduced overhead
            "batch": 2.5,     # larger batch increases AI
        },
        "prefill": {
            "flash": 1.8,     # helps with long-sequence prefill (memory savings)
            "int4": 1.05,     # already compute-bound, BW not the bottleneck
            "fusion": 1.1,    # minor
            "graphs": 1.03,   # negligible
            "batch": 1.02,    # already large batch
        },
        "vision": {
            "flash": 1.0,     # no attention in vision CNN
            "int4": 1.8,      # near-ridge, bandwidth helps some
            "fusion": 1.25,   # useful for residual blocks
            "graphs": 1.1,    # moderate
            "batch": 1.15,    # near saturation
        },
    }

    _wl = pE_workload.value
    _opt = pE_optim.value
    _speedup = _speedups[_wl][_opt]

    # Diagnosis info
    _wl_info = {
        "decode": ("Memory-Bound", 1, "Deeply memory-bound (AI=1, ridge=295). Fix: reduce data movement."),
        "prefill": ("Compute-Bound", 512, "Compute-bound (AI=512 > ridge=295). Fix: reduce compute."),
        "vision": ("Near Ridge", 180, "Near the ridge point (AI=180, ridge=295). Mixed constraint."),
    }
    _regime, _ai, _diagnosis = _wl_info[_wl]

    # Color-coded speedup
    _sp_color = (COLORS["GreenLine"] if _speedup > 1.5 else
                 COLORS["OrangeLine"] if _speedup > 1.1 else COLORS["RedLine"])

    # Build bar chart of all optimizations for this workload
    _opt_names = ["FlashAttention", "INT4 Quant", "Fusion", "CUDA Graphs", "Larger Batch"]
    _opt_keys = ["flash", "int4", "fusion", "graphs", "batch"]
    _all_speedups = [_speedups[_wl][k] for k in _opt_keys]
    _bar_colors = [COLORS["GreenLine"] if s > 1.5 else COLORS["OrangeLine"] if s > 1.1 else COLORS["RedLine"]
                   for s in _all_speedups]

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        x=_opt_names, y=_all_speedups,
        marker_color=_bar_colors,
        text=[f"{s:.2f}x" for s in _all_speedups],
        textposition="outside",
    ))
    _fig.add_hline(y=1.0, line_dash="dash", line_color="#94a3b8",
                   annotation_text="No improvement", annotation_position="bottom right")
    _fig.update_layout(
        height=340,
        yaxis=dict(title="Speedup", range=[0, max(_all_speedups) + 0.5]),
        title=dict(text=f"Optimization Impact on: {_regime} Workload (AI={_ai})", font_size=13),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    apply_plotly_theme(_fig)

    _match_text = (
        f"**Your choice: {_opt}** on a **{_regime}** workload. "
        f"Speedup: **{_speedup:.2f}x**."
    )

    if _speedup < 1.1:
        _match_callout = mo.callout(mo.md(
            f"{_match_text} This optimization attacks the wrong bottleneck. "
            f"Diagnosis: {_diagnosis}"
        ), kind="danger")
    elif _speedup < 1.5:
        _match_callout = mo.callout(mo.md(
            f"{_match_text} Marginal improvement. There is a better option for this workload. "
            f"Diagnosis: {_diagnosis}"
        ), kind="warn")
    else:
        _match_callout = mo.callout(mo.md(
            f"{_match_text} Good match. This optimization addresses the binding constraint. "
            f"Diagnosis: {_diagnosis}"
        ), kind="success")

    mo.vstack([
        mo.as_html(_fig),
        _match_callout,
    ])
    return


# ─── CELL 29: PART E REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pE_pred):
    if pE_pred.value == "C":
        _pE_msg = (
            "**Correct.** The right optimization depends entirely on the bottleneck. "
            "INT4 quantization gives 3.2x speedup on memory-bound decode but only 1.05x on "
            "compute-bound prefill. FlashAttention helps prefill (1.8x) but barely helps "
            "decode at batch=1 (1.02x). The meta-skill is: profile on the roofline, identify "
            "the bottleneck, then select the matching optimization."
        )
        _pE_kind = "success"
    else:
        _pE_msg = (
            "**There is no universally best optimization.** The right choice depends on whether "
            "the workload is compute-bound or memory-bound. Applying FlashAttention to a "
            "compute-bound workload, or INT4 quantization to a compute-bound workload, yields "
            "less than 5% improvement. Always profile first: diagnose the bottleneck, then treat."
        )
        _pE_kind = "warn"

    mo.vstack([
        mo.callout(mo.md(_pE_msg), kind=_pE_kind),
        mo.accordion({
            "Optimization Matching Framework": mo.md("""
| Bottleneck | Best Optimization | Why |
|---|---|---|
| Memory-bound (AI << ridge) | INT4 quantization, larger batch | Increases effective bandwidth or AI |
| Compute-bound (AI >> ridge) | FlashAttention (for attention), better kernels | Reduces compute or HBM traffic |
| Overhead-dominated | CUDA Graphs, operator fusion | Eliminates kernel launch overhead |
| Near ridge | Profile carefully -- small changes shift regime | Mixed strategies needed |

**The diagnostic sequence:**
1. Profile on roofline -- place workload dot
2. Identify bottleneck (compute / memory / overhead)
3. Select matching optimization from the table above
4. Reprofile after optimization -- the regime may have shifted

*Source: @sec-performance-engineering, optimization methodology*
"""),
        }),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE G: SYNTHESIS + LEDGER
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 30: SYNTHESIS ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. The roofline is a diagnostic tool, not a decoration.</strong>
                    LLM decode at batch=1 achieves 0.34% of H100 compute capacity because
                    AI=1 is 295x below the ridge point. Faster GPUs do not help memory-bound workloads.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Not all fusion is equal.</strong>
                    FlashAttention saves 60x more HBM traffic than elementwise fusion because
                    the attention score matrix grows as O(N^2). The savings ratio widens to 256x
                    at 32K tokens and 1024x at 128K tokens.
                </div>
                <div>
                    <strong>3. Profile first, optimize second.</strong>
                    Applying the wrong optimization yields &lt;5% improvement. The meta-skill is:
                    diagnose the bottleneck (roofline placement), then select the matching
                    optimization. Reprofile after each change -- the regime may shift.
                </div>
            </div>
        </div>
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab V2-10: The Edge Thermodynamics Lab</strong> &mdash; You optimized
                    inference on H100s. Now discover what happens when you move training to a
                    smartphone: memory amplifies 4-12x, and the battery becomes the binding constraint.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-performance-engineering for the full roofline derivation
                    and FlashAttention tiling analysis.<br/>
                    <strong>Build:</strong> TinyTorch attention module &mdash; implement tiled attention
                    and measure HBM traffic reduction.
                </div>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 31: LEDGER ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, ledger, mo):
    ledger.save(chapter=9, design={
        "roofline_diagnostic": "memory-bound",
        "flash_savings_ratio_32k": 256,
        "optimization_methodology": "profile-diagnose-treat",
    })

    mo.Html(f"""
    <div style="background: #0f172a; border-radius: 10px; padding: 18px 24px;
                margin-top: 32px; font-family: 'SF Mono', 'Fira Code', monospace;">
        <div style="color: #475569; font-size: 0.7rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 10px;">
            Design Ledger &middot; Lab V2-09 Saved
        </div>
        <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.8;">
            <span style="color: #64748b;">roofline_diagnostic:</span>
            <span style="color: {COLORS['RedLine']};">memory-bound</span><br/>
            <span style="color: #64748b;">flash_savings_32k:</span>
            <span style="color: {COLORS['GreenLine']};">256x</span><br/>
            <span style="color: #64748b;">methodology:</span>
            <span style="color: {COLORS['BlueLine']};">profile &rarr; diagnose &rarr; treat</span>
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
