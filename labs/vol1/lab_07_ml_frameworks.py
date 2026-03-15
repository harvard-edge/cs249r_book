import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 07: THE KERNEL FUSION DIVIDEND
#
# Chapter: ML Frameworks (@sec-ml-frameworks)
# Core Invariant: Every unfused kernel launch pays a dispatch tax (5–20 μs).
#   Kernel fusion reduces memory round-trips by keeping data on-chip. A fused
#   LayerNorm + Dropout + ReLU sequence can yield 5× wall-clock speedup vs
#   3 separate kernels. torch.compile provides 1.3–2× throughput gain by
#   reducing kernel launch overhead.
#
# Act 1 (12 min): The Dispatch Tax Audit
#   Wrong prior: students believe larger models benefit most from compilation.
#   Reality: KWS (1,000 small kernels) has 33% GPU utilization at 5 μs compute
#   + 10 μs dispatch per kernel. Compilation raises utilization to ~67%.
#   GPT-2 (20 large kernels) already has 90% utilization — dispatch is negligible.
#
# Act 2 (22 min): The Compilation Break-Even
#   Wrong prior: "compile once, run fast forever" is always net positive.
#   Reality: 30-second compile time on ResNet-50 requires ~134,000 images to
#   break even. KWS on a Cloud server recovers quickly; on edge, the deployment
#   may expire before break-even is reached.
#
# 2 Contexts: Cloud (A100, 312 TFLOPS FP16, 2.0 TB/s HBM2e, 10M req/day)
#             Edge  (Jetson Orin NX, 100 TOPS INT8, 102 GB/s, 100 req/hr)
#
# Design Ledger: saves execution_mode, fusion_enabled, compilation_roi_positive,
#   breakeven_inferences, kws_utilization_eager_pct, kws_utilization_compiled_pct
# Downstream: Lab 08 (Training MFU), Lab 11 (Roofline arithmetic intensity)
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP (hide_code=False — leave visible) ─────────────────────────
@app.cell
async def _():
    import marimo as mo
    import sys
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np
    import math

    # WASM bootstrap: install mlsysim from hosted wheel when running in browser
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

    # ── HARDWARE CONSTANTS (traceability: frameworks.qmd) ───────────────────
    # A100 FP16 dense TFLOPS — frameworks.qmd line ~101 (A100BLAS.dense_tflops_str = 312)
    A100_TFLOPS_FP16 = 312          # TFLOPS dense FP16
    # A100 HBM2e bandwidth — frameworks.qmd line ~293 (MemoryWallSpecs: "2.0 TB/s")
    A100_BW_TBS = 2.0               # TB/s = 2,000 GB/s
    A100_BW_GBS = 2000              # GB/s
    # Roofline ridge point: 312 TFLOPS / 2.0 TB/s = 156 FLOP/byte
    A100_RIDGE_FLOP_PER_BYTE = 156  # frameworks.qmd implied (§ Memory Wall)
    # A100 RAM
    A100_RAM_GB = 80                # GB HBM2e

    # Jetson Orin NX specs (NVIDIA datasheet, representative edge device)
    ORIN_TOPS_INT8 = 100            # TOPS INT8
    ORIN_BW_GBS = 102               # GB/s
    ORIN_RAM_GB = 16                # GB
    ORIN_TDP_W = 25                 # Watts

    # Dispatch tax constants — frameworks.qmd line ~307
    # "Each kernel launch incurs 5–20 μs of CPU-side overhead"
    DISPATCH_US_CLOUD = 10          # μs per kernel launch on cloud (mid of 5–20 range)
    DISPATCH_US_EDGE = 50           # μs per kernel on edge (higher OS overhead)

    # torch.compile ResNet-50 data — frameworks.qmd line ~1283
    # "torch.compile provides ~48% speedup on ResNet-50 (2,150 vs 1,450 img/sec)"
    RESNET50_THROUGHPUT_EAGER = 1450    # img/sec
    RESNET50_THROUGHPUT_COMPILED = 2150  # img/sec
    RESNET50_COMPILE_TIME_S = 30        # seconds

    # Break-even formula (frameworks.qmd, derivable from the above)
    # N_breakeven = t_compile / (1/R_eager - 1/R_compiled)
    _delta_t_per_img = (1 / RESNET50_THROUGHPUT_EAGER) - (1 / RESNET50_THROUGHPUT_COMPILED)
    RESNET50_BREAKEVEN_IMAGES = int(RESNET50_COMPILE_TIME_S / _delta_t_per_img)  # ~134,000

    # GPU utilization range from framework choice — frameworks.qmd line ~2663
    # "whether a training loop achieves 30% or 80% of theoretical hardware throughput"
    UTILIZATION_LOW_PCT = 30        # % (eager, small-kernel model)
    UTILIZATION_HIGH_PCT = 80       # % (compiled, optimized)

    return (
        mo, go, np, math,
        ledger, COLORS, LAB_CSS, apply_plotly_theme,
        A100_TFLOPS_FP16, A100_BW_TBS, A100_BW_GBS, A100_RIDGE_FLOP_PER_BYTE, A100_RAM_GB,
        ORIN_TOPS_INT8, ORIN_BW_GBS, ORIN_RAM_GB, ORIN_TDP_W,
        DISPATCH_US_CLOUD, DISPATCH_US_EDGE,
        RESNET50_THROUGHPUT_EAGER, RESNET50_THROUGHPUT_COMPILED,
        RESNET50_COMPILE_TIME_S, RESNET50_BREAKEVEN_IMAGES,
        UTILIZATION_LOW_PCT, UTILIZATION_HIGH_PCT,
    )


# ─── CELL 1: HEADER ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2027 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume I · Lab 07
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.2rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.15; letter-spacing: -0.02em;">
                The Kernel Fusion Dividend
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.0rem; color: #94a3b8;
                      max-width: 660px; line-height: 1.65;">
                Every unfused kernel launch pays a dispatch tax. At 10 μs per launch and
                1,000 kernels per forward pass, the GPU is busy for only 33% of wall time.
                Kernel fusion and compilation recover the rest — but the payoff depends entirely
                on your kernel count and deployment volume.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    2 Acts · 35–40 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    Chapter 7: ML Frameworks
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Read @sec-ml-frameworks first
                </span>
            </div>
            <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 12px 18px; min-width: 160px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: #64748b;
                                text-transform: uppercase; letter-spacing: 0.1em;">Cloud (A100)</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: {COLORS['Cloud']}; margin-top: 4px;">
                        312 TFLOPS · 2.0 TB/s
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 12px 18px; min-width: 160px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: #64748b;
                                text-transform: uppercase; letter-spacing: 0.1em;">Edge (Jetson Orin NX)</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: {COLORS['Edge']}; margin-top: 4px;">
                        100 TOPS · 102 GB/s
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 12px 18px; min-width: 160px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: #64748b;
                                text-transform: uppercase; letter-spacing: 0.1em;">Invariant</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #e2e8f0; margin-top: 4px;">
                        Dispatch Tax: 5–20 μs/kernel
                    </div>
                </div>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">

        <!-- LEARNING OBJECTIVES -->
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Quantify GPU utilization loss from the dispatch tax</strong> — compute the exact fraction of wall-clock time a GPU performs arithmetic when 1,000 small kernels each carry a 10 &mu;s launch overhead.</div>
                <div style="margin-bottom: 3px;">2. <strong>Predict which workloads benefit most from kernel fusion</strong> — identify why KWS (1,000 small kernels) gains &gt;30% from compilation while GPT-2 (20 large kernels) gains near zero.</div>
                <div style="margin-bottom: 3px;">3. <strong>Calculate the compilation break-even threshold</strong> — derive the minimum inference volume at which a 30-second torch.compile cost is recovered for a given throughput gain.</div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- PREREQUISITES + DURATION (side by side) -->
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    Kernel launch overhead concept from @sec-ml-frameworks &middot;
                    Memory bandwidth vs. compute TFLOPS from @sec-ml-frameworks-execution-strategy-matters-memory-wall-1ce8
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35&ndash;40 min</strong><br/>
                    Act I: ~12 min &middot; Act II: ~25 min
                </div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- CORE QUESTION -->
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                "If a model launches 1,000 kernels per forward pass and each kernel computes
                for only 5 &mu;s, why does buying a faster GPU make the utilization problem
                worse &mdash; and when does compiling the model actually help?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: READING ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following sections of @sec-ml-frameworks before this lab:

    - **The Memory Wall** (@sec-ml-frameworks-execution-strategy-matters-memory-wall-1ce8) — The gap between 312 TFLOPS of compute and 2.0 TB/s of bandwidth is why element-wise ops like ReLU achieve less than 1% of peak compute.
    - **Kernel Fusion and the Dispatch Tax** — Framework compilation fuses adjacent operations, reducing kernel launches from N to 1 and eliminating intermediate HBM reads/writes.
    - **Compilation Continuum** — The spectrum from eager execution (debug-friendly, optimization-limited) to fully compiled graphs (latency to compile, 1.3–2× throughput gain).
    """), kind="info")
    return


# ─── CELL 4: CONTEXT TOGGLE ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    context_toggle = mo.ui.radio(
        options={
            "Cloud (A100 — 10M requests/day)": "cloud",
            "Edge (Jetson Orin NX — 100 requests/hour)": "edge",
        },
        value="Cloud (A100 — 10M requests/day)",
        label="Deployment context:",
        inline=True,
    )

    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
                    padding: 16px 20px; margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #64748b;
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                Deployment Context — Applied to Both Acts
            </div>
            <div style="font-size: 0.88rem; color: #475569; line-height: 1.55; margin-bottom: 12px;">
                The dispatch tax per kernel is
                <strong style="color:{COLORS['BlueLine']};">10 μs on Cloud (A100)</strong>
                and
                <strong style="color:{COLORS['Edge']};">50 μs on Edge (Jetson)</strong> —
                higher OS and driver overhead per launch relative to compute time.
                Switch contexts to see how the same model behaves under different overhead regimes.
            </div>
        </div>
        """),
        context_toggle,
    ])
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, context_toggle):
    _ctx = context_toggle.value
    _dispatch_us = 10 if _ctx == "cloud" else 50
    _ctx_label = "Cloud (A100)" if _ctx == "cloud" else "Edge (Jetson Orin NX)"
    _act_num = "I"
    _act_color = COLORS["BlueLine"]
    _act_title = "The Dispatch Tax Audit"
    _act_duration = "12 min"
    _act_why = (
        f"You expect that a model with fewer, larger operations is harder to optimize. "
        f"The instruments will show the opposite: it is the model with 1,000 tiny kernels "
        f"that is broken, because the fixed {_dispatch_us}\u00a0\u03bcs launch cost per kernel "
        f"dwarfs the 5\u00a0\u03bcs of actual computation \u2014 and a faster GPU makes it worse."
    )
    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {_act_color}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">{_act_num}</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Act {_act_num} &middot; {_act_duration} &middot; {_ctx_label}</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                {_act_title}
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                {_act_why}
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 6: ACT1_STAKEHOLDER ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: {COLORS['BlueL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['BlueLine']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; ML Infra Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "Our Keyword Spotting model's transformer inference is 3&times; slower than the
            paper reports. Same hardware, same model. The profiler shows 1,000 kernel
            launches per forward pass. Each kernel averages 5 &mu;s of GPU compute. What
            fraction of wall time is the GPU actually doing math?"
        </div>
    </div>
    """)
    return


# ─── ACT 1: PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) About 90% — the GPU is the bottleneck, not the CPU": "A",
            "B) About 50% — half compute, half overhead": "B",
            "C) About 33% — compute (5 μs) is one-third of total time (5 + 10 μs)": "C",
            "D) About 5% — overhead completely dominates": "D",
        },
        label="""**Commit your prediction before unlocking the instruments.**

A Keyword Spotting model performs 1,000 kernel launches per forward pass.
Each kernel computes for 5 μs on average. Each kernel *launch* costs 10 μs of
CPU-side dispatch overhead. What fraction of wall-clock time is the GPU
actually performing tensor operations?""",
    )
    mo.vstack([
        act1_pred,
        mo.callout(mo.md("Select your prediction to unlock the Act 1 instruments."), kind="warn")
        if act1_pred.value is None else mo.md(""),
    ])
    return (act1_pred,)


# ─── ACT 1: GATE ──────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(mo.md("Select your Act 1 prediction above to unlock the Dispatch Tax Waterfall."), kind="warn"),
    )
    return


# ─── ACT 1: INSTRUMENTS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_model_selector = mo.ui.dropdown(
        options={
            "KWS (Keyword Spotting) — 1,000 small kernels, 5 μs each": "kws",
            "ResNet-50 — 200 medium kernels, 50 μs each": "resnet50",
            "GPT-2 Layer — 20 large kernels, 500 μs each": "gpt2",
        },
        value="KWS (Keyword Spotting) — 1,000 small kernels, 5 μs each",
        label="Model type",
    )
    act1_mode_toggle = mo.ui.radio(
        options={
            "Eager (unfused)": "eager",
            "Compiled (fused)": "compiled",
        },
        value="Eager (unfused)",
        label="Execution mode",
        inline=True,
    )
    mo.hstack([
        mo.vstack([mo.md("**Model type**"), act1_model_selector]),
        mo.vstack([mo.md("**Execution mode**"), act1_mode_toggle]),
    ], justify="start", gap=3)
    return (act1_model_selector, act1_mode_toggle)


# ─── ACT 1: PHYSICS ENGINE + VISUALIZATION ────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, np,
    act1_model_selector, act1_mode_toggle, context_toggle,
    COLORS, apply_plotly_theme,
    DISPATCH_US_CLOUD, DISPATCH_US_EDGE,
):
    # ── Physics constants (traceability: frameworks.qmd) ─────────────────────
    # Dispatch tax per kernel — line ~307: "5–20 μs of CPU-side overhead"
    _ctx = context_toggle.value
    _dispatch_us = DISPATCH_US_CLOUD if _ctx == "cloud" else DISPATCH_US_EDGE

    # Model parameters: (kernel_count, compute_us_per_kernel, compiled_kernel_reduction_pct)
    # KWS: 1,000 small kernels — frameworks.qmd plan §3
    # ResNet-50: 200 medium kernels — realistic estimate for 50-layer residual network
    # GPT-2 Layer: 20 large kernels (matmul-dominant) — minimal dispatch
    _model_params = {
        "kws":      {"n": 1000, "compute_us": 5,   "compiled_reduction": 0.50},
        "resnet50": {"n": 200,  "compute_us": 50,  "compiled_reduction": 0.35},
        "gpt2":     {"n": 20,   "compute_us": 500, "compiled_reduction": 0.15},
    }

    _mode = act1_mode_toggle.value
    _model_key = act1_model_selector.value.split(" — ")[0].lower().replace("-", "").replace(" ", "")
    # Map display names to keys
    _key_map = {
        "kws": "kws", "keyword": "kws",
        "resnet50": "resnet50", "resnet": "resnet50",
        "gpt2": "gpt2", "gpt": "gpt2",
    }
    _mk = "kws"
    for _k in _key_map:
        if _k in _model_key:
            _mk = _key_map[_k]
            break

    _p = _model_params[_mk]

    # In compiled mode: kernel count reduces, dispatch overhead drops proportionally
    # Compute time is unchanged (same work, less fragmentation overhead)
    if _mode == "compiled":
        _n_kernels = int(_p["n"] * (1 - _p["compiled_reduction"]))
    else:
        _n_kernels = _p["n"]

    # Total times (μs)
    _total_compute_us = _p["n"] * _p["compute_us"]   # same total computation always
    _total_dispatch_us = _n_kernels * _dispatch_us
    # Memory transfer: estimated from arithmetic intensity
    # Element-wise ops (KWS): low AI (~0.1 FLOP/byte)
    # ResNet conv: medium AI (~10 FLOP/byte)
    # GPT-2 matmul: high AI (~100 FLOP/byte)
    _ai_map = {"kws": 0.1, "resnet50": 8.0, "gpt2": 80.0}
    _ai = _ai_map[_mk]
    # Memory transfer from arithmetic intensity: t_mem = compute_flops / (AI * BW)
    # Using A100 BW (2000 GB/s) as baseline; scale for edge
    _bw_gbs = 2000 if _ctx == "cloud" else 102
    _total_flops = _total_compute_us * 1e-6 * (312e12 if _ctx == "cloud" else 100e12) / 1e6
    # Simplified: memory time in μs
    _total_memory_us = max(50, int((_total_compute_us / _ai) * (2000 / _bw_gbs)))

    _total_us = _total_compute_us + _total_dispatch_us + _total_memory_us
    _gpu_utilization_pct = 100.0 * _total_compute_us / _total_us

    # ── Build stacked bar chart ───────────────────────────────────────────────
    _bar_color_compute = COLORS["BlueLine"]
    _bar_color_dispatch = COLORS["OrangeLine"]
    _bar_color_memory = COLORS["GreenLine"]

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        name="Kernel Compute",
        x=["Forward Pass Breakdown"],
        y=[_total_compute_us / 1000],  # convert to ms
        marker_color=_bar_color_compute,
        text=[f"Compute: {_total_compute_us/1000:.2f} ms"],
        textposition="inside",
        textfont=dict(color="white", size=11, family="SF Mono, monospace"),
    ))
    _fig.add_trace(go.Bar(
        name="Dispatch Overhead",
        x=["Forward Pass Breakdown"],
        y=[_total_dispatch_us / 1000],
        marker_color=_bar_color_dispatch,
        text=[f"Dispatch: {_total_dispatch_us/1000:.2f} ms ({_n_kernels} kernels × {_dispatch_us}μs)"],
        textposition="inside",
        textfont=dict(color="white", size=11, family="SF Mono, monospace"),
    ))
    _fig.add_trace(go.Bar(
        name="Memory Transfer",
        x=["Forward Pass Breakdown"],
        y=[_total_memory_us / 1000],
        marker_color=_bar_color_memory,
        text=[f"Memory: {_total_memory_us/1000:.2f} ms"],
        textposition="inside",
        textfont=dict(color="white", size=11, family="SF Mono, monospace"),
    ))
    _fig.update_layout(
        barmode="stack",
        height=320,
        yaxis=dict(title="Time (ms)", gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
    )

    # ── Model comparison table ────────────────────────────────────────────────
    _rows = ""
    for _model_name, _mp in [
        ("KWS (1,000 kernels, 5 μs each)", _model_params["kws"]),
        ("ResNet-50 (200 kernels, 50 μs each)", _model_params["resnet50"]),
        ("GPT-2 Layer (20 kernels, 500 μs each)", _model_params["gpt2"]),
    ]:
        _tc = _mp["n"] * _mp["compute_us"]
        _td_eager = _mp["n"] * _dispatch_us
        _util_eager = 100.0 * _tc / (_tc + _td_eager + max(50, int((_tc / _ai_map[_model_name[:3].lower().strip()]) * (2000 / _bw_gbs) if _model_name[:3].lower().strip() in {"kws", "res", "gpt"} else 1000)))
        _n_compiled = int(_mp["n"] * (1 - _mp["compiled_reduction"]))
        _td_compiled = _n_compiled * _dispatch_us
        _util_compiled = 100.0 * _tc / (_tc + _td_compiled + max(50, int((_tc / _ai_map[_model_name[:3].lower().strip()]) * (2000 / _bw_gbs) if _model_name[:3].lower().strip() in {"kws", "res", "gpt"} else 1000)))

        _eager_color = "#008F45" if _util_eager >= 70 else ("#CC5500" if _util_eager >= 40 else "#CB202D")
        _comp_color = "#008F45" if _util_compiled >= 70 else ("#CC5500" if _util_compiled >= 40 else "#CB202D")

        _rows += f"""
        <tr>
            <td style="padding: 8px 12px; font-size:0.85rem;">{_model_name}</td>
            <td style="padding: 8px 12px; font-family:monospace; font-size:0.85rem;">{_mp['n']}</td>
            <td style="padding: 8px 12px; font-family:monospace; font-size:0.85rem;">{_mp['compute_us']} μs</td>
            <td style="padding: 8px 12px; font-family:monospace; font-size:0.85rem;">{_dispatch_us} μs</td>
            <td style="padding: 8px 12px; font-family:monospace; font-weight:700; color:{_eager_color};">{_util_eager:.0f}%</td>
            <td style="padding: 8px 12px; font-family:monospace; font-weight:700; color:{_comp_color};">{_util_compiled:.0f}%</td>
        </tr>
        """

    _table_html = f"""
    <div style="overflow-x: auto; margin-top: 16px;">
        <table style="width:100%; border-collapse: collapse; font-size:0.88rem;">
            <thead>
                <tr style="background:#f8fafc; border-bottom: 2px solid #e2e8f0;">
                    <th style="padding:8px 12px; text-align:left; color:#475569;">Model</th>
                    <th style="padding:8px 12px; text-align:left; color:#475569;">Kernels</th>
                    <th style="padding:8px 12px; text-align:left; color:#475569;">Compute/Kernel</th>
                    <th style="padding:8px 12px; text-align:left; color:#475569;">Dispatch/Kernel</th>
                    <th style="padding:8px 12px; text-align:left; color:{COLORS['OrangeLine']};">Eager Util %</th>
                    <th style="padding:8px 12px; text-align:left; color:{COLORS['GreenLine']};">Compiled Util %</th>
                </tr>
            </thead>
            <tbody style="border-bottom: 1px solid #e2e8f0;">
                {_rows}
            </tbody>
        </table>
    </div>
    """

    # ── Physics formula display ───────────────────────────────────────────────
    _formula_block = f"""
    <div style="background: #0f172a; border-radius: 10px; padding: 16px 20px; margin: 12px 0;
                font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.83rem; color: #e2e8f0;">
        <div style="color: #94a3b8; font-size:0.72rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom: 8px;">Dispatch Tax Physics</div>
        GPU Utilization = N_kernels × t_compute / (N_kernels × (t_compute + t_dispatch) + t_memory)<br/>
        <br/>
        KWS eager:  {_total_compute_us:,} μs compute / ({_total_us:,} μs total) = <span style="color:#6ee7b7; font-weight:700;">{_gpu_utilization_pct:.1f}%</span><br/>
        t_dispatch = {_n_kernels} kernels × {_dispatch_us} μs = <span style="color:{COLORS['OrangeLine']};">{_total_dispatch_us:,} μs</span><br/>
        t_compute  = {_total_compute_us:,} μs (unchanged by compilation)
    </div>
    """

    return mo.vstack([
        mo.md(f"""### Dispatch Tax Waterfall — {act1_model_selector.value.split(' — ')[0]} · {act1_mode_toggle.value}"""),
        mo.ui.plotly(_fig),
        mo.Html(_formula_block),
        mo.md("**Model Comparison Table** — GPU utilization across model types and execution modes"),
        mo.Html(_table_html),
    ])


# ─── ACT 1: PREDICTION REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, COLORS):
    # Correct answer is C: 33%
    # KWS: 1,000 × 5 μs compute = 5,000 μs total compute
    # 1,000 × 10 μs dispatch = 10,000 μs total dispatch
    # Compute fraction = 5,000 / (5,000 + 10,000 + ~500 memory) ≈ 33%
    _correct_pct = 33.3
    _predicted_map = {"A": 90, "B": 50, "C": 33, "D": 5}
    _predicted_pct = _predicted_map.get(act1_pred.value, 0)
    _is_correct = act1_pred.value == "C"
    _gap_pct = abs(_correct_pct - _predicted_pct)

    if _is_correct:
        _reveal_kind = "success"
        _reveal_text = f"""**Your prediction was correct: ~33%.** The arithmetic: 1,000 kernels × 5 μs compute = 5,000 μs of actual GPU work; 1,000 kernels × 10 μs dispatch = 10,000 μs of overhead. GPU utilization = 5,000 / (5,000 + 10,000 + memory) ≈ **33%**. The GPU spends two-thirds of wall time waiting for kernel launches, not computing."""
    else:
        _reveal_text = f"""**You predicted {_predicted_pct}%. The actual value is ~33%. You were off by {_gap_pct:.0f} percentage points.** The arithmetic: 1,000 kernels × 5 μs compute = 5,000 μs of GPU work; 1,000 kernels × 10 μs dispatch = 10,000 μs of overhead. GPU utilization = 5,000 / (5,000 + 10,000 + memory) ≈ **33%**. {'You overestimated — dispatch overhead dominates when kernels are small.' if _predicted_pct > _correct_pct else 'Overhead is real but compute is not zero — the ratio is 33%, not near zero.'}"""
        _reveal_kind = "warn"

    return mo.callout(mo.md(_reveal_text + """

**The key asymmetry:** A 2× faster GPU would not fix 33% utilization. It would complete the 5 μs compute in 2.5 μs — and then wait 10 μs for the next launch. Faster hardware *amplifies* the dispatch tax."""), kind=_reveal_kind)


# ─── ACT 1: REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A) Faster GPUs have higher power draw, which causes thermal throttling": "A",
            "B) Faster GPUs require more time to warm up before peak throughput": "B",
            "C) Faster compute reduces the compute fraction of each kernel, making the fixed dispatch overhead a larger share of total time": "C",
            "D) Faster GPUs use different memory hierarchies that are incompatible with small models": "D",
        },
        label="""**Structured Reflection.**

A faster GPU sometimes produces *lower* utilization for small models because:""",
    )
    act1_reflection
    return (act1_reflection,)


@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(
        act1_reflection.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn"),
    )

    _correct = act1_reflection.value == "C"
    _feedback_map = {
        "A": "**Not quite.** Thermal throttling affects sustained throughput, but it is not the mechanism here. A faster GPU running the same dispatch-limited workload would finish the 5 μs compute faster — say in 2.5 μs — and then idle for 10 μs waiting for the next launch. Power draw is not the variable.",
        "B": "**Not quite.** Warm-up time affects the first few iterations (the 'cold start' problem), not steady-state utilization. Once the GPU is at operating temperature, the dispatch tax per kernel is independent of how long the GPU has been running.",
        "C": "**Correct.** This is the fundamental asymmetry: the dispatch tax (5–20 μs) is a CPU-side constant set by driver overhead. If the GPU computes faster, it finishes sooner but the dispatch timer does not compress. The ratio t_compute / (t_compute + t_dispatch) falls as t_compute shrinks — faster silicon, lower utilization for the same kernel structure.",
        "D": "**Not quite.** Memory hierarchy compatibility is not the mechanism. The same kernel code runs on all NVIDIA architectures. The dispatch overhead difference is in the OS scheduler and CUDA driver layer, not the memory subsystem.",
    }
    return mo.callout(mo.md(_feedback_map[act1_reflection.value]), kind="success" if _correct else "warn")


# ─── ACT 1: MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation": mo.md(r"""
        **GPU Utilization Formula:**

        $$\text{GPU Utilization} = \frac{N \cdot t_{compute}}{N \cdot (t_{compute} + t_{launch}) + t_{memory}}$$

        - **N** — number of kernel launches per forward pass
        - **t\_compute** — GPU compute time per kernel (μs) — set by the operation size
        - **t\_launch** — CPU-side dispatch overhead per kernel (5–20 μs on Cloud; 50 μs on Edge) — *constant*
        - **t\_memory** — HBM transfer time for the full forward pass (μs)

        **Numerical check (KWS eager, Cloud):**

        $$\text{Utilization} = \frac{1000 \times 5}{1000 \times (5 + 10) + 500} = \frac{5000}{15500} \approx 33\%$$

        **What compilation changes:** Compiled mode reduces N by 30–80% via operator fusion.
        Compute time is unchanged (same total FLOP count). Dispatch overhead drops in proportion to N reduction.

        $$\text{Utilization}_{compiled} = \frac{N \cdot t_{compute}}{(N \cdot r_{fusion}) \cdot (t_{compute} + t_{launch}) + t_{memory}}$$

        where $r_{fusion} \in [0.2, 0.7]$ is the fraction of kernels remaining after fusion.
        """)
    })
    return


# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, context_toggle):
    _ctx = context_toggle.value
    _ctx_label = "Cloud (A100)" if _ctx == "cloud" else "Edge (Jetson Orin NX)"
    _volume_default = "10M requests/day" if _ctx == "cloud" else "100 requests/hour"
    _act_num = "II"
    _act_color = COLORS["OrangeLine"]
    _act_title = "The Compilation Break-Even"
    _act_duration = "22 min"
    _act_why = (
        f"Act I showed that compilation raises KWS utilization from 33% to 67%. "
        f"Now discover when \u201ccompile once, run fast forever\u201d is actually net-positive: "
        f"a 30-second compile time on ResNet-50 requires ~134,000 inferences to break even, "
        f"and on a {_volume_default} deployment, that crossover tells you whether to compile at all."
    )
    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {_act_color}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">{_act_num}</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Act {_act_num} &middot; {_act_duration} &middot; {_ctx_label}</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                {_act_title}
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                {_act_why}
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 13: ACT2_STAKEHOLDER ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""**Design Challenge:** You have a production serving system. You must decide: run eagerly (flexible, no compile cost) or compile (latency to compile, permanent throughput gain). The break-even analysis determines whether compilation is net-positive for your deployment. A wrong decision costs either throughput or deployment time."""), kind="info")
    return


# ─── ACT 2: PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(
        act1_reflection.value is None,
        mo.callout(mo.md("Complete the Act 1 reflection above to unlock Act 2."), kind="warn"),
    )

    act2_pred = mo.ui.radio(
        options={
            "A) About 1,000 images — the overhead is tiny": "A",
            "B) About 10,000 images — roughly 10 seconds of inference at baseline throughput": "B",
            "C) About 130,000 images — the time saved per image is small, so many images are needed": "C",
            "D) About 10 million images — compilation is almost never worth it": "D",
        },
        label="""**Commit your prediction before unlocking the Act 2 instruments.**

torch.compile on ResNet-50 improves throughput by 48% (from 1,450 to 2,150 images/sec)
but requires 30 seconds of one-time compilation.

Approximately how many images must you process before the compilation time cost is recovered?""",
    )
    mo.vstack([
        act2_pred,
        mo.callout(mo.md("Select your prediction to unlock the compilation instruments."), kind="warn")
        if act2_pred.value is None else mo.md(""),
    ])
    return (act2_pred,)


# ─── ACT 2: GATE ──────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(mo.md("Select your Act 2 prediction above to unlock the compilation instruments."), kind="warn"),
    )
    return


# ─── ACT 2: PANEL A — KERNEL FUSION EXPLORER ──────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Panel A: Kernel Fusion Explorer

**Operation sequence:** LayerNorm → Dropout → ReLU

Without fusion, these three element-wise operations each read their input from HBM, compute, and write their output back to HBM.
With fusion, the input is read once, all three operations execute in registers/L1, and only the final output is written.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_fusion_toggle = mo.ui.radio(
        options={
            "Unfused (3 separate kernel launches)": "unfused",
            "Fused (1 kernel launch)": "fused",
        },
        value="Unfused (3 separate kernel launches)",
        label="Fusion state",
        inline=True,
    )
    act2_hbm_bw_selector = mo.ui.dropdown(
        options={
            "1.0 TB/s (older A100 equivalent)": 1000,
            "2.0 TB/s (A100 HBM2e)": 2000,
            "3.35 TB/s (H100 HBM3)": 3350,
        },
        value="2.0 TB/s (A100 HBM2e)",
        label="HBM bandwidth",
    )
    mo.hstack([
        mo.vstack([mo.md("**Fusion state**"), act2_fusion_toggle]),
        mo.vstack([mo.md("**HBM bandwidth**"), act2_hbm_bw_selector]),
    ], justify="start", gap=3)
    return (act2_fusion_toggle, act2_hbm_bw_selector)


@app.cell(hide_code=True)
def _(
    mo, go,
    act2_fusion_toggle, act2_hbm_bw_selector, context_toggle,
    COLORS, apply_plotly_theme,
    DISPATCH_US_CLOUD, DISPATCH_US_EDGE,
):
    _ctx = context_toggle.value
    _dispatch_us = DISPATCH_US_CLOUD if _ctx == "cloud" else DISPATCH_US_EDGE
    _bw_gbs = act2_hbm_bw_selector.value
    _is_fused = act2_fusion_toggle.value == "fused"

    # ── Fusion physics (traceability: frameworks.qmd §4, lab plan §4 Panel A) ─
    # LayerNorm + Dropout + ReLU fusion: 3 element-wise ops
    # Unfused: 3 reads + 3 writes = 6 HBM ops per tensor element
    # Fused:   1 read  + 1 write  = 2 HBM ops per tensor element
    # HBM traffic ratio: 6/2 = 3× reduction — frameworks.qmd plan (derived, not FlashAttention)
    # FlashAttention provides 10–20× which is tiling of attention matrix — different mechanism

    # Representative tensor size: 1024 tokens × 768 hidden = 786,432 elements × 2 bytes (FP16)
    _tensor_bytes = 1024 * 768 * 2  # bytes per tensor transfer
    _n_ops_unfused = 3  # LayerNorm, Dropout, ReLU

    # HBM traffic
    _hbm_ops_unfused = 2 * _n_ops_unfused  # 3 reads + 3 writes
    _hbm_ops_fused = 2                      # 1 read + 1 write
    _hbm_traffic_unfused_mb = (_hbm_ops_unfused * _tensor_bytes) / 1e6
    _hbm_traffic_fused_mb = (_hbm_ops_fused * _tensor_bytes) / 1e6

    # Transfer time in μs
    _bw_bytes_per_us = _bw_gbs * 1000  # GB/s → MB/μs → bytes/μs = _bw_gbs * 1e9 / 1e6
    _t_mem_unfused_us = (_hbm_traffic_unfused_mb * 1e6) / (_bw_gbs * 1e9 / 1e6)
    _t_mem_fused_us = (_hbm_traffic_fused_mb * 1e6) / (_bw_gbs * 1e9 / 1e6)

    # Kernel launch overhead
    _t_dispatch_unfused_us = _n_ops_unfused * _dispatch_us
    _t_dispatch_fused_us = 1 * _dispatch_us

    # Compute time (same total work, 3 element-wise ops — fixed)
    _t_compute_us = 5  # μs for all 3 ops combined (small element-wise)

    # Total times
    _total_unfused_us = _t_compute_us + _t_dispatch_unfused_us + _t_mem_unfused_us
    _total_fused_us = _t_compute_us + _t_dispatch_fused_us + _t_mem_fused_us

    _speedup = _total_unfused_us / _total_fused_us if _total_fused_us > 0 else 1.0
    _traffic_reduction = _hbm_ops_unfused / _hbm_ops_fused  # = 3×

    # Arithmetic intensity
    _compute_flops = 1024 * 768 * 10  # ~10 FLOP per element for all 3 ops combined
    _ai_unfused = _compute_flops / (_hbm_traffic_unfused_mb * 1e6)
    _ai_fused = _compute_flops / (_hbm_traffic_fused_mb * 1e6)

    _active_state = "fused" if _is_fused else "unfused"
    _active_total = _total_fused_us if _is_fused else _total_unfused_us
    _active_mem = _t_mem_fused_us if _is_fused else _t_mem_unfused_us
    _active_dispatch = _t_dispatch_fused_us if _is_fused else _t_dispatch_unfused_us
    _active_traffic = _hbm_traffic_fused_mb if _is_fused else _hbm_traffic_unfused_mb
    _active_ai = _ai_fused if _is_fused else _ai_unfused

    # ── Bar chart: memory traffic comparison ────────────────────────────────
    _fig_fusion = go.Figure()
    _active_color_un = COLORS["OrangeLine"] if not _is_fused else COLORS["Grey"]
    _active_color_f = COLORS["GreenLine"] if _is_fused else COLORS["Grey"]

    _fig_fusion.add_trace(go.Bar(
        name="Unfused (3 kernels)",
        x=["HBM Reads", "HBM Writes", "Dispatch Overhead"],
        y=[
            _hbm_traffic_unfused_mb / 2,  # reads only
            _hbm_traffic_unfused_mb / 2,  # writes only
            _t_dispatch_unfused_us / 100,  # scale for visibility
        ],
        marker_color=COLORS["OrangeLine"],
        opacity=0.5 if _is_fused else 1.0,
    ))
    _fig_fusion.add_trace(go.Bar(
        name="Fused (1 kernel)",
        x=["HBM Reads", "HBM Writes", "Dispatch Overhead"],
        y=[
            _hbm_traffic_fused_mb / 2,
            _hbm_traffic_fused_mb / 2,
            _t_dispatch_fused_us / 100,
        ],
        marker_color=COLORS["GreenLine"],
        opacity=0.5 if not _is_fused else 1.0,
    ))
    _fig_fusion.update_layout(
        barmode="group",
        height=280,
        yaxis=dict(title="MB (reads/writes) / scaled overhead", gridcolor="#f1f5f9"),
        xaxis=dict(gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
    )

    # ── Results panel ────────────────────────────────────────────────────────
    _result_color = COLORS["GreenLine"] if _speedup >= 3.0 else (COLORS["OrangeLine"] if _speedup >= 1.5 else COLORS["RedLine"])
    _ai_color = COLORS["OrangeLine"]  # still memory-bound below ridge point (156 FLOP/byte)

    _panel_a_results = f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0;">
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px;
                    padding: 14px 16px; text-align: center;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">HBM Traffic</div>
            <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['BlueLine']}; font-family: monospace;">
                {_active_traffic:.2f} MB
            </div>
            <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">
                {"1 read + 1 write" if _is_fused else "3 reads + 3 writes"}
            </div>
        </div>
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px;
                    padding: 14px 16px; text-align: center;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">Traffic Reduction</div>
            <div style="font-size: 1.6rem; font-weight: 900; color: {_result_color}; font-family: monospace;">
                {_traffic_reduction:.0f}×
            </div>
            <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">vs unfused baseline</div>
        </div>
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px;
                    padding: 14px 16px; text-align: center;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">Wall-Clock Speedup</div>
            <div style="font-size: 1.6rem; font-weight: 900; color: {_result_color}; font-family: monospace;">
                {_speedup:.1f}×
            </div>
            <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">fused vs unfused</div>
        </div>
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px;
                    padding: 14px 16px; text-align: center;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">Arithmetic Intensity</div>
            <div style="font-size: 1.6rem; font-weight: 900; color: {_ai_color}; font-family: monospace;">
                {_active_ai:.2f} FLOP/B
            </div>
            <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">
                {"Still memory-bound (ridge = 156 FLOP/B)" if _active_ai < 156 else "Compute-bound"}
            </div>
        </div>
    </div>
    """

    _fusion_formula = f"""
    <div style="background: #0f172a; border-radius: 10px; padding: 14px 18px; margin: 8px 0;
                font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.82rem; color: #e2e8f0;">
        <div style="color: #94a3b8; font-size:0.72rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom: 6px;">Fusion Physics (LayerNorm + Dropout + ReLU)</div>
        Unfused: {_n_ops_unfused} reads + {_n_ops_unfused} writes = {_hbm_ops_unfused} HBM ops → {_hbm_traffic_unfused_mb:.2f} MB<br/>
        Fused:   1 read  + 1 write  = {_hbm_ops_fused} HBM ops → {_hbm_traffic_fused_mb:.2f} MB<br/>
        Traffic reduction: {_hbm_ops_unfused}/{_hbm_ops_fused} = <span style="color:#6ee7b7; font-weight:700;">{_traffic_reduction:.0f}× less HBM traffic</span><br/>
        Wall-clock speedup: {_total_unfused_us:.1f} μs → {_total_fused_us:.1f} μs = <span style="color:#6ee7b7; font-weight:700;">{_speedup:.1f}× faster</span><br/>
        Arithmetic intensity: {_ai_unfused:.3f} → {_ai_fused:.3f} FLOP/byte (still below ridge point of 156 FLOP/B)
    </div>
    """

    return mo.vstack([
        mo.ui.plotly(_fig_fusion),
        mo.Html(_panel_a_results),
        mo.Html(_fusion_formula),
        mo.callout(mo.md(f"""**Key distinction:** The 3× HBM traffic reduction applies to *element-wise fusion* (LayerNorm + Dropout + ReLU — 3 ops → 1 op). The 10–20× figure cited in @sec-ml-frameworks applies to **FlashAttention** specifically, which tiles the attention matrix to avoid materializing the full N×N attention scores in HBM. These are different mechanisms. Element-wise fusion is proportional to the number of ops fused; FlashAttention's gain comes from tiling mathematics."""), kind="info"),
    ])


# ─── ACT 2: PANEL B — COMPILATION ROI CALCULATOR ──────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Panel B: Compilation ROI Calculator

Compilation has a one-time cost and a permanent throughput gain. Whether it is net-positive depends on how many inferences you run before the deployment ends.
""")
    return


@app.cell(hide_code=True)
def _(mo, context_toggle):
    _ctx = context_toggle.value

    # Default values reflect the two deployment contexts
    # Cloud: high volume, compilation always pays off quickly
    # Edge: low volume (100 req/hr), short deployments
    _default_compile_s = 30      # ResNet-50 default
    _default_gain_pct = 48       # ResNet-50 default from chapter
    _default_volume = 10000000 if _ctx == "cloud" else 100
    _default_duration_days = 30 if _ctx == "cloud" else 1

    act2_compile_time_slider = mo.ui.slider(
        start=5, stop=300, step=5, value=_default_compile_s,
        label="Compilation time (seconds)",
        full_width=True,
    )
    act2_gain_pct_slider = mo.ui.slider(
        start=5, stop=100, step=5, value=_default_gain_pct,
        label="Throughput gain from compilation (%)",
        full_width=True,
    )
    act2_volume_slider = mo.ui.slider(
        start=100, stop=10000000, step=100, value=_default_volume,
        label="Inferences per day",
        full_width=True,
    )
    act2_duration_slider = mo.ui.slider(
        start=1, stop=365, step=1, value=_default_duration_days,
        label="Deployment duration (days)",
        full_width=True,
    )

    mo.vstack([
        mo.hstack([act2_compile_time_slider, act2_gain_pct_slider], justify="start", gap=2),
        mo.hstack([act2_volume_slider, act2_duration_slider], justify="start", gap=2),
    ])
    return (
        act2_compile_time_slider,
        act2_gain_pct_slider,
        act2_volume_slider,
        act2_duration_slider,
    )


# ─── ACT 2: BREAK-EVEN CHART + FAILURE STATE ─────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, np,
    act2_compile_time_slider, act2_gain_pct_slider,
    act2_volume_slider, act2_duration_slider,
    context_toggle,
    COLORS, apply_plotly_theme,
    RESNET50_THROUGHPUT_EAGER, RESNET50_THROUGHPUT_COMPILED,
):
    _ctx = context_toggle.value
    _compile_s = act2_compile_time_slider.value
    _gain_pct = act2_gain_pct_slider.value / 100.0
    _volume_per_day = act2_volume_slider.value
    _duration_days = act2_duration_slider.value

    # ── Break-even calculation ────────────────────────────────────────────────
    # frameworks.qmd line ~1283: ResNet-50 baseline 1,450 img/sec; compiled 2,150 img/sec
    # User-configurable gain applied to baseline
    _throughput_eager = RESNET50_THROUGHPUT_EAGER
    _throughput_compiled = _throughput_eager * (1 + _gain_pct)

    # Time saved per image (seconds)
    _t_per_img_eager = 1.0 / _throughput_eager
    _t_per_img_compiled = 1.0 / _throughput_compiled
    _delta_t_per_img = _t_per_img_eager - _t_per_img_compiled  # seconds saved per image

    # Break-even: how many images until compile_time is recovered
    # N_breakeven = t_compile / delta_t_per_image
    _breakeven_images = int(_compile_s / _delta_t_per_img) if _delta_t_per_img > 0 else float('inf')
    _breakeven_days = _breakeven_images / _volume_per_day if _volume_per_day > 0 else float('inf')

    _total_images = _volume_per_day * _duration_days
    _roi_positive = _total_images >= _breakeven_images

    # ── Build break-even timeline ────────────────────────────────────────────
    # X-axis: days into deployment
    # Y-axis: cumulative time saved (seconds) minus compile overhead
    _days = np.linspace(0, _duration_days, 500)
    _images_elapsed = _days * _volume_per_day
    _cumulative_saved_s = _images_elapsed * _delta_t_per_img - _compile_s
    _compile_overhead_line = np.zeros_like(_days)

    _fig_roi = go.Figure()

    # Compilation overhead baseline
    _fig_roi.add_trace(go.Scatter(
        x=_days, y=_compile_overhead_line,
        name="Break-even line",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        showlegend=True,
    ))

    # Cumulative time saved
    _line_color = COLORS["GreenLine"] if _roi_positive else COLORS["OrangeLine"]
    _fig_roi.add_trace(go.Scatter(
        x=_days, y=_cumulative_saved_s,
        name="Cumulative time saved",
        fill="tozeroy",
        fillcolor=f"rgba(0, 143, 69, 0.12)" if _roi_positive else f"rgba(204, 85, 0, 0.12)",
        line=dict(color=_line_color, width=2.5),
    ))

    # Mark break-even point if within deployment
    if _breakeven_days <= _duration_days and _breakeven_days > 0:
        _fig_roi.add_vline(
            x=_breakeven_days,
            line_color=COLORS["BlueLine"],
            line_dash="dash",
            line_width=2,
            annotation_text=f"Break-even: day {_breakeven_days:.1f}",
            annotation_position="top right",
            annotation_font_color=COLORS["BlueLine"],
            annotation_font_size=11,
        )

    _fig_roi.update_layout(
        height=320,
        yaxis=dict(
            title="Cumulative time saved (seconds)",
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            zeroline=True,
            zerolinecolor="#e2e8f0",
            zerolinewidth=1,
        ),
        xaxis=dict(title="Days deployed", gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
    )

    # ── Metric cards ────────────────────────────────────────────────────────
    _be_display = f"{_breakeven_images:,.0f}" if _breakeven_images != float('inf') else "∞"
    _be_days_display = f"{_breakeven_days:.1f}" if _breakeven_days != float('inf') else "∞"
    _total_saved_s = max(0, _cumulative_saved_s[-1])

    _card_color_be = COLORS["GreenLine"] if _roi_positive else COLORS["OrangeLine"]
    _card_color_saved = COLORS["GreenLine"] if _total_saved_s > 0 else COLORS["RedLine"]

    _metrics_html = f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 12px 0;">
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px;
                    padding: 14px 16px; text-align: center;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">Break-Even Images</div>
            <div style="font-size: 1.5rem; font-weight: 900; color: {_card_color_be}; font-family: monospace;">{_be_display}</div>
            <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">images needed</div>
        </div>
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px;
                    padding: 14px 16px; text-align: center;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">Break-Even Day</div>
            <div style="font-size: 1.5rem; font-weight: 900; color: {_card_color_be}; font-family: monospace;">{_be_days_display}</div>
            <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">of {_duration_days}-day deployment</div>
        </div>
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px;
                    padding: 14px 16px; text-align: center;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">Total Images Served</div>
            <div style="font-size: 1.5rem; font-weight: 900; color: {COLORS['BlueLine']}; font-family: monospace;">{_total_images:,.0f}</div>
            <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">over deployment</div>
        </div>
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px;
                    padding: 14px 16px; text-align: center;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">Net Time Saved</div>
            <div style="font-size: 1.5rem; font-weight: 900; color: {_card_color_saved}; font-family: monospace;">{_total_saved_s:,.0f}s</div>
            <div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">after compile cost</div>
        </div>
    </div>
    """

    # ── FAILURE STATE ─────────────────────────────────────────────────────────
    # Trigger: break-even > deployment duration (compilation not justified)
    if not _roi_positive:
        _failure_banner = mo.callout(
            mo.md(
                f"**Compilation Not Justified.** "
                f"Break-even requires **{_be_display} images** (day {_be_days_display}), "
                f"but your deployment ends after **day {_duration_days}** "
                f"({_total_images:,.0f} total images). "
                f"Eager mode is faster overall for this deployment window. "
                f"To fix: increase deployment duration, inference volume, or throughput gain — "
                f"or use eager mode and accept the {_gain_pct*100:.0f}% throughput cost."
            ),
            kind="danger",
        )
    else:
        _net_days_positive = _duration_days - _breakeven_days
        _failure_banner = mo.callout(
            mo.md(
                f"**Compilation ROI Positive.** "
                f"Break-even at day {_be_days_display} leaves "
                f"**{_net_days_positive:.1f} days** of net-positive throughput. "
                f"Total time saved after compile cost: **{_total_saved_s:,.0f} seconds**."
            ),
            kind="success",
        )

    _formula_text = f"""
    <div style="background: #0f172a; border-radius: 10px; padding: 14px 18px; margin: 8px 0;
                font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.82rem; color: #e2e8f0;">
        <div style="color: #94a3b8; font-size:0.72rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom: 6px;">Break-Even Formula (frameworks.qmd §Compilation)</div>
        N_breakeven = t_compile / (1/R_eager - 1/R_compiled)<br/>
        = {_compile_s}s / (1/{_throughput_eager:.0f} - 1/{_throughput_compiled:.0f})<br/>
        = {_compile_s}s / {_delta_t_per_img*1000:.4f} ms/image<br/>
        = <span style="color:#6ee7b7; font-weight:700;">{_be_display} images</span> = day {_be_days_display} at {_volume_per_day:,} images/day
    </div>
    """

    return mo.vstack([
        mo.ui.plotly(_fig_roi),
        mo.Html(_metrics_html),
        _failure_banner,
        mo.Html(_formula_text),
    ])


# ─── ACT 2: PREDICTION REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred, RESNET50_BREAKEVEN_IMAGES):
    _correct_images = RESNET50_BREAKEVEN_IMAGES  # ~134,000
    _predicted_map = {"A": 1000, "B": 10000, "C": 130000, "D": 10000000}
    _predicted_images = _predicted_map.get(act2_pred.value, 0)
    _is_correct = act2_pred.value == "C"
    _ratio = _correct_images / _predicted_images if _predicted_images > 0 else float('inf')

    if _is_correct:
        _kind = "success"
        _text = f"**Your prediction was correct: ~130,000 images.** The calculation: Δt per image = 1/1,450 − 1/2,150 ≈ 0.224 ms. Break-even = 30 s / 0.000224 s/image ≈ **{_correct_images:,} images**. The gain per image is small because the base throughput is already fast — you need high volume to recover the fixed compile cost."
    elif _ratio > 5:
        _kind = "warn"
        _text = f"**You predicted {_predicted_images:,}. The actual value is ~{_correct_images:,}. You were off by {_ratio:.1f}×.** The gain per image is only 0.224 ms — much smaller than it feels. A 48% throughput gain sounds large, but when you are processing at 1,450 img/sec, each image takes just 0.69 ms. Saving 0.224 ms per image requires processing 134,000 images to recover 30 seconds of compile time."
    else:
        _kind = "warn"
        _text = f"**You predicted {_predicted_images:,}. The actual value is ~{_correct_images:,}. You were off by {_ratio:.1f}×.** {'Compilation requires more volume than expected because the per-image gain is small relative to the compile cost.' if _predicted_images < _correct_images else 'Compilation is more cost-effective than you expected — the per-image gain accumulates quickly at high throughput.'}"

    return mo.callout(mo.md(_text), kind=_kind)


# ─── ACT 2: REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_reflection_a = mo.ui.radio(
        options={
            "A) ~5× wall-clock speedup only — no HBM traffic change": "A",
            "B) ~3× HBM traffic reduction and ~5× wall-clock speedup": "B",
            "C) 10–20× HBM traffic reduction (same as FlashAttention)": "C",
            "D) ~3× HBM traffic reduction only — no wall-clock change": "D",
        },
        label="""**Structured Reflection — Part 1.**

Kernel fusion of LayerNorm + Dropout + ReLU (3 element-wise ops → 1 fused kernel) provides:""",
    )
    act2_reflection_a
    return (act2_reflection_a,)


@app.cell(hide_code=True)
def _(mo, act2_reflection_a):
    mo.stop(
        act2_reflection_a.value is None,
        mo.callout(mo.md("Select your answer to continue."), kind="warn"),
    )

    _correct_a = act2_reflection_a.value == "B"
    _fb_a_map = {
        "A": "**Not quite.** The wall-clock speedup (approximately 5×) comes from *two* combined effects: eliminating 2 of 3 kernel launch overheads (dispatch reduction) AND eliminating intermediate HBM writes and reads (memory traffic reduction). You cannot get the full speedup without both. The 5× figure without the 3× HBM reduction is not accurate.",
        "B": "**Correct.** Fusing 3 element-wise ops into 1 eliminates 2 intermediate HBM read-write pairs: 6 HBM operations (3 reads + 3 writes) reduce to 2 (1 read + 1 write) = **3× HBM traffic reduction**. Combined with eliminating 2 of 3 kernel launch overheads, the total wall-clock speedup is approximately **5×** — as reported in @sec-ml-frameworks.",
        "C": "**Not quite.** The 10–20× figure applies to **FlashAttention**, which tiles the N×N attention matrix to avoid materializing it in HBM entirely. That is a different mechanism — tiling reduces quadratic memory cost to linear. Element-wise op fusion (LayerNorm, Dropout, ReLU) reduces traffic proportionally to the number of ops fused: 3 ops = 3× reduction. Different operations, different physics.",
        "D": "**Not quite.** Both effects occur simultaneously. Eliminating intermediate HBM traffic (3× reduction) speeds up the memory-bound portion. Eliminating 2 of 3 kernel launches reduces dispatch overhead. Together these produce the ~5× wall-clock speedup — you cannot separate them in practice.",
    }
    return mo.callout(mo.md(_fb_a_map[act2_reflection_a.value]), kind="success" if _correct_a else "warn")


@app.cell(hide_code=True)
def _(mo, act2_reflection_a):
    mo.stop(act2_reflection_a.value is None)

    act2_reflection_b = mo.ui.radio(
        options={
            "A) Near-zero speedup — dispatch overhead (10 μs) is negligible vs. 200 ms compute": "A",
            "B) About 48% speedup — the same as ResNet-50": "B",
            "C) Greater than 2× speedup — large kernels benefit most from optimization": "C",
            "D) Negative speedup — compilation makes large kernels slower": "D",
        },
        label="""**Structured Reflection — Part 2.**

For a model with one giant matrix multiply (200 ms compute per kernel, 1 kernel total),
torch.compile will provide:""",
    )
    act2_reflection_b
    return (act2_reflection_b,)


@app.cell(hide_code=True)
def _(mo, act2_reflection_b):
    mo.stop(
        act2_reflection_b.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn"),
    )

    _correct_b = act2_reflection_b.value == "A"
    _fb_b_map = {
        "A": "**Correct.** The dispatch tax for 1 kernel at 10 μs is 10 μs / (10 μs + 200,000 μs) = **0.005%** of total time. Compilation fuses zero ops (there is only one kernel) and eliminates essentially zero dispatch overhead. For large matmul-dominant workloads like GPT-2 forward passes, compilation provides marginal gains — the hardware is already efficient at that scale.",
        "B": "**Not quite.** The 48% figure applies to ResNet-50, which has approximately 200 medium-sized kernels where dispatch overhead is a meaningful fraction of total time. A single 200 ms kernel has a dispatch ratio of 10 μs / 200 ms = 0.005%. Compilation cannot exploit what is not fragmented.",
        "C": "**Not quite.** Large kernels benefit *least* from compilation because the dispatch tax is already negligible. The 5–20 μs overhead per launch is fixed by CPU driver latency — it does not scale with kernel size. When compute time is 200 ms, the dispatch component is invisible.",
        "D": "**Not quite.** Compilation does not make correct kernels slower (it may add negligible warm-up cost). The answer is near-zero speedup, not negative. Compilation at worst adds trace overhead; the 30-second compile time has no effect on per-call performance after the first run.",
    }
    return mo.callout(mo.md(_fb_b_map[act2_reflection_b.value]), kind="success" if _correct_b else "warn")


# ─── ACT 2: MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations": mo.md(r"""
        **Total Forward Pass Time:**

        $$T_{total} = N_{kernels} \cdot t_{dispatch} + t_{compute} + t_{memory}$$

        - **N\_kernels** — kernel launches per forward pass (reduced by compilation via fusion)
        - **t\_dispatch** — CPU-side overhead per launch: 5–20 μs (Cloud); 50 μs (Edge)
        - **t\_compute** — GPU arithmetic time (unchanged by compilation)
        - **t\_memory** — HBM transfer time (reduced by kernel fusion via fewer intermediate tensors)

        **Compilation Break-Even:**

        $$N_{break-even} = \frac{t_{compile}}{\Delta t_{per\text{-}inference}} = \frac{t_{compile}}{\frac{1}{R_{eager}} - \frac{1}{R_{compiled}}}$$

        **ResNet-50 numerical check (frameworks.qmd §Compilation):**

        $$N_{break-even} = \frac{30\text{ s}}{\frac{1}{1450} - \frac{1}{2150}} \approx 134{,}000 \text{ images}$$

        **Element-wise fusion HBM traffic reduction:**

        $$\frac{\text{HBM ops}_{unfused}}{\text{HBM ops}_{fused}} = \frac{2N_{ops}}{2} = N_{ops}\text{-fold reduction}$$

        For LayerNorm + Dropout + ReLU ($N_{ops} = 3$): **3× HBM traffic reduction**, **~5× wall-clock speedup** (includes dispatch elimination).

        *Note: FlashAttention's 10–20× reduction comes from attention matrix tiling ($O(N^2) \to O(N)$ HBM access) — a different mechanism from element-wise fusion.*
        """)
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
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
                    <strong>1. The dispatch tax determines GPU utilization, not model size.</strong>
                    A KWS model with 1,000 kernels at 5 &mu;s compute + 10 &mu;s dispatch achieves
                    only 33% utilization. A GPT-2 layer with 20 large kernels already achieves ~90%.
                    Faster hardware amplifies this gap rather than closing it.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Kernel fusion is the fix for dispatch-bound workloads.</strong>
                    Fusing LayerNorm + Dropout + ReLU into one kernel reduces HBM traffic 3&times;
                    and achieves ~5&times; wall-clock speedup. For GPT-2-sized ops the gain is
                    near zero &mdash; compilation only helps when fragmentation was the problem.
                </div>
                <div>
                    <strong>3. Compilation has a break-even threshold of ~134,000 inferences.</strong>
                    A 30-second torch.compile on ResNet-50 costs 0.224 ms per image in saved
                    time. That gap accumulates slowly: you need high inference volume before the
                    fixed compile cost is recovered. On edge deployments with episodic traffic,
                    eager mode may win overall.
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
                    <strong>Lab 08: The Training Memory Budget</strong> &mdash; This lab measured
                    the dispatch overhead inside one forward pass. Lab 08 zooms out to the full
                    training loop and asks: where does the other 55% of wall time go when MFU is
                    only 45%?
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
                    <strong>Read:</strong> @sec-ml-frameworks for kernel fusion mechanics and
                    the compilation continuum.<br/>
                    <strong>Build:</strong> TinyTorch Module 07 &mdash; implement a fused
                    forward-pass executor and observe the dispatch count reduction firsthand.
                </div>
            </div>

        </div>
        """),


        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. A KWS model with 1,000 kernels at 5 us compute each and 10 us launch overhead per kernel achieves what GPU utilization percentage — and what is the only way to increase it without changing the hardware?

    2. A LayerNorm + Dropout + ReLU sequence fused into one kernel reduces HBM traffic by approximately 3x and achieves ~5x wall-clock speedup. Why does eliminating intermediate read/write round-trips account for most of this gain?

    3. Compilation takes 30 seconds and provides a 35% throughput improvement. Below what number of inference iterations does the compilation overhead make it net-negative — and for what class of model (many small kernels vs. few large ones) is compilation most valuable?

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ──────────────────────────────────────────────────────
@app.cell
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    mo,
    ledger, COLORS,
    act1_pred, act1_reflection,
    act2_pred, act2_reflection_a, act2_reflection_b,
    act2_fusion_toggle, act2_compile_time_slider,
    act2_volume_slider, act2_duration_slider,
    context_toggle,
    RESNET50_THROUGHPUT_EAGER, RESNET50_THROUGHPUT_COMPILED,
    RESNET50_BREAKEVEN_IMAGES,
, decision_input, decision_ui):
    _ctx = context_toggle.value

    # Determine completion state
    _act1_done = act1_pred.value is not None and act1_reflection.value is not None
    _act2_done = (
        act2_pred.value is not None
        and act2_reflection_a.value is not None
        and act2_reflection_b.value is not None
    )

    # Compute break-even with current slider state
    _compile_s = act2_compile_time_slider.value
    _gain_pct = 0.48  # default ResNet-50 gain
    _throughput_eager = RESNET50_THROUGHPUT_EAGER
    _throughput_compiled = _throughput_eager * (1 + _gain_pct)
    _delta_t = (1 / _throughput_eager) - (1 / _throughput_compiled)
    _breakeven_images = int(_compile_s / _delta_t) if _delta_t > 0 else 999999
    _total_images = act2_volume_slider.value * act2_duration_slider.value
    _roi_positive = _total_images >= _breakeven_images

    # KWS utilization values (fixed physics, for ledger record)
    _kws_eager_util = 33
    _kws_compiled_util = 67

    # Save to Design Ledger (only when both acts complete)
    if _act1_done and _act2_done:
        ledger.save(
            chapter=7,
            design={
                "context": _ctx,
                "execution_mode": "compiled" if act2_fusion_toggle.value == "fused" else "eager",
                "fusion_enabled": act2_fusion_toggle.value == "fused",
                "compilation_roi_positive": _roi_positive,
                "breakeven_inferences": _breakeven_images,
                "kws_utilization_eager_pct": _kws_eager_util,
                "kws_utilization_compiled_pct": _kws_compiled_util,
                "act1_prediction": act1_pred.value,
                "act1_correct": act1_pred.value == "C",
                "act2_result": float(_breakeven_images),
                "act2_decision": "compiled" if _roi_positive else "eager",
                "constraint_hit": not _roi_positive,
        "student_justification": str(decision_input.value),
            },
        )

    # Progress indicators
    _act1_indicator = f'<span style="color:{COLORS["GreenLine"]}; font-weight:700;">COMPLETE</span>' if _act1_done else f'<span style="color:{COLORS["OrangeLine"]};">IN PROGRESS</span>'
    _act2_indicator = f'<span style="color:{COLORS["GreenLine"]}; font-weight:700;">COMPLETE</span>' if _act2_done else f'<span style="color:{COLORS["OrangeLine"]};">IN PROGRESS</span>'
    _ctx_color = COLORS["Cloud"] if _ctx == "cloud" else COLORS["Edge"]
    _ctx_label = "Cloud (A100)" if _ctx == "cloud" else "Edge (Jetson Orin NX)"

    mo.vstack([
        mo.md("---"),
        decision_ui,
        mo.Html(f"""
        <div class="lab-hud">
            <div>
                <span class="hud-label">LAB</span>
                <span class="hud-value"> 07 · Frameworks</span>
            </div>
            <div>
                <span class="hud-label">CONTEXT</span>
                <span style="color:{_ctx_color}; font-family: monospace; font-size: 0.8rem;"> {_ctx_label}</span>
            </div>
            <div>
                <span class="hud-label">ACT 1</span>
                <span> {_act1_indicator}</span>
            </div>
            <div>
                <span class="hud-label">ACT 2</span>
                <span> {_act2_indicator}</span>
            </div>
            <div>
                <span class="hud-label">FUSION</span>
                <span class="{'hud-active' if act2_fusion_toggle.value == 'fused' else 'hud-none'}">
                    {' ENABLED' if act2_fusion_toggle.value == 'fused' else ' DISABLED'}
                </span>
            </div>
            <div>
                <span class="hud-label">COMPILE ROI</span>
                <span class="{'hud-active' if _roi_positive else 'hud-none'}">
                    {' POSITIVE' if _roi_positive else ' NEGATIVE'}
                </span>
            </div>
            <div>
                <span class="hud-label">BREAK-EVEN</span>
                <span class="hud-value"> {_breakeven_images:,} imgs</span>
            </div>
            <div>
                <span class="hud-label">KWS UTIL</span>
                <span class="hud-value"> {_kws_eager_util}% eager → {_kws_compiled_util}% compiled</span>
            </div>
        </div>
        """),
        mo.callout(mo.md("""**Ledger saved to chapter 7.** Lab 08 (Training) reads `kws_utilization_compiled_pct` to initialize the MFU pipeline breakdown. Lab 11 (HW Acceleration) reads `fusion_enabled` and arithmetic intensity values to set the starting position on the Roofline curve."""), kind="info") if _act1_done and _act2_done else mo.callout(mo.md("Complete both acts to save results to the Design Ledger and unlock cross-lab continuity."), kind="warn"),
    ])
    return


if __name__ == "__main__":
    app.run()
