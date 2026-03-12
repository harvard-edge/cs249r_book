import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 11: THE ROOFLINE
#
# Core Invariant:
#   Every compute kernel is either compute-bound or memory-bandwidth-bound.
#   The Roofline Model makes this concrete:
#       attainable_perf = min(peak_flops, bandwidth x arithmetic_intensity)
#   The ridge point = peak_flops / peak_bandwidth separates the two regimes.
#   MFU (Model FLOP Utilization) = achieved_FLOPS / peak_FLOPS.
#
# Contexts: Cloud H100 (80 GB HBM3e, 3350 GB/s, 989 TFLOPS FP16) vs
#           Edge  Jetson Orin NX (16 GB, 102 GB/s, ~12 TFLOPS FP16)
#
# New Instrument: Interactive Roofline Model (first introduction in curriculum)
#
# Act I  — The Memory Wall (12-15 min)
#   Scenario: GPU Kernel Engineer — GEMM achieves 31.5% MFU on H100. Why?
#   Prediction: Why is MFU only 31.5%? (memory-bound, below ridge point)
#   Instrument: Interactive Roofline — adjust M×N×K, precision, device
#   Reflection: Most effective MFU improvement when memory-bound?
#   Correct: Increase tile dimensions to raise arithmetic intensity
#
# Act II — The Design Challenge (20-25 min)
#   Scenario: ML Infra Lead — 3 LLM kernel types, which are bottlenecks?
#   Prediction: Which kernel types are memory-bound vs compute-bound?
#   Instrument: Multi-operation Roofline + kernel fusion strategies
#   Failure state: KV-cache + activations exceed device RAM
#   Reflection: Why does kernel fusion improve memory-bound ops?
#   Correct: Eliminates redundant reads/writes — fused ops load data once
#
# Key constants (all from NVIDIA spec sheets and hw_acceleration.qmd):
#   H100_BW_GBS      = 3350   # H100 SXM5 HBM3e bandwidth, NVIDIA spec
#   H100_TFLOPS_FP16 = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
#   H100_RAM_GB      = 80     # H100 HBM3e capacity, NVIDIA spec
#   H100_RIDGE_PT    = 295    # FLOP/byte = 989e12 / 3350e9, derived
#   ORIN_BW_GBS      = 102    # Jetson Orin NX 16GB, NVIDIA spec
#   ORIN_TFLOPS_FP16 = 12     # Jetson Orin NX FP16 TFLOPS, estimated from GPU die
#   ORIN_RAM_GB      = 16     # Jetson Orin NX RAM, NVIDIA spec
#   ORIN_RIDGE_PT    = 10     # conservative FLOP/byte estimate per chapter text
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP (hide_code=False — leave visible for instructors) ──────────
@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    ledger = DesignLedger()

    # ── Hardware constants (all plain floats, no pint units) ──────────────────

    # Cloud: NVIDIA H100 SXM5 (NVIDIA spec sheet, 2023)
    H100_BW_GBS      = 3350   # GB/s HBM3e memory bandwidth
    H100_TFLOPS_FP16 = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_TFLOPS_FP32 = 67     # TFLOPS FP32 (non-tensor)
    H100_TFLOPS_INT8 = 3958   # TFLOPS INT8 tensor core peak (2x FP8)
    H100_RAM_GB      = 80     # GB HBM3e total capacity
    H100_TDP_W       = 700    # Watts TDP
    H100_RIDGE_PT    = 295    # FLOP/byte = 989e12 / 3350e9 (derived)

    # Edge: NVIDIA Jetson Orin NX 16GB (NVIDIA Jetson product brief 2023)
    ORIN_BW_GBS      = 102    # GB/s LPDDR5 memory bandwidth
    ORIN_TFLOPS_FP16 = 12     # TFLOPS FP16 estimated from GPU die specs
    ORIN_TFLOPS_INT8 = 100    # TOPS INT8 (advertised on product page)
    ORIN_RAM_GB      = 16     # GB LPDDR5
    ORIN_TDP_W       = 25     # Watts maximum TDP
    ORIN_RIDGE_PT    = 80     # FLOP/byte sustained estimate (12 TFLOPS / 102 GB/s = 118 peak;
    # ~80 assumes ~70% bandwidth utilization under realistic sustained workloads,
    # consistent with LPDDR5 efficiency measurements in embedded inference contexts)

    # Bytes per element by precision
    BYTES_FP32  = 4     # bytes per FP32 element
    BYTES_FP16  = 2     # bytes per FP16/BF16 element
    BYTES_INT8  = 1     # bytes per INT8 element

    # Llama-3-8B architecture constants (public model card, Meta 2024)
    LLAMA3_DMODEL   = 4096   # embedding dimension
    LLAMA3_LAYERS   = 32     # transformer layers
    LLAMA3_HEADS    = 32     # attention heads
    LLAMA3_PARAMS   = 8e9    # total parameter count

    return (
        mo, ledger, go, np,
        COLORS, LAB_CSS, apply_plotly_theme,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_TFLOPS_FP32, H100_TFLOPS_INT8,
        H100_RAM_GB, H100_TDP_W, H100_RIDGE_PT,
        ORIN_BW_GBS, ORIN_TFLOPS_FP16, ORIN_TFLOPS_INT8,
        ORIN_RAM_GB, ORIN_TDP_W, ORIN_RIDGE_PT,
        BYTES_FP32, BYTES_FP16, BYTES_INT8,
        LLAMA3_DMODEL, LLAMA3_LAYERS, LLAMA3_HEADS, LLAMA3_PARAMS,
    )


# ─── HEADER ──────────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS):
    mo.vstack([
        LAB_CSS,
        mo.md("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume I · Lab 11
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Roofline
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 680px; line-height: 1.65;">
                A kernel that achieves 312 TFLOPS on a 989 TFLOPS accelerator
                is running at 31.5% efficiency. Is the algorithm broken?
                No — the roof is somewhere else. This lab forces you to find it.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Cloud vs Edge
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min
                </span>
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Prerequisite: hw_acceleration.qmd
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    New instrument: Roofline Model
                </span>
            </div>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 14px;">
                <span class="badge badge-ok">AI above ridge = Compute-Bound</span>
                <span class="badge badge-warn">AI below ridge = Memory-Bound</span>
                <span class="badge badge-fail">KV-cache exceeds device RAM = OOM</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Diagnose whether a GEMM kernel is memory-bound or compute-bound</strong> — compute its arithmetic intensity and compare it to the hardware ridge point (295 FLOP/byte for H100 FP16) to determine which ceiling limits performance.</div>
                <div style="margin-bottom: 3px;">2. <strong>Quantify the H100 compute utilization of a transformer attention layer at 2.7%</strong> — explain why a kernel achieving 26.8 TFLOPS out of 989 TFLOPS peak is correctly behaving, not broken, and identify the correct optimization strategy.</div>
                <div style="margin-bottom: 3px;">3. <strong>Predict the minimum batch size that moves a memory-bound workload to compute-bound</strong> — determine at what point increasing batch size raises arithmetic intensity above the ridge point, and discover how this creates an OOM trade-off.</div>
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
                    Roofline model equation from @sec-hardware-acceleration-roofline-model-42ff &middot;
                    Arithmetic intensity definition from @sec-hardware-acceleration-roofline-model-42ff &middot;
                    MFU (Model FLOPS Utilization) concept from @sec-model-training-iron-law-training-performance-a53f
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
                "A transformer attention layer achieves only 2.7% of the H100's peak compute
                and three weeks of arithmetic optimization have not moved it &mdash; what is
                the actual bottleneck, and would doubling the accelerator's TFLOPS help at all?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: READING ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-roofline-model** — The Roofline Model: memory wall, compute ceiling, ridge point
    - **@sec-arithmetic-intensity** — Arithmetic intensity definition and GEMM derivation
    - **@sec-mfu** — Model FLOP Utilization: what it measures and why 100% is never achievable
    - **@sec-kernel-fusion** — Fusing elementwise ops: why it reduces memory traffic

    The lab assumes you know what arithmetic intensity (FLOP/byte) means.
    If the term *ridge point* is unfamiliar, re-read @sec-roofline-model first.
    """), kind="info")
    return


# ─── CELL 4: CONTEXT TOGGLE ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Cloud (H100 SXM5, 80 GB HBM (High Bandwidth Memory), 989 TFLOPS FP16)": "cloud",
            "Edge (Jetson Orin NX, 16 GB LPDDR5, ~12 TFLOPS FP16)": "edge",
        },
        value="Cloud (H100 SXM5, 80 GB HBM (High Bandwidth Memory), 989 TFLOPS FP16)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md("## Select Your Deployment Context"),
        mo.md(
            "The ridge point separating memory-bound from compute-bound is a hardware "
            "constant. On an H100 it is 295 FLOP/byte; on a Jetson Orin NX it is roughly "
            "10 FLOP/byte. Select your context — the entire roofline changes with it."
        ),
        context_toggle,
    ])
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS):
    _ctx   = context_toggle.value
    _act_num = "I"
    _act_color = COLORS["BlueLine"]
    _act_title = "The Memory Wall"
    _act_duration = "12 min"
    _ridge_str = "295 FLOP/byte" if _ctx == "cloud" else "~10 FLOP/byte"
    _device_name = "H100 SXM5" if _ctx == "cloud" else "Jetson Orin NX"
    _act_why = (
        f"You expect that a kernel achieving only 31.5% MFU has a bug or a suboptimal algorithm. "
        f"The Roofline will show that the kernel is behaving correctly: its arithmetic intensity "
        f"is below the ridge point ({_ridge_str}), so memory bandwidth \u2014 not arithmetic "
        f"throughput \u2014 is the binding constraint. No amount of compute tuning can help."
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
                    Act {_act_num} &middot; {_act_duration} &middot; {_device_name}</div>
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
def _(mo, context_toggle, COLORS):
    _ctx   = context_toggle.value
    _color = COLORS["Cloud"] if _ctx == "cloud" else COLORS["Edge"]
    _bg    = "#EBF4FA"        if _ctx == "cloud" else COLORS["RedLL"]

    _device_name  = "H100 SXM5"        if _ctx == "cloud" else "Jetson Orin NX"
    _peak_tflops  = "989 TFLOPS FP16"  if _ctx == "cloud" else "~12 TFLOPS FP16"
    _peak_bw      = "3350 GB/s"        if _ctx == "cloud" else "102 GB/s"
    _ridge_str    = "295 FLOP/byte"    if _ctx == "cloud" else "~10 FLOP/byte"

    mo.vstack([
        mo.Html(f"""
        <div style="border-left:4px solid {_color}; background:{_bg};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; GPU Kernel Engineer
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "Our matrix multiply kernel achieves 312 TFLOPS on the {_device_name}.
                Peak is {_peak_tflops}. MFU is about 31.5%.
                We have spent three weeks tuning the arithmetic &mdash; register tiling,
                loop unrolling, mixed precision. The math has not moved one TFLOP.
                Our manager is asking why we keep hitting the same wall.
                What ceiling are we running into?"
            </div>
        </div>
        """),
        mo.md(f"""
        The engineer has been optimizing the wrong bottleneck for three weeks.
        The {_device_name} has a peak compute of {_peak_tflops} but a memory
        bandwidth of {_peak_bw}. The **ridge point** is the arithmetic intensity
        at which a kernel transitions from memory-bound to compute-bound: **{_ridge_str}**.

        A kernel running *below* the ridge point is bottlenecked by data movement,
        not by floating-point throughput. No amount of arithmetic optimization
        will improve a memory-bound kernel.
        """),
    ])
    return


# ─── ACT I PREDICTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) The algorithm has a bug — matrix multiply should always be compute-bound": "A",
            "B) The kernel is not memory-bound — increase the tile size": "B",
            "C) The operation is memory-bandwidth-bound — arithmetic intensity is below the ridge point": "C",
            "D) 15% MFU is already the hardware maximum for matrix multiply": "D",
        },
        label="Your prediction: why is MFU only 15.7% despite correct arithmetic?",
    )
    mo.vstack([
        mo.md("### Your Prediction"),
        mo.md(
            "*Before touching the simulator, commit to your hypothesis. "
            "The roofline is locked until you do.*"
        ),
        act1_pred,
    ])
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Roofline simulator."),
            kind="warn",
        )
    )
    mo.callout(
        mo.md(f"**Prediction locked:** Option {act1_pred.value}. Now explore the simulator below."),
        kind="info",
    )
    return


# ─── ACT I ROOFLINE CONTROLS ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_matrix_n = mo.ui.slider(
        start=128, stop=8192, value=512, step=128,
        label="Matrix dimension N (square M=N=K)",
    )
    act1_precision = mo.ui.dropdown(
        options={
            "FP32 (4 bytes/element)": "fp32",
            "FP16 / BF16 (2 bytes/element)": "fp16",
            "INT8 (1 byte/element)": "int8",
        },
        value="FP16 / BF16 (2 bytes/element)",
        label="Precision",
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### The Roofline Simulator"),
        mo.md(
            "Adjust matrix dimension and precision. Watch the operation point move. "
            "The vertical dashed line is the ridge point — the boundary between "
            "memory-bound (left) and compute-bound (right)."
        ),
        mo.hstack([act1_matrix_n, act1_precision], justify="start", gap="2rem"),
    ])
    return (act1_matrix_n, act1_precision)


# ─── ACT I ROOFLINE PLOT ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, np,
    context_toggle,
    act1_matrix_n, act1_precision,
    COLORS,
    H100_BW_GBS, H100_TFLOPS_FP16, H100_TFLOPS_FP32, H100_TFLOPS_INT8, H100_RIDGE_PT,
    ORIN_BW_GBS, ORIN_TFLOPS_FP16, ORIN_TFLOPS_INT8, ORIN_RIDGE_PT,
    BYTES_FP32, BYTES_FP16, BYTES_INT8,
):
    _ctx       = context_toggle.value
    _N         = act1_matrix_n.value
    _precision = act1_precision.value

    # ── Device specs based on context and precision ───────────────────────────
    if _ctx == "cloud":
        _peak_bw_gbs  = H100_BW_GBS
        _ridge_pt     = H100_RIDGE_PT
        _device_label = "H100 SXM5"
        _ctx_color    = COLORS["Cloud"]
        if _precision == "fp32":
            _peak_flops_t = H100_TFLOPS_FP32
        elif _precision == "int8":
            _peak_flops_t = H100_TFLOPS_INT8
        else:
            _peak_flops_t = H100_TFLOPS_FP16
    else:
        _peak_bw_gbs  = ORIN_BW_GBS
        _ridge_pt     = ORIN_RIDGE_PT
        _device_label = "Jetson Orin NX"
        _ctx_color    = COLORS["Edge"]
        if _precision == "int8":
            _peak_flops_t = ORIN_TFLOPS_INT8
        else:
            _peak_flops_t = ORIN_TFLOPS_FP16

    # ── Precision bytes ────────────────────────────────────────────────────────
    if _precision == "fp32":
        _bytes_elem = BYTES_FP32
        _prec_label = "FP32"
    elif _precision == "int8":
        _bytes_elem = BYTES_INT8
        _prec_label = "INT8"
    else:
        _bytes_elem = BYTES_FP16
        _prec_label = "FP16"

    # ── GEMM arithmetic intensity ─────────────────────────────────────────────
    # For square M=N=K GEMM at B bytes/element:
    #   FLOPs = 2 x N^3   (N multiply-adds per dot product, N^2 outputs)
    #   Bytes = (MN + NK + MK) x B = 3 x N^2 x B  (read A, B, write C)
    #   AI    = 2N^3 / (3N^2 x B) = 2N / (3B)
    _flops_gemm = 2.0 * _N * _N * _N
    _bytes_gemm = 3.0 * _N * _N * _bytes_elem
    _ai_gemm    = _flops_gemm / _bytes_gemm

    # ── Attainable performance ────────────────────────────────────────────────
    _peak_flops_per_s  = _peak_flops_t * 1e12
    _peak_bw_per_s     = _peak_bw_gbs * 1e9
    _attain_t = min(_peak_flops_t, (_peak_bw_per_s * _ai_gemm) / 1e12)
    _mfu_pct  = (_attain_t / _peak_flops_t) * 100.0

    _is_mem_bound   = _ai_gemm < _ridge_pt
    _regime_label   = "Memory-Bound"  if _is_mem_bound else "Compute-Bound"
    _regime_color   = COLORS["OrangeLine"] if _is_mem_bound else COLORS["GreenLine"]

    # ── Build roofline curve ───────────────────────────────────────────────────
    _ai_axis    = np.logspace(-1, 4, 500)
    _mem_slope  = (_peak_bw_per_s * _ai_axis) / 1e12
    _comp_ceil  = np.full_like(_ai_axis, _peak_flops_t)
    _roofline   = np.minimum(_mem_slope, _comp_ceil)

    # ── Build figure ──────────────────────────────────────────────────────────
    _fig = go.Figure()

    # Memory-bound segment (left of ridge)
    _mask_m = _ai_axis <= _ridge_pt
    _fig.add_trace(go.Scatter(
        x=_ai_axis[_mask_m], y=_roofline[_mask_m],
        mode="lines", name="Memory-bound roof",
        line=dict(color=COLORS["OrangeLine"], width=3),
    ))

    # Compute-bound segment (right of ridge)
    _mask_c = _ai_axis >= _ridge_pt
    _fig.add_trace(go.Scatter(
        x=_ai_axis[_mask_c], y=_roofline[_mask_c],
        mode="lines", name="Compute ceiling",
        line=dict(color=COLORS["GreenLine"], width=3),
    ))

    # Ridge point vertical dashed line
    _fig.add_vline(
        x=_ridge_pt,
        line=dict(color="#64748b", width=2, dash="dash"),
        annotation_text=f"Ridge: {_ridge_pt:.0f} FLOP/byte",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#64748b"),
    )

    # Shaded zones
    _fig.add_vrect(x0=0.1, x1=_ridge_pt,
        fillcolor=COLORS["OrangeLine"], opacity=0.06, layer="below", line_width=0)
    _fig.add_vrect(x0=_ridge_pt, x1=10000,
        fillcolor=COLORS["GreenLine"], opacity=0.04, layer="below", line_width=0)

    # Zone text labels
    _fig.add_annotation(x=0.18, y=0.12, xref="paper", yref="paper",
        text="Memory-Bound Zone", font=dict(size=10, color=COLORS["OrangeLine"]),
        showarrow=False)
    _fig.add_annotation(x=0.82, y=0.12, xref="paper", yref="paper",
        text="Compute-Bound Zone", font=dict(size=10, color=COLORS["GreenLine"]),
        showarrow=False)

    # Vertical drop to x-axis from operation point
    _fig.add_shape(
        type="line",
        x0=_ai_gemm, y0=1e-3,
        x1=_ai_gemm, y1=_attain_t,
        line=dict(color=_regime_color, width=1, dash="dot"),
        layer="below",
    )

    # Operation point
    _fig.add_trace(go.Scatter(
        x=[_ai_gemm], y=[_attain_t],
        mode="markers+text",
        name=f"GEMM {_N}x{_N}x{_N} ({_prec_label})",
        marker=dict(size=16, color=_regime_color,
                    line=dict(color="white", width=2), symbol="circle"),
        text=[f"  {_attain_t:.0f} TFLOPS ({_mfu_pct:.1f}% MFU)"],
        textposition="middle right",
        textfont=dict(size=11, color=_regime_color),
    ))

    _fig.update_layout(
        title=dict(
            text=f"Roofline Model — {_device_label} ({_prec_label})",
            font=dict(size=15, color=COLORS["Text"]), x=0.02,
        ),
        xaxis=dict(type="log", title="Arithmetic Intensity (FLOP / byte)",
                   range=[-1, 4], gridcolor="#f1f5f9", linecolor=COLORS["Border"]),
        yaxis=dict(type="log", title="Attainable Performance (TFLOPS)",
                   range=[-2, 4] if _ctx == "cloud" else [-2, 2.5],
                   gridcolor="#f1f5f9", linecolor=COLORS["Border"]),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
        height=460,
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter, sans-serif", font_color=COLORS["Text"],
        margin=dict(l=60, r=30, t=50, b=60),
    )

    # ── Metric cards ──────────────────────────────────────────────────────────
    _mfu_color = (COLORS["GreenLine"]   if _mfu_pct > 50
                  else COLORS["OrangeLine"] if _mfu_pct > 20
                  else COLORS["RedLine"])
    _ai_color  = COLORS["GreenLine"] if not _is_mem_bound else COLORS["OrangeLine"]

    mo.vstack([
        mo.as_html(_fig),
        mo.Html(f"""
        <div style="display:flex; gap:16px; flex-wrap:wrap; margin:16px 0 8px 0;">
            <div style="padding:18px 24px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:165px; text-align:center; background:white;
                        box-shadow:0 2px 6px rgba(0,0,0,0.04);">
                <div style="color:#64748b; font-size:0.78rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">
                    Arithmetic Intensity
                </div>
                <div style="font-size:2rem; font-weight:800; color:{_ai_color};">{_ai_gemm:.0f}</div>
                <div style="font-size:0.73rem; color:#94a3b8;">FLOP / byte</div>
            </div>
            <div style="padding:18px 24px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:165px; text-align:center; background:white;
                        box-shadow:0 2px 6px rgba(0,0,0,0.04);">
                <div style="color:#64748b; font-size:0.78rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">
                    Attainable Perf
                </div>
                <div style="font-size:2rem; font-weight:800; color:{_mfu_color};">{_attain_t:.0f}</div>
                <div style="font-size:0.73rem; color:#94a3b8;">TFLOPS</div>
            </div>
            <div style="padding:18px 24px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:165px; text-align:center; background:white;
                        box-shadow:0 2px 6px rgba(0,0,0,0.04);">
                <div style="color:#64748b; font-size:0.78rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">
                    MFU
                </div>
                <div style="font-size:2rem; font-weight:800; color:{_mfu_color};">{_mfu_pct:.1f}%</div>
                <div style="font-size:0.73rem; color:#94a3b8;">vs peak {_peak_flops_t:.0f} TFLOPS</div>
            </div>
            <div style="padding:18px 24px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:165px; text-align:center; background:white;
                        box-shadow:0 2px 6px rgba(0,0,0,0.04);">
                <div style="color:#64748b; font-size:0.78rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">
                    Regime
                </div>
                <div style="font-size:1.4rem; font-weight:800; color:{_regime_color}; margin:6px 0 2px 0;">
                    {_regime_label}
                </div>
                <div style="font-size:0.73rem; color:#94a3b8;">AI {'<' if _is_mem_bound else '>='} {_ridge_pt:.0f}</div>
            </div>
        </div>
        """),
        mo.md(f"""
**Physics (visible):**

```
GEMM FLOPs     = 2 x M x N x K = 2 x {_N}^3 = {_flops_gemm/1e9:.1f} GFLOPs
GEMM Bytes     = (MN + NK + MK) x {_bytes_elem} = 3 x {_N}^2 x {_bytes_elem} = {_bytes_gemm/1e6:.1f} MB
Arith. Intens. = {_flops_gemm:.2e} / {_bytes_gemm:.2e} = {_ai_gemm:.1f} FLOP/byte

Ridge Point    = peak_FLOPS / BW = {_peak_flops_t} TFLOPS / {_peak_bw_gbs} GB/s = {_ridge_pt:.0f} FLOP/byte

Attainable     = min(peak_FLOPS, BW x AI)
               = min({_peak_flops_t}, {_peak_bw_gbs}e9 x {_ai_gemm:.1f} / 1e12) TFLOPS
               = min({_peak_flops_t:.0f}, {(_peak_bw_gbs*1e9*_ai_gemm/1e12):.0f}) TFLOPS
               = {_attain_t:.1f} TFLOPS

MFU            = {_attain_t:.1f} / {_peak_flops_t:.0f} = {_mfu_pct:.1f}%
```
        """),
    ])
    return (_ai_gemm, _attain_t, _mfu_pct, _is_mem_bound, _peak_flops_t, _ridge_pt, _regime_label)


# ─── ACT I PREDICTION VS REALITY ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, _ai_gemm, _ridge_pt, _is_mem_bound, _regime_label):
    _pred = act1_pred.value

    _feedback = {
        "A": (
            "**Not quite.** There is no bug. A square GEMM with N=512 has an arithmetic "
            "intensity of only ~170 FLOP/byte — far below the H100 ridge point of 295 FLOP/byte. "
            "Matrix multiply *can* be compute-bound, but only for large enough matrices. "
            "The ratio of FLOPs to bytes grows linearly with N."
        ),
        "B": (
            "**Close, but the logic is inverted.** The kernel *is* memory-bound — "
            "that is exactly the problem. Increasing tile size is the correct *solution*, "
            "not evidence against memory-boundedness. Larger tiles increase arithmetic "
            "intensity by reusing data in on-chip SRAM instead of re-fetching from HBM."
        ),
        "C": (
            "**Correct.** With N=512, arithmetic intensity is ~170 FLOP/byte — well below "
            "the H100 ridge point of 295 FLOP/byte. The kernel is memory-bandwidth-bound. "
            "Optimizing FP arithmetic does nothing when the bottleneck is data movement. "
            "The kernel spends most cycles waiting for data, not computing."
        ),
        "D": (
            "**Incorrect.** 15% MFU is not a hardware maximum. It reflects a specific "
            "operating point below the ridge. At N=8192 the same hardware achieves >70% MFU "
            "because arithmetic intensity rises to ~2730 FLOP/byte, well past the ridge."
        ),
    }

    _correct = _pred == "C"
    _actual  = "memory-bound" if _is_mem_bound else "compute-bound"

    mo.vstack([
        mo.md("### Prediction vs Reality"),
        mo.callout(
            mo.md(
                f"{_feedback.get(_pred, '')}\n\n"
                f"**Actual result:** AI = {_ai_gemm:.0f} FLOP/byte, "
                f"ridge = {_ridge_pt:.0f} FLOP/byte. "
                f"Operation is **{_actual}** ({_regime_label})."
            ),
            kind="success" if _correct else "warn",
        ),
    ])
    return


# ─── ACT I REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) Increase clock speed": "A",
            "B) Use larger batch sizes or tile dimensions to raise arithmetic intensity": "B",
            "C) Reduce floating-point precision": "C",
            "D) Add more GPU memory capacity": "D",
        },
        label="Reflection: what is the most effective way to improve MFU for a memory-bound kernel?",
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Act I Reflection"),
        act1_reflect,
    ])
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn")
    )

    _reflect_feedback = {
        "A": (
            "**Incorrect.** Higher clock speed increases raw FLOP/s, but bandwidth is the "
            "bottleneck. If data cannot move faster, the compute units remain starved "
            "regardless of frequency."
        ),
        "B": (
            "**Correct.** Larger tiles allow the kernel to reuse data held in fast on-chip "
            "SRAM across more arithmetic operations before evicting it back to HBM. "
            "This increases FLOPs per byte — shifting the operation rightward on the "
            "roofline toward and past the ridge point. This is exactly what cuBLAS tiles "
            "and FlashAttention's block structure accomplish."
        ),
        "C": (
            "**Incorrect.** Reducing precision (FP32 to FP16) halves bytes-per-element and "
            "doubles FLOP/s, but the *ratio* FLOPs/bytes stays roughly constant. "
            "The arithmetic intensity of the GEMM is unchanged for a given matrix shape."
        ),
        "D": (
            "**Incorrect.** More capacity (larger HBM) is not the same as more bandwidth. "
            "The bottleneck is the *rate* of data transfer. A larger memory pool that "
            "arrives at the same rate still starves the compute units."
        ),
    }

    _correct = act1_reflect.value == "B"
    mo.callout(
        mo.md(_reflect_feedback[act1_reflect.value]),
        kind="success" if _correct else "warn",
    )
    return


# ─── ACT I MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations": mo.md("""
**Roofline attainable performance:**

```
attainable_perf(I) = min(peak_flops, BW x I)
```

`I` = arithmetic intensity (FLOP / byte), `BW` = peak bandwidth (byte/s),
`peak_flops` = peak compute (FLOP/s).

**The ridge point** is where the two bounds are equal:

```
ridge_point = peak_flops / BW    [FLOP / byte]
```

- H100 SXM5:     ridge = 989e12 / 3350e9 = 295 FLOP/byte
- Jetson Orin NX: ridge ~10 FLOP/byte (conservative, per chapter text)

**GEMM arithmetic intensity** for square M=N=K at B bytes/element:

```
FLOPs  = 2 x N^3
Bytes  = 3 x N^2 x B   (read A, read B, write C)
I_gemm = 2N^3 / (3N^2 x B) = 2N / (3B)
```

At FP16 (B=2): I_gemm = N / 3.  So I grows *linearly with N*.

| N     | AI (FP16) | H100 regime        |
|-------|-----------|--------------------|
| 128   |  43       | Memory-Bound       |
| 512   | 170       | Memory-Bound       |
| 885   | 295       | AT ridge point     |
| 1024  | 341       | Compute-Bound      |
| 4096  | 1365      | Compute-Bound      |
| 8192  | 2730      | Compute-Bound      |

**MFU (Model FLOP Utilization):**

```
MFU = achieved_FLOPS / peak_FLOPS
```

MFU directly tells you how much of the advertised peak you are using.
A memory-bound kernel always has MFU < (BW x I) / peak_flops.
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS):
    _ctx    = context_toggle.value
    _act_num = "II"
    _act_color = COLORS["OrangeLine"]
    _act_title = "The Design Challenge"
    _act_duration = "25 min"
    _device = "H100" if _ctx == "cloud" else "Jetson Orin NX"
    _ridge2 = "295 FLOP/byte" if _ctx == "cloud" else "~10 FLOP/byte"
    _act_why = (
        f"Act I showed that a single GEMM kernel has a fixed ridge point ({_ridge2}). "
        f"Now discover that an LLM inference stack mixes kernels across the full arithmetic "
        f"intensity spectrum \u2014 and that elementwise ops (layer norm, softmax, ReLU) at "
        f"~0.8 FLOP/byte are always memory-bound on any accelerator built this decade."
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
                    Act {_act_num} &middot; {_act_duration} &middot; {_device}</div>
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
def _(mo, context_toggle, COLORS):
    _ctx    = context_toggle.value
    _color  = COLORS["Cloud"] if _ctx == "cloud" else COLORS["Edge"]
    _bg     = "#EBF4FA"        if _ctx == "cloud" else COLORS["RedLL"]
    _device = "H100"           if _ctx == "cloud" else "Jetson Orin NX"
    _ram    = "80 GB"          if _ctx == "cloud" else "16 GB"
    _ridge2 = "295 FLOP/byte"  if _ctx == "cloud" else "~10 FLOP/byte"

    mo.vstack([
        mo.Html(f"""
        <div style="border-left:4px solid {_color}; background:{_bg};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; ML Infrastructure Lead
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "We are designing an LLM inference service on the {_device}
                ({_ram} RAM, ridge point {_ridge2}).
                We have three kernel types: (1) large GEMM for the attention
                projection layers, (2) layer normalization, (3) softmax over
                attention scores. We have a budget to optimize exactly one kernel.
                Half the team says GEMM. Half says fuse the elementwise ops.
                Which operations are the actual bottlenecks and what do we do first?"
            </div>
        </div>
        """),
        mo.md(f"""
        An LLM inference stack mixes kernels with wildly different arithmetic intensities.
        Large GEMM operations can become compute-bound at large batch sizes.
        Elementwise operations — layer norm, softmax — have arithmetic intensities
        near 0.8 FLOP/byte and are *always* memory-bound regardless of batch size.

        Kernel fusion addresses memory-bound ops by eliminating redundant HBM round-trips.
        But push batch size too far and the KV-cache alone exhausts device RAM.
        """),
    ])
    return


# ─── ACT II PREDICTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) Matrix multiply is always the bottleneck — optimize it first": "A",
            "B) Softmax and layer norm are likely memory-bound; GEMM may be compute-bound at large batch sizes": "B",
            "C) All three operations have similar arithmetic intensity": "C",
            "D) Layer norm is compute-bound because it computes square roots": "D",
        },
        label="Your prediction: which kernel types are memory-bound vs compute-bound in LLM inference?",
    )
    mo.vstack([
        mo.md("### Your Prediction"),
        mo.md("*Commit before configuring the multi-operation design.*"),
        act2_pred,
    ])
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the design instruments."),
            kind="warn",
        )
    )
    mo.callout(
        mo.md(f"**Prediction locked:** Option {act2_pred.value}. Configure the design below."),
        kind="info",
    )
    return


# ─── ACT II DESIGN CONTROLS ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_batch = mo.ui.slider(
        start=1, stop=256, value=8, step=1,
        label="Batch size (concurrent sequences)",
    )
    act2_seqlen = mo.ui.slider(
        start=256, stop=4096, value=512, step=256,
        label="Sequence length (tokens)",
    )
    act2_fusion = mo.ui.dropdown(
        options={
            "Separate kernels (no fusion)":                "none",
            "Fuse softmax + layer norm":                   "partial",
            "Fuse all elementwise ops (softmax+LN+bias)":  "full",
        },
        value="Separate kernels (no fusion)",
        label="Kernel fusion strategy",
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Design Instruments"),
        mo.md(
            "Adjust batch size, sequence length, and fusion strategy. "
            "Watch the operation points shift across the roofline and monitor the OOM boundary."
        ),
        mo.hstack([act2_batch, act2_seqlen], justify="start", gap="2rem"),
        act2_fusion,
    ])
    return (act2_batch, act2_seqlen, act2_fusion)


# ─── ACT II MULTI-OPERATION ROOFLINE ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, np,
    context_toggle,
    act2_batch, act2_seqlen, act2_fusion,
    COLORS,
    H100_BW_GBS, H100_TFLOPS_FP16, H100_RIDGE_PT, H100_RAM_GB,
    ORIN_BW_GBS, ORIN_TFLOPS_FP16, ORIN_RIDGE_PT, ORIN_RAM_GB,
    LLAMA3_DMODEL, LLAMA3_LAYERS, LLAMA3_HEADS, LLAMA3_PARAMS,
    BYTES_FP16,
):
    _ctx    = context_toggle.value
    _B      = act2_batch.value
    _S      = act2_seqlen.value
    _fusion = act2_fusion.value

    # ── Device ────────────────────────────────────────────────────────────────
    if _ctx == "cloud":
        _peak_bw_gbs  = H100_BW_GBS
        _peak_flops_t = H100_TFLOPS_FP16
        _ridge_pt2    = H100_RIDGE_PT
        _device_ram   = H100_RAM_GB
        _dev_label    = "H100 SXM5"
        _ctx_color    = COLORS["Cloud"]
    else:
        _peak_bw_gbs  = ORIN_BW_GBS
        _peak_flops_t = ORIN_TFLOPS_FP16
        _ridge_pt2    = ORIN_RIDGE_PT
        _device_ram   = ORIN_RAM_GB
        _dev_label    = "Jetson Orin NX"
        _ctx_color    = COLORS["Edge"]

    _peak_flops_per_s = _peak_flops_t * 1e12
    _peak_bw_per_s    = _peak_bw_gbs * 1e9

    # ── Op 1: GEMM (attention Q/K/V projection) ───────────────────────────────
    # Shape: [B, S, D] x [D, D] -> [B, S, D]
    # FLOPs = 2 x B x S x D x D   (one projection; 4 total Q,K,V,out)
    # Bytes = (B*S*D + D*D + B*S*D) x 2  (read input, weight, write output)
    _D          = LLAMA3_DMODEL
    _gemm_flops = 2.0 * _B * _S * _D * _D
    _gemm_bytes = (_B * _S * _D + _D * _D + _B * _S * _D) * BYTES_FP16
    _gemm_ai2   = _gemm_flops / _gemm_bytes

    # ── Op 2: Layer Norm [B, S, D] ────────────────────────────────────────────
    # FLOPs per token: 5*D (mean, var, normalize, scale, shift)
    # Bytes per token: 3*D*bytes (load, compute, store)
    # AI = 5D / (3D * 2) = 5/6 ~ 0.83 FLOP/byte
    _ln_flops_tok = 5.0 * _D
    _ln_bytes_tok = 3.0 * _D * BYTES_FP16
    _ln_ai_base   = _ln_flops_tok / _ln_bytes_tok  # ~0.83 FLOP/byte

    # ── Op 3: Softmax [B, heads, S, S] ────────────────────────────────────────
    # FLOPs per element: ~5 (max sub, exp, sum, div, log2)
    # Bytes: 3 * bytes (load, intermediate, store)
    # AI = 5 / (3*2) ~ 0.83 FLOP/byte
    _sm_ai_base = 5.0 / (3.0 * BYTES_FP16)  # ~0.83 FLOP/byte

    # ── Fusion adjustment ─────────────────────────────────────────────────────
    # Fusion eliminates intermediate HBM round-trips.
    # Unfused: each op reads+writes full tensor — byte factor = 1.0
    # partial (softmax+LN fused): one fewer round-trip each — factor ~0.50
    # full (all elementwise fused): two fewer round-trips — factor ~0.35
    # This raises effective AI by reducing denominator bytes.
    _ff = {"none": 1.0, "partial": 0.5, "full": 0.35}[_fusion]

    _ln_ai_eff = _ln_ai_base / _ff
    _sm_ai_eff = _sm_ai_base / _ff

    # ── Attainable performance ────────────────────────────────────────────────
    def _attain2(ai_val):
        return min(_peak_flops_t, (_peak_bw_per_s * ai_val) / 1e12)

    _gemm_perf2 = _attain2(_gemm_ai2)
    _ln_perf2   = _attain2(_ln_ai_eff)
    _sm_perf2   = _attain2(_sm_ai_eff)

    # ── Memory footprint ──────────────────────────────────────────────────────
    # KV-cache: 2 tensors (K,V) each [B, S, heads, d_head, layers] at FP16
    # d_head = D / heads = 4096/32 = 128
    _d_head       = _D // LLAMA3_HEADS   # 128
    _kv_gb        = (2.0 * _B * _S * LLAMA3_HEADS * _d_head
                     * LLAMA3_LAYERS * BYTES_FP16) / 1e9

    # Weights in FP16: params * 2 bytes / 1e9
    _weights_gb   = (LLAMA3_PARAMS * BYTES_FP16) / 1e9   # ~16 GB for 8B

    # Activations (inference only, no gradients): rough 0.5x weights
    _activ_gb     = 0.5 * _weights_gb

    _total_mem_gb = _weights_gb + _kv_gb + _activ_gb
    _oom2         = _total_mem_gb > _device_ram

    # ── Throughput estimate (tokens/sec) ─────────────────────────────────────
    # Dominated by GEMM: FLOPs per token = 2 * D^2 * 4 projections * layers
    _flops_per_tok   = 2.0 * _D * _D * 4 * LLAMA3_LAYERS
    _tokens_per_sec2 = (_gemm_perf2 * 1e12) / _flops_per_tok if not _oom2 else 0.0

    # ── Regime helper ─────────────────────────────────────────────────────────
    def _regime2(ai_val):
        if ai_val < _ridge_pt2:
            return "Memory-Bound", COLORS["OrangeLine"]
        return "Compute-Bound", COLORS["GreenLine"]

    _gemm_reg2, _gemm_rc2 = _regime2(_gemm_ai2)
    _ln_reg2,   _ln_rc2   = _regime2(_ln_ai_eff)
    _sm_reg2,   _sm_rc2   = _regime2(_sm_ai_eff)

    # ── OOM banner ────────────────────────────────────────────────────────────
    if _oom2:
        _oom_banner = mo.callout(
            mo.md(
                f"**OOM — Infeasible Design.** "
                f"Required: {_total_mem_gb:.1f} GB "
                f"(weights {_weights_gb:.1f} GB + KV-cache {_kv_gb:.1f} GB "
                f"+ activations {_activ_gb:.1f} GB) "
                f"| Available: {_device_ram:.0f} GB ({_dev_label}). "
                f"Reduce batch size or sequence length to stay within device RAM."
            ),
            kind="danger",
        )
    else:
        _head_gb = _device_ram - _total_mem_gb
        _oom_banner = mo.callout(
            mo.md(
                f"**Memory budget OK.** "
                f"Total: {_total_mem_gb:.1f} GB / {_device_ram:.0f} GB used. "
                f"Headroom: {_head_gb:.1f} GB."
            ),
            kind="success",
        )

    # ── Build multi-operation Roofline figure ─────────────────────────────────
    _ai_axis2  = np.logspace(-2, 4, 500)
    _mem_perf2 = (_peak_bw_per_s * _ai_axis2) / 1e12
    _comp_c2   = np.full_like(_ai_axis2, _peak_flops_t)
    _roof2     = np.minimum(_mem_perf2, _comp_c2)

    _fig2 = go.Figure()

    _mask_m2 = _ai_axis2 <= _ridge_pt2
    _fig2.add_trace(go.Scatter(
        x=_ai_axis2[_mask_m2], y=_roof2[_mask_m2],
        mode="lines", name="Memory-bound roof",
        line=dict(color=COLORS["OrangeLine"], width=3),
    ))
    _mask_c2 = _ai_axis2 >= _ridge_pt2
    _fig2.add_trace(go.Scatter(
        x=_ai_axis2[_mask_c2], y=_roof2[_mask_c2],
        mode="lines", name="Compute ceiling",
        line=dict(color=COLORS["GreenLine"], width=3),
    ))

    _fig2.add_vline(
        x=_ridge_pt2,
        line=dict(color="#64748b", width=2, dash="dash"),
        annotation_text=f"Ridge: {_ridge_pt2:.0f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#64748b"),
    )

    _fig2.add_vrect(x0=0.01, x1=_ridge_pt2,
        fillcolor=COLORS["OrangeLine"], opacity=0.05, layer="below", line_width=0)
    _fig2.add_vrect(x0=_ridge_pt2, x1=10000,
        fillcolor=COLORS["GreenLine"], opacity=0.04, layer="below", line_width=0)

    # Drop lines
    for _ai_pt, _perf_pt, _col_pt in [
        (_gemm_ai2, _gemm_perf2, "#6366f1"),
        (_ln_ai_eff, _ln_perf2, COLORS["OrangeLine"]),
        (_sm_ai_eff, _sm_perf2, COLORS["BlueLine"]),
    ]:
        _fig2.add_shape(
            type="line",
            x0=_ai_pt, y0=1e-4, x1=_ai_pt, y1=_perf_pt,
            line=dict(color=_col_pt, width=1, dash="dot"), layer="below",
        )

    # Operation scatter points
    _ops_data = [
        ("GEMM (attn. projection)", _gemm_ai2,   _gemm_perf2, "#6366f1",            "circle"),
        ("Layer Norm",              _ln_ai_eff,  _ln_perf2,   COLORS["OrangeLine"],  "diamond"),
        ("Softmax",                 _sm_ai_eff,  _sm_perf2,   COLORS["BlueLine"],    "square"),
    ]
    for _op_nm, _op_ai, _op_perf, _op_color, _op_sym in _ops_data:
        _fig2.add_trace(go.Scatter(
            x=[_op_ai], y=[_op_perf],
            mode="markers+text",
            name=_op_nm,
            marker=dict(size=15, color=_op_color, symbol=_op_sym,
                        line=dict(color="white", width=2)),
            text=[f"  {_op_ai:.1f} F/B"],
            textposition="middle right",
            textfont=dict(size=10, color=_op_color),
        ))

    _fig2.update_layout(
        title=dict(
            text=f"Multi-Operation Roofline — {_dev_label} | B={_B}, S={_S}, fusion={_fusion}",
            font=dict(size=14, color=COLORS["Text"]), x=0.02,
        ),
        xaxis=dict(type="log", title="Arithmetic Intensity (FLOP / byte)",
                   range=[-2, 4], gridcolor="#f1f5f9"),
        yaxis=dict(type="log", title="Attainable Performance (TFLOPS)",
                   range=[-3, 4] if _ctx == "cloud" else [-3, 2],
                   gridcolor="#f1f5f9"),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.88)"),
        height=490,
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter, sans-serif", font_color=COLORS["Text"],
        margin=dict(l=60, r=30, t=50, b=60),
    )

    # ── Metric cards ──────────────────────────────────────────────────────────
    def _card_html(label, value, unit, color):
        return f"""
        <div style="padding:14px 18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:148px; text-align:center; background:white;
                    box-shadow:0 2px 6px rgba(0,0,0,0.04);">
            <div style="color:#64748b; font-size:0.73rem; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">
                {label}
            </div>
            <div style="font-size:1.5rem; font-weight:800; color:{color}; margin:4px 0 2px 0;">
                {value}
            </div>
            <div style="font-size:0.7rem; color:#94a3b8;">{unit}</div>
        </div>"""

    _tps_color  = (COLORS["GreenLine"]   if _tokens_per_sec2 > 1000
                   else COLORS["OrangeLine"] if _tokens_per_sec2 > 100
                   else COLORS["RedLine"])
    _mem_color  = (COLORS["RedLine"]     if _oom2
                   else COLORS["OrangeLine"] if _total_mem_gb > _device_ram * 0.8
                   else COLORS["GreenLine"])

    _cards_html = f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin:14px 0 8px 0;">
        {_card_html("GEMM",       _gemm_reg2,  f"AI={_gemm_ai2:.0f} F/B",      _gemm_rc2)}
        {_card_html("LayerNorm",  _ln_reg2,    f"AI={_ln_ai_eff:.1f} F/B",     _ln_rc2)}
        {_card_html("Softmax",    _sm_reg2,    f"AI={_sm_ai_eff:.1f} F/B",     _sm_rc2)}
        {_card_html("Throughput", f"{_tokens_per_sec2:,.0f}" if not _oom2 else "OOM",
                                  "tokens / sec", _tps_color)}
        {_card_html("Memory",     f"{_total_mem_gb:.1f}",
                                  f"/ {_device_ram:.0f} GB used", _mem_color)}
    </div>"""

    mo.vstack([
        mo.as_html(_fig2),
        mo.Html(_cards_html),
        _oom_banner,
        mo.md(f"""
**Physics (visible):**

```
Layer Norm AI (unfused) = 5D / (3D x 2) = {_ln_ai_base:.2f} FLOP/byte
Softmax AI   (unfused)  = 5 / (3 x 2)  = {_sm_ai_base:.2f} FLOP/byte
Fusion factor ({_fusion}):  {_ff:.2f}x bytes eliminated per round-trip
Layer Norm AI (fused)   = {_ln_ai_eff:.2f} FLOP/byte
Softmax AI   (fused)    = {_sm_ai_eff:.2f} FLOP/byte

KV-cache = 2 x B x S x heads x d_head x layers x 2 bytes
         = 2 x {_B} x {_S} x 32 x 128 x 32 x 2
         = {_kv_gb:.2f} GB
Weights  = {LLAMA3_PARAMS/1e9:.0f}B params x 2 bytes = {_weights_gb:.1f} GB
Total    = {_total_mem_gb:.1f} GB  |  Limit: {_device_ram:.0f} GB
```
        """),
    ])
    return (
        _oom2, _total_mem_gb, _kv_gb,
        _gemm_ai2, _gemm_reg2,
        _ln_ai_eff, _ln_reg2,
        _sm_ai_eff, _sm_reg2,
        _tokens_per_sec2, _ridge_pt2,
        _fusion,
    )


# ─── ACT II PREDICTION VS REALITY ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, act2_pred,
    _gemm_ai2, _gemm_reg2,
    _ln_reg2, _sm_reg2,
    _ridge_pt2,
):
    _pred2 = act2_pred.value

    _feedback2 = {
        "A": (
            "**Incorrect.** GEMM can be the bottleneck at large batch sizes, but "
            "it is *not always* so. At small batch sizes GEMM arithmetic intensity "
            "is low enough to be memory-bound too. The elementwise ops (softmax, LN) "
            "are *always* memory-bound at any batch size — their AI near 0.8 F/B "
            "never approaches any accelerator's ridge point."
        ),
        "B": (
            "**Correct.** Softmax and layer norm have arithmetic intensities near "
            "0.8 FLOP/byte — permanently below every accelerator's ridge point. "
            "They are always memory-bound. GEMM's AI grows with batch × sequence "
            "length, and can eventually cross the ridge to become compute-bound. "
            "The correct strategy is to fuse the elementwise ops and push GEMM to "
            "larger tiles/batches."
        ),
        "C": (
            "**Incorrect.** The spread is enormous: GEMM at large batches can reach "
            "hundreds of FLOP/byte, while softmax stays at ~0.8 FLOP/byte regardless "
            "of batch size. The difference is how much arithmetic reuse each kernel "
            "can perform on each loaded byte."
        ),
        "D": (
            "**Incorrect.** The complexity of the arithmetic operations is irrelevant "
            "to the bound regime. Layer norm reads each element, applies a few FLOPs, "
            "and writes the result. The ratio of FLOPs to bytes is tiny regardless "
            "of whether one of those FLOPs is a sqrt. Expensive single operations "
            "contribute negligible throughput compared to the memory traffic."
        ),
    }

    _correct2 = _pred2 == "B"

    mo.vstack([
        mo.md("### Prediction vs Reality"),
        mo.callout(
            mo.md(
                f"{_feedback2.get(_pred2, '')}\n\n"
                f"**Actual:** GEMM AI = {_gemm_ai2:.0f} FLOP/byte ({_gemm_reg2}), "
                f"LayerNorm = {_ln_reg2}, Softmax = {_sm_reg2}. "
                f"Ridge = {_ridge_pt2:.0f} FLOP/byte."
            ),
            kind="success" if _correct2 else "warn",
        ),
    ])
    return


# ─── ACT II REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) It reduces the number of FLOPs needed": "A",
            "B) It eliminates redundant memory reads/writes — fused ops load data only once": "B",
            "C) It increases clock frequency by reducing instruction overhead": "C",
            "D) It allows using higher precision without accuracy loss": "D",
        },
        label="Reflection: why does kernel fusion improve performance for memory-bound operations?",
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Act II Reflection"),
        act2_reflect,
    ])
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_reflect):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn")
    )

    _reflect2_feedback = {
        "A": (
            "**Incorrect.** Fusion does not change the number of FLOPs executed. "
            "The same arithmetic still runs. What changes is the *memory traffic* "
            "between operations."
        ),
        "B": (
            "**Correct.** Without fusion, each elementwise kernel writes its output "
            "to HBM and the next kernel reads it back — pure bandwidth round-trips "
            "with almost no arithmetic reuse. A fused kernel keeps the tensor in "
            "on-chip registers or SRAM across all operations, loading each element "
            "once and storing once. This is why FlashAttention achieves 2–4x speedup "
            "over naive attention: it fuses softmax + matrix multiply and avoids "
            "materializing the full O(S^2) attention matrix in HBM."
        ),
        "C": (
            "**Incorrect.** Kernel fusion has no effect on clock frequency. "
            "The hardware runs at the same frequency. The improvement is architectural "
            "— eliminating unnecessary memory round-trips."
        ),
        "D": (
            "**Incorrect.** Precision is orthogonal to fusion. You can fuse FP32 or "
            "FP16 or INT8 ops equally. The benefit of fusion is reducing memory traffic, "
            "not changing the numeric format."
        ),
    }

    _correct2r = act2_reflect.value == "B"
    mo.callout(
        mo.md(_reflect2_feedback[act2_reflect.value]),
        kind="success" if _correct2r else "warn",
    )
    return


# ─── ACT II MATHPEEK ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations": mo.md("""
**Elementwise operation AI (general):**

```
AI_elementwise = FLOPs_per_element / bytes_per_element
               ~ 1 FLOP / 2 bytes  (FP16)
               = 0.5 FLOP/byte
```

Elementwise ops are **always** memory-bound on any modern accelerator.
The ridge point is never below ~10 FLOP/byte; 0.5 < 10.

**Layer normalization AI:**

```
FLOPs per token = 5 x D   (mean, variance, normalize, scale, shift)
Bytes per token = 3 x D x 2   (load input, intermediate, write output)

AI_layernorm = 5D / (3D x 2) ~ 0.83 FLOP/byte
```

**Softmax AI (over attention scores [B, H, S, S]):**

```
FLOPs per element ~ 5   (subtract max, exp, accumulate sum, divide)
Bytes per element = 3 x 2   (load, intermediate, store)

AI_softmax = 5 / (3 x 2) ~ 0.83 FLOP/byte
```

**Kernel fusion benefit:**

Unfused N-op pipeline on tensor T:
```
Memory traffic = N x (read(T) + write(T)) = 2N x |T| bytes
```

Fully fused single-pass kernel:
```
Memory traffic = read(T) + write(T) = 2 x |T| bytes
```

Theoretical speedup (memory-bound) = N, limited by on-chip SRAM capacity.

**KV-cache memory footprint:**

```
KV_cache = 2 x B x S x n_heads x d_head x n_layers x bytes_per_elem
```

For Llama-3-8B at FP16 (n_heads=32, d_head=128, layers=32):

```
KV_cache = B x S x 524288 bytes = B x S x 0.5 MB
```

At B=64, S=2048: KV-cache alone = 64 GB — exceeds Orin NX entirely.
Kernel fusion buys performance only if the design fits in RAM first.
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="background:{COLORS['Surface2']}; border-radius:12px;
                padding:28px 32px; margin:24px 0; border:1px solid {COLORS['Border']};">
        <h2 style="margin:0 0 20px; font-size:1.1rem; font-weight:700;
                   color:{COLORS['Text']};">Key Takeaways</h2>

        <div style="display:flex; flex-direction:column; gap:14px; margin-bottom:24px;">

            <div style="background:#fff; border-radius:8px; padding:16px 20px;
                        border-left:4px solid {COLORS['BlueLine']};
                        border:1px solid {COLORS['Border']}; border-left-width:4px;">
                <div style="font-weight:700; color:{COLORS['Text']}; margin-bottom:4px;">
                    1. Diagnose the regime before optimising the kernel.
                </div>
                <div style="color:{COLORS['TextSec']}; font-size:0.9rem; line-height:1.5;">
                    A transformer attention layer at ~8 FLOP/byte sits far below the
                    H100 ridge point of 295 FLOP/byte, achieving only 2.7% of peak
                    compute. Adding faster arithmetic changes nothing — the fix is
                    always to increase data reuse (more FLOPs per loaded byte), not
                    to add more compute units.
                </div>
            </div>

            <div style="background:#fff; border-radius:8px; padding:16px 20px;
                        border-left:4px solid {COLORS['GreenLine']};
                        border:1px solid {COLORS['Border']}; border-left-width:4px;">
                <div style="font-weight:700; color:{COLORS['Text']}; margin-bottom:4px;">
                    2. Elementwise ops are always memory-bound — fuse them.
                </div>
                <div style="color:{COLORS['TextSec']}; font-size:0.9rem; line-height:1.5;">
                    Layer norm, softmax, ReLU, and every single-pass elementwise
                    operation has an arithmetic intensity below 1 FLOP/byte — well
                    below the ridge point on any silicon built this decade. The only
                    lever is kernel fusion: eliminate the redundant HBM round-trips
                    between consecutive ops. This is why FlashAttention and fused
                    layer norm exist. The specific accelerator will change. This
                    principle will not.
                </div>
            </div>

            <div style="background:#fff; border-radius:8px; padding:16px 20px;
                        border-left:4px solid {COLORS['OrangeLine']};
                        border:1px solid {COLORS['Border']}; border-left-width:4px;">
                <div style="font-weight:700; color:{COLORS['Text']}; margin-bottom:4px;">
                    3. GEMM crosses into compute-bound territory only at large batch sizes.
                </div>
                <div style="color:{COLORS['TextSec']}; font-size:0.9rem; line-height:1.5;">
                    As matrix dimensions grow, arithmetic intensity rises proportionally
                    to the tile edge length. At small batch sizes (batch=1 inference),
                    even GEMM is memory-bound. At large batch sizes (training),
                    GEMM becomes compute-bound and MFU can approach 50–60% on H100.
                    Batch size is a first-order knob for utilization.
                </div>
            </div>

        </div>

        <div style="display:flex; gap:16px; flex-wrap:wrap;">

            <div style="flex:1; min-width:220px; background:#fff; border-radius:8px;
                        padding:16px 20px; border:1px solid {COLORS['Border']};">
                <div style="font-weight:700; color:{COLORS['Text']}; margin-bottom:8px;
                            font-size:0.9rem;">What's Next</div>
                <div style="color:{COLORS['TextSec']}; font-size:0.85rem; line-height:1.5;">
                    <strong>Lab 12 — Serving Systems</strong> applies the Roofline
                    lens to inference serving: how does batch size interact with
                    latency SLAs and throughput targets in a production deployment?
                </div>
            </div>

            <div style="flex:1; min-width:220px; background:#fff; border-radius:8px;
                        padding:16px 20px; border:1px solid {COLORS['Border']};">
                <div style="font-weight:700; color:{COLORS['Text']}; margin-bottom:8px;
                            font-size:0.9rem;">Textbook &amp; TinyTorch</div>
                <div style="color:{COLORS['TextSec']}; font-size:0.85rem; line-height:1.5;">
                    Textbook: <code>@sec-hw-acceleration</code> — Roofline model,
                    arithmetic intensity, MFU, and kernel fusion.<br><br>
                    TinyTorch: <code>tinytorch/src/11_hw_accel/</code> — implement
                    a tiled GEMM kernel and measure achieved vs theoretical bandwidth.
                </div>
            </div>

        </div>
    </div>
    """)
    return


# ─── CELL 20B: SYNTHESIS SELF-ASSESSMENT ──────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Self-Assessment: Can you answer these?": mo.md("""
    1. GPT-2 at batch=1 is 80% parallelizable. Using Amdahl's Law, what is the maximum system speedup achievable regardless of how fast the GPU is — and what serial overhead accounts for the remaining 20%?

    2. On the Roofline plot, the H100's ridge point is approximately 295 FLOP/byte. A workload with arithmetic intensity of 5 FLOP/byte sits where relative to the ridge — and does optimizing FLOPS or bandwidth improve its performance?

    3. ResNet-50 achieves ~20x system speedup on H100 while GPT-2 achieves ~5x on the same chip. Both use identical hardware. Explain the two separate reasons (parallelizable fraction and arithmetic intensity regime) that produce this 4x gap in system speedup.

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
    })
    return


# ─── CELL 21: LEDGER_HUD ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle,
    act1_pred, act2_pred,
    _ai_gemm, _is_mem_bound, _mfu_pct,
    _oom2, _tokens_per_sec2,
    _fusion,
):
    _ctx    = context_toggle.value
    _p1     = act1_pred.value  or "none"
    _p2     = act2_pred.value  or "none"
    _fus    = _fusion          if _fusion else "none"

    _act1_correct = _p1 == "C"
    _act2_correct = _p2 == "B"

    ledger.save(chapter=11, design={
        "context":              _ctx,
        "operation":            "gemm",
        "arithmetic_intensity": float(_ai_gemm),
        "bound_type":           "memory" if _is_mem_bound else "compute",
        "act1_prediction":      _p1,
        "act1_correct":         _act1_correct,
        "act2_result":          float(_mfu_pct),
        "act2_decision":        _fus,
        "constraint_hit":       bool(_oom2),
        "mfu_percent":          float(_mfu_pct),
    })

    _track     = ledger.get_track() or _ctx
    _color_map = {
        "cloud":  COLORS["Cloud"],
        "edge":   COLORS["Edge"],
        "mobile": COLORS["Mobile"],
        "tiny":   COLORS["Tiny"],
        "NONE":   "#475569",
    }
    _hud_color = _color_map.get(_track, "#475569")

    _p1_icon  = "CORRECT" if _act1_correct else "WRONG"
    _p2_icon  = "CORRECT" if _act2_correct else "WRONG"
    _oom_icon = "TRIGGERED" if _oom2 else "CLEAR"
    _tps_str  = f"{_tokens_per_sec2:,.0f}" if not _oom2 else "OOM"

    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="display:flex; gap:22px; align-items:center; padding:12px 24px;
                    background:#0f172a; border-radius:10px; margin-top:16px; flex-wrap:wrap;
                    font-family:'SF Mono','Fira Code',monospace; font-size:0.79rem;
                    border:1px solid #1e293b;">
            <div style="color:#475569; font-weight:600; letter-spacing:0.08em;">
                DESIGN LEDGER
            </div>
            <div>
                <span style="color:#475569;">Context: </span>
                <span style="color:{_hud_color}; font-weight:700;">{_ctx.upper()}</span>
            </div>
            <div>
                <span style="color:#475569;">Chapter: </span>
                <span style="color:#e2e8f0;">11</span>
            </div>
            <div>
                <span style="color:#475569;">AI (GEMM): </span>
                <span style="color:#e2e8f0;">{_ai_gemm:.0f} FLOP/byte</span>
            </div>
            <div>
                <span style="color:#475569;">MFU: </span>
                <span style="color:{'#4ade80' if _mfu_pct > 50 else '#fbbf24'};">
                    {_mfu_pct:.1f}%
                </span>
            </div>
            <div>
                <span style="color:#475569;">Act I: </span>
                <span style="color:{'#4ade80' if _act1_correct else '#f87171'};">
                    {_p1_icon} [{_p1}]
                </span>
            </div>
            <div>
                <span style="color:#475569;">Act II: </span>
                <span style="color:{'#4ade80' if _act2_correct else '#f87171'};">
                    {_p2_icon} [{_p2}]
                </span>
            </div>
            <div>
                <span style="color:#475569;">OOM: </span>
                <span style="color:{'#f87171' if _oom2 else '#4ade80'};">
                    {_oom_icon}
                </span>
            </div>
            <div>
                <span style="color:#475569;">Fusion: </span>
                <span style="color:#e2e8f0;">{_fus}</span>
            </div>
            <div>
                <span style="color:#475569;">Tokens/s: </span>
                <span style="color:#e2e8f0;">{_tps_str}</span>
            </div>
        </div>
        """),

    ])
    return


if __name__ == "__main__":
    app.run()
