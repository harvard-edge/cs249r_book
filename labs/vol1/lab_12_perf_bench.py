import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 12: THE SPEEDUP CEILING
#
# Chapter: Benchmarking (@sec-benchmarking)
# Core Invariants:
#   1. Amdahl's Law: Speedup = 1 / ((1−p) + p/n)
#      The serial fraction, not the number of processors, determines the ceiling.
#   2. Benchmark validity: a benchmark is valid only when it exercises the same
#      bottleneck present in the production workload.
#
# Two deployment contexts: Cloud (H100) vs Edge (Jetson Orin NX).
# 2-Act structure: ~35-40 minutes total.
#
# Act I — The Amdahl Ceiling (12-15 min)
#   Stakeholder: HPC Architect. Training job = 100 hours on 1 GPU.
#   Plan: buy 64 GPUs. CTO expects 64× speedup → 1.56 hours.
#   Profile shows 15% serial (data loading, checkpoint I/O).
#   Prediction: what speedup will we actually get?
#   Instrument: Amdahl explorer with serial fraction + processor count sliders.
#   Show: theoretical vs actual speedup curves for multiple serial fractions.
#   Prediction-vs-reality overlay.
#   MathPeek: Full derivation of Amdahl's Law, Gustafson's Law, efficiency formula.
#
# Act II — Benchmark Validity (20-25 min)
#   Stakeholder: MLPerf Committee. Vendor A=85 synthetic, Vendor B=72 synthetic,
#   but Vendor B is 20% faster in production. Why?
#   Prediction: why does the benchmark lie?
#   Instrument: Benchmark validity analyzer. Configure benchmark type + production
#   workload profile. Show validity ratio. Failure state when batch_size diverges >10×.
#   MathPeek: Validity ratio formula, standard deviation of benchmark scores.
#
# Design Ledger save:
#   chapter=12, context, serial_fraction, num_processors, amdahl_speedup,
#   act1_prediction, act1_correct, act2_result, act2_decision,
#   constraint_hit (validity gap triggered), expected_speedup
# ─────────────────────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP (hide_code=False — leave visible for instructor inspection) ─
@app.cell
def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    # ── Hardware constants (plain floats — no pint, sourced from @sec-benchmarking) ─
    H100_BW_GBS       = 3350.0   # GB/s  — H100 SXM5 HBM3e, NVIDIA spec
    H100_TFLOPS_FP16  = 989.0    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_RAM_GB       = 80.0     # GB    — H100 HBM capacity
    H100_TDP_W        = 700.0    # Watts — H100 TDP

    ORIN_BW_GBS       = 102.0    # GB/s  — Jetson Orin NX LPDDR5, NVIDIA spec
    ORIN_TFLOPS       = 100.0    # TFLOPS — Orin NX INT8-equivalent DLA+GPU
    ORIN_RAM_GB       = 16.0     # GB    — Orin NX unified memory
    ORIN_TDP_W        = 25.0     # Watts — Orin NX TDP

    # ── Act I constants ─────────────────────────────────────────────────────────
    # Training job baseline from the CTO scenario (@sec-benchmarking-endtoend-benchmarks-51bb)
    BASELINE_HOURS    = 100.0    # hours — baseline 1-GPU training time
    GPU_PURCHASE      = 64       # number of GPUs the CTO approved purchasing
    CTO_EXPECTED_SPEEDUP = 64.0  # naive expectation: linear scaling

    # Serial fraction from kernel profiling: data loading + checkpoint I/O
    # This is the fraction that cannot be parallelized across GPUs
    CLOUD_SERIAL_FRAC = 0.15     # 15% serial on H100 (data loading, checkpoint I/O)
    EDGE_SERIAL_FRAC  = 0.25     # 25% serial on Orin NX (sensor preprocessing dominates)

    # Amdahl's Law: Speedup = 1 / (S + P/N) where S=serial, P=parallel, N=processors
    # Cloud: S=0.15, P=0.85, N=64 → Speedup = 1 / (0.15 + 0.85/64) = 1 / 0.1633 ≈ 6.12×
    # Edge:  S=0.25, P=0.75, N=64 → Speedup = 1 / (0.25 + 0.75/64) = 1 / 0.2617 ≈ 3.82×
    CLOUD_AMDAHL_64 = 1.0 / (CLOUD_SERIAL_FRAC + (1.0 - CLOUD_SERIAL_FRAC) / GPU_PURCHASE)
    EDGE_AMDAHL_64  = 1.0 / (EDGE_SERIAL_FRAC  + (1.0 - EDGE_SERIAL_FRAC)  / GPU_PURCHASE)

    # ── Act II constants ─────────────────────────────────────────────────────────
    # Benchmark validity scenario: vendor scores vs production performance
    # Synthetic benchmark favors cache-friendly, fixed batch-size access patterns
    # Production uses variable sequence lengths and random memory access patterns
    VENDOR_A_SYNTHETIC   = 85.0    # synthetic benchmark score (higher is better)
    VENDOR_B_SYNTHETIC   = 72.0    # synthetic benchmark score
    VENDOR_B_PROD_BOOST  = 0.20    # vendor B is 20% faster in production

    # Batch size divergence threshold triggering validity failure
    VALIDITY_BATCH_THRESHOLD = 10  # ratio above which benchmark is invalid

    ledger = DesignLedger()
    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme, go, np, math,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB, H100_TDP_W,
        ORIN_BW_GBS, ORIN_TFLOPS, ORIN_RAM_GB, ORIN_TDP_W,
        BASELINE_HOURS, GPU_PURCHASE, CTO_EXPECTED_SPEEDUP,
        CLOUD_SERIAL_FRAC, EDGE_SERIAL_FRAC,
        CLOUD_AMDAHL_64, EDGE_AMDAHL_64,
        VENDOR_A_SYNTHETIC, VENDOR_B_SYNTHETIC, VENDOR_B_PROD_BOOST,
        VALIDITY_BATCH_THRESHOLD,
    )


# ─── CELL 1: HEADER ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS, CLOUD_AMDAHL_64, GPU_PURCHASE, CTO_EXPECTED_SPEEDUP):
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
                Machine Learning Systems · Volume I · Lab 12
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Speedup Ceiling
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 720px; line-height: 1.65;">
                Your CTO approved {GPU_PURCHASE} GPUs and expects {CTO_EXPECTED_SPEEDUP:.0f}&times; speedup.
                Kernel profiling shows 15% of the job is serial. Amdahl's Law will deliver
                a number far below what was promised — and a benchmark designed for the
                wrong workload made the situation worse.
            </p>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Amdahl&apos;s Law</span>
                <span class="badge badge-info">Benchmark Validity</span>
                <span class="badge badge-info">Serial Fraction</span>
                <span class="badge badge-ok">Cloud (H100)</span>
                <span class="badge badge-fail">Edge (Jetson Orin NX)</span>
                <span class="badge badge-warn">35&ndash;40 min</span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px;">
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
                             font-weight: 700; border: 1px solid rgba(203,32,45,0.25);">
                    Cloud ceiling at 64 GPUs: {CLOUD_AMDAHL_64:.1f}&times;
                </span>
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
                             font-weight: 700; border: 1px solid rgba(99,102,241,0.25);">
                    CTO expects: {CTO_EXPECTED_SPEEDUP:.0f}&times;
                </span>
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the Amdahl ceiling for a 64-GPU training cluster given a measured serial fraction, and predict end-to-end speedup before adding hardware.</strong></div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose why a benchmark score of 85 (Vendor A) correctly predicted the winner on synthetic workloads but the wrong winner in production.</strong></div>
                <div style="margin-bottom: 3px;">3. <strong>Identify the batch-size divergence threshold above which a benchmark loses validity, and compute the resulting production throughput gap.</strong></div>
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
                    Amdahl&apos;s Law formula from @sec-benchmarking-endtoend-benchmarks-51bb &middot;
                    Benchmark validity concepts from @sec-benchmarking-machine-learning-benchmarking-framework-70b8
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35-40 min</strong><br/>
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
                "Your CTO approved 64 GPUs expecting 64x speedup, and the vendor benchmark confirmed the hardware purchase — so why will your training job only run 6x faster, and why did you buy the slower vendor?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete these sections before starting:

    - **@sec-benchmarking-machine-learning-benchmarking-framework-70b8** — The three principles: benchmarks as proxies, Goodhart's Law, end-to-end beats component metrics
    - **@sec-benchmarking-benchmarking-granularity-3855** — Micro vs. macro vs. end-to-end granularity; when each level reveals what
    - **@sec-benchmarking-endtoend-benchmarks-51bb** — Why a 3&times; inference speedup can yield only 1.3&times; end-to-end improvement
    - **@sec-benchmarking-fallacies-pitfalls-9781** — The benchmark trap in practice
    """), kind="info")
    return


# ─── CELL 4: CONTEXT TOGGLE ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={"Cloud (H100)": "cloud", "Edge (Jetson Orin NX)": "edge"},
        value="Cloud (H100)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("### Deployment Context"),
        mo.md(
            "Select the deployment target. The serial fraction differs between cloud and edge, "
            "shifting where Amdahl's Law imposes its ceiling."
        ),
        context_toggle,
    ])
    return (context_toggle,)


# ─── CELL 4: CONTEXT SPEC DISPLAY ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS,
      H100_BW_GBS, H100_RAM_GB, H100_TDP_W,
      ORIN_BW_GBS, ORIN_RAM_GB, ORIN_TDP_W,
      CLOUD_SERIAL_FRAC, EDGE_SERIAL_FRAC):

    _ctx = context_toggle.value
    if _ctx == "cloud":
        _color  = COLORS["Cloud"]
        _name   = "H100 SXM5"
        _bw     = f"{H100_BW_GBS:,.0f} GB/s"
        _ram    = f"{H100_RAM_GB:.0f} GB HBM3e"
        _tdp    = f"{H100_TDP_W:.0f} W"
        _serial = f"{CLOUD_SERIAL_FRAC*100:.0f}%"
        _note   = (
            "Serial fraction = 15% (data loading + checkpoint I/O). "
            "The GPU is fast enough that these CPU-bound operations dominate the non-compute time. "
            "This fraction is the hard ceiling on parallel scaling."
        )
    else:
        _color  = COLORS["Edge"]
        _name   = "Jetson Orin NX"
        _bw     = f"{ORIN_BW_GBS:.0f} GB/s"
        _ram    = f"{ORIN_RAM_GB:.0f} GB unified"
        _tdp    = f"{ORIN_TDP_W:.0f} W"
        _serial = f"{EDGE_SERIAL_FRAC*100:.0f}%"
        _note   = (
            "Serial fraction = 25% (sensor preprocessing + camera decode dominates). "
            "On Orin NX, the DLA/iGPU is slower than H100, so sensor pipelines become a "
            "larger fraction of end-to-end time — the ceiling is even lower."
        )

    mo.Html(f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin:8px 0;">
        <div style="flex:1; min-width:220px; background:white; border:1.5px solid {_color};
                    border-top:4px solid {_color}; border-radius:10px; padding:16px 20px;">
            <div style="font-size:0.72rem; font-weight:700; color:{_color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Hardware
            </div>
            <div style="font-size:1.25rem; font-weight:800; color:#0f172a;">{_name}</div>
            <div style="margin-top:10px; font-size:0.85rem; color:#475569; line-height:1.8;">
                Memory BW: <strong>{_bw}</strong><br>
                RAM: <strong>{_ram}</strong><br>
                TDP: <strong>{_tdp}</strong><br>
                Serial fraction: <strong style="color:{_color};">{_serial}</strong>
            </div>
        </div>
        <div style="flex:3; min-width:300px; background:#f8fafc; border:1px solid #e2e8f0;
                    border-radius:10px; padding:16px 20px; font-size:0.88rem;
                    color:#475569; line-height:1.65; display:flex; align-items:center;">
            {_note}
        </div>
    </div>
    """)
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num = "I"
    _act_color = COLORS["BlueLine"]
    _act_title = "The 64\u00d7 Speedup That Cannot Exist"
    _act_duration = "12\u201315 min"
    _act_why = (
        "You expect 64 GPUs to deliver 64\u00d7 speedup. They will not. A 15% serial fraction "
        "in your data-loading pipeline caps the ceiling at 6.1\u00d7 by mathematical law \u2014 "
        "and the CTO\u2019s $2M hardware purchase cannot change that number."
    )
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
                Act {_act_num} &middot; {_act_duration}</div>
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
    """)
    return


# ─── ACT I: STAKEHOLDER MESSAGE ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, BASELINE_HOURS, GPU_PURCHASE, CTO_EXPECTED_SPEEDUP,
      CLOUD_SERIAL_FRAC):
    _color = COLORS["BlueLine"]
    _bg    = COLORS["BlueLL"]
    _expected_hours = BASELINE_HOURS / CTO_EXPECTED_SPEEDUP
    mo.Html(f"""
    <div style="border-left:4px solid {_color}; background:{_bg};
                border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
        <div style="font-size:0.72rem; font-weight:700; color:{_color};
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
            Incoming Message &middot; HPC Architect
        </div>
        <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
            "We have a training job that runs in {BASELINE_HOURS:.0f} hours on 1 GPU.
            We just bought {GPU_PURCHASE} more GPUs. Our CTO expects
            {CTO_EXPECTED_SPEEDUP:.0f}&times; speedup &mdash; that would bring it down
            to {_expected_hours:.2f} hours. But kernel profiling shows that
            {CLOUD_SERIAL_FRAC*100:.0f}% of runtime is serial: data loading and checkpoint
            I/O that cannot be parallelized. How fast will this actually run?"
        </div>
    </div>
    """)
    return


# ─── ACT I: CONCEPT ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, CLOUD_SERIAL_FRAC, GPU_PURCHASE, CLOUD_AMDAHL_64,
      CTO_EXPECTED_SPEEDUP, BASELINE_HOURS):
    _actual_hours  = BASELINE_HOURS / CLOUD_AMDAHL_64
    _expected_hours = BASELINE_HOURS / CTO_EXPECTED_SPEEDUP
    mo.vstack([
        mo.md("""
        ## Amdahl's Law

        Gene Amdahl proved in 1967 that the speedup achievable by adding more parallel
        resources is bounded by the fraction of work that must remain sequential.
        No matter how many processors you add, the serial fraction imposes a hard ceiling.

        The formula is exact, not approximate:
        """),
        mo.Html(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin:16px 0;">
            <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
                        padding:16px; border-top:4px solid #16a34a; text-align:center;">
                <div style="font-size:0.8rem; color:#64748b; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
                    Serial Fraction
                </div>
                <div style="font-size:1.8rem; font-weight:900; color:#15803d;">
                    {CLOUD_SERIAL_FRAC*100:.0f}%
                </div>
                <div style="font-size:0.78rem; color:#64748b; margin-top:4px;">
                    data loading + checkpoint I/O
                </div>
            </div>
            <div style="background:#eef2ff; border:1px solid #c7d2fe; border-radius:10px;
                        padding:16px; border-top:4px solid #6366f1; text-align:center;">
                <div style="font-size:0.8rem; color:#64748b; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
                    CTO Expects
                </div>
                <div style="font-size:1.8rem; font-weight:900; color:#4338ca;">
                    {CTO_EXPECTED_SPEEDUP:.0f}&times;
                </div>
                <div style="font-size:0.78rem; color:#64748b; margin-top:4px;">
                    = {_expected_hours:.2f} hours
                </div>
            </div>
            <div style="background:#fff7ed; border:1px solid #fed7aa; border-radius:10px;
                        padding:16px; border-top:4px solid #ea580c; text-align:center;">
                <div style="font-size:0.8rem; color:#64748b; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">
                    Amdahl Predicts
                </div>
                <div style="font-size:1.8rem; font-weight:900; color:#c2410c;">
                    {CLOUD_AMDAHL_64:.1f}&times;
                </div>
                <div style="font-size:0.78rem; color:#64748b; margin-top:4px;">
                    = {_actual_hours:.1f} hours
                </div>
            </div>
        </div>
        """),
        mo.callout(mo.md(
            f"**The invariant:** The {CLOUD_SERIAL_FRAC*100:.0f}% serial fraction imposes a maximum speedup "
            f"of 1/{CLOUD_SERIAL_FRAC:.2f} = **{1/CLOUD_SERIAL_FRAC:.1f}&times;**, regardless of how many "
            f"GPUs you add. With {GPU_PURCHASE} GPUs the actual speedup is **{CLOUD_AMDAHL_64:.1f}&times;** "
            f"— not {CTO_EXPECTED_SPEEDUP:.0f}&times;. "
            f"The gap between what was promised and what physics allows is "
            f"**{CTO_EXPECTED_SPEEDUP/CLOUD_AMDAHL_64:.1f}&times;**."
        ), kind="info"),
    ])
    return


# ─── ACT I: PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Your Prediction — Lock In Before Exploring

    *Commit to your answer before touching any slider. The gap between your prediction
    and the physics is where the learning happens.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) ~64&times; speedup &rarr; 1.56 hours (linear scaling, ignores serial fraction)": "A",
            "B) ~54&times; speedup &rarr; 1.85 hours (underestimates the serial bottleneck)":    "B",
            "C) ~6.4&times; speedup &rarr; 15.6 hours (Amdahl with 15% serial fraction)":       "C",
            "D) ~2&times; speedup &rarr; 50 hours (too pessimistic, overcounts the serial cost)": "D",
        },
        label=(
            "A training job runs in 100 hours on 1 GPU. You add 64 GPUs. "
            "Profiling shows 15% of time is serial (data loading + checkpoint I/O). "
            "What is the actual expected runtime?"
        ),
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(mo.md("Select your prediction to continue to the simulator."), kind="warn"),
    )
    return


# ─── ACT I: SIMULATOR CONTROLS ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### The Amdahl Simulator

    Adjust the sliders to explore Amdahl's Law. Move the serial fraction from the
    profiled 15% toward 0% to see how much you would gain by parallelizing data loading.
    Increase the processor count beyond 64 to see the diminishing returns curve flatten.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sim_serial_pct = mo.ui.slider(
        start=0, stop=50, value=15, step=1,
        label="Serial fraction (%)",
        show_value=True,
    )
    sim_num_proc = mo.ui.slider(
        start=1, stop=1024, value=64, step=1,
        label="Number of processors",
        show_value=True,
    )
    sim_efficiency = mo.ui.slider(
        start=70, stop=100, value=90, step=5,
        label="Parallel efficiency (%)",
        show_value=True,
    )
    mo.vstack([
        mo.md("**Amdahl parameters:**"),
        mo.hstack([sim_serial_pct, sim_num_proc, sim_efficiency], gap="2rem"),
    ])
    return (sim_serial_pct, sim_num_proc, sim_efficiency)


# ─── ACT I: AMDAHL CHART + FORMULA CARD ──────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, go, np, apply_plotly_theme, COLORS,
      sim_serial_pct, sim_num_proc, sim_efficiency,
      BASELINE_HOURS, CTO_EXPECTED_SPEEDUP, GPU_PURCHASE,
      CLOUD_SERIAL_FRAC):

    _s    = sim_serial_pct.value / 100.0      # serial fraction
    _p    = 1.0 - _s                          # parallel fraction
    _n    = float(sim_num_proc.value)
    _eff  = sim_efficiency.value / 100.0      # parallel efficiency

    # Theoretical Amdahl speedup (assumes perfect parallel efficiency)
    _speedup_theoretical = 1.0 / (_s + _p / _n)
    # Actual speedup accounting for parallel efficiency
    _speedup_actual = 1.0 / (_s + _p / (_n * _eff))
    # Theoretical maximum (infinite processors)
    _max_speedup = 1.0 / _s if _s > 0 else float("inf")

    # Curves over processor count 1..1024
    _proc_range = np.logspace(0, 3, 300)  # 1 to 1000

    def _amdahl(s, p, n_arr, eff):
        return 1.0 / (s + p / (n_arr * eff))

    _curve_theoretical = _amdahl(_s, _p, _proc_range, 1.0)
    _curve_actual      = _amdahl(_s, _p, _proc_range, _eff)

    # Additional reference curves for the canonical serial fractions
    _reference_serials = [0.05, 0.15, 0.25, 0.50]
    _ref_labels        = ["5% serial", "15% serial (profiled)", "25% serial", "50% serial"]
    _ref_colors        = [COLORS["GreenLine"], COLORS["BlueLine"],
                          COLORS["OrangeLine"], COLORS["RedLine"]]

    _fig = go.Figure()

    # Reference curves (grey/subtle)
    for _ref_s, _ref_label, _ref_c in zip(_reference_serials, _ref_labels, _ref_colors):
        _ref_p = 1.0 - _ref_s
        _ref_curve = 1.0 / (_ref_s + _ref_p / _proc_range)
        _fig.add_trace(go.Scatter(
            x=_proc_range,
            y=_ref_curve,
            mode="lines",
            name=_ref_label,
            line=dict(color=_ref_c, width=1.5, dash="dot"),
            opacity=0.55,
        ))

    # Theoretical curve for current selection
    _fig.add_trace(go.Scatter(
        x=_proc_range,
        y=_curve_theoretical,
        mode="lines",
        name=f"Theoretical ({sim_serial_pct.value}% serial, 100% efficiency)",
        line=dict(color=COLORS["BlueLine"], width=2.5),
    ))

    # Actual curve (with efficiency)
    if _eff < 1.0:
        _fig.add_trace(go.Scatter(
            x=_proc_range,
            y=_curve_actual,
            mode="lines",
            name=f"Actual ({sim_serial_pct.value}% serial, {sim_efficiency.value}% efficiency)",
            line=dict(color=COLORS["OrangeLine"], width=2.5, dash="dash"),
        ))

    # CTO expectation line
    _fig.add_hline(
        y=CTO_EXPECTED_SPEEDUP,
        line_dash="longdash",
        line_color=COLORS["RedLine"],
        annotation_text=f"CTO expects: {CTO_EXPECTED_SPEEDUP:.0f}×",
        annotation_position="top left",
        annotation_font_color=COLORS["RedLine"],
        annotation_font_size=11,
    )

    # Amdahl ceiling for current serial fraction
    if _s > 0:
        _fig.add_hline(
            y=_max_speedup,
            line_dash="dot",
            line_color=COLORS["GreenLine"],
            annotation_text=f"Ceiling = {_max_speedup:.1f}× (1/{_s:.2f})",
            annotation_position="bottom right",
            annotation_font_color=COLORS["GreenLine"],
            annotation_font_size=11,
        )

    # Marker at selected processor count
    _marker_color = (
        COLORS["GreenLine"]  if _speedup_theoretical >= 10.0
        else COLORS["OrangeLine"] if _speedup_theoretical >= 4.0
        else COLORS["RedLine"]
    )
    _fig.add_trace(go.Scatter(
        x=[_n],
        y=[_speedup_theoretical],
        mode="markers+text",
        name=f"Your selection: {_n:.0f} procs",
        marker=dict(color=_marker_color, size=14, symbol="diamond",
                    line=dict(color="white", width=2)),
        text=[f"  {_speedup_theoretical:.1f}×"],
        textposition="middle right",
        textfont=dict(size=13, color=_marker_color, family="monospace"),
        showlegend=True,
    ))

    _fig.update_layout(
        title=dict(
            text=f"Amdahl Speedup Curve — {sim_serial_pct.value}% serial fraction",
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Number of processors",
            type="log",
            tickvals=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            ticktext=["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"],
        ),
        yaxis_title="Speedup (×)",
        height=380,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.92)",
                    font=dict(size=11)),
    )
    apply_plotly_theme(_fig)

    # Formula card
    _hours_at_speedup = BASELINE_HOURS / _speedup_theoretical
    _speedup_color = (
        COLORS["GreenLine"]  if _speedup_theoretical >= 10.0
        else COLORS["OrangeLine"] if _speedup_theoretical >= 4.0
        else COLORS["RedLine"]
    )

    mo.vstack([
        mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0 8px 0;">
            <div style="flex:1; min-width:150px; background:white; border:1px solid #e2e8f0;
                        border-radius:10px; padding:16px 18px; text-align:center;">
                <div style="font-size:0.72rem; color:#94a3b8; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">Speedup</div>
                <div style="font-size:2.2rem; font-weight:900; color:{_speedup_color};
                            font-family:monospace; margin-top:4px;">{_speedup_theoretical:.2f}&times;</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">theoretical</div>
            </div>
            <div style="flex:1; min-width:150px; background:white; border:1px solid #e2e8f0;
                        border-radius:10px; padding:16px 18px; text-align:center;">
                <div style="font-size:0.72rem; color:#94a3b8; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">Actual</div>
                <div style="font-size:2.2rem; font-weight:900; color:{COLORS['OrangeLine']};
                            font-family:monospace; margin-top:4px;">{_speedup_actual:.2f}&times;</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">at {sim_efficiency.value}% efficiency</div>
            </div>
            <div style="flex:1; min-width:150px; background:white; border:1px solid #e2e8f0;
                        border-radius:10px; padding:16px 18px; text-align:center;">
                <div style="font-size:0.72rem; color:#94a3b8; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">Ceiling</div>
                <div style="font-size:2.2rem; font-weight:900; color:{COLORS['GreenLine']};
                            font-family:monospace; margin-top:4px;">
                    {"&infin;" if _s == 0 else f"{_max_speedup:.1f}&times;"}
                </div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">max possible</div>
            </div>
            <div style="flex:2; min-width:280px; background:#f8fafc; border:1px solid #e2e8f0;
                        border-radius:10px; padding:16px 18px; font-family:monospace;
                        font-size:0.83rem; color:#0f172a; line-height:1.9;">
                <strong>Amdahl&apos;s Law:</strong><br>
                Speedup = 1 / (S + P/N)<br>
                &nbsp;&nbsp;= 1 / ({_s:.2f} + {_p:.2f}/{_n:.0f})<br>
                &nbsp;&nbsp;= 1 / ({_s:.2f} + {_p/_n:.4f})<br>
                &nbsp;&nbsp;= <strong>{_speedup_theoretical:.4f}&times;</strong><br>
                New runtime = {BASELINE_HOURS:.0f}h / {_speedup_theoretical:.2f} = <strong>{_hours_at_speedup:.1f}h</strong>
            </div>
        </div>
        """),
        mo.as_html(_fig),
    ])
    return (_speedup_theoretical,)


# ─── ACT I: MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation — Amdahl's Law, Gustafson's Law, and efficiency": mo.md("""
        **Amdahl's Law** (1967) — upper bound on speedup with fixed workload size

        `Speedup = 1 / (S + P/N)`

        - **S** — serial fraction (cannot be parallelized; 0.0–1.0)
        - **P** — parallel fraction (P = 1 − S)
        - **N** — number of processors (or speedup factor on the parallel part)

        **Maximum speedup** (limit as N → ∞):

        `Speedup_max = 1 / S`

        For S = 0.15: `Speedup_max = 1/0.15 = 6.67×` regardless of how many GPUs you add.

        **Example from this lab** (profiled: S=0.15, P=0.85, N=64):

        `Speedup = 1 / (0.15 + 0.85/64) = 1 / (0.15 + 0.01328) = 1 / 0.16328 ≈ 6.12×`

        **Efficiency formula** — fraction of peak speedup actually achieved:

        `E = Speedup / N`

        At 6.12× speedup with 64 GPUs: `E = 6.12/64 ≈ 9.6%` — we are using 9.6% of the theoretical maximum.

        ---

        **Gustafson's Law** (1988) — a different framing for scaled workloads

        Amdahl assumes fixed problem size. Gustafson observed that in practice,
        engineers scale the problem size with the number of processors:

        `Scaled Speedup = N − S × (N − 1)`

        For S=0.15, N=64: `Scaled Speedup = 64 − 0.15 × 63 = 64 − 9.45 = 54.55×`

        Gustafson's Law is more optimistic because it asks: if we can make the problem
        64× larger in the same wall-clock time, what speedup did we achieve on the
        *scaled* problem? The answer is much better — but only if your workload
        naturally scales (training on more data, not just training faster on the same data).

        **The key insight:** Amdahl bounds you when problem size is fixed (inference latency,
        single-model training). Gustafson applies when you scale the problem (larger dataset,
        bigger model). Both are correct — they answer different questions.
        """)
    })
    return


# ─── ACT I: PREDICTION-VS-REALITY REVEAL ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, CLOUD_SERIAL_FRAC, GPU_PURCHASE,
      CLOUD_AMDAHL_64, BASELINE_HOURS, CTO_EXPECTED_SPEEDUP):

    _actual_speedup = CLOUD_AMDAHL_64
    _actual_hours   = BASELINE_HOURS / _actual_speedup
    _expected_hours = BASELINE_HOURS / CTO_EXPECTED_SPEEDUP

    _correct = act1_pred.value == "C"

    _pred_label = {
        "A": f"~{CTO_EXPECTED_SPEEDUP:.0f}× speedup (linear scaling)",
        "B": "~54× speedup (underestimates serial bottleneck)",
        "C": f"~{_actual_speedup:.1f}× speedup (Amdahl-correct)",
        "D": "~2× speedup (too pessimistic)",
    }[act1_pred.value]

    _pred_val = {"A": 64.0, "B": 54.0, "C": _actual_speedup, "D": 2.0}[act1_pred.value]
    _pred_hours = BASELINE_HOURS / _pred_val

    if _correct:
        mo.callout(mo.md(
            f"**Correct.** You predicted option C: {_pred_label}. "
            f"Amdahl's Law with S=0.15, P=0.85, N=64: "
            f"Speedup = 1/(0.15 + 0.85/64) = 1/0.1633 = **{_actual_speedup:.2f}×**. "
            f"The job completes in **{_actual_hours:.1f} hours**, not "
            f"{_expected_hours:.2f} hours as the CTO expected. "
            f"The {CLOUD_SERIAL_FRAC*100:.0f}% serial fraction "
            f"sets a hard ceiling at {1/CLOUD_SERIAL_FRAC:.1f}×, and "
            f"{GPU_PURCHASE} GPUs only close {(_actual_speedup/(1/CLOUD_SERIAL_FRAC))*100:.0f}% of that gap."
        ), kind="success")
    else:
        _ratio = abs(_pred_val - _actual_speedup) / _actual_speedup
        mo.callout(mo.md(
            f"**Not quite.** You predicted option {act1_pred.value}: {_pred_label} "
            f"({_pred_hours:.1f} hours). "
            f"The Amdahl-correct answer is option C: **{_actual_speedup:.2f}×** "
            f"({_actual_hours:.1f} hours). "
            f"Your prediction was off by {_ratio*100:.0f}% of the correct speedup. "
            f"**The arithmetic:** Speedup = 1/(S + P/N) = "
            f"1/(0.15 + 0.85/64) = 1/0.1633 = {_actual_speedup:.2f}×. "
            f"Option A (linear) ignores the serial fraction entirely. "
            f"Option B (54×) is what Gustafson's Law would predict for a *scaled* workload — "
            f"a different question. Option D (2×) grossly over-weights the serial cost."
        ), kind="warn")
    return


# ─── ACT I: REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Reflection — The Most Impactful Optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) Add more processors beyond 64 &mdash; the serial fraction is still fine.": "A",
            "B) Use a faster GPU interconnect &mdash; NVLink instead of PCIe.":           "B",
            "C) Reduce the serial fraction &mdash; parallelize data loading and async checkpointing.": "C",
            "D) Use lower precision (FP16) to reduce compute time per step.":             "D",
        },
        label=(
            "The CTO now wants to approach linear scaling (close to 64×). "
            "What is the most impactful single optimization?"
        ),
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn"),
    )
    return


@app.cell(hide_code=True)
def _(mo, act1_reflect, CLOUD_SERIAL_FRAC, GPU_PURCHASE):
    _correct = act1_reflect.value == "C"
    _new_s   = 0.02   # if you parallelize data loading, serial drops to ~2%
    _new_speedup = 1.0 / (_new_s + (1.0 - _new_s) / GPU_PURCHASE)

    if _correct:
        mo.callout(mo.md(
            f"**Correct.** The serial fraction S is the denominator of the ceiling formula: "
            f"1/S. Reducing S from {CLOUD_SERIAL_FRAC*100:.0f}% to ~2% (async prefetch + "
            f"async checkpointing) shifts the ceiling from "
            f"{1/CLOUD_SERIAL_FRAC:.1f}× to {1/_new_s:.0f}×. "
            f"At N={GPU_PURCHASE} GPUs with S={_new_s:.2f}: "
            f"Speedup = 1/({_new_s:.2f} + {(1-_new_s):.2f}/{GPU_PURCHASE}) = "
            f"**{_new_speedup:.1f}×**. "
            f"Adding more GPUs (A) cannot exceed a ceiling that is determined by S, not N. "
            f"Faster interconnect (B) reduces communication overhead in the parallel part, "
            f"not the serial part. Lower precision (D) speeds up compute in the parallel part "
            f"but leaves the serial fraction unchanged."
        ), kind="success")
    else:
        _explanations = {
            "A": (
                f"Adding more processors beyond {GPU_PURCHASE} approaches the ceiling "
                f"1/S = 1/{CLOUD_SERIAL_FRAC:.2f} = {1/CLOUD_SERIAL_FRAC:.1f}× asymptotically. "
                f"With {GPU_PURCHASE} processors you are already at "
                f"{1.0/(CLOUD_SERIAL_FRAC + (1-CLOUD_SERIAL_FRAC)/GPU_PURCHASE)/((1/CLOUD_SERIAL_FRAC))*100:.0f}% "
                f"of the ceiling. More GPUs give diminishing returns. The ceiling itself must move."
            ),
            "B": (
                "NVLink reduces communication latency in the parallel portion of the job. "
                "This improves the *effective* parallel efficiency — similar to increasing "
                "the efficiency slider above. It does not change the serial fraction. "
                f"The ceiling remains at {1/CLOUD_SERIAL_FRAC:.1f}×."
            ),
            "D": (
                "Lower precision reduces FLOP time on the compute-intensive forward/backward pass. "
                "This speeds up the parallel portion P — equivalent to increasing N in Amdahl's formula. "
                f"But the serial fraction S remains {CLOUD_SERIAL_FRAC*100:.0f}%, so the ceiling "
                f"stays at {1/CLOUD_SERIAL_FRAC:.1f}×. You are optimizing the parallel path "
                "when the serial path is the constraint."
            ),
        }
        _msg = _explanations.get(act1_reflect.value, "")
        mo.callout(mo.md(
            f"**Not quite — you selected option {act1_reflect.value}.** {_msg} "
            f"**The correct answer is C:** reduce the serial fraction directly. "
            f"If async data loading and async checkpointing cut S to ~2%, "
            f"the speedup ceiling rises from {1/CLOUD_SERIAL_FRAC:.1f}× to {1/_new_s:.0f}×, "
            f"and the actual speedup at {GPU_PURCHASE} GPUs reaches {_new_speedup:.1f}×."
        ), kind="warn")
    return


# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num = "II"
    _act_color = COLORS["OrangeLine"]
    _act_title = "The Benchmark That Ranked the Wrong Vendor First"
    _act_duration = "20\u201325 min"
    _act_why = (
        "Act I showed that the serial fraction caps parallel scaling. Now discover a second "
        "failure mode: the benchmark that measured the right thing on the wrong workload "
        "\u2014 ranking Vendor A first on synthetic tests while Vendor B is 20% faster "
        "in the production traffic that actually matters."
    )
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
                Act {_act_num} &middot; {_act_duration}</div>
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
    """)
    return


# ─── ACT II: STAKEHOLDER MESSAGE ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, VENDOR_A_SYNTHETIC, VENDOR_B_SYNTHETIC, VENDOR_B_PROD_BOOST):
    _color = COLORS["OrangeLine"]
    _bg    = COLORS["OrangeLL"]
    _a_score = VENDOR_A_SYNTHETIC
    _b_score = VENDOR_B_SYNTHETIC
    _boost   = VENDOR_B_PROD_BOOST * 100
    mo.Html(f"""
    <div style="border-left:4px solid {_color}; background:{_bg};
                border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
        <div style="font-size:0.72rem; font-weight:700; color:{_color};
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
            Incoming Message &middot; MLPerf Committee Chair
        </div>
        <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
            "We evaluated three vendors. Vendor A scores {_a_score:.0f} on our synthetic
            benchmark. Vendor B scores {_b_score:.0f}. We ranked Vendor A the winner and
            recommended it to procurement. But after deployment, teams running production
            workloads report that Vendor B is actually {_boost:.0f}% faster in practice.
            The committee is under fire. How did our benchmark get the ranking wrong?"
        </div>
    </div>
    """)
    return


# ─── ACT II: CONCEPT ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("""
        ## Benchmark Validity

        A benchmark is a proxy for production behavior. The fundamental question is:
        does the benchmark exercise the **same bottleneck** as the production workload?

        When a synthetic benchmark uses fixed, cache-friendly batch sizes with sequential
        memory access, it measures peak throughput on well-structured data — but production
        inference runs variable-length sequences with random memory access patterns that
        bust the cache and expose memory latency the benchmark never tests.

        Vendor A's microarchitecture is optimized for cache-resident, uniform-batch
        computation. Vendor B's microarchitecture has higher memory-latency tolerance
        and better random-access patterns. The synthetic benchmark systematically favors A.
        Production systematically favors B. The benchmark was reproducible and fair.
        It was not valid.

        The **validity ratio** quantifies how accurately a benchmark predicts production:
        """),
        mo.callout(mo.md(
            "**Validity Ratio = Benchmark-predicted rank correlation with production rank.** "
            "A ratio of 1.0 means the benchmark perfectly predicts the production ranking. "
            "A ratio of 0 means the benchmark is uncorrelated with production. "
            "A negative ratio means the benchmark ranks systems in the *reverse* order of "
            "production performance — worse than random."
        ), kind="info"),
    ])
    return


# ─── ACT II: PREDICTION LOCK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Before You Analyze — Lock In Your Hypothesis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) The benchmark has measurement errors — statistical noise in the timing.": "A",
            "B) The synthetic workload uses cache-friendly access patterns that production does not exhibit — the benchmark exercises the wrong bottleneck.": "B",
            "C) Vendor A uses a faster language runtime for the benchmark harness.": "C",
            "D) The benchmark uses a larger model than production, favoring Vendor A's larger cache.": "D",
        },
        label=(
            "Vendor A scores higher on the synthetic benchmark but is 20% slower in "
            "production. The most likely explanation is:"
        ),
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(
            mo.md("Select your hypothesis to unlock the benchmark validity analyzer."),
            kind="warn",
        ),
    )
    return


# ─── ACT II: SIMULATOR CONTROLS ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Benchmark Validity Analyzer

    Configure the benchmark and production workload parameters below. The simulator
    computes a **validity score** measuring how well the benchmark predicts production.
    Watch what happens when batch sizes diverge or memory access patterns diverge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bench_type = mo.ui.dropdown(
        options={
            "MLPerf Inference (standard)":    "mlperf",
            "Synthetic matrix multiply":       "matmul",
            "In-house microbenchmark":         "micro",
        },
        value="MLPerf Inference (standard)",
        label="Benchmark type",
    )
    bench_batch_size = mo.ui.slider(
        start=1, stop=512, value=32, step=1,
        label="Benchmark batch size",
        show_value=True,
    )
    prod_batch_size = mo.ui.slider(
        start=1, stop=512, value=1, step=1,
        label="Production batch size",
        show_value=True,
    )
    mem_access_divergence = mo.ui.slider(
        start=0, stop=100, value=60, step=10,
        label="Memory access pattern divergence (0=identical, 100=completely different)",
        show_value=True,
    )
    mo.vstack([
        mo.hstack([bench_type, bench_batch_size], gap="2rem"),
        mo.hstack([prod_batch_size, mem_access_divergence], gap="2rem"),
    ])
    return (bench_type, bench_batch_size, prod_batch_size, mem_access_divergence)


# ─── ACT II: PHYSICS ENGINE + VALIDITY CHART ──────────────────────────────────
@app.cell(hide_code=True)
def _(mo, go, np, apply_plotly_theme, COLORS, context_toggle,
      bench_type, bench_batch_size, prod_batch_size, mem_access_divergence,
      VENDOR_A_SYNTHETIC, VENDOR_B_SYNTHETIC, VENDOR_B_PROD_BOOST,
      VALIDITY_BATCH_THRESHOLD):

    _ctx = context_toggle.value

    # ── Benchmark type base validity ─────────────────────────────────────────
    # MLPerf has the most rigorous workload coverage; synthetic matmul the least
    _bench_base_validity = {
        "mlperf": 0.80,   # MLPerf exercises realistic inference workloads
        "matmul": 0.45,   # synthetic matmul: optimized for cache-resident data
        "micro":  0.30,   # microbenchmark: measures peak, not realistic
    }[bench_type.value]

    _bench_bs   = bench_batch_size.value
    _prod_bs    = prod_batch_size.value
    _mem_div    = mem_access_divergence.value / 100.0
    _bs_ratio   = max(_bench_bs, _prod_bs) / max(min(_bench_bs, _prod_bs), 1)

    # ── Batch size penalty ───────────────────────────────────────────────────
    # Validity decays logarithmically as batch sizes diverge.
    # Beyond VALIDITY_BATCH_THRESHOLD× divergence, the benchmark is invalid.
    # Source: @sec-benchmarking-benchmarking-granularity-3855
    _bs_penalty = min(np.log10(max(_bs_ratio, 1)) / np.log10(VALIDITY_BATCH_THRESHOLD), 1.0)
    _bs_validity_factor = 1.0 - 0.6 * _bs_penalty  # max 60% penalty from batch divergence

    # ── Memory access penalty ────────────────────────────────────────────────
    # Cache-friendly synthetic → random production = worst case
    _mem_validity_factor = 1.0 - 0.4 * _mem_div  # max 40% penalty from access pattern

    # ── Combined validity score (0.0 = invalid, 1.0 = perfect) ──────────────
    _validity_score = _bench_base_validity * _bs_validity_factor * _mem_validity_factor
    _validity_score = max(0.0, min(1.0, _validity_score))

    # ── Simulate predicted vs actual vendor ranking ──────────────────────────
    # Vendor B's true production advantage over Vendor A
    _b_prod_score = VENDOR_A_SYNTHETIC * (1.0 + VENDOR_B_PROD_BOOST)

    # Benchmark prediction: validity score interpolates between synthetic and true
    # At validity=1.0: prediction = production truth
    # At validity=0.0: prediction = random (here: pure synthetic score)
    _a_bench_pred = VENDOR_A_SYNTHETIC
    _b_bench_pred = VENDOR_B_SYNTHETIC + (
        (VENDOR_B_SYNTHETIC - _b_prod_score) * (1.0 - _validity_score)
    )

    # Production truth
    _a_prod_true = VENDOR_A_SYNTHETIC * (1.0 - VENDOR_B_PROD_BOOST / 2.0)
    _b_prod_true = _b_prod_score

    # Validity ratio: does benchmark correctly rank B above A?
    _correct_rank = _b_prod_true > _a_prod_true   # always True in this scenario
    _bench_correct_rank = _b_bench_pred > _a_bench_pred
    _rank_validity = 1.0 if _bench_correct_rank else -1.0

    # ── Chart: benchmark score vs production throughput ──────────────────────
    _vendors = ["Vendor A", "Vendor B"]
    _bench_scores  = [_a_bench_pred, _b_bench_pred]
    _prod_scores   = [_a_prod_true,  _b_prod_true]

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        name="Benchmark score",
        x=_vendors,
        y=_bench_scores,
        marker_color=[COLORS["BlueLine"] + "AA", COLORS["BlueLine"] + "AA"],
        marker_line_color=[COLORS["BlueLine"], COLORS["BlueLine"]],
        marker_line_width=2,
        text=[f"{v:.1f}" for v in _bench_scores],
        textposition="outside",
    ))
    _fig.add_trace(go.Bar(
        name="Production throughput (normalized)",
        x=_vendors,
        y=_prod_scores,
        marker_color=[COLORS["GreenLine"] + "AA", COLORS["GreenLine"] + "AA"],
        marker_line_color=[COLORS["GreenLine"], COLORS["GreenLine"]],
        marker_line_width=2,
        text=[f"{v:.1f}" for v in _prod_scores],
        textposition="outside",
    ))
    _fig.update_layout(
        barmode="group",
        title=dict(
            text=f"Benchmark vs Production — Validity Score: {_validity_score:.2f}",
            font=dict(size=14),
        ),
        xaxis_title="Vendor",
        yaxis_title="Score (higher = better)",
        height=300,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)"),
    )
    apply_plotly_theme(_fig)

    # ── Validity score color ──────────────────────────────────────────────────
    _val_color = (
        COLORS["GreenLine"]   if _validity_score >= 0.70
        else COLORS["OrangeLine"] if _validity_score >= 0.45
        else COLORS["RedLine"]
    )

    _batch_divergence_label = f"{_bs_ratio:.0f}×"
    _rank_label = "CORRECT" if _bench_correct_rank else "WRONG"
    _rank_color = COLORS["GreenLine"] if _bench_correct_rank else COLORS["RedLine"]

    # ── Failure state: batch divergence beyond threshold ───────────────────────
    _failure = _bs_ratio > VALIDITY_BATCH_THRESHOLD

    mo.vstack([
        mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0 8px 0;">
            <div style="flex:1; min-width:150px; background:white; border:1px solid #e2e8f0;
                        border-radius:10px; padding:16px 18px; text-align:center;">
                <div style="font-size:0.72rem; color:#94a3b8; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">Validity Score</div>
                <div style="font-size:2.2rem; font-weight:900; color:{_val_color};
                            font-family:monospace; margin-top:4px;">{_validity_score:.2f}</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">0=invalid · 1=perfect</div>
            </div>
            <div style="flex:1; min-width:150px; background:white; border:1px solid #e2e8f0;
                        border-radius:10px; padding:16px 18px; text-align:center;">
                <div style="font-size:0.72rem; color:#94a3b8; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">Batch Divergence</div>
                <div style="font-size:2.2rem; font-weight:900;
                            color:{"#CB202D" if _failure else "#0f172a"};
                            font-family:monospace; margin-top:4px;">{_batch_divergence_label}</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">
                    bench={_bench_bs} vs prod={_prod_bs}
                </div>
            </div>
            <div style="flex:1; min-width:150px; background:white; border:1px solid #e2e8f0;
                        border-radius:10px; padding:16px 18px; text-align:center;">
                <div style="font-size:0.72rem; color:#94a3b8; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">Rank Prediction</div>
                <div style="font-size:1.6rem; font-weight:900; color:{_rank_color};
                            font-family:monospace; margin-top:4px;">{_rank_label}</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">
                    B &gt; A in production: {"YES" if _correct_rank else "NO"}
                </div>
            </div>
            <div style="flex:2; min-width:280px; background:#f8fafc; border:1px solid #e2e8f0;
                        border-radius:10px; padding:16px 18px; font-size:0.83rem;
                        color:#0f172a; line-height:1.8;">
                <strong>Validity decomposition:</strong><br>
                Base ({bench_type.value}): {_bench_base_validity:.2f}<br>
                &times; batch penalty factor: {_bs_validity_factor:.2f}<br>
                &times; memory pattern factor: {_mem_validity_factor:.2f}<br>
                = validity score: <strong>{_validity_score:.3f}</strong>
            </div>
        </div>
        """),
        mo.as_html(_fig),
    ])
    return (_validity_score, _failure, _bench_bs, _prod_bs, _bs_ratio,
            _bench_correct_rank, _val_color, _rank_label, _rank_color)


# ─── ACT II: FAILURE STATE ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, _failure, _bench_bs, _prod_bs, _bs_ratio,
      VALIDITY_BATCH_THRESHOLD, _validity_score):

    if _failure:
        mo.callout(mo.md(
            f"**Benchmark validity compromised.** "
            f"Benchmark batch = {_bench_bs}, production batch = {_prod_bs}. "
            f"Divergence ratio = {_bs_ratio:.0f}&times; exceeds the validity threshold "
            f"of {VALIDITY_BATCH_THRESHOLD}&times;. "
            f"**Validity score = {_validity_score:.2f}** — this benchmark cannot be used "
            f"to predict production ranking. "
            f"At this batch divergence, cache occupancy patterns, memory bandwidth "
            f"utilization, and arithmetic intensity all differ structurally between "
            f"benchmark and production. The measurement is technically correct; "
            f"the benchmark is measuring the wrong workload."
        ), kind="danger")
    else:
        mo.md("")
    return


# ─── ACT II: CONTEXT COMPARISON ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Toggle the context (Cloud vs. Edge) and observe the production batch size impact:**

    On **H100 (Cloud):** Production inference typically runs with batch sizes of 1–8
    for latency-sensitive APIs. Benchmarks run at batch=32–512 to maximize throughput
    on the chart. The divergence is structural. A benchmark at batch=64 on an H100
    measuring matrix throughput tells you nothing about batch=1 API latency.

    On **Orin NX (Edge):** Production runs at batch=1 (real-time sensor stream,
    one frame at a time). Benchmarks that run at batch=16+ on Orin NX are exercising
    the DLA throughput mode — a mode that production never uses. The gap is even worse.

    This is the core lesson from @sec-benchmarking-endtoend-benchmarks-51bb: the benchmark
    must match the production batch size, memory access pattern, and thermal state.
    A benchmark that does not is not wrong — it is answering a different question.
    """), kind="info")
    return


# ─── ACT II: MATHPEEK ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation — Benchmark validity ratio and the Bedchamber Effect": mo.md("""
        **Benchmark Validity Ratio**

        The validity ratio V measures how well a benchmark predicts production ranking:

        `V = Spearman rank correlation(benchmark_scores, production_scores)`

        - V = 1.0: benchmark perfectly predicts production ranking
        - V = 0.0: benchmark is uncorrelated with production
        - V < 0.0: benchmark ranks systems in the opposite order of production

        **Validity decomposition** (simplified multiplicative model):

        `V = V_base × V_batch × V_memory × V_thermal`

        Where each factor captures one axis of workload divergence:
        - **V_base**: inherent validity of the benchmark design (MLPerf > synthetic > micro)
        - **V_batch**: penalty from batch size divergence (1× = 1.0, 100× = ~0.4)
        - **V_memory**: penalty from access pattern divergence (sequential → random)
        - **V_thermal**: penalty from thermal condition divergence (cold → sustained-load)

        **Standard deviation of benchmark scores**

        A benchmark suite run k times on the same system has variance:
        `σ² = σ²_measurement + σ²_thermal + σ²_workload_variation`

        If σ_total > |score_A - score_B| / 2, the ranking is within noise and cannot
        distinguish the two systems, regardless of validity.

        ---

        **The Bedchamber Effect** (workload mismatch → invalid ranking)

        Named after the historical practice of judging a kingdom's wealth by its reception
        chambers rather than its treasury, the Bedchamber Effect describes how optimizing
        a system for the visible (benchmark) measure causes it to underperform on the true
        (production) measure.

        In ML systems: Vendor A optimizes its microarchitecture for MLPerf benchmark patterns.
        Vendor B optimizes for the production workload. The benchmark correctly reports that
        Vendor A is faster on the benchmark. The benchmark incorrectly implies that Vendor A
        is faster in production. Goodhart's Law applied to hardware evaluation.
        """)
    })
    return


# ─── ACT II: REFLECTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Reflection — The Gold Standard
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) Run the exact MLPerf benchmark suite on all competing systems.":
                "A",
            "B) Measure production workloads on actual production traffic with P99 latency.":
                "B",
            "C) Use the vendor&apos;s own benchmark tools — they know their hardware best.":
                "C",
            "D) Average multiple independent synthetic benchmarks.":
                "D",
        },
        label="What is the gold standard for validating a benchmark?",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_reflect):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn"),
    )
    return


@app.cell(hide_code=True)
def _(mo, act2_reflect, act2_pred):
    _reflect_correct = act2_reflect.value == "B"
    _hypo_correct    = act2_pred.value == "B"

    if _reflect_correct:
        mo.callout(mo.md(
            "**Correct.** The gold standard is production traffic with P99 latency. "
            "This is the only measurement that exercises exactly the same workload, "
            "batch size, memory access pattern, thermal state, and request distribution "
            "that the system will face in deployment. "
            "MLPerf (A) is a rigorous standardized benchmark — it improves on synthetic "
            "benchmarks significantly, but it is still a proxy. A system that scores well "
            "on MLPerf Inference may still underperform on your specific model and traffic pattern. "
            "Vendor benchmarks (C) have obvious incentive misalignment. "
            "Averaging synthetic benchmarks (D) averages invalid results — the average of "
            "five measurements that all miss the production bottleneck still misses it. "
            "From @sec-benchmarking-endtoend-benchmarks-51bb: only the production workload "
            "itself is guaranteed to exercise the production bottleneck."
        ), kind="success")
    else:
        _options = {
            "A": (
                "MLPerf is far better than purely synthetic benchmarks — it uses standardized "
                "real models and reproducible run rules. But it is still a proxy. "
                "The MLPerf Inference benchmark may use a different batch size, model version, "
                "or input distribution than your production traffic. "
                "The correct answer is B: only production traffic guarantees production conditions."
            ),
            "C": (
                "Vendor benchmarks are optimized to show the vendor's hardware in the best possible "
                "light. This is not bias — it is rational behavior. But the result is that vendor "
                "benchmarks systematically over-represent the scenarios where that vendor's "
                "microarchitecture excels. They are not a neutral proxy for production. "
                "The correct answer is B: production traffic is vendor-agnostic by definition."
            ),
            "D": (
                "If a benchmark does not exercise the production bottleneck, averaging it with "
                "five other benchmarks that also miss the bottleneck does not fix the validity gap. "
                "You are averaging measurements of the wrong thing. "
                "The correct answer is B: only the production workload is guaranteed to exercise "
                "the production bottleneck, regardless of how many synthetic benchmarks you average."
            ),
        }
        _msg = _options.get(act2_reflect.value, "")
        mo.callout(mo.md(
            f"**Not quite — you selected option {act2_reflect.value}.** {_msg}"
        ), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo, act2_pred):
    _hypo_correct = act2_pred.value == "B"

    if _hypo_correct:
        mo.callout(mo.md(
            "**Your hypothesis was correct.** The synthetic benchmark uses cache-friendly, "
            "uniform-batch access patterns that production does not exhibit. "
            "Vendor A's microarchitecture is optimized for those patterns. "
            "Vendor B's is optimized for the variable-length, random-access patterns "
            "present in production. The benchmark measures the wrong bottleneck and "
            "systematically inverts the ranking."
        ), kind="success")
    else:
        _corrections = {
            "A": (
                "Statistical measurement error (option A) would cause *random* mis-ranking, "
                "not the systematic 20% production advantage for Vendor B. "
                "The correct answer is B: the synthetic benchmark exercises a different "
                "memory access pattern than production, systematically favoring Vendor A's "
                "cache-optimized microarchitecture."
            ),
            "C": (
                "A faster benchmark harness (option C) would inflate all scores equally or "
                "near-equally — it would not cause a consistent 20% production advantage for "
                "one vendor over the other. The correct answer is B: workload mismatch, "
                "not harness bias, explains the systematic production gap."
            ),
            "D": (
                "A larger model in the benchmark (option D) would change the absolute numbers "
                "but would not necessarily invert the ranking unless the larger model happened "
                "to change which subsystem is the bottleneck. The correct answer is B: "
                "the memory access pattern divergence is the structural explanation for "
                "why Vendor A wins on synthetic and Vendor B wins in production."
            ),
        }
        _msg = _corrections.get(act2_pred.value, "")
        mo.callout(mo.md(
            f"**Your hypothesis (option {act2_pred.value}) was not the root cause.** {_msg}"
        ), kind="warn")
    return


# ─── CONNECTIONS ──────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, CLOUD_SERIAL_FRAC, GPU_PURCHASE, CLOUD_AMDAHL_64, CTO_EXPECTED_SPEEDUP):
    mo.vstack([
        mo.md("---"),

        # ── KEY TAKEAWAYS ──
        mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. The serial fraction, not the number of processors, sets the speedup ceiling.</strong>
                    With S={CLOUD_SERIAL_FRAC*100:.0f}%, 64 GPUs deliver {CLOUD_AMDAHL_64:.1f}&times; &mdash;
                    not {CTO_EXPECTED_SPEEDUP:.0f}&times;. The Amdahl ceiling of 1/S = {1/CLOUD_SERIAL_FRAC:.1f}&times;
                    is invariant regardless of how many processors you add.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. A benchmark is valid only if it exercises the production bottleneck.</strong>
                    Vendor A scored 85 on the synthetic workload but lost in production because
                    synthetic and production use different memory access patterns. The benchmark
                    was reproducible and fair; it was not valid.
                </div>
                <div>
                    <strong>3. The batch-size divergence threshold quantifies benchmark invalidity.</strong>
                    When benchmark batch size diverges more than 10&times; from production,
                    the validity score drops below 0.5 and the ranking reversal becomes predictable
                    from first principles, not hindsight.
                </div>
            </div>
        </div>
        """),

        # ── CONNECTIONS ──
        mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">

            <!-- What's Next -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab 13: The Tail Latency Trap.</strong> This lab showed that the serial
                    fraction caps parallel scaling. Lab 13 asks: once your system is serving
                    requests, why does the average latency look healthy while users are complaining
                    &mdash; and how does Little&apos;s Law predict the P99 explosion before it happens?
                </div>
            </div>

            <!-- Textbook Connection -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-benchmarking-endtoend-benchmarks-51bb for the
                    full Amdahl derivation and @sec-benchmarking-fallacies-pitfalls-9781 for the
                    benchmark trap as a named anti-pattern.<br/>
                    <strong>Build:</strong> TinyTorch Module 12 &mdash; instrument a real inference
                    pipeline, measure component fractions, and verify Amdahl&apos;s Law on hardware.
                    See <code>tinytorch/src/12_benchmarking/</code>.
                </div>
            </div>

        </div>
        """),

        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. A vision pipeline has 8 ms preprocessing and 10 ms inference. After achieving a 5x inference speedup, the end-to-end speedup is only 1.8x. What is the Amdahl ceiling (maximum possible speedup with infinitely fast inference) — and what fraction of the pipeline causes it?

    2. Peak TFLOPS predict sustained performance poorly: an A100 achieving 90% utilization on ResNet-50 may achieve only 40% on a recommendation system. From Act II, what configuration (batch size, context) allowed the Jetson Orin NX to meet the latency SLA more efficiently than the H100?

    3. A vendor reports 10x faster inference on their new accelerator. Based on the chapter's data that model inference is only 10-20% of total production latency (the rest is preprocessing, queuing, postprocessing), what end-to-end speedup should you actually expect — and what should you benchmark to verify?

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
        }),
    ])
    return


# ─── CELL 21: DESIGN LEDGER SAVE + HUD FOOTER ────────────────────────────────
@app.cell(hide_code=True)
def _(mo, ledger, COLORS,
      act1_pred, act1_reflect, act2_pred, act2_reflect,
      context_toggle,
      sim_serial_pct, sim_num_proc,
      _speedup_theoretical, _validity_score, _failure,
      bench_type,
      BASELINE_HOURS, CTO_EXPECTED_SPEEDUP,
      CLOUD_SERIAL_FRAC, GPU_PURCHASE, CLOUD_AMDAHL_64):

    _ctx = context_toggle.value

    # ── Act I computed values for ledger ─────────────────────────────────────
    _s = sim_serial_pct.value / 100.0
    _p = 1.0 - _s
    _n = float(sim_num_proc.value)
    _amdahl_speedup = float(f"{_speedup_theoretical:.3f}")
    _expected        = CTO_EXPECTED_SPEEDUP

    # ── Ledger save ──────────────────────────────────────────────────────────
    ledger.save(chapter=12, design={
        "context":           _ctx,
        "serial_fraction":   round(_s, 3),
        "num_processors":    int(_n),
        "amdahl_speedup":    _amdahl_speedup,
        "act1_prediction":   act1_pred.value or "none",
        "act1_correct":      act1_pred.value == "C",
        "act2_result":       round(float(_validity_score), 3),
        "act2_decision":     bench_type.value,
        "constraint_hit":    bool(_failure),
        "expected_speedup":  _expected,
    })

    # ── HUD build ────────────────────────────────────────────────────────────
    _a1_ok      = act1_pred.value == "C"
    _a1_col     = COLORS["GreenLine"] if _a1_ok     else COLORS["RedLine"]
    _ref_ok     = act1_reflect.value == "C" if act1_reflect.value else False
    _ref_col    = COLORS["GreenLine"] if _ref_ok    else COLORS["OrangeLine"]
    _a2_ok      = act2_pred.value    == "B"
    _a2_col     = COLORS["GreenLine"] if _a2_ok     else COLORS["OrangeLine"]
    _a2r_ok     = act2_reflect.value == "B" if act2_reflect.value else False
    _a2r_col    = COLORS["GreenLine"] if _a2r_ok    else COLORS["OrangeLine"]
    _valid_col  = (
        COLORS["GreenLine"]  if _validity_score >= 0.70
        else COLORS["OrangeLine"] if _validity_score >= 0.45
        else COLORS["RedLine"]
    )
    _trap_color = "#f87171" if _failure else "#4ade80"

    mo.Html(f"""
    <div class="lab-hud" style="margin-top:40px;">
        <span><span class="hud-label">LAB</span>&nbsp;
              <span class="hud-value">12 &middot; Speedup Ceiling</span></span>
        <span><span class="hud-label">CTX</span>&nbsp;
              <span class="hud-value">{_ctx.upper()}</span></span>
        <span><span class="hud-label">A1 PRED</span>&nbsp;
              <span style="color:{_a1_col}; font-weight:700;">
                  {"CORRECT" if _a1_ok else "WRONG"}</span></span>
        <span><span class="hud-label">A1 REFLECT</span>&nbsp;
              <span style="color:{_ref_col}; font-weight:700;">
                  {"CORRECT" if _ref_ok else "&#8212;"}</span></span>
        <span><span class="hud-label">AMDAHL</span>&nbsp;
              <span class="hud-value">{_amdahl_speedup:.2f}&times;</span></span>
        <span><span class="hud-label">A2 PRED</span>&nbsp;
              <span style="color:{_a2_col}; font-weight:700;">
                  {"CORRECT" if _a2_ok else "&#8212;"}</span></span>
        <span><span class="hud-label">VALIDITY</span>&nbsp;
              <span style="color:{_valid_col}; font-weight:700;">
                  {_validity_score:.2f}</span></span>
        <span><span class="hud-label">TRAP</span>&nbsp;
              <span style="color:{_trap_color}; font-weight:700;">
                  {"HIT" if _failure else "OK"}</span></span>
        <span><span class="hud-label">SAVED</span>&nbsp;
              <span class="hud-active">ch12</span></span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
