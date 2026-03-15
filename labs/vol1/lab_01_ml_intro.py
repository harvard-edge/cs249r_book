import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 01: THE MAGNITUDE AWAKENING
#
# Chapter: Introduction to ML Systems (@sec-introduction)
# Core Invariant: The D·A·M Triad (Data, Algorithm, Machine) and the
#                 9-order-of-magnitude gap that prevents a universal software stack.
#
# 2-Act Structure (35-40 minutes):
#   Act I  — The Scale Blindspot (12-15 min)
#             Calibrate student intuition about the H100 ↔ Cortex-M7 compute gap.
#             The central question: does a 6-order-of-magnitude gap matter for
#             software architecture? The answer is: yes — it forces separate stacks.
#
#   Act II — The Iron Law Preview (20-25 min)
#             Apply T = D/BW + O/R + L to ResNet-50 on both deployment contexts.
#             The central question: which Iron Law term dominates at batch=1?
#             The OOM failure state: ResNet-50 (100 MB) > Cortex-M7 (512 KB).
#
# Deployment Contexts:
#   Cloud:  NVIDIA H100 SXM5 (989 TFLOPs FP16, 3350 GB/s HBM3, 80 GB, 700 W)
#   TinyML: Cortex-M7 (0.001 TFLOPs, 0.05 GB/s, 512 KB SRAM, 0.1 W)
#
# Design Ledger: saves chapter=1 with context, prediction accuracy, OOM trigger.
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible for instructor inspection) ─
@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

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
    return COLORS, LAB_CSS, DesignLedger, apply_plotly_theme, go, ledger, math, mo, np


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, LAB_CSS, mo):
    _c_cloud = COLORS["Cloud"]
    _c_tiny = COLORS["Tiny"]
    _c_surface0 = COLORS["Surface0"]
    _c_surface1 = COLORS["Surface1"]
    _header = mo.Html(f"""
    {LAB_CSS}
    <div style="background: linear-gradient(135deg, {_c_surface0} 0%, {_c_surface1} 100%);
                border-radius: 16px; padding: 32px 40px; margin-bottom: 8px;
                border: 1px solid #2d3748;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                            text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;">
                    Vol 1 · Lab 01 · Introduction to ML Systems
                </div>
                <div style="font-size: 2.0rem; font-weight: 800; color: #f1f5f9; line-height: 1.15; margin-bottom: 10px;">
                    The Magnitude Awakening
                </div>
                <div style="font-size: 0.95rem; color: #94a3b8; max-width: 580px; line-height: 1.6;">
                    Nine orders of magnitude separate a cloud accelerator from a microcontroller.
                    This lab forces you to confront that gap quantitatively — and discover why it
                    makes a universal ML software stack physically impossible.
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;">
                <span class="badge badge-info">D·A·M Triad</span>
                <span class="badge badge-info">Iron Law T = D/BW + O/R + L</span>
                <span class="badge badge-warn">35-40 minutes · 2 Acts</span>
            </div>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap;">
            <div style="background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_cloud}; font-weight: 700;">Cloud Context</span>
                <span style="color: #94a3b8;"> — NVIDIA H100 · 989 TFLOPS · 3350 GB/s · 80 GB · 700 W</span>
            </div>
            <div style="background: rgba(0,143,69,0.12); border: 1px solid rgba(0,143,69,0.35);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_tiny}; font-weight: 700;">TinyML Context</span>
                <span style="color: #94a3b8;"> — Cortex-M7 · 0.001 TFLOPS · 0.05 GB/s · 512 KB · 0.1 W</span>
            </div>
        </div>
    </div>
    """)
    _header
    return


# ─── CELL 2: BRIEFING ──────────────────────────────────────────────────────────
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the hardware gap</strong> between cloud and TinyML deployment targets — measuring the 10&sup6; compute, 67,000&times; bandwidth, and 160,000&times; memory ratios that prevent a universal software stack.</div>
                <div style="margin-bottom: 3px;">2. <strong>Predict which Iron Law term dominates</strong> ResNet-50 inference at batch=1 on the H100, using T&nbsp;=&nbsp;D/BW&nbsp;+&nbsp;O/R&nbsp;+&nbsp;L to decompose the 100&nbsp;MB data-movement vs. 4&nbsp;GFLOPs compute trade-off.</div>
                <div style="margin-bottom: 3px;">3. <strong>Identify the binding constraint</strong> that makes ResNet-50 infeasible on a Cortex-M7, distinguishing between memory-capacity failure (OOM) and compute-speed limitation.</div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- PREREQUISITES + DURATION -->
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    D&middot;A&middot;M Taxonomy from @sec-introduction-scaling-regimes &middot;
                    Deployment spectrum (Cloud/Edge/Mobile/TinyML) from @sec-introduction-deployment-spectrum-a38c &middot;
                    Iron Law equation T&nbsp;=&nbsp;D/BW&nbsp;+&nbsp;O/R&nbsp;+&nbsp;L from @sec-introduction-iron-law-ml-systems-c32a
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
                "If the H100 is one million times faster than a microcontroller, why can't you
                simply shrink a cloud model to make it run anywhere &mdash; and which physical
                term of the Iron Law tells you why the cloud itself is already slower than you think?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-introduction-scaling-regimes** — The AI Triad and the two scaling regimes (single-node vs. distributed fleet)
    - **@sec-introduction-deployment-spectrum-a38c** — The four deployment paradigms and their power/memory constraints
    - **@sec-introduction-iron-law-ml-systems-c32a** — The Iron Law equation `T = D/BW + O/R + L` and its three terms
    - **@sec-introduction-dam** — D·A·M taxonomy as a diagnostic lens for bottleneck analysis
    """), kind="info")
    return


# ─── CELL 4: CONTEXT TOGGLE + LEDGER LOAD ──────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _prior_ctx = "cloud"  # default — overridden by ledger if available
    context_toggle = mo.ui.radio(
        options={"Cloud (H100)": "cloud", "TinyML (Cortex-M7)": "tiny"},
        value="Cloud (H100)",
        label="Deployment context for this session:",
        inline=True,
    )
    mo.hstack([
        mo.Html(f"""
        <div style="font-size:0.78rem; font-weight:700; color:{COLORS['TextMuted']};
                    text-transform:uppercase; letter-spacing:0.08em; margin-right:8px; padding-top:2px;">
            Active Context:
        </div>
        """),
        context_toggle,
    ], justify="start", gap=0)
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "I"
    _act_color    = COLORS["BlueLine"]
    _act_title    = "The Scale Blindspot"
    _act_duration = "12–15 min"
    _act_why      = (
        "Most engineers estimate the cloud-to-microcontroller compute gap at 100&times; or "
        "10,000&times; — a manageable engineering difference. "
        "The data will show a 1,000,000&times; gap: not an optimization problem, "
        "but an architectural boundary that forces separate software stacks."
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


# ─── ACT I: STAKEHOLDER MESSAGE ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _color = COLORS["Cloud"]
    _bg = COLORS["BlueL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; VP of Engineering
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We need keyword spotting on every smart doorbell we ship. Our cloud team
            says just run the model on our H100 cluster — same code we use for everything else.
            The microcontrollers cost $2.40 each and have 512 KB of RAM. Before we commit
            to $50,000 in cloud infrastructure, can you explain why we can't use the same
            software stack for both?"
        </div>
    </div>
    """)
    return


# ─── ACT I: CONCEPT FRAMING ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The VP is asking a systems engineering question, not a machine learning question.
    The answer depends on quantifying the physical gap between the two deployment targets —
    not in vague terms like "the cloud is faster," but in exact orders of magnitude.

    The **D·A·M Triad** (Data, Algorithm, Machine) from @sec-introduction provides the
    diagnostic lens. When the *Machine* axis spans nine orders of magnitude, the
    *Algorithm* and *Data* pipelines cannot be shared. The gap is not an implementation
    detail — it is a physical constraint that forces architectural separation.

    Before looking at any data, commit to a prediction.
    """)
    return


# ─── ACT I: PREDICTION LOCK ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Your Prediction")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) ~100× (roughly two orders of magnitude)": "A",
            "B) ~10,000× (roughly four orders of magnitude)": "B",
            "C) ~1,000,000× (roughly six orders of magnitude)": "C",
            "D) ~1,000,000,000× (roughly nine orders of magnitude)": "D",
        },
        label="By what factor does the H100 exceed the Cortex-M7 in peak compute throughput (TFLOPS)?",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(act1_pred, mo):
    mo.stop(
        act1_pred.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Act I instruments."),
            kind="warn",
        ),
    )
    mo.md("")
    return


# ─── ACT I: INSTRUMENT — 4-WAY COMPARISON BAR CHART (LOG SCALE) ────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### The Magnitude Landscape")
    return


@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, context_toggle, go, math, mo, np):
    # ── Hardware constants (source: @sec-introduction-deployment-spectrum-a38c,
    #    NVIDIA H100 SXM5 spec sheet, ARM Cortex-M7 TRM, Jetson Orin NX spec)
    #
    # Cloud  — NVIDIA H100 SXM5 (Hopper, 2022)
    H100_TFLOPS   = 989.0     # TFLOPs FP16 Tensor Core; source: NVIDIA H100 Data Sheet
    H100_BW_GBS   = 3350.0    # GB/s HBM3; source: NVIDIA H100 Data Sheet
    H100_RAM_GB   = 80.0      # GB HBM3e
    H100_TDP_W    = 700.0     # Watts; SXM variant
    H100_TFLOPS_MFLOPS = H100_TFLOPS * 1e6  # convert to MFLOPS for ratio

    # Edge   — NVIDIA Jetson Orin NX (2023)
    ORIN_TFLOPS   = 100.0     # TFLOPs INT8 equivalent; source: Jetson Orin NX spec
    ORIN_BW_GBS   = 102.0     # GB/s
    ORIN_RAM_GB   = 16.0      # GB
    ORIN_TDP_W    = 25.0      # Watts

    # Mobile — Apple A17-class NPU (2023, representative smartphone)
    MOBILE_TOPS   = 35.0      # TOPS INT8; source: @sec-introduction-deployment-spectrum-a38c
    MOBILE_BW_GBS = 68.0      # GB/s
    MOBILE_RAM_GB = 8.0       # GB
    MOBILE_TDP_W  = 5.0       # Watts sustained

    # TinyML — ARM Cortex-M7 (representative MCU, e.g. STM32H7 class)
    MCU_TFLOPS    = 0.001     # TFLOPs (~1 GFLOPS FP32 DSP); source: hardware.py Tiny.Generic_MCU
    MCU_BW_GBS    = 0.05      # GB/s; source: hardware.py Tiny.Generic_MCU
    MCU_SRAM_MB   = 0.512     # MB (512 KB); source: constants.py MCU_RAM_KIB
    MCU_TDP_W     = 0.1       # Watts; source: @sec-introduction-deployment-spectrum-a38c

    # ── Regime labels and colors
    _regimes = ["TinyML\n(Cortex-M7)", "Mobile\n(Smartphone NPU)", "Edge\n(Jetson Orin NX)", "Cloud\n(H100)"]
    _colors_bar = [COLORS["Tiny"], COLORS["Mobile"], COLORS["Edge"], COLORS["Cloud"]]

    # ── Highlight the selected context
    _ctx = context_toggle.value
    _bar_opacities = []
    for _r in ["tiny", "mobile", "edge", "cloud"]:
        if _r == _ctx:
            _bar_opacities.append(1.0)
        else:
            _bar_opacities.append(0.35)

    _highlight_colors = []
    for _i, (_c, _o) in enumerate(zip(_colors_bar, _bar_opacities)):
        _highlight_colors.append(_c if _o == 1.0 else "#94a3b8")

    # ── Compute (TFLOPs) — log scale
    _compute = [MCU_TFLOPS, MOBILE_TOPS, ORIN_TFLOPS, H100_TFLOPS]
    _compute_log = [math.log10(v) for v in _compute]

    # ── Memory BW (GB/s) — log scale
    _bw = [MCU_BW_GBS, MOBILE_BW_GBS, ORIN_BW_GBS, H100_BW_GBS]
    _bw_log = [math.log10(v) for v in _bw]

    # ── Power (W) — log scale
    _power = [MCU_TDP_W, MOBILE_TDP_W, ORIN_TDP_W, H100_TDP_W]
    _power_log = [math.log10(v) for v in _power]

    # ── Build figure with subplots
    from plotly.subplots import make_subplots
    _fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Peak Compute (TFLOPs)", "Memory Bandwidth (GB/s)", "Power Budget (W)"],
        horizontal_spacing=0.12,
    )

    _reg_display = ["TinyML", "Mobile", "Edge", "Cloud"]

    for _col, (_log_vals, _raw_vals, _fmt_unit) in enumerate(
        zip(
            [_compute_log, _bw_log, _power_log],
            [_compute, _bw, _power],
            ["TFLOPS", "GB/s", "W"],
        ),
        start=1,
    ):
        _bar_colors_col = []
        for _j in range(4):
            _bar_colors_col.append(_colors_bar[_j] if _bar_opacities[_j] == 1.0 else "#c7cdd4")

        _hover = [f"{_reg_display[_j]}: {_raw_vals[_j]:,.3g} {_fmt_unit}<br>log₁₀ = {_log_vals[_j]:.1f}" for _j in range(4)]

        _fig.add_trace(
            go.Bar(
                x=_reg_display,
                y=_log_vals,
                marker_color=_bar_colors_col,
                text=[f"10^{v:.0f}" for v in _log_vals],
                textposition="outside",
                hovertext=_hover,
                hoverinfo="text",
                showlegend=False,
            ),
            row=1, col=_col,
        )

    _fig.update_layout(
        height=380,
        title_text="",
        margin=dict(t=50, b=60, l=40, r=20),
    )
    _fig.update_yaxes(
        title_text="log₁₀ (value)",
        range=[-3.5, 4.0],
        tickvals=[-3, -2, -1, 0, 1, 2, 3],
        ticktext=["10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²", "10³"],
    )

    apply_plotly_theme(_fig)
    mo.ui.plotly(_fig)
    return (
        H100_BW_GBS,
        H100_RAM_GB,
        H100_TFLOPS,
        H100_TDP_W,
        MCU_BW_GBS,
        MCU_SRAM_MB,
        MCU_TFLOPS,
        MCU_TDP_W,
        MOBILE_BW_GBS,
        MOBILE_RAM_GB,
        MOBILE_TDP_W,
        MOBILE_TOPS,
        ORIN_BW_GBS,
        ORIN_RAM_GB,
        ORIN_TDP_W,
        ORIN_TFLOPS,
    )


# ─── ACT I: QUANTITATIVE GAP TABLE ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS,
    H100_BW_GBS,
    H100_RAM_GB,
    H100_TFLOPS,
    H100_TDP_W,
    MCU_BW_GBS,
    MCU_SRAM_MB,
    MCU_TFLOPS,
    MCU_TDP_W,
    mo,
):
    _c_cloud = COLORS["Cloud"]
    _c_tiny = COLORS["Tiny"]
    _c_red = COLORS["RedLine"]
    _c_border = COLORS["Border"]

    _compute_ratio = H100_TFLOPS / MCU_TFLOPS
    _bw_ratio      = H100_BW_GBS / MCU_BW_GBS
    _ram_ratio_num = (H100_RAM_GB * 1e3) / MCU_SRAM_MB  # both in MB
    _power_ratio   = H100_TDP_W / MCU_TDP_W

    def _ratio_badge(r):
        if r >= 1e6:
            return f'<span style="color:{_c_red}; font-weight:800;">{r:,.0f}&times;</span>'
        elif r >= 1e4:
            return f'<span style="color:#CC5500; font-weight:700;">{r:,.0f}&times;</span>'
        else:
            return f'<span style="color:{COLORS["BlueLine"]}; font-weight:700;">{r:,.0f}&times;</span>'

    mo.Html(f"""
    <div class="lab-card" style="margin: 8px 0;">
        <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
            <thead>
                <tr style="border-bottom:2px solid {_c_border};">
                    <th style="text-align:left; padding:8px 12px; color:{COLORS['TextMuted']}; font-weight:700; text-transform:uppercase; font-size:0.72rem; letter-spacing:0.08em;">Axis</th>
                    <th style="text-align:right; padding:8px 12px; color:{_c_cloud}; font-weight:700;">Cloud (H100)</th>
                    <th style="text-align:right; padding:8px 12px; color:{_c_tiny}; font-weight:700;">TinyML (Cortex-M7)</th>
                    <th style="text-align:right; padding:8px 12px; color:{COLORS['Text']}; font-weight:700;">Gap (H100 / MCU)</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid {_c_border};">
                    <td style="padding:8px 12px; color:{COLORS['TextSec']}; font-weight:600;">Peak Compute (TFLOPs)</td>
                    <td style="text-align:right; padding:8px 12px; font-family:monospace;">{H100_TFLOPS:,}</td>
                    <td style="text-align:right; padding:8px 12px; font-family:monospace;">{MCU_TFLOPS:.3f}</td>
                    <td style="text-align:right; padding:8px 12px;">{_ratio_badge(_compute_ratio)}</td>
                </tr>
                <tr style="border-bottom:1px solid {_c_border};">
                    <td style="padding:8px 12px; color:{COLORS['TextSec']}; font-weight:600;">Memory Bandwidth (GB/s)</td>
                    <td style="text-align:right; padding:8px 12px; font-family:monospace;">{H100_BW_GBS:,}</td>
                    <td style="text-align:right; padding:8px 12px; font-family:monospace;">{MCU_BW_GBS}</td>
                    <td style="text-align:right; padding:8px 12px;">{_ratio_badge(_bw_ratio)}</td>
                </tr>
                <tr style="border-bottom:1px solid {_c_border};">
                    <td style="padding:8px 12px; color:{COLORS['TextSec']}; font-weight:600;">Memory Capacity</td>
                    <td style="text-align:right; padding:8px 12px; font-family:monospace;">{H100_RAM_GB:,} GB</td>
                    <td style="text-align:right; padding:8px 12px; font-family:monospace;">{MCU_SRAM_MB * 1000:.0f} KB</td>
                    <td style="text-align:right; padding:8px 12px;">{_ratio_badge(_ram_ratio_num)}</td>
                </tr>
                <tr>
                    <td style="padding:8px 12px; color:{COLORS['TextSec']}; font-weight:600;">Power Budget (W)</td>
                    <td style="text-align:right; padding:8px 12px; font-family:monospace;">{H100_TDP_W:,}</td>
                    <td style="text-align:right; padding:8px 12px; font-family:monospace;">{MCU_TDP_W}</td>
                    <td style="text-align:right; padding:8px 12px;">{_ratio_badge(_power_ratio)}</td>
                </tr>
            </tbody>
        </table>
    </div>
    """)
    return


# ─── ACT I: PREDICTION REVEAL ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(H100_TFLOPS, MCU_TFLOPS, act1_pred, mo):
    # Actual gap (TFLOPs ratio)
    _actual_gap = H100_TFLOPS / MCU_TFLOPS  # 989,000× ≈ ~10^6

    _pred_map = {"A": 100, "B": 10_000, "C": 1_000_000, "D": 1_000_000_000}
    _pred_val = _pred_map[act1_pred.value]
    _off_factor = _actual_gap / _pred_val

    _correct = act1_pred.value == "C"

    if _correct:
        _reveal = mo.callout(mo.md(
            f"**Correct.** You predicted ~1,000,000×. "
            f"The actual compute ratio is {_actual_gap:,.0f}× (H100 FP16 / Cortex-M7). "
            f"This is approximately 10⁶ — six orders of magnitude. "
            f"The memory bandwidth gap is similar: 3350 GB/s vs. 0.05 GB/s = 67,000×. "
            f"The power gap is 7000×. Across all three D·A·M axes, the hardware is "
            f"separated by 4–7 orders of magnitude — making a shared software stack "
            f"physically infeasible."
        ), kind="success")
    elif act1_pred.value == "D":
        _reveal = mo.callout(mo.md(
            f"**Close, but one order of magnitude too large.** "
            f"You predicted ~10⁹. The actual ratio is {_actual_gap:,.0f}× (~10⁶). "
            f"Nine orders of magnitude is the span across *all* D·A·M axes combined "
            f"(the chapter describes this as 9 orders total when you include power, memory, "
            f"and compute together). On compute alone the gap is ~10⁶. "
            f"The principle holds: separate stacks are mandatory."
        ), kind="warn")
    elif act1_pred.value == "B":
        _reveal = mo.callout(mo.md(
            f"**You underestimated by {_off_factor:.0f}×.** "
            f"You predicted ~10,000×. The actual compute ratio is {_actual_gap:,.0f}× (~10⁶). "
            f"At 10⁴, a performance difference is an engineering optimization — "
            f"at 10⁶, it is an architectural boundary. The TinyML device cannot even hold "
            f"a single modern model in memory, let alone execute it at useful speed."
        ), kind="warn")
    else:
        _reveal = mo.callout(mo.md(
            f"**You significantly underestimated.** "
            f"You predicted ~100×. The actual ratio is {_actual_gap:,.0f}× (~10⁶). "
            f"A 100× gap is manageable with smart caching and batching. "
            f"A 10⁶ gap is a different kind of physics: it means the hardware fundamentally "
            f"cannot run the same model code, triggering the need for separate stacks, "
            f"separate model formats, and separate compilation targets."
        ), kind="warn")

    _reveal
    return


# ─── ACT I: MATH PEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "View the D·A·M Axes Definition": mo.md("""
        **The D·A·M Taxonomy** (from @sec-introduction-scaling-regimes):

        Every ML system is a three-way interaction between three axes:

        | Axis | Governs | Lab Instruments |
        |------|---------|----------------|
        | **D — Data** | Volume moved (`D_vol`), bandwidth consumed | Memory BW bar chart |
        | **A — Algorithm** | Operation count (`O`), model architecture | FLOPs bar chart |
        | **M — Machine** | Peak throughput (`R_peak`), memory capacity, power | All three charts |

        The D·A·M framework predicts that optimizing one axis in isolation shifts bottlenecks
        rather than eliminating them. A 10× faster algorithm on a power-constrained device
        does not help if the model still exceeds the device's memory capacity.

        **The chapter claim** (@sec-introduction, line ~1778):
        > "Modern models demand resources nine orders of magnitude larger [than early neural nets]."

        The lab measures the *hardware* gap (6 orders on compute), not the historical *model* gap.
        Both support the same conclusion: a universal software stack is physically impossible.
        """)
    })
    return


# ─── ACT I: STRUCTURED REFLECTION ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Act I Reflection")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) We need faster algorithms for the TinyML device": "A",
            "B) We need separate software stacks per deployment regime": "B",
            "C) We need more training data to handle the hardware gap": "C",
            "D) We need better compilers to bridge the compute difference": "D",
        },
        label="What does the 10⁶ compute gap between Cloud and TinyML imply for software architecture?",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(act1_reflect, mo):
    mo.stop(
        act1_reflect.value is None,
        mo.md(""),
    )

    if act1_reflect.value == "B":
        _fb = mo.callout(mo.md(
            "**Correct.** The magnitude gap is not an optimization problem — it is an architectural "
            "boundary. An H100 kernel compiled for CUDA cannot run on a Cortex-M7. The MCU has no "
            "FP16 Tensor Cores, no HBM, and 512 KB of SRAM instead of 80 GB. The hardware gap "
            "forces separate compilation targets, separate model formats (TFLite, ONNX vs. CUDA PTX), "
            "and separate runtime environments. This is why @sec-ml-systems introduces four distinct "
            "deployment paradigms rather than one universal stack."
        ), kind="success")
    elif act1_reflect.value == "D":
        _fb = mo.callout(mo.md(
            "**Partially correct, but incomplete.** Better compilers (like TVM or MLIR) do help "
            "bridge execution targets — but they cannot add SRAM that does not exist. A ResNet-50 "
            "requires 100 MB of RAM at inference. The Cortex-M7 has 512 KB. No compiler can resolve "
            "a 200× memory deficit. The fundamental answer is B: separate stacks, because the "
            "physical constraints are incommensurable."
        ), kind="warn")
    elif act1_reflect.value == "A":
        _fb = mo.callout(mo.md(
            "**Not quite.** A faster algorithm reduces operation count (the *Algorithm* axis of D·A·M), "
            "but does not change memory bandwidth or capacity (the *Machine* axis). Even if an algorithm "
            "ran in zero FLOPs, it would still need to load model weights — and 100 MB of weights do not "
            "fit in 512 KB of SRAM. The constraint is physical, not algorithmic."
        ), kind="warn")
    else:
        _fb = mo.callout(mo.md(
            "**Incorrect.** Data volume (the *Data* axis) is independent of hardware capacity. "
            "More training data makes a better model but does not change the memory bandwidth "
            "or SRAM size of the Cortex-M7. The hardware gap is a physical constraint on the "
            "*Machine* axis that exists regardless of how much data the model was trained on."
        ), kind="warn")

    _fb
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "II"
    _act_color    = COLORS["OrangeLine"]
    _act_title    = "The Iron Law Preview"
    _act_duration = "20–25 min"
    _act_why      = (
        "Act I established the 10&sup6;&times; compute gap between H100 and Cortex-M7. "
        "Act II asks: given ResNet-50 (4 GFLOPs, 100 MB) on each target, "
        "which Iron Law term &mdash; memory movement (D/BW), arithmetic (O/R), "
        "or overhead (L) &mdash; actually dominates at batch=1? "
        "On H100 the answer is counterintuitive. On Cortex-M7 there is a harder constraint: "
        "the model simply does not fit in 512 KB of SRAM."
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


# ─── ACT II: INTRO TEXT ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Act I established *that* a gap exists. Act II asks: given a specific model
    and a specific hardware target, which physical constraint actually dominates?

    The **Iron Law of ML Systems** (@sec-introduction-iron-law-ml-systems-c32a)
    decomposes total execution time into three additive terms:

    ```
    T = D/BW  +  O/R  +  L
        Data     Compute  Overhead
        Term     Term     Term
    ```

    Where:
    - `D` = data volume moved (bytes)
    - `BW` = memory bandwidth (bytes/sec)
    - `O` = arithmetic operations (FLOPs)
    - `R` = peak throughput (FLOPs/sec)
    - `L` = dispatch/overhead latency (sec)

    Whichever term is largest determines the **binding constraint** on that hardware.

    **The workload**: ResNet-50 forward pass (batch size = 1).
    ResNet-50 requires 4 GFLOPs of compute and moves approximately 100 MB of data
    (weights + activations) per inference. These values are from the chapter's
    Lighthouse Model definitions.
    """)
    return


# ─── ACT II: PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Your Prediction")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) Compute term dominates (O/R is largest)": "A",
            "B) Memory term dominates (D/BW is largest)": "B",
            "C) Both terms are approximately equal": "C",
            "D) It depends on which hardware context is selected": "D",
        },
        label="On the H100 at batch=1, for ResNet-50 (4 GFLOPs, 100 MB): which Iron Law term dominates?",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(act2_pred, mo):
    mo.stop(
        act2_pred.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Act II instruments."),
            kind="warn",
        ),
    )
    mo.md("")
    return


# ─── ACT II: HARDWARE CONTEXT SELECTOR ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    act2_context = mo.ui.radio(
        options={"Cloud (H100)": "cloud", "TinyML (Cortex-M7)": "tiny"},
        value="Cloud (H100)",
        label="Select hardware context:",
        inline=True,
    )
    mo.hstack([
        mo.Html(f"""
        <div style="font-size:0.78rem; font-weight:700; color:{COLORS['TextMuted']};
                    text-transform:uppercase; letter-spacing:0.08em; margin-right:8px; padding-top:2px;">
            Hardware:
        </div>
        """),
        act2_context,
    ], justify="start", gap=0)
    return (act2_context,)


# ─── ACT II: IRON LAW COMPUTATION + OOM DETECTION ──────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS,
    H100_BW_GBS,
    H100_RAM_GB,
    H100_TFLOPS,
    MCU_BW_GBS,
    MCU_SRAM_MB,
    MCU_TFLOPS,
    act2_context,
    apply_plotly_theme,
    go,
    mo,
):
    # ── ResNet-50 workload constants (source: @sec-introduction Lighthouse Models)
    # ResNet-50 single forward pass at batch=1:
    RESNET50_GFLOPS   = 4.0     # GFLOPs; source: He et al. (2016) / chapter Lighthouse table
    RESNET50_DATA_MB  = 100.0   # MB (weights + activation footprint); source: chapter Lighthouse

    # Convert to consistent units
    _resnet_ops_gflops   = RESNET50_GFLOPS          # GFLOPs = 10^9 FLOPs
    _resnet_data_gb      = RESNET50_DATA_MB / 1000.0 # GB

    _ctx = act2_context.value

    if _ctx == "cloud":
        _hw_name      = "NVIDIA H100"
        _hw_color     = COLORS["Cloud"]
        _hw_tflops    = H100_TFLOPS          # TFLOPs = 10^12 FLOPs/s
        _hw_bw        = H100_BW_GBS          # GB/s
        _hw_ram_gb    = H100_RAM_GB          # GB
        _overhead_ms  = 0.01                 # ms dispatch tax (hardware.py Cloud.H100)
    else:
        _hw_name      = "Cortex-M7"
        _hw_color     = COLORS["Tiny"]
        _hw_tflops    = MCU_TFLOPS           # TFLOPs
        _hw_bw        = MCU_BW_GBS           # GB/s
        _hw_ram_gb    = MCU_SRAM_MB / 1000.0 # GB (convert KB→GB: 512KB = 0.000512 GB)
        _overhead_ms  = 2.0                  # ms dispatch tax (hardware.py Tiny.Generic_MCU)

    # ── Iron Law computation
    # T_mem  = D_vol (GB) / BW (GB/s) → seconds → ms
    # T_comp = O (GFLOPs) / R (TFLOPs/s) = O (GFLOPs) / (R * 1000 GFLOPs/s) → seconds → ms
    #        (1 TFLOPs = 1000 GFLOPs)
    _T_mem_s   = _resnet_data_gb / _hw_bw              # seconds
    _T_comp_s  = _resnet_ops_gflops / (_hw_tflops * 1000.0)  # seconds (convert TFLOPS→GFLOPS)
    _T_overhead_s = _overhead_ms / 1000.0              # seconds

    _T_mem_ms  = _T_mem_s   * 1000.0
    _T_comp_ms = _T_comp_s  * 1000.0
    _T_total_ms = _T_mem_ms + _T_comp_ms + _overhead_ms

    # ── OOM detection
    # ResNet-50 needs 100 MB of RAM at minimum (weights alone).
    # Cortex-M7 has 512 KB = 0.512 MB of SRAM.
    _oom = _resnet_data_gb > _hw_ram_gb
    _oom_ratio = _resnet_data_gb / _hw_ram_gb if _hw_ram_gb > 0 else float("inf")

    # ── Determine bottleneck
    _bottleneck = "memory" if _T_mem_ms > _T_comp_ms else "compute"

    # ── Color bars: red if OOM, otherwise by bottleneck
    if _oom:
        _bar_colors = [COLORS["RedLine"], COLORS["RedLine"], COLORS["OrangeLine"]]
    else:
        _bar_colors = [
            COLORS["RedLine"] if _bottleneck == "memory" else COLORS["BlueLine"],
            COLORS["RedLine"] if _bottleneck == "compute" else COLORS["BlueLine"],
            COLORS["OrangeLine"],
        ]

    # ── Chart
    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        x=["Memory Term (D/BW)", "Compute Term (O/R)", "Overhead (L)"],
        y=[_T_mem_ms, _T_comp_ms, _overhead_ms],
        marker_color=_bar_colors,
        text=[f"{_T_mem_ms:.4f} ms", f"{_T_comp_ms:.4f} ms", f"{_overhead_ms:.4f} ms"],
        textposition="outside",
        width=0.5,
    ))
    _fig.update_layout(
        height=300,
        yaxis=dict(title="Latency (ms)", type="log"),
        margin=dict(t=30, b=40, l=50, r=20),
    )
    apply_plotly_theme(_fig)
    _chart = mo.ui.plotly(_fig)

    # ── Metric cards
    _mem_dominant_label = "BINDING" if _bottleneck == "memory" else "not binding"
    _comp_dominant_label = "BINDING" if _bottleneck == "compute" else "not binding"
    _mem_card_color = COLORS["RedLine"] if _bottleneck == "memory" else COLORS["BlueLine"]
    _comp_card_color = COLORS["RedLine"] if _bottleneck == "compute" else COLORS["BlueLine"]

    _cards = mo.Html(f"""
    <div style="display:flex; gap:16px; justify-content:center; margin:16px 0; flex-wrap:wrap;">
        <div style="padding:20px 24px; border:2px solid {_mem_card_color};
                    border-radius:10px; min-width:170px; text-align:center;
                    background:{'#FEF2F2' if _bottleneck == 'memory' else '#f8fafc'};">
            <div style="font-size:0.78rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.08em;">Memory Term</div>
            <div style="font-size:0.72rem; color:#94a3b8; margin:2px 0;">D / BW</div>
            <div style="font-size:1.9rem; font-weight:800; color:{_mem_card_color}; font-family:monospace;">
                {_T_mem_ms:.4f}
            </div>
            <div style="font-size:0.82rem; color:#64748b;">ms</div>
            <div style="font-size:0.72rem; font-weight:700; color:{_mem_card_color}; margin-top:6px; text-transform:uppercase;">
                {_mem_dominant_label}
            </div>
        </div>
        <div style="padding:20px 24px; border:2px solid {_comp_card_color};
                    border-radius:10px; min-width:170px; text-align:center;
                    background:{'#FEF2F2' if _bottleneck == 'compute' else '#f8fafc'};">
            <div style="font-size:0.78rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.08em;">Compute Term</div>
            <div style="font-size:0.72rem; color:#94a3b8; margin:2px 0;">O / R</div>
            <div style="font-size:1.9rem; font-weight:800; color:{_comp_card_color}; font-family:monospace;">
                {_T_comp_ms:.4f}
            </div>
            <div style="font-size:0.82rem; color:#64748b;">ms</div>
            <div style="font-size:0.72rem; font-weight:700; color:{_comp_card_color}; margin-top:6px; text-transform:uppercase;">
                {_comp_dominant_label}
            </div>
        </div>
        <div style="padding:20px 24px; border:2px solid {COLORS['OrangeLine']};
                    border-radius:10px; min-width:170px; text-align:center; background:#FFF7ED;">
            <div style="font-size:0.78rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.08em;">Total Latency</div>
            <div style="font-size:0.72rem; color:#94a3b8; margin:2px 0;">T = D/BW + O/R + L</div>
            <div style="font-size:1.9rem; font-weight:800; color:{COLORS['OrangeLine']}; font-family:monospace;">
                {_T_total_ms:.3f}
            </div>
            <div style="font-size:0.82rem; color:#64748b;">ms</div>
            <div style="font-size:0.72rem; font-weight:700; color:{_hw_color}; margin-top:6px;">
                {_hw_name}
            </div>
        </div>
    </div>
    """)

    # ── Physics formula display
    _formula = mo.Html(f"""
    <div class="lab-card" style="margin:8px 0; background:#f8fafc; font-family:monospace; font-size:0.85rem;">
        <div style="color:#64748b; font-weight:700; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">Iron Law Computation — {_hw_name}</div>
        <div style="line-height:2.0; color:#1e293b;">
            <span style="color:#006395;">D/BW</span>&nbsp; = {_resnet_data_gb:.3f} GB &divide; {_hw_bw:,.1f} GB/s = <strong style="color:{_mem_card_color};">{_T_mem_ms:.4f} ms</strong><br>
            <span style="color:#006395;">O/R</span>&nbsp;&nbsp; = {_resnet_ops_gflops:.1f} GFLOPs &divide; {_hw_tflops * 1000:.0f} GFLOPs/s = <strong style="color:{_comp_card_color};">{_T_comp_ms:.4f} ms</strong><br>
            <span style="color:#CC5500;">L</span>&nbsp;&nbsp;&nbsp;&nbsp; = {_overhead_ms:.2f} ms (dispatch overhead)<br>
            <strong>T_total = {_T_total_ms:.4f} ms</strong>&nbsp;&nbsp;&nbsp;
            Bottleneck: <strong style="color:{'#CB202D' if not _oom else '#CB202D'};">
                {'OOM — infeasible' if _oom else _bottleneck.upper() + '-BOUND'}
            </strong>
        </div>
    </div>
    """)

    mo.vstack([_cards, _chart, _formula])
    return (
        RESNET50_DATA_MB,
        RESNET50_GFLOPS,
        _T_comp_ms,
        _T_mem_ms,
        _T_total_ms,
        _bottleneck,
        _oom,
        _oom_ratio,
    )


# ─── ACT II: OOM FAILURE STATE BANNER ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(RESNET50_DATA_MB, MCU_SRAM_MB, _oom, _oom_ratio, mo):
    if _oom:
        _oom_banner = mo.callout(mo.md(
            f"**OOM — Infeasible.** "
            f"ResNet-50 requires {RESNET50_DATA_MB:.0f} MB of RAM for weights and activations. "
            f"The Cortex-M7 has {MCU_SRAM_MB * 1000:.0f} KB of SRAM. "
            f"The model exceeds available memory by **{_oom_ratio:.0f}×**. "
            f"This is not a performance bottleneck — the model cannot be loaded at all. "
            f"Switch back to Cloud (H100) to see a feasible execution, or use the toggle "
            f"to continue exploring the hardware boundary."
        ), kind="danger")
    else:
        _oom_banner = mo.md("")
    _oom_banner
    return


# ─── ACT II: PREDICTION REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    H100_BW_GBS,
    H100_TFLOPS,
    RESNET50_DATA_MB,
    RESNET50_GFLOPS,
    _T_comp_ms,
    _T_mem_ms,
    act2_pred,
    mo,
):
    # Reveal is always computed for H100 context (the prediction question specifies H100)
    _actual_bw   = H100_BW_GBS           # GB/s
    _actual_tflops = H100_TFLOPS         # TFLOPs
    _data_gb = RESNET50_DATA_MB / 1000.0
    _ops_gflops = RESNET50_GFLOPS

    _h100_mem_ms  = (_data_gb / _actual_bw) * 1000.0
    _h100_comp_ms = (_ops_gflops / (_actual_tflops * 1000.0)) * 1000.0
    _h100_dominant = "memory" if _h100_mem_ms > _h100_comp_ms else "compute"

    _correct_act2 = act2_pred.value == "B"

    if _correct_act2:
        _reveal2 = mo.callout(mo.md(
            f"**Correct.** On the H100 at batch=1, the Iron Law is **memory-bound**. "
            f"Memory term (D/BW) = {_h100_mem_ms:.4f} ms vs. Compute term (O/R) = {_h100_comp_ms:.5f} ms. "
            f"The H100's 989 TFLOPs of compute finishes the 4 GFLOPs in {_h100_comp_ms*1000:.2f} microseconds, "
            f"but loading 100 MB through 3350 GB/s takes {_h100_mem_ms*1000:.0f} microseconds. "
            f"The data movement is ~{_h100_mem_ms/_h100_comp_ms:.0f}× slower than the arithmetic. "
            f"This is the **Memory Wall** — it persists even on the fastest accelerators at small batch sizes."
        ), kind="success")
    elif act2_pred.value == "A":
        _reveal2 = mo.callout(mo.md(
            f"**Incorrect — the opposite is true.** "
            f"The H100 is so fast at arithmetic (989 TFLOPs) that 4 GFLOPs takes only "
            f"{_h100_comp_ms*1000:.2f} microseconds. But 100 MB through 3350 GB/s takes "
            f"{_h100_mem_ms*1000:.0f} microseconds. The memory term is "
            f"~{_h100_mem_ms/_h100_comp_ms:.0f}× larger. "
            f"At batch=1, almost every modern inference workload is memory-bound on cloud hardware."
        ), kind="warn")
    elif act2_pred.value == "C":
        _reveal2 = mo.callout(mo.md(
            f"**Not quite.** The two terms are not equal — they differ by "
            f"~{_h100_mem_ms/_h100_comp_ms:.0f}×. "
            f"Memory = {_h100_mem_ms:.4f} ms, Compute = {_h100_comp_ms:.5f} ms. "
            f"The H100's extreme compute throughput makes the arithmetic trivially fast "
            f"at batch=1, while data movement time is determined by the memory bandwidth "
            f"ceiling, which cannot be exceeded."
        ), kind="warn")
    else:
        _reveal2 = mo.callout(mo.md(
            f"**Not quite.** The prediction question specifies H100 at batch=1, so the "
            f"answer is deterministic for that context. On the H100, the memory term "
            f"({_h100_mem_ms:.4f} ms) dominates the compute term ({_h100_comp_ms:.5f} ms) "
            f"by ~{_h100_mem_ms/_h100_comp_ms:.0f}×. "
            f"On the Cortex-M7, the memory term also dominates — but both terms are much "
            f"larger, and the model is infeasible anyway (OOM). "
            f"The binding constraint at batch=1 is always the Memory Wall."
        ), kind="warn")

    _reveal2
    return


# ─── ACT II: MATH PEEK ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "View the Iron Law — T = D/BW + O/R + L": mo.md("""
        **The Iron Law of ML Systems** (@sec-introduction-iron-law-ml-systems-c32a):

        ```
        T_total = D_vol/BW  +  O/(R_peak × η)  +  L_lat
                  ────────     ─────────────────    ─────
                  Data Term    Compute Term         Overhead
        ```

        **Variable definitions:**

        | Symbol | Meaning | ResNet-50 Value | H100 Value |
        |--------|---------|----------------|------------|
        | `D_vol` | Data volume moved (weights + activations) | 100 MB = 0.1 GB | — |
        | `BW` | Memory bandwidth | — | 3,350 GB/s |
        | `O` | Arithmetic operations | 4 GFLOPs | — |
        | `R_peak` | Peak compute throughput | — | 989 TFLOPs = 989,000 GFLOPs/s |
        | `η` | Hardware utilization (assumed 1.0 here) | — | 1.0 |
        | `L_lat` | Dispatch / overhead latency | — | 0.01 ms |

        **The systems conclusion (from @sec-introduction):**
        At batch=1, the compute term becomes negligible on high-throughput accelerators.
        The H100's 989 TFLOPs finishes 4 GFLOPs in ~4 nanoseconds. The memory wall
        (loading 100 MB at 3350 GB/s) takes ~30 microseconds — 7,500× longer.
        This is why inference serving systems use **batching**: grouping requests
        increases `O` without proportionally increasing `D_vol`, moving the workload
        from the memory-bound regime toward the compute-bound regime.
        """)
    })
    return


# ─── ACT II: STRUCTURED REFLECTION ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Act II Reflection")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) Too slow — the TFLOPS count is too low": "A",
            "B) Memory capacity — the model exceeds available SRAM": "B",
            "C) Power — the microcontroller overheats": "C",
            "D) Accuracy — quantization degrades the model": "D",
        },
        label="What is the PRIMARY constraint preventing ResNet-50 from running on a Cortex-M7 microcontroller?",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(MCU_SRAM_MB, RESNET50_DATA_MB, act2_reflect, mo):
    mo.stop(
        act2_reflect.value is None,
        mo.md(""),
    )

    if act2_reflect.value == "B":
        _fb2 = mo.callout(mo.md(
            f"**Correct.** ResNet-50 requires {RESNET50_DATA_MB:.0f} MB of RAM. "
            f"The Cortex-M7 has {MCU_SRAM_MB * 1000:.0f} KB of SRAM — a {RESNET50_DATA_MB / MCU_SRAM_MB:.0f}× "
            f"deficit. This is not a question of speed: the model literally cannot be loaded. "
            f"Memory capacity is the absolute constraint. This is why TinyML requires "
            f"completely different model architectures (MobileNet, EfficientNet-Lite, "
            f"quantized INT8 networks) that fit within 100–200 KB, not 100 MB."
        ), kind="success")
    elif act2_reflect.value == "A":
        _fb2 = mo.callout(mo.md(
            f"**True, but not the primary constraint.** Yes, the Cortex-M7's 0.001 TFLOPs "
            f"would make ResNet-50 extremely slow (~4 seconds per inference). But it would "
            f"never reach that calculation, because it cannot load the model into RAM first. "
            f"Memory capacity failure precedes any speed analysis."
        ), kind="warn")
    elif act2_reflect.value == "C":
        _fb2 = mo.callout(mo.md(
            "**Not the primary constraint.** At 0.1 W, the Cortex-M7 operates well within "
            "its thermal envelope — microcontrollers are designed for sustained embedded "
            "operation. Power is not the binding constraint here. Memory capacity is."
        ), kind="warn")
    else:
        _fb2 = mo.callout(mo.md(
            "**Incorrect.** Quantization degrades accuracy but solves the memory problem — "
            "a quantized INT8 ResNet-50 still requires ~25 MB, still exceeding 512 KB "
            "by 50×. Accuracy tradeoff is downstream of feasibility. The model must first "
            "fit in memory before any inference can occur."
        ), kind="warn")

    _fb2
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
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
                    <strong>1. The 10&sup6; compute gap is an architectural boundary, not an optimization problem.</strong>
                    The H100 delivers 989 TFLOPs; the Cortex-M7 delivers 0.001 TFLOPs &mdash; a 989,000&times; gap.
                    No compiler flag bridges this. The gap forces separate model architectures,
                    separate compilation targets, and separate runtime stacks across the four deployment paradigms.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. At batch=1, cloud accelerators are memory-bound &mdash; not compute-bound.</strong>
                    The H100 finishes 4 GFLOPs of ResNet-50 arithmetic in ~4 nanoseconds, but moving
                    100 MB through HBM3 takes ~30 microseconds. The Memory Wall persists even on the
                    fastest hardware when batch size is 1; batching amortizes data movement across requests.
                </div>
                <div>
                    <strong>3. Memory capacity, not compute speed, is the binding constraint for TinyML feasibility.</strong>
                    ResNet-50 requires 100 MB of RAM. The Cortex-M7 has 512 KB &mdash; a 200&times; deficit.
                    The model cannot be loaded at all. Compute speed is irrelevant until memory fits.
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
                    <strong>Lab 02: The Iron Law.</strong> This lab revealed that the H100 is
                    memory-bound at batch=1. Lab 02 asks: by exactly how much does a $2M hardware
                    upgrade improve latency for a memory-bound workload &mdash; and when does
                    a physical law make cloud inference categorically impossible?
                </div>
            </div>

            <!-- Textbook & TinyTorch -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-introduction for the D&middot;A&middot;M Taxonomy,
                    Hardware Twins, and the Degradation Equation.<br/>
                    <strong>Build:</strong> TinyTorch Module 01 &mdash; implement a minimal forward-pass
                    engine and observe the Iron Law&apos;s three terms in profiling output.
                    See <code>tinytorch/src/01_foundations/</code>.
                </div>
            </div>

        </div>
        """),


        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. The H100 has ~2,000,000x more compute than the Cortex-M7. Why does this gap force separate model architectures and runtime stacks rather than a single universal stack?

    2. At batch=1, which term of the Iron Law (T = D/BW + O/R + L) dominates for ResNet-50 on the H100 — and what does that tell you about which hardware resource to upgrade?

    3. Silent degradation: a recommendation system shows 100% uptime and <50 ms P99 latency for 6 months while its accuracy falls below an acceptable floor. What monitoring gap allows this, and what threshold parameter from Act II would have caught it earlier?

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ───────────────────────────────────────────────────────
@app.cell
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    COLORS,
    _T_comp_ms,
    _T_mem_ms,
    _bottleneck,
    _oom,
    act1_pred,
    act2_context,
    act2_pred,
    ledger,
    mo,
, decision_input, decision_ui):
    # ── Save chapter 1 results to Design Ledger
    _act1_correct = act1_pred.value == "C"
    _act2_correct = act2_pred.value == "B"

    ledger.save(
        chapter=1,
        design={
            "context": act2_context.value,
            "act1_prediction": act1_pred.value,
            "act1_correct": _act1_correct,
            "act2_bottleneck": _bottleneck,
            "act2_prediction": act2_pred.value,
            "act2_correct": _act2_correct,
            "constraint_hit": bool(_oom),
        "student_justification": str(decision_input.value),
            "oom_triggered": bool(_oom),
        },
    )

    # ── HUD footer
    _act1_status = "correct" if _act1_correct else "incorrect"
    _act2_status = "correct" if _act2_correct else "incorrect"
    _ctx_display = act2_context.value.upper()
    _oom_display = "YES" if _oom else "NO"
    _bn_display  = _bottleneck.upper() if not _oom else "OOM"

    _hud_color_act1 = COLORS["GreenLine"] if _act1_correct else COLORS["RedLine"]
    _hud_color_act2 = COLORS["GreenLine"] if _act2_correct else COLORS["RedLine"]
    _hud_color_oom  = COLORS["RedLine"] if _oom else COLORS["GreenLine"]

    mo.Html(f"""
    <div class="lab-hud">
        <div>
            <span class="hud-label">LAB </span>
            <span class="hud-value">01 · Magnitude Awakening</span>
        </div>
        <div>
            <span class="hud-label">CONTEXT </span>
            <span class="hud-value">{_ctx_display}</span>
        </div>
        <div>
            <span class="hud-label">ACT I PRED </span>
            <span style="color:{_hud_color_act1}; font-weight:700;">{act1_pred.value} ({_act1_status})</span>
        </div>
        <div>
            <span class="hud-label">ACT II PRED </span>
            <span style="color:{_hud_color_act2}; font-weight:700;">{act2_pred.value} ({_act2_status})</span>
        </div>
        <div>
            <span class="hud-label">BOTTLENECK </span>
            <span class="hud-value">{_bn_display}</span>
        </div>
        <div>
            <span class="hud-label">OOM </span>
            <span style="color:{_hud_color_oom}; font-weight:700;">{_oom_display}</span>
        </div>
        <div>
            <span class="hud-label">LEDGER </span>
            <span class="hud-active">CH01 SAVED</span>
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
