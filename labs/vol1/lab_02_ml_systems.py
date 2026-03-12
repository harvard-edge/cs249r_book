import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 02: THE IRON LAW — PHYSICS OF PERFORMANCE
#
# Chapter: ml_systems.qmd  (@sec-ml-systems)
# Core Invariant: T = D/BW + O/R + L
#
# Two Acts:
#   Act I  — The Memory Wall Revelation (12-15 min)
#     The $2M H100 upgrade only sped up the compute term. Memory term dominates.
#     First introduction of the Latency Waterfall instrument.
#
#   Act II — The Light Barrier (20-25 min)
#     Speed of light in fiber sets a hard latency floor.
#     AV use case: 3000 km datacenter = 30 ms propagation delay. SLA = 10 ms.
#     Failure state: propagation delay alone violates the SLA budget.
#
# Design Ledger: chapter=2, context, act1 prediction, bottleneck_type, sla_violated
#
# Hardware constants (all from @sec-ml-systems-deployment-spectrum-71be and NVIDIA specs):
#   H100_BW_GBS      = 3350    # H100 SXM5 HBM3e, NVIDIA spec
#   H100_TFLOPS_FP16 = 989     # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
#   A100_BW_GBS      = 2000    # A100 SXM4 HBM2e, NVIDIA spec (approx 1935, rounded)
#   ORIN_BW_GBS      = 102     # Jetson Orin NX, NVIDIA spec
#   ORIN_TOPS        = 100     # INT8 TOPS, Jetson Orin NX
#   FIBER_SPEED_KMS  = 200000  # km/s speed of light in fiber (~0.67c),
#                              # from @eq-latency-physics in ml_systems.qmd
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP (hide_code=False — leave visible for instructor inspection) ─


@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np
    import math

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    # ── Hardware constants (NVIDIA published specs + chapter equations) ──
    # H100 SXM5: https://www.nvidia.com/en-us/data-center/h100/
    H100_BW_GBS = 3350       # GB/s  HBM3e memory bandwidth
    H100_TFLOPS_FP16 = 989   # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_RAM_GB = 80         # GB  HBM capacity

    # A100 SXM4: prior-generation reference for CTO upgrade scenario
    A100_BW_GBS = 2000       # GB/s  HBM2e (actual: 1935, rounded up for scenario)
    A100_TFLOPS_FP16 = 312   # TFLOPS  FP16 Tensor Core peak

    # Jetson Orin NX: NVIDIA embedded inference platform
    ORIN_BW_GBS = 102        # GB/s  LPDDR5 memory bandwidth
    ORIN_TOPS = 100          # TOPS  INT8 equivalent
    ORIN_RAM_GB = 16         # GB

    # Speed of light in fiber — from @eq-latency-physics (ml_systems.qmd):
    #   Latency_min = 2 * Distance / (0.67 * c) ≈ 2 * Distance / 200,000 km/s
    FIBER_SPEED_KM_S = 200_000  # km/s  (~0.67c, refractive index ~1.5)

    # Overhead constant — fixed dispatch / PCIe / driver tax per inference call
    # Typical DNN framework dispatch overhead: 0.3–1.0 ms
    # Source: @sec-ml-systems-architectural-anchor (OS/Runtime layer dispatch)
    OVERHEAD_MS = 0.5        # ms  fixed overhead per inference call

    ledger = DesignLedger()

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB,
        A100_BW_GBS, A100_TFLOPS_FP16,
        ORIN_BW_GBS, ORIN_TOPS, ORIN_RAM_GB,
        FIBER_SPEED_KM_S, OVERHEAD_MS,
    )


# ─── CELL 1: HEADER ───────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume I · Lab 02
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Iron Law
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                T = D/BW + O/R + L
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Every millisecond of inference latency decomposes into three physical terms:
                data movement, arithmetic, and fixed overhead. The term that dominates
                determines where optimization effort pays off — and where $2M upgrades disappear.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    2 Acts · 35–40 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter: @sec-ml-systems
                </span>
                <span style="background: rgba(0,143,69,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(0,143,69,0.25);">
                    First Instrument: Latency Waterfall
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">T = D/BW + O/R + L</span>
                <span class="badge badge-warn">Memory Wall</span>
                <span class="badge badge-fail">Light Barrier</span>
            </div>
        </div>
        """),
    ])
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
                <div style="margin-bottom: 3px;">1. <strong>Diagnose why a 6&times; compute upgrade yields only ~8% latency improvement</strong> for a memory-bound workload, using the Bottleneck Principle to quantify that the Memory term (D/BW) accounts for &gt;99% of inference latency at AI&nbsp;=&nbsp;5&nbsp;FLOPs/Byte.</div>
                <div style="margin-bottom: 3px;">2. <strong>Identify the Ridge Point</strong> &mdash; the Arithmetic Intensity threshold (AI&nbsp;=&nbsp;R/BW&nbsp;&asymp;&nbsp;590 FLOPs/Byte on H100) where a workload transitions from memory-bound to compute-bound, and predict which batch sizes cross it.</div>
                <div style="margin-bottom: 3px;">3. <strong>Calculate the propagation delay floor</strong> for a 3,000 km datacenter and determine at what distance the speed of light makes a 10 ms AV safety SLA physically impossible to satisfy from the cloud.</div>
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
                    Iron Law equation T&nbsp;=&nbsp;D/BW&nbsp;+&nbsp;O/R&nbsp;+&nbsp;L
                    from @sec-introduction-iron-law-ml-systems-c32a &middot;
                    Arithmetic Intensity definition from @sec-ml-systems-deployment-spectrum-71be &middot;
                    Deployment spectrum constraints from @sec-ml-systems-architectural-anchor
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
                "If you double the compute power of your inference server, why doesn&apos;t
                latency halve &mdash; and when does the speed of light make cloud inference
                physically impossible?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-ml-systems-deployment-spectrum-71be** (Physical Constraints) — The Light Barrier equation
      and Memory Wall divergence rates that govern both acts.
    - **@sec-ml-systems-architectural-anchor** (Single-Node Stack) — The four layers and the
      Iron Law's relationship to memory bandwidth at the hardware layer.
    - **Iron Law** from @sec-introduction-iron-law-ml-systems-c32a — The equation
      `T = D/BW + O/R + L` with variable definitions.
    """), kind="info")
    return


# ─── CELL 4: CONTEXT TOGGLE ────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "☁️  Cloud (H100 · 3,350 GB/s · 989 TFLOPS FP16)": "cloud",
            "🤖  Edge  (Jetson Orin NX · 102 GB/s · 100 TOPS)":   "edge",
        },
        value="☁️  Cloud (H100 · 3,350 GB/s · 989 TFLOPS FP16)",
        label="Deployment context (sets hardware for all computations):",
        inline=True,
    )
    context_toggle
    return (context_toggle,)


# ═════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "I"
    _act_color    = COLORS["BlueLine"]
    _act_title    = "The Memory Wall Revelation"
    _act_duration = "12–15 min"
    _act_why      = (
        "You expect a faster GPU to reduce latency proportionally to its TFLOPS gain. "
        "The data will show that when Arithmetic Intensity sits far below the Ridge Point "
        "(AI = 5 vs. ridge &asymp; 590 FLOPs/Byte on H100), the Memory term D/BW accounts for "
        "&gt;99% of total latency &mdash; and the $2M compute upgrade attacked the wrong term."
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
def _(mo, COLORS):
    _color = COLORS["BlueLine"]
    _bg    = COLORS["BlueL"]
    mo.Html(f"""
    <div style="border-left:4px solid {_color}; background:{_bg};
                border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
        <div style="font-size:0.72rem; font-weight:700; color:{_color};
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
            Incoming Message · CTO, NovaSight AI
        </div>
        <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
            "We spent $2M upgrading our inference cluster from A100 to H100. Compute
            throughput is 6× higher on paper. But end-to-end inference latency dropped
            by only 8%. My engineering team cannot explain why. Every benchmark says the
            H100 is faster. I need to understand what we're actually paying for."
        </div>
        <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
            — Dr. Priya Nair, CTO · NovaSight AI (vision inference, 40M API calls/day)
        </div>
    </div>
    """)
    return


# ─── ACT I: CONCEPT SETUP ─────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("""
        ## The Iron Law Decomposes Latency into Three Terms

        The **Iron Law of ML Systems** states that total inference time has three additive components:

        ```
        T  =  D/BW  +  O/R  +  L
              ────     ───     ─
              Memory   Compute  Overhead
              term     term     term
        ```

        Each term can only be reduced by attacking the physical resource it depends on.
        Doubling compute throughput (R) only speeds up the **Compute term** — it does nothing
        to the **Memory term** (D/BW). When a workload is memory-bandwidth-bound, the Memory
        term dominates. A faster GPU changes only the smaller term.

        **Arithmetic Intensity (AI)** is the ratio of compute operations to data moved:

        ```
        AI  =  O / D      (FLOPs per byte loaded from memory)
        ```

        The **Ridge Point** is where the Memory term equals the Compute term — where
        `D/BW = O/R`, or equivalently `AI_ridge = R / BW`. Below the ridge point,
        the workload is memory-bound. Above it, compute-bound.
        """),
        mo.callout(mo.md(
            "**First Instrument Introduction — The Latency Waterfall.** "
            "The three bars in the chart below correspond directly to the three terms "
            "of the Iron Law: D/BW (memory), O/R (compute), and L (overhead). "
            "Watch how their relative heights shift as you change Arithmetic Intensity below."
        ), kind="info"),
    ])
    return


# ─── ACT I: PREDICTION LOCK ───────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A) Halve inference latency (6× compute → ~2× total latency reduction)": "halve",
            "B) Reduce latency by ~10–15% (compute is a minority term)":              "ten_pct",
            "C) Have no measurable effect on latency (model is fully memory-bound)":  "no_effect",
            "D) Double throughput while latency stays the same":                      "throughput",
        },
        label="""**Prediction Lock — Act I.**
A transformer inference workload has Arithmetic Intensity = 5 FLOPs/Byte.
The H100 ridge point is R/BW = 989 TFLOPS / 3,350 GB/s ≈ 295 FLOPs/Byte.
The A100 ridge point is 312 TFLOPS / 2,000 GB/s ≈ 156 FLOPs/Byte.

Upgrading from A100 to H100 multiplies peak compute (R) by 6×, but only increases
memory bandwidth (BW) by 1.7×. For this workload, upgrading from A100 to H100 will:""",
    )
    act1_prediction
    return (act1_prediction,)


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(
        act1_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Latency Waterfall instruments."),
            kind="warn",
        )
    )
    return


# ─── ACT I: INSTRUMENTS ───────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(act1_prediction.value is None)

    act1_ai_slider = mo.ui.slider(
        start=1, stop=200, value=5, step=1,
        label="Arithmetic Intensity (FLOPs/Byte)",
    )
    act1_ai_slider
    return (act1_ai_slider,)


@app.cell(hide_code=True)
def _(
    mo, act1_prediction, act1_ai_slider, context_toggle,
    go, apply_plotly_theme, COLORS,
    H100_BW_GBS, H100_TFLOPS_FP16,
    A100_BW_GBS, A100_TFLOPS_FP16,
    ORIN_BW_GBS, ORIN_TOPS,
    OVERHEAD_MS,
):
    mo.stop(act1_prediction.value is None)

    # ── Hardware selection ────────────────────────────────────────────────────
    _ctx = context_toggle.value
    if _ctx == "cloud":
        _bw_new  = H100_BW_GBS       # GB/s
        _r_new   = H100_TFLOPS_FP16  # TFLOPS = 10^12 FLOPs/s
        _bw_old  = A100_BW_GBS
        _r_old   = A100_TFLOPS_FP16
        _hw_new  = "H100"
        _hw_old  = "A100"
    else:
        # Edge context: show Orin vs hypothetical "upgraded" Orin with 2× compute
        _bw_new  = ORIN_BW_GBS
        _r_new   = ORIN_TOPS         # TOPS ≈ TFLOPS for this comparison
        _bw_old  = ORIN_BW_GBS * 0.5  # hypothetical prior Orin
        _r_old   = ORIN_TOPS * 0.2
        _hw_new  = "Jetson Orin NX"
        _hw_old  = "Prior Orin (20% compute)"

    # ── Iron Law calculation ──────────────────────────────────────────────────
    # Model parameters for a representative transformer layer inference:
    #   D  = 2 GB (weights loaded per inference for a ~1B param slice in FP16)
    #   O  = AI * D  (by definition of Arithmetic Intensity)
    # Source: @sec-ml-systems-architectural-anchor (Memory Wall)
    _AI = act1_ai_slider.value       # FLOPs/Byte — user control
    _D_GB  = 2.0                     # GB data moved per inference call
    _D_bytes = _D_GB * 1e9           # bytes

    # Operations = AI × data
    _O_flops = _AI * _D_bytes        # FLOPs

    # ── New hardware (H100 or Orin) ──────────────────────────────────────────
    _bw_new_bps  = _bw_new  * 1e9   # bytes/s
    _r_new_fps   = _r_new   * 1e12  # FLOPs/s

    _t_mem_new_ms  = (_D_bytes / _bw_new_bps)  * 1000   # ms
    _t_comp_new_ms = (_O_flops / _r_new_fps)   * 1000   # ms
    _t_ovh_new_ms  = OVERHEAD_MS                         # ms (fixed)
    _t_total_new   = _t_mem_new_ms + _t_comp_new_ms + _t_ovh_new_ms

    # ── Old hardware (A100 or prior Orin) ───────────────────────────────────
    _bw_old_bps  = _bw_old  * 1e9
    _r_old_fps   = _r_old   * 1e12

    _t_mem_old_ms  = (_D_bytes / _bw_old_bps)  * 1000
    _t_comp_old_ms = (_O_flops / _r_old_fps)   * 1000
    _t_ovh_old_ms  = OVERHEAD_MS
    _t_total_old   = _t_mem_old_ms + _t_comp_old_ms + _t_ovh_old_ms

    # ── Ridge points ─────────────────────────────────────────────────────────
    # Ridge point = R / BW  (FLOPs/Byte at which compute term = memory term)
    _ridge_new = (_r_new * 1e12) / (_bw_new * 1e9)   # FLOPs/Byte
    _ridge_old = (_r_old * 1e12) / (_bw_old * 1e9)

    # ── Bottleneck classification ─────────────────────────────────────────────
    _is_mem_bound_new = _AI < _ridge_new
    _bottleneck_new   = "Memory-bound" if _is_mem_bound_new else "Compute-bound"
    _bottleneck_color = COLORS["RedLine"] if _is_mem_bound_new else COLORS["BlueLine"]

    # ── Latency improvement ───────────────────────────────────────────────────
    _pct_improvement = (1 - _t_total_new / _t_total_old) * 100 if _t_total_old > 0 else 0

    # ── Build Latency Waterfall chart ─────────────────────────────────────────
    # This is the FIRST appearance of the Latency Waterfall instrument.
    # Three bars: Memory (D/BW), Compute (O/R), Overhead (L)
    # Color coding: Memory=Red (often limiting), Compute=Blue, Overhead=Orange

    _bar_colors_new = [
        COLORS["RedLine"]    if _is_mem_bound_new  else COLORS["BlueLine"],  # mem bar
        COLORS["BlueLine"]   if _is_mem_bound_new  else COLORS["RedLine"],   # comp bar
        COLORS["OrangeLine"],                                                  # overhead
    ]
    _bar_colors_old = [COLORS["Grey"], COLORS["Grey"], COLORS["Grey"]]

    _fig = go.Figure()

    # Old hardware bars
    _fig.add_trace(go.Bar(
        name=_hw_old,
        x=["Memory (D/BW)", "Compute (O/R)", "Overhead (L)"],
        y=[_t_mem_old_ms, _t_comp_old_ms, _t_ovh_old_ms],
        marker_color=_bar_colors_old,
        opacity=0.45,
        width=0.35,
        offset=-0.2,
        text=[f"{v:.2f} ms" for v in [_t_mem_old_ms, _t_comp_old_ms, _t_ovh_old_ms]],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
    ))

    # New hardware bars
    _fig.add_trace(go.Bar(
        name=_hw_new,
        x=["Memory (D/BW)", "Compute (O/R)", "Overhead (L)"],
        y=[_t_mem_new_ms, _t_comp_new_ms, _t_ovh_new_ms],
        marker_color=_bar_colors_new,
        width=0.35,
        offset=0.15,
        text=[f"{v:.2f} ms" for v in [_t_mem_new_ms, _t_comp_new_ms, _t_ovh_new_ms]],
        textposition="outside",
        textfont=dict(size=10, color="#1e293b", family="SF Mono, monospace"),
    ))

    _fig.update_layout(
        barmode="overlay",
        height=340,
        yaxis=dict(title="Latency (ms)", gridcolor="#f1f5f9"),
        xaxis=dict(title="Iron Law Term"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=True,
    )
    apply_plotly_theme(_fig)

    # ── Physics formula display ───────────────────────────────────────────────
    _formula_block = f"""
    **Iron Law — Live Calculation** (`AI = {_AI} FLOPs/Byte, D = {_D_GB:.1f} GB`)

    ```
    Memory  D/BW  =  {_D_GB:.1f} GB  /  {_bw_new:,} GB/s  =  {_t_mem_new_ms:.3f} ms   ← {_hw_new}
    Compute O/R   =  {_AI * _D_GB:.1f} GFLOPS  /  {_r_new:,} TFLOPS  =  {_t_comp_new_ms:.4f} ms
    Overhead  L   =  {_t_ovh_new_ms:.1f} ms  (fixed dispatch tax)
    ─────────────────────────────────────────────────────────────
    Total    T    =  {_t_total_new:.3f} ms   (vs {_t_total_old:.3f} ms on {_hw_old})

    Ridge Point  =  R / BW  =  {_ridge_new:,.0f} FLOPs/Byte   ({_hw_new})
    AI = {_AI} FLOPs/Byte  →  {'BELOW ridge → Memory-bound' if _is_mem_bound_new else 'ABOVE ridge → Compute-bound'}
    ```
    """

    # ── Metric cards ──────────────────────────────────────────────────────────
    _mem_pct  = _t_mem_new_ms  / _t_total_new * 100 if _t_total_new > 0 else 0
    _comp_pct = _t_comp_new_ms / _t_total_new * 100 if _t_total_new > 0 else 0
    _ovh_pct  = _t_ovh_new_ms  / _t_total_new * 100 if _t_total_new > 0 else 0

    _mem_col  = COLORS["RedLine"]    if _mem_pct  > 50 else COLORS["OrangeLine"]
    _comp_col = COLORS["GreenLine"]  if _comp_pct < 30 else COLORS["BlueLine"]
    _impr_col = COLORS["GreenLine"]  if _pct_improvement > 20 else COLORS["OrangeLine"]

    _cards_html = f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin:16px 0;">
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:160px; text-align:center; background:white;
                    border-top:3px solid {_bottleneck_color};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Bottleneck
            </div>
            <div style="font-size:1.2rem; font-weight:800; color:{_bottleneck_color};">
                {_bottleneck_new}
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                AI={_AI} vs ridge={_ridge_new:,.0f}
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:160px; text-align:center; background:white;
                    border-top:3px solid {_mem_col};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Memory Share
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{_mem_col};">
                {_mem_pct:.0f}%
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                of total latency
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:160px; text-align:center; background:white;
                    border-top:3px solid {_comp_col};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Compute Share
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{_comp_col};">
                {_comp_pct:.0f}%
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                of total latency
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:160px; text-align:center; background:white;
                    border-top:3px solid {_impr_col};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Upgrade Benefit
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{_impr_col};">
                {_pct_improvement:.1f}%
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                latency reduction
            </div>
        </div>
    </div>
    """

    mo.vstack([
        mo.md("### Latency Waterfall — Three Terms of the Iron Law"),
        mo.md(
            "_Each bar represents one term of T = D/BW + O/R + L. "
            "Faded bars = prior hardware. Solid bars = current hardware. "
            "The dominant bar is your bottleneck._"
        ),
        mo.as_html(_fig),
        mo.Html(_cards_html),
        mo.md(_formula_block),
    ])
    # Export bottleneck for downstream cells
    return (
        _bottleneck_new,
        _pct_improvement,
        _ridge_new,
        _t_mem_new_ms,
        _t_comp_new_ms,
        _t_total_new,
        _t_total_old,
        _is_mem_bound_new,
    )


# ─── ACT I: REVEAL — PREDICTION vs REALITY ────────────────────────────────────


@app.cell(hide_code=True)
def _(
    mo, act1_prediction,
    _bottleneck_new, _pct_improvement, _ridge_new,
    _t_mem_new_ms, _t_comp_new_ms, _t_total_new,
    _is_mem_bound_new, COLORS,
    act1_ai_slider,
):
    _ai = act1_ai_slider.value
    _pred = act1_prediction.value

    # Correct answer depends on the current AI value:
    # At AI=5 (well below ridge ~590), model is memory-bound → answer is B or C
    # The "correct" answer in the prediction is B (10-15%) since the compute term
    # is not literally zero — it just shrinks from ~4% to ~0.7% of total.
    _correct = _pred in ("ten_pct", "no_effect")

    _actual_improvement = _pct_improvement
    _pred_labels = {
        "halve":      "A) Halve latency (~50%)",
        "ten_pct":    "B) ~10–15% reduction",
        "no_effect":  "C) No measurable effect",
        "throughput": "D) Double throughput only",
    }
    _pred_text = _pred_labels.get(_pred, _pred)

    if _correct and _pred == "ten_pct":
        _msg = (
            f"**Correct.** You predicted a modest improvement (~10–15%), and the physics "
            f"confirms it: the upgrade delivers **{_actual_improvement:.1f}%** latency reduction "
            f"at AI = {_ai} FLOPs/Byte. "
            f"The H100 ridge point is ≈ {_ridge_new:,.0f} FLOPs/Byte. "
            f"At AI = {_ai}, the workload sits {_ridge_new / _ai:.0f}× *below* the ridge — "
            f"firmly memory-bound. The Memory term ({_t_mem_new_ms:.2f} ms) dwarfs "
            f"the Compute term ({_t_comp_new_ms:.4f} ms). "
            f"Upgrading compute (R) from A100 to H100 sped up the *smaller* term. "
            f"The $2M upgrade made an already-tiny term even smaller."
        )
        _kind = "success"
    elif _correct and _pred == "no_effect":
        _msg = (
            f"**Close — the physics partially confirms this.** The actual improvement is "
            f"**{_actual_improvement:.1f}%**, not exactly zero, but your intuition is "
            f"correct: at AI = {_ai} FLOPs/Byte (ridge = {_ridge_new:,.0f}), the workload "
            f"is so deeply memory-bound that compute improvements produce negligible change. "
            f"The Memory term ({_t_mem_new_ms:.2f} ms) is the dominant term. "
            f"The Compute term is {_t_comp_new_ms:.4f} ms — nearly four orders of magnitude smaller."
        )
        _kind = "success"
    elif _pred == "halve":
        _msg = (
            f"**Not quite.** You predicted ~50% improvement (halving latency), which would require "
            f"both the Memory AND Compute terms to improve proportionally. "
            f"But at AI = {_ai} FLOPs/Byte, the workload is *memory-bound* "
            f"(ridge = {_ridge_new:,.0f} FLOPs/Byte). "
            f"The H100 upgrade increased memory bandwidth by only 1.7× (3,350 vs 2,000 GB/s), "
            f"not 6×. The dominant term (Memory: {_t_mem_new_ms:.2f} ms) barely moved. "
            f"Actual improvement: **{_actual_improvement:.1f}%**. "
            f"This is the CTO's $2M surprise."
        )
        _kind = "warn"
    else:  # throughput
        _msg = (
            f"**Not quite.** Throughput and latency are coupled through the Iron Law. "
            f"The H100 upgrade did improve peak FLOPs throughput by 6×, but for a "
            f"memory-bound workload (AI = {_ai}, ridge = {_ridge_new:,.0f}), the memory "
            f"term still dominates end-to-end latency. "
            f"Actual latency improvement: **{_actual_improvement:.1f}%**. "
            f"Higher compute throughput only helps if the Compute term is the bottleneck."
        )
        _kind = "warn"

    mo.vstack([
        mo.md(f"**You predicted:** {_pred_text}  |  **Actual improvement:** {_actual_improvement:.1f}%"),
        mo.callout(mo.md(_msg), kind=_kind),
    ])
    return


# ─── ACT I: REFLECTION ────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(act1_prediction.value is None)

    act1_reflection = mo.ui.radio(
        options={
            "A) Compute-bound — AI exceeds the ridge point":             "compute",
            "B) Memory-bound — AI is far below the ridge point":         "memory",
            "C) Overhead-bound — fixed dispatch tax dominates":          "overhead",
            "D) Balanced — both Memory and Compute terms are equal":     "balanced",
        },
        label="""**Reflection.** A transformer encoder running on H100 has Arithmetic Intensity = 5 FLOPs/Byte.
The H100 ridge point is approximately 295 FLOPs/Byte (R/BW = 989 TFLOPS / 3,350 GB/s).
This workload is:""",
    )
    act1_reflection
    return (act1_reflection,)


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, _ridge_new):
    mo.stop(act1_prediction.value is None or act1_reflection.value is None)

    _correct = act1_reflection.value == "memory"
    _feedbacks = {
        "compute": (
            "**Not quite.** Compute-bound means AI exceeds the ridge point, so the Compute term "
            "dominates. At AI = 5 FLOPs/Byte with a ridge point of ≈590 FLOPs/Byte, the workload "
            "is 118× *below* the ridge — deeply in the memory-bound regime. Compute-bound workloads "
            "are rare for transformer inference because weight loading generates low arithmetic intensity."
        ),
        "memory": (
            f"**Correct.** At AI = 5 FLOPs/Byte versus a ridge point of ≈{_ridge_new:,.0f} FLOPs/Byte, "
            "this workload sits far to the left of the ridge — firmly memory-bound. "
            "The Memory term (D/BW) dominates total latency. "
            "This is the regime where GPU TFLOPS upgrades produce minimal latency improvement, "
            "and where memory bandwidth (BW) is the engineering lever that actually matters. "
            "The CTO's $2M upgrade bought 6× more compute for a workload that did not need it."
        ),
        "overhead": (
            "**Not quite.** Overhead-bound means the fixed dispatch tax (L) exceeds both D/BW and O/R. "
            "That occurs at extremely small batch sizes or very short models. "
            "For a 2 GB weight-load scenario at AI = 5, the Memory term alone is multiple milliseconds "
            "— far larger than the 0.5 ms overhead constant."
        ),
        "balanced": (
            "**Not quite.** A balanced workload sits exactly at the ridge point (AI ≈ ridge), "
            f"where Memory term = Compute term. At AI = 5, the ridge is ≈{_ridge_new:,.0f} FLOPs/Byte — "
            "the workload is not balanced; it is deeply memory-bound, with the Compute term "
            "being orders of magnitude smaller than the Memory term."
        ),
    }

    mo.vstack([
        act1_reflection,
        mo.callout(
            mo.md(_feedbacks[act1_reflection.value]),
            kind="success" if _correct else "warn",
        ),
    ])
    return


# ─── ACT I: MATH PEEK ─────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(act1_prediction.value is None)

    mo.accordion({
        "The governing equation — Iron Law of ML Systems": mo.md("""
        **Formula:**

        $$T = \\frac{D}{BW} + \\frac{O}{R} + L$$

        **Variables:**

        | Symbol | Name | Units | Source |
        |--------|------|-------|--------|
        | T | Total inference latency | ms | Output |
        | D | Data volume moved (weights + activations) | GB | Model size × precision |
        | BW | Memory bandwidth | GB/s | Hardware spec |
        | O | Floating-point operations | FLOPs | 2 × parameters × batch |
        | R | Peak compute throughput × MFU | FLOPs/s | Hardware spec × efficiency |
        | L | Fixed overhead latency | ms | Dispatch + driver tax |

        **Arithmetic Intensity** (AI) = O / D (FLOPs/Byte)

        **Ridge Point** = R / BW (FLOPs/Byte — where Memory term = Compute term)

        - If AI < Ridge Point → **Memory-bound**: reduce D or increase BW
        - If AI > Ridge Point → **Compute-bound**: increase R or reduce O

        *From @sec-introduction-iron-law-ml-systems-c32a and @sec-ml-systems-deployment-spectrum-71be*
        """),
    })
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, COLORS):
    mo.stop(
        act1_prediction.value is None or act1_reflection.value is None,
        mo.callout(
            mo.md("Complete Act I (prediction + reflection) to unlock Act II."),
            kind="warn",
        )
    )

    _act_num      = "II"
    _act_color    = COLORS["RedLine"]
    _act_title    = "The Light Barrier"
    _act_duration = "20–25 min"
    _act_why      = (
        "Act I showed that memory bandwidth, not compute, is the binding constraint for "
        "transformer inference at low Arithmetic Intensity. "
        "Now discover a constraint that even memory bandwidth cannot fix: "
        "the speed of light in fiber sets a hard latency floor that no GPU upgrade "
        "can remove &mdash; and a 3,000 km datacenter already exceeds a 10 ms AV safety SLA "
        "before a single FLOP is computed."
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
def _(mo, act1_prediction, act1_reflection, COLORS):
    mo.stop(act1_prediction.value is None or act1_reflection.value is None)

    _color = COLORS["RedLine"]
    _bg    = COLORS["RedL"]
    mo.Html(f"""
    <div style="border-left:4px solid {_color}; background:{_bg};
                border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
        <div style="font-size:0.72rem; font-weight:700; color:{_color};
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
            Incoming Message · CTO, HelixDrive Autonomy
        </div>
        <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
            "Our perception pipeline processes LIDAR and camera frames in the cloud.
            The AI team says our 12 ms round-trip is acceptable for 'normal driving.'
            Safety certification requires a 10 ms end-to-end decision loop.
            We're planning to upgrade to faster GPUs. Will that fix it?"
        </div>
        <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
            — Marcus Chen, CTO · HelixDrive Autonomy (Level 4 AV, production fleet)
        </div>
    </div>
    """)
    return


# ─── ACT II: CONCEPT SETUP ────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection):
    mo.stop(act1_prediction.value is None or act1_reflection.value is None)

    mo.md("""
    ## The L Term: Propagation Delay Is Not Negotiable

    The Iron Law overhead term L has two components:

    ```
    L  =  L_dispatch  +  L_propagation
          ──────────     ──────────────
          Software tax   Physics tax
    ```

    The **dispatch overhead** (L_dispatch ≈ 0.5 ms) can be reduced with engineering.
    The **propagation delay** (L_propagation) cannot. It is bounded by the speed of light:

    ```
    L_propagation  =  2 × Distance / c_fiber
                   =  2 × Distance / 200,000 km/s
    ```

    At 3,000 km (approximate US coast-to-coast): 2 × 3,000 / 200,000 = **30 ms one-way minimum**.
    Faster GPUs speed up the Compute term (O/R). They do not move the datacenter closer.
    A GPU upgrade that shrinks the Compute term from 5 ms to 0.5 ms saves 4.5 ms.
    The propagation delay remains 30 ms.
    """)
    return


# ─── ACT II: PREDICTION LOCK ──────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection):
    mo.stop(act1_prediction.value is None or act1_reflection.value is None)

    act2_prediction = mo.ui.radio(
        options={
            "A) ~1 ms — fiber is nearly instantaneous at this scale": "1ms",
            "B) ~10 ms — close to the SLA budget":                    "10ms",
            "C) ~30 ms — speed of light in fiber over 3,000 km":      "30ms",
            "D) ~100 ms or more — cloud datacenter typical":           "100ms",
        },
        label="""**Prediction Lock — Act II.**
An autonomous vehicle sends a perception request to a datacenter 3,000 km away.
The speed of light in fiber is approximately 200,000 km/s.

The minimum round-trip propagation delay (no compute, just photons in fiber) is:""",
    )
    act2_prediction
    return (act2_prediction,)


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction):
    mo.stop(
        act1_prediction.value is None
        or act1_reflection.value is None
        or act2_prediction.value is None,
        mo.callout(
            mo.md("Select your Act II prediction to unlock the Light Barrier instruments."),
            kind="warn",
        )
    )
    return


# ─── ACT II: INSTRUMENTS ──────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction):
    mo.stop(
        act1_prediction.value is None
        or act1_reflection.value is None
        or act2_prediction.value is None,
    )

    act2_distance_slider = mo.ui.slider(
        start=0, stop=5000, value=3000, step=50,
        label="Distance to datacenter (km)",
    )
    act2_batch_slider = mo.ui.slider(
        start=1, stop=64, value=1, step=1,
        label="Inference batch size",
    )
    mo.hstack([act2_distance_slider, act2_batch_slider], justify="start", gap=4)
    return (act2_distance_slider, act2_batch_slider)


@app.cell(hide_code=True)
def _(
    mo, act1_prediction, act1_reflection, act2_prediction,
    act2_distance_slider, act2_batch_slider,
    context_toggle,
    go, apply_plotly_theme, COLORS,
    H100_BW_GBS, H100_TFLOPS_FP16,
    ORIN_BW_GBS, ORIN_TOPS,
    FIBER_SPEED_KM_S, OVERHEAD_MS,
):
    mo.stop(
        act1_prediction.value is None
        or act1_reflection.value is None
        or act2_prediction.value is None,
    )

    # ── Hardware selection ────────────────────────────────────────────────────
    _ctx = context_toggle.value
    if _ctx == "cloud":
        _bw_gbs   = H100_BW_GBS
        _r_tflops = H100_TFLOPS_FP16
        _hw_label = "H100 (Cloud)"
    else:
        _bw_gbs   = ORIN_BW_GBS
        _r_tflops = ORIN_TOPS
        _hw_label = "Jetson Orin NX (Edge)"

    # ── Parameters ───────────────────────────────────────────────────────────
    _dist_km    = act2_distance_slider.value
    _batch_size = act2_batch_slider.value
    _SLA_MS     = 10.0   # ms  — AV safety SLA from chapter scenario

    # ── Light Barrier computation ─────────────────────────────────────────────
    # Propagation delay = round-trip = 2 × Distance / c_fiber
    # Source: @eq-latency-physics in ml_systems.qmd
    #   Latency_min = 2 × Distance / 200,000 km/s
    _prop_ms = (2 * _dist_km / FIBER_SPEED_KM_S) * 1000   # ms

    # ── Iron Law: compute + memory terms for the inference workload ───────────
    # Representative AV perception model (e.g. ResNet-50 inference):
    #   D = 0.1 GB  (ResNet-50 weights in FP16 ≈ 50M params × 2B = 100 MB)
    #   AI = 20 FLOPs/Byte  (typical CNN inference AI)
    # Batch size scales both D and O linearly
    _D_GB_single    = 0.1         # GB  weights per inference
    _AI_perception  = 20.0        # FLOPs/Byte  typical CNN
    _D_GB_batch     = _D_GB_single * _batch_size
    _D_bytes_batch  = _D_GB_batch * 1e9
    _O_flops_batch  = _AI_perception * _D_bytes_batch

    _bw_bps   = _bw_gbs   * 1e9
    _r_fps    = _r_tflops * 1e12

    _t_mem_ms  = (_D_bytes_batch / _bw_bps) * 1000
    _t_comp_ms = (_O_flops_batch / _r_fps)  * 1000
    _t_ovh_ms  = OVERHEAD_MS

    # Total end-to-end latency (inference at edge/cloud + propagation for cloud)
    _t_inference_ms = _t_mem_ms + _t_comp_ms + _t_ovh_ms
    _t_total_ms     = _t_inference_ms + _prop_ms

    # ── SLA violation check ───────────────────────────────────────────────────
    _sla_violated   = _t_total_ms > _SLA_MS
    _prop_violates  = _prop_ms    > _SLA_MS   # propagation alone exceeds SLA

    # ── Throughput (samples/s for given batch) ────────────────────────────────
    # Throughput = batch_size / latency_inference_only (not including propagation)
    _throughput = _batch_size / (_t_inference_ms / 1000) if _t_inference_ms > 0 else 0

    # ── Build stacked latency chart ────────────────────────────────────────────
    _bar_h_color = COLORS["RedLine"]    if _sla_violated       else COLORS["GreenLine"]
    _prop_color  = COLORS["RedLine"]    if _prop_ms  > _SLA_MS  else COLORS["OrangeLine"]
    _mem_color   = COLORS["BlueLine"]
    _comp_color  = COLORS["BlueLine"]
    _ovh_color   = COLORS["OrangeLine"]

    _fig2 = go.Figure()

    # Stacked bars showing each component
    _components = [
        ("Memory (D/BW)",    _t_mem_ms,  _mem_color),
        ("Compute (O/R)",    _t_comp_ms, _comp_color),
        ("Dispatch (L)",     _t_ovh_ms,  _ovh_color),
        ("Propagation (RTT)", _prop_ms,  _prop_color),
    ]

    for _cname, _cval, _ccol in _components:
        _fig2.add_trace(go.Bar(
            name=_cname,
            x=["End-to-End Latency"],
            y=[_cval],
            marker_color=_ccol,
            text=[f"{_cval:.2f} ms"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=10, family="SF Mono, monospace"),
        ))

    # SLA reference line
    _fig2.add_hline(
        y=_SLA_MS,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        line_width=2,
        annotation_text=f"SLA = {_SLA_MS:.0f} ms",
        annotation_position="top right",
        annotation_font_color=COLORS["RedLine"],
        annotation_font_size=11,
    )

    _fig2.update_layout(
        barmode="stack",
        height=360,
        yaxis=dict(title="Latency (ms)", gridcolor="#f1f5f9"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", y=1.1, x=0),
        margin=dict(l=50, r=20, t=60, b=30),
    )
    apply_plotly_theme(_fig2)

    # ── Failure state banner ───────────────────────────────────────────────────
    if _prop_violates:
        _failure_banner = mo.callout(
            mo.md(
                f"**SLA VIOLATED — Propagation delay alone is {_prop_ms:.1f} ms. "
                f"The speed of light cannot be optimized.** "
                f"Even with zero compute time and zero overhead, the round-trip through "
                f"{_dist_km:,} km of fiber costs {_prop_ms:.1f} ms — "
                f"{_prop_ms / _SLA_MS:.1f}× the {_SLA_MS:.0f} ms SLA. "
                f"No GPU upgrade fixes this. The datacenter must move, or the model must move."
            ),
            kind="danger",
        )
    elif _sla_violated:
        _gap_ms = _t_total_ms - _SLA_MS
        _failure_banner = mo.callout(
            mo.md(
                f"**SLA VIOLATED — Total latency is {_t_total_ms:.1f} ms "
                f"({_gap_ms:.1f} ms over budget).** "
                f"Propagation ({_prop_ms:.1f} ms) + compute ({_t_inference_ms:.1f} ms) "
                f"exceeds the {_SLA_MS:.0f} ms limit. "
                f"Reducing distance is more effective than reducing compute at this scale."
            ),
            kind="danger",
        )
    else:
        _failure_banner = mo.callout(
            mo.md(
                f"**SLA met — Total latency is {_t_total_ms:.1f} ms "
                f"(budget: {_SLA_MS:.0f} ms, headroom: {_SLA_MS - _t_total_ms:.1f} ms).** "
                f"At {_dist_km:,} km, propagation delay is {_prop_ms:.1f} ms — "
                f"within budget at this range. Move the slider past the threshold to see the failure state."
            ),
            kind="success",
        )

    # ── Metric cards ──────────────────────────────────────────────────────────
    _prop_col2 = COLORS["RedLine"] if _prop_ms > _SLA_MS * 0.8 else COLORS["OrangeLine"]
    _total_col = COLORS["RedLine"] if _sla_violated else COLORS["GreenLine"]
    _tp_col    = COLORS["BlueLine"]

    _cards2_html = f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin:16px 0;">
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:160px; text-align:center; background:white;
                    border-top:3px solid {_prop_col2};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Propagation RTT
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{_prop_col2};">
                {_prop_ms:.1f} ms
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                2 × {_dist_km:,} km / {FIBER_SPEED_KM_S:,} km/s
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:160px; text-align:center; background:white;
                    border-top:3px solid {COLORS['BlueLine']};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Inference Latency
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{COLORS['BlueLine']};">
                {_t_inference_ms:.2f} ms
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                mem + compute + dispatch
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:160px; text-align:center; background:white;
                    border-top:3px solid {_total_col};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Total End-to-End
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{_total_col};">
                {_t_total_ms:.1f} ms
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                SLA = {_SLA_MS:.0f} ms {"VIOLATED" if _sla_violated else "MET"}
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:160px; text-align:center; background:white;
                    border-top:3px solid {_tp_col};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Throughput
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{_tp_col};">
                {_throughput:.0f}/s
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                batch={_batch_size} / inference only
            </div>
        </div>
    </div>
    """

    # ── Physics formula ────────────────────────────────────────────────────────
    _formula2 = f"""
    **Light Barrier — Live Calculation** (`distance = {_dist_km:,} km, batch = {_batch_size}`)

    ```
    L_propagation  =  2 × {_dist_km:,} km  /  {FIBER_SPEED_KM_S:,} km/s  =  {_prop_ms:.2f} ms  (RTT)
    T_inference    =  {_t_mem_ms:.3f} ms  +  {_t_comp_ms:.4f} ms  +  {_t_ovh_ms:.1f} ms  =  {_t_inference_ms:.3f} ms
    T_total        =  {_t_inference_ms:.3f} ms  +  {_prop_ms:.2f} ms  =  {_t_total_ms:.2f} ms
    SLA budget     =  {_SLA_MS:.0f} ms
    SLA status     =  {"VIOLATED (T_total > SLA)" if _sla_violated else "MET (T_total <= SLA)"}
    ```
    """

    mo.vstack([
        mo.md("### End-to-End Latency Breakdown — Iron Law with Light Barrier"),
        mo.md(
            "_The four stacked segments are the four latency components. "
            "The red dashed line is the 10 ms AV safety SLA. "
            "Move the Distance slider until the bar crosses the line._"
        ),
        mo.as_html(_fig2),
        _failure_banner,
        mo.Html(_cards2_html),
        mo.md(_formula2),
    ])
    # Export key values for downstream cells
    return (
        _sla_violated,
        _prop_ms,
        _t_total_ms,
        _t_inference_ms,
        _dist_km,
    )


# ─── ACT II: PREDICTION vs REALITY REVEAL ────────────────────────────────────


@app.cell(hide_code=True)
def _(
    mo, act2_prediction,
    _prop_ms, _sla_violated,
    FIBER_SPEED_KM_S,
):
    _pred = act2_prediction.value
    # At default 3000 km: 2 × 3000 / 200,000 = 30 ms
    _actual_prop_ms = _prop_ms

    _pred_values = {
        "1ms": 1,
        "10ms": 10,
        "30ms": 30,
        "100ms": 100,
    }
    _pred_ms = _pred_values.get(_pred, 0)
    _correct = _pred == "30ms"

    if _correct:
        _msg = (
            f"**Correct.** At 3,000 km, the minimum round-trip propagation delay is "
            f"2 × 3,000 / {FIBER_SPEED_KM_S:,} = **{_actual_prop_ms:.1f} ms**. "
            f"This is the minimum — before any compute, any queueing, any software overhead. "
            f"The AV's 10 ms SLA is physically impossible to satisfy with a 3,000 km datacenter. "
            f"No GPU upgrade changes this. This is why Edge ML exists as a deployment paradigm."
        )
        _kind = "success"
    elif _pred == "1ms":
        _msg = (
            f"**Not quite.** You predicted ~1 ms, which would require a fiber speed of "
            f"2 × 3,000 km / 0.001 s = 6,000,000 km/s — 20× faster than light. "
            f"Light travels at ~{FIBER_SPEED_KM_S:,} km/s in fiber. "
            f"The actual minimum round-trip is **{_actual_prop_ms:.1f} ms** — "
            f"already 3× the 10 ms SLA, before any computation."
        )
        _kind = "warn"
    elif _pred == "10ms":
        _msg = (
            f"**Close, but optimistic.** You predicted ~10 ms, which is the SLA limit. "
            f"The actual minimum is **{_actual_prop_ms:.1f} ms** — "
            f"exactly 3× the 10 ms SLA. At 3,000 km, physics forbids sub-10 ms round-trips "
            f"regardless of network equipment quality. Your prediction understated the distance "
            f"penalty by 3×."
        )
        _kind = "warn"
    else:  # 100ms
        _msg = (
            f"**Close to real-world overhead, but the minimum is lower.** "
            f"The physical minimum (photons in fiber only) is **{_actual_prop_ms:.1f} ms**. "
            f"Real cloud round-trips are indeed 60–150 ms due to software overhead, routing, "
            f"and queuing — which is closer to your estimate. "
            f"But even the physical floor (30 ms) already violates the 10 ms SLA. "
            f"The hardware and software overhead is the secondary problem; physics is the primary one."
        )
        _kind = "warn"

    mo.vstack([
        mo.md(f"**You predicted:** {_pred_ms} ms  |  **Actual at 3,000 km:** {_actual_prop_ms:.1f} ms"),
        mo.callout(mo.md(_msg), kind=_kind),
    ])
    return


# ─── ACT II: REFLECTION ───────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction):
    mo.stop(
        act1_prediction.value is None
        or act1_reflection.value is None
        or act2_prediction.value is None,
    )

    act2_reflection = mo.ui.radio(
        options={
            "A) Edge GPUs are faster than cloud GPUs for perception tasks": "faster_gpu",
            "B) Network latency violates real-time SLAs imposed by physics": "network",
            "C) Edge deployment is cheaper than cloud inference at scale":   "cheaper",
            "D) Cloud GPUs cannot run small models efficiently":             "small_model",
        },
        label="""**Reflection.** Based on the Light Barrier analysis, why does Edge ML
exist as a distinct deployment paradigm — rather than simply using faster cloud GPUs?""",
    )
    act2_reflection
    return (act2_reflection,)


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction, act2_reflection, _prop_ms):
    mo.stop(
        act1_prediction.value is None
        or act1_reflection.value is None
        or act2_prediction.value is None
        or act2_reflection.value is None,
    )

    _correct = act2_reflection.value == "network"
    _feedbacks2 = {
        "faster_gpu": (
            "**Not quite.** Edge GPUs are not faster than H100s for raw inference throughput — "
            "cloud GPUs are superior in compute capacity. "
            "Edge ML exists because *proximity* matters, not because edge silicon is faster. "
            "The constraint is the L term (propagation delay), not the O/R term (compute speed)."
        ),
        "network": (
            f"**Correct.** Edge ML exists because propagation delay is bounded by physics. "
            f"At 3,000 km, the minimum round-trip is {_prop_ms:.1f} ms — "
            "already exceeding a 10 ms AV safety SLA before any computation occurs. "
            "No hardware upgrade removes propagation delay. "
            "The only solution is to move the compute closer to the sensor — "
            "which is the defining engineering rationale for Edge ML. "
            "This is not a preference; it is a constraint analysis."
        ),
        "cheaper": (
            "**Not the primary reason.** Cost is a secondary consideration for real-time safety systems. "
            "An AV perception pipeline with a 10 ms SLA cannot use a 3,000 km datacenter at any price. "
            "The primary reason Edge ML exists is physical: propagation delay floors are immovable. "
            "Cost savings are a benefit, not the cause."
        ),
        "small_model": (
            "**Not quite.** Cloud GPUs can run small models efficiently — in fact, they excel at "
            "batching many small inference requests. "
            "Edge ML exists not because of model size constraints but because the L term "
            "(propagation delay) violates real-time latency SLAs. "
            "A 1-ms model on an H100 3,000 km away still costs 30 ms in propagation."
        ),
    }

    mo.vstack([
        act2_reflection,
        mo.callout(
            mo.md(_feedbacks2[act2_reflection.value]),
            kind="success" if _correct else "warn",
        ),
    ])
    return


# ─── ACT II: MATH PEEK ────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction):
    mo.stop(
        act1_prediction.value is None
        or act1_reflection.value is None
        or act2_prediction.value is None,
    )

    mo.accordion({
        "The governing equation — Light Barrier (Propagation Delay)": mo.md("""
        **Formula:**

        $$L_{\\text{propagation}} = \\frac{2 \\times \\text{Distance}}{c_{\\text{fiber}}}
          \\approx \\frac{2 \\times \\text{Distance}}{200{,}000 \\text{ km/s}}$$

        where $c_{\\text{fiber}} \\approx 0.67 \\times c$ due to fiber refractive index (~1.5).

        **The full Iron Law with Light Barrier:**

        $$T = \\underbrace{\\frac{D}{BW}}_{\\text{Memory}} +
              \\underbrace{\\frac{O}{R}}_{\\text{Compute}} +
              \\underbrace{L_{\\text{dispatch}} + L_{\\text{propagation}}}_{\\text{Overhead}}$$

        | Symbol | Value | Reducible? |
        |--------|-------|------------|
        | L_dispatch | ~0.5 ms | Yes — software optimization |
        | L_propagation (coast-to-coast) | ~30 ms | **No — physics floor** |
        | L_propagation (local edge, <10 km) | <0.1 ms | Only by moving compute |

        *Source: @eq-latency-physics in @sec-ml-systems-deployment-spectrum-71be*

        **The engineering implication:** When L_propagation > SLA budget, no amount
        of compute optimization can fix the system. The deployment paradigm must change.
        This is the quantitative basis for the Cloud vs Edge architectural split.
        """),
    })
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS ───────────────────────────────────────────────────────
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
                    <strong>1. The Memory Wall dominates transformer inference at low Arithmetic Intensity.</strong>
                    At AI&nbsp;=&nbsp;5 FLOPs/Byte (far below the H100 ridge point of &asymp;590 FLOPs/Byte),
                    the Memory term D/BW accounts for &gt;99% of inference latency. A 6&times; compute
                    upgrade produces only &asymp;8% latency improvement &mdash; because it attacked the wrong term.
                    The correct lever is memory bandwidth (BW), not compute throughput (R).
                    At ~$3/GPU-hour, the $2M H100 upgrade attacks &lt;1% of latency.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. The Ridge Point determines which hardware upgrade is worth buying.</strong>
                    The H100 Ridge Point is R/BW&nbsp;&asymp;&nbsp;590 FLOPs/Byte.
                    Only workloads with AI above this threshold benefit from more TFLOPS.
                    Below it, the Memory Wall is the binding constraint regardless of how many TFLOPS
                    the accelerator advertises.
                </div>
                <div>
                    <strong>3. The speed of light cannot be optimized.</strong>
                    Propagation delay at 3,000 km is 30 ms &mdash; 3&times; the 10 ms AV safety SLA,
                    before any computation. This physical floor is the quantitative reason Edge ML
                    exists as a deployment paradigm. When L_propagation &gt; SLA budget,
                    the deployment architecture must change; no GPU upgrade can fix it.
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
                    <strong>Lab 03: The ML Workflow.</strong> The Iron Law governs performance
                    at a point in time. Lab 03 asks what happens as time passes and the world
                    drifts away from training data &mdash; introducing silent degradation,
                    the MLOps feedback loop, and the Degradation Equation.
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
                    <strong>Read:</strong> @sec-ml-systems-analyzing-workloads-cbb8 for the full
                    Iron Law derivation and @sec-ml-systems-bottleneck-principle-3514 for the
                    Bottleneck Principle and Ridge Point.<br/>
                    <strong>Build:</strong> TinyTorch Module 02 &mdash; implement the Iron Law
                    profiler and observe the three terms in a real forward pass.
                    See <code>tinytorch/src/02_systems/</code>.
                </div>
            </div>

        </div>
        """),


        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. When a workload is memory-bound at batch=1, upgrading from A100 to H100 yields ~1.6x speedup instead of ~3.2x. Why does only the bandwidth ratio matter, not the FLOPS ratio?

    2. What specific batch size and precision combination moved your ResNet-50 deployment across the ridge point from memory-bound to compute-bound on the H100?

    3. The Bottleneck Principle states that optimizing the non-dominant term yields exactly 0% speedup. If your deployment is memory-bound, name two optimization actions that target the correct term and two that waste effort entirely.

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    act1_prediction, act1_reflection, act2_prediction, act2_reflection,
    context_toggle,
    _bottleneck_new, _pct_improvement, _sla_violated,
):
    mo.stop(
        act1_prediction.value is None
        or act1_reflection.value is None
        or act2_prediction.value is None
        or act2_reflection.value is None,
        mo.callout(
            mo.md("Complete both acts (predictions + reflections) to save your results to the Design Ledger."),
            kind="info",
        )
    )

    # ── Save to Design Ledger ─────────────────────────────────────────────────
    _ctx = context_toggle.value
    _act1_correct = act1_prediction.value in ("ten_pct", "no_effect")
    _act2_correct = act2_prediction.value == "30ms"

    ledger.save(
        chapter=2,
        design={
            "context":          _ctx,
            "act1_prediction":  act1_prediction.value,
            "act1_correct":     _act1_correct,
            "bottleneck_type":  _bottleneck_new,
            "act2_result":      float(f"{_pct_improvement:.1f}"),
            "act2_decision":    "edge_deploy" if _sla_violated else "cloud_ok",
            "constraint_hit":   _sla_violated,
        }
    )

    # ── HUD Footer ────────────────────────────────────────────────────────────
    _act1_label  = "correct" if _act1_correct  else "incorrect"
    _act2_label  = "correct" if _act2_correct  else "incorrect"
    _sla_label   = "VIOLATED" if _sla_violated else "met"
    _sla_color   = "#f87171" if _sla_violated  else "#4ade80"
    _ctx_color   = COLORS["Cloud"] if _ctx == "cloud" else COLORS["Edge"]
    _ctx_label   = "Cloud (H100)" if _ctx == "cloud" else "Edge (Orin NX)"

    mo.Html(f"""
    <div style="display:flex; gap:28px; align-items:center; padding:14px 24px;
                background:#0f172a; border-radius:12px; margin-top:24px;
                font-family:'SF Mono','Fira Code',monospace; font-size:0.8rem;
                border:1px solid #1e293b; flex-wrap:wrap;">
        <div>
            <span style="color:#94a3b8; font-weight:600; letter-spacing:0.06em;">
                LAB
            </span>
            <span style="color:#e2e8f0; margin-left:8px;">02 · The Iron Law</span>
        </div>
        <div>
            <span style="color:#94a3b8; font-weight:600; letter-spacing:0.06em;">
                CONTEXT
            </span>
            <span style="color:{_ctx_color}; margin-left:8px; font-weight:700;">
                {_ctx_label}
            </span>
        </div>
        <div>
            <span style="color:#94a3b8; font-weight:600; letter-spacing:0.06em;">
                ACT I
            </span>
            <span style="color:{'#4ade80' if _act1_correct else '#f87171'}; margin-left:8px;">
                {_act1_label} · bottleneck = {_bottleneck_new}
            </span>
        </div>
        <div>
            <span style="color:#94a3b8; font-weight:600; letter-spacing:0.06em;">
                ACT II
            </span>
            <span style="color:{'#4ade80' if _act2_correct else '#f87171'}; margin-left:8px;">
                {_act2_label} · SLA {_sla_label}
            </span>
        </div>
        <div>
            <span style="color:#94a3b8; font-weight:600; letter-spacing:0.06em;">
                LEDGER
            </span>
            <span style="color:#4ade80; margin-left:8px;">ch02 saved</span>
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
