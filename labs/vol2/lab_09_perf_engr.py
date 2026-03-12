import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 09: YOU CAN'T OPTIMIZE WHAT YOU CAN'T MEASURE
#
# Chapter: performance_engineering.qmd (@sec-performance-engineering)
# Core Invariant: Profile-guided optimization finds the true bottleneck — which
#   is rarely where engineers expect it. Amdahl's Law at distributed scale has
#   a second level: the serial fraction includes not just in-process serial code
#   but also distributed coordination (barrier synchronization, AllReduce,
#   checkpoint I/O, pipeline bubbles).
#
# 2-Act structure (35–40 min total):
#   Act I:  The Profiling Revelation (12–15 min)
#     Performance engineer spent 3 weeks on a CUDA attention kernel (3× speedup)
#     but end-to-end training improved by only 8%. Why?
#     Prediction lock → Amdahl explorer → reveal → reflection
#   Act II: Distributed Performance Analysis (20–25 min)
#     512-GPU LLM training at 35% MFU. Budget: fix exactly one bottleneck.
#     Which bottleneck has the highest ROI?
#     Prediction lock → distributed optimizer → failure state (budget exceeded)
#     → reflection
#
# Deployment contexts:
#   Batch training:    Large-scale LLM training, 512 × H100 cluster
#   Streaming inference: Real-time serving, single H100 node
#
# Key hardware constants (NVIDIA specs):
#   H100_BW_GBS         = 3350  # H100 SXM5 HBM3e bandwidth, NVIDIA spec
#   H100_TFLOPS_FP16    = 989   # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
#   H100_RAM_GB         = 80    # H100 HBM3e capacity, NVIDIA spec
#   NVLINK4_BW_GBS      = 900   # NVLink 4.0 bidirectional bandwidth, NVIDIA spec
#   IB_HDR200_BW_GBS    = 400   # InfiniBand HDR200 unidirectional, Mellanox spec
#   NSYS_OVERHEAD_PCT   = 1     # Nsight Systems profiling overhead (~1%)
#
# Design Ledger save: chapter="v2_09"
# ─────────────────────────────────────────────────────────────────────────────


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

    ledger = DesignLedger()

    # ── Hardware constants (all values sourced from NVIDIA datasheets) ─────────
    H100_BW_GBS      = 3350   # GB/s  — H100 SXM5 HBM3e, NVIDIA spec
    H100_TFLOPS_FP16 = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_RAM_GB      = 80     # GB    — H100 HBM3e capacity, NVIDIA spec
    NVLINK4_BW_GBS   = 900    # GB/s  — NVLink 4.0 bidirectional, NVIDIA spec
    IB_HDR200_BW_GBS = 400    # GB/s  — InfiniBand HDR200 unidirectional, Mellanox spec
    NSYS_OVERHEAD_PCT = 1     # %     — Nsight Systems profiling overhead, empirical

    return (
        COLORS, LAB_CSS, apply_plotly_theme, go, ledger, math, mo, np,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB,
        NVLINK4_BW_GBS, IB_HDR200_BW_GBS, NSYS_OVERHEAD_PCT,
    )


# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, LAB_CSS, mo):
    _c_batch  = COLORS["Cloud"]    # indigo — batch training context
    _c_stream = COLORS["Edge"]     # red    — streaming inference context
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume II · Lab 09
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.2rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                You Can't Optimize What You Can't Measure
            </h1>
            <p style="margin: 0 0 22px 0; font-size: 1.02rem; color: #94a3b8;
                      max-width: 700px; line-height: 1.65;">
                A 3× faster attention kernel produced only 8% end-to-end speedup.
                A 512-GPU cluster runs at 35% MFU when industry achieves 55%.
                Both failures share one cause: engineers optimized the code they
                understood, not the bottleneck the profiler would have shown them.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; align-items: center;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: The Profiling Revelation · 12–15 min
                </span>
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act II: Distributed Performance Analysis · 20–25 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min total
                </span>
                <span class="badge badge-info">Chapter 9: Performance Engineering</span>
                <span class="badge badge-warn">Amdahl's Law at Scale</span>
            </div>
            <div style="display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap;">
                <div style="background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.35);
                            border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                    <span style="color: {_c_batch}; font-weight: 700;">Batch Training</span>
                    <span style="color: #94a3b8;"> — 512 × H100 cluster · LLM pre-training · AllReduce-heavy</span>
                </div>
                <div style="background: rgba(203,32,45,0.10); border: 1px solid rgba(203,32,45,0.30);
                            border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                    <span style="color: {_c_stream}; font-weight: 700;">Streaming Inference</span>
                    <span style="color: #94a3b8;"> — Single H100 node · real-time LLM serving · latency-bound</span>
                </div>
            </div>
            <div style="display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap;">
                <span class="badge badge-ok">Constraint: 8-week engineering budget</span>
                <span class="badge badge-ok">Iron Law: Time = max(Compute/FLOPS, Mem/BW) + Overhead</span>
                <span class="badge badge-warn">New: Profile-guided Amdahl Explorer</span>
            </div>
        </div>
        """),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 2: BRIEFING ───────────────────────────────────────────────────────────
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
                <div style="margin-bottom: 3px;">1. <strong>Predict the end-to-end speedup from Amdahl's Law</strong> given a component's time fraction and speedup factor &mdash; a 3x attention-kernel improvement on a 12% bottleneck yields only 8% end-to-end gain.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose the dominant bottleneck in a 512-GPU training profile</strong> (AllReduce, pipeline bubble, or data loading) and calculate which fix delivers the highest ROI for a fixed engineering budget.</div>
                <div style="margin-bottom: 3px;">3. <strong>Identify why optimizing the non-binding constraint yields zero speedup</strong> using the Iron Law: Time = max(Compute/FLOPS, Memory/BW) + Overhead.</div>
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
                    Iron Law of ML Performance from @sec-performance-engineering-efficiency-frontier &middot;
                    Roofline Model and Arithmetic Intensity from @sec-performance-engineering-roofline
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
                "Your team spent 3 weeks building a 3x faster attention kernel and got
                only 8% end-to-end speedup &mdash; why? And which component of your
                512-GPU training profile actually controls the cluster's efficiency?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following sections of the Performance
    Engineering chapter before this lab:

    - **@sec-performance-engineering-efficiency-frontier** — The Iron Law of ML Performance:
      `Time = max(Compute/FLOPS, MemAccess/BW) + Overhead`. Every optimization targets
      exactly one term.
    - **@sec-performance-engineering-roofline** — The Roofline Model and Arithmetic Intensity.
      Diagnose compute-bound vs. memory-bound before optimizing.
    - **@sec-performance-engineering-memory-wall** — Why memory bandwidth, not FLOPS, is the
      binding constraint for most LLM workloads.

    The Amdahl's Law formula used throughout this lab:
    `Speedup = 1 / ((1 - f) + f/k)` where `f` is the fraction of time affected
    by the optimization and `k` is the speedup factor applied to that fraction.
    """), kind="info")
    return


# ─── CELL 4: CONTEXT TOGGLE ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Batch Training (512-GPU LLM cluster)":      "batch",
            "Streaming Inference (single H100 node)":    "streaming",
        },
        value="Batch Training (512-GPU LLM cluster)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("## Select Your Deployment Context"),
        context_toggle,
        mo.md("""
        Both acts use the same profile-guided optimization framework, but the bottleneck
        distribution differs: batch training is dominated by AllReduce communication and
        pipeline bubbles; streaming inference is dominated by memory bandwidth and
        KV-cache management.
        """),
    ])
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I -- CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _act_num      = "I"
    _act_color    = COLORS["BlueLine"]
    _act_title    = "The Profiling Revelation"
    _act_duration = "12-15 min"
    _act_why      = ("You expect that a 3x faster attention kernel will produce approximately "
                     "3x end-to-end training speedup. Amdahl's Law will reveal that attention "
                     "accounts for only 12% of total training time &mdash; making the maximum "
                     "possible speedup 1.14x regardless of how fast the kernel runs.")

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


# ─── CELL 6: ACT1_STAKEHOLDER ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _c = COLORS["BlueLine"]
    mo.vstack([
        mo.Html(f"""
        <div style="border-left:4px solid {_c}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_c};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message · Senior Performance Engineer, LLM Training Team
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "We spent three weeks writing a custom CUDA kernel for multi-head attention.
                Benchmarked in isolation, the kernel is 3× faster than our baseline.
                We shipped it last Thursday. End-to-end training time improved by 8%.
                Eight percent. My team is demoralized. What did we miss?"
            </div>
        </div>
        """),
        mo.md("""
        The engineer is not wrong about the kernel benchmark. The kernel *is* 3× faster.
        The gap between 3× and 8% is not a bug — it is Amdahl's Law in action, and it
        points directly to a failure in the profiling process that preceded the optimization.

        Before running the simulator, commit to your interpretation.
        """),
    ])
    return


# ─── CELL 5: ACT I PREDICTION ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A: 8% is expected — kernel optimizations always have limited impact on end-to-end training.":
                "A",
            "B: The profiler would show attention was only 12% of total training time — they optimized the wrong bottleneck.":
                "B",
            "C: Their custom kernel has a correctness bug — otherwise the improvement would be higher.":
                "C",
            "D: A 3× kernel speedup should produce roughly 3× end-to-end speedup — something else is broken.":
                "D",
        },
        label="Your prediction: What explains the gap between 3× kernel speedup and 8% end-to-end improvement?",
    )
    mo.vstack([
        mo.md("### Your Prediction"),
        mo.md("*Commit before touching the simulator. Your prediction is locked once you proceed.*"),
        act1_pred,
    ])
    return (act1_pred,)


# ─── CELL 6: PREDICTION GATE ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(act1_pred, mo):
    mo.stop(
        act1_pred.value is None,
        mo.callout(mo.md("Select your prediction above to unlock the Act I simulator."), kind="warn"),
    )
    mo.callout(
        mo.md(f"**Prediction locked:** Option **{act1_pred.value}**. Now explore the simulator to test your hypothesis."),
        kind="info",
    )
    return


# ─── CELL 7: ACT I SIMULATOR CONTROLS ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Profile-Guided Amdahl Explorer

    The training profile below shows the time breakdown for one training step.
    These numbers come from Nsight Systems traces on a 512-GPU H100 cluster
    training a 70B transformer. Adjust the **optimization target** and
    **speedup factor** to see what Amdahl's Law predicts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Training step time fractions (must sum to 1.0)
    # Source: empirical profiling of 512-GPU H100 LLM training runs
    # AllReduce: 45% — NCCL ring-AllReduce over InfiniBand HDR200
    # Optimizer:  18% — AdamW update, GPU-local
    # Attention:  12% — multi-head attention forward + backward
    # DataLoad:   15% — async data prefetch (bottleneck varies by storage)
    # KernelLaunch: 10% — CUDA kernel launch overhead + synchronization

    opt_target = mo.ui.dropdown(
        options={
            "Attention Kernel":       "attention",
            "AllReduce Communication": "allreduce",
            "Data Loading":           "dataloading",
            "Optimizer Step":         "optimizer",
            "Kernel Launch Overhead": "kernellaunch",
        },
        value="Attention Kernel",
        label="Optimization target",
    )
    speedup_factor = mo.ui.slider(
        start=1.0, stop=10.0, value=3.0, step=0.5,
        label="Speedup factor applied to target (×)",
    )
    mo.hstack([opt_target, speedup_factor], gap="3rem", justify="start")
    return (opt_target, speedup_factor)


# ─── CELL 8: ACT I SIMULATION ENGINE ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, go, mo, opt_target, speedup_factor):
    # ── Training step profile (source: NVIDIA Nsight Systems traces on H100 cluster)
    # Fractions represent share of total wall-clock time per training step.
    _PROFILE = {
        "attention":    0.12,   # 12% — MHA forward+backward; target of 3-week effort
        "allreduce":    0.45,   # 45% — NCCL AllReduce over InfiniBand HDR200
        "optimizer":    0.18,   # 18% — AdamW parameter update; GPU-local
        "dataloading":  0.15,   # 15% — async prefetch from distributed storage
        "kernellaunch": 0.10,   # 10% — CUDA launch overhead, barrier sync
    }

    _LABELS = {
        "attention":    "Attention Kernel",
        "allreduce":    "AllReduce Comm.",
        "optimizer":    "Optimizer Step",
        "dataloading":  "Data Loading",
        "kernellaunch": "Kernel Launch",
    }

    _COLORS_PROFILE = {
        "attention":    "#6366f1",
        "allreduce":    COLORS["OrangeLine"],
        "optimizer":    COLORS["BlueLine"],
        "dataloading":  COLORS["GreenLine"],
        "kernellaunch": "#8b5cf6",
    }

    _target = opt_target.value
    _k      = speedup_factor.value

    # Amdahl's Law: Speedup = 1 / ((1 - f) + f/k)
    # where f = fraction of time affected, k = speedup factor
    _f = _PROFILE[_target]
    _amdahl_speedup = 1.0 / ((1.0 - _f) + _f / _k)
    _pct_improvement = (_amdahl_speedup - 1.0) * 100.0

    # New profile after optimization
    _new_profile = {}
    for _key, _frac in _PROFILE.items():
        if _key == _target:
            _new_profile[_key] = _frac / _k
        else:
            _new_profile[_key] = _frac
    _total_new = sum(_new_profile.values())
    # Renormalize to show fractions of new total step time
    _new_frac_norm = {k: v / _total_new for k, v in _new_profile.items()}
    _old_frac_norm = {k: v for k, v in _PROFILE.items()}

    # Find the true hotspot (largest fraction in original profile)
    _hotspot_key = max(_PROFILE, key=_PROFILE.get)
    _hotspot_label = _LABELS[_hotspot_key]
    _hotspot_frac  = _PROFILE[_hotspot_key]

    # Hypothetical: what speedup if we had optimized the hotspot instead?
    _hotspot_speedup = 1.0 / ((1.0 - _hotspot_frac) + _hotspot_frac / _k)
    _hotspot_pct = (_hotspot_speedup - 1.0) * 100.0

    # ── Flame chart: before vs. after (stacked horizontal bar) ────────────────
    _components = list(_PROFILE.keys())
    _labels     = [_LABELS[c] for c in _components]
    _before_pct = [_old_frac_norm[c] * 100 for c in _components]
    _after_pct  = [_new_frac_norm[c] * 100 for c in _components]
    _bar_colors = [_COLORS_PROFILE[c] for c in _components]

    _fig = go.Figure()
    for _i, (_comp, _label, _b_pct, _a_pct, _clr) in enumerate(
        zip(_components, _labels, _before_pct, _after_pct, _bar_colors)
    ):
        _fig.add_trace(go.Bar(
            name=_label + " (before)",
            x=[_b_pct], y=["Before"],
            orientation="h",
            marker_color=_clr, marker_opacity=0.85,
            hovertemplate=f"{_label}: {{x:.1f}}%<extra></extra>",
            legendgroup=_label,
            showlegend=True,
        ))
        _fig.add_trace(go.Bar(
            name=_label + " (after)",
            x=[_a_pct], y=["After"],
            orientation="h",
            marker_color=_clr, marker_opacity=0.45,
            marker_pattern_shape="/" if _comp == _target else "",
            hovertemplate=f"{_label}: {{x:.1f}}%<extra></extra>",
            legendgroup=_label,
            showlegend=False,
        ))

    _fig.update_layout(
        barmode="stack",
        height=220,
        legend=dict(orientation="h", y=-0.45, x=0, font_size=11),
        xaxis=dict(title="Share of training step time (%)", range=[0, 100]),
        yaxis=dict(title=""),
        margin=dict(l=70, r=20, t=20, b=120),
        title=dict(
            text=f"Training Step Profile — optimizing <b>{_LABELS[_target]}</b> by <b>{_k:.1f}×</b>",
            font_size=13, x=0,
        ),
    )
    apply_plotly_theme(_fig)

    # ── Color coding for result metrics ───────────────────────────────────────
    _color_speedup = (
        COLORS["GreenLine"]  if _pct_improvement >= 20 else
        COLORS["OrangeLine"] if _pct_improvement >= 10 else
        COLORS["RedLine"]
    )
    _color_hotspot = (
        COLORS["GreenLine"]  if _hotspot_pct >= 20 else
        COLORS["OrangeLine"] if _hotspot_pct >= 10 else
        COLORS["RedLine"]
    )
    _optimal_badge = (
        f'<span class="badge badge-ok">Optimal target selected</span>'
        if _target == _hotspot_key
        else f'<span class="badge badge-fail">Suboptimal — true hotspot is {_hotspot_label} ({_hotspot_frac*100:.0f}%)</span>'
    )

    mo.vstack([
        mo.as_html(_fig),
        mo.Html(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-top: 16px;">
            <div style="padding: 18px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                        background: white; text-align: center;">
                <div style="color: #64748b; font-size: 0.82rem; font-weight: 600; margin-bottom: 4px;">
                    Target fraction
                </div>
                <div style="font-size: 2rem; font-weight: 900; color: {COLORS['BlueLine']};">
                    {_f*100:.0f}%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                    {_LABELS[_target]}
                </div>
            </div>
            <div style="padding: 18px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                        background: white; text-align: center;">
                <div style="color: #64748b; font-size: 0.82rem; font-weight: 600; margin-bottom: 4px;">
                    End-to-end speedup
                </div>
                <div style="font-size: 2rem; font-weight: 900; color: {_color_speedup};">
                    +{_pct_improvement:.1f}%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                    Amdahl: {_amdahl_speedup:.3f}× total
                </div>
            </div>
            <div style="padding: 18px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                        background: white; text-align: center;">
                <div style="color: #64748b; font-size: 0.82rem; font-weight: 600; margin-bottom: 4px;">
                    Hotspot alternative ({_hotspot_label})
                </div>
                <div style="font-size: 2rem; font-weight: 900; color: {_color_hotspot};">
                    +{_hotspot_pct:.1f}%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                    Same {_k:.1f}× factor, optimal target
                </div>
            </div>
        </div>
        <div style="margin-top: 12px;">
            {_optimal_badge}
        </div>
        """),
        mo.md(f"""
        **Amdahl's Law formula:**

        ```
        Speedup = 1 / ((1 - f) + f/k)
                = 1 / ((1 - {_f:.2f}) + {_f:.2f}/{_k:.1f})
                = 1 / ({1-_f:.2f} + {_f/_k:.4f})
                = 1 / {(1-_f) + _f/_k:.4f}
                = {_amdahl_speedup:.4f}×
                → +{_pct_improvement:.1f}% end-to-end improvement
        ```

        The attention kernel optimization (f = {_f:.2f}, k = {_k:.1f}×) gives the
        result we see in the scenario: **+{_pct_improvement if abs(_f - 0.12) < 0.01 and abs(_k - 3.0) < 0.1 else _pct_improvement:.1f}% end-to-end improvement**.
        The bottleneck was AllReduce ({_PROFILE['allreduce']*100:.0f}% of step time).
        Applying the same {_k:.1f}× speedup to AllReduce would yield
        **+{_hotspot_pct if _hotspot_key == 'allreduce' else (_pct_improvement if _target == 'allreduce' else _hotspot_pct):.1f}% end-to-end improvement** — roughly
        {_hotspot_pct / max(_pct_improvement, 0.001):.1f}× more impact.
        """),
    ])
    return (
        _amdahl_speedup, _f, _hotspot_key, _hotspot_label, _hotspot_pct,
        _pct_improvement, _target, _LABELS, _PROFILE,
    )


# ─── CELL 9: ACT I PREDICTION-VS-REALITY OVERLAY ──────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, _LABELS, _PROFILE, _pct_improvement, act1_pred, mo):
    # The canonical numbers from the scenario with attention at 12%, k=3×
    _actual_pct = 8.7   # Amdahl: 1/(0.88 + 0.12/3) = 1/0.92 = 1.087 → +8.7%
    _allreduce_opt_pct = 29.0  # Amdahl: 1/(0.55 + 0.45/2) = 1/0.775 = 1.29 → +29%

    _pred_map = {
        "A": ("8% is within normal variance for kernel optimizations", False),
        "B": ("The profiler shows attention was only 12% of total time — they optimized the wrong bottleneck", True),
        "C": ("The kernel has a correctness bug causing incorrect but fast results", False),
        "D": ("A 3× kernel speedup should yield roughly 3× end-to-end speedup", False),
    }

    _selected = act1_pred.value or "B"
    _pred_text, _is_correct = _pred_map.get(_selected, ("", False))

    if _is_correct:
        _overlay = mo.callout(mo.md(f"""
        **Correct. Amdahl's Law is exact here.**

        Attention was **12%** of total step time. A 3× kernel speedup on 12% of the
        workload gives:

        `Speedup = 1 / (0.88 + 0.12/3) = 1 / 0.92 = 1.087 → +8.7% end-to-end`

        The actual measurement (+8%) matches the Amdahl prediction (+8.7%) within
        profiling noise. The optimization is not broken — the target was wrong.

        The true hotspot was **AllReduce communication ({_PROFILE['allreduce']*100:.0f}% of step time)**.
        A 2× AllReduce speedup (achievable with NCCL topology optimization or
        gradient compression) would give:

        `Speedup = 1 / (0.55 + 0.45/2) = 1 / 0.775 = 1.29 → +29% end-to-end`

        That is **3.3× more impact** from the same engineering effort, directed at
        the bottleneck the profiler would have identified in the first hour.
        """), kind="success")
    elif _selected == "D":
        _overlay = mo.callout(mo.md(f"""
        **Not quite. This is the Amdahl fallacy.**

        The 3× claim is correct for the kernel in isolation. But end-to-end speedup
        is bounded by the fraction of time the optimization affects:

        `Speedup = 1 / ((1 - f) + f/k)`

        When f = 0.12 (attention = 12% of step time) and k = 3.0:

        `Speedup = 1 / (0.88 + 0.04) = 1.087 → +8.7%`

        The 3× speedup on 12% of the workload can never produce more than
        `1 / (1 - 0.12) = 1.136` — a theoretical maximum of **+13.6%** even if the
        attention kernel took zero time. The ceiling is set by the other 88%.
        """), kind="warn")
    elif _selected == "C":
        _overlay = mo.callout(mo.md(f"""
        **Not quite. The kernel is working correctly.**

        The 8% improvement is not evidence of a bug — it is the exact prediction of
        Amdahl's Law. A 3× speedup on 12% of total step time gives:

        `Speedup = 1 / (0.88 + 0.12/3) = 1 / 0.92 = 1.087 → +8.7%`

        The measurement matches the prediction. The issue is not code correctness;
        it is optimization *targeting*: the team optimized 12% of total time when
        AllReduce represented 45% and was the true bottleneck.
        """), kind="warn")
    else:  # A
        _overlay = mo.callout(mo.md(f"""
        **Partially correct framing, but the wrong explanation.**

        8% improvement from a 3× kernel speedup is not typical variance — it is
        a precise Amdahl prediction. The issue is not that "kernel optimizations have
        limited impact." The issue is that *this particular kernel* was optimized on
        only 12% of total training time:

        `Speedup = 1 / (0.88 + 0.12/3) = 1.087 → +8.7%`

        Had the team profiled first and targeted AllReduce instead (45% of time),
        the same engineering effort would have delivered +29% end-to-end improvement
        — roughly 3.3× more impact. The lesson is not "kernel optimizations are
        limited." The lesson is "profile first, always."
        """), kind="warn")

    mo.vstack([
        mo.md("### Prediction vs. Reality"),
        _overlay,
    ])
    return


# ─── CELL 10: ACT I MATHPEEK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — Amdahl's Law and its distributed extension": mo.md("""
        **Amdahl's Law (single-stage):**

        ```
        Speedup = 1 / ((1 - f) + f/k)
        ```

        - **f** — fraction of total execution time affected by the optimization (0 ≤ f ≤ 1)
        - **k** — speedup factor applied to the affected fraction (k ≥ 1)
        - The term `(1 - f)` is the *serial residual* — the work that cannot be sped up

        **The maximum speedup ceiling** (as k → ∞, making the target fraction zero):

        ```
        Speedup_max = 1 / (1 - f)
        ```

        For f = 0.12 (attention): Speedup_max = 1/0.88 = 1.136 → **+13.6% ceiling**.
        No matter how fast the attention kernel, end-to-end improvement is capped at 13.6%.

        **Multi-stage Amdahl (distributed training):**

        In distributed training, the "serial residual" includes not just in-process serial
        code but all distributed coordination:

        ```
        Speedup = 1 / (f_compute/k_compute + f_allreduce/k_allreduce
                        + f_pipeline_bubble/k_bubble + f_dataload/k_dataload
                        + f_kernellaunch/k_launch)
        ```

        where each fraction `f_i` must satisfy `sum(f_i) = 1`.

        **Gustafson's Law** (for workload-scaled problems):

        When problem size scales with the number of processors (as in data-parallel training
        with larger batch sizes), the relevant law is Gustafson's, not Amdahl's:

        ```
        Speedup_Gustafson = k - f_serial × (k - 1)
        ```

        Gustafson's Law is more optimistic: it assumes the serial fraction stays *constant*
        as the workload grows, rather than staying constant as a *fraction* of the total.
        For batch training where you can always increase batch size, Gustafson applies;
        for fixed-dataset training, Amdahl applies.

        **Roofline-guided optimization priority:**

        Profile the workload with Nsight Systems. Identify the component with the largest
        time fraction. Compute the Arithmetic Intensity (AI = FLOPs / bytes) for that
        component. If AI < ridge point (≈295 FLOP/byte for H100 FP16), the component is
        *memory-bound* — optimize memory access patterns, not compute. If AI > ridge point,
        it is *compute-bound* — optimize kernel arithmetic efficiency.
        """),
    })
    return


# ─── CELL 11: ACT I REFLECTION ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A: Profile the codebase to identify the largest time fraction (hotspot) — optimize the bottleneck, not the code you understand best.":
                "A",
            "B: Rewrite hot loops in C++/CUDA for maximum low-level control.":
                "B",
            "C: Increase batch size to amortize per-step overhead across more samples.":
                "C",
            "D: Reduce model size to decrease computation per step.":
                "D",
        },
        label="Reflection: What is the mandatory first step in any optimization project?",
    )
    mo.vstack([
        mo.md("### Act I Reflection"),
        act1_reflection,
    ])
    return (act1_reflection,)


@app.cell(hide_code=True)
def _(act1_reflection, mo):
    mo.stop(
        act1_reflection.value is None,
        mo.callout(mo.md("Select your reflection answer to continue to Act II."), kind="warn"),
    )
    _selected = act1_reflection.value or "A"
    if _selected == "A":
        mo.callout(mo.md("""
        **Correct. Profile first, always.**

        The invariant: *you cannot optimize what you cannot measure*. A profiler
        (Nsight Systems, PyTorch Profiler, or nsys) gives you the ground truth
        time breakdown in minutes. Skipping this step means you are optimizing
        based on intuition — and intuition is reliably wrong about distributed
        system bottlenecks.

        The consequence of profiling first: you discover that AllReduce is 45% of
        your training step, that your attention kernel is 12%, and that three weeks
        of kernel engineering will produce less than a quarter of the gain that
        NCCL topology tuning would provide in two days.
        """), kind="success")
    elif _selected == "B":
        mo.callout(mo.md("""
        **Wrong direction.** Rewriting in C++/CUDA is a valid optimization
        *technique*, but it is not a strategy. Applied to the wrong component, a
        perfectly optimized C++ kernel still delivers the same Amdahl-bounded
        fraction of improvement. Profile first. Then decide whether C++ or CUDA is
        the right tool for the *actual bottleneck*.
        """), kind="warn")
    elif _selected == "C":
        mo.callout(mo.md("""
        **Wrong direction.** Increasing batch size changes the workload, not the
        profiling methodology. It can improve GPU utilization (by giving the hardware
        more parallelism to exploit), but it does not tell you *where* time is being
        spent in the current training configuration. Profile first; then batch size
        tuning may appear as one option among several.
        """), kind="warn")
    else:
        mo.callout(mo.md("""
        **Wrong direction.** Reducing model size changes what you are building, not
        how efficiently you build it. A smaller model that trains faster may not meet
        quality requirements. Profile the model you have first; then determine whether
        the bottleneck is structural (requiring architectural changes) or engineering
        (addressable through optimization without changing the model).
        """), kind="warn")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II -- DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _act_num      = "II"
    _act_color    = COLORS["OrangeLine"]
    _act_title    = "Distributed Performance Analysis"
    _act_duration = "20-25 min"
    _act_why      = ("Act I showed that optimizing the wrong bottleneck wastes engineering weeks. "
                     "Now apply the same logic at 512-GPU scale: a real training profile shows "
                     "compute = 45%, AllReduce = 30%, pipeline bubble = 15%, data loading = 10%. "
                     "One quarter's engineering budget. Amdahl determines the only correct choice.")

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


# ─── CELL 13: ACT2_STAKEHOLDER ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _c = COLORS["OrangeLine"]
    mo.vstack([
        mo.Html(f"""
        <div style="border-left:4px solid {_c}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_c};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message · Production ML Lead, Foundation Model Team
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "Our 512-GPU H100 cluster is training a 70B LLM at 35% MFU.
                We know industry leaders are hitting 50–60% MFU at this scale.
                I have a Nsight Systems trace showing: compute = 45%, AllReduce = 30%,
                pipeline bubble = 15%, data loading = 10%.
                We have budget to fix exactly one bottleneck this quarter.
                My recommendation to leadership is due Friday. What do I optimize?"
            </div>
        </div>
        """),
        mo.md("""
        **Model Flop Utilization (MFU)** measures the fraction of peak hardware FLOPS
        actually used for productive computation. At 35% MFU on H100s (989 TFLOPS FP16),
        the cluster delivers 35% × 989 = 346 effective TFLOPS per GPU, leaving 65%
        of purchased capability idle.

        The four candidate bottlenecks have different Amdahl impacts *and* different
        implementation costs. Before running the optimizer, commit to your recommendation.
        """),
    ])
    return


# ─── CELL 13: ACT II PREDICTION ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A: Fix AllReduce (30%) — largest parallelizable fraction means biggest Amdahl gain.":
                "A",
            "B: Fix pipeline bubble (15%) — configuration change only (increase micro-batches), zero hardware cost, gets closest to 50% MFU target.":
                "B",
            "C: Fix data loading (10%) — smallest and easiest to fix with prefetch workers.":
                "C",
            "D: Fix compute (45%) — largest fraction must be the primary bottleneck.":
                "D",
        },
        label="Your recommendation: Which single bottleneck has the highest ROI this quarter?",
    )
    mo.vstack([
        mo.md("### Your Recommendation"),
        mo.md("*Commit before running the optimizer.*"),
        act2_pred,
    ])
    return (act2_pred,)


@app.cell(hide_code=True)
def _(act2_pred, mo):
    mo.stop(
        act2_pred.value is None,
        mo.callout(mo.md("Select your recommendation above to unlock the Act II optimizer."), kind="warn"),
    )
    mo.callout(
        mo.md(f"**Recommendation locked:** Option **{act2_pred.value}**. Now use the optimizer to test your analysis."),
        kind="info",
    )
    return


# ─── CELL 14: ACT II SIMULATOR CONTROLS ───────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Distributed Performance Optimizer

    Use the levers below to apply targeted optimizations to each bottleneck.
    The simulator computes new MFU using multi-stage Amdahl's Law and tracks
    engineering implementation cost. The constraint: **8 weeks of engineering
    budget**. Exceeding the budget triggers the failure state.

    **Pipeline bubble formula:**
    `B = (PP - 1) / (PP × m)` where PP = pipeline stages, m = micro-batches per step.
    Increasing m reduces bubble fraction: doubling m from 4 → 8 halves the bubble
    from ~15% → ~7.5% of step time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    micro_batches = mo.ui.slider(
        start=1, stop=32, value=4, step=1,
        label="Pipeline micro-batches per step (m) — increases to reduce bubble",
    )
    allreduce_compression = mo.ui.slider(
        start=1.0, stop=4.0, value=1.0, step=0.5,
        label="AllReduce compression ratio (×) — gradient compression speedup",
    )
    data_prefetch_workers = mo.ui.slider(
        start=1, stop=16, value=4, step=1,
        label="Data prefetch workers — parallel I/O workers per GPU",
    )
    mo.vstack([
        micro_batches,
        allreduce_compression,
        data_prefetch_workers,
    ])
    return (allreduce_compression, data_prefetch_workers, micro_batches)


# ─── CELL 15: ACT II SIMULATION ENGINE ────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, allreduce_compression, apply_plotly_theme,
    data_prefetch_workers, go, micro_batches, mo,
):
    # ── Baseline training step profile (512-GPU H100, 70B LLM, 3D-parallel)
    # Source: empirical characterization of large-scale LLM training
    _BASE = {
        "compute":         0.45,   # 45% — GPU compute (forward + backward pass)
        "allreduce":       0.30,   # 30% — NCCL AllReduce over InfiniBand HDR200
        "pipeline_bubble": 0.15,   # 15% — pipeline idle time, B = (PP-1)/(PP×m)
        "data_loading":    0.10,   # 10% — async data prefetch bottleneck
    }
    _LABELS2 = {
        "compute":         "Compute (fwd+bwd)",
        "allreduce":       "AllReduce Comm.",
        "pipeline_bubble": "Pipeline Bubble",
        "data_loading":    "Data Loading",
    }
    _COLORS2 = {
        "compute":         COLORS["BlueLine"],
        "allreduce":       COLORS["OrangeLine"],
        "pipeline_bubble": COLORS["RedLine"],
        "data_loading":    COLORS["GreenLine"],
    }

    # ── Pipeline bubble reduction ─────────────────────────────────────────────
    # Formula: B = (PP - 1) / (PP × m) where PP = pipeline stages = 8 (70B, 3D-parallel)
    # Baseline: m = 4 micro-batches → B_base = 7 / 32 ≈ 0.219 of compute time
    # But bubble is expressed as fraction of *total* step time = 15%
    # New bubble fraction of total = 15% × (4 / m)  (halves when m doubles)
    _PP = 8          # pipeline stages (70B model, 8-way pipeline parallelism)
    _m_base = 4      # baseline micro-batches per step
    _m_new  = micro_batches.value

    # Bubble fraction of step time scales as 1/m (keeping PP constant)
    _bubble_new_frac = _BASE["pipeline_bubble"] * (_m_base / _m_new)
    # Excess time saved by reducing bubble returns to compute
    _bubble_saved = _BASE["pipeline_bubble"] - _bubble_new_frac

    # ── AllReduce compression speedup ─────────────────────────────────────────
    # Gradient compression (TopK, PowerSGD, etc.) reduces bytes transmitted.
    # Speedup factor = compression ratio; bandwidth wall means 2× compression ≈ 2× AllReduce speedup
    _ar_k = allreduce_compression.value
    _allreduce_new_frac = _BASE["allreduce"] / _ar_k

    # ── Data loading speedup ─────────────────────────────────────────────────
    # Prefetch workers scale approximately as sqrt (I/O parallelism diminishing returns)
    # Empirical: 4→8 workers ≈ 1.5× throughput; 4→16 workers ≈ 2.2× throughput
    import math as _math
    _dl_speedup = _math.sqrt(_data_prefetch_workers := data_prefetch_workers.value) / _math.sqrt(4)
    _dl_speedup = max(1.0, _dl_speedup)
    _dataload_new_frac = _BASE["data_loading"] / _dl_speedup

    # ── New step time (as fraction of baseline) ───────────────────────────────
    # compute fraction is unchanged (assumed at hardware ceiling already)
    _new_fracs = {
        "compute":         _BASE["compute"],
        "allreduce":       _allreduce_new_frac,
        "pipeline_bubble": _bubble_new_frac,
        "data_loading":    _dataload_new_frac,
    }
    _total_new = sum(_new_fracs.values())
    _speedup_total = 1.0 / _total_new  # normalized: baseline = 1.0

    # MFU calculation
    # Baseline MFU = 35%. New MFU = 35% × speedup (bounded at 60% practical ceiling)
    _MFU_BASE = 35.0
    _MFU_PRACTICAL_CEIL = 60.0   # industry top-of-range for 512-GPU training
    _mfu_new = min(_MFU_BASE * _speedup_total, _MFU_PRACTICAL_CEIL)
    _mfu_improvement = _mfu_new - _MFU_BASE

    # ── Engineering cost model ────────────────────────────────────────────────
    # Each optimization has an implementation cost in engineer-weeks
    # Pipeline micro-batches: only configuration change, 0.5 wks baseline + 0.1/increment
    _cost_pipeline = 0.5 + max(0, (_m_new - _m_base)) * 0.1
    # AllReduce compression: requires gradient compression library integration
    _cost_allreduce = 0.0 if _ar_k == 1.0 else (2.0 + (_ar_k - 1.0) * 1.5)
    # Data prefetch: DALI or custom prefetcher per additional worker tier
    _cost_dataload = 0.0 if _data_prefetch_workers <= 4 else (1.0 + (_data_prefetch_workers - 4) * 0.25)

    _total_cost_weeks = _cost_pipeline + _cost_allreduce + _cost_dataload
    _BUDGET_WEEKS = 8.0  # quarterly engineering budget
    _budget_exceeded = _total_cost_weeks > _BUDGET_WEEKS

    # ── Before/after grouped bar chart ────────────────────────────────────────
    _comps  = list(_BASE.keys())
    _labels = [_LABELS2[c] for c in _comps]
    _before_pcts = [_BASE[c] * 100 for c in _comps]
    _after_pcts  = [_new_fracs[c] * 100 for c in _comps]
    _clrs        = [_COLORS2[c] for c in _comps]

    _fig2 = go.Figure()
    _fig2.add_trace(go.Bar(
        name="Before optimization",
        x=_labels, y=_before_pcts,
        marker_color=_clrs, marker_opacity=0.9,
        text=[f"{v:.1f}%" for v in _before_pcts],
        textposition="outside",
    ))
    _fig2.add_trace(go.Bar(
        name="After optimization",
        x=_labels, y=_after_pcts,
        marker_color=_clrs, marker_opacity=0.45,
        marker_pattern_shape="\\",
        text=[f"{v:.1f}%" for v in _after_pcts],
        textposition="outside",
    ))
    _fig2.update_layout(
        barmode="group",
        height=320,
        yaxis=dict(title="Share of training step time (%)", range=[0, 60]),
        xaxis=dict(title=""),
        legend=dict(orientation="h", y=-0.28, x=0),
        margin=dict(l=60, r=20, t=40, b=100),
        title=dict(text="Training Step Profile: Before vs. After Optimization", font_size=13, x=0),
    )
    apply_plotly_theme(_fig2)

    # ── Color-coded result metrics ────────────────────────────────────────────
    _mfu_color = (
        COLORS["GreenLine"]  if _mfu_new >= 50 else
        COLORS["OrangeLine"] if _mfu_new >= 42 else
        COLORS["RedLine"]
    )
    _cost_color = (
        COLORS["GreenLine"]  if _total_cost_weeks <= 4 else
        COLORS["OrangeLine"] if _total_cost_weeks <= _BUDGET_WEEKS else
        COLORS["RedLine"]
    )

    # ROI: MFU points gained per engineer-week spent
    _roi = _mfu_improvement / max(_total_cost_weeks, 0.5)

    _pipeline_formula = f"B = (PP-1)/(PP×m) = ({_PP}-1)/({_PP}×{_m_new}) = {(_PP-1)/(_PP*_m_new):.3f}"

    mo.vstack([
        mo.as_html(_fig2),
        mo.Html(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 14px; margin-top: 16px;">
            <div style="padding: 16px 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        background: white; text-align: center;">
                <div style="color: #64748b; font-size: 0.80rem; font-weight: 600; margin-bottom: 4px;">
                    New MFU
                </div>
                <div style="font-size: 2rem; font-weight: 900; color: {_mfu_color};">
                    {_mfu_new:.1f}%
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">
                    was 35.0% (baseline)
                </div>
            </div>
            <div style="padding: 16px 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        background: white; text-align: center;">
                <div style="color: #64748b; font-size: 0.80rem; font-weight: 600; margin-bottom: 4px;">
                    MFU improvement
                </div>
                <div style="font-size: 2rem; font-weight: 900; color: {_mfu_color};">
                    +{_mfu_improvement:.1f}pp
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">
                    percentage points
                </div>
            </div>
            <div style="padding: 16px 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        background: white; text-align: center;">
                <div style="color: #64748b; font-size: 0.80rem; font-weight: 600; margin-bottom: 4px;">
                    Engineering cost
                </div>
                <div style="font-size: 2rem; font-weight: 900; color: {_cost_color};">
                    {_total_cost_weeks:.1f} wks
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">
                    budget: {_BUDGET_WEEKS:.0f} weeks
                </div>
            </div>
            <div style="padding: 16px 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        background: white; text-align: center;">
                <div style="color: #64748b; font-size: 0.80rem; font-weight: 600; margin-bottom: 4px;">
                    ROI
                </div>
                <div style="font-size: 2rem; font-weight: 900; color: {COLORS['BlueLine']};">
                    {_roi:.1f}
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">
                    MFU pts / eng-week
                </div>
            </div>
        </div>
        """),
        mo.Html(f"""
        <div style="background: #f8fafc; border-radius: 10px; padding: 14px 18px; margin-top: 12px;
                    border: 1px solid #e2e8f0; font-size: 0.85rem; color: #475569; line-height: 1.7;">
            <strong>Pipeline bubble formula:</strong>
            <code style="background: #e2e8f0; padding: 2px 6px; border-radius: 4px; font-size: 0.82rem;">
                {_pipeline_formula}
            </code>
            — bubble fraction of step time with m={_m_new} micro-batches:
            <strong>{_bubble_new_frac*100:.1f}%</strong>
            (was {_BASE['pipeline_bubble']*100:.0f}% at m={_m_base}).
            <br>
            <strong>Cost breakdown:</strong>
            Pipeline config: {_cost_pipeline:.1f} wk |
            AllReduce compression: {_cost_allreduce:.1f} wk |
            Data prefetch: {_cost_dataload:.1f} wk =
            <strong>{_total_cost_weeks:.1f} wk total</strong>.
        </div>
        """),
    ])
    return (
        _budget_exceeded, _cost_allreduce, _cost_dataload, _cost_pipeline,
        _mfu_improvement, _mfu_new, _roi, _total_cost_weeks,
        _BUDGET_WEEKS, _LABELS2,
    )


# ─── CELL 16: ACT II FAILURE STATE ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, _BUDGET_WEEKS, _budget_exceeded, _cost_allreduce,
    _cost_dataload, _cost_pipeline, _mfu_improvement, _mfu_new,
    _total_cost_weeks, mo,
):
    if _budget_exceeded:
        # Determine which cost is largest (suggest deprioritizing it)
        _costs = {
            "pipeline config": _cost_pipeline,
            "AllReduce compression": _cost_allreduce,
            "data prefetch tuning": _cost_dataload,
        }
        _largest_cost_name = max(_costs, key=_costs.get)
        _largest_cost_val  = _costs[_largest_cost_name]
        mo.callout(mo.md(
            f"**Engineering budget exceeded:** Selected optimizations require "
            f"**{_total_cost_weeks:.1f} weeks**. Budget: **{_BUDGET_WEEKS:.0f} weeks**. "
            f"Deprioritize **{_largest_cost_name}** ({_largest_cost_val:.1f} wks) — "
            f"reduce scope or defer to next quarter. The highest-ROI single fix "
            f"(pipeline micro-batch tuning) costs under 1 week and delivers "
            f"~3–4 MFU percentage points."
        ), kind="warn")
    elif _mfu_new >= 50.0:
        mo.callout(mo.md(
            f"**Target reached.** New MFU: **{_mfu_new:.1f}%** — above the 50% industry "
            f"threshold. Total engineering cost: **{_total_cost_weeks:.1f} weeks** "
            f"(within {_BUDGET_WEEKS:.0f}-week budget). "
            f"MFU improvement: **+{_mfu_improvement:.1f} percentage points**."
        ), kind="success")
    else:
        mo.callout(mo.md(
            f"**Feasible but below target.** New MFU: **{_mfu_new:.1f}%** "
            f"(target: 50%). Cost: {_total_cost_weeks:.1f} weeks. "
            f"Adjust levers to find the configuration that crosses 50% MFU "
            f"within the {_BUDGET_WEEKS:.0f}-week budget."
        ), kind="info")
    return


# ─── CELL 17: ACT II PREDICTION-VS-REALITY ────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, _mfu_improvement, _mfu_new, _roi, _total_cost_weeks, act2_pred, mo):
    _selected2 = act2_pred.value or "B"

    # Reference numbers for the "optimal" pipeline bubble fix:
    # m: 4→8, bubble: 15%→7.5%, new profile: {compute:0.45, ar:0.30, bubble:0.075, dl:0.10}
    # total = 0.925, speedup = 1/0.925 = 1.081, MFU = 35 × 1.081 = 37.8%, cost ≈ 0.5 wk
    _pipeline_mfu   = 37.8   # %
    _pipeline_cost  = 0.5    # engineer-weeks
    _pipeline_roi   = (_pipeline_mfu - 35.0) / _pipeline_cost  # MFU pts / wk

    # AllReduce 2× compression:
    # new profile: {compute:0.45, ar:0.15, bubble:0.15, dl:0.10}, total=0.85
    # speedup = 1.176, MFU = 35 × 1.176 = 41.2%, cost ≈ 3.5 wk
    _ar_mfu   = 41.2
    _ar_cost  = 3.5
    _ar_roi   = (_ar_mfu - 35.0) / _ar_cost   # MFU pts / wk

    if _selected2 == "B":
        mo.callout(mo.md(f"""
        **Correct. Pipeline bubble is the highest-ROI fix.**

        Pipeline bubbles scale with pipeline parallelism degree (PP) and are fixed
        entirely by configuration — no hardware, no library changes, no model
        architecture changes:

        ```
        B = (PP - 1) / (PP × m)
          = (8 - 1) / (8 × 4) = 7/32 ≈ 21.9% of compute time
        ```

        As fraction of total step time, B ≈ 15%. Doubling micro-batches (m: 4 → 8)
        halves the bubble:

        ```
        New B = (8-1)/(8×8) = 7/64 ≈ 10.9% of compute time → 7.5% of step time
        ```

        New MFU ≈ **{_pipeline_mfu:.1f}%** at **{_pipeline_cost:.1f} engineer-week** of cost.
        ROI: **{_pipeline_roi:.1f} MFU pts/week** — compared to AllReduce compression's
        {_ar_roi:.1f} MFU pts/week at {_ar_cost:.1f} weeks of implementation cost.

        AllReduce compression delivers more absolute MFU gain ({_ar_mfu:.1f}% vs {_pipeline_mfu:.1f}%)
        but requires 7× more engineering effort. Pipeline tuning gets to 50% MFU
        faster, meets the deadline, and leaves budget for AllReduce next quarter.
        """), kind="success")
    elif _selected2 == "A":
        mo.callout(mo.md(f"""
        **Reasonable analysis, but ROI tells a different story.**

        AllReduce is 30% of step time and a 2× compression speedup gives:

        ```
        Speedup = 1 / (0.45 + 0.15 + 0.15 + 0.10) = 1 / 0.85 = 1.176 → +17.6% step time
        MFU = 35% × 1.176 = 41.2%
        ```

        That is genuine improvement (+6.2 pp) but costs ~3.5 engineer-weeks to implement
        gradient compression with correctness guarantees. ROI: **{_ar_roi:.1f} MFU pts/week**.

        Pipeline bubble tuning (m: 4→8) delivers +2.8 pp MFU in **0.5 engineer-weeks**
        with zero risk — it is a configuration file change. ROI: **{_pipeline_roi:.1f} MFU pts/week**,
        which is {_pipeline_roi / max(_ar_roi, 0.01):.1f}× higher.

        The ranking is: pipeline first (this quarter), AllReduce second (next quarter).
        """), kind="warn")
    elif _selected2 == "C":
        mo.callout(mo.md(f"""
        **Correct instinct (low implementation cost), but too small to matter.**

        Data loading is 10% of step time. Even a perfect fix (data loading → 0%)
        gives at most:

        ```
        Speedup_max = 1 / (1 - 0.10) = 1.111 → +11.1% step time → MFU ≈ 38.9%
        ```

        That is a real gain (+3.9 pp), but it is the *ceiling* — you cannot make
        data loading faster than zero. In practice, 16 prefetch workers improve
        throughput by roughly 2.2× (sqrt scaling), not infinity. The achievable gain
        is smaller than the pipeline bubble fix while requiring similar implementation
        effort. Optimize the data pipeline after the pipeline bubble and AllReduce.
        """), kind="warn")
    else:  # D
        mo.callout(mo.md(f"""
        **Wrong interpretation.** Compute at 45% is already at the hardware ceiling.

        The 45% compute fraction does not mean compute is the bottleneck — it means
        45% of step time is productively used for GPU arithmetic. The remaining 55% is
        time spent in AllReduce (30%), pipeline bubbles (15%), and data loading (10%).
        These are the *gaps* between compute bursts, not the compute itself.

        Attempting to "fix compute" at 45% utilization is undefined: you cannot make
        hardware faster. You can make the *other 55%* smaller, which is exactly what
        AllReduce compression, pipeline micro-batch tuning, and data prefetch do.
        """), kind="warn")
    return


# ─── CELL 18: ACT II MATHPEEK ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — Distributed Amdahl, pipeline bubble formula, optimization ROI": mo.md("""
        **Multi-component Amdahl's Law for distributed training:**

        ```
        Speedup = 1 / sum_i(f_i / k_i)
        ```

        where `f_i` = fraction of step time for component i, `k_i` = speedup factor
        applied to component i. When only one component is optimized (k_j, all others = 1):

        ```
        Speedup = 1 / ((1 - f_j) + f_j/k_j)
        ```

        **Pipeline bubble formula (3D-parallel training):**

        ```
        B = (PP - 1) / (PP × m)
        ```

        - **PP** — pipeline parallelism degree (number of pipeline stages)
        - **m** — number of micro-batches per training step (gradient accumulation steps)
        - **B** — bubble fraction of compute time (idle cycles waiting for pipeline fill)

        At PP=8, m=4: B = 7/32 = 21.9% of compute time.
        At PP=8, m=8: B = 7/64 = 10.9% of compute time.
        At PP=8, m=32: B = 7/256 = 2.7% of compute time.

        Note: increasing m also increases the effective batch size, which may require
        learning rate scaling (linear or sqrt scaling rule). The configuration change
        is not free — validate convergence after tuning.

        **MFU definition:**

        ```
        MFU = (Achieved FLOPS) / (Peak FLOPS × num_GPUs)
             = (tokens/sec × FLOPs/token) / (989 TFLOPS × 512)
        ```

        **Optimization ROI model:**

        ```
        ROI = (MFU_after - MFU_before) / engineering_cost_weeks
        ```

        Pipeline micro-batch tuning: ~5.6 MFU pts/week (low cost, moderate gain).
        AllReduce compression: ~1.8 MFU pts/week (high cost, larger gain).
        Data prefetch: ~2.5 MFU pts/week (moderate cost, limited ceiling).

        **When to use Gustafson's Law instead:**

        Amdahl's Law applies to fixed-workload optimization (same model, same dataset,
        same batch size). If the optimization enables *scaling* the batch size (e.g.,
        pipeline micro-batch tuning freeing memory budget for larger batches), Gustafson's
        Law provides a more accurate projection:

        ```
        Speedup_Gustafson = k - f_serial × (k - 1)
        ```

        where `f_serial` is the fraction of time that does *not* scale with workload size.
        """),
    })
    return


# ─── CELL 19: ACT II REFLECTION ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A: Pipeline bubbles represent hardware failures that require GPU replacement.":
                "A",
            "B: Pipeline bubbles scale with pipeline parallelism degree and are eliminated by configuration change (increasing micro-batches) — zero hardware cost, immediate ROI.":
                "B",
            "C: Pipeline bubbles only occur in data-parallel training when gradient synchronization is slow.":
                "C",
            "D: Reducing pipeline bubble fraction requires changing the model architecture (fewer transformer layers).":
                "D",
        },
        label="Reflection: Why is pipeline bubble reduction often the highest-ROI fix in 3D-parallel training?",
    )
    mo.vstack([
        mo.md("### Act II Reflection"),
        act2_reflection,
    ])
    return (act2_reflection,)


@app.cell(hide_code=True)
def _(act2_reflection, mo):
    mo.stop(
        act2_reflection.value is None,
        mo.callout(mo.md("Select your reflection answer to continue to the Design Ledger save."), kind="warn"),
    )
    _r = act2_reflection.value or "B"
    if _r == "B":
        mo.callout(mo.md("""
        **Correct.** Pipeline bubbles are *scheduled idle time*, not hardware faults.

        In pipeline-parallel training, the first micro-batch must propagate through all
        PP stages before the pipeline is full. The startup and drain phases always
        consume `(PP - 1)` steps of idle time. With PP=8 and m=4 micro-batches,
        7 out of every 32 compute slots are wasted as bubble:

        `B = (PP-1) / (PP × m) = 7/32 = 21.9% of compute slots`

        Increasing m to 8 (doubling gradient accumulation steps) halves this:

        `B = 7/64 = 10.9% of compute slots`

        The change requires editing one configuration parameter. No hardware changes.
        No library upgrades. No model architecture changes. No correctness risk beyond
        standard batch-size tuning. This is why it is the first optimization every
        3D-parallel training system should make before touching anything else.
        """), kind="success")
    elif _r == "A":
        mo.callout(mo.md("""
        **Incorrect.** Pipeline bubbles are expected behavior, not hardware failures.

        They occur because GPT-style transformers are divided into PP sequential stages,
        and the forward pass must complete through all stages before the backward pass
        begins. The "bubble" is the portion of each training step where some pipeline
        stages are idle, waiting for upstream stages to finish. This is fully predictable:
        `B = (PP-1)/(PP×m)`. Replacing GPUs would have no effect.
        """), kind="warn")
    elif _r == "C":
        mo.callout(mo.md("""
        **Incorrect.** Pipeline bubbles occur specifically in *pipeline-parallel* training
        (the P in 3D-parallel: Data Parallel × Tensor Parallel × Pipeline Parallel).

        Data-parallel training uses AllReduce, not pipeline stages — it has no pipeline
        bubble. Pipeline bubbles are the cost of subdividing the model into sequential
        stages across multiple nodes and training a single stream of micro-batches through
        that pipeline. They increase with PP degree and decrease with micro-batch count.
        """), kind="warn")
    else:
        mo.callout(mo.md("""
        **Incorrect.** Pipeline bubbles are independent of model architecture (number of layers).

        The bubble formula `B = (PP-1)/(PP×m)` depends only on the *pipeline partitioning*
        (how many stages PP you divide the model into) and the *micro-batch count* (m).
        A transformer with 96 layers and a transformer with 32 layers, both partitioned
        into PP=8 stages, will have the same bubble fraction at the same m. Reducing the
        layer count is a model quality decision, not a pipeline efficiency decision.
        """), kind="warn")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS ─────────────────────────────────────────────────────────
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
                    <strong>1. Profile first, always.</strong>
                    Amdahl's Law is exact: a component at f = 12% of total time with a
                    k = 3x speedup yields at most +13.6% end-to-end improvement.
                    The three weeks spent on the attention kernel would have delivered
                    3.3x more impact applied to AllReduce &mdash; a fact the profiler
                    reveals in the first hour. You cannot optimize what you cannot measure.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. In distributed training, the serial residual is distributed coordination.</strong>
                    At 512-GPU scale, the non-parallelizable fraction includes AllReduce
                    barriers (30%), pipeline bubble idle time (15%), and data loading
                    stalls (10%). Pipeline bubble reduction via micro-batch tuning
                    (m: 4 &rarr; 8) is a configuration change costing zero hardware
                    and zero library changes &mdash; typically the highest-ROI first fix.
                    A +2.8 MFU point gain on a 512-GPU H100 cluster at $3/GPU-hour
                    saves ~$47,000 per week of training time recovered.
                </div>
                <div>
                    <strong>3. Optimizing the non-binding constraint yields zero speedup.</strong>
                    The Iron Law (Time = max(Compute/FLOPS, Memory/BW) + Overhead) is not
                    a guideline &mdash; it is exact. A 95%-efficient compute kernel on a
                    memory-bound workload produces 0% end-to-end improvement because compute
                    is not in the max() term.
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
                    <strong>Lab 10: Distributed Inference</strong> &mdash; Training
                    performance is dominated by AllReduce and pipeline bubbles. The
                    next lab asks: when the same model moves from training to serving,
                    what shifts? KV-cache memory pressure and queuing dynamics replace
                    gradient synchronization as the binding constraint.
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
                    <strong>Read:</strong> @sec-performance-engineering-efficiency-frontier
                    for the full Iron Law derivation and @sec-performance-engineering-roofline
                    for the Roofline Model and arithmetic intensity analysis.<br/>
                    <strong>Build:</strong> TinyTorch Module 19 &mdash; implement a
                    profile-guided optimizer that identifies the Amdahl bottleneck
                    from a simulated trace and recommends the highest-ROI fix.
                </div>
            </div>

        </div>
        """),

        mo.accordion({


            "Self-Assessment": mo.md("""
**Check your understanding:**

1. A component at f = 12% of total time receives a k = 3x speedup. Using Amdahl's Law, what is the maximum end-to-end improvement? Why would the same engineering effort applied to a component at f = 40% deliver far more impact?
2. At 512-GPU scale, the serial residual includes AllReduce barriers (30%), pipeline bubble idle time (15%), and data loading stalls (10%). Which of these is a configuration change costing zero hardware, and why is it typically the highest-ROI first fix?
3. The Iron Law states Time = max(Compute/FLOPS, Memory/BW) + Overhead. If a workload is memory-bound, why does a 2x improvement to the compute kernel yield exactly 0% end-to-end speedup?

**You're ready to move on if you can:**
- Use Amdahl's Law to prioritize optimization targets by their maximum possible end-to-end impact
- Identify whether a workload is compute-bound or memory-bound using the roofline model and arithmetic intensity
- Explain why profiling must come before optimization and why the binding constraint determines the only productive optimization target
""")


        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ────────────────────────────────────────────────────────
# ─── CELL 21: DESIGN LEDGER SAVE + HUD ────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, _budget_exceeded, _mfu_improvement, _mfu_new,
    _total_cost_weeks, act1_pred, act1_reflection, act2_pred,
    act2_reflection, context_toggle, ledger, mo,
):
    _ctx     = context_toggle.value or "batch"
    _a1_pred = act1_pred.value or "none"
    _a1_ok   = (_a1_pred == "B")
    _a2_pred = act2_pred.value or "none"
    _a2_ok   = (_a2_pred == "B")

    ledger.save(
        chapter="v2_09",
        design={
            "context":               _ctx,
            "bottleneck_identified": "attention_12pct" if _a1_ok else "misidentified",
            "optimization_target":   _a1_pred,
            "mfu_before":            35.0,
            "mfu_after":             _mfu_new,
            "act1_prediction":       _a1_pred,
            "act1_correct":          _a1_ok,
            "act2_result":           _mfu_improvement,
            "act2_decision":         _a2_pred,
            "constraint_hit":        _budget_exceeded,
            "budget_exceeded":       _budget_exceeded,
        },
    )

    _c = COLORS["BlueLine"]
    _status_a1 = ("Correct" if _a1_ok else "Incorrect") if _a1_pred != "none" else "Not answered"
    _status_a2 = ("Correct" if _a2_ok else "Incorrect") if _a2_pred != "none" else "Not answered"
    _status_a1_color = COLORS["GreenLine"] if _a1_ok else (COLORS["RedLine"] if _a1_pred != "none" else "#94a3b8")
    _status_a2_color = COLORS["GreenLine"] if _a2_ok else (COLORS["RedLine"] if _a2_pred != "none" else "#94a3b8")
    _budget_color = COLORS["RedLine"] if _budget_exceeded else COLORS["GreenLine"]
    _budget_status = "EXCEEDED" if _budget_exceeded else "OK"

    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">Vol2 · Lab 09 · Performance Engineering</span>
            <span class="hud-label">CONTEXT</span>
            <span class="hud-value">{_ctx.replace('_', ' ').title()}</span>
            <span class="hud-label">ACT I</span>
            <span style="color: {_status_a1_color}; font-family: var(--font-mono); font-size: 0.8rem;">
                {_status_a1} ({_a1_pred})
            </span>
            <span class="hud-label">ACT II</span>
            <span style="color: {_status_a2_color}; font-family: var(--font-mono); font-size: 0.8rem;">
                {_status_a2} ({_a2_pred})
            </span>
            <span class="hud-label">MFU</span>
            <span class="hud-value">35.0% → {_mfu_new:.1f}%</span>
            <span class="hud-label">BUDGET</span>
            <span style="color: {_budget_color}; font-family: var(--font-mono); font-size: 0.8rem;">
                {_total_cost_weeks:.1f}/{8.0:.0f} wk [{_budget_status}]
            </span>
            <span class="hud-label">LEDGER</span>
            <span class="hud-active">SAVED (v2_09)</span>
        </div>
        """),
    ])
    return




if __name__ == "__main__":
    app.run()
