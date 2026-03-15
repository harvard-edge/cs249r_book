import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-01: THE SCALE ILLUSION
#
# Volume II, Chapter 1 — Introduction to Scale
#
# Core Invariant: Scale laws — single-node → fleet
#   Cost and time grow super-linearly with N; 1000× hardware does NOT deliver
#   1000× speedup. Communication overhead and coordination latency dominate.
#
# 2 Contexts:
#   Single Node  — One H100 SXM5 (baseline)
#   Fleet        — 1024-H100 cluster (the illusion target)
#
# Act I  (12–15 min): Scale Efficiency Explorer
#   Stakeholder: VP Engineering with $10M budget
#   Instruments: cluster size, parallel efficiency, communication overhead
#   Prediction: speedup achieved with 1000 GPUs
#   Overlay: predicted speedup vs. actual from physics
#   Reflection: why AllReduce limits scaling
#
# Act II (20–25 min): Fleet TCO Calculator
#   Stakeholder: CFO comparing 3 infrastructure paths
#   Instruments: GPU count, utilization, years, pricing
#   Prediction: cheapest 3-year TCO path
#   Failure state: on-demand cost > $50M budget
#   Reflection: CAPEX vs. OpEx, utilization breakeven
#
# Design Ledger: saves chapter="v2_01"
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible) ─────────────────────────
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

    # ── Hardware constants (all from NVIDIA H100 SXM5 spec and Vol2 intro) ──
    H100_BW_GBS       = 3350    # GB/s HBM3e — NVIDIA H100 SXM5 spec
    H100_TFLOPS_FP16  = 989     # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_RAM_GB       = 80      # GB HBM3e — NVIDIA spec
    H100_TDP_W        = 700     # Watts TDP — NVIDIA spec
    H100_NVLINK_BW    = 900     # GB/s bidirectional NVLink4 — NVIDIA spec
    INFINIBAND_BW_GBS = 400     # GB/s HDR200 per link — Mellanox/NVIDIA spec

    # ── Training compute constant (from Vol2 introduction.qmd) ──────────────
    # GPT-4 class model: ~2.2×10²⁴ FLOPs training compute
    # Single H100 at 50% MFU: 989 TFLOPS effective
    # Source: @sec-vol2-introduction-scale-moment
    GPT4_TRAINING_FLOPS = 2.2e24    # FLOPs — GPT-4 scale estimate
    H100_MFU_DEFAULT    = 0.50      # 50% MFU — realistic single-GPU efficiency

    ledger = DesignLedger()
    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB, H100_TDP_W,
        H100_NVLINK_BW, INFINIBAND_BW_GBS,
        GPT4_TRAINING_FLOPS, H100_MFU_DEFAULT,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 1: HEADER (hide_code=True) ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _fleet_color = COLORS["Cloud"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #1a1040 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume II · Lab 01
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Scale Illusion
            </h1>
            <p style="margin: 0 0 22px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 640px; line-height: 1.65;">
                1,000 GPUs. 1,000× speedup? The physics of distributed training
                says otherwise. Communication overhead, coordination cost, and
                failure probability all grow with cluster size — and they grow
                faster than your compute budget.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    Act I: Scaling Efficiency · Act II: Fleet TCO
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Requires: @sec-vol2-introduction-scale-moment
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Single Node: 1× H100</span>
                <span class="badge badge-info">Fleet: 1024× H100 cluster</span>
                <span class="badge badge-warn">Invariant: Speedup &lt; N</span>
                <span class="badge badge-warn">Invariant: TCO ≠ hourly rate × N</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ────────────────────────────────────────────────────────
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify fleet availability collapse: predict why a 1,000-GPU cluster at 99.9% per-GPU reliability has only a 37% chance of being fully healthy at any moment.</strong></div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose the communication wall: identify at what GPU count and interconnect type Ethernet scaling efficiency drops below 30% for a 175B-parameter model.</strong></div>
                <div style="margin-bottom: 3px;">3. <strong>Compare infrastructure TCO paths: determine the utilization breakeven point where on-premises hardware becomes cheaper than reserved cloud over a 3-year horizon.</strong></div>
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
                    Fleet reliability and the exponential collapse formula from @sec-vol2-introduction-scale-moment &middot;
                    Iron Law of Scale and AllReduce volume from @sec-vol2-introduction-fleet-law
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
                "If each GPU in your fleet has 99.9% uptime and each GPU delivers 989 TFLOPS, why does a 1,000-GPU cluster fail multiple times per day and deliver far less than 1,000&times; the throughput of a single machine?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING (hide_code=True) ────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-vol2-introduction-scale-moment** — The Scale Moment: 10-million-fold compute growth
      from AlexNet to GPT-4; why fleet scale is qualitatively different from single-node scale.
    - **@sec-vol2-introduction-engineering-crux** — The Engineering Crux: the four-layer stack
      (Hardware, Systems, Workloads, Missions) that governs every distributed design decision.
    - **@sec-vol2-introduction-breed-apart** — ML workload character: synchronous tight coupling,
      iterative statefulness, and why AllReduce dominates communication cost.

    If you have not read these sections, the predictions in this lab will not map to the physics.
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE (hide_code=True) ─────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Single Node (1 H100)": "single",
            "Fleet (1024 H100 cluster)": "fleet",
        },
        value="Single Node (1 H100)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Select your deployment context to orient the instruments:"),
        context_toggle,
    ])
    return (context_toggle,)


@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS):
    _ctx = context_toggle.value
    _is_fleet = _ctx == "fleet"
    _color = COLORS["Cloud"] if _is_fleet else COLORS["GreenLine"]
    _label = "Fleet (1024 H100 cluster)" if _is_fleet else "Single Node (1 H100)"
    _specs = (
        "1,024 H100 SXM5 GPUs · InfiniBand 400 GB/s fabric · H100 NVLink4 within nodes"
        if _is_fleet else
        "1 H100 SXM5 · 80 GB HBM3e · 3,350 GB/s memory bandwidth · 989 TFLOPS FP16"
    )
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {'#f0f4ff' if _is_fleet else '#ecfdf5'};
                border-radius: 0 10px 10px 0; padding: 14px 20px; margin: 10px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
            Active Context
        </div>
        <div style="font-weight: 700; font-size: 1.05rem; color: #1e293b;">{_label}</div>
        <div style="font-size: 0.85rem; color: #475569; margin-top: 3px;">{_specs}</div>
    </div>
    """)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER (hide_code=True) ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "I"
    _act_color    = COLORS["BlueLine"]
    _act_title    = "The Scale Illusion"
    _act_duration = "12&ndash;15 min"
    _act_why      = (
        "You expect that 1,000&times; hardware delivers 1,000&times; speedup. "
        "Amdahl&rsquo;s Law and communication overhead will show that a 1,024-GPU cluster "
        "at realistic parallel efficiency delivers a fraction of the theoretical maximum &mdash; "
        "and the gap widens as you add more nodes."
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


@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["BlueLine"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {COLORS['BlueL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · VP Engineering
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We have a GPT-4 scale model to train. Our compute team estimates it would take
            a single H100 roughly 71 years at 50% MFU. We have board approval for a $10M
            compute budget and can buy 1,000 H100s. My CFO is expecting this to take about
            0.45 years — 5.4 months. Is that realistic? What will the actual training time be?"
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Single-Node Baseline

    Before exploring the cluster, establish the physics on one H100.

    An H100 SXM5 delivers **989 TFLOPS** peak FP16 dense throughput. Real training
    workloads achieve roughly **50% Model FLOP Utilization (MFU)** — the rest is
    memory access latency, kernel launch overhead, and data loading. At 50% MFU:

    ```
    Effective throughput = 989 TFLOPS × 0.50 = 494.5 TFLOPS
    GPT-4 training compute ≈ 2.2 × 10²⁴ FLOPs
    Single-GPU time = 2.2×10²⁴ / (494.5×10¹²) / (86,400 × 365) ≈ 141 years
    ```

    The VP's $10M budget buys 1,000 H100s. Perfect linear scaling would give
    **71 years ÷ 1,000 = 0.071 years ≈ 26 days**. But distributed training
    is never perfectly linear.
    """)
    return


# ─── ACT I PREDICTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Your Prediction

    *Before touching the simulator, commit to your hypothesis:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) ~1,000× speedup — perfect linear scaling is achievable with good hardware":
                "option_a",
            "B) ~800× speedup — 80% parallel efficiency is realistic for modern clusters":
                "option_b",
            "C) ~200–400× speedup — communication overhead and stragglers reduce efficiency to 20–40%":
                "option_c",
            "D) ~100× speedup — distributed training rarely exceeds 10% parallel efficiency":
                "option_d",
        },
        label="With 1,000 H100s (instead of 1), what speedup over the single-GPU baseline can we realistically expect?",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(
            mo.md("Select your prediction to unlock the Scale Efficiency Explorer."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(f"**Prediction locked:** {act1_pred.value}. Now explore the physics below."),
        kind="info",
    )
    return


# ─── ACT I INSTRUMENTS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Scale Efficiency Explorer")
    return


@app.cell(hide_code=True)
def _(mo):
    cluster_size = mo.ui.slider(
        start=1, stop=4096, value=1000, step=1,
        label="Cluster size (GPUs)",
        show_value=True,
    )
    parallel_efficiency_pct = mo.ui.slider(
        start=10, stop=100, value=40, step=5,
        label="Parallel efficiency (%)",
        show_value=True,
    )
    comm_overhead_pct = mo.ui.slider(
        start=0, stop=60, value=20, step=5,
        label="Communication overhead (% of compute time)",
        show_value=True,
    )
    mo.vstack([
        mo.md("""
        Adjust the sliders to explore how cluster size and efficiency interact.
        **Parallel efficiency** captures how much of each GPU's compute capacity
        is usable — the rest is lost to synchronization barriers, straggler waits,
        and load imbalance. **Communication overhead** is the fraction of total
        step time consumed by AllReduce gradient synchronization.
        """),
        mo.hstack([cluster_size, parallel_efficiency_pct, comm_overhead_pct],
                  justify="start", gap="2rem"),
    ])
    return (cluster_size, parallel_efficiency_pct, comm_overhead_pct)


@app.cell(hide_code=True)
def _(
    mo, go, np, apply_plotly_theme, COLORS,
    cluster_size, parallel_efficiency_pct, comm_overhead_pct,
    GPT4_TRAINING_FLOPS, H100_TFLOPS_FP16, H100_MFU_DEFAULT,
):
    # ── Physics engine ────────────────────────────────────────────────────────
    # Source: @sec-vol2-introduction-scale-moment
    #
    # Effective throughput per GPU (TFLOPS):
    #   T_eff = H100_peak × MFU_default × (parallel_efficiency / 100)
    #
    # AllReduce communication model (ring-allreduce):
    #   T_comm_fraction = comm_overhead / 100
    #   Compute fraction = 1 - T_comm_fraction
    #
    # Actual cluster throughput:
    #   T_cluster = N × T_eff × (1 - T_comm_fraction)
    #
    # Actual speedup vs 1 GPU:
    #   speedup = T_cluster / T_single
    #
    # Training time:
    #   T_train = GPT4_FLOPS / (T_cluster × 10^12) seconds

    N = cluster_size.value
    E = parallel_efficiency_pct.value / 100.0    # parallel efficiency fraction
    C = comm_overhead_pct.value / 100.0          # communication overhead fraction

    # Single H100 effective throughput (TFLOPS)
    _t_single_tflops = H100_TFLOPS_FP16 * H100_MFU_DEFAULT
    _t_single_flops_s = _t_single_tflops * 1e12

    # Cluster effective throughput
    _t_cluster_tflops = N * H100_TFLOPS_FP16 * H100_MFU_DEFAULT * E * (1.0 - C)
    _t_cluster_flops_s = _t_cluster_tflops * 1e12

    # Ideal cluster throughput (perfect linear scaling)
    _t_ideal_tflops = N * _t_single_tflops
    _t_ideal_flops_s = _t_ideal_tflops * 1e12

    # Training times
    _SECONDS_PER_YEAR = 86400 * 365
    _SECONDS_PER_DAY  = 86400

    _t_single_years  = GPT4_TRAINING_FLOPS / _t_single_flops_s / _SECONDS_PER_YEAR
    _t_ideal_seconds = GPT4_TRAINING_FLOPS / _t_ideal_flops_s
    _t_actual_seconds = GPT4_TRAINING_FLOPS / _t_cluster_flops_s if _t_cluster_flops_s > 0 else float("inf")

    _t_ideal_days  = _t_ideal_seconds / _SECONDS_PER_DAY
    _t_actual_days = _t_actual_seconds / _SECONDS_PER_DAY

    # Speedups
    _ideal_speedup  = N                                                         # linear
    _actual_speedup = _t_single_flops_s / _t_cluster_flops_s * N if _t_cluster_flops_s > 0 else 0
    # Simplification: actual_speedup = N × E × (1 - C)
    _actual_speedup_simple = N * E * (1.0 - C)

    _scaling_efficiency = _actual_speedup_simple / N  # = E × (1 - C)

    # ── Color coding ──────────────────────────────────────────────────────────
    _eff_pct = _scaling_efficiency * 100
    _eff_color = (
        COLORS["GreenLine"] if _eff_pct >= 60 else
        COLORS["OrangeLine"] if _eff_pct >= 30 else
        COLORS["RedLine"]
    )

    # ── Speedup curve: actual vs ideal as N varies ────────────────────────────
    _ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    _ideal_curve  = _ns.astype(float)
    _actual_curve = _ns * E * (1.0 - C)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_ns, y=_ideal_curve,
        mode="lines", name="Ideal (linear scaling)",
        line=dict(color=COLORS["GreenLine"], width=2, dash="dash"),
    ))
    _fig.add_trace(go.Scatter(
        x=_ns, y=_actual_curve,
        mode="lines", name=f"Actual (E={E:.0%}, C={C:.0%})",
        line=dict(color=COLORS["BlueLine"], width=3),
        fill="tonexty", fillcolor="rgba(0,99,149,0.08)",
    ))
    # Mark current cluster size
    _current_actual = N * E * (1.0 - C)
    _fig.add_trace(go.Scatter(
        x=[N], y=[_current_actual],
        mode="markers", name=f"Current ({N} GPUs)",
        marker=dict(color=COLORS["RedLine"], size=12, symbol="diamond",
                    line=dict(color="white", width=2)),
    ))
    _fig.update_layout(
        xaxis=dict(title="Cluster Size (GPUs)", type="log",
                   tickvals=[1, 8, 64, 512, 4096],
                   ticktext=["1", "8", "64", "512", "4096"]),
        yaxis=dict(title="Speedup over single GPU", type="log"),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=10, b=50),
    )
    apply_plotly_theme(_fig)

    # ── Result display ────────────────────────────────────────────────────────
    _formula_block = f"""
**Scaling physics:**

```
Parallel efficiency (E)   = {E:.0%}
Communication overhead (C) = {C:.0%}

Actual speedup            = N × E × (1 − C)
                          = {N} × {E:.2f} × {1.0 - C:.2f}
                          = {_actual_speedup_simple:,.0f}×

Scaling efficiency        = Actual speedup / N
                          = {_actual_speedup_simple:,.0f} / {N}
                          = {_scaling_efficiency:.1%}

Ideal training time       = {_t_ideal_days:.1f} days  ({N} GPUs, perfect scaling)
Actual training time      = {_t_actual_days:.1f} days  ({N} GPUs, realistic)
Single-GPU baseline       = {_t_single_years:.0f} years
```
"""

    mo.vstack([
        mo.md(_formula_block),
        mo.Html(f"""
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin: 16px 0;">
            <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 170px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 4px;">Actual Speedup</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {_eff_color};">
                    {_actual_speedup_simple:,.0f}×
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">vs. {N:,}× ideal</div>
            </div>
            <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 170px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 4px;">Scaling Efficiency</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {_eff_color};">
                    {_scaling_efficiency:.0%}
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">speedup / N</div>
            </div>
            <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 170px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 4px;">Actual Training Time</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {COLORS['BlueLine']};">
                    {_t_actual_days:.0f} days
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">GPT-4 scale model</div>
            </div>
            <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 170px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 4px;">Ideal Training Time</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {COLORS['GreenLine']};">
                    {_t_ideal_days:.0f} days
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">perfect linear scaling</div>
            </div>
        </div>
        """),
        mo.as_html(_fig),
    ])
    return (
        _actual_speedup_simple, _scaling_efficiency,
        _t_actual_days, _t_ideal_days, _t_single_years,
        N, E, C,
    )


# ─── ACT I FEEDBACK (efficiency zones) ───────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, _scaling_efficiency, N, E, C):
    _eff_pct = _scaling_efficiency * 100
    if _eff_pct >= 60:
        mo.callout(mo.md(
            f"**Excellent scaling at {_eff_pct:.0f}% efficiency.** "
            f"With parallel efficiency E={E:.0%} and communication overhead C={C:.0%}, "
            f"this is on the optimistic end for large clusters. Real deployments at "
            f"{N:,} GPUs typically require tensor parallelism, gradient compression, "
            f"and careful AllReduce scheduling to sustain this. Validate by measuring "
            f"actual MFU during the first training run."
        ), kind="success")
    elif _eff_pct >= 30:
        mo.callout(mo.md(
            f"**Realistic scaling at {_eff_pct:.0f}% efficiency.** "
            f"This range (20–60%) is what most production clusters achieve. "
            f"At E={E:.0%} parallel efficiency and C={C:.0%} communication overhead, "
            f"the cluster is delivering meaningful throughput, but there is significant "
            f"headroom. The gap between ideal and actual reflects AllReduce synchronization "
            f"time — this is the **Bisection Bandwidth Wall** in practice."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**Poor scaling at {_eff_pct:.0f}% efficiency.** "
            f"Below 30%, communication overhead ({C:.0%}) or low parallel efficiency "
            f"({E:.0%}) is consuming most of the cluster's potential. This is a "
            f"**communication-bound** regime: adding more GPUs makes training slower "
            f"in relative terms. Reduce model size, use gradient compression, or "
            f"switch to pipeline parallelism to escape this regime."
        ), kind="danger")
    return


# ─── ACT I PREDICTION-VS-REALITY OVERLAY ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, _actual_speedup_simple, N):
    _pred_map = {
        "option_a": N,          # 1000× — linear
        "option_b": int(N * 0.8),   # 800× — 80% efficiency
        "option_c": int(N * 0.30),  # 300× midpoint of 200–400 range
        "option_d": int(N * 0.1),   # 100× — 10% efficiency
    }
    _pred_value = _pred_map.get(act1_pred.value, N)
    _actual_rounded = int(_actual_speedup_simple)
    _ratio = _actual_rounded / _pred_value if _pred_value > 0 else float("inf")
    _is_correct = act1_pred.value == "option_c"

    if _is_correct:
        mo.callout(mo.md(
            f"**Correct.** You predicted ~{_pred_value:,}×. "
            f"With the current parameters, the actual speedup is **{_actual_rounded:,}×** "
            f"— in the 200–400× range. "
            f"Communication overhead and parallel efficiency together explain the gap. "
            f"AllReduce gradient synchronization grows with N, making it the primary "
            f"bottleneck at large cluster sizes."
        ), kind="success")
    elif _ratio < 0.5:
        mo.callout(mo.md(
            f"**You predicted {_pred_value:,}× but the simulator shows {_actual_rounded:,}×** "
            f"— you were {1/_ratio:.1f}× too pessimistic. "
            f"Distributed training *can* achieve higher efficiency with "
            f"well-tuned AllReduce topology and modern NVLink interconnects. "
            f"But efficiency depends critically on communication overlap and "
            f"parallel efficiency, which you can now tune with the sliders."
        ), kind="warn")
    elif _ratio > 2.0:
        mo.callout(mo.md(
            f"**You predicted {_pred_value:,}× but the simulator shows {_actual_rounded:,}×** "
            f"— you were {_ratio:.1f}× too optimistic. "
            f"Perfect or near-perfect scaling is the **Scale Illusion**: "
            f"AllReduce communication time grows as O(N-1/N × model_size / BW), "
            f"parallel efficiency rarely exceeds 60% at large cluster sizes, "
            f"and stragglers introduce synchronization barriers. "
            f"The correct mental model: expect 20–40% scaling efficiency at {N:,} GPUs."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**You predicted {_pred_value:,}× and the simulator shows {_actual_rounded:,}×** "
            f"— within {abs(1 - _ratio):.0%}. "
            f"The scaling regime you selected matches the current efficiency parameters. "
            f"Try pushing the cluster size to 4,096 GPUs and watch how efficiency degrades."
        ), kind="success")
    return


# ─── ACT I REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reflection: What Limits Scaling?

    *Now that you have seen the physics, diagnose the root cause:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) GPUs run slower when connected over a network":
                "reflect_a",
            "B) AllReduce communication time grows with cluster size — gradient synchronization becomes the bottleneck":
                "reflect_b",
            "C) Power delivery limits per-GPU performance at large cluster sizes":
                "reflect_c",
            "D) Larger models always have lower MFU regardless of cluster size":
                "reflect_d",
        },
        label="What is the primary cause of sub-linear scaling in large GPU clusters?",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select your answer to continue to Act II."), kind="warn"),
    )
    _feedback = {
        "reflect_a": (
            "**Incorrect.** Individual GPUs do not run slower when networked — "
            "their peak TFLOPS are unchanged. The bottleneck is not per-GPU compute "
            "but the *synchronization* that networking requires: every parameter update "
            "must be globally consistent before the next forward pass begins. "
            "The GPU is idle while waiting for that synchronization to complete.",
            "warn",
        ),
        "reflect_b": (
            "**Correct.** AllReduce gradient synchronization is the primary bottleneck. "
            "In ring-AllReduce, each step transfers `2(N-1)/N × model_size` bytes "
            "across the fabric. For a 175B parameter model in FP16, that is "
            "~700 GB per step — over a 400 GB/s InfiniBand link, that is ~1.75 seconds "
            "of communication *per step*. At 1,000 GPUs, even a 20% communication "
            "overhead means 20% of every step is dead time.",
            "success",
        ),
        "reflect_c": (
            "**Incorrect.** Power delivery is a datacenter design concern but does not "
            "fundamentally limit per-GPU throughput in well-designed facilities. "
            "The GPUs continue to execute at full TFLOPS within their TDP envelope. "
            "The constraint is *network synchronization time*, not power budget.",
            "warn",
        ),
        "reflect_d": (
            "**Incorrect.** MFU is a per-GPU metric measuring how efficiently the "
            "arithmetic units are utilized. It is affected by model architecture and "
            "batch size, not cluster size per se. The scaling issue is that even "
            "perfectly MFU-efficient GPUs must stop and wait for AllReduce to complete "
            "before the next iteration — that wait time grows with N.",
            "warn",
        ),
    }
    _msg, _kind = _feedback.get(act1_reflect.value, ("", "info"))
    mo.callout(mo.md(_msg), kind=_kind)
    return


# ─── ACT I MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations": mo.md("""
        **Scaling Efficiency Formula** (from @sec-vol2-introduction-scale-moment):

        ```
        Scaling Efficiency  E_scale = Speedup / N
        Actual Speedup              = N × E_parallel × (1 - C_comm)
        E_scale                     = E_parallel × (1 - C_comm)
        ```

        **AllReduce Communication Time** (ring-AllReduce):

        ```
        T_comm = 2 × (N - 1) / N × model_size_bytes / BW_per_link
        ```

        For 175B parameters (FP16 = 2 bytes/param), N = 1024 GPUs, BW = 400 GB/s:

        ```
        model_size = 175×10⁹ × 2 = 350 GB
        T_comm     = 2 × 1023/1024 × 350 / 400
                   ≈ 1.75 seconds per AllReduce step
        ```

        **Effective Cluster Throughput**:

        ```
        T_cluster = N × T_single × E_parallel × (1 - C_comm)
        ```

        **Variables:**
        - **N** — cluster size (number of GPUs)
        - **E_parallel** — parallel efficiency (fraction of peak compute usable after load imbalance and straggler losses)
        - **C_comm** — communication overhead fraction (AllReduce time / total step time)
        - **BW_per_link** — InfiniBand bandwidth per bidirectional link (GB/s)
        - **model_size_bytes** — total parameter bytes transferred per AllReduce
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER (hide_code=True) ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "II"
    _act_color    = COLORS["OrangeLine"]
    _act_title    = "The Fleet Cost Model"
    _act_duration = "20&ndash;25 min"
    _act_why      = (
        "Act I showed that scale efficiency is far below the theoretical maximum. "
        "Now discover a deeper trap: the infrastructure path that looks cheapest per GPU-hour "
        "becomes the most expensive once utilization drops below the breakeven threshold &mdash; "
        "a number your CFO has never calculated."
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


@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["OrangeLine"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {COLORS['OrangeL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · CFO
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We need to train and serve a 70B parameter model continuously for three years.
            Our options are: (1) on-demand cloud at $2.10/GPU-hour, no commitment,
            (2) 1-year reserved cloud instances at 35% discount, or
            (3) buying our own cluster — 1,000 H100s at $40,000 each.
            I need a 3-year TCO comparison before the board meeting. Which path is cheapest?
            And what utilization rate do we need to break even on the on-prem investment?"
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Infrastructure Cost Physics

    Three paths to the same compute capacity, with fundamentally different cost structures:

    - **On-demand cloud**: Pay per GPU-hour actually used. No fixed cost. Maximum flexibility.
      Cost scales directly with utilization — but the per-hour rate is highest.
    - **Reserved instances**: Commit to 1 year at a discounted rate. Pay whether used or not.
      The discount makes sense only above a utilization breakeven point.
    - **On-premises**: CAPEX purchase. Zero marginal cost per GPU-hour after purchase.
      But: amortized hardware, power (~$0.10/kWh × 700W), cooling (1.4× power), staff.

    The key insight: **utilization determines which path wins.** On-prem with 90% utilization
    looks very different from on-prem with 20% utilization.
    """)
    return


# ─── ACT II PREDICTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) On-demand is cheapest — you only pay when actually training":
                "pred2_a",
            "B) On-prem is cheapest — zero per-hour cost after the hardware purchase":
                "pred2_b",
            "C) Reserved instances give best TCO for steady workloads; on-prem only wins above ~70% utilization":
                "pred2_c",
            "D) All three paths are within 20% of each other over 3 years":
                "pred2_d",
        },
        label="For a 1,000-GPU cluster running 3 years, which infrastructure path has the lowest TCO?",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(mo.md("Select your prediction to unlock the TCO Calculator."), kind="warn"),
    )
    mo.callout(
        mo.md(f"**Prediction locked.** Now explore the TCO model below."),
        kind="info",
    )
    return


# ─── ACT II INSTRUMENTS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Fleet TCO Calculator")
    return


@app.cell(hide_code=True)
def _(mo):
    tco_gpu_count = mo.ui.slider(
        start=100, stop=4096, value=1000, step=100,
        label="GPU count",
        show_value=True,
    )
    tco_utilization = mo.ui.slider(
        start=10, stop=100, value=60, step=5,
        label="Cluster utilization (%)",
        show_value=True,
    )
    tco_years = mo.ui.slider(
        start=1, stop=5, value=3, step=1,
        label="Planning horizon (years)",
        show_value=True,
    )
    tco_ondemand_price = mo.ui.slider(
        start=1.0, stop=5.0, value=2.10, step=0.10,
        label="On-demand price ($/GPU-hour)",
        show_value=True,
    )
    tco_reserved_discount = mo.ui.slider(
        start=10, stop=60, value=35, step=5,
        label="Reserved instance discount (%)",
        show_value=True,
    )
    tco_onprem_gpu_price = mo.ui.slider(
        start=20000, stop=80000, value=40000, step=5000,
        label="On-prem GPU purchase price ($/GPU)",
        show_value=True,
    )
    mo.vstack([
        mo.md("""
        Adjust the sliders to model different infrastructure scenarios.
        **Utilization** is the fraction of time the cluster is running training
        or inference workloads (vs. idle). **On-demand price** is the public
        cloud list price per GPU-hour (H100 class).
        """),
        mo.hstack([tco_gpu_count, tco_utilization, tco_years], justify="start", gap="2rem"),
        mo.hstack([tco_ondemand_price, tco_reserved_discount, tco_onprem_gpu_price],
                  justify="start", gap="2rem"),
    ])
    return (
        tco_gpu_count, tco_utilization, tco_years,
        tco_ondemand_price, tco_reserved_discount, tco_onprem_gpu_price,
    )


@app.cell(hide_code=True)
def _(
    mo, go, apply_plotly_theme, COLORS,
    tco_gpu_count, tco_utilization, tco_years,
    tco_ondemand_price, tco_reserved_discount, tco_onprem_gpu_price,
):
    # ── TCO physics engine ────────────────────────────────────────────────────
    # Source: @sec-vol2-introduction-engineering-crux
    #
    # On-demand TCO:
    #   Hours used = utilization × 8760 h/yr × years
    #   Cost = GPUs × hours_used × price_per_hour
    #
    # Reserved TCO (pay regardless of utilization, but at discount):
    #   Hours committed = 8760 × years (always-on commitment)
    #   Cost = GPUs × 8760 × years × price_per_hour × (1 - discount)
    #
    # On-prem TCO:
    #   CAPEX = GPUs × price_per_gpu
    #   Power cost = GPUs × TDP_W/1000 × utilization × 8760 × years × $0.10/kWh
    #   Cooling overhead = power_cost × 0.4  (PUE ~1.4)
    #   Staff = $200,000/yr per 100 GPUs (conservative estimate)
    #   Total = CAPEX + power + cooling + staff
    #
    # Breakeven utilization (on-demand vs on-prem):
    #   on_demand(U) = on_prem → solve for U

    _G   = tco_gpu_count.value
    _U   = tco_utilization.value / 100.0
    _Y   = tco_years.value
    _P   = tco_ondemand_price.value          # $/GPU-hour on-demand
    _D   = tco_reserved_discount.value / 100.0  # discount fraction
    _GPC = tco_onprem_gpu_price.value         # $/GPU purchase price

    _H_PER_YEAR = 8760                        # hours per year
    _H100_TDP_KW = 700 / 1000                 # kW per GPU — NVIDIA spec
    _POWER_COST_PER_KWH = 0.10                # $/kWh — datacenter typical
    _PUE = 1.4                                # Power Usage Effectiveness — industry average
    _STAFF_COST_PER_GPU_YEAR = 200_000 / 100  # $2,000/GPU/year (1 engineer per 100 GPUs)

    # On-demand: only pay for hours used
    _hours_used_total = _U * _H_PER_YEAR * _Y * _G
    _cost_ondemand_m  = (_hours_used_total * _P) / 1e6

    # Reserved: pay for all hours (committed), discounted
    _hours_committed_total = _H_PER_YEAR * _Y * _G
    _cost_reserved_m = (_hours_committed_total * _P * (1.0 - _D)) / 1e6

    # On-prem: CAPEX + OpEx
    _capex_m      = (_G * _GPC) / 1e6
    _power_kwh    = _G * _H100_TDP_KW * _U * _H_PER_YEAR * _Y
    _power_cost_m = (_power_kwh * _POWER_COST_PER_KWH) / 1e6
    _cooling_m    = _power_cost_m * (_PUE - 1.0)
    _staff_m      = (_G * _STAFF_COST_PER_GPU_YEAR * _Y) / 1e6
    _cost_onprem_m = _capex_m + _power_cost_m + _cooling_m + _staff_m

    # Breakeven utilization: on-demand cost = on-prem cost
    # G × U_be × H × Y × P = on-prem_total
    # U_be = on-prem_total / (G × H × Y × P)
    # (on-prem OpEx also has U in it, so iterate or approximate)
    # Approximation: treat CAPEX + staff as fixed, power+cooling as variable
    _fixed_m    = _capex_m + _staff_m
    # power_cost_m at utilization U_be:
    # power(U_be) = G × H100_TDP_KW × U_be × H × Y × $/kWh / 1e6
    # cooling(U_be) = power × (PUE - 1)
    # total_onprem(U_be) = fixed + power_factor × U_be
    _power_factor = _G * _H100_TDP_KW * _H_PER_YEAR * _Y * _POWER_COST_PER_KWH * _PUE / 1e6
    # ondemand(U_be) = G × U_be × H × Y × P / 1e6
    _demand_factor = _G * _H_PER_YEAR * _Y * _P / 1e6
    # G × U_be × H × Y × P / 1e6 = fixed + power_factor × U_be
    # U_be × (demand_factor - power_factor) = fixed
    _U_breakeven = _fixed_m / (_demand_factor - _power_factor) if (_demand_factor - _power_factor) > 0 else 1.0
    _U_breakeven = max(0.0, min(1.0, _U_breakeven))

    # ── Failure state: on-demand exceeds $50M budget ──────────────────────────
    _BUDGET_M = 50.0
    _budget_exceeded = _cost_ondemand_m > _BUDGET_M

    # ── Colors ────────────────────────────────────────────────────────────────
    _costs = [_cost_ondemand_m, _cost_reserved_m, _cost_onprem_m]
    _min_cost = min(_costs)
    _bar_colors = [
        COLORS["GreenLine"] if c == _min_cost else COLORS["BlueLine"]
        for c in _costs
    ]

    # ── TCO bar chart ─────────────────────────────────────────────────────────
    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        x=["On-Demand", "Reserved (1yr)", "On-Premises"],
        y=_costs,
        marker_color=_bar_colors,
        text=[f"${c:.1f}M" for c in _costs],
        textposition="outside",
        width=0.5,
    ))
    # Breakeven line on on-demand bar to show cost at breakeven utilization
    _fig.add_hline(
        y=_cost_onprem_m,
        line_dash="dot",
        line_color=COLORS["OrangeLine"],
        annotation_text=f"On-prem TCO: ${_cost_onprem_m:.1f}M",
        annotation_position="right",
    )
    _fig.update_layout(
        yaxis=dict(title=f"{_Y}-Year TCO ($M)", rangemode="tozero"),
        height=340,
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    apply_plotly_theme(_fig)

    # ── Utilization breakeven curve ───────────────────────────────────────────
    import numpy as _np_local
    _u_range = _np_local.linspace(0.05, 1.0, 100)
    _od_curve    = _G * _u_range * _H_PER_YEAR * _Y * _P / 1e6
    _onprem_var  = _G * _H100_TDP_KW * _u_range * _H_PER_YEAR * _Y * _POWER_COST_PER_KWH * _PUE / 1e6
    _onprem_curve = _fixed_m + _onprem_var

    _fig2 = go.Figure()
    _fig2.add_trace(go.Scatter(
        x=_u_range * 100, y=_od_curve,
        mode="lines", name="On-Demand",
        line=dict(color=COLORS["BlueLine"], width=2),
    ))
    _fig2.add_trace(go.Scatter(
        x=_u_range * 100, y=_onprem_curve,
        mode="lines", name="On-Premises",
        line=dict(color=COLORS["GreenLine"], width=2),
    ))
    # Breakeven vertical marker
    _fig2.add_vline(
        x=_U_breakeven * 100,
        line_dash="dash",
        line_color=COLORS["OrangeLine"],
        annotation_text=f"Breakeven: {_U_breakeven:.0%}",
        annotation_position="top right",
    )
    # Current utilization marker
    _fig2.add_vline(
        x=_U * 100,
        line_dash="dot",
        line_color=COLORS["RedLine"],
        annotation_text=f"Current: {_U:.0%}",
        annotation_position="bottom right",
    )
    _fig2.update_layout(
        xaxis=dict(title="Cluster Utilization (%)"),
        yaxis=dict(title=f"{_Y}-Year TCO ($M)", rangemode="tozero"),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=30, b=40),
    )
    apply_plotly_theme(_fig2)

    # ── Formula display ───────────────────────────────────────────────────────
    _cheapest = ["On-Demand", "Reserved", "On-Premises"][_costs.index(_min_cost)]
    _formula_block2 = f"""
**TCO physics ({_G:,} GPUs, {_U:.0%} utilization, {_Y} years):**

```
On-Demand:     {_G:,} × {_U:.0%} × {_H_PER_YEAR:,} h/yr × {_Y} yr × ${_P:.2f}/GPU-hr
             = ${_cost_ondemand_m:.1f}M

Reserved:      {_G:,} × 100% × {_H_PER_YEAR:,} h/yr × {_Y} yr × ${_P:.2f} × (1 − {_D:.0%})
             = ${_cost_reserved_m:.1f}M

On-Premises:
  CAPEX:       {_G:,} GPUs × ${_GPC:,}/GPU               = ${_capex_m:.1f}M
  Power:       {_G:,} × 0.70kW × {_U:.0%} × {_H_PER_YEAR:,} × {_Y}yr × $0.10/kWh  = ${_power_cost_m:.1f}M
  Cooling:     Power × (PUE−1) = ${_power_cost_m:.1f}M × 0.4  = ${_cooling_m:.1f}M
  Staff:       {_G:,} GPUs × $2k/GPU/yr × {_Y}yr          = ${_staff_m:.1f}M
  TOTAL:                                              = ${_cost_onprem_m:.1f}M

Cheapest at {_U:.0%} utilization: {_cheapest}
Breakeven utilization (on-demand vs on-prem): {_U_breakeven:.0%}
```
"""

    _result_ui = mo.vstack([
        mo.md(_formula_block2),
        mo.Html(f"""
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin: 16px 0;">
            <div style="padding: 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 150px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">On-Demand 3yr TCO</div>
                <div style="font-size: 1.8rem; font-weight: 800;
                            color: {'#CB202D' if _cost_ondemand_m == _min_cost else '#475569'};">
                    ${_cost_ondemand_m:.1f}M
                </div>
            </div>
            <div style="padding: 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 150px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">Reserved 3yr TCO</div>
                <div style="font-size: 1.8rem; font-weight: 800;
                            color: {'#008F45' if _cost_reserved_m == _min_cost else '#475569'};">
                    ${_cost_reserved_m:.1f}M
                </div>
            </div>
            <div style="padding: 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 150px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">On-Premises 3yr TCO</div>
                <div style="font-size: 1.8rem; font-weight: 800;
                            color: {'#008F45' if _cost_onprem_m == _min_cost else '#475569'};">
                    ${_cost_onprem_m:.1f}M
                </div>
            </div>
            <div style="padding: 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 150px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">Breakeven Utilization</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {COLORS['OrangeLine']};">
                    {_U_breakeven:.0%}
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">on-demand vs on-prem</div>
            </div>
        </div>
        """),
        mo.md("**3-Year TCO Comparison**"),
        mo.as_html(_fig),
        mo.md(f"**Breakeven Curve — On-Demand vs. On-Premises** (current utilization: {_U:.0%})"),
        mo.as_html(_fig2),
    ])

    # ── Failure state: on-demand exceeds $50M budget ──────────────────────────
    if _budget_exceeded:
        mo.vstack([
            mo.callout(mo.md(
                f"**On-demand cost exceeds budget.** "
                f"Required: **${_cost_ondemand_m:.1f}M** | Budget: **$50M**. "
                f"At {_G:,} GPUs × {_U:.0%} utilization × ${_P:.2f}/hr for {_Y} years, "
                f"on-demand cloud is infeasible. "
                f"Consider reserved instances (${_cost_reserved_m:.1f}M) or "
                f"on-premises infrastructure (${_cost_onprem_m:.1f}M). "
                f"Alternatively, reduce cluster utilization below "
                f"{_BUDGET_M / (_G * _H_PER_YEAR * _Y * _P):.0%} to stay within budget."
            ), kind="danger"),
            _result_ui,
        ])
    else:
        _result_ui

    return (
        _cost_ondemand_m, _cost_reserved_m, _cost_onprem_m,
        _U_breakeven, _cheapest, _budget_exceeded,
    )


# ─── ACT II FEEDBACK (TCO path analysis) ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred, _cost_ondemand_m, _cost_reserved_m, _cost_onprem_m, _U_breakeven, _cheapest):
    _feedback2 = {
        "pred2_a": (
            f"**Incorrect.** On-demand is the most expensive path at ${_cost_ondemand_m:.1f}M. "
            f"The per-hour price ($2.10/GPU-hr) is the list price for flexibility — "
            f"you pay a premium for not committing. For sustained workloads, "
            f"the premium compounds over three years. "
            f"Reserved instances reduce this by the discount factor applied to all committed hours.",
            "warn",
        ),
        "pred2_b": (
            f"**Partially correct.** On-prem at ${_cost_onprem_m:.1f}M can be cheapest, "
            f"but only above the breakeven utilization of **{_U_breakeven:.0%}**. "
            f"Below that threshold, on-prem's fixed CAPEX (hardware purchase + staff) "
            f"is not amortized over enough productive GPU-hours to beat cloud pricing. "
            f"The key insight: on-prem TCO includes power, cooling, and staff — "
            f"not just the GPU purchase price.",
            "warn" if _cheapest != "On-Premises" else "success",
        ),
        "pred2_c": (
            f"**Correct.** Reserved instances at ${_cost_reserved_m:.1f}M are optimal for "
            f"steady, predictable workloads. The breakeven between on-demand and on-prem "
            f"is **{_U_breakeven:.0%}** utilization. Above that, on-prem wins; below it, "
            f"reserved wins. On-demand is always dominated by reserved for any utilization "
            f"above zero, because you pay the undiscounted rate for every hour used.",
            "success" if _cheapest in ("Reserved", "On-Premises") else "warn",
        ),
        "pred2_d": (
            f"**Incorrect.** The three paths differ by up to "
            f"{abs(_cost_ondemand_m - _cost_onprem_m) / min(_cost_ondemand_m, _cost_onprem_m):.0%} "
            f"at this utilization level. On-demand: ${_cost_ondemand_m:.1f}M. "
            f"Reserved: ${_cost_reserved_m:.1f}M. On-prem: ${_cost_onprem_m:.1f}M. "
            f"The structure of fixed vs. variable costs creates large divergence "
            f"at scale, especially over multi-year horizons.",
            "warn",
        ),
    }
    _msg2, _kind2 = _feedback2.get(act2_pred.value, ("", "info"))
    mo.callout(mo.md(_msg2), kind=_kind2)
    return


# ─── ACT II REFLECTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) On-prem hardware degrades faster at low utilization, increasing maintenance costs":
                "r2_a",
            "B) On-prem has fixed CAPEX and OpEx regardless of use — idle hardware still costs money":
                "r2_b",
            "C) On-demand pricing scales with utilization, so both have proportional costs":
                "r2_c",
            "D) On-prem power consumption drops to zero when GPUs are idle":
                "r2_d",
        },
        label="Why does utilization dramatically affect on-prem TCO but not on-demand TCO?",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_reflect):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn"),
    )
    _r2_feedback = {
        "r2_a": (
            "**Incorrect.** Hardware degradation is a real long-term concern, but it does not "
            "explain the utilization sensitivity. An idle H100 ages similarly to an active one "
            "from a thermal-cycle perspective. The TCO sensitivity comes from the cost structure, "
            "not from accelerated wear.",
            "warn",
        ),
        "r2_b": (
            "**Correct.** On-prem TCO has a large fixed component: "
            "CAPEX (hardware purchase), staff costs, and baseline facility overhead accrue "
            "whether the GPUs are running training or sitting idle. "
            "At 20% utilization, you pay the full fixed cost but amortize it over only 20% "
            "of available GPU-hours — making your effective cost per productive GPU-hour 5× higher "
            "than the theoretical peak. On-demand eliminates this: you pay only for hours used.",
            "success",
        ),
        "r2_c": (
            "**Incorrect.** On-demand pricing is pay-per-hour-used, so your total cost "
            "is proportional to hours used (and thus utilization). But on-prem has a "
            "large *fixed* CAPEX component that does not scale with utilization. "
            "That asymmetry — fixed cost vs. variable cost — is what makes "
            "the breakeven utilization meaningful.",
            "warn",
        ),
        "r2_d": (
            "**Incorrect.** Idle GPUs consume roughly 50% of their TDP in idle power states "
            "(not zero). An H100 at idle draws ~350W vs 700W at full load. "
            "Additionally, facility cooling and staff costs are nearly constant regardless "
            "of GPU activity. These fixed ongoing costs are why on-prem TCO does not scale "
            "down linearly with utilization.",
            "warn",
        ),
    }
    _msg3, _kind3 = _r2_feedback.get(act2_reflect.value, ("", "info"))
    mo.callout(mo.md(_msg3), kind=_kind3)
    return


# ─── ACT II MATHPEEK ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations": mo.md("""
        **3-Year TCO Models:**

        ```
        On-Demand TCO  = N × U × 8760 × Y × P_demand

        Reserved TCO   = N × 8760 × Y × P_demand × (1 − D_reserved)
                       [committed whether used or not]

        On-Prem TCO    = CAPEX + Power + Cooling + Staff

          CAPEX        = N × P_gpu
          Power        = N × TDP_kW × U × 8760 × Y × $/kWh
          Cooling      = Power × (PUE − 1)        [PUE ≈ 1.4]
          Staff        = N × $2,000/GPU/yr × Y
        ```

        **Breakeven Utilization (on-demand vs. on-prem):**

        Solving `OnDemand(U_be) = OnPrem(U_be)`:

        ```
        N × U_be × H × Y × P = CAPEX + Staff + N × TDP × U_be × H × Y × $/kWh × PUE

        U_be = (CAPEX + Staff) / (N × H × Y × (P − TDP × $/kWh × PUE))
        ```

        **Variables:**
        - **N** — GPU count
        - **U** — utilization fraction (0–1)
        - **Y** — planning horizon (years)
        - **P** — on-demand price ($/GPU-hour)
        - **D** — reserved discount fraction
        - **TDP_kW** — GPU thermal design power in kilowatts (H100: 0.70 kW)
        - **PUE** — Power Usage Effectiveness (total facility power / IT power)
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 21: LEDGER_HUD ─────────────────────────────────────────────────────
@app.cell
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS, context_toggle,
    act1_pred, act1_reflect, act2_pred, act2_reflect,
    cluster_size, parallel_efficiency_pct, comm_overhead_pct,
    _actual_speedup_simple, _scaling_efficiency,
    _cost_ondemand_m, _cost_reserved_m, _cost_onprem_m,
    _U_breakeven, _cheapest, _budget_exceeded,
    tco_gpu_count, tco_utilization,
, decision_input, decision_ui):
    _ctx = context_toggle.value
    _infra_map = {
        "pred2_a": "on_demand",
        "pred2_b": "on_prem",
        "pred2_c": "reserved",
        "pred2_d": "on_demand",
    }
    _infra = _infra_map.get(act2_pred.value or "pred2_a", "on_demand")

    _design = {
        "context":              _ctx,
        "cluster_size":         cluster_size.value,
        "parallel_efficiency":  parallel_efficiency_pct.value / 100.0,
        "communication_overhead": comm_overhead_pct.value / 100.0,
        "infrastructure_choice": _infra,
        "tco_3yr":              round(min(_cost_ondemand_m, _cost_reserved_m, _cost_onprem_m), 2),
        "act1_prediction":      act1_pred.value or "none",
        "act1_correct":         act1_pred.value == "option_c",
        "act1_reflect_correct": act1_reflect.value == "reflect_b",
        "act2_result":          round(min(_cost_ondemand_m, _cost_reserved_m, _cost_onprem_m), 2),
        "act2_decision":        _cheapest,
        "constraint_hit":       _budget_exceeded,
        "student_justification": str(decision_input.value),
        "scaling_efficiency":   round(_scaling_efficiency, 3),
        "breakeven_utilization": round(_U_breakeven, 3),
    }
    ledger.save(chapter="v2_01", design=_design)

    # ── HUD footer ────────────────────────────────────────────────────────────
    _act1_done = act1_pred.value is not None
    _act2_done = act2_pred.value is not None
    _reflect1_done = act1_reflect.value is not None
    _reflect2_done = act2_reflect.value is not None

    _dot = lambda done: (
        f'<span style="color: #4ade80;">&#9679;</span>' if done
        else f'<span style="color: #f87171;">&#9675;</span>'
    )

    mo.Html(f"""
    <div style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap;
                padding: 14px 24px; background: #0f172a; border-radius: 12px;
                margin-top: 32px; font-family: 'SF Mono', monospace; font-size: 0.8rem;
                border: 1px solid #1e293b;">
        <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">
            LAB v2-01
        </span>
        <span>
            {_dot(_act1_done)} <span style="color: #e2e8f0;">Act I Prediction</span>
        </span>
        <span>
            {_dot(_reflect1_done)} <span style="color: #e2e8f0;">Act I Reflection</span>
        </span>
        <span>
            {_dot(_act2_done)} <span style="color: #e2e8f0;">Act II Prediction</span>
        </span>
        <span>
            {_dot(_reflect2_done)} <span style="color: #e2e8f0;">Act II Reflection</span>
        </span>
        <span style="margin-left: auto; color: #94a3b8;">
            Ledger: <span style="color: {'#4ade80' if _act1_done and _act2_done else '#f87171'};">
                {'SAVED' if _act1_done and _act2_done else 'INCOMPLETE'}
            </span>
        </span>
        <span style="color: #94a3b8;">
            Context: <span style="color: #a5b4fc;">{_ctx}</span>
        </span>
        <span style="color: #94a3b8;">
            Scaling Eff: <span style="color: #fcd34d;">{_scaling_efficiency:.0%}</span>
        </span>
        <span style="color: #94a3b8;">
            Best TCO: <span style="color: #6ee7b7;">{_cheapest} (${min(_cost_ondemand_m, _cost_reserved_m, _cost_onprem_m):.1f}M)</span>
        </span>
    </div>
    """)
    return


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
                    <strong>1. Fleet reliability collapses exponentially.</strong>
                    At 1,000 GPUs and 99.9% per-GPU uptime, fleet availability is (0.999)^1000 &asymp; 37% &mdash;
                    more likely broken than healthy. At GPT-4 scale (25,000 GPUs), a hardware failure
                    occurs every 4.4 hours. Engineering for failure is not optional; it is the primary design constraint.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Communication, not compute, sets the scaling ceiling.</strong>
                    For a 175B-parameter model on Ethernet (12.5 GB/s), Ring AllReduce takes 56 seconds
                    against 1.2 seconds of compute &mdash; communication consumes 97% of step time and
                    Amdahl&apos;s Law caps maximum speedup at 5&times; regardless of GPU count.
                </div>
                <div>
                    <strong>3. Infrastructure TCO is governed by utilization, not hourly rate.</strong>
                    On-premises beats reserved cloud only above the breakeven utilization threshold
                    (typically 60&ndash;75%). Below breakeven, fixed CAPEX, power, and cooling costs
                    accumulate regardless of whether GPUs are computing &mdash; a trap invisible
                    in per-hour pricing comparisons.
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
                    <strong>Lab V2-02: The Interconnect Wall</strong> &mdash; This lab showed that
                    Ethernet collapses scaling efficiency to &lt;30%. The next lab asks: why does
                    NVLink at 900 GB/s create a hard boundary between intra-node and inter-node
                    parallelism strategies?
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
                    <strong>Read:</strong> @sec-vol2-introduction-scale-moment for the Fleet Law derivation
                    and @sec-vol2-introduction-fleet-law for the Iron Law of Scale.<br/>
                    <strong>Build:</strong> TinyTorch distributed module &mdash; implement a ring-AllReduce
                    in <code>tinytorch/src/distributed/</code>.
                </div>
            </div>

        </div>
        """),


        mo.accordion({
            "Self-Assessment": mo.md("""
**Check your understanding:**

1. A 1,000-GPU cluster with 99.9% per-GPU uptime has what fleet availability? Why does the Fleet Law (MTBF_cluster = MTBF_gpu / N) mean that engineering for failure is the primary design constraint at scale?
2. For a 175B-parameter model on Ethernet, Ring AllReduce takes 56 seconds against 1.2 seconds of compute. Why does Amdahl's Law cap maximum speedup at roughly 5x regardless of how many GPUs you add?
3. On-premises infrastructure accumulates fixed costs whether GPUs are computing or idle. At what utilization range does on-prem break even against reserved cloud, and why is this threshold invisible in per-hour pricing comparisons?

**You're ready to move on if you can:**
- Calculate fleet availability from per-GPU uptime using (reliability)^N
- Explain why communication overhead, not compute, sets the scaling ceiling for large models
- Identify the utilization breakeven point that determines the TCO winner between on-prem and cloud
""")
        }),
    ])
    return


if __name__ == "__main__":
    app.run()
