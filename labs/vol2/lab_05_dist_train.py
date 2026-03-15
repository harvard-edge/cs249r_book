import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 05: THE PARALLELISM PARADOX
#
# Chapter: Distributed Training Systems (@sec-distributed-training-systems)
# Core Invariant: The Parallelism Paradox — adding more GPUs to data parallel
#                 training increases communication overhead, which can decrease
#                 MFU below single-GPU levels for large models. 3D parallelism
#                 (Tensor + Pipeline + Data) is required for models that don't
#                 fit on a single GPU, but each dimension adds overhead.
#
# 2-Act Structure (35-40 minutes):
#   Act I  — The Data Parallel Wall (12-15 min)
#             A 7B model trained with DP across 8→64→512 GPUs shows MFU
#             collapsing from 52% to 19%. The central question: why does MFU
#             fall as we add more GPUs? Students must confront that communication
#             time grows relative to compute time as cluster size grows.
#
#   Act II — 3D Parallelism Design Challenge (20-25 min)
#             Design the TP×PP×DP configuration for GPT-3 175B on 1024 H100s.
#             The failure state: per-GPU memory exceeds 80 GB (model doesn't fit)
#             and a bandwidth penalty warning when TP crosses node boundaries.
#
# Deployment Contexts:
#   DP:         Data Parallel — replicate model, sync gradients via AllReduce
#   3D Parallel: TP×PP×DP — within-node TP (NVLink), cross-node PP (IB), DP
#
# Hardware Constants:
#   H100_TFLOPS_FP16  = 989     # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
#   H100_BW_GBS       = 3350    # GB/s HBM3e; source: NVIDIA H100 spec sheet
#   H100_RAM_GB       = 80      # GB HBM3e; source: NVIDIA H100 spec sheet
#   NVLINK4_BW_GBS    = 900     # GB/s NVLink 4; source: NVIDIA DGX H100 spec
#   IB_HDR200_BW_GBS  = 400     # GB/s InfiniBand HDR200; source: Mellanox spec
#   GPUS_PER_NODE     = 8       # Standard DGX H100 node size
#
# Design Ledger: saves chapter="v2_05" with DP vs 3D context, parallelism
#                degrees, MFU achieved, prediction accuracy, failure states.
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
    from plotly.subplots import make_subplots

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
    return COLORS, LAB_CSS, apply_plotly_theme, go, ledger, math, mo, np, make_subplots


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, LAB_CSS, mo):
    _c_dp   = COLORS["BlueLine"]
    _c_3d   = COLORS["Cloud"]
    _c_s0   = COLORS["Surface0"]
    _c_s1   = COLORS["Surface1"]
    _header = mo.Html(f"""
    {LAB_CSS}
    <div style="background: linear-gradient(135deg, {_c_s0} 0%, {_c_s1} 100%);
                border-radius: 16px; padding: 32px 40px; margin-bottom: 8px;
                border: 1px solid #2d3748;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                            text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;">
                    Vol 2 &middot; Lab 05 &middot; Distributed Training Systems
                </div>
                <div style="font-size: 2.0rem; font-weight: 800; color: #f1f5f9; line-height: 1.15; margin-bottom: 10px;">
                    The Parallelism Paradox
                </div>
                <div style="font-size: 0.95rem; color: #94a3b8; max-width: 600px; line-height: 1.6;">
                    Adding GPUs to a data-parallel job can reduce MFU (Model FLOPS Utilization)
                    below single-GPU levels. This lab forces you to confront the
                    communication-computation ratio and design the 3D parallelism
                    configuration that keeps 1024 H100s productive.
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;">
                <span class="badge badge-info">Parallelism Paradox</span>
                <span class="badge badge-info">AllReduce Bandwidth Model</span>
                <span class="badge badge-info">3D Parallel: TP &times; PP &times; DP</span>
                <span class="badge badge-warn">35&ndash;40 minutes &middot; 2 Acts</span>
            </div>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap;">
            <div style="background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.35);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_dp}; font-weight: 700;">Context A — Data Parallel</span>
                <span style="color: #94a3b8;"> &mdash; 7B model &middot; 8&ndash;512 GPUs &middot; AllReduce via NVLink / IB</span>
            </div>
            <div style="background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.35);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_3d}; font-weight: 700;">Context B — 3D Parallel</span>
                <span style="color: #94a3b8;"> &mdash; 175B model &middot; 1024 H100s &middot; TP&times;PP&times;DP design</span>
            </div>
        </div>
    </div>
    """)
    _header
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the communication wall:</strong> derive the AllReduce transfer time as a function of model size, GPU count, and interconnect bandwidth, and explain why DP (Data Parallelism) parallel efficiency collapses when crossing the intra-node to inter-node boundary.</div>
                <div style="margin-bottom: 3px;">2. <strong>Identify the budget-optimal configuration:</strong> use the scaling efficiency formula to determine whether gradient accumulation on fewer GPUs can match the effective batch size of a larger cluster at lower cost, and explain when TP (Tensor Parallelism) within-node and PP (Pipeline Parallelism) across-node are required.</div>
                <div style="margin-bottom: 3px;">3. <strong>Diagnose pipeline bubble waste:</strong> apply the bubble fraction formula B = (PP &minus; 1) / (PP &times; m) to evaluate when pipeline overhead renders a configuration infeasible.</div>
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
                    Scaling efficiency formula and data parallelism from @sec-distributed-training-systems &middot;
                    AllReduce communication volume (2&times; gradient size) from @sec-collective-communication
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
                "You have a fixed training budget and need to complete GPT-2 training as fast as possible &mdash; does adding more GPUs always help, and at what point does the communication overhead of a larger cluster outweigh the compute benefit?"
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

    - **@sec-distributed-training-systems-systems-multimachine-scaling-fundamentals-ff96** — The Iron Law of Scale: `T_step(N) = T_compute/N + T_comm(N) - T_overlap` and the Communication-Computation Ratio
    - **@sec-distributed-training-systems** — Why distribution is necessary: memory exhaustion, training duration, and dataset scale thresholds
    - The Data Parallelism section — AllReduce gradient synchronization, Ring-AllReduce bandwidth formula, gradient bucketing
    - The 3D Parallelism section — Tensor Parallelism (within-node), Pipeline Parallelism (across nodes), pipeline bubble fraction `B = (PP-1)/(PP * m)`
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _c_muted = COLORS["TextMuted"]
    context_toggle = mo.ui.radio(
        options={
            "Data Parallel (DP)": "dp",
            "3D Parallel (TP+PP+DP)": "3d",
        },
        value="Data Parallel (DP)",
        label="Deployment context for this session:",
        inline=True,
    )
    mo.hstack([
        mo.Html(f"""
        <div style="font-size:0.78rem; font-weight:700; color:{_c_muted};
                    text-transform:uppercase; letter-spacing:0.08em;
                    margin-right:8px; padding-top:2px;">
            Active Context:
        </div>
        """),
        context_toggle,
    ], justify="start", gap=0)
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 5: ACT1_BANNER (hide_code=True) ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "I"
    _act_color    = COLORS["BlueLine"]
    _act_title    = "The Data Parallel Wall"
    _act_duration = "12&ndash;15 min"
    _act_why      = (
        "You expect that moving from 8 intra-node GPUs to 32 inter-node GPUs roughly quadruples "
        "training throughput. Quantify how parallel efficiency changes when data parallelism crosses "
        "the intra-node NVLink boundary to inter-node Ethernet, and identify the dominant term "
        "in the AllReduce cost that causes the efficiency cliff."
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
    _color = COLORS["BlueLine"]
    _bg    = COLORS["BlueL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Training Infrastructure Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We are training a 7B parameter model using data parallelism. I measured MFU at
            8 GPUs = 52%. At 64 GPUs it dropped to 38%. At 512 GPUs it's 19%. The model
            hasn't changed. The batch size per GPU hasn't changed. We just added more GPUs
            and it got worse. We're wasting $40,000 per day in idle compute. Can you tell
            me exactly why MFU falls as we scale data parallelism?"
        </div>
    </div>
    """)
    return


# ─── ACT I: CONCEPT FRAMING ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The training lead's observation is not a software bug. It is the **Parallelism Paradox**:
    the physical law that governs every data-parallel training job.

    Data parallel training replicates the model on every GPU, runs a forward and backward
    pass on each device's local batch, then synchronizes gradients across all devices via
    **AllReduce** before the optimizer step. The AllReduce is unavoidable — without it,
    each replica would diverge. The critical question is how long AllReduce takes relative
    to the compute step.

    The **Communication-Computation Ratio** from @sec-distributed-training-systems determines
    whether a cluster behaves as a supercomputer or as a collection of idling heaters:

    - **Compute-Bound (Low Ratio)**: `T_compute >> T_comm`. GPUs spend most time on matrix
      multiplications. This is the ideal state.
    - **Communication-Bound (High Ratio)**: `T_comm ≈ T_compute`. GPUs spend significant
      time waiting for gradients. This is the common state for LLMs at scale.

    Before looking at any numbers, commit to a prediction about what causes MFU to fall.
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
            "A) Software overhead in NCCL and collective libraries grows with cluster size": "A",
            "B) AllReduce communication time grows with cluster size while compute time stays constant — the comm/compute ratio rises": "B",
            "C) Larger clusters cause L2 cache pressure and HBM bandwidth saturation per GPU": "C",
            "D) 512 GPUs exceeds optimal batch size — gradient quality degrades and more steps are needed": "D",
        },
        label="Why does Model FLOPs Utilization (MFU) fall as we scale a data-parallel job from 8 to 512 GPUs?",
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


# ─── ACT I: INSTRUMENT PANEL INTRO ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Data Parallel Scaling Explorer

    Adjust the parameters below to see how AllReduce communication time compares to
    compute time — and how their ratio determines MFU at each scale point.
    """)
    return


# ─── ACT I: SLIDERS ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    dp_model_b = mo.ui.slider(
        start=1, stop=175, value=7, step=1,
        label="Model size (B params)",
    )
    dp_gpus = mo.ui.slider(
        start=8, stop=1024, value=8, step=8,
        label="Number of GPUs (DP degree)",
    )
    dp_batch_per_gpu = mo.ui.slider(
        start=8, stop=128, value=32, step=8,
        label="Micro-batch size per GPU",
    )
    dp_interconnect = mo.ui.dropdown(
        options={"NVLink 4 (within DGX node, 8 GPUs)": "nvlink", "InfiniBand HDR200 (cross-node)": "ib"},
        value="NVLink 4 (within DGX node, 8 GPUs)",
        label="Interconnect fabric",
    )
    mo.hstack([
        mo.vstack([dp_model_b, dp_gpus]),
        mo.vstack([dp_batch_per_gpu, dp_interconnect]),
    ], justify="center", gap=2)
    return dp_batch_per_gpu, dp_gpus, dp_interconnect, dp_model_b


# ─── ACT I: PHYSICS ENGINE ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, dp_batch_per_gpu, dp_gpus, dp_interconnect, dp_model_b, go, math, mo, np):
    # ── Flight Simulator: MLSys·im Engine Evaluation ─────────────────────────────
    import mlsysim

    _params_b  = dp_model_b.value
    _gpus      = dp_gpus.value
    _batch_gpu = dp_batch_per_gpu.value
    _fabric    = dp_interconnect.value

    # Use the hardware registry
    h100 = mlsysim.Hardware.Cloud.H100
    
    # Network fabric based on selection
    if _fabric == "nvlink" and _gpus <= 8:
        fabric = mlsysim.Systems.Fabrics.NVLink_4
    else:
        fabric = mlsysim.Systems.Fabrics.InfiniBand_HDR
        
    _forced_ib = (_fabric == "nvlink" and _gpus > 8)
    _effective_bw = fabric.bandwidth.m_as("GB/s")
    _fabric_label = fabric.name

    # Build the cluster
    node = mlsysim.Systems.Node(name="DGX H100", accelerator=h100, accelerators_per_node=8, intra_node_bw=mlsysim.Systems.Fabrics.NVLink_4.bandwidth)
    fleet = mlsysim.Systems.Fleet(name="Custom Cluster", node=node, count=max(1, _gpus // 8), fabric=fabric)
    
    # Build the model workload
    model = mlsysim.Models.Generic(
        parameters=mlsysim.Q_(_params_b * 1e9, "count"),
        inference_flops=mlsysim.Q_(2.0 * _params_b * 1e9, "flop") # 2N inference, 6N training handled inside engine
    )

    # Solve via the engine
    solver = mlsysim.DistributedModel()
    res = solver.solve(
        model=model,
        fleet=fleet,
        batch_size=_batch_gpu * _gpus, # Global batch size
        precision="fp16",
        efficiency=0.52, # Reference MFU
        tp_size=1,
        pp_size=1,
        overlap_comm=False # Turn off overlap for raw AllReduce math
    )

    _compute_time_s = res.node_profile.latency.m_as("s")
    _allreduce_time_s = res.dp_communication_latency.m_as("s")
    _total_time_s = res.step_latency_total.m_as("s")
    _mfu_pct = res.scaling_efficiency * 52.0 # Effective MFU = scaling_efficiency * baseline
    _cc_ratio = _allreduce_time_s / _compute_time_s if _compute_time_s > 0 else 0

    # ── Build MFU vs GPU count curve ─────────────────────────────────────────────
    _gpu_range     = [8, 16, 32, 64, 128, 256, 512, 1024]
    _mfu_curve     = []
    for _g in _gpu_range:
        # Evaluate each point on the curve using the engine
        _curve_fleet = mlsysim.Systems.Fleet(
            name="Sweep Cluster", 
            node=node, 
            count=max(1, _g // 8), 
            fabric=mlsysim.Systems.Fabrics.NVLink_4 if (_fabric == "nvlink" and _g <= 8) else mlsysim.Systems.Fabrics.InfiniBand_HDR
        )
        _curve_res = solver.solve(
            model=model, fleet=_curve_fleet, batch_size=_batch_gpu * _g,
            precision="fp16", efficiency=0.52, tp_size=1, pp_size=1, overlap_comm=False
        )
        _mfu_curve.append(_curve_res.scaling_efficiency * 52.0)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_gpu_range, y=_mfu_curve,
        mode="lines+markers",
        line=dict(color=COLORS["BlueLine"], width=2.5),
        marker=dict(size=8, color=COLORS["BlueLine"]),
        name="MFU (model)",
        hovertemplate="<b>%{x} GPUs</b><br>MFU: %{y:.1f}%<extra></extra>",
    ))
    # Mark the current selection
    _fig.add_trace(go.Scatter(
        x=[_gpus], y=[_mfu_pct],
        mode="markers",
        marker=dict(size=16, color=COLORS["RedLine"], symbol="diamond",
                    line=dict(color="white", width=2)),
        name="Current config",
        hovertemplate="<b>Current: %{x} GPUs</b><br>MFU: %{y:.1f}%<extra></extra>",
    ))
    # Reference points from stakeholder message
    _ref_x = [8, 64, 512]
    _ref_y = [52, 38, 19]
    _fig.add_trace(go.Scatter(
        x=_ref_x, y=_ref_y,
        mode="markers",
        marker=dict(size=12, color=COLORS["OrangeLine"], symbol="x",
                    line=dict(color=COLORS["OrangeLine"], width=3)),
        name="Measured (stakeholder)",
        hovertemplate="<b>Measured: %{x} GPUs</b><br>MFU: %{y:.0f}%<extra></extra>",
    ))
    _fig.add_hline(y=40.0, line=dict(color=COLORS["GreenLine"], width=1.5, dash="dash"),
                   annotation_text="40% — practical floor", annotation_position="bottom right")
    _fig.update_layout(
        height=320,
        xaxis=dict(title="GPU Count (DP degree)", type="log",
                   tickvals=[8, 16, 32, 64, 128, 256, 512, 1024],
                   ticktext=["8", "16", "32", "64", "128", "256", "512", "1024"]),
        yaxis=dict(title="Model FLOPs Utilization (%)", range=[0, 65]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=50, l=50, r=20),
    )
    apply_plotly_theme(_fig)

    # ── Color coding ─────────────────────────────────────────────────────────────
    _mfu_color  = COLORS["GreenLine"] if _mfu_pct >= 45 else (COLORS["OrangeLine"] if _mfu_pct >= 25 else COLORS["RedLine"])
    _cc_color   = COLORS["GreenLine"] if _cc_ratio <= 0.3 else (COLORS["OrangeLine"] if _cc_ratio <= 0.8 else COLORS["RedLine"])

    # ── Forced-IB warning ────────────────────────────────────────────────────────
    _ib_warn = ""
    if _forced_ib:
        _ib_warn = f"""
        <div style="background:{COLORS['OrangeLL']}; border:1px solid {COLORS['OrangeLine']};
                    border-radius:8px; padding:10px 14px; margin:8px 0; font-size:0.85rem;">
            <strong style="color:{COLORS['OrangeLine']};">Interconnect Upgrade Applied:</strong>
            NVLink operates within a single DGX node (8 GPUs). At {_gpus} GPUs, traffic crosses
            node boundaries. The physics engine automatically fell back to InfiniBand.
        </div>
        """

    # ── Physics display ──────────────────────────────────────────────────────────
    mo.vstack([
        mo.Html(f"""
        {_ib_warn}
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">MFU</div>
                <div style="font-size:2.2rem; font-weight:800; color:{_mfu_color};
                            font-family:monospace;">{_mfu_pct:.1f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">model FLOPs utilization</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">Comm / Compute</div>
                <div style="font-size:2.2rem; font-weight:800; color:{_cc_color};
                            font-family:monospace;">{_cc_ratio:.2f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">ratio (lower = better)</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">AllReduce Time</div>
                <div style="font-size:2.2rem; font-weight:800; color:{COLORS['BlueLine']};
                            font-family:monospace;">{_allreduce_time_s*1000:.1f}ms</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">per step</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">Compute Time</div>
                <div style="font-size:2.2rem; font-weight:800; color:{COLORS['BlueLine']};
                            font-family:monospace;">{_compute_time_s*1000:.1f}ms</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">per step (ref MFU)</div>
            </div>
        </div>
        """),
        mo.ui.plotly(_fig),
        mo.accordion({
            "⚙️ Under the Hood: How MLSys·im Calculates This": mo.md(f"""
            This "Flight Simulator" runs the exact `mlsysim` engine used in the textbook.
            Here is the code executing the distributed scaling model in the background:
            
            ```python
            import mlsysim
            
            # 1. Define the distributed fleet
            fabric = mlsysim.Systems.Fabrics.InfiniBand_HDR
            node = mlsysim.Systems.Node(name="DGX", accelerator=mlsysim.Hardware.Cloud.H100)
            fleet = mlsysim.Systems.Fleet(name="Cluster", node=node, count={max(1, _gpus // 8)}, fabric=fabric)
            
            # 2. Evaluate the Ring-AllReduce communication overhead
            solver = mlsysim.DistributedModel()
            res = solver.solve(
                model=mlsysim.Models.Generic(parameters={_params_b}e9),
                fleet=fleet,
                batch_size={_batch_gpu * _gpus}, # Global batch size
                precision="fp16",
                efficiency=0.52
            )
            
            print(f"Scaling Efficiency: {{res.scaling_efficiency * 100}}%")
            print(f"AllReduce Penalty: {{res.dp_communication_latency}}")
            ```
            """)
        })
    ])
    return (
        _gpus,
    )


# ─── ACT I: PREDICTION REVEAL ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(act1_pred, mo):
    _correct = act1_pred.value == "B"
    if _correct:
        mo.callout(mo.md(
            "**Correct.** Option B identifies the root cause: the Ring-AllReduce transfer volume "
            "is essentially constant (approximately 2 × gradient size regardless of N for large N), "
            "but it must traverse InfiniBand at 400 GB/s when the cluster spans multiple nodes "
            "instead of NVLink at 900 GB/s within a node. The compute time per GPU does not change "
            "as you add GPUs. Therefore the comm/compute ratio rises with cluster size, "
            "directly reducing MFU. The simulator above shows this as the cliff in the MFU curve "
            "between 8 and 64 GPUs where traffic transitions from NVLink to InfiniBand."
        ), kind="success")
    elif act1_pred.value == "A":
        mo.callout(mo.md(
            "**Not the primary cause.** NCCL is highly optimized and adds minimal overhead "
            "relative to wire transfer time. The dominant factor is the physical bandwidth "
            "of the interconnect, not the software library overhead. At 512 GPUs, "
            "the AllReduce transfer itself consumes ~140 ms on InfiniBand while the compute "
            "step takes ~60 ms — NCCL overhead is negligible compared to this ratio."
        ), kind="warn")
    elif act1_pred.value == "C":
        mo.callout(mo.md(
            "**Not the primary cause.** Each GPU's local computation is unchanged — the "
            "same model, same batch size per GPU, same forward and backward pass. "
            "Cache pressure and HBM bandwidth utilization per GPU are essentially identical "
            "regardless of whether you are running with 8 or 512 GPUs. The bottleneck "
            "is between nodes, not within them."
        ), kind="warn")
    elif act1_pred.value == "D":
        mo.callout(mo.md(
            "**A real phenomenon, but not the cause here.** Gradient quality degradation "
            "with very large global batch sizes is a real concern (the linear scaling rule "
            "breaks above a critical batch size), but the stakeholder explicitly notes that "
            "batch size per GPU is unchanged. Total global batch = 512 GPUs × 32 = 16,384. "
            "For a 7B model this is well within the stable scaling regime. The MFU drop "
            "is communication-bound, not convergence-bound."
        ), kind="warn")
    else:
        mo.callout(mo.md("Select a prediction above to see the reveal."), kind="info")
    return


# ─── ACT I: ACT I MATHPEEK ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — AllReduce bandwidth model and DP efficiency": mo.md("""
        **Ring-AllReduce Transfer Volume (per GPU)**

        ```
        T_allreduce = [2 × (N-1)/N × grad_bytes] / BW_interconnect
        ```

        - `N` — number of data-parallel replicas (GPU count)
        - `grad_bytes` — gradient tensor size = params × 2 bytes (FP16)
        - `BW_interconnect` — 900 GB/s (NVLink 4, within node) or 400 GB/s (IB HDR200, cross-node)
        - For large N: the factor 2×(N-1)/N → 2, so AllReduce volume saturates at ~2× gradient size
        - **Key insight**: AllReduce volume does NOT grow linearly with N — it saturates. But the
          bandwidth cliff when crossing the node boundary (NVLink → IB) creates a step-change in latency.

        **DP Efficiency Formula**

        ```
        MFU_effective = (T_compute / (T_compute + T_allreduce)) × MFU_ref
        ```

        - `T_compute` — forward + backward FLOPs / (peak_TFLOPS × MFU_ref)
        - `T_allreduce` — grows as cluster spans more nodes (IB replaces NVLink)
        - When T_allreduce ≈ T_compute (ratio ≈ 1), effective MFU ≈ MFU_ref / 2

        **Gradient Bucketing Analysis**

        ```
        T_effective = max(T_compute_late_layers, T_allreduce_early_gradients)
        ```

        Gradient bucketing starts AllReduce for early-layer gradients while later layers
        are still computing. Ideal overlap: T_effective → T_compute (hiding communication).
        In practice, overlapping achieves 60–80% communication hiding for large models.
        """)
    })
    return


# ─── ACT I: REFLECTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reflection

    Now that you have explored the AllReduce bottleneck, consider the primary technique
    practitioners use to reclaim efficiency: overlapping communication with computation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) Use faster GPUs to reduce compute time — this shrinks the total step time": "A",
            "B) Gradient bucketing + async AllReduce — begin communicating early-layer gradients while computing late-layer gradients": "B",
            "C) Reduce batch size per GPU to reduce the gradient tensor size and shorten AllReduce": "C",
            "D) Quantize gradients to INT8 for AllReduce communication, then dequantize before the optimizer step": "D",
        },
        label="What is the primary technique to overlap communication with computation in data parallel training?",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(act1_reflect, mo):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"),
    )
    if act1_reflect.value == "B":
        mo.callout(mo.md(
            "**Correct.** Gradient bucketing partitions the gradient tensor into chunks. "
            "During the backward pass, as soon as the gradients for the last few layers are "
            "computed, AllReduce begins on those buckets while the backward pass continues "
            "computing gradients for earlier layers. This overlaps the two operations. "
            "PyTorch DDP implements this via `bucket_cap_mb` (default: 25 MB). "
            "For a 7B model with 14 GB of gradients, effective overlap can hide 60–80% of "
            "the AllReduce latency, recovering significant MFU at scale."
        ), kind="success")
    elif act1_reflect.value == "A":
        mo.callout(mo.md(
            "**This does not reduce the comm/compute ratio.** A faster GPU shortens T_compute, "
            "which makes the numerator in the ratio smaller — but it also reduces the time "
            "available to overlap communication. The ratio T_allreduce/T_compute can actually "
            "worsen as compute gets faster while interconnect bandwidth stays constant. "
            "This is a common misconception: hardware upgrades on the compute side do not "
            "solve interconnect-bound scaling."
        ), kind="warn")
    elif act1_reflect.value == "C":
        mo.callout(mo.md(
            "**This reduces the wrong dimension.** Gradient dimensions are determined by model "
            "architecture, not batch size. A 7B model has 7B parameters regardless of whether "
            "the local batch is 8 or 128 samples. Reducing batch size per GPU does reduce "
            "gradient noise (smaller effective batch = higher gradient variance), but it does "
            "not reduce AllReduce volume. The gradient tensor size is `params × 2 bytes` in FP16."
        ), kind="warn")
    elif act1_reflect.value == "D":
        mo.callout(mo.md(
            "**Partially true, but not the primary technique.** INT8 gradient compression "
            "can reduce AllReduce volume by 2× compared to FP16, but it introduces gradient "
            "quantization error that can harm convergence for sensitive training runs. "
            "BF16 gradients are standard in modern training. The more reliable approach is "
            "gradient bucketing and async AllReduce, which hides rather than reduces "
            "communication — recovering throughput without precision loss."
        ), kind="warn")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 12: ACT2_BANNER (hide_code=True) ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "II"
    _act_color    = COLORS["OrangeLine"]
    _act_title    = "3D Parallelism Design Challenge"
    _act_duration = "20&ndash;25 min"
    _act_why      = (
        "Act I showed that data parallelism collapses at inter-node scale. Now tackle a "
        "model that does not fit on a single GPU: GPT-3 (175B parameters) on 1,024 H100s. "
        "3D parallelism (TP&times;PP&times;DP) is the only option &mdash; but each "
        "dimension adds its own overhead, and violating the bandwidth hierarchy triggers "
        "OOM failures that no amount of clever scheduling can fix."
    )
    mo.Html(f"""
    <div style="margin: 32px 0 12px 0; border-top: 2px solid {COLORS['Border']}; padding-top: 32px;">
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


# ─── ACT II: STAKEHOLDER MESSAGE ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _color = COLORS["Cloud"]
    _bg    = COLORS["BlueLL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; MLOps Architect
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We need to train GPT-3 (175B parameters). A single H100 holds 80 GB of HBM3e.
            With FP16 weights, FP32 optimizer states, and activation buffers, a 175B model
            needs roughly 10 bytes per parameter in practice — about 1.75 TB total, which
            doesn't fit in any single GPU. We have 1024 H100s available across 128 DGX nodes.
            Design the 3D parallel configuration (TP × PP × DP) that maximizes MFU without
            exceeding per-GPU memory or creating a pipeline bubble fraction above 10%."
        </div>
    </div>
    """)
    return


# ─── ACT II: CONCEPT FRAMING ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    When a model exceeds single-GPU memory capacity, data parallelism alone cannot help.
    Three orthogonal strategies exist for distributing a model:

    - **Tensor Parallelism (TP)**: Split individual matrix operations across GPUs within a layer.
      Every forward pass requires an AllReduce across the TP group. TP must operate at high
      bandwidth — otherwise the AllReduce overhead dominates. This constrains TP to
      **within a single DGX node** (NVLink, 900 GB/s).

    - **Pipeline Parallelism (PP)**: Assign consecutive layers to consecutive GPUs.
      Requires microbatching to keep all pipeline stages busy. Introduces **bubble overhead**:
      `B = (PP - 1) / (PP × m)` where `m` is the number of microbatches.

    - **Data Parallelism (DP)**: Replicate the TP×PP model group and distribute the
      global batch. This scales to the remaining GPU budget after TP and PP are fixed.
      `DP = total_GPUs / (TP × PP)`.

    The 3D configuration space has a hard constraint: `TP × PP × DP = 1024`.

    Before using the configurator, predict the optimal configuration.
    """)
    return


# ─── ACT II: PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Your Configuration Prediction")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) TP=128, PP=1, DP=8 — maximize tensor parallelism to spread every layer across 128 GPUs": "A",
            "B) TP=8, PP=4, DP=32 — within-node TP on NVLink, pipeline across nodes, DP for throughput scale": "B",
            "C) TP=1, PP=1024, DP=1 — pure pipeline parallelism to avoid AllReduce entirely": "C",
            "D) TP=4, PP=256, DP=1 — deep pipeline to maximize layer-level parallelism": "D",
        },
        label="Which 3D parallel configuration (TP × PP × DP) best balances memory, compute, and communication for GPT-3 175B on 1024 H100s?",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(act2_pred, mo):
    mo.stop(
        act2_pred.value is None,
        mo.callout(
            mo.md("Select your configuration prediction above to unlock the Act II instruments."),
            kind="warn",
        ),
    )
    mo.md("")
    return


# ─── ACT II: INSTRUMENT PANEL INTRO ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3D Parallelism Configurator

    Adjust TP and PP degrees. DP is computed automatically from the constraint
    `TP × PP × DP = 1024`. The configurator will enforce per-GPU memory and
    pipeline bubble constraints.
    """)
    return


# ─── ACT II: SLIDERS ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    tp_degree = mo.ui.slider(
        start=1, stop=64, value=8, step=1,
        label="Tensor Parallelism degree (TP)",
    )
    pp_degree = mo.ui.slider(
        start=1, stop=64, value=4, step=1,
        label="Pipeline Parallelism degree (PP)",
    )
    n_microbatches = mo.ui.slider(
        start=1, stop=64, value=8, step=1,
        label="Microbatches per pipeline flush (m)",
    )
    mo.hstack([
        mo.vstack([tp_degree, pp_degree]),
        mo.vstack([n_microbatches]),
    ], justify="center", gap=2)
    return n_microbatches, pp_degree, tp_degree


# ─── ACT II: PHYSICS ENGINE ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS,
    GPUS_PER_NODE,
    H100_RAM_GB,
    H100_TFLOPS_FP16,
    IB_HDR200_BW_GBS,
    NVLINK4_BW_GBS,
    apply_plotly_theme,
    go,
    math,
    mo,
    n_microbatches,
    np,
    pp_degree,
    tp_degree,
):
    # ── Model constants — GPT-3 175B ─────────────────────────────────────────────
    # Source: @sec-distributed-training-systems, Brown et al. 2020 (GPT-3 paper)
    GPT3_PARAMS_B       = 175.0       # 175B parameters; Brown et al. 2020
    GPT3_LAYERS         = 96          # transformer layers; Brown et al. 2020
    BYTES_PER_PARAM_FP16 = 2          # FP16 model weights
    OPTIMIZER_OVERHEAD   = 8          # FP32 optimizer states (m1+m2+master) ≈ 8 bytes/param
    ACTIVATION_BYTES_GB  = 8.0        # activation buffers per pipeline stage (estimate)
    TOTAL_BYTES_PER_PARAM = 10        # practical: weights + grads + optimizer ≈ 10 bytes/param
    TOTAL_GPUS           = 1024       # available H100s
    MFU_BASE             = 0.52       # reference MFU for calibration
    TP_ALLREDUCE_LAYERS  = GPT3_LAYERS  # TP AllReduce happens every layer

    # ── Extract widget values ────────────────────────────────────────────────────
    _tp = tp_degree.value
    _pp = pp_degree.value
    _m  = n_microbatches.value

    # ── Constraint: TP × PP × DP = 1024 ─────────────────────────────────────────
    _tp_pp_product = _tp * _pp
    _dp = TOTAL_GPUS // _tp_pp_product if _tp_pp_product <= TOTAL_GPUS else 0
    _dp_remainder  = TOTAL_GPUS % _tp_pp_product if _tp_pp_product > 0 else 1
    _config_valid  = (_dp > 0) and (_dp_remainder == 0)

    # ── Memory analysis ──────────────────────────────────────────────────────────
    # Per-GPU memory = model shards + optimizer + activations
    # TP shards model parameters: each GPU holds 1/TP of each tensor
    # PP assigns GPT3_LAYERS/PP layers to each stage
    _params_per_gpu_b  = GPT3_PARAMS_B / (_tp * _pp)   # billions
    _params_per_gpu    = _params_per_gpu_b * 1e9
    _model_mem_gb      = _params_per_gpu * BYTES_PER_PARAM_FP16 / 1e9
    _optim_mem_gb      = _params_per_gpu * OPTIMIZER_OVERHEAD / 1e9
    _activ_mem_gb      = ACTIVATION_BYTES_GB / (_tp if _tp > 1 else 1)  # TP partitions activations
    _total_mem_gb      = _model_mem_gb + _optim_mem_gb + _activ_mem_gb

    # ── Failure state: OOM ───────────────────────────────────────────────────────
    _oom = _total_mem_gb > H100_RAM_GB

    # ── Pipeline bubble fraction ─────────────────────────────────────────────────
    # Source: @sec-distributed-training-systems pipeline parallelism section
    # B = (PP - 1) / (PP × m)
    _bubble_frac = (_pp - 1) / (_pp * _m) if _pp > 1 else 0.0
    _bubble_pct  = _bubble_frac * 100.0
    _bubble_warn = _bubble_pct > 10.0

    # ── TP communication overhead ────────────────────────────────────────────────
    # TP AllReduce volume per layer = 2 × hidden_dim × seq_len × 2 bytes (FP16)
    # Simplified: TP communication time relative to compute
    # Each TP AllReduce per layer uses NVLink (within node) or IB (cross-node)
    _tp_crosses_node = _tp > GPUS_PER_NODE
    _tp_fabric_bw    = IB_HDR200_BW_GBS if _tp_crosses_node else NVLINK4_BW_GBS
    _tp_bw_penalty   = NVLINK4_BW_GBS / _tp_fabric_bw  # 1.0 if NVLink, 2.25 if IB
    _tp_warn         = _tp_crosses_node

    # ── Effective MFU estimation ─────────────────────────────────────────────────
    # TP penalty: communication overhead from intra-layer AllReduce
    # PP penalty: pipeline bubble fraction
    # DP penalty: AllReduce for gradients (small for large DP with gradient bucketing)
    _tp_comm_penalty  = 1.0 - (0.05 * math.log2(max(_tp, 1)) * _tp_bw_penalty)   # rough empirical model
    _pp_efficiency    = 1.0 - _bubble_frac
    _dp_comm_penalty  = 1.0 - (0.02 * math.log2(max(_dp, 1)))  # gradient AllReduce overhead
    _mfu_effective    = MFU_BASE * _tp_comm_penalty * _pp_efficiency * _dp_comm_penalty
    _mfu_effective    = max(0.0, min(_mfu_effective, MFU_BASE))
    _mfu_pct_3d       = _mfu_effective * 100.0

    # ── Color coding ─────────────────────────────────────────────────────────────
    _mem_color    = COLORS["RedLine"]  if _oom           else (COLORS["OrangeLine"] if _total_mem_gb > 60 else COLORS["GreenLine"])
    _bubble_color = COLORS["RedLine"]  if _bubble_warn   else (COLORS["OrangeLine"] if _bubble_pct > 5 else COLORS["GreenLine"])
    _mfu_color_3d = COLORS["RedLine"]  if _mfu_pct_3d < 25 else (COLORS["OrangeLine"] if _mfu_pct_3d < 40 else COLORS["GreenLine"])
    _cfg_color    = COLORS["GreenLine"] if _config_valid else COLORS["RedLine"]

    # ── FAILURE STATE: OOM ───────────────────────────────────────────────────────
    _oom_banner = ""
    if _oom:
        _oom_banner = f"""
        <div style="background:{COLORS['RedLL']}; border:2px solid {COLORS['RedLine']};
                    border-radius:10px; padding:14px 18px; margin:10px 0;">
            <div style="font-size:0.88rem; font-weight:800; color:{COLORS['RedLine']}; margin-bottom:4px;">
                OOM — Configuration Infeasible
            </div>
            <div style="font-size:0.85rem; color:#7f1d1d; line-height:1.6;">
                <strong>Required per GPU: {_total_mem_gb:.1f} GB</strong> &mdash; exceeds H100 limit: {H100_RAM_GB:.0f} GB.<br>
                Model shard: {_model_mem_gb:.1f} GB &nbsp;|&nbsp; Optimizer states: {_optim_mem_gb:.1f} GB &nbsp;|&nbsp; Activations: {_activ_mem_gb:.1f} GB.<br>
                Increase TP or PP to reduce the per-GPU model shard below {H100_RAM_GB - _activ_mem_gb:.0f} GB (leaving room for activations).
            </div>
        </div>
        """

    # ── WARNING STATE: TP crosses node boundary ───────────────────────────────────
    _tp_bw_banner = ""
    if _tp_warn:
        _penalty_x = NVLINK4_BW_GBS / IB_HDR200_BW_GBS
        _tp_bw_banner = f"""
        <div style="background:{COLORS['OrangeLL']}; border:1px solid {COLORS['OrangeLine']};
                    border-radius:8px; padding:12px 16px; margin:8px 0;">
            <div style="font-size:0.85rem; font-weight:700; color:{COLORS['OrangeLine']}; margin-bottom:4px;">
                Tensor Parallelism Crosses Node Boundary
            </div>
            <div style="font-size:0.83rem; color:#7c2d12; line-height:1.6;">
                TP={_tp} exceeds GPUS_PER_NODE={GPUS_PER_NODE}. TP AllReduce uses
                InfiniBand HDR200 ({IB_HDR200_BW_GBS:.0f} GB/s) instead of NVLink 4
                ({NVLINK4_BW_GBS:.0f} GB/s) &mdash; a <strong>{_penalty_x:.1f}&times; bandwidth penalty</strong>
                on every layer's AllReduce. TP should remain &le; {GPUS_PER_NODE} to exploit
                NVLink within a single DGX node.
            </div>
        </div>
        """

    # ── Config validity warning ───────────────────────────────────────────────────
    _cfg_banner = ""
    if not _config_valid:
        _cfg_banner = f"""
        <div style="background:{COLORS['RedLL']}; border:1px solid {COLORS['RedLine']};
                    border-radius:8px; padding:12px 16px; margin:8px 0;">
            <div style="font-size:0.85rem; font-weight:700; color:{COLORS['RedLine']};">
                Invalid Configuration: TP &times; PP = {_tp} &times; {_pp} = {_tp_pp_product}
                does not divide 1024 evenly. Choose TP and PP such that 1024 / (TP &times; PP)
                is a positive integer.
            </div>
        </div>
        """

    # ── Build bubble fraction vs PP/m chart ──────────────────────────────────────
    _pp_range = list(range(1, 33))
    _bubble_m1  = [(_p - 1) / (_p * 1)  * 100 for _p in _pp_range]
    _bubble_m4  = [(_p - 1) / (_p * 4)  * 100 for _p in _pp_range]
    _bubble_m8  = [(_p - 1) / (_p * 8)  * 100 for _p in _pp_range]
    _bubble_m16 = [(_p - 1) / (_p * 16) * 100 for _p in _pp_range]

    _fig2 = go.Figure()
    for _vals, _label, _clr in [
        (_bubble_m1,  "m=1 microbatch",  "#cb202d"),
        (_bubble_m4,  "m=4 microbatches", "#cc5500"),
        (_bubble_m8,  "m=8 microbatches", "#006395"),
        (_bubble_m16, "m=16 microbatches", "#008f45"),
    ]:
        _fig2.add_trace(go.Scatter(
            x=_pp_range, y=_vals, mode="lines", name=_label,
            line=dict(color=_clr, width=2),
            hovertemplate=f"PP=%{{x}} {_label}<br>Bubble: %{{y:.1f}}%<extra></extra>",
        ))
    _fig2.add_hline(y=10.0, line=dict(color="#1e293b", width=1.5, dash="dash"),
                    annotation_text="10% bubble ceiling", annotation_position="top right")
    # Mark current config
    _fig2.add_trace(go.Scatter(
        x=[_pp], y=[_bubble_pct],
        mode="markers", name="Current config",
        marker=dict(size=14, color=COLORS["RedLine"], symbol="diamond",
                    line=dict(color="white", width=2)),
        hovertemplate=f"PP={_pp}, m={_m}<br>Bubble: {_bubble_pct:.1f}%<extra></extra>",
    ))
    _fig2.update_layout(
        height=300,
        xaxis=dict(title="Pipeline Parallelism (PP stages)", range=[1, 32]),
        yaxis=dict(title="Pipeline Bubble Fraction (%)", range=[0, 55]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=50, l=50, r=20),
    )
    apply_plotly_theme(_fig2)

    # ── Render all outputs ────────────────────────────────────────────────────────
    mo.vstack([
        mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;
                        font-family:sans-serif;">
                Physics — 3D Parallel Memory and Bubble Analysis
            </div>
            <div>Configuration: TP={_tp} &times; PP={_pp} &times; DP={_dp if _config_valid else "N/A"}
                 {'= ' + str(TOTAL_GPUS) if _config_valid else '(INVALID: TP&times;PP=' + str(_tp_pp_product) + ' does not divide 1024)'}</div>
            <div>Params per GPU = {GPT3_PARAMS_B}B / (TP={_tp} &times; PP={_pp}) = <strong>{_params_per_gpu_b:.2f}B params</strong></div>
            <div>Model memory (FP16) = {_params_per_gpu_b:.2f}B &times; 2 bytes = <strong>{_model_mem_gb:.1f} GB</strong></div>
            <div>Optimizer states (FP32) = {_params_per_gpu_b:.2f}B &times; 8 bytes = <strong>{_optim_mem_gb:.1f} GB</strong></div>
            <div>Activation buffer = <strong>{_activ_mem_gb:.1f} GB</strong> (estimated)</div>
            <div>Total per-GPU memory = <strong style="color:{_mem_color};">{_total_mem_gb:.1f} GB</strong> / {H100_RAM_GB:.0f} GB limit</div>
            <div>Pipeline bubble B = (PP-1)/(PP&times;m) = ({_pp}-1)/({_pp}&times;{_m}) = <strong style="color:{_bubble_color};">{_bubble_pct:.1f}%</strong></div>
            <div>TP bandwidth = <strong>{'InfiniBand ' + str(IB_HDR200_BW_GBS) + ' GB/s (CROSS-NODE)' if _tp_crosses_node else 'NVLink 4 ' + str(NVLINK4_BW_GBS) + ' GB/s (within node)'}</strong></div>
            <div>Effective MFU = <strong style="color:{_mfu_color_3d};">{_mfu_pct_3d:.1f}%</strong></div>
        </div>
        {_oom_banner}
        {_tp_bw_banner}
        {_cfg_banner}
        """),
        mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">Per-GPU Memory</div>
                <div style="font-size:2.2rem; font-weight:800; color:{_mem_color};
                            font-family:monospace;">{_total_mem_gb:.0f}GB</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">/ {H100_RAM_GB:.0f} GB limit</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">Pipeline Bubble</div>
                <div style="font-size:2.2rem; font-weight:800; color:{_bubble_color};
                            font-family:monospace;">{_bubble_pct:.1f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">ceiling: 10%</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">Effective MFU</div>
                <div style="font-size:2.2rem; font-weight:800; color:{_mfu_color_3d};
                            font-family:monospace;">{_mfu_pct_3d:.1f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">3D parallel</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">DP degree</div>
                <div style="font-size:2.2rem; font-weight:800; color:{_cfg_color};
                            font-family:monospace;">{_dp if _config_valid else 'N/A'}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">= 1024 / (TP&times;PP)</div>
            </div>
        </div>
        """),
        mo.ui.plotly(_fig2),
    ])
    return (
        _bubble_pct,
        _config_valid,
        _dp,
        _mfu_pct_3d,
        _oom,
        _total_mem_gb,
        _tp_crosses_node,
    )


# ─── ACT II: PREDICTION REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(act2_pred, mo):
    _correct = act2_pred.value == "B"
    if _correct:
        mo.callout(mo.md(
            "**Correct.** TP=8, PP=4, DP=32 is the principled baseline for GPT-3 scale training. "
            "TP=8 maps exactly to one DGX node (8 GPUs per node), keeping TP AllReduce on "
            "NVLink at 900 GB/s. PP=4 assigns 96/4=24 transformer layers per stage, "
            "requiring 4 nodes per pipeline. With 8 microbatches, the pipeline bubble "
            "B=(4-1)/(4×8)=9.375% stays just under the 10% ceiling. DP=32 then "
            "replicates the TP×PP group 32 times across the remaining 1024/(8×4)=32 GPU groups. "
            "This matches the configuration used in real GPT-3-scale training runs "
            "on DGX clusters (Megatron-LM, 2021)."
        ), kind="success")
    elif act2_pred.value == "A":
        mo.callout(mo.md(
            "**Infeasible.** TP=128 distributes each layer across 128 GPUs. Each tensor "
            "parallel AllReduce must traverse 16 DGX nodes (128/8=16), using InfiniBand "
            "instead of NVLink — a 2.25× bandwidth penalty on every single layer forward and "
            "backward pass. The AllReduce occurs 96 times per forward pass (once per transformer "
            "layer). At IB bandwidth this becomes the dominant bottleneck, crushing MFU. "
            "Configure TP in the simulator with TP > 8 to observe the bandwidth penalty warning."
        ), kind="warn")
    elif act2_pred.value == "C":
        mo.callout(mo.md(
            "**Catastrophic bubble overhead.** PP=1024 with a single microbatch gives "
            "B=(1024-1)/(1024×1)≈99.9% bubble fraction — the cluster is 99.9% idle. "
            "Even with m=64 microbatches: B=(1024-1)/(1024×64)≈1.5%, but now each "
            "gradient accumulation step is enormous, harming optimizer convergence. "
            "Pure pipeline parallelism with depth matching GPU count is never used in practice. "
            "Use the configurator to set PP=1024 and observe the bubble fraction."
        ), kind="warn")
    elif act2_pred.value == "D":
        mo.callout(mo.md(
            "**Pipeline bubble too large.** PP=256 with m=8 microbatches gives "
            "B=(256-1)/(256×8)=12.4% — already over the 10% ceiling. "
            "You would need m=32 microbatches to bring the bubble to 3.1%, "
            "but that requires a batch size of 32×256=8,192 sequences through the pipeline "
            "before each optimizer step, creating a very large effective batch. "
            "With DP=1, there is no data parallelism to amortize the batch size requirement. "
            "This is an over-pipelined design."
        ), kind="warn")
    else:
        mo.callout(mo.md("Select a configuration prediction above to see the analysis."), kind="info")
    return


# ─── ACT II: MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — 3D parallel memory, bubble fraction, and TP communication": mo.md("""
        **3D Parallel Per-GPU Memory**

        ```
        mem_per_gpu = (params / (TP × PP)) × bytes_per_param
                    + (params / (TP × PP)) × optimizer_bytes_per_param
                    + activation_buffer
        ```

        - `params` — total model parameters (e.g. 175B for GPT-3)
        - `TP × PP` — reduces the parameter shard on each GPU
        - `bytes_per_param` — FP16 = 2 bytes; FP32 master copy = 4 bytes
        - `optimizer_bytes_per_param` — Adam states: 2 FP32 moments + master = ~8 bytes/param
        - **Key insight**: TP and PP jointly reduce per-GPU memory — TP shards each matrix
          horizontally, PP shards the depth (layers). DP does NOT reduce memory: every DP replica
          holds the full TP×PP model shard.

        **Pipeline Bubble Fraction**

        ```
        B = (PP - 1) / (PP × m)
        ```

        - `PP` — pipeline parallelism degree (stages)
        - `m` — number of microbatches per pipeline flush
        - At PP=4, m=8: B = 3/32 = 9.375%
        - **Key insight**: increasing m (microbatches) reduces bubble but increases pipeline latency
          and may harm optimizer convergence at very large effective batch sizes.
        - Practical ceiling: B < 10% is standard in production (Megatron-LM guidelines).

        **Tensor Parallelism Communication Volume (per layer)**

        ```
        TP AllReduce per layer = 2 × (TP - 1)/TP × hidden_dim × seq_len × 2 bytes (FP16)
        ```

        - Occurs **every layer** in both forward and backward passes
        - At 900 GB/s (NVLink): ~0.5 ms per layer for a 175B model configuration
        - At 400 GB/s (IB): ~1.1 ms per layer — 2.25× slower, applied 96 times per forward pass
        - **Key insight**: TP communication is not a one-time cost — it is a per-layer tax.
          This is why TP > 8 (crossing node boundary to IB) destroys MFU.
        """)
    })
    return


# ─── ACT II: REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reflection

    You observed that TP=8 is the natural constraint boundary. Before finishing, confirm
    your understanding of why this boundary is fundamental.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) PyTorch does not support cross-node tensor parallelism in its distributed primitives": "A",
            "B) Tensor parallel AllReduce happens every layer — at InfiniBand bandwidth this becomes the dominant bottleneck": "B",
            "C) Tensor parallelism requires shared GPU memory, which is unavailable across separate nodes": "C",
            "D) Cross-node tensor parallelism causes numerical instability due to floating-point rounding across nodes": "D",
        },
        label="Why must tensor parallelism be confined within a single DGX node (TP ≤ 8)?",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(act2_reflect, mo):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"),
    )
    if act2_reflect.value == "B":
        mo.callout(mo.md(
            "**Correct.** Tensor parallelism introduces an AllReduce after every transformer "
            "layer's matrix operations — both in the forward pass and the backward pass. "
            "For a 96-layer model like GPT-3, that is 192 AllReduce calls per training step. "
            "At NVLink bandwidth (900 GB/s) this adds ~1 ms per step — tolerable. "
            "At InfiniBand bandwidth (400 GB/s), the penalty is 2.25× higher and accumulates "
            "across all 96 layers, making TP communication the dominant step time. "
            "The constraint TP ≤ GPUS_PER_NODE (≤ 8) is not a software limitation; "
            "it is a bandwidth physics constraint."
        ), kind="success")
    elif act2_reflect.value == "A":
        mo.callout(mo.md(
            "**Incorrect.** PyTorch (via Megatron-LM's column/row parallel linear layers) "
            "and frameworks like DeepSpeed fully support cross-node tensor parallelism "
            "using the standard NCCL AllReduce over InfiniBand. The constraint is physical, "
            "not a software limitation. The code works fine; the bandwidth penalty is what "
            "makes cross-node TP undesirable."
        ), kind="warn")
    elif act2_reflect.value == "C":
        mo.callout(mo.md(
            "**Incorrect.** Tensor parallelism does not require shared memory. It is a "
            "message-passing strategy: each GPU holds a shard of the weight matrix, "
            "computes a partial matrix multiply on its shard, then the partial results are "
            "reduced via AllReduce across all TP ranks. This works equally over NVLink "
            "or InfiniBand — the difference is only bandwidth and therefore latency."
        ), kind="warn")
    elif act2_reflect.value == "D":
        mo.callout(mo.md(
            "**Incorrect.** Floating-point arithmetic in distributed training uses deterministic "
            "reduction primitives (NCCL's AllReduce). The numerical behavior is identical whether "
            "the AllReduce traverses NVLink or InfiniBand — both use the same FP16/BF16 precision "
            "operations. Numerical instability in distributed training typically arises from "
            "gradient accumulation order (non-associative floating-point operations), not from "
            "the physical transport medium."
        ), kind="warn")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

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

            <div style="background:linear-gradient(to right, #f8fafc, #f1f5f9); border-radius:8px; padding:20px; margin-bottom:24px; border-left:4px solid #8b5cf6;">
                <div style="font-weight:800; font-size:1.1rem; color:#6d28d9; margin-bottom:8px;">💎 The Iron Law Nugget</div>
                <div style="color:#334155; font-size:1rem; font-style:italic; line-height:1.6;">
                    "Scaling a model across more GPUs does not guarantee faster training. The physical bandwidth of the network fabric strictly bounds scaling efficiency. When communication time exceeds compute time, adding more hardware yields negative returns."
                </div>
                <div style="margin-top:12px; font-size:0.8rem; color:#64748b;">
                    <strong>Source:</strong> Adapted from the Megatron-LM 3D Parallelism constraints defined in <em>Shoeybi, M., et al. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.</em>
                </div>
            </div>

            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. Scaling from 8 to 32 GPUs on 10GbE collapses parallel efficiency from 97% to 13%.</strong>
                    The 5.6 GB gradient sync at 1.25 GB/s takes 4,805 ms per step against 1,800 ms of compute.
                    GPUs spend 73% of each step waiting for gradients. The 720&times; bandwidth gap between NVLink
                    (900 GB/s) and 10GbE (1.25 GB/s) is the root cause &mdash; not the GPU count.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Eight GPUs with gradient accumulation beats 32 GPUs at 87% lower cost.</strong>
                    8 GPUs with 4-step accumulation achieves the same effective batch size of 512 as 32 GPUs,
                    costs $422 vs. $3,021, and keeps all communication on NVLink (overhead &lt;0.1%).
                    The cheapest configuration often uses the fewest GPUs connected by the fastest network.
                </div>
                <div>
                    <strong>3. Pipeline bubble waste scales as (PP &minus; 1) / (PP &times; m) and exceeds 40% when PP &gg; m.</strong>
                    At 8 pipeline stages and 32 microbatches, bubble fraction is (8&minus;1)/(8&times;32) = 2.7% &mdash; acceptable.
                    At 16 stages and 8 microbatches, it rises to (16&minus;1)/(16&times;8) = 11.7% &mdash; over the 10% ceiling.
                    Keeping m &gg; PP is the constraint that determines pipeline parallelism feasibility.
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
                    <strong>Lab V2-06: The Bandwidth Invariant</strong> &mdash; This lab showed that
                    AllReduce communication dominates training cost at scale. The next lab asks: which
                    AllReduce algorithm (Ring vs. Tree) wins for large gradient payloads, and where
                    exactly is the crossover where Tree outperforms Ring?
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
                    <strong>Read:</strong> @sec-distributed-training-systems for the full scaling
                    efficiency derivation, gradient accumulation analysis, and pipeline bubble formula.<br/>
                    <strong>Build:</strong> TinyTorch distributed training module &mdash; implement
                    data parallelism with gradient accumulation in <code>tinytorch/src/distributed/</code>.
                </div>
            </div>

        </div>
        """),

        mo.accordion({


            "Self-Assessment": mo.md("""
**Check your understanding:**

1. Scaling from 8 intra-node GPUs (NVLink) to 32 cross-node GPUs (10GbE) collapses parallel efficiency from 97% to 13%. What is the gradient sync time on 10GbE versus compute time, and at what GPU count does adding more hardware yield negative returns?
2. Pipeline bubble waste follows the formula (PP - 1) / (PP x m). At 8 pipeline stages, what microbatch count keeps bubble fraction below 10%? Why does this formula, not software tuning, determine pipeline parallelism feasibility?
3. Eight GPUs with gradient accumulation achieve the same effective batch size as 32 GPUs at 87% lower cost. What makes this possible, and why is the cheapest configuration often the one using the fewest GPUs on the fastest interconnect?

**You're ready to move on if you can:**
- Calculate parallel efficiency given compute time and gradient synchronization time for a specific interconnect
- Use the pipeline bubble formula to determine feasible PP-stage and microbatch configurations
- Compare cost-effectiveness of gradient accumulation versus scaling out across a slow network
""")


        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ─────────────────────────────────────────────────────
# ─── LEDGER SAVE + HUD ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    COLORS,
    _bubble_pct,
    _config_valid,
    _dp,
    _mfu_pct_3d,
    _oom,
    _total_mem_gb,
    _tp_crosses_node,
    act1_pred,
    act2_pred,
    act2_reflect,
    act1_reflect,
    ledger,
    mo,
    n_microbatches,
    pp_degree,
    tp_degree,
    decision_input, decision_ui
):
    # ── Save to Design Ledger ────────────────────────────────────────────────────
    _context = "3d_parallel" if tp_degree.value > 1 or pp_degree.value > 1 else "data_parallel"

    ledger.save(
        chapter="v2_05",
        design={
            "context":         _context,
            "tp_degree":       tp_degree.value,
            "pp_degree":       pp_degree.value,
            "dp_degree":       _dp,
            "total_gpus":      1024,
            "mfu_percent":     round(_mfu_pct_3d, 2),
            "act1_prediction": act1_pred.value if act1_pred.value else "no_selection",
            "act1_correct":    act1_pred.value == "B",
            "act1_reflect":    act1_reflect.value if act1_reflect.value else "no_selection",
            "act2_result":     round(_mfu_pct_3d, 2),
            "act2_decision":   act2_pred.value if act2_pred.value else "no_selection",
            "constraint_hit":  _oom or _tp_crosses_node,
        "student_justification": str(decision_input.value),
            "memory_feasible": not _oom,
        },
    )

    # ── Determine overall performance tier ──────────────────────────────────────
    _act1_ok  = act1_pred.value == "B"
    _act2_ok  = act2_pred.value == "B"
    _mfu_ok   = _mfu_pct_3d >= 40.0 and not _oom and _bubble_pct <= 10.0

    _tier = "Optimal" if (_act1_ok and _act2_ok and _mfu_ok) else ("Partial" if (_act1_ok or _act2_ok) else "Developing")
    _tier_color = COLORS["GreenLine"] if _tier == "Optimal" else (COLORS["OrangeLine"] if _tier == "Partial" else COLORS["TextMuted"])

    # ── HUD Footer ───────────────────────────────────────────────────────────────
    decision_ui
    _hud = mo.Html(f"""
    <div class="lab-hud">
        <div>
            <span class="hud-label">LAB</span>&nbsp;
            <span class="hud-value">Vol2 · Lab 05</span>
        </div>
        <div>
            <span class="hud-label">CHAPTER</span>&nbsp;
            <span class="hud-value">v2_05 · Distributed Training</span>
        </div>
        <div>
            <span class="hud-label">CONTEXT</span>&nbsp;
            <span class="hud-value">{_context.upper()}</span>
        </div>
        <div>
            <span class="hud-label">CONFIG</span>&nbsp;
            <span class="hud-value">TP={tp_degree.value} &times; PP={pp_degree.value} &times; DP={_dp}</span>
        </div>
        <div>
            <span class="hud-label">MFU</span>&nbsp;
            <span style="color:{COLORS['GreenLine'] if _mfu_pct_3d >= 40 else COLORS['OrangeLine']}; font-family:var(--font-mono); font-size:0.8rem;">
                {_mfu_pct_3d:.1f}%
            </span>
        </div>
        <div>
            <span class="hud-label">ACT I</span>&nbsp;
            <span class="{'hud-active' if _act1_ok else 'hud-none'}">&nbsp;{"CORRECT" if _act1_ok else "REVIEW"}</span>
        </div>
        <div>
            <span class="hud-label">ACT II</span>&nbsp;
            <span class="{'hud-active' if _act2_ok else 'hud-none'}">&nbsp;{"CORRECT" if _act2_ok else "REVIEW"}</span>
        </div>
        <div>
            <span class="hud-label">TIER</span>&nbsp;
            <span style="color:{_tier_color}; font-family:var(--font-mono); font-size:0.8rem;">{_tier.upper()}</span>
        </div>
        <div>
            <span class="hud-label">OOM</span>&nbsp;
            <span class="{'hud-none' if _oom else 'hud-active'}">&nbsp;{"YES" if _oom else "NO"}</span>
        </div>
    </div>
    """)
    _hud
    return


# ─── CONNECTIONS ──────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Connections

    **Textbook:** This lab explores the core concepts of
    @sec-distributed-training-systems — the Iron Law of Scale, the
    Communication-Computation Ratio, and the 3D Parallelism Cube
    (@fig-3d-parallelism-cube).

    **Prior Labs:** Lab 03 (Network Fabrics) established the physical bandwidth
    limits — NVLink vs InfiniBand — that constrain TP degree here.
    Lab 04 (Data Storage) established the I/O pipeline that feeds each DP replica.

    **Next Lab:** Vol2 Lab 06 (Collective Communications) examines the
    Ring-AllReduce and Tree-AllReduce algorithms in detail, quantifying
    why Ring-AllReduce achieves near-linear scaling efficiency while
    parameter-server approaches hit coordination bottlenecks.
    """)
    return


# ─── KEY TAKEAWAYS ─────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    1. **The Parallelism Paradox is a bandwidth ratio, not a software bug.**
       AllReduce volume saturates at approximately 2× gradient size regardless of GPU count,
       but the transition from NVLink (900 GB/s, within node) to InfiniBand (400 GB/s, cross-node)
       creates a step-change in communication time that drives the MFU cliff observed at
       8→64 GPUs. MFU falls because T_allreduce grows while T_compute stays constant.

    2. **The 3D parallelism constraint TP ≤ GPUS_PER_NODE is physics, not convention.**
       Tensor parallelism performs AllReduce after every transformer layer. At 96 layers,
       the per-layer AllReduce penalty accumulates into a step-time budget that InfiniBand
       cannot satisfy. Confine TP within a single DGX node to keep every layer's
       synchronization on NVLink. Then PP crosses nodes over InfiniBand — but PP
       AllReduce happens only once per pipeline stage, not once per layer.
    """)
    return


if __name__ == "__main__":
    app.run()
