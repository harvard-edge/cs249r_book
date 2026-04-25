import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-05: THE PARALLELISM PUZZLE
#
# Chapter: Distributed Training Systems (@sec-distributed-training-systems)
# Core Invariant: Each parallelism strategy (DP, ZeRO, PP) solves one
#                 constraint while creating another. The Conservation of
#                 Overhead governs every choice. 3D parallelism maps
#                 strategies to the bandwidth hierarchy.
#
# 3-Part + Synthesis Structure (35-40 minutes):
#   Part A — The Communication Wall (12-15 min)
#             Pure DP on 256 GPUs with IB NDR: 175B model achieves only
#             ~50-55% efficiency because AllReduce consumes half the step.
#
#   Part B — The ZeRO Memory Trap (8-10 min)
#             ZeRO-3 alone cannot train 175B — activations push OOM.
#
#   Part C — 3D Parallelism Design Challenge (15-18 min)
#             Design the full 3D config for 175B on 256 H100s. Pipeline
#             bubbles impose a minimum microbatch count. 3D parallelism is
#             the only viable approach.
#
# Deployment Contexts:
#   DP:         Data Parallel — replicate model, sync gradients via AllReduce
#   3D Parallel: TP x PP x DP — within-node TP (NVLink), cross-node PP/DP
#
# Hardware Constants (sourced from mlsysim registry):
#   H100_TFLOPS_FP16  = 989     TFLOPS (NVIDIA H100 SXM5 spec)
#   H100_BW_GBS       = 3350    GB/s HBM3e
#   H100_RAM_GB       = 80      GB HBM3e
#   NVLINK4_BW_GBS    = 900     GB/s NVLink 4 (DGX H100)
#   IB_NDR_BW_GBS     = 50      GB/s InfiniBand NDR per port
#   GPUS_PER_NODE     = 8       Standard DGX H100 node
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 0: SETUP ─────────────────────────────────────────────────────────────
@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import numpy as np

    # WASM bootstrap
    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install(
            "../../wheels/mlsysim-0.1.1-py3-none-any.whl", keep_going=False
        )
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    # plotly must be imported AFTER micropip install, since it's installed at runtime on WASM
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog
    import mlsysim
    from mlsysim import Hardware
    from mlsysim.core.defaults import INFINIBAND_NDR_BW_GBS
    from mlsysim.core.constants import (
        A100_FLOPS_FP16_TENSOR, T4_FLOPS_FP16_TENSOR,
    )

    # ── Hardware registry ─────────────────────────────────────────────────
    H100 = Hardware.Cloud.H100
    A100 = Hardware.Cloud.A100

    # ── Hardware constants (from registry) ──────────────────────────────
    H100_TFLOPS_FP16  = H100.compute.peak_flops.m_as("TFLOPs/s")
    H100_BW_GBS       = H100.memory.bandwidth.m_as("GB/s")
    H100_RAM_GB       = H100.memory.capacity.m_as("GB")
    A100_RAM_GB       = A100.memory.capacity.m_as("GB")
    A100_TFLOPS_FP16  = A100_FLOPS_FP16_TENSOR.m_as("TFLOPs/s")
    T4_TFLOPS_FP16    = T4_FLOPS_FP16_TENSOR.m_as("TFLOPs/s")
    # Interconnect specs (from defaults — not in per-device registry)
    NVLINK4_BW_GBS    = 900.0     # GB/s NVLink 4 (DGX H100)
    IB_NDR_BW_GBS     = INFINIBAND_NDR_BW_GBS  # 50 GB/s, alias for consistent naming
    IB_HDR_BW_GBS     = 25.0      # GB/s InfiniBand HDR per port
    ETH_100G_BW_GBS   = 12.5      # GB/s 100GbE
    GPUS_PER_NODE     = 8

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        COLORS, LAB_CSS, apply_plotly_theme, go, ledger, math, mo, np,
        make_subplots, mlsysim, DecisionLog, Hardware, H100, A100,
        H100_TFLOPS_FP16, H100_BW_GBS, H100_RAM_GB, A100_RAM_GB,
        A100_TFLOPS_FP16, T4_TFLOPS_FP16,
        NVLINK4_BW_GBS, IB_NDR_BW_GBS, IB_HDR_BW_GBS, ETH_100G_BW_GBS,
        GPUS_PER_NODE,
    )


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
                    The Parallelism Puzzle
                </div>
                <div style="font-size: 0.95rem; color: #94a3b8; max-width: 600px; line-height: 1.6;">
                    Each parallelism strategy solves one constraint while creating another.
                    Data parallelism hits a communication wall. ZeRO trades memory for
                    communication. Pipeline parallelism creates bubbles. Only 3D parallelism,
                    mapped to the bandwidth hierarchy, can train frontier models.
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;">
                <span class="badge badge-info">Communication Wall</span>
                <span class="badge badge-info">ZeRO Memory Sharding</span>
                <span class="badge badge-info">Pipeline Bubbles</span>
                <span class="badge badge-info">3D Parallel: TP &times; PP &times; DP</span>
                <span class="badge badge-warn">35&ndash;40 minutes &middot; 3 Parts + Synthesis</span>
            </div>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap;">
            <div style="background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.35);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_dp}; font-weight: 700;">Context A &mdash; Data Parallel</span>
                <span style="color: #94a3b8;"> &mdash; 175B model &middot; 1&ndash;512 GPUs &middot; AllReduce via IB NDR</span>
            </div>
            <div style="background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.35);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_3d}; font-weight: 700;">Context B &mdash; 3D Parallel</span>
                <span style="color: #94a3b8;"> &mdash; 175B model &middot; 256 H100s &middot; TP&times;PP&times;DP design</span>
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
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Quantify the communication wall:</strong> calculate Ring-AllReduce time for a 175B model and explain why DP efficiency collapses to ~50% at 256 GPUs on IB NDR.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose why ZeRO-3 alone cannot train 175B:</strong> compute per-GPU memory with and without activation sharding to identify the OOM boundary.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a 3D parallelism configuration:</strong> select TP, PP, and DP degrees that satisfy memory, bubble, and bandwidth constraints simultaneously.</div>
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
                    Ring-AllReduce bandwidth model from the Communication chapter &middot;
                    Data parallelism and scaling efficiency from the Distributed Training chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35-40 min</strong><br/>
                    Part A: ~12 min &middot; Part B: ~10 min &middot; Part C: ~15 min
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
                "If data parallelism, ZeRO, and pipeline parallelism each solve part of the scaling
                puzzle, why does every frontier model require all three simultaneously &mdash; and what
                determines the correct ratio?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **The Distributed Training chapter** -- Data parallelism, the Communication-Computation Ratio,
      and the Iron Law of Scale: `T_step(N) = T_compute/N + T_comm(N) - T_overlap`
    - **The Communication chapter** -- Ring-AllReduce bandwidth formula, gradient bucketing
    - The ZeRO section -- Memory sharding stages (ZeRO-1/2/3), activation memory
    - The Pipeline Parallelism section -- Microbatching, bubble fraction `B = (PP-1)/(PP*m)`
    - The 3D Parallelism section -- TP within NVLink, PP across IB, DP for throughput
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: WIDGET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 4: PART A WIDGETS ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    partA_prediction = mo.ui.radio(
        options={
            "A) ~90% -- InfiniBand is fast enough to handle the gradients": "A",
            "B) ~70% -- moderate communication overhead": "B",
            "C) ~50-55% -- communication is nearly half of step time": "C",
            "D) ~25% -- communication dominates": "D",
        },
        label="You train a 175B model with pure data parallelism on 256 GPUs with InfiniBand NDR. What scaling efficiency do you achieve?",
    )
    a1_model_select = mo.ui.dropdown(
        options={"1B": 1.0, "7B": 7.0, "70B": 70.0, "175B": 175.0},
        value="175B",
        label="Model size (parameters)",
    )
    a1_gpu_slider = mo.ui.slider(
        start=1, stop=512, value=256, step=1,
        label="Number of GPUs (DP degree)",
    )
    a1_interconnect = mo.ui.dropdown(
        options={
            "IB NDR (50 GB/s)": 50.0,
            "IB HDR (25 GB/s)": 25.0,
            "100GbE (12.5 GB/s)": 12.5,
        },
        value="IB NDR (50 GB/s)",
        label="Interconnect",
    )
    partA_reflection = mo.ui.radio(
        options={
            "A) Use faster GPUs -- reduce compute time so communication is a smaller fraction": "A",
            "B) Shard the model across GPUs so each holds only a fraction -- this is ZeRO or model parallelism": "B",
            "C) Use gradient compression to reduce AllReduce volume by 4-8x": "C",
            "D) Accept the wall -- 55% efficiency is good enough for production": "D",
        },
        label="What is the most effective strategy to overcome the DP communication wall for frontier models?",
    )
    return (a1_gpu_slider, a1_interconnect, a1_model_select, partA_prediction, partA_reflection)


# ─── CELL 5: PART B WIDGETS ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partA_prediction):
    partB_prediction = mo.ui.radio(
        options={
            "A) TP=1, PP=1, DP=256 -- Pure DP: maximize data parallelism": "A",
            "B) TP=8, PP=4, DP=8 -- Full 3D parallelism mapped to bandwidth hierarchy": "B",
            "C) TP=8, PP=1, DP=32 -- TP within node + DP only": "C",
            "D) TP=8, PP=32, DP=1 -- Aggressive pipeline, no DP": "D",
        },
        label="Which 3D configuration (TP x PP x DP = 256) best trains a 175B model on 256 H100s?",
    )
    return (partB_prediction,)


# ─── CELL 6: PART C WIDGETS ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partB_prediction):
    a2_tp = mo.ui.slider(start=1, stop=16, value=8, step=1, label="Tensor Parallelism (TP)")
    a2_pp = mo.ui.slider(start=1, stop=32, value=4, step=1, label="Pipeline Parallelism (PP)")
    a2_microbatches = mo.ui.slider(start=1, stop=64, value=16, step=1, label="Microbatches per flush (m)")
    a2_zero_stage = mo.ui.dropdown(
        options={"ZeRO-0 (no sharding)": 0, "ZeRO-1 (optimizer)": 1, "ZeRO-2 (+gradients)": 2, "ZeRO-3 (+parameters)": 3},
        value="ZeRO-3 (+parameters)",
        label="ZeRO Stage",
    )
    partC_reflection = mo.ui.radio(
        options={
            "A) TP handles inter-layer communication, PP handles intra-layer, DP handles gradient sync": "A",
            "B) TP maps to NVLink (highest BW), PP maps to IB (moderate BW), DP AllReduce uses remaining BW": "B",
            "C) The mapping is arbitrary -- any assignment of TP/PP/DP to the hierarchy works equally well": "C",
            "D) TP and PP should both use NVLink; DP should use Ethernet for cost savings": "D",
        },
        label="Why must TP map to NVLink, PP to InfiniBand, and DP to the remaining bandwidth?",
    )
    return (a2_microbatches, a2_pp, a2_tp, a2_zero_stage, partC_reflection)


# ─── CELL 6b: PART D WIDGETS ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partC_reflection,
      H100_TFLOPS_FP16, A100_TFLOPS_FP16, T4_TFLOPS_FP16,
      IB_NDR_BW_GBS, IB_HDR_BW_GBS, ETH_100G_BW_GBS):
    partD_prediction = mo.ui.radio(
        options={
            "A) A100 80GB -- same HBM so same result": "A",
            "B) A100 achieves ~35% efficiency -- slower interconnect (IB HDR 25 GB/s) doubles the communication wall": "B",
            "C) A100 achieves ~70% -- slower compute means comm/compute ratio improves": "C",
            "D) A100 cannot train 175B at all -- insufficient memory": "D",
        },
        label="You switch from H100 + IB NDR (50 GB/s) to A100 + IB HDR (25 GB/s) for 175B DP training on 256 GPUs. What happens to scaling efficiency?",
    )
    d1_hw_tier = mo.ui.dropdown(
        options={
            f"H100 + IB NDR ({IB_NDR_BW_GBS} GB/s, {H100_TFLOPS_FP16:.0f} TFLOPS)": ("H100", H100_TFLOPS_FP16, IB_NDR_BW_GBS),
            f"A100 + IB HDR ({IB_HDR_BW_GBS} GB/s, {A100_TFLOPS_FP16:.0f} TFLOPS)": ("A100", A100_TFLOPS_FP16, IB_HDR_BW_GBS),
            f"T4 + 100GbE ({ETH_100G_BW_GBS} GB/s, {T4_TFLOPS_FP16:.0f} TFLOPS)": ("T4", T4_TFLOPS_FP16, ETH_100G_BW_GBS),
        },
        value=f"H100 + IB NDR ({IB_NDR_BW_GBS} GB/s, {H100_TFLOPS_FP16:.0f} TFLOPS)",
        label="Hardware tier",
    )
    d1_model_size = mo.ui.dropdown(
        options={"7B": 7.0, "70B": 70.0, "175B": 175.0},
        value="175B",
        label="Model size",
    )
    d1_gpu_count = mo.ui.slider(start=8, stop=512, value=256, step=8, label="GPU count")
    partD_reflection = mo.ui.radio(
        options={
            "A) Always use the fastest GPUs regardless of model size": "A",
            "B) Match hardware tier to model scale -- small models on T4, large models require H100": "B",
            "C) Use A100 for everything -- best price/performance": "C",
            "D) Interconnect does not matter -- only GPU compute speed determines efficiency": "D",
        },
        label="What principle governs hardware tier selection for distributed training?",
    )
    return (d1_gpu_count, d1_hw_tier, d1_model_size, partD_prediction, partD_reflection)


# ─── CELL 7: SYNTHESIS WIDGETS ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(DecisionLog, mo, partD_reflection):
    synth_decision_input, synth_decision_ui = DecisionLog(
        placeholder="Based on what I learned in this lab, the most important insight about "
                    "distributed training parallelism is..."
    )
    return (synth_decision_input, synth_decision_ui)



# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: SINGLE TABS CELL
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 8: ALL PARTS + TABS COMPOSITION ─────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, apply_plotly_theme, go, math,
    mo, np, H100_TFLOPS_FP16, H100_RAM_GB,
    A100_TFLOPS_FP16, T4_TFLOPS_FP16,
    IB_NDR_BW_GBS, IB_HDR_BW_GBS, ETH_100G_BW_GBS,
    NVLINK4_BW_GBS, GPUS_PER_NODE, synth_decision_input, synth_decision_ui,
    a1_gpu_slider, a1_interconnect, a1_model_select, a2_microbatches,
    a2_pp, a2_tp, a2_zero_stage, d1_gpu_count,
    d1_hw_tier, d1_model_size, partA_prediction, partA_reflection,
    partB_prediction, partC_reflection, partD_prediction, partD_reflection,
):
    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER -- The Communication Wall
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['BlueLine']}; background: {COLORS['BlueLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; Training Infrastructure Lead
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "We are scaling a 175B parameter model with pure data parallelism on 256 H100s
                connected via InfiniBand NDR (50 GB/s per port). InfiniBand is the fastest
                interconnect money can buy. I expect scaling efficiency above 90%. Am I wrong?"
            </div>
        </div>
        """))

        # Concept framing
        items.append(mo.md("""
    The stakeholder's expectation sounds reasonable: InfiniBand NDR is the fastest
    datacenter interconnect. But the critical variable is not the absolute bandwidth
    -- it is the **ratio** of communication time to compute time.

    For a 175B model in FP16, the gradient tensor is ~350 GB. Ring-AllReduce transfers
    approximately `2 x (N-1)/N x gradient_bytes` per step. At N=256, this saturates
    at ~700 GB of data traversing 50 GB/s InfiniBand -- a transfer that takes ~14 seconds.

    If the compute step (forward + backward) on each GPU takes ~10 seconds, the
    communication/compute ratio is 1.4. Efficiency = 1 / (1 + ratio) = ~42%.
    Even with 50% overlap via gradient bucketing, efficiency reaches only ~55%.

    Before seeing the numbers, commit to your prediction.
        """))

        # Prediction lock
        items.append(mo.md("### Your Prediction"))
        items.append(partA_prediction)
        items.append(mo.stop(
            partA_prediction.value is None,
            mo.callout(
                mo.md("Select your prediction above to unlock the Part A instruments."),
                kind="warn",
            ),
        ))

        # Instruments header
        items.append(mo.md("""
    ### Data Parallel Scaling Explorer

    Adjust GPU count and model size to see how AllReduce communication time
    compares to compute time -- and how the ratio determines efficiency.
        """))

        # Controls
        items.append(mo.hstack([a1_model_select, a1_gpu_slider, a1_interconnect], justify="center", gap=2))

        # ── Physics: Ring-AllReduce ────────────────────────────────────────
        _params_b   = a1_model_select.value
        _n_gpus     = a1_gpu_slider.value
        _bw_gbs     = a1_interconnect.value

        _grad_bytes     = _params_b * 1e9 * 2                    # FP16 gradients
        _grad_gb        = _grad_bytes / 1e9
        _ring_factor    = 2.0 * (_n_gpus - 1) / max(_n_gpus, 1)  # saturates at ~2
        _allreduce_gb   = _ring_factor * _grad_gb
        _allreduce_s    = _allreduce_gb / _bw_gbs if _bw_gbs > 0 else 999

        # Compute time: 6 * params * tokens_per_batch / (peak_flops * MFU_ref)
        _flops_per_step = 6.0 * _params_b * 1e9 * 2048         # 6PD with seq_len=2048, micro_batch=1
        _compute_s      = _flops_per_step / (_n_gpus * H100_TFLOPS_FP16 * 1e12 * 0.50)

        # Overlap: gradient bucketing hides ~50% of AllReduce
        _overlap_frac   = 0.50
        _effective_comm = _allreduce_s * (1 - _overlap_frac)
        _step_time_s    = _compute_s + _effective_comm
        _efficiency     = _compute_s / _step_time_s if _step_time_s > 0 else 0
        _efficiency_pct = _efficiency * 100

        # ── Sweep: efficiency vs GPU count ────────────────────────────────
        _gpu_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        _eff_curve = []
        for _g in _gpu_range:
            _c_s = _flops_per_step / (_g * H100_TFLOPS_FP16 * 1e12 * 0.50)
            _ar_s = (2.0 * (_g - 1) / max(_g, 1) * _grad_gb) / _bw_gbs
            _st = _c_s + _ar_s * (1 - _overlap_frac)
            _eff_curve.append((_c_s / _st * 100) if _st > 0 else 100)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_gpu_range, y=_eff_curve,
            mode="lines+markers",
            line=dict(color=COLORS["BlueLine"], width=2.5),
            marker=dict(size=7),
            name="Scaling efficiency",
            hovertemplate="<b>%{x} GPUs</b><br>Efficiency: %{y:.1f}%<extra></extra>",
        ))
        _fig.add_trace(go.Scatter(
            x=[_n_gpus], y=[_efficiency_pct],
            mode="markers",
            marker=dict(size=14, color=COLORS["RedLine"], symbol="diamond",
                        line=dict(color="white", width=2)),
            name="Current config",
        ))
        _fig.add_hline(y=50, line=dict(color=COLORS["OrangeLine"], width=1.5, dash="dash"),
                       annotation_text="50% efficiency", annotation_position="bottom right")
        _fig.update_layout(
            height=320,
            xaxis=dict(title="GPU Count (DP degree)", type="log",
                       tickvals=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
            yaxis=dict(title="Scaling Efficiency (%)", range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=50, r=20),
        )
        apply_plotly_theme(_fig)

        # Color coding
        _eff_color = COLORS["GreenLine"] if _efficiency_pct >= 70 else (COLORS["OrangeLine"] if _efficiency_pct >= 40 else COLORS["RedLine"])
        _comm_color = COLORS["GreenLine"] if _effective_comm < _compute_s * 0.3 else (COLORS["OrangeLine"] if _effective_comm < _compute_s else COLORS["RedLine"])

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;
                        font-family:sans-serif;">
                Physics &mdash; Ring-AllReduce Communication Model
            </div>
            <div>Gradient size = {_params_b}B &times; 2 bytes (FP16) = <strong>{_grad_gb:.1f} GB</strong></div>
            <div>Ring-AllReduce volume = 2 &times; ({_n_gpus}-1)/{_n_gpus} &times; {_grad_gb:.1f} GB = <strong>{_allreduce_gb:.1f} GB</strong></div>
            <div>AllReduce time = {_allreduce_gb:.1f} GB / {_bw_gbs} GB/s = <strong>{_allreduce_s:.2f}s</strong> (raw)</div>
            <div>Compute time per step = <strong>{_compute_s:.2f}s</strong> (at 50% MFU reference)</div>
            <div>Effective comm (50% overlap) = <strong>{_effective_comm:.2f}s</strong></div>
            <div>Efficiency = T_compute / (T_compute + T_comm_eff) = {_compute_s:.2f} / {_step_time_s:.2f} = <strong style="color:{_eff_color};">{_efficiency_pct:.1f}%</strong></div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">Efficiency</div>
                <div style="font-size:2.2rem; font-weight:800; color:{_eff_color};
                            font-family:monospace;">{_efficiency_pct:.1f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">scaling efficiency</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">AllReduce</div>
                <div style="font-size:2.2rem; font-weight:800; color:{_comm_color};
                            font-family:monospace;">{_allreduce_s:.1f}s</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">per step (raw)</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">Compute</div>
                <div style="font-size:2.2rem; font-weight:800; color:{COLORS['BlueLine']};
                            font-family:monospace;">{_compute_s:.1f}s</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">per step</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.06em;">Comm/Compute</div>
                <div style="font-size:2.2rem; font-weight:800; color:{_comm_color};
                            font-family:monospace;">{_effective_comm / _compute_s:.2f}x</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']}; margin-top:2px;">ratio (lower = better)</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # Prediction reveal
        _correct = partA_prediction.value == "C"
        if _correct:
            items.append(mo.callout(mo.md(
                "**Correct.** At 256 GPUs on IB NDR, the 175B model's gradient AllReduce "
                "transfers ~700 GB per step. Even at 50 GB/s, this takes ~14 seconds raw. "
                "Compute per GPU (~10s at this scale) cannot hide such a large transfer. "
                "With 50% overlap, efficiency reaches ~55%. The communication wall is real "
                "even on the fastest interconnect."
            ), kind="success"))
        elif partA_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Too optimistic.** InfiniBand NDR is fast (50 GB/s), but 175B parameters "
                "produce 350 GB of FP16 gradients. Ring-AllReduce transfers ~700 GB per step. "
                "At 50 GB/s, that is ~14 seconds -- comparable to the compute time itself. "
                "90% efficiency would require communication to be <11% of compute time."
            ), kind="warn"))
        elif partA_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Close, but still optimistic.** 70% efficiency requires comm/compute ratio "
                "below 0.43. With 700 GB of AllReduce traffic and only 50% overlap, the "
                "effective communication is ~7 seconds against ~10 seconds of compute. "
                "The actual ratio is ~0.7, yielding ~55% efficiency."
            ), kind="warn"))
        elif partA_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Too pessimistic.** 25% would mean communication is 3x compute time. "
                "InfiniBand NDR is genuinely fast -- the issue is that gradient size for "
                "175B models is so large that even fast interconnects are overwhelmed, but "
                "not to the point of 3x domination. With overlap, efficiency stabilizes ~55%."
            ), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "The governing equations -- Ring-AllReduce and DP efficiency": mo.md("""
        **Ring-AllReduce Transfer Volume**

        ```
        V_allreduce = 2 * (N-1)/N * gradient_bytes
        T_allreduce = V_allreduce / BW_interconnect
        ```

        - For large N, the factor 2*(N-1)/N approaches 2
        - 175B FP16 gradients = 350 GB; AllReduce volume = ~700 GB
        - At IB NDR (50 GB/s): T_allreduce = 14.0 seconds

        **DP Scaling Efficiency**

        ```
        eta = T_compute / (T_compute + T_comm_effective)
        T_comm_effective = T_allreduce * (1 - overlap_fraction)
        ```

        - Gradient bucketing achieves ~50% overlap for large models
        - At 256 GPUs: eta = 10s / (10s + 7s) = 58.8%
        - The communication wall appears when T_comm approaches T_compute
            """)
        }))

        # Reflection
        items.append(mo.md("### Reflection"))
        items.append(partA_reflection)
        if partA_reflection.value is not None:
            if partA_reflection.value == "B":
                items.append(mo.callout(mo.md(
                    "**Correct.** Model parallelism (TP, PP) and ZeRO sharding reduce the "
                    "gradient AllReduce volume by distributing the model itself. With TP=8 "
                    "within a node, each TP group handles 175B/8 parameters, and the DP "
                    "AllReduce is over a smaller DP degree, dramatically reducing the "
                    "communication wall. This is why 3D parallelism exists."
                ), kind="success"))
            elif partA_reflection.value == "A":
                items.append(mo.callout(mo.md(
                    "**Counterproductive.** Faster GPUs reduce T_compute, but the comm/compute "
                    "ratio actually *worsens* because T_allreduce stays the same. Each GPU "
                    "generation doubles compute TFLOPS but interconnect bandwidth grows slower. "
                    "The communication wall gets *worse* with faster GPUs, not better."
                ), kind="warn"))
            elif partA_reflection.value == "C":
                items.append(mo.callout(mo.md(
                    "**Partially effective but not the primary solution.** INT8 gradient compression "
                    "can halve AllReduce volume, but it introduces convergence risk for sensitive "
                    "training runs. The standard approach is to restructure the parallelism itself "
                    "(TP/PP) rather than compromise gradient fidelity."
                ), kind="warn"))
            elif partA_reflection.value == "D":
                items.append(mo.callout(mo.md(
                    "**Not viable at frontier scale.** At 55% efficiency, you are paying for 256 "
                    "H100s but getting the throughput of ~140. At $3/GPU-hour, the wasted compute "
                    "costs ~$350/hour. Over a 90-day training run, that is $756,000 in idle GPUs."
                ), kind="warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER -- The ZeRO Memory Trap
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['Cloud']}; background: {COLORS['BlueLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['Cloud']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; MLOps Architect
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "My team says ZeRO-3 can shard everything across 64 A100 GPUs (80 GB each).
                The static memory math works: 175B x 14 bytes / 64 = 38 GB per GPU. That
                fits in 80 GB HBM. So ZeRO-3 alone should handle this model. Right?"
            </div>
        </div>
        """))

        # Concept framing
        items.append(mo.callout(mo.md(
            "**Systems Bridge:** ZeRO (Zero Redundancy Optimizer) shards optimizer state across GPUs — "
            "analogous to how a distributed database shards table partitions across nodes. ZeRO Stage 1 "
            "shards optimizer state, Stage 2 adds gradient sharding, Stage 3 adds parameter sharding. "
            "Each stage trades communication for memory."
        ), kind="info"))

        items.append(mo.md("""
    The stakeholder computed only **static memory**: parameters + gradients + optimizer states.
    ZeRO-3 shards all three across workers, reducing per-GPU static memory to ~38 GB.

    But **activation memory is NOT sharded** by ZeRO. Each GPU stores activations for its
    own micro-batch during the forward pass, needed for the backward pass. For a 175B
    Transformer at seq_len=2048 and batch_size=1, activations consume ~50 GB per GPU.

    Total per-GPU memory: 38 GB (ZeRO-3 static) + 50 GB (activations) = **88 GB > 80 GB HBM**.

    This is the OOM trap: ZeRO-3 is necessary but not sufficient for frontier models.
    Pipeline parallelism (PP) reduces per-GPU layers, reducing activation memory.
    Tensor parallelism (TP) shards each layer's activations across the TP group.
    Only the combination (3D parallelism) makes training feasible.
        """))

        # Prediction lock
        items.append(mo.md("### Your Prediction"))
        items.append(partB_prediction)
        items.append(mo.stop(
            partB_prediction.value is None,
            mo.callout(
                mo.md("Select your configuration prediction above to unlock the Part B instruments."),
                kind="warn",
            ),
        ))

        # Prediction reveal
        _correct = partB_prediction.value == "B"
        if _correct:
            items.append(mo.callout(mo.md(
                "**Correct.** TP=8 maps to one DGX node (8 GPUs on NVLink at 900 GB/s), keeping "
                "the per-layer AllReduce fast. PP=4 assigns 96/4=24 layers per stage, cutting "
                "activation memory by 4x. With m=16 microbatches, bubble = 3/64 = 4.7%. "
                "DP=8 replicates across 8 groups for throughput. This matches Megatron-LM practice."
            ), kind="success"))
        elif partB_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Pure DP cannot train 175B.** Each of 256 GPUs must hold the full model "
                "(175B x 2 bytes = 350 GB in FP16 weights alone), which exceeds 80 GB HBM "
                "by 4.4x. Even ZeRO-3 sharding across 256 GPUs leaves 38 GB static + 50 GB "
                "activations = 88 GB > 80 GB. TP or PP must reduce per-GPU model size."
            ), kind="warn"))
        elif partB_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Memory infeasible.** Without PP, each GPU holds all 96 layers' activations. "
                "TP=8 reduces weights to 175B/8 = 21.9B per GPU, but activation memory "
                "remains ~50/8 = 6.25 GB (TP partitions activations). Static memory: "
                "21.9B x 14 bytes / 32 (ZeRO-3) = 9.6 GB. Total ~16 GB -- feasible, "
                "but DP=32 means AllReduce of 21.9B x 2 bytes across 32 nodes, still a "
                "significant communication wall. PP would reduce this further."
            ), kind="warn"))
        elif partB_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**DP=1 wastes the cluster.** With DP=1, there is no data parallelism for "
                "throughput scaling. PP=32 requires m >> 32 microbatches to keep bubble "
                "fraction below 10%, implying an enormous global batch size. This is "
                "an over-pipelined design that wastes most of the cluster on bubbles."
            ), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Governing equations -- ZeRO memory and the OOM boundary": mo.md("""
        **ZeRO-3 Per-GPU Memory**

        ```
        Static memory = (P / (TP * PP)) * (2 + 2/DP + 12/DP) bytes
                       = params_per_gpu * (2 + 14/DP) bytes
        Activation memory = base_activation / (TP * PP)
        Total = static + activation
        ```

        For 175B, TP=1, PP=1, DP=64, ZeRO-3:
        - params_per_gpu = 175B
        - Static = 175B * (2 + 14/64) = 175B * 2.22 = 38.3 GB
        - Activations = 50 GB (NOT sharded by ZeRO)
        - Total = 88.3 GB > 80 GB HBM -- **OOM**

        The fix: TP and PP reduce activations per GPU. With TP=8, PP=4:
        - Activations = 50 / (8*4) = 1.56 GB
        - Static = 5.47B * (2 + 14/8) = 20.5 GB
        - Total = ~22 GB -- fits comfortably
            """)
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER -- 3D Parallelism Design Challenge
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        items.append(mo.md("""
    ### 3D Parallelism Configurator

    Adjust TP and PP degrees below. DP is computed automatically from `TP x PP x DP = 256`.
    The configurator enforces per-GPU memory and pipeline bubble constraints.
        """))

        # Controls
        items.append(mo.hstack([
            mo.vstack([a2_tp, a2_pp]),
            mo.vstack([a2_microbatches, a2_zero_stage]),
        ], justify="center", gap=2))

        # ── Model constants: 175B ──────────────────────────────────────────
        _PARAMS_B        = 175.0
        _LAYERS          = 96
        _TOTAL_GPUS      = 256
        _BYTES_WEIGHT    = 2          # FP16
        _BYTES_OPTIMIZER = 12         # FP32 m1+m2+master = 4+4+4 = 12 bytes/param
        _BYTES_GRADIENT  = 2          # FP16
        _ACTIVATION_BASE = 50.0       # GB base activation for 175B at seq=2048, batch=1

        _tp = a2_tp.value
        _pp = a2_pp.value
        _m  = a2_microbatches.value
        _zero = a2_zero_stage.value

        # ── DP degree ──────────────────────────────────────────────────────
        _tp_pp = _tp * _pp
        _dp = _TOTAL_GPUS // _tp_pp if _tp_pp <= _TOTAL_GPUS and _TOTAL_GPUS % _tp_pp == 0 else 0
        _config_valid = _dp > 0

        # ── Memory analysis per GPU ────────────────────────────────────────
        _params_per_gpu_b = _PARAMS_B / (_tp * _pp)

        # ZeRO sharding of static memory across DP workers
        _dp_shard = max(_dp, 1)
        if _zero == 0:
            _weight_gb = _params_per_gpu_b * _BYTES_WEIGHT
            _grad_gb = _params_per_gpu_b * _BYTES_GRADIENT
            _optim_gb = _params_per_gpu_b * _BYTES_OPTIMIZER
        elif _zero == 1:
            _weight_gb = _params_per_gpu_b * _BYTES_WEIGHT
            _grad_gb = _params_per_gpu_b * _BYTES_GRADIENT
            _optim_gb = _params_per_gpu_b * _BYTES_OPTIMIZER / _dp_shard
        elif _zero == 2:
            _weight_gb = _params_per_gpu_b * _BYTES_WEIGHT
            _grad_gb = _params_per_gpu_b * _BYTES_GRADIENT / _dp_shard
            _optim_gb = _params_per_gpu_b * _BYTES_OPTIMIZER / _dp_shard
        else:  # ZeRO-3
            _weight_gb = _params_per_gpu_b * _BYTES_WEIGHT / _dp_shard
            _grad_gb = _params_per_gpu_b * _BYTES_GRADIENT / _dp_shard
            _optim_gb = _params_per_gpu_b * _BYTES_OPTIMIZER / _dp_shard

        # Activations: reduced by TP (partitioned across TP group) and PP (fewer layers)
        _act_gb = _ACTIVATION_BASE / (_tp * (_pp if _pp > 1 else 1))
        _total_mem_gb = _weight_gb + _grad_gb + _optim_gb + _act_gb
        _oom = _total_mem_gb > H100_RAM_GB

        # ── Pipeline bubble ────────────────────────────────────────────────
        _bubble_frac = (_pp - 1) / (_pp * _m) if _pp > 1 else 0.0
        _bubble_pct = _bubble_frac * 100

        # ── TP bandwidth penalty ───────────────────────────────────────────
        _tp_crosses_node = _tp > GPUS_PER_NODE
        _tp_penalty_x = 1.0 if not _tp_crosses_node else (NVLINK4_BW_GBS / 50.0)

        # ── Effective MFU ──────────────────────────────────────────────────
        _mfu_base = 0.52
        _tp_eff = 1.0 - (0.04 * math.log2(max(_tp, 1)) * (1 if not _tp_crosses_node else 2.25))
        _pp_eff = 1.0 - _bubble_frac
        _dp_eff = 1.0 - (0.02 * math.log2(max(_dp, 1)))
        _mfu_eff = max(0.0, min(_mfu_base * _tp_eff * _pp_eff * _dp_eff, _mfu_base))
        _mfu_pct = _mfu_eff * 100

        # ── Colors ─────────────────────────────────────────────────────────
        _mem_color = COLORS["RedLine"] if _oom else (COLORS["OrangeLine"] if _total_mem_gb > 60 else COLORS["GreenLine"])
        _bubble_color = COLORS["RedLine"] if _bubble_pct > 10 else (COLORS["OrangeLine"] if _bubble_pct > 5 else COLORS["GreenLine"])
        _mfu_color = COLORS["RedLine"] if _mfu_pct < 25 else (COLORS["OrangeLine"] if _mfu_pct < 40 else COLORS["GreenLine"])
        _cfg_color = COLORS["GreenLine"] if _config_valid else COLORS["RedLine"]

        # ── Failure banners ────────────────────────────────────────────────
        _oom_banner = ""
        if _oom:
            _oom_banner = f"""
            <div style="background:{COLORS['RedLL']}; border:2px solid {COLORS['RedLine']};
                        border-radius:10px; padding:14px 18px; margin:10px 0;">
                <div style="font-size:0.88rem; font-weight:800; color:{COLORS['RedLine']}; margin-bottom:4px;">
                    OOM &mdash; Configuration Infeasible
                </div>
                <div style="font-size:0.85rem; color:#7f1d1d; line-height:1.6;">
                    <strong>Required per GPU: {_total_mem_gb:.1f} GB</strong> &mdash; exceeds H100 limit: {H100_RAM_GB:.0f} GB.<br>
                    Weights: {_weight_gb:.1f} GB | Gradients: {_grad_gb:.1f} GB | Optimizer: {_optim_gb:.1f} GB | Activations: {_act_gb:.1f} GB<br>
                    Increase TP or PP to reduce per-GPU memory, or enable a higher ZeRO stage.
                </div>
            </div>
            """

        _tp_banner = ""
        if _tp_crosses_node:
            _tp_banner = f"""
            <div style="background:{COLORS['OrangeLL']}; border:1px solid {COLORS['OrangeLine']};
                        border-radius:8px; padding:12px 16px; margin:8px 0;">
                <div style="font-size:0.85rem; font-weight:700; color:{COLORS['OrangeLine']};">
                    TP Crosses Node Boundary &mdash; {_tp_penalty_x:.1f}x bandwidth penalty on every layer's AllReduce
                </div>
            </div>
            """

        _cfg_banner = ""
        if not _config_valid:
            _cfg_banner = f"""
            <div style="background:{COLORS['RedLL']}; border:1px solid {COLORS['RedLine']};
                        border-radius:8px; padding:12px 16px; margin:8px 0;">
                <div style="font-size:0.85rem; font-weight:700; color:{COLORS['RedLine']};">
                    Invalid: TP({_tp}) &times; PP({_pp}) = {_tp_pp} does not divide 256 evenly.
                </div>
            </div>
            """

        # ── Stacked memory bar chart ───────────────────────────────────────
        _fig_mem = go.Figure()
        _categories = ["Weights", "Gradients", "Optimizer", "Activations"]
        _values = [_weight_gb, _grad_gb, _optim_gb, _act_gb]
        _bar_colors = [COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"], COLORS["Cloud"]]
        if _oom:
            _bar_colors = [COLORS["RedLine"]] * 4

        for _cat, _val, _clr in zip(_categories, _values, _bar_colors):
            _fig_mem.add_trace(go.Bar(
                x=[_cat], y=[_val], name=_cat,
                marker_color=_clr,
                hovertemplate=f"{_cat}: %{{y:.1f}} GB<extra></extra>",
            ))
        _fig_mem.add_hline(y=H100_RAM_GB, line=dict(color=COLORS["RedLine"], width=2, dash="dash"),
                           annotation_text=f"H100 HBM: {H100_RAM_GB:.0f} GB", annotation_position="top right")
        _fig_mem.update_layout(
            height=280, barmode="stack",
            xaxis=dict(title=""), yaxis=dict(title="Per-GPU Memory (GB)", range=[0, max(_total_mem_gb * 1.2, 90)]),
            showlegend=False, margin=dict(t=30, b=40, l=50, r=20),
        )
        apply_plotly_theme(_fig_mem)

        # ── Bubble chart ───────────────────────────────────────────────────
        _pp_range = list(range(1, 33))
        _fig_bubble = go.Figure()
        for _mval, _label, _clr in [(4, "m=4", COLORS["OrangeLine"]), (8, "m=8", COLORS["BlueLine"]),
                                      (16, "m=16", COLORS["GreenLine"]), (32, "m=32", COLORS["Grey"])]:
            _bvals = [(_p - 1) / (_p * _mval) * 100 for _p in _pp_range]
            _fig_bubble.add_trace(go.Scatter(x=_pp_range, y=_bvals, mode="lines", name=_label,
                                              line=dict(color=_clr, width=2)))
        _fig_bubble.add_trace(go.Scatter(x=[_pp], y=[_bubble_pct], mode="markers", name="Current",
                                          marker=dict(size=14, color=COLORS["RedLine"], symbol="diamond")))
        _fig_bubble.add_hline(y=10, line=dict(color=COLORS["Surface1"], width=1.5, dash="dash"),
                              annotation_text="10% ceiling", annotation_position="top right")
        _fig_bubble.update_layout(
            height=260, xaxis=dict(title="Pipeline Stages (PP)"), yaxis=dict(title="Bubble %", range=[0, 55]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=50, r=20),
        )
        apply_plotly_theme(_fig_bubble)

        # ── Render instruments ─────────────────────────────────────────────
        items.append(mo.Html(f"""
        {_oom_banner}{_tp_banner}{_cfg_banner}
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; 3D Parallel Memory + Bubble Analysis
            </div>
            <div>Config: TP={_tp} &times; PP={_pp} &times; DP={_dp if _config_valid else 'N/A'} {'= 256' if _config_valid else ''} | ZeRO Stage {_zero}</div>
            <div>Params/GPU = 175B / (TP={_tp} &times; PP={_pp}) = <strong>{_params_per_gpu_b:.2f}B</strong></div>
            <div>Per-GPU memory = <strong style="color:{_mem_color};">{_total_mem_gb:.1f} GB</strong> / {H100_RAM_GB:.0f} GB limit</div>
            <div>Bubble = (PP-1)/(PP&times;m) = ({_pp}-1)/({_pp}&times;{_m}) = <strong style="color:{_bubble_color};">{_bubble_pct:.1f}%</strong></div>
            <div>Effective MFU = <strong style="color:{_mfu_color};">{_mfu_pct:.1f}%</strong></div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:150px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Per-GPU Mem</div>
                <div style="font-size:2rem; font-weight:800; color:{_mem_color}; font-family:monospace;">{_total_mem_gb:.0f}GB</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">/ {H100_RAM_GB:.0f} GB</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:150px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Bubble</div>
                <div style="font-size:2rem; font-weight:800; color:{_bubble_color}; font-family:monospace;">{_bubble_pct:.1f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">ceiling: 10%</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:150px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">MFU</div>
                <div style="font-size:2rem; font-weight:800; color:{_mfu_color}; font-family:monospace;">{_mfu_pct:.1f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">3D parallel</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:150px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">DP degree</div>
                <div style="font-size:2rem; font-weight:800; color:{_cfg_color}; font-family:monospace;">{_dp if _config_valid else 'N/A'}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">= 256/(TP&times;PP)</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig_mem))
        items.append(mo.ui.plotly(_fig_bubble))

        # MathPeek
        items.append(mo.accordion({
            "Governing equations -- ZeRO memory, bubble fraction, 3D parallelism": mo.md("""
        **ZeRO-3 Per-GPU Memory**

        ```
        Static memory = (P / (TP * PP)) * (2 + 2/DP + 12/DP) bytes
                       = params_per_gpu * (2 + 14/DP) bytes
        Activation memory = base_activation / (TP * PP)
        Total = static + activation
        ```

        For 175B, TP=8, PP=4, DP=8, ZeRO-3:
        - params_per_gpu = 175B/(8*4) = 5.47B
        - Static = 5.47B * (2 + 14/8) = 5.47B * 3.75 = 20.5 GB
        - Activations = 50 GB / (8*4) = 1.56 GB
        - Total = ~22 GB -- fits in 80 GB HBM

        **Pipeline Bubble Fraction**

        ```
        B = (PP - 1) / (PP * m)
        ```

        At PP=4, m=16: B = 3/64 = 4.7% (under 10% ceiling)

        **Conservation of Overhead**: Each parallelism dimension eliminates one
        bottleneck while introducing another. 3D parallelism is not overhead-free;
        it distributes overhead across the bandwidth hierarchy where each type
        can be absorbed most efficiently.
            """)
        }))

        # Reflection
        items.append(mo.md("### Reflection"))
        items.append(partC_reflection)
        if partC_reflection.value is not None:
            if partC_reflection.value == "B":
                items.append(mo.callout(mo.md(
                    "**Correct.** TP performs AllReduce after every layer (96 times per forward pass). "
                    "This requires the highest bandwidth -- NVLink at 900 GB/s. PP transfers only "
                    "small activation tensors (~200 MB) between pipeline stages -- IB's 50 GB/s "
                    "handles this in milliseconds. DP AllReduce is performed once per step on the "
                    "reduced model shard (after TP+PP partitioning), so the smaller DP degree and "
                    "reduced gradient volume can tolerate the remaining bandwidth."
                ), kind="success"))
            else:
                items.append(mo.callout(mo.md(
                    "**Not quite.** The key insight is that TP has the highest communication frequency "
                    "(once per layer), PP has moderate frequency (once per pipeline stage), and DP has "
                    "the lowest frequency (once per step). The bandwidth hierarchy must match: "
                    "highest frequency -> highest bandwidth. TP -> NVLink, PP -> IB, DP -> remaining."
                ), kind="warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER -- Hardware Tier Comparison
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['OrangeLine']}; background: {COLORS['OrangeLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['OrangeLine']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; VP of Infrastructure
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "Our budget team says we can save 40% by using A100s instead of H100s for
                distributed training. The A100 has the same 80 GB HBM. Since the model fits
                on both, switching should be straightforward. Can you verify the efficiency
                impact before we commit?"
            </div>
        </div>
        """))

        # Concept framing
        items.append(mo.md("""
    The VP is right that both GPUs have 80 GB HBM -- but memory capacity is only
    one constraint. Distributed training efficiency depends on the **ratio** of
    communication time to compute time. Switching hardware changes both:

    - **A100**: 312 TFLOPS FP16 (vs H100's 989) -- 3.2x slower compute
    - **IB HDR**: 25 GB/s (vs NDR's 50 GB/s) -- 2x slower interconnect

    Slower compute *helps* the comm/compute ratio (more time to hide AllReduce).
    Slower interconnect *hurts* it. The net effect depends on the model size.

    For small models (7B), the gradient is only 14 GB -- even 25 GB/s handles it.
    For 175B, the gradient is 350 GB -- halving bandwidth is devastating.
        """))

        # Prediction lock
        items.append(mo.md("### Your Prediction"))
        items.append(partD_prediction)

        items.append(mo.stop(partD_prediction.value is None,
            mo.callout(mo.md("Select your prediction above to unlock the Part D instruments."), kind="warn")))

        # Controls
        items.append(mo.md("### Hardware Tier Comparison"))
        items.append(mo.hstack([d1_hw_tier, d1_model_size, d1_gpu_count], justify="center", gap=2))

        # Physics
        _hw_name, _hw_tflops, _hw_bw = d1_hw_tier.value
        _params_b = d1_model_size.value
        _n_gpus = d1_gpu_count.value

        _grad_gb = _params_b * 1e9 * 2 / 1e9
        _ring_factor = 2.0 * (_n_gpus - 1) / max(_n_gpus, 1)
        _allreduce_gb = _ring_factor * _grad_gb
        _allreduce_s = _allreduce_gb / _hw_bw if _hw_bw > 0 else 999

        _flops_per_step = 6.0 * _params_b * 1e9 * 2048
        _compute_s = _flops_per_step / (_n_gpus * _hw_tflops * 1e12 * 0.50)

        _overlap_frac = 0.50
        _effective_comm = _allreduce_s * (1 - _overlap_frac)
        _step_time_s = _compute_s + _effective_comm
        _efficiency = _compute_s / _step_time_s if _step_time_s > 0 else 0
        _efficiency_pct = _efficiency * 100

        # Compare all three tiers
        _tiers = [
            ("H100 + IB NDR", H100_TFLOPS_FP16, IB_NDR_BW_GBS),
            ("A100 + IB HDR", A100_TFLOPS_FP16, IB_HDR_BW_GBS),
            ("T4 + 100GbE", T4_TFLOPS_FP16, ETH_100G_BW_GBS),
        ]
        _tier_effs = []
        for _tname, _tflops, _tbw in _tiers:
            _ar = _ring_factor * _grad_gb / _tbw
            _cs = _flops_per_step / (_n_gpus * _tflops * 1e12 * 0.50)
            _ec = _ar * (1 - _overlap_frac)
            _st = _cs + _ec
            _eff = (_cs / _st * 100) if _st > 0 else 0
            _tier_effs.append((_tname, _eff, _tflops, _tbw))

        # Bar chart
        _fig = go.Figure()
        _colors_bar = [COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["RedLine"]]
        for _i, (_tname, _eff, _, _) in enumerate(_tier_effs):
            _fig.add_trace(go.Bar(
                x=[_tname], y=[_eff],
                marker_color=_colors_bar[_i],
                text=[f"{_eff:.1f}%"],
                textposition="auto",
                hovertemplate=f"{_tname}<br>Efficiency: %{{y:.1f}}%<extra></extra>",
                showlegend=False,
            ))
        _fig.add_hline(y=50, line=dict(color=COLORS["TextMuted"], width=1, dash="dot"),
                       annotation_text="50% threshold", annotation_position="top right")
        _fig.update_layout(
            height=300,
            xaxis=dict(title="Hardware Tier"),
            yaxis=dict(title="DP Scaling Efficiency (%)", range=[0, 105]),
            margin=dict(t=30, b=50, l=50, r=20),
        )
        apply_plotly_theme(_fig)

        _eff_color = COLORS["GreenLine"] if _efficiency_pct >= 70 else (COLORS["OrangeLine"] if _efficiency_pct >= 40 else COLORS["RedLine"])

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Hardware Tier Scaling Comparison
            </div>
            <div>Hardware: <strong>{_hw_name}</strong> &mdash; {_hw_tflops} TFLOPS &middot; {_hw_bw} GB/s interconnect</div>
            <div>Model: {_params_b}B params &times; {_n_gpus} GPUs</div>
            <div>Compute time: <strong>{_compute_s:.2f}s</strong> &mdash; AllReduce: <strong>{_allreduce_s:.2f}s</strong></div>
            <div>Effective comm (50% overlap): <strong>{_effective_comm:.2f}s</strong></div>
            <div>Efficiency: <strong style="color:{_eff_color};">{_efficiency_pct:.1f}%</strong></div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:150px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Efficiency</div>
                <div style="font-size:2rem; font-weight:800; color:{_eff_color}; font-family:monospace;">{_efficiency_pct:.1f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">{_hw_name}</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:150px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Compute</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']}; font-family:monospace;">{_hw_tflops:.0f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">TFLOPS FP16</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:150px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Network</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['OrangeLine']}; font-family:monospace;">{_hw_bw:.0f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">GB/s</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # Prediction reveal
        if partD_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** A100 compute is 3.2x slower (312 vs 989 TFLOPS), which increases "
                "compute time from ~10s to ~32s per step. But IB HDR is 2x slower (25 vs 50 GB/s), "
                "doubling AllReduce time from ~14s to ~28s. With overlap, effective comm is ~14s. "
                "Efficiency = 32 / (32 + 14) = ~69%. Wait -- that is *higher* than H100's 55%! "
                "The slower GPU gives more time to hide communication. But throughput (steps/second) "
                "drops 3x, so training takes 3x longer. Efficiency != throughput."
            ), kind="success"))
        elif partD_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Wrong metric.** Same HBM capacity means the model fits, but efficiency depends "
                "on the comm/compute ratio, not memory capacity. The A100's slower compute actually "
                "improves the ratio, but the slower interconnect partially offsets this gain."
            ), kind="warn"))
        elif partD_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Directionally correct!** Slower compute does improve the comm/compute ratio. "
                "But 70% may overestimate -- the halved interconnect bandwidth (25 vs 50 GB/s) "
                "partially offsets the benefit. The actual efficiency depends on the exact model "
                "size and GPU count."
            ), kind="warn"))
        elif partD_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Incorrect.** A100 has 80 GB HBM, same as H100. With ZeRO-3 or 3D parallelism, "
                "175B trains on either GPU. The question is about efficiency, not feasibility."
            ), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Governing equations -- hardware tier scaling": mo.md("""
        **Comm/Compute Ratio Across Tiers**

        ```
        R = T_comm_effective / T_compute
        eta = 1 / (1 + R)
        ```

        When you switch hardware, both numerator and denominator change:
        - Slower GPU: T_compute increases (R decreases -> higher eta)
        - Slower network: T_comm increases (R increases -> lower eta)

        **Key insight**: Efficiency is scale-dependent. A T4 training a 7B model
        may achieve higher *efficiency* than an H100, but H100 has 15x higher
        *throughput*. The cost-optimal choice depends on total training time
        x cost-per-GPU-hour, not efficiency alone.

        **Hardware-Bandwidth Matching Rule**

        ```
        Optimal tier: min(cost_per_hour * T_step(N) * total_steps)
        ```

        Large models on slow interconnects have low efficiency AND low throughput.
        Small models on fast GPUs have high efficiency but waste expensive hardware.
            """)
        }))

        # Reflection
        items.append(mo.md("### Reflection"))
        items.append(partD_reflection)
        if partD_reflection.value is not None:
            if partD_reflection.value == "B":
                items.append(mo.callout(mo.md(
                    "**Correct.** Hardware tier selection is a matching problem: the comm/compute "
                    "ratio depends on both model size (gradient volume) and hardware capabilities "
                    "(compute TFLOPS and network bandwidth). Small models (7B) have small gradients "
                    "that even 100GbE handles; large models (175B) need IB NDR + NVLink. "
                    "Over-provisioning wastes budget; under-provisioning wastes GPU cycles on communication."
                ), kind="success"))
            else:
                items.append(mo.callout(mo.md(
                    "**Not quite.** The correct principle is matching hardware tier to model scale. "
                    "Both compute speed and interconnect bandwidth must be considered together. "
                    "A 7B model does not need H100 + IB NDR; a 175B model cannot efficiently use T4 + 100GbE."
                ), kind="warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def build_synthesis():
        items = []

        items.append(mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. The communication wall caps pure DP at ~55% efficiency for 175B models on IB NDR.</strong>
                    Ring-AllReduce transfers ~700 GB per step. Even at 50 GB/s with 50% overlap,
                    communication consumes nearly half the step time. No interconnect upgrade
                    eliminates this wall for frontier model sizes.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. ZeRO-3 is necessary but not sufficient: activation memory pushes OOM.</strong>
                    ZeRO-3 shards static memory to ~38 GB per GPU, but activations (~50 GB)
                    are not sharded. Total = 88 GB > 80 GB HBM. Pipeline or tensor parallelism
                    must reduce per-GPU activation memory.
                </div>
                <div>
                    <strong>3. 3D parallelism maps each strategy to its natural bandwidth tier.</strong>
                    TP (per-layer AllReduce) -> NVLink (900 GB/s). PP (inter-stage activation transfer)
                    -> IB (50 GB/s). DP (per-step gradient AllReduce) -> remaining bandwidth.
                    This mapping is physics, not convention.
                </div>
            </div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab V2-06: When Failure is Routine</strong> &mdash; You designed a 256-GPU
                    training configuration. But at this scale, hardware fails every few hours.
                    The next lab asks: how often should you checkpoint, and what happens when the
                    checkpoint storm itself becomes the bottleneck?
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> the Distributed Training chapter for the full 3D parallelism
                    derivation and the Conservation of Overhead principle.<br/>
                    <strong>Build:</strong> TinyTorch distributed module &mdash; implement data parallelism
                    with gradient accumulation in <code>tinytorch/src/distributed/</code>.
                </div>
            </div>
        </div>
        """))

        items.append(mo.accordion({
            "Self-Assessment": mo.md("""
1. Why does pure DP efficiency collapse for 175B models even on InfiniBand NDR?
2. Why can ZeRO-3 on 64 A100s not train a 175B model despite static memory fitting?
3. For TP=8, PP=4, DP=8 on 256 H100s, what is the pipeline bubble fraction at m=16?
4. Why must TP be confined within a single DGX node?

*If you cannot answer all four from memory, revisit Parts A, B, and C.*
""")
        }))

        items.append(mo.md("---"))
        items.append(mo.md("### Decision Log"))
        items.append(mo.md("Record the single most important insight from this lab. "
                           "This entry carries forward to Lab 06 and beyond via the Design Ledger."))
        items.append(synth_decision_ui)

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    tabs = mo.ui.tabs({
        "Part A -- The Communication Wall":         build_part_a(),
        "Part B -- The ZeRO Memory Trap":            build_part_b(),
        "Part C -- 3D Parallelism Design Challenge": build_part_c(),
        "Part D -- Hardware Tier Comparison":        build_part_d(),
        "Synthesis":                                  build_synthesis(),
    })
    tabs
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 9: LEDGER_HUD ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, partA_prediction, partB_prediction, partA_reflection, partC_reflection,
      partD_prediction, partD_reflection,
      a2_tp, a2_pp, a2_microbatches, ledger, mo, synth_decision_input):
    _tp = a2_tp.value
    _pp = a2_pp.value
    _m = a2_microbatches.value
    _tp_pp = _tp * _pp
    _dp = 256 // _tp_pp if _tp_pp <= 256 and 256 % _tp_pp == 0 else 0
    _config_valid = _dp > 0
    _bubble_pct = ((_pp - 1) / (_pp * _m) * 100) if _pp > 1 else 0.0

    # Effective MFU (simplified recalc for HUD)
    import math as _math
    _mfu_base = 0.52
    _tp_crosses_node = _tp > 8
    _tp_eff = 1.0 - (0.04 * _math.log2(max(_tp, 1)) * (1 if not _tp_crosses_node else 2.25))
    _pp_eff = 1.0 - ((_pp - 1) / (_pp * _m) if _pp > 1 else 0.0)
    _dp_eff = 1.0 - (0.02 * _math.log2(max(_dp, 1))) if _dp > 0 else 0
    _mfu_eff = max(0.0, min(_mfu_base * _tp_eff * _pp_eff * _dp_eff, _mfu_base))
    _mfu_pct = _mfu_eff * 100

    _context = "3d_parallel" if _tp > 1 or _pp > 1 else "data_parallel"

    ledger.save(
        chapter="v2_05",
        design={
            "context": _context,
            "tp_degree": _tp,
            "pp_degree": _pp,
            "dp_degree": _dp,
            "total_gpus": 256,
            "mfu_percent": round(_mfu_pct, 2),
            "partA_prediction": partA_prediction.value or "no_selection",
            "partA_correct": partA_prediction.value == "C",
            "partA_reflection": partA_reflection.value or "no_selection",
            "partB_prediction": partB_prediction.value or "no_selection",
            "partB_correct": partB_prediction.value == "B",
            "partC_reflection": partC_reflection.value or "no_selection",
            "partD_prediction": partD_prediction.value or "no_selection",
            "partD_correct": partD_prediction.value == "B",
            "partD_reflection": partD_reflection.value or "no_selection",
            "student_justification": str(synth_decision_input.value),
        },
    )

    _a1_ok = partA_prediction.value == "C"
    _a2_ok = partB_prediction.value == "B"
    _tier = "Optimal" if (_a1_ok and _a2_ok) else ("Partial" if (_a1_ok or _a2_ok) else "Developing")
    _tier_color = COLORS["GreenLine"] if _tier == "Optimal" else (COLORS["OrangeLine"] if _tier == "Partial" else COLORS["TextMuted"])

    mo.Html(f"""
    <div class="lab-hud">
        <div><span class="hud-label">LAB</span> <span class="hud-value">Vol2 &middot; Lab 05</span></div>
        <div><span class="hud-label">CHAPTER</span> <span class="hud-value">v2_05 &middot; Distributed Training</span></div>
        <div><span class="hud-label">CONFIG</span> <span class="hud-value">TP={_tp}&times;PP={_pp}&times;DP={_dp}</span></div>
        <div><span class="hud-label">MFU</span> <span style="color:{COLORS['GreenLine'] if _mfu_pct >= 40 else COLORS['OrangeLine']}; font-family:var(--font-mono);">{_mfu_pct:.1f}%</span></div>
        <div><span class="hud-label">PART A</span> <span class="{'hud-active' if _a1_ok else 'hud-none'}">{"CORRECT" if _a1_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">PART B</span> <span class="{'hud-active' if _a2_ok else 'hud-none'}">{"CORRECT" if _a2_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">TIER</span> <span style="color:{_tier_color}; font-family:var(--font-mono);">{_tier.upper()}</span></div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
