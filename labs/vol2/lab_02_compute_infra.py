import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 02: THE INTERCONNECT WALL
#
# Volume II, Chapter 2: Compute Infrastructure
#
# Core invariant: The interconnect hierarchy creates bandwidth cliffs.
# NVLink (900 GB/s) is 18× faster than PCIe Gen4 (50 GB/s per GPU).
# Crossing the node boundary causes a 4–9× additional bandwidth drop.
# This "interconnect wall" determines whether distributed training is feasible.
#
# Structure:
#   Act I  — The Interconnect Cliff (12–15 min)
#     Stakeholder: Infra Architect comparing NVLink DGX vs PCIe servers
#     Instruments: Bandwidth explorer with model size, batch, interconnect type
#     Prediction-vs-reality overlay after instruments run
#     Reflection: Why AllReduce volume = 2× model size
#
#   Act II — The Multi-Node Scaling Wall (20–25 min)
#     Stakeholder: ML Infra Lead scaling from 1 DGX node to 16 nodes
#     Instruments: Multi-node scaling analyzer with IB link count
#     Failure state: >100% overhead triggers danger callout
#     Reflection: Best remedy for inter-node bandwidth bottleneck
#
# 2 Contexts: Single-node (NVLink) vs Multi-node (InfiniBand)
#
# Design Ledger: saves context, interconnect, nodes, model size,
#                comm overhead, predictions, decisions.
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible) ────────────────────────
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

    # ── Hardware constants (all from @sec-compute-infrastructure) ───────────
    # Bandwidth figures
    NVLINK4_BW_GBS      = 900   # NVLink4 bidirectional per DGX H100 node, NVIDIA spec
    PCIE_GEN4_BW_GBS    = 50    # PCIe Gen4 ×16 per GPU, NVIDIA P2P effective BW
    IB_HDR200_BW_GBS    = 400   # InfiniBand HDR200 peak bidirectional (200 Gbps × 2 directions)
    IB_HDR200_EFF_GBS   = 50    # InfiniBand HDR200 effective per-port AllReduce bandwidth
                                 # HDR200 = 200 Gbps = 25 GB/s raw; ~50 GB/s effective with
                                 # bidirectional pipelining (reduce-scatter + allgather overlap)
                                 # Source: @sec-compute-infrastructure bandwidth hierarchy table

    # H100 compute specs
    H100_TFLOPS_FP16   = 1979   # H100 SXM5 FP16 tensor core TFLOPS, NVIDIA spec
    H100_RAM_GB        = 80     # H100 SXM5 HBM3e capacity, NVIDIA spec

    # FP16 bytes per parameter
    BYTES_PER_PARAM    = 2      # FP16 = 2 bytes/parameter

    # Calibrated compute constant: K_COMP × params_b × batch = compute_time_s
    # Calibrated so that 70B params × batch 32 = 2.1 s (spec reference point)
    # Derivation: 2.1 = (6 × 70e9 × seq_139 × 32) / (1979e12 × 0.45), seq_139 ≈ 139 tokens
    # Equivalent to: K_COMP = 2.1 / (70 × 32) = 9.375e-4
    K_COMP = 2.1 / (70.0 * 32.0)   # s / (B_params × batch)

    ledger = DesignLedger()
    return (
        mo, ledger, go, np,
        COLORS, LAB_CSS, apply_plotly_theme,
        NVLINK4_BW_GBS, PCIE_GEN4_BW_GBS,
        IB_HDR200_BW_GBS, IB_HDR200_EFF_GBS,
        H100_TFLOPS_FP16, H100_RAM_GB, BYTES_PER_PARAM,
        K_COMP,
    )


# ─── CELL 1: HEADER ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _indigo = COLORS["Cloud"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume II · Lab 02
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Interconnect Wall
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 660px; line-height: 1.65;">
                NVLink runs at 900 GB/s. PCIe runs at 50 GB/s. InfiniBand runs at 400 GB/s.
                These are not implementation details — they are the physical cliffs
                that determine whether distributed training is feasible at all.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: Interconnect Cliff · Act II: Multi-Node Scaling Wall
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Prereq: @sec-compute-infrastructure
                </span>
                <span class="badge badge-fail">
                    Interconnect Wall Active
                </span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: RECOMMENDED READING ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **Bandwidth Hierarchy** (@sec-compute-infrastructure) — HBM, NVLink, InfiniBand, PCIe: what each tier is, where it sits in the stack, and the order-of-magnitude gaps between them.
    - **AllReduce and Ring Topology** (@sec-compute-infrastructure) — How gradient synchronization works: why the data volume is 2× model size, not 1×, and why ring topology is bandwidth-optimal.
    - **Node vs. Pod Boundaries** (@sec-compute-infrastructure) — Why crossing the server boundary causes a bandwidth cliff, and how hierarchical AllReduce attempts to bridge it.
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    context_toggle = mo.ui.radio(
        options={
            "Single-node (NVLink DGX)": "single_node",
            "Multi-node (InfiniBand Cluster)": "multi_node",
        },
        value="Single-node (NVLink DGX)",
        label="Deployment context:",
        inline=True,
    )

    _c = COLORS["BlueLine"]
    mo.vstack([
        mo.Html(f"""
        <div style="border-bottom: 2px solid {COLORS['Border']}; padding-bottom: 16px; margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                Infrastructure Context
            </div>
        </div>
        """),
        context_toggle,
    ])
    return (context_toggle,)


# ═════════════════════════════════════════════════════════════════════════════
# ACT I — THE INTERCONNECT CLIFF
# ═════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div style="margin: 32px 0 8px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: #006395; color: white; border-radius: 6px;
                        padding: 3px 10px; font-size: 0.72rem; font-weight: 800;
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act I
            </div>
            <div style="font-size: 1.6rem; font-weight: 900; color: #0f172a;">
                The Interconnect Cliff
            </div>
            <div style="flex: 1; height: 1px; background: #e2e8f0;"></div>
            <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;">
                12–15 min
            </div>
        </div>
    </div>
    """)
    return


# ─── ACT I: STAKEHOLDER MESSAGE ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["BlueLine"]
    _bg    = COLORS["BlueL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · Infrastructure Architect
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We need to gradient-sync a 70B model (140 GB FP16) across 8 GPUs.
            We have two options: a single NVLink DGX H100 node, or 8 separate PCIe servers.
            The PCIe option is 40% cheaper. My manager is asking whether the bandwidth difference
            actually matters in practice. Can you model this out before we sign the purchase order?"
        </div>
    </div>
    """)
    return


# ─── ACT I: SCENARIO SETUP ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("""
        ## The Physics of Gradient Synchronization

        Training a large model across multiple GPUs requires synchronizing gradients after every
        backward pass. The dominant algorithm — **Ring AllReduce** — sends each gradient tensor
        in two phases across the interconnect:

        1. **Reduce-Scatter**: Each GPU sends its gradients around the ring once (1× model volume)
        2. **AllGather**: The reduced result is broadcast back to all GPUs (1× model volume)

        Total data transferred per synchronization step: **2 × model size in bytes**.

        For a 70B parameter model in FP16 (2 bytes/parameter):

        ```
        AllReduce volume = 2 × (70 × 10⁹ parameters × 2 bytes) = 2 × 140 GB = 280 GB per step
        ```

        The interconnect bandwidth determines how long this synchronization takes — and whether
        that time is negligible or catastrophic relative to compute time.
        """),
    ])
    return


# ─── ACT I: PREDICTION LOCK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("---")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) Barely any difference — gradient compression will fix the bandwidth gap": "A",
            "B) ~2× throughput difference — PCIe is slower but still practical for production": "B",
            "C) ~3–5× throughput difference — PCIe communication overhead dominates training": "C",
            "D) PCIe makes training impossible — infinite overhead, cannot converge": "D",
        },
        label="""**Commit to your prediction before running the instruments.**

When training a 70B parameter model across 8 GPUs, how much worse is the training throughput
on 8 PCIe-connected servers compared to one NVLink DGX node (assuming the same H100 GPUs)?""",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.vstack([
            act1_pred,
            mo.callout(
                mo.md("Select your prediction to continue. Commit before the instruments run."),
                kind="warn",
            ),
        ])
    )
    mo.callout(
        mo.md(f"**Prediction locked:** {act1_pred.value[:2]}. Now run the simulator below to test your hypothesis."),
        kind="info",
    )
    return


# ─── ACT I: INSTRUMENTS ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Bandwidth Comparison Explorer")
    return


@app.cell(hide_code=True)
def _(mo):
    model_params_b = mo.ui.slider(
        start=1, stop=175, value=70, step=1,
        label="Model parameters (billions)",
    )
    batch_size = mo.ui.slider(
        start=1, stop=128, value=32, step=1,
        label="Batch size per GPU",
    )
    interconnect_type = mo.ui.dropdown(
        options={
            "NVLink4 (DGX H100) — 900 GB/s": "nvlink",
            "PCIe Gen4 — 50 GB/s": "pcie",
            "InfiniBand HDR200 — 400 GB/s": "infiniband",
        },
        value="NVLink4 (DGX H100) — 900 GB/s",
        label="Interconnect type",
    )
    mo.vstack([
        mo.hstack([model_params_b, batch_size], justify="start", gap=2),
        interconnect_type,
    ])
    return (model_params_b, batch_size, interconnect_type)


@app.cell(hide_code=True)
def _(
    mo, go, np,
    model_params_b, batch_size, interconnect_type,
    COLORS, apply_plotly_theme,
    NVLINK4_BW_GBS, PCIE_GEN4_BW_GBS, IB_HDR200_EFF_GBS,
    BYTES_PER_PARAM, K_COMP,
):
    # ── Physics engine ────────────────────────────────────────────────────────
    # Model memory in GB (FP16)
    model_gb    = model_params_b.value * BYTES_PER_PARAM  # GB

    # AllReduce volume: 2 × model size
    # Ring AllReduce: reduce-scatter (1× model) + allgather (1× model)
    allreduce_gb = 2.0 * model_gb

    # Select interconnect bandwidth
    # For Act I comparison: NVLink vs PCIe vs IB as single-link options
    _bw_map = {
        "nvlink":      NVLINK4_BW_GBS,
        "pcie":        PCIE_GEN4_BW_GBS,
        "infiniband":  IB_HDR200_EFF_GBS,   # effective single-port IB BW
    }
    _bw_name_map = {
        "nvlink":      "NVLink4 900 GB/s",
        "pcie":        "PCIe Gen4 50 GB/s",
        "infiniband":  "InfiniBand HDR200 ~50 GB/s effective",
    }
    bw_gbs      = _bw_map[interconnect_type.value]
    bw_name     = _bw_name_map[interconnect_type.value]

    # Communication time (seconds)
    comm_time_s = allreduce_gb / bw_gbs

    # Compute time per step (calibrated to spec reference point)
    # Formula: K_COMP × params_b × batch_size
    # Calibrated: 70B × batch 32 = 2.1 s (matching @sec-compute-infrastructure numbers)
    # K_COMP = 2.1 / (70 × 32); equivalent to 6N FLOPs at seq_len=139, MFU=0.45 on H100
    comp_time_s  = K_COMP * model_params_b.value * batch_size.value

    # Overhead ratio and efficiency
    overhead_pct = (comm_time_s / comp_time_s) * 100.0
    efficiency   = comp_time_s / (comp_time_s + comm_time_s) * 100.0

    # Step time
    total_time_s = comp_time_s + comm_time_s
    eff_mfu_pct  = (comp_time_s / total_time_s) * 0.45 * 100.0

    # Color coding
    if overhead_pct <= 20:
        ovhd_color = COLORS["GreenLine"]
        ovhd_label = "Acceptable"
    elif overhead_pct <= 100:
        ovhd_color = COLORS["OrangeLine"]
        ovhd_label = "High"
    else:
        ovhd_color = COLORS["RedLine"]
        ovhd_label = "Bottleneck"

    eff_color = COLORS["GreenLine"] if efficiency >= 80 else (
        COLORS["OrangeLine"] if efficiency >= 40 else COLORS["RedLine"]
    )

    # ── Bar chart: time breakdown ─────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Compute",
        x=["Step Time Breakdown"],
        y=[comp_time_s],
        marker_color=COLORS["BlueLine"],
        width=0.4,
        text=[f"{comp_time_s:.2f}s"],
        textposition="inside",
        textfont=dict(color="white", size=13, family="SF Mono, Fira Code, monospace"),
    ))
    fig.add_trace(go.Bar(
        name="AllReduce (Communication)",
        x=["Step Time Breakdown"],
        y=[comm_time_s],
        marker_color=ovhd_color,
        width=0.4,
        text=[f"{comm_time_s:.2f}s"],
        textposition="inside",
        textfont=dict(color="white", size=13, family="SF Mono, Fira Code, monospace"),
    ))
    fig.update_layout(
        barmode="stack",
        height=260,
        legend=dict(orientation="h", y=-0.25),
        yaxis=dict(title="Seconds per step"),
        showlegend=True,
        margin=dict(l=40, r=20, t=16, b=60),
    )
    apply_plotly_theme(fig)

    # ── Display ───────────────────────────────────────────────────────────────
    mo.vstack([
        mo.Html(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0;
                    border-radius: 12px; padding: 18px 22px; margin: 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Physics
            </div>
            <div style="font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem;
                        color: #1e293b; line-height: 2.0;">
                AllReduce volume = 2 × model_size = 2 × {model_gb:.0f} GB = <strong>{allreduce_gb:.0f} GB</strong><br>
                Communication time = {allreduce_gb:.0f} GB ÷ {bw_gbs} GB/s = <strong style="color:{ovhd_color}">{comm_time_s:.3f} s</strong><br>
                Compute time = K × {model_params_b.value}B params × batch {batch_size.value} = <strong>{comp_time_s:.3f} s</strong><br>
                Communication overhead = {comm_time_s:.3f} ÷ {comp_time_s:.3f} = <strong style="color:{ovhd_color}">{overhead_pct:.1f}%</strong>
            </div>
        </div>
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 4px 0 12px 0;">
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.08em;">
                    Comm Overhead
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {ovhd_color};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {overhead_pct:.1f}%
                </div>
                <div style="font-size: 0.75rem; font-weight: 700; color: {ovhd_color};">
                    {ovhd_label}
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.08em;">
                    Training Efficiency
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {eff_color};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {efficiency:.1f}%
                </div>
                <div style="font-size: 0.75rem; font-weight: 700; color: {eff_color};">
                    GPU-compute utilization
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.08em;">
                    Comm Time
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {ovhd_color};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {comm_time_s:.2f}s
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">
                    {bw_name}
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.08em;">
                    Compute Time
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {COLORS['BlueLine']};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {comp_time_s:.2f}s
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">
                    H100 FP16 @ 45% MFU
                </div>
            </div>
        </div>
        """),
        mo.as_html(fig),
    ])
    return (
        comm_time_s, comp_time_s, overhead_pct, efficiency,
        eff_mfu_pct, allreduce_gb, model_gb,
    )


# ─── ACT I: CONTEXTUAL FEEDBACK ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, overhead_pct, comm_time_s, comp_time_s, interconnect_type):
    _iv = interconnect_type.value
    if _iv == "nvlink":
        if overhead_pct <= 20:
            mo.callout(mo.md(
                f"**NVLink efficiency.** {overhead_pct:.1f}% communication overhead. "
                "At 900 GB/s, the AllReduce completes in well under a second — less than a typical "
                "compute step. This is the design point NVLink was engineered for: keeping the "
                "interconnect invisible to the training loop."
            ), kind="success")
        else:
            mo.callout(mo.md(
                f"**NVLink at its limits.** {overhead_pct:.1f}% overhead. "
                "Even NVLink can be stressed by very large models with small batch sizes — "
                "the compute-to-communication ratio collapses as batch size shrinks."
            ), kind="warn")
    elif _iv == "pcie":
        mo.callout(mo.md(
            f"**PCIe bandwidth cliff.** {overhead_pct:.1f}% communication overhead. "
            f"AllReduce takes {comm_time_s:.2f}s against a compute step of {comp_time_s:.2f}s. "
            "At 50 GB/s, the interconnect is 18× slower than NVLink. The GPU spends most of its "
            "time waiting for gradients — not computing. The 40% hardware cost savings is "
            "immediately offset by training throughput collapse."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**InfiniBand: the inter-node tier.** {overhead_pct:.1f}% overhead. "
            "At 400 GB/s, InfiniBand falls between NVLink and PCIe — sufficient for moderate "
            "workloads but problematic for the largest models. This is the bandwidth available "
            "when you cross the node boundary in Act II."
        ), kind="info")
    return


# ─── ACT I: PREDICTION-VS-REALITY OVERLAY ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, COLORS):
    # Compute the NVLink and PCIe throughput ratio from physics
    # NVLink: overhead = 280/900 / 2.1 ≈ 15%  → efficiency ≈ 85%
    # PCIe:   overhead = 280/50  / 2.1 ≈ 267% → efficiency ≈ 27%
    # Throughput ratio ≈ 85% / 27% ≈ 3.1×
    # This is Act I physics for 70B default config

    _nvlink_eff  = 0.85   # 85% training efficiency on NVLink (15% comm overhead)
    _pcie_eff    = 0.27   # 27% training efficiency on PCIe (267% comm overhead)
    _actual_ratio = _nvlink_eff / _pcie_eff   # ≈ 3.1×

    _predicted_ratio_map = {
        "A": 1.1,   # "barely any difference"
        "B": 2.0,   # "~2× throughput difference"
        "C": 4.0,   # "~3–5× throughput difference" (mid of range)
        "D": 999.0, # "impossible / infinite"
    }

    _letter = act1_pred.value[0] if act1_pred.value else "A"
    _predicted_ratio = _predicted_ratio_map.get(_letter, 1.0)

    _correct = _letter == "C"
    _gap_desc = ""
    if _letter == "A":
        _gap_desc = (
            f"You predicted ~{_predicted_ratio:.1f}× throughput difference (gradient compression makes it negligible). "
            f"The actual ratio is **{_actual_ratio:.1f}×**. Gradient compression reduces volume by 10–100× "
            "but requires an extra compute pass and introduces approximation error. It does not close "
            "an 18× bandwidth gap without severe accuracy cost."
        )
    elif _letter == "B":
        _gap_desc = (
            f"You predicted ~{_predicted_ratio:.0f}× throughput difference. "
            f"The actual ratio is **{_actual_ratio:.1f}×**. At 267% communication overhead, the PCIe "
            "configuration spends 72% of its time in AllReduce — not computing. "
            "A 2× estimate is too optimistic: the math gives 3–4×."
        )
    elif _letter == "C":
        _gap_desc = (
            f"Correct. You predicted ~3–5× throughput difference. "
            f"The physics gives **{_actual_ratio:.1f}×**: NVLink at 85% efficiency vs. PCIe at 27% efficiency. "
            "The 40% hardware savings on PCIe servers translates to 3× slower training — "
            "a wall-clock cost that erases the hardware savings within weeks."
        )
    elif _letter == "D":
        _gap_desc = (
            f"You predicted PCIe makes training impossible. "
            f"The actual ratio is **{_actual_ratio:.1f}×** — substantial, but not infinite. "
            "Training at 267% overhead is extremely inefficient but technically converges. "
            "The practical barrier is cost: training takes 3× as long, which means 3× the GPU-hours."
        )

    _kind = "success" if _correct else "warn"
    _header = "Correct prediction." if _correct else f"You predicted option {_letter}."

    mo.callout(mo.md(
        f"**Prediction vs. Reality — Act I**\n\n"
        f"{_header} {_gap_desc}\n\n"
        f"**The governing numbers at 70B / batch 32:**\n"
        f"- NVLink: 280 GB ÷ 900 GB/s = 0.31 s comm | 2.1 s compute → **15% overhead, 85% efficiency**\n"
        f"- PCIe: 280 GB ÷ 50 GB/s = 5.6 s comm | 2.1 s compute → **267% overhead, 27% efficiency**\n"
        f"- Throughput ratio: 85% ÷ 27% ≈ **{_actual_ratio:.1f}×** — PCIe trains {_actual_ratio:.1f}× slower."
    ), kind=_kind)
    return


# ─── ACT I: REFLECTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("---")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) You send weights forward and gradients backward — two passes across the network": "A",
            "B) Ring AllReduce sends each parameter twice: reduce-scatter (1×) + allgather (1×)": "B",
            "C) FP16 precision requires 2× more data than FP32 for the same gradient accuracy": "C",
            "D) Gradient accumulation over multiple micro-batches doubles the sync volume": "D",
        },
        label="""**Reflection.** The AllReduce data volume equals 2× the model size (280 GB for a 70B FP16 model).
Why is the factor exactly 2×, and not 1× or some other number?""",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.vstack([
            act1_reflect,
            mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"),
        ])
    )

    _feedback = {
        "A": mo.callout(mo.md(
            "**Not quite.** Ring AllReduce does not transmit model weights at all — only gradients. "
            "The forward pass and backward pass are local operations on each GPU. "
            "The factor-of-2 comes from the two phases of the ring algorithm itself, "
            "not from the forward/backward decomposition."
        ), kind="warn"),
        "B": mo.callout(mo.md(
            "**Correct.** Ring AllReduce operates in two phases of identical volume. "
            "In **reduce-scatter**, each of N GPUs sends 1/N of the gradient tensor to its neighbor "
            "and receives a partial reduction, cycling through the ring. Total data: 1× model size. "
            "In **allgather**, each GPU then broadcasts its reduced shard back around the ring. "
            "Total data: another 1× model size. Combined: exactly 2× model size, independent of N. "
            "This is why ring AllReduce is called bandwidth-optimal: doubling the number of GPUs "
            "does not increase the per-GPU data volume."
        ), kind="success"),
        "C": mo.callout(mo.md(
            "**Not quite.** The 2× factor is not a precision artifact. FP16 already determines the "
            "per-parameter byte count (2 bytes). The factor-of-2 comes from the ring AllReduce "
            "algorithm structure — reduce-scatter plus allgather — which each contribute 1× model "
            "volume. If you used FP32 gradients, the per-parameter cost would be 4 bytes, "
            "but the multiplier would still be 2×."
        ), kind="warn"),
        "D": mo.callout(mo.md(
            "**Not quite.** Gradient accumulation reduces the sync frequency (you sync every K steps "
            "instead of every step), but each sync still involves 2× model size — not 4×. "
            "Accumulation helps by amortizing the AllReduce cost over K compute steps, "
            "but it does not change the per-sync data volume."
        ), kind="warn"),
    }

    mo.vstack([
        act1_reflect,
        _feedback[act1_reflect.value[0]],
    ])
    return


# ─── ACT I: MATH PEEK ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The Governing Equation — AllReduce Bandwidth Model": mo.md("""
        **AllReduce Communication Time**

        ```
        T_comm = (2 × N_params × bytes_per_param) / BW_interconnect
        ```

        - **N_params** — model parameter count (e.g., 70 × 10⁹)
        - **bytes_per_param** — 2 for FP16, 4 for FP32
        - **BW_interconnect** — effective interconnect bandwidth in GB/s
        - Factor of **2** — ring AllReduce reduce-scatter + allgather

        **Communication Overhead**

        ```
        overhead = T_comm / T_compute
        efficiency = T_compute / (T_compute + T_comm)
        ```

        When `overhead > 1.0` (i.e., T_comm > T_compute), communication is the bottleneck.
        The GPU waits for gradients longer than it spends computing them.

        **The Interconnect Hierarchy (from @sec-compute-infrastructure)**

        | Interconnect | Bandwidth | Relative to NVLink |
        |---|---|---|
        | NVLink4 (DGX H100) | 900 GB/s | 1× (baseline) |
        | InfiniBand HDR200 | 400 GB/s | 0.44× |
        | PCIe Gen4 ×16 | 50 GB/s | 0.056× (18× slower) |

        NVLink achieves 900 GB/s because it is a dedicated point-to-point bus
        etched onto the server backplane, with no contention from I/O traffic.
        PCIe is a general-purpose bus shared with storage controllers,
        network cards, and other peripherals — its ML bandwidth is further degraded
        by protocol overhead and peer-to-peer routing through the CPU.
        """),
    })
    return


# ═════════════════════════════════════════════════════════════════════════════
# ACT II — THE MULTI-NODE SCALING WALL
# ═════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(act1_reflect.value is None)
    mo.Html("""
    <div style="margin: 40px 0 8px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: #CB202D; color: white; border-radius: 6px;
                        padding: 3px 10px; font-size: 0.72rem; font-weight: 800;
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act II
            </div>
            <div style="font-size: 1.6rem; font-weight: 900; color: #0f172a;">
                The Multi-Node Scaling Wall
            </div>
            <div style="flex: 1; height: 1px; background: #e2e8f0;"></div>
            <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;">
                20–25 min
            </div>
        </div>
    </div>
    """)
    return


# ─── ACT II: STAKEHOLDER MESSAGE ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflect, COLORS):
    mo.stop(act1_reflect.value is None)
    _color = COLORS["RedLine"]
    _bg    = COLORS["RedL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · ML Infrastructure Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We're scaling from 1 DGX H100 node (8 GPUs, NVLink) to 16 nodes (128 GPUs, InfiniBand).
            We expected 16× more throughput — we budgeted 6 months to hit this target.
            First week of benchmarking shows we're only getting 6× throughput on 128 GPUs
            versus 1 DGX node. The engineering team is pointing at software bugs in
            distributed PyTorch. But I'm not convinced. What is actually happening?"
        </div>
    </div>
    """)
    return


# ─── ACT II: SCENARIO SETUP ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(act1_reflect.value is None)
    mo.vstack([
        mo.md("""
        ## Why Ideal Scaling Does Not Survive the Node Boundary

        Scaling from 1 node to 16 nodes introduces a fundamental topological change:
        intra-node communication (NVLink) is replaced by inter-node communication
        (InfiniBand) for the final AllReduce aggregation across nodes.

        A 16-node cluster with 8 GPUs per node must perform a hierarchical AllReduce:

        1. **Intra-node AllReduce** (NVLink, 900 GB/s): Each node reduces its 8 GPUs locally.
           Data volume per GPU: (N_gpus_per_node - 1) / N_gpus_per_node × model_size
        2. **Inter-node AllReduce** (InfiniBand, 400 GB/s × IB_links): The 16 node representatives
           synchronize across the fabric. Data volume per node: model_size / N_gpus_per_node

        The inter-node step is the bottleneck — it runs at InfiniBand bandwidth,
        which is a 2.25× cliff below NVLink and is shared by all 16 nodes simultaneously.
        """),
    ])
    return


# ─── ACT II: PREDICTION LOCK ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(act1_reflect.value is None)
    mo.md("---")
    return


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(act1_reflect.value is None)
    act2_pred = mo.ui.radio(
        options={
            "A) Software bugs — distributed PyTorch has framework overhead that scales badly with node count": "A",
            "B) InfiniBand latency — each network hop adds 1–2 µs, compounding at 128 GPUs": "B",
            "C) The node boundary creates a bandwidth cliff — inter-node AllReduce is bottlenecked by InfiniBand": "C",
            "D) 128 GPUs exceeds the fat-tree topology's bisection bandwidth capacity": "D",
        },
        label="""**Commit to your prediction.**

The team observed only 6× throughput scaling on 128 GPUs (16 nodes) versus 8 GPUs (1 node).
The expectation was 16× scaling. What is the primary cause of this 2.7× shortfall?""",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act1_reflect, act2_pred):
    mo.stop(act1_reflect.value is None)
    mo.stop(
        act2_pred.value is None,
        mo.vstack([
            act2_pred,
            mo.callout(mo.md("Select your prediction to continue."), kind="warn"),
        ])
    )
    mo.callout(
        mo.md(f"**Prediction locked:** {act2_pred.value[:2]}. Run the multi-node analyzer to test your hypothesis."),
        kind="info",
    )
    return


# ─── ACT II: INSTRUMENTS ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(act1_reflect.value is None)
    mo.md("## Multi-Node Scaling Analyzer")
    return


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(act1_reflect.value is None)
    n_nodes = mo.ui.slider(
        start=1, stop=64, value=16, step=1,
        label="Number of nodes",
    )
    gpus_per_node = mo.ui.slider(
        start=4, stop=8, value=8, step=4,
        label="GPUs per node (NVLink)",
    )
    act2_model_b = mo.ui.slider(
        start=7, stop=175, value=70, step=7,
        label="Model size (billions of parameters)",
    )
    ib_links = mo.ui.slider(
        start=1, stop=8, value=1, step=1,
        label="InfiniBand links per node",
    )
    mo.vstack([
        mo.hstack([n_nodes, gpus_per_node], justify="start", gap=2),
        mo.hstack([act2_model_b, ib_links], justify="start", gap=2),
    ])
    return (n_nodes, gpus_per_node, act2_model_b, ib_links)


@app.cell(hide_code=True)
def _(
    mo, go, np,
    n_nodes, gpus_per_node, act2_model_b, ib_links,
    act1_reflect,
    COLORS, apply_plotly_theme,
    NVLINK4_BW_GBS, IB_HDR200_EFF_GBS, IB_HDR200_BW_GBS,
    BYTES_PER_PARAM, K_COMP,
):
    mo.stop(act1_reflect.value is None)

    # ── Physics engine ────────────────────────────────────────────────────────
    N_nodes      = n_nodes.value
    N_gpu_node   = gpus_per_node.value
    N_gpu_total  = N_nodes * N_gpu_node
    M_gb         = act2_model_b.value * BYTES_PER_PARAM   # model GB (FP16)

    # Intra-node AllReduce (NVLink)
    # Ring AllReduce within 1 node: each GPU sends/receives 2 × (N-1)/N × model_size
    intra_data_gb   = 2.0 * (N_gpu_node - 1.0) / N_gpu_node * M_gb
    intra_comm_s    = intra_data_gb / NVLINK4_BW_GBS

    # Inter-node AllReduce (InfiniBand)
    # After intra-node reduce, each node holds the full gradient.
    # Inter-node ring AllReduce over N_nodes nodes:
    #   Total data per node: 2 × model_size (bandwidth-optimal ring)
    #   Effective IB bandwidth per node = IB_HDR200_EFF_GBS × ib_links
    # Source: ring AllReduce bandwidth analysis, @sec-compute-infrastructure
    ib_bw_node      = IB_HDR200_EFF_GBS * ib_links.value
    inter_comm_s    = 2.0 * M_gb / ib_bw_node

    # Total communication time is the sum (hierarchical, sequential phases)
    total_comm_s    = intra_comm_s + inter_comm_s

    # Compute time per step (calibrated K_COMP formula, batch 32 reference)
    batch_ref       = 32
    comp_time2_s    = K_COMP * act2_model_b.value * batch_ref

    # Overhead and efficiency
    overhead2_pct   = (total_comm_s / comp_time2_s) * 100.0
    efficiency2     = comp_time2_s / (comp_time2_s + total_comm_s) * 100.0

    # Scaling efficiency vs 1 DGX node (8 GPUs, NVLink only)
    # Single-node baseline (N_gpu_node = 8 by default):
    intra_base_s    = (2.0 * 7.0 / 8.0 * M_gb) / NVLINK4_BW_GBS
    step_base_s     = comp_time2_s + intra_base_s

    # Multi-node step time
    step_multi_s    = comp_time2_s + total_comm_s

    # Ideal: linear speedup with number of nodes
    ideal_speedup   = float(N_nodes)    # vs 1 node
    actual_speedup  = (step_base_s / step_multi_s) * ideal_speedup
    scaling_eff_pct = (actual_speedup / ideal_speedup) * 100.0

    # ── Failure state ─────────────────────────────────────────────────────────
    _oom = overhead2_pct > 100.0

    # ── Scaling curve ─────────────────────────────────────────────────────────
    node_range  = np.arange(1, 65, 1)
    ideal_curve = node_range * N_gpu_node / N_gpu_node  # normalized to 1 node

    actual_curve = []
    for _n in node_range:
        # For _n=1: no inter-node comm (pure NVLink), intra only
        if _n == 1:
            _intra_time_n = intra_comm_s
            _inter_time   = 0.0
        else:
            _intra_time_n = intra_comm_s
            # Inter-node: 2 × M_gb / ib_bw_node (same formula regardless of N_nodes)
            _inter_time   = 2.0 * M_gb / ib_bw_node
        _total_comm_n = _intra_time_n + _inter_time
        _step_n = comp_time2_s + _total_comm_n
        _speedup_n = (step_base_s / _step_n) * _n
        actual_curve.append(_speedup_n)

    actual_curve = np.array(actual_curve)
    ideal_curve  = node_range.astype(float)   # ideal = linear in nodes

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=node_range, y=ideal_curve,
        mode="lines", name="Ideal (linear)",
        line=dict(color=COLORS["GreenLine"], width=2, dash="dash"),
    ))
    fig2.add_trace(go.Scatter(
        x=node_range, y=actual_curve,
        mode="lines", name="Actual (bandwidth-limited)",
        line=dict(color=COLORS["BlueLine"], width=2.5),
    ))
    # Mark current configuration
    fig2.add_trace(go.Scatter(
        x=[N_nodes], y=[actual_speedup],
        mode="markers", name=f"Current ({N_nodes} nodes)",
        marker=dict(
            color=COLORS["RedLine"] if _oom else COLORS["OrangeLine"],
            size=14, symbol="circle",
            line=dict(color="white", width=2),
        ),
    ))
    fig2.update_layout(
        height=300,
        xaxis=dict(title="Number of nodes", range=[1, 64]),
        yaxis=dict(title="Throughput speedup vs 1 node"),
        legend=dict(orientation="h", y=-0.32),
        margin=dict(l=40, r=20, t=16, b=80),
    )
    apply_plotly_theme(fig2)

    # ── Color coding ──────────────────────────────────────────────────────────
    eff2_color = (
        COLORS["GreenLine"] if scaling_eff_pct >= 70 else
        COLORS["OrangeLine"] if scaling_eff_pct >= 40 else
        COLORS["RedLine"]
    )
    ovhd2_color = COLORS["RedLine"] if _oom else (
        COLORS["OrangeLine"] if overhead2_pct > 20 else COLORS["GreenLine"]
    )

    # ── Display ───────────────────────────────────────────────────────────────
    _bandwidth_cliff_note = ""
    if N_nodes > 1:
        _cliff_ratio = NVLINK4_BW_GBS / ib_bw_node
        _bandwidth_cliff_note = (
            f"Node boundary: NVLink {NVLINK4_BW_GBS} GB/s → "
            f"IB {ib_bw_node} GB/s "
            f"({_cliff_ratio:.1f}× bandwidth cliff)"
        )

    _failure_banner = mo.Html("")
    if _oom:
        _failure_banner = mo.callout(mo.md(
            f"**Communication Bottleneck — Configuration Impractical.** "
            f"AllReduce requires {total_comm_s:.1f}s; compute step is {comp_time2_s:.2f}s. "
            f"**{overhead2_pct:.0f}% overhead** means the cluster spends {overhead2_pct:.0f}% of every "
            f"compute step waiting for gradient synchronization. "
            f"At this configuration, adding more nodes makes training *slower* in wall-clock time. "
            f"Remedies: reduce model size, increase IB links per node, or switch to pipeline parallelism "
            f"(which does not synchronize full gradients across node boundaries)."
        ), kind="danger")

    mo.vstack([
        mo.Html(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0;
                    border-radius: 12px; padding: 18px 22px; margin: 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Physics
            </div>
            <div style="font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem;
                        color: #1e293b; line-height: 2.0;">
                Intra-node comm = 2 × {(N_gpu_node-1)}/{N_gpu_node} × {M_gb:.0f} GB ÷ {NVLINK4_BW_GBS} GB/s = <strong>{intra_comm_s:.3f} s</strong> (NVLink)<br>
                Inter-node comm = 2 × {M_gb:.0f} GB ÷ {ib_bw_node} GB/s = <strong style="color:{ovhd2_color}">{inter_comm_s:.3f} s</strong> (IB {ib_bw_node} GB/s)<br>
                Total comm = {intra_comm_s:.3f} + {inter_comm_s:.3f} = <strong style="color:{ovhd2_color}">{total_comm_s:.3f} s</strong><br>
                Compute = <strong>{comp_time2_s:.3f} s</strong> | Overhead = <strong style="color:{ovhd2_color}">{overhead2_pct:.1f}%</strong><br>
                {"<strong style='color:#CB202D;'>"+_bandwidth_cliff_note+"</strong>" if _bandwidth_cliff_note else ""}
            </div>
        </div>
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 4px 0 12px 0;">
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.08em;">
                    Throughput Speedup
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {eff2_color};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {actual_speedup:.1f}×
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">
                    Ideal: {ideal_speedup:.0f}×
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.08em;">
                    Scaling Efficiency
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {eff2_color};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {scaling_eff_pct:.1f}%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">
                    vs ideal linear
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.08em;">
                    Comm Overhead
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {ovhd2_color};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {overhead2_pct:.1f}%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">
                    {N_gpu_total} total GPUs
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.08em;">
                    IB Bandwidth
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {COLORS['BlueLine']};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {ib_bw_node} GB/s
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">
                    {ib_links.value} link(s) × {IB_HDR200_BW_GBS} GB/s
                </div>
            </div>
        </div>
        """),
        _failure_banner,
        mo.md("### Scaling Curve: Actual vs. Ideal"),
        mo.as_html(fig2),
        mo.callout(mo.md(
            f"**Bandwidth cliff at node boundary.** "
            f"Intra-node AllReduce runs at {NVLINK4_BW_GBS} GB/s (NVLink). "
            f"Inter-node AllReduce runs at {ib_bw_node} GB/s (InfiniBand × {ib_links.value}). "
            f"The cliff ratio is {NVLINK4_BW_GBS / ib_bw_node:.1f}×. "
            "As you add nodes, each additional node does not add proportional compute — "
            "it adds proportional gradient volume that must cross the lower-bandwidth inter-node fabric. "
            "This is the source of the scaling wall."
        ), kind="info"),
    ])
    return (
        actual_speedup, ideal_speedup, scaling_eff_pct,
        overhead2_pct, total_comm_s, comp_time2_s,
        N_nodes, N_gpu_total, act2_model_b,
    )


# ─── ACT II: PREDICTION-VS-REALITY OVERLAY ───────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, act2_pred,
    actual_speedup, ideal_speedup, scaling_eff_pct,
    overhead2_pct,
    act1_reflect,
):
    mo.stop(act1_reflect.value is None)
    mo.stop(act2_pred.value is None)

    _letter2 = act2_pred.value[0]
    _correct2 = _letter2 == "C"

    _gap_map = {
        "A": (
            "You predicted distributed PyTorch framework overhead. "
            "Framework overhead is real — it typically costs 5–15% of step time — but it does not "
            "explain a 2.7× shortfall from ideal. The actual cause is the **18× bandwidth cliff** "
            "when crossing from NVLink (900 GB/s) to InfiniBand (~50 GB/s effective per port). "
            "Framework profiling would show the GPU sitting idle waiting for AllReduce — not executing Python."
        ),
        "B": (
            "You predicted InfiniBand latency (per-hop microseconds). "
            "InfiniBand latency is ~1 µs per hop — at 64 nodes, this is ~6 µs total. "
            "Against a 2+ second gradient sync, 6 µs is negligible (0.0003% overhead). "
            "The bottleneck is **bandwidth**, not latency. The 280 GB of gradient data "
            "must flow through ~50 GB/s of effective InfiniBand bandwidth per port — and that takes seconds, not microseconds."
        ),
        "C": (
            f"Correct. The bandwidth cliff at the node boundary is the root cause. "
            f"Scaling from 1 node (8 GPUs, NVLink 900 GB/s) to 16 nodes introduces inter-node "
            "AllReduce at ~50 GB/s effective IB bandwidth — an 18× drop from NVLink for the inter-node phase. "
            f"At 70B parameters, this dominates the total communication time. "
            f"The scaling curve shows {scaling_eff_pct:.0f}% efficiency against ideal: "
            f"approximately {actual_speedup:.1f}× actual vs {ideal_speedup:.0f}× expected."
        ),
        "D": (
            "You predicted the topology's bisection bandwidth is exceeded. "
            "A properly provisioned fat-tree network does not have a fixed bisection bandwidth cap "
            "for 128 GPUs — it scales. The bottleneck is the per-node IB link count (1–8 links), "
            "not the topology itself. The scaling wall would appear even on an ideal non-blocking "
            "fabric, because the fundamental issue is the NVLink→IB bandwidth drop at the node boundary."
        ),
    }

    _kind2 = "success" if _correct2 else "warn"
    mo.callout(mo.md(
        f"**Prediction vs. Reality — Act II**\n\n"
        f"You predicted option {_letter2}. {_gap_map[_letter2]}\n\n"
        f"**At 16 nodes / 70B / 1 IB link per node:**\n"
        f"- Actual speedup: **{actual_speedup:.1f}×** vs ideal **{ideal_speedup:.0f}×**\n"
        f"- Scaling efficiency: **{scaling_eff_pct:.0f}%**\n"
        f"- Communication overhead: **{overhead2_pct:.1f}%**\n"
        f"- The inter-node AllReduce is the bottleneck, not software or topology."
    ), kind=_kind2)
    return


# ─── ACT II: REFLECTION ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(act1_reflect.value is None)
    mo.md("---")
    return


@app.cell(hide_code=True)
def _(mo, act1_reflect, act2_pred):
    mo.stop(act1_reflect.value is None)
    mo.stop(act2_pred.value is None)

    act2_reflect = mo.ui.radio(
        options={
            "A) Use faster InfiniBand HDR400 — double the inter-node bandwidth from 400 to 800 GB/s": "A",
            "B) Switch to pipeline parallelism — only pipeline-boundary activations cross node boundaries, not full gradients": "B",
            "C) Reduce model size to fit within a single NVLink node — eliminate inter-node traffic entirely": "C",
            "D) Increase batch size — fewer sync steps per epoch reduces total gradient communication": "D",
        },
        label="""**Reflection.** The inter-node bandwidth bottleneck is structural.
Given that NVLink within a node is 900 GB/s but InfiniBand across nodes is 400 GB/s
(a 2.25× cliff at best, 18× for PCIe), what is the most effective architectural remedy?""",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect, act2_pred, act2_reflect):
    mo.stop(act1_reflect.value is None)
    mo.stop(act2_pred.value is None)
    mo.stop(
        act2_reflect.value is None,
        mo.vstack([
            act2_reflect,
            mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"),
        ])
    )

    _reflect2_feedback = {
        "A": mo.callout(mo.md(
            "**Partially correct, but insufficient.** HDR400 doubles bandwidth to 800 GB/s, "
            "closing the gap from 2.25× to 1.125× vs NVLink. For modest-scale training "
            "(16–32 nodes, models up to 30B), this can be effective. But it does not eliminate "
            "the architectural mismatch: you are still synchronizing full gradient tensors across "
            "a lower-bandwidth inter-node fabric. At 70B+ parameters and 64+ nodes, even HDR400 "
            "produces unacceptable overhead. It is a bandwidth tax, not an architectural remedy."
        ), kind="warn"),
        "B": mo.callout(mo.md(
            "**Correct.** Pipeline parallelism partitions the model across nodes by **layers**, "
            "not by data. Each node holds a contiguous slice of the model's layers. "
            "The data that crosses node boundaries is not gradients (model_size × 2) "
            "but **pipeline activations**: one micro-batch of activations flowing forward, "
            "and one of gradients flowing backward. Activation volume is typically "
            "batch_size × seq_len × hidden_dim × 2 bytes — orders of magnitude smaller than "
            "full model gradients. The inter-node bandwidth constraint is reduced from "
            "`2 × model_size / IB_bw` to `2 × activation_volume / IB_bw`. "
            "This is the architectural insight behind how Megatron-LM and GPT-4-scale systems "
            "cross node boundaries without a bandwidth wall."
        ), kind="success"),
        "C": mo.callout(mo.md(
            "**Correct premise, wrong conclusion.** Fitting everything on a single NVLink node "
            "does eliminate inter-node traffic — but it limits you to models that fit in "
            "8 × 80 GB = 640 GB of HBM. A 70B FP16 model is 140 GB; a 175B model is 350 GB. "
            "At 70B you fit; at 175B you do not. More importantly, this is not a remedy for "
            "multi-node scaling — it is a retreat from it. The question is how to scale, "
            "not how to avoid scaling."
        ), kind="warn"),
        "D": mo.callout(mo.md(
            "**Not the most effective remedy.** Increasing batch size does reduce sync frequency "
            "per epoch, but it does not reduce the per-sync gradient volume — each AllReduce "
            "still transfers 2 × model_size bytes. Larger batches also risk statistical efficiency "
            "degradation (larger batches require more learning rate tuning to avoid accuracy loss). "
            "The inter-node bandwidth bottleneck is a function of gradient volume, "
            "not sync frequency."
        ), kind="warn"),
    }

    mo.vstack([
        act2_reflect,
        _reflect2_feedback[act2_reflect.value[0]],
    ])
    return


# ─── ACT II: MATH PEEK ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(act1_reflect.value is None)
    mo.accordion({
        "The Governing Equation — Hierarchical AllReduce at Scale": mo.md("""
        **Hierarchical AllReduce Time**

        For a cluster of N_nodes nodes, each with N_gpu GPUs connected by NVLink,
        and inter-node InfiniBand:

        ```
        T_intra = 2 × (N_gpu - 1)/N_gpu × M_bytes / BW_nvlink
        T_inter = 2 × M_bytes / (BW_ib × N_ib_links)
        T_allreduce = T_intra + T_inter
        ```

        - **T_intra** — intra-node reduce time (NVLink, fast path)
        - **T_inter** — inter-node reduce time (InfiniBand, bottleneck)
        - **M_bytes** — model size in bytes (N_params × bytes_per_param)
        - **BW_nvlink** — NVLink4 bidirectional bandwidth, 900 GB/s
        - **BW_ib** — InfiniBand per-port bandwidth, 400 GB/s
        - **N_ib_links** — number of IB ports per node (1–8)

        **Scaling Efficiency**

        ```
        T_step_1node   = T_compute + T_intra_1node
        T_step_Nnodes  = T_compute + T_allreduce
        actual_speedup = (T_step_1node / T_step_Nnodes) × N_nodes
        scaling_eff    = actual_speedup / N_nodes
        ```

        **The Bandwidth Cliff**

        | Phase | Bandwidth | Volume (70B FP16) | Time |
        |---|---|---|---|
        | Intra-node (NVLink) | 900 GB/s | ~122.5 GB | 0.14 s |
        | Inter-node (1× IB) | 400 GB/s | ~8.75 GB | 0.35 s |
        | Inter-node (8× IB) | 3200 GB/s | ~8.75 GB | 0.04 s |

        Adding more IB links per node is the hardware remedy within data-parallel training.
        Switching to pipeline parallelism is the architectural remedy that eliminates
        the gradient-crossing requirement altogether.

        **Pipeline Parallelism Volume Comparison**

        ```
        AllReduce activation volume = 2 × batch × seq_len × hidden_dim × 2 bytes
        ```
        For batch=32, seq=2048, hidden=8192 (70B-class):
        ```
        2 × 32 × 2048 × 8192 × 2 = ~2.15 GB
        ```
        Pipeline boundary traffic (**~2 GB**) vs. data-parallel gradient traffic (**280 GB**) —
        a 130× reduction in inter-node data movement.
        """),
    })
    return


# ═════════════════════════════════════════════════════════════════════════════
# LEDGER SAVE + HUD FOOTER
# ═════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(
    mo, ledger,
    COLORS,
    act1_pred, act2_pred, act2_reflect,
    context_toggle,
    overhead_pct, efficiency,
    actual_speedup, ideal_speedup, scaling_eff_pct,
    overhead2_pct, N_nodes, N_gpu_total,
    act2_model_b,
    act1_reflect,
):
    # Only save when Act II reflection is answered
    _act1_done  = act1_pred.value is not None
    _act1_corr  = (act1_pred.value or "")[0] == "C"
    _act2_done  = act2_pred.value is not None
    _act2_corr  = (act2_pred.value or "")[0] == "C"
    _ref1_done  = act1_reflect.value is not None
    _ref1_corr  = (act1_reflect.value or "")[0] == "B"
    _ref2_done  = act2_reflect.value is not None and act1_reflect.value is not None

    _ctx        = context_toggle.value
    _interconnect = "nvlink" if _ctx == "single_node" else "infiniband"
    _constraint_hit = overhead_pct > 100.0 or overhead2_pct > 100.0

    if _act2_done and _ref1_done:
        ledger.save(
            chapter="v2_02",
            design={
                "context":            _ctx,
                "interconnect":       _interconnect,
                "nodes":              N_nodes,
                "model_params_b":     float(act2_model_b.value),
                "comm_overhead_pct":  float(overhead2_pct),
                "act1_prediction":    act1_pred.value or "",
                "act1_correct":       _act1_corr,
                "act2_result":        float(scaling_eff_pct),
                "act2_decision":      act2_reflect.value or "",
                "constraint_hit":     _constraint_hit,
            }
        )

    # ── HUD footer ─────────────────────────────────────────────────────────
    def _hud_badge(label, value, active):
        _color = COLORS["GreenLine"] if active else COLORS["RedLine"]
        _bg    = COLORS["GreenLL"] if active else COLORS["RedLL"]
        return f"""
        <div style="display:flex; flex-direction:column; gap:2px; min-width:100px;">
            <div style="font-size:0.68rem; font-weight:700; color:#94a3b8;
                        text-transform:uppercase; letter-spacing:0.1em;">{label}</div>
            <div style="font-size:0.88rem; font-weight:700; color:{_color};
                        font-family:'SF Mono','Fira Code',monospace;">{value}</div>
        </div>
        """

    _progress_pct = (
        (1 if _act1_done else 0) +
        (1 if _ref1_done else 0) +
        (1 if _act2_done else 0) +
        (1 if _ref2_done else 0)
    ) / 4 * 100

    _hud_items = [
        _hud_badge("Context",     _ctx.replace("_", "-"), True),
        _hud_badge("Act I Pred",  (act1_pred.value or "—")[:1], _act1_corr),
        _hud_badge("Reflect I",   (act1_reflect.value or "—")[:1], _ref1_corr),
        _hud_badge("Act II Pred", (act2_pred.value or "—")[:1] if _act2_done else "—", _act2_corr),
        _hud_badge("Scaling Eff", f"{scaling_eff_pct:.0f}%" if _act2_done else "—", scaling_eff_pct >= 50 if _act2_done else False),
        _hud_badge("Lab Progress", f"{_progress_pct:.0f}%", _progress_pct >= 75),
    ]

    mo.Html(f"""
    <div style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap;
                padding: 14px 24px; background: #0f172a; border-radius: 12px;
                margin-top: 32px; border: 1px solid #1e293b;">
        <div style="font-size: 0.72rem; font-weight: 800; color: #475569;
                    text-transform: uppercase; letter-spacing: 0.14em; white-space: nowrap;
                    border-right: 1px solid #1e293b; padding-right: 16px; margin-right: 4px;">
            Design Ledger · v2_02
        </div>
        {"".join(_hud_items)}
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
