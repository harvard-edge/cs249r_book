import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 03: THE BISECTION BANDWIDTH WALL
#
# Chapter: network_fabrics.qmd  (@sec-network-fabrics)
# Volume II · Lab 03
#
# Core Invariant: Fat-tree topology provides full bisection bandwidth.
#   As clusters scale, bisection bandwidth per GPU decreases unless the fabric
#   is properly overprovisioned. The bisection bandwidth wall is why not all
#   topologies scale equally.
#
# Two Acts:
#   Act I  — The Bisection Bandwidth Blindspot (12–15 min)
#     A 128-GPU cluster uses 2:1 oversubscription. AllReduce is 40% of expected.
#     Why? Bisection bandwidth halved by oversubscription explains the drop.
#
#   Act II — The 1024-GPU Fabric Design (20–25 min)
#     $50M budget, 3 fabric options: IB fat-tree, Eth fat-tree, 3D-torus.
#     AllReduce at scale is dominated by bisection bandwidth, not per-link speed.
#     Failure state: bisection BW insufficient → AllReduce > 1-second target.
#
# Contexts: 8-GPU cluster (single-node reference) vs 1024-GPU cluster (scale)
#
# Design Ledger: chapter="v2_03", context, fabric_type, cluster_size,
#                oversubscription, bisection_bw_gbps, act1_prediction,
#                act1_correct, act2_result, act2_decision, constraint_hit
#
# Hardware constants (source in comments on each constant):
#   IB_HDR200_PORT_GBPS  = 200   # InfiniBand HDR200 single port, Gb/s
#   IB_NDR400_PORT_GBPS  = 400   # InfiniBand NDR400 single port, Gb/s
#   ETH_100G_PORT_GBPS   = 100   # 100GbE port line rate, Gb/s
#   ETH_400G_PORT_GBPS   = 400   # 400GbE port line rate, Gb/s
#   IB_HDR200_EFF_GBS    = 22.5  # InfiniBand HDR200 effective unidirectional GB/s
#   IB_NDR400_EFF_GBS    = 45.0  # InfiniBand NDR400 effective unidirectional GB/s
#   ETH_100G_EFF_GBS     = 11.0  # 100GbE effective GB/s after TCP/IP overhead
#   ETH_400G_EFF_GBS     = 44.0  # 400GbE effective GB/s after overhead
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

    # ── Hardware constants (source: vendor datasheets + @sec-network-fabrics) ──

    # InfiniBand port rates — NVIDIA/Mellanox datasheet
    IB_HDR200_PORT_GBPS = 200   # Gb/s per port, InfiniBand HDR200 (HDR)
    IB_NDR400_PORT_GBPS = 400   # Gb/s per port, InfiniBand NDR (NDR400)

    # Ethernet port rates — IEEE 802.3 standards
    ETH_100G_PORT_GBPS  = 100   # Gb/s per port, 100GbE
    ETH_400G_PORT_GBPS  = 400   # Gb/s per port, 400GbE

    # Effective unidirectional bandwidth (Gb/s → GB/s with protocol overhead)
    # InfiniBand: ~90% efficiency (minimal overhead), source: MLPerf network BW
    IB_HDR200_EFF_GBS   = 22.5  # GB/s unidirectional (200 Gbps × 0.9 / 8)
    IB_NDR400_EFF_GBS   = 45.0  # GB/s unidirectional (400 Gbps × 0.9 / 8)

    # Ethernet: ~88% efficiency after TCP/UDP overhead
    ETH_100G_EFF_GBS    = 11.0  # GB/s unidirectional (100 Gbps × 0.88 / 8)
    ETH_400G_EFF_GBS    = 44.0  # GB/s unidirectional (400 Gbps × 0.88 / 8)

    # Cost estimates for 1024-GPU fabric (Act II), source: industry white papers
    # These are order-of-magnitude estimates for pedagogical framing
    IB_NDR_FABRIC_COST_M  = 20.0   # $M for 1024-GPU IB NDR fat-tree, non-blocking
    ETH_400G_FABRIC_COST_M = 8.0   # $M for 1024-GPU 400GbE fat-tree, 2:1 oversub
    TORUS_3D_FABRIC_COST_M = 5.0   # $M for 1024-GPU 3D-torus (fixed links)

    # AllReduce ring-allreduce formula constant
    # Ring AllReduce time = 2 * (N-1)/N * message_size / bisection_bw
    # For large N: approaches 2 * message_size / bisection_bw
    # Source: @sec-network-fabrics-allreduce-algorithms (Rabenseifner's algorithm)

    ledger = DesignLedger()

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        IB_HDR200_PORT_GBPS, IB_NDR400_PORT_GBPS,
        ETH_100G_PORT_GBPS, ETH_400G_PORT_GBPS,
        IB_HDR200_EFF_GBS, IB_NDR400_EFF_GBS,
        ETH_100G_EFF_GBS, ETH_400G_EFF_GBS,
        IB_NDR_FABRIC_COST_M, ETH_400G_FABRIC_COST_M, TORUS_3D_FABRIC_COST_M,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

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
                Machine Learning Systems · Volume II · Lab 03
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Bisection Bandwidth Wall
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                BW_bisection = N × link_BW / (2 × oversubscription_ratio)
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 700px; line-height: 1.65;">
                Fat-tree topology promises full bisection bandwidth — every GPU reaches
                every other GPU at full link speed. That promise holds only when the fabric
                is non-blocking. Oversubscription quietly halves it. At 1024 GPUs, the choice
                of topology determines whether AllReduce finishes in 200 ms or 4 seconds.
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
                    Chapter: @sec-network-fabrics
                </span>
                <span style="background: rgba(0,143,69,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(0,143,69,0.25);">
                    Instrument: Bisection BW Calculator
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Bisection Bandwidth</span>
                <span class="badge badge-warn">Oversubscription</span>
                <span class="badge badge-fail">AllReduce Bottleneck</span>
                <span class="badge badge-ok">Fat-Tree Non-Blocking</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Identify the alpha-beta crossover: predict whether upgrading InfiniBand from 200 Gbps to 400 Gbps improves transfer time for a 4 KB control message, and calculate where the crossover from latency-dominated to bandwidth-dominated regime occurs (~75 KB).</strong></div>
                <div style="margin-bottom: 3px;">2. <strong>Quantify the oversubscription tax: measure how a 4:1 oversubscribed spine reduces bisection bandwidth from 25.6 TB/s to 6.4 TB/s, making every global AllReduce 4&times; slower.</strong></div>
                <div style="margin-bottom: 3px;">3. <strong>Design a fabric topology: determine the maximum oversubscription ratio that keeps AllReduce below 30% of step time for a 70B model on a 1,024-GPU cluster.</strong></div>
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
                    Alpha-beta latency model T(n) = &alpha; + n/&beta; from @sec-network-fabrics-performance-model &middot;
                    Bisection bandwidth definition and fat-tree construction from @sec-network-fabrics-fat-tree
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
                "If you upgrade your InfiniBand links from 200 Gbps to 400 Gbps, which workloads actually benefit &mdash; and why does oversubscribing the spine tier make even your 400 Gbps investment worthless for large-model AllReduce?"
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

    - **@sec-network-fabrics-bisection-bandwidth** (Bisection Bandwidth) — The minimum-cut
      definition, why it sets the all-to-all communication ceiling, and the fat-tree formula.
    - **@sec-network-fabrics-fat-tree-topology** (Fat-Tree Structure) — k-ary fat-tree
      construction, oversubscription ratios, and the non-blocking guarantee.
    - **@sec-network-fabrics-allreduce** (AllReduce Algorithms) — Ring-AllReduce bandwidth
      analysis and why bisection bandwidth is the bottleneck for large clusters.
    - **@sec-network-fabrics-topology-comparison** (Topology Trade-offs) — Fat-tree vs
      3D-torus bisection bandwidth scaling: O(N) vs O(N^(2/3)).
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE ───────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "8-GPU Cluster (single-node reference scale)":    "small",
            "1024-GPU Cluster (production training scale)":   "large",
        },
        value="8-GPU Cluster (single-node reference scale)",
        label="Cluster context (sets scale for all computations):",
        inline=True,
    )
    context_toggle
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER (hide_code=True) ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "I"
    _act_color    = COLORS["BlueLine"]
    _act_title    = "The Bisection Bandwidth Blindspot"
    _act_duration = "12&ndash;15 min"
    _act_why      = (
        "You expect a bandwidth upgrade from 200 Gbps to 400 Gbps to halve transfer times. "
        "The alpha-beta model will show that for a 4 KB pipeline coordination message, the "
        "improvement is only 5% &mdash; startup latency dominates below the 75 KB crossover, "
        "and upgrading bandwidth in the latency-dominated regime wastes engineering budget."
    )
    mo.Html(f"""
    <div style="margin: 24px 0 12px 0;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="background:{_act_color}; color:white; border-radius:50%;
                         width:32px; height:32px; display:inline-flex; align-items:center;
                         justify-content:center; font-size:0.9rem; font-weight:800;
                         flex-shrink:0;">{_act_num}</div>
            <div style="flex:1; height:2px; background:{COLORS['Border']};"></div>
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.12em;">
                Act {_act_num} &middot; {_act_duration}
            </div>
        </div>
        <div style="font-size:1.5rem; font-weight:800; color:{COLORS['Text']};
                    margin-top:8px; line-height:1.2;">
            {_act_title}
        </div>
        <div style="color:{COLORS['TextSec']}; font-size:0.92rem; margin-top:6px;
                    line-height:1.55; max-width:700px;">
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
            Incoming Message · Network Architect, Meridian AI Infrastructure
        </div>
        <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
            "We commissioned a new 128-GPU training cluster. To reduce fabric cost we used
            a 2:1 oversubscription on the spine switches — half the uplinks of a full fat-tree.
            The vendor assured us this is standard for most workloads. But our distributed
            training AllReduce is running at 40% of the throughput we calculated. The GPUs are
            idle 60% of the time waiting on gradients. What did we get wrong?"
        </div>
        <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
            — Kenji Watanabe, Network Architect · Meridian AI Infrastructure (128-GPU cluster)
        </div>
    </div>
    """)
    return


# ─── ACT I: CONCEPT SETUP ─────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("""
        ## Bisection Bandwidth is the All-to-All Ceiling

        **Bisection bandwidth** is the minimum bandwidth when you cut a network into two
        equal halves. It is the theoretical ceiling for any all-to-all communication pattern
        — the kind AllReduce uses when every GPU must exchange gradients with every other GPU.

        For a **fat-tree with no oversubscription** (non-blocking), the bisection bandwidth is:

        ```
        BW_bisection = N × link_BW / 2
        ```

        Where `N` is the number of GPUs and `link_BW` is the bandwidth of each GPU uplink.
        Every GPU can communicate at full link speed simultaneously — "no blocking" means
        no link is ever the bottleneck.

        When **oversubscription ratio `r`** is applied (e.g., 2:1 means half the spine
        uplinks are removed), bisection bandwidth is cut by the same factor:

        ```
        BW_bisection (oversubscribed) = N × link_BW / (2 × r)
        ```

        A 2:1 oversubscription **halves bisection bandwidth**. AllReduce throughput for
        an all-to-all workload is bounded by bisection bandwidth, so performance is also
        halved — not slightly degraded.
        """),
        mo.callout(mo.md(
            "**The 40% observation.** A 2:1 oversubscription halves bisection bandwidth. "
            "If AllReduce efficiency was ~80% of theoretical on a non-blocking fabric, "
            "it becomes ~40% on the oversubscribed fabric — exactly the observation above. "
            "The GPU idleness is not a software bug. It is a physics constraint."
        ), kind="info"),
    ])
    return


# ─── ACT I: PREDICTION LOCK ───────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A) 40% of expected is close enough — oversubscription rarely matters for ML workloads": "option_a",
            "B) The 2:1 oversubscription halves bisection bandwidth — all-to-all patterns see 50% of expected throughput": "option_b",
            "C) The issue is latency, not bandwidth — too many hops in the fat-tree add queueing delay": "option_c",
            "D) 128 GPUs exceeds the practical scaling limit for fat-tree topologies": "option_d",
        },
        label="""**Prediction Lock — Act I.**
A 128-GPU cluster uses InfiniBand HDR200 (200 Gbps per port) with a 2:1 oversubscription
ratio at the spine layer. A non-blocking fat-tree of the same cluster would have
bisection bandwidth = 128 × 25 GB/s / 2 = 1,600 GB/s total.

The AllReduce throughput is 40% of expected. Which explanation best accounts for this?""",
    )
    act1_prediction
    return (act1_prediction,)


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(
        act1_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Bisection Bandwidth instruments."),
            kind="warn",
        )
    )
    return


# ─── ACT I: INSTRUMENTS ───────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(act1_prediction.value is None)

    act1_cluster_size = mo.ui.slider(
        start=8, stop=512, value=128, step=8,
        label="Cluster size N (GPUs)",
    )
    act1_link_bw = mo.ui.dropdown(
        options={
            "IB HDR200 — 200 Gbps (22.5 GB/s eff)": "ib_hdr200",
            "IB NDR400 — 400 Gbps (45.0 GB/s eff)": "ib_ndr400",
            "100GbE    — 100 Gbps (11.0 GB/s eff)": "eth_100g",
            "400GbE    — 400 Gbps (44.0 GB/s eff)": "eth_400g",
        },
        value="IB HDR200 — 200 Gbps (22.5 GB/s eff)",
        label="Link type (per GPU uplink)",
    )
    act1_oversub = mo.ui.dropdown(
        options={
            "1:1 — Non-blocking (full fat-tree)": 1.0,
            "2:1 — Half spine uplinks":           2.0,
            "4:1 — Quarter spine uplinks":        4.0,
        },
        value="1:1 — Non-blocking (full fat-tree)",
        label="Oversubscription ratio",
    )

    mo.vstack([
        mo.md("### Bisection Bandwidth Calculator"),
        mo.hstack([act1_cluster_size, act1_link_bw], justify="start", gap="2rem"),
        act1_oversub,
    ])
    return (act1_cluster_size, act1_link_bw, act1_oversub)


@app.cell(hide_code=True)
def _(
    mo, act1_prediction, act1_cluster_size, act1_link_bw, act1_oversub,
    go, apply_plotly_theme, COLORS,
    IB_HDR200_EFF_GBS, IB_NDR400_EFF_GBS, ETH_100G_EFF_GBS, ETH_400G_EFF_GBS,
):
    mo.stop(act1_prediction.value is None)

    # ── Link bandwidth lookup ─────────────────────────────────────────────────
    _link_map = {
        "ib_hdr200": IB_HDR200_EFF_GBS,
        "ib_ndr400": IB_NDR400_EFF_GBS,
        "eth_100g":  ETH_100G_EFF_GBS,
        "eth_400g":  ETH_400G_EFF_GBS,
    }
    _link_label_map = {
        "ib_hdr200": "IB HDR200",
        "ib_ndr400": "IB NDR400",
        "eth_100g":  "100GbE",
        "eth_400g":  "400GbE",
    }
    _link_bw_gbs = _link_map[act1_link_bw.value]
    _link_label  = _link_label_map[act1_link_bw.value]
    _N           = act1_cluster_size.value
    _r           = act1_oversub.value

    # ── Bisection bandwidth formula ───────────────────────────────────────────
    # BW_bisection = N × link_BW / (2 × oversubscription_ratio)
    # Source: @sec-network-fabrics-bisection-bandwidth
    _bw_full      = _N * _link_bw_gbs / 2.0          # GB/s, non-blocking
    _bw_oversub   = _N * _link_bw_gbs / (2.0 * _r)   # GB/s, with oversubscription
    _bw_per_gpu   = _bw_oversub / _N                   # GB/s per GPU (bisection share)

    # ── AllReduce bandwidth efficiency ────────────────────────────────────────
    # Ring-AllReduce sends 2*(N-1)/N * message_size bytes total per GPU.
    # For large N, effective per-GPU AllReduce bandwidth ≈ bisection_bw / N.
    # Efficiency relative to non-blocking fabric:
    _efficiency_pct = (_bw_oversub / _bw_full) * 100.0 if _bw_full > 0 else 0.0

    # ── Bottleneck classification ─────────────────────────────────────────────
    if _r <= 1.0:
        _fabric_status   = "Non-blocking"
        _status_color    = COLORS["GreenLine"]
        _status_bg       = COLORS["GreenLL"]
    elif _r <= 2.0:
        _fabric_status   = "2:1 Oversubscribed"
        _status_color    = COLORS["OrangeLine"]
        _status_bg       = COLORS["OrangeLL"]
    else:
        _fabric_status   = "4:1 Oversubscribed"
        _status_color    = COLORS["RedLine"]
        _status_bg       = COLORS["RedLL"]

    # ── Comparison bar chart: topology bandwidth comparison ───────────────────
    _oversub_labels  = ["1:1 (Non-blocking)", "2:1 Oversubscribed", "4:1 Oversubscribed"]
    _oversub_ratios  = [1.0, 2.0, 4.0]
    _bw_values       = [_N * _link_bw_gbs / (2.0 * r) for r in _oversub_ratios]
    _bar_colors      = [COLORS["GreenLine"], COLORS["OrangeLine"], COLORS["RedLine"]]
    _selected_idx    = _oversub_ratios.index(_r)

    _fig = go.Figure()

    _fig.add_trace(go.Bar(
        x=_oversub_labels,
        y=_bw_values,
        marker_color=[
            _bar_colors[i] if i != _selected_idx else "#1e293b"
            for i in range(len(_bar_colors))
        ],
        marker_line_color=[
            _bar_colors[i] for i in range(len(_bar_colors))
        ],
        marker_line_width=[
            3 if i == _selected_idx else 1 for i in range(len(_bar_colors))
        ],
        opacity=[0.55 if i != _selected_idx else 1.0 for i in range(3)],
        text=[f"{v:,.0f} GB/s" for v in _bw_values],
        textposition="outside",
        textfont=dict(size=11, family="SF Mono, monospace"),
        width=0.55,
    ))

    # Highlight selected configuration
    _fig.add_trace(go.Scatter(
        x=[_oversub_labels[_selected_idx]],
        y=[_bw_values[_selected_idx]],
        mode="markers",
        marker=dict(
            symbol="star",
            size=14,
            color=_bar_colors[_selected_idx],
            line=dict(color="white", width=1),
        ),
        name="Current config",
        showlegend=False,
    ))

    _fig.update_layout(
        title=dict(
            text=f"Bisection Bandwidth by Oversubscription — {_N}-GPU cluster, {_link_label}",
            font=dict(size=13, color="#1e293b"),
            x=0,
        ),
        height=320,
        yaxis=dict(title="Total Bisection Bandwidth (GB/s)", gridcolor="#f1f5f9"),
        xaxis=dict(title="Oversubscription Ratio"),
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    apply_plotly_theme(_fig)

    # ── Metric cards ──────────────────────────────────────────────────────────
    _cards_html = f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin:16px 0;">
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:175px; text-align:center; background:white;
                    border-top:3px solid {_status_color};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Total Bisection BW
            </div>
            <div style="font-size:1.35rem; font-weight:800; color:{_status_color};
                        font-family:'SF Mono',monospace;">
                {_bw_oversub:,.0f} GB/s
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                {_fabric_status}
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:175px; text-align:center; background:white;
                    border-top:3px solid {COLORS['BlueLine']};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Per-GPU BW Share
            </div>
            <div style="font-size:1.35rem; font-weight:800; color:{COLORS['BlueLine']};
                        font-family:'SF Mono',monospace;">
                {_bw_per_gpu:.2f} GB/s
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                of {_link_bw_gbs:.1f} GB/s link speed
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:175px; text-align:center; background:white;
                    border-top:3px solid {'#008F45' if _efficiency_pct >= 90 else '#CC5500' if _efficiency_pct >= 60 else '#CB202D'};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                AllReduce Efficiency
            </div>
            <div style="font-size:1.35rem; font-weight:800;
                        color:{'#008F45' if _efficiency_pct >= 90 else '#CC5500' if _efficiency_pct >= 60 else '#CB202D'};
                        font-family:'SF Mono',monospace;">
                {_efficiency_pct:.0f}%
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                vs non-blocking fabric
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:175px; text-align:center; background:white;
                    border-top:3px solid {COLORS['BlueLine']};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Non-blocking Peak
            </div>
            <div style="font-size:1.35rem; font-weight:800; color:{COLORS['TextSec']};
                        font-family:'SF Mono',monospace;">
                {_bw_full:,.0f} GB/s
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                1:1 fat-tree reference
            </div>
        </div>
    </div>
    """

    # ── Physics formula display ───────────────────────────────────────────────
    _formula_text = f"""
    **Bisection Bandwidth — Live Calculation** (N={_N} GPUs, link={_link_bw_gbs:.1f} GB/s, r={_r:.0f}:1)

    ```
    BW_bisection  =  N × link_BW / (2 × r)
                  =  {_N} × {_link_bw_gbs:.1f} GB/s / (2 × {_r:.0f})
                  =  {_bw_oversub:,.1f} GB/s   ← {_fabric_status}

    Non-blocking  =  {_N} × {_link_bw_gbs:.1f} GB/s / 2
                  =  {_bw_full:,.1f} GB/s   ← full fat-tree

    AllReduce efficiency  =  {_bw_oversub:,.1f} / {_bw_full:,.1f}
                          =  {_efficiency_pct:.1f}%   ← degradation from oversubscription
    ```
    """

    mo.vstack([
        mo.Html(_cards_html),
        mo.md(_formula_text),
        mo.ui.plotly(_fig),
    ])
    return (
        _bw_oversub, _bw_full, _efficiency_pct,
        _N, _r, _link_bw_gbs, _link_label,
    )


# ─── ACT I: PREDICTION VS REALITY OVERLAY ─────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, _bw_oversub, _bw_full, _efficiency_pct, _r):
    mo.stop(act1_prediction.value is None)

    # Prediction-vs-reality overlay
    # The 2:1 oversubscription halves bisection BW → exactly option_b
    _predicted_pct = {
        "option_a": 90.0,   # Student predicted oversubscription "rarely matters"
        "option_b": 50.0,   # Student predicted 50% — correct physics
        "option_c": 75.0,   # Student predicted latency (hops), not BW reduction
        "option_d": 20.0,   # Student predicted capacity limit, not oversubscription
    }[act1_prediction.value]

    _actual_pct   = _efficiency_pct
    _gap          = abs(_actual_pct - _predicted_pct)
    _is_correct   = act1_prediction.value == "option_b"

    if _is_correct:
        _overlay = mo.callout(mo.md(
            f"**Correct.** Your prediction of {_predicted_pct:.0f}% matches the physics. "
            f"The actual AllReduce efficiency for {_r:.0f}:1 oversubscription is "
            f"**{_actual_pct:.0f}%** of a non-blocking fabric — "
            f"bisection bandwidth is cut by exactly the oversubscription ratio. "
            f"There is no partial degradation: the halved spine capacity directly halves "
            f"the cross-bisection bandwidth available for AllReduce."
        ), kind="success")
    else:
        _overlay = mo.callout(mo.md(
            f"**Not quite.** You predicted {_predicted_pct:.0f}% efficiency. "
            f"The actual value for {_r:.0f}:1 oversubscription is **{_actual_pct:.0f}%** "
            f"— off by {_gap:.0f} percentage points. "
            f"The correct answer is B: the 2:1 oversubscription **halves** bisection "
            f"bandwidth (from {_bw_full:,.0f} GB/s to {_bw_oversub:,.0f} GB/s). "
            f"AllReduce is an all-to-all pattern — its throughput is bounded by bisection "
            f"bandwidth, so the degradation is proportional, not incidental."
        ), kind="warn")

    _overlay
    return


# ─── ACT I: MATHPEEK ACCORDION ────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(act1_prediction.value is None)

    mo.accordion({
        "The governing equations: Fat-tree bisection bandwidth": mo.md("""
        **Bisection Bandwidth Formula (fat-tree):**

        ```
        BW_bisection = N × link_BW / (2 × r)
        ```

        - **N** — Number of end-hosts (GPUs) in the cluster
        - **link_BW** — Unidirectional bandwidth of each GPU uplink (GB/s)
        - **r** — Oversubscription ratio (1 = non-blocking, 2 = half spine, 4 = quarter spine)
        - **2** — Factor of 2 because bisection cuts the network in half;
          each half can send at most half the total uplink bandwidth

        **Non-blocking condition (r = 1):**

        ```
        BW_bisection_max = N × link_BW / 2
        ```

        A k-ary fat-tree has k/2 spine switches, each with k ports.
        Full non-blocking requires as many uplink ports at each layer as downlink ports.
        Oversubscription reduces spine uplinks, creating the bottleneck at the bisection cut.

        **AllReduce time (ring-allreduce, large N):**

        ```
        T_allreduce ≈ 2 × M / (BW_bisection / N)
                    = 2 × N × M / BW_bisection
        ```

        - **M** — Message size (gradient tensor, GB)
        - The "2" comes from the two phases of ring-allreduce (reduce-scatter + all-gather)

        **Scaling behavior by topology:**

        | Topology       | Bisection BW scales as | Notes                        |
        |----------------|------------------------|------------------------------|
        | Full fat-tree  | O(N)                   | Linear — ideal for AllReduce |
        | 2:1 fat-tree   | O(N/2)                 | Linear, but halved           |
        | 3D-torus       | O(N^(2/3))             | Sub-linear — poor at scale   |
        | Ring           | O(1) per node          | Fixed 2 links per GPU        |
        """),
    })
    return


# ─── ACT I: STRUCTURED REFLECTION ─────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(act1_prediction.value is None)

    mo.md("---")
    return


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(act1_prediction.value is None)

    act1_reflection = mo.ui.radio(
        options={
            "A) All switches have equal port count — k ports in, k ports out at every layer": "ref_a",
            "B) Every leaf-to-leaf path has the same bandwidth as a direct link — no bandwidth bottleneck at any layer": "ref_b",
            "C) The tree has exactly 3 layers: edge, aggregation, core": "ref_c",
            "D) Maximum path length is log₂(N) hops, minimizing queueing latency": "ref_d",
        },
        label="""**Reflection — Act I.**
What is the defining property of a non-blocking fat-tree topology?""",
    )
    act1_reflection
    return (act1_reflection,)


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None, mo.callout(
        mo.md("Select your answer to see the explanation."), kind="warn"
    ))

    _ref_correct = act1_reflection.value == "ref_b"

    _ref_feedback = {
        "ref_a": (
            "**Not quite.** Equal port counts is a property of symmetric switches "
            "but does not guarantee non-blocking behavior. A switch can have equal "
            "ingress and egress ports and still create bandwidth contention if "
            "aggregation-layer uplinks are fewer than downlinks — which is exactly "
            "what oversubscription does."
        ),
        "ref_b": (
            "**Correct.** A non-blocking fat-tree guarantees that any leaf-to-leaf "
            "path can sustain full link bandwidth simultaneously with all other "
            "leaf-to-leaf paths. This requires that at every layer, the total "
            "uplink capacity equals the total downlink capacity. When this holds, "
            "no switch layer is ever a bandwidth bottleneck — bisection bandwidth "
            "equals N × link_BW / 2, the theoretical maximum."
        ),
        "ref_c": (
            "**Not quite.** Fat-trees are commonly depicted with 3 layers "
            "(edge / aggregation / core), but this is a specific implementation "
            "choice, not the defining property. A fat-tree can have more layers. "
            "The defining property is the bandwidth guarantee at every cut, "
            "not the number of layers."
        ),
        "ref_d": (
            "**Not quite.** O(log N) path length is a property of fat-trees "
            "but is not what makes them non-blocking. Latency and bandwidth are "
            "independent properties. A topology can have short paths and still "
            "be bandwidth-bottlenecked at the bisection if uplink capacity is "
            "insufficient."
        ),
    }

    mo.callout(
        mo.md(_ref_feedback[act1_reflection.value]),
        kind="success" if _ref_correct else "warn",
    )
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER (hide_code=True) ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, COLORS):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)
    _act_num      = "II"
    _act_color    = COLORS["OrangeLine"]
    _act_title    = "The 1024-GPU Fabric Design"
    _act_duration = "20&ndash;25 min"
    _act_why      = (
        "Act I showed that bandwidth upgrades are irrelevant below the 75 KB crossover. "
        "Now apply the other side: at 1,024 GPU scale, AllReduce messages are hundreds of GB &mdash; "
        "firmly bandwidth-dominated. Here, oversubscribing the spine by 4:1 reduces bisection "
        "bandwidth by 4&times;, turning the network into the dominant bottleneck regardless "
        "of per-link speed."
    )
    mo.Html(f"""
    <div style="margin: 24px 0 12px 0;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="background:{_act_color}; color:white; border-radius:50%;
                         width:32px; height:32px; display:inline-flex; align-items:center;
                         justify-content:center; font-size:0.9rem; font-weight:800;
                         flex-shrink:0;">{_act_num}</div>
            <div style="flex:1; height:2px; background:{COLORS['Border']};"></div>
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.12em;">
                Act {_act_num} &middot; {_act_duration}
            </div>
        </div>
        <div style="font-size:1.5rem; font-weight:800; color:{COLORS['Text']};
                    margin-top:8px; line-height:1.2;">
            {_act_title}
        </div>
        <div style="color:{COLORS['TextSec']}; font-size:0.92rem; margin-top:6px;
                    line-height:1.55; max-width:700px;">
            {_act_why}
        </div>
    </div>
    """)
    return


# ─── ACT II: STAKEHOLDER MESSAGE ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, COLORS):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)

    _color = COLORS["OrangeLine"]
    _bg    = COLORS["OrangeL"]
    mo.Html(f"""
    <div style="border-left:4px solid {_color}; background:{_bg};
                border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
        <div style="font-size:0.72rem; font-weight:700; color:{_color};
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
            Incoming Message · Infrastructure VP, Apex Foundation Models
        </div>
        <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
            "We are building a 1024-GPU cluster for large language model pre-training.
            Total hardware budget: $50M, with $20–25M allocated to compute (GPUs).
            I have three fabric proposals on my desk:
            (A) 400G InfiniBand NDR fat-tree — $20M, 1:1 non-blocking;
            (B) 200G Ethernet fat-tree — $8M, 2:1 oversubscription;
            (C) 3D-torus — $5M, fixed bisection per plane.
            Our primary workload is Llama-class model training with large AllReduce passes
            (32 GB gradient tensors every iteration). Which fabric do I choose?"
        </div>
        <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
            — Dr. Amara Osei, VP Infrastructure · Apex Foundation Models (1024-GPU cluster)
        </div>
    </div>
    """)
    return


# ─── ACT II: CONCEPT SETUP ────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)

    mo.vstack([
        mo.md("""
        ## Three Fabrics, One Bottleneck

        Each fabric architecture has a different bisection bandwidth model at scale:

        **InfiniBand NDR fat-tree (1:1 non-blocking):**
        All 1024 GPU uplinks have matching spine capacity. Bisection bandwidth scales
        linearly with cluster size. The most expensive option per port, but no contention.

        **Ethernet fat-tree (2:1 oversubscribed):**
        Half the spine uplinks are removed to reduce cost. Bisection bandwidth is halved.
        AllReduce sees only 50% of the per-GPU link speed at the bisection.

        **3D-torus:**
        Each GPU connects to 6 neighbors in three dimensions. There is no central spine.
        Bisection bandwidth cuts one dimension of the torus — for a 1024-GPU cube,
        that is a 10×10 plane of links. Bisection bandwidth scales as O(N^(2/3)), not O(N).

        ```
        3D-torus bisection BW  =  (N)^(2/3) × link_BW
        Fat-tree bisection BW  =  N × link_BW / (2 × r)
        ```

        At small N, the 3D-torus can be competitive. At 1024 GPUs, the fat-tree
        advantage is pronounced. At 16,384 GPUs, the torus bisection bandwidth is
        roughly **16× lower** per GPU than a non-blocking fat-tree.
        """),
        mo.callout(mo.md(
            "**AllReduce at scale.** Every iteration of large-model training exchanges "
            "gradient tensors across all GPUs. For a 70B-parameter model in FP16, "
            "each AllReduce moves ~140 GB of data. The time to complete AllReduce "
            "determines whether GPUs are computing or waiting — it is the direct "
            "multiplier on total training time."
        ), kind="info"),
    ])
    return


# ─── ACT II: PREDICTION LOCK ──────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)

    act2_prediction = mo.ui.radio(
        options={
            "A) Option C — 3D-torus: cheapest option and bandwidth scales well for mesh workloads": "pred_torus",
            "B) Option B — Ethernet 2:1: good enough at 50% of IB performance for 40% of the cost": "pred_eth",
            "C) Option A — IB NDR fat-tree: AllReduce is all-to-all, bisection bandwidth is the bottleneck": "pred_ib",
            "D) All three topologies perform equally for AllReduce — link speed is all that matters": "pred_equal",
        },
        label="""**Prediction Lock — Act II.**
For a 1024-GPU cluster running 32 GB AllReduce tensors with a target AllReduce
completion time < 1 second, which fabric design should the Infrastructure VP choose?""",
    )
    act2_prediction
    return (act2_prediction,)


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)
    mo.stop(
        act2_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Fabric Design instruments."),
            kind="warn",
        )
    )
    return


# ─── ACT II: INSTRUMENTS ──────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)

    act2_fabric = mo.ui.dropdown(
        options={
            "IB NDR fat-tree 1:1 (non-blocking)":   "ib_fat_tree",
            "Ethernet fat-tree 2:1 (oversubscribed)": "eth_fat_tree",
            "3D-torus (fixed bisection)":             "torus_3d",
        },
        value="IB NDR fat-tree 1:1 (non-blocking)",
        label="Fabric type",
    )
    act2_cluster_n = mo.ui.slider(
        start=64, stop=4096, value=1024, step=64,
        label="Cluster size N (GPUs)",
    )
    act2_model_gb = mo.ui.slider(
        start=1, stop=200, value=32, step=1,
        label="AllReduce tensor size (GB)",
    )

    mo.vstack([
        mo.md("### Fabric Design Dashboard"),
        mo.hstack([act2_fabric, act2_cluster_n], justify="start", gap="2rem"),
        act2_model_gb,
    ])
    return (act2_fabric, act2_cluster_n, act2_model_gb)


@app.cell(hide_code=True)
def _(
    mo, act1_prediction, act1_reflection, act2_prediction,
    act2_fabric, act2_cluster_n, act2_model_gb,
    go, apply_plotly_theme, COLORS, math,
    IB_NDR400_EFF_GBS, ETH_100G_EFF_GBS,
    IB_NDR_FABRIC_COST_M, ETH_400G_FABRIC_COST_M, TORUS_3D_FABRIC_COST_M,
):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)

    _fabric    = act2_fabric.value
    _N         = act2_cluster_n.value
    _msg_gb    = act2_model_gb.value

    # ── Per-fabric bisection bandwidth physics ────────────────────────────────
    # Source: @sec-network-fabrics-topology-comparison

    if _fabric == "ib_fat_tree":
        # IB NDR400 fat-tree, 1:1 non-blocking
        # link_BW = 45.0 GB/s per GPU (NDR400 effective unidirectional)
        _link_bw_gbs     = IB_NDR400_EFF_GBS       # 45.0 GB/s
        _oversub_ratio   = 1.0
        _bw_bisect       = _N * _link_bw_gbs / (2.0 * _oversub_ratio)  # O(N)
        _fabric_name     = "IB NDR fat-tree (1:1)"
        _fabric_color    = COLORS["GreenLine"]
        _fabric_cost_m   = IB_NDR_FABRIC_COST_M    # $20M
        _topology_note   = "Non-blocking · O(N) bisection BW"
    elif _fabric == "eth_fat_tree":
        # Ethernet 100GbE fat-tree, 2:1 oversubscription
        # Using 100GbE at $8M price point (200GbE ports at 2:1 = 100G effective)
        _link_bw_gbs     = ETH_100G_EFF_GBS * 2.0  # 22.0 GB/s (200GbE effective)
        _oversub_ratio   = 2.0
        _bw_bisect       = _N * _link_bw_gbs / (2.0 * _oversub_ratio)  # O(N/2)
        _fabric_name     = "Ethernet fat-tree (2:1)"
        _fabric_color    = COLORS["OrangeLine"]
        _fabric_cost_m   = ETH_400G_FABRIC_COST_M   # $8M
        _topology_note   = "2:1 Oversubscribed · O(N/2) bisection BW"
    else:
        # 3D-torus: bisection BW = N^(2/3) × link_BW per plane
        # Each GPU has 6 neighbors (±x, ±y, ±z); bisection cuts one dimension
        # Bisection = (N)^(2/3) × link_BW / 2  (cut through a cross-sectional plane)
        # Source: @sec-network-fabrics-topology-comparison (torus bisection derivation)
        _link_bw_gbs     = ETH_100G_EFF_GBS        # 11.0 GB/s (cost-equivalent links)
        _plane_links     = int(_N ** (2.0/3.0))     # number of links crossing bisection
        _bw_bisect       = _plane_links * _link_bw_gbs / 2.0   # O(N^(2/3))
        _fabric_name     = "3D-torus"
        _fabric_color    = COLORS["RedLine"]
        _fabric_cost_m   = TORUS_3D_FABRIC_COST_M  # $5M
        _topology_note   = f"Fixed mesh · O(N^(2/3)) bisection BW · plane links={_plane_links}"

    # ── AllReduce time calculation ─────────────────────────────────────────────
    # Ring-allreduce: T = 2 × (N-1)/N × M / (BW_bisection / N)
    # For large N → T ≈ 2 × M / (BW_bisection / N) = 2 × N × M / BW_bisection
    # Source: @sec-network-fabrics-allreduce (Rabenseifner ring-allreduce analysis)
    _bw_per_gpu    = _bw_bisect / _N                 # GB/s per GPU
    _factor        = 2.0 * (_N - 1) / _N if _N > 1 else 2.0
    _allreduce_sec = _factor * _msg_gb / _bw_per_gpu if _bw_per_gpu > 0 else float("inf")

    # ── SLA threshold: AllReduce must complete < 1 second ─────────────────────
    # Target from stakeholder: AllReduce < 1s for productive training throughput
    # Source: @sec-network-fabrics (AllReduce latency target, LLM training)
    _SLA_SEC       = 1.0
    _sla_violated  = _allreduce_sec > _SLA_SEC

    # ── Training efficiency (fraction of time GPUs are computing, not waiting) ─
    # Assume compute time per iteration ≈ 5 seconds for a 70B-parameter model step
    # on 1024 GPUs; AllReduce is the synchronization barrier.
    # Source: rough model from @sec-network-fabrics (communication overhead)
    _COMPUTE_SEC   = 5.0
    _train_eff_pct = _COMPUTE_SEC / (_COMPUTE_SEC + _allreduce_sec) * 100.0
    _eff_color     = (
        COLORS["GreenLine"]  if _train_eff_pct >= 85 else
        COLORS["OrangeLine"] if _train_eff_pct >= 60 else
        COLORS["RedLine"]
    )

    # ── Cost-performance ratio ────────────────────────────────────────────────
    # How many GB/s of bisection bandwidth per $1M of fabric cost
    _bw_per_dollar_m = _bw_bisect / _fabric_cost_m if _fabric_cost_m > 0 else 0.0

    # ── Topology comparison across cluster sizes ───────────────────────────────
    _n_range      = list(range(64, 4097, 64))
    _bw_ib_fat    = [n * IB_NDR400_EFF_GBS / 2.0            for n in _n_range]
    _bw_eth_fat   = [n * ETH_100G_EFF_GBS * 2.0 / (2 * 2.0) for n in _n_range]
    _bw_torus     = [int(n ** (2.0/3.0)) * ETH_100G_EFF_GBS / 2.0 for n in _n_range]

    _fig = go.Figure()

    _fig.add_trace(go.Scatter(
        x=_n_range, y=_bw_ib_fat, mode="lines", name="IB NDR fat-tree (1:1)",
        line=dict(color=COLORS["GreenLine"], width=2.5),
    ))
    _fig.add_trace(go.Scatter(
        x=_n_range, y=_bw_eth_fat, mode="lines", name="Ethernet fat-tree (2:1)",
        line=dict(color=COLORS["OrangeLine"], width=2.5, dash="dash"),
    ))
    _fig.add_trace(go.Scatter(
        x=_n_range, y=_bw_torus, mode="lines", name="3D-torus",
        line=dict(color=COLORS["RedLine"], width=2.5, dash="dot"),
    ))

    # Mark the current configuration
    _fig.add_trace(go.Scatter(
        x=[_N], y=[_bw_bisect], mode="markers",
        name=f"Current: {_fabric_name}",
        marker=dict(symbol="star", size=14, color=_fabric_color,
                    line=dict(color="white", width=1.5)),
        showlegend=True,
    ))

    # SLA line (AllReduce < 1s → minimum bisection BW needed)
    # T = 2 × N × M / BW_bisect < 1s  →  BW_bisect > 2 × N × M
    _min_bw_for_sla = [2.0 * n * _msg_gb for n in _n_range]
    _fig.add_trace(go.Scatter(
        x=_n_range, y=_min_bw_for_sla, mode="lines",
        name=f"Min BW for <1s AllReduce ({_msg_gb} GB tensor)",
        line=dict(color=COLORS["RedLine"], width=1.5, dash="longdash"),
        opacity=0.65,
    ))

    _fig.update_layout(
        title=dict(
            text="Bisection Bandwidth vs Cluster Size by Topology",
            font=dict(size=13, color="#1e293b"), x=0,
        ),
        height=380,
        yaxis=dict(title="Total Bisection Bandwidth (GB/s)", gridcolor="#f1f5f9"),
        xaxis=dict(title="Cluster Size N (GPUs)"),
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10)),
        margin=dict(l=60, r=20, t=50, b=100),
    )
    apply_plotly_theme(_fig)

    # ── Cost breakdown bar chart ──────────────────────────────────────────────
    _fabrics       = ["IB NDR fat-tree\n(1:1)", "Ethernet fat-tree\n(2:1)", "3D-torus"]
    _costs_m       = [IB_NDR_FABRIC_COST_M, ETH_400G_FABRIC_COST_M, TORUS_3D_FABRIC_COST_M]
    _bw_at_1024    = [
        1024 * IB_NDR400_EFF_GBS / 2.0,
        1024 * ETH_100G_EFF_GBS * 2.0 / (2 * 2.0),
        int(1024 ** (2.0/3.0)) * ETH_100G_EFF_GBS / 2.0,
    ]
    _ar_times      = [
        2.0 * _msg_gb / (bw / 1024) for bw in _bw_at_1024
    ]
    _cost_colors   = [COLORS["GreenLine"], COLORS["OrangeLine"], COLORS["RedLine"]]
    _selected_fab_idx = ["ib_fat_tree", "eth_fat_tree", "torus_3d"].index(_fabric)

    _fig2 = go.Figure()
    _fig2.add_trace(go.Bar(
        x=_fabrics,
        y=_costs_m,
        marker_color=[
            c if i == _selected_fab_idx else c
            for i, c in enumerate(_cost_colors)
        ],
        marker_line_color=["white"] * 3,
        marker_line_width=[0] * 3,
        opacity=[1.0 if i == _selected_fab_idx else 0.45 for i in range(3)],
        text=[f"${c:.0f}M" for c in _costs_m],
        textposition="outside",
        textfont=dict(size=11, family="SF Mono, monospace"),
        width=0.55,
        name="Fabric cost ($M)",
    ))

    # Overlay: AllReduce time as text annotation
    for _i, (_fab, _ar_t) in enumerate(zip(_fabrics, _ar_times)):
        _ar_color = "#008F45" if _ar_t < 1.0 else "#CB202D"
        _fig2.add_annotation(
            x=_fab, y=_costs_m[_i] + 0.5,
            text=f"AllReduce: {_ar_t:.2f}s",
            showarrow=False,
            font=dict(size=9, color=_ar_color, family="SF Mono, monospace"),
        )

    _fig2.update_layout(
        title=dict(
            text=f"Fabric Cost vs AllReduce Time — 1024 GPUs, {_msg_gb} GB tensor",
            font=dict(size=13, color="#1e293b"), x=0,
        ),
        height=320,
        yaxis=dict(title="Fabric Cost ($M)", gridcolor="#f1f5f9"),
        xaxis=dict(title="Fabric Type"),
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    apply_plotly_theme(_fig2)

    # ── Physics formula display ───────────────────────────────────────────────
    if _fabric == "torus_3d":
        _plane_links_disp = int(_N ** (2.0/3.0))
        _formula_text = f"""
    **Bisection Bandwidth — 3D-torus** (N={_N}, link={_link_bw_gbs:.1f} GB/s)

    ```
    Bisection links  =  N^(2/3)  =  {_N}^(2/3)  =  {_plane_links_disp}  links crossing bisection plane
    BW_bisection     =  {_plane_links_disp} × {_link_bw_gbs:.1f} GB/s / 2  =  {_bw_bisect:,.1f} GB/s
    BW per GPU       =  {_bw_bisect:,.1f} / {_N}  =  {_bw_per_gpu:.3f} GB/s
    AllReduce time   ≈  2 × {_msg_gb} GB / {_bw_per_gpu:.3f} GB/s  =  {_allreduce_sec:.2f} s
    ```
    **Scaling penalty at {_N} GPUs vs IB NDR fat-tree:**
    ```
    IB fat-tree BW   =  {_N} × {IB_NDR400_EFF_GBS:.1f} / 2  =  {_N * IB_NDR400_EFF_GBS / 2:.0f} GB/s
    3D-torus BW      =  {_bw_bisect:,.1f} GB/s
    Ratio            =  {(_N * IB_NDR400_EFF_GBS / 2) / _bw_bisect:.1f}× lower bisection BW
    ```
    """
    else:
        _formula_text = f"""
    **Bisection Bandwidth — {_fabric_name}** (N={_N}, link={_link_bw_gbs:.1f} GB/s, r={_oversub_ratio:.0f}:1)

    ```
    BW_bisection  =  N × link_BW / (2 × r)
                  =  {_N} × {_link_bw_gbs:.1f} GB/s / (2 × {_oversub_ratio:.0f})
                  =  {_bw_bisect:,.1f} GB/s   ← {_topology_note}

    BW per GPU    =  {_bw_bisect:,.1f} / {_N}  =  {_bw_per_gpu:.3f} GB/s
    AllReduce     ≈  2 × (N-1)/N × M / (BW/N)
                  ≈  2 × {_msg_gb} GB / {_bw_per_gpu:.3f} GB/s
                  =  {_allreduce_sec:.2f} s
    ```
    """

    # ── Metric cards ──────────────────────────────────────────────────────────
    _ar_color   = COLORS["GreenLine"] if not _sla_violated else COLORS["RedLine"]
    _cards_html = f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin:16px 0;">
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:175px; text-align:center; background:white;
                    border-top:3px solid {_fabric_color};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Bisection BW
            </div>
            <div style="font-size:1.35rem; font-weight:800; color:{_fabric_color};
                        font-family:'SF Mono',monospace;">
                {_bw_bisect:,.0f} GB/s
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                {_topology_note.split(' · ')[0]}
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:175px; text-align:center; background:white;
                    border-top:3px solid {_ar_color};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                AllReduce Time
            </div>
            <div style="font-size:1.35rem; font-weight:800; color:{_ar_color};
                        font-family:'SF Mono',monospace;">
                {_allreduce_sec:.2f} s
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                target: &lt; {_SLA_SEC:.0f} s
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:175px; text-align:center; background:white;
                    border-top:3px solid {_eff_color};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Training Efficiency
            </div>
            <div style="font-size:1.35rem; font-weight:800; color:{_eff_color};
                        font-family:'SF Mono',monospace;">
                {_train_eff_pct:.1f}%
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                GPU compute fraction
            </div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:10px;
                    min-width:175px; text-align:center; background:white;
                    border-top:3px solid {COLORS['BlueLine']};">
            <div style="color:#94a3b8; font-size:0.78rem; font-weight:600; margin-bottom:4px;">
                Fabric Cost
            </div>
            <div style="font-size:1.35rem; font-weight:800; color:{COLORS['TextSec']};
                        font-family:'SF Mono',monospace;">
                ${_fabric_cost_m:.0f}M
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">
                {_bw_per_dollar_m:,.0f} GB/s per $1M
            </div>
        </div>
    </div>
    """

    mo.vstack([
        mo.Html(_cards_html),
        mo.md(_formula_text),
        mo.ui.plotly(_fig),
        mo.ui.plotly(_fig2),
    ])
    return (
        _allreduce_sec, _bw_bisect, _sla_violated,
        _fabric_name, _train_eff_pct, _fabric_cost_m,
    )


# ─── ACT II: FAILURE STATE ────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction,
      _allreduce_sec, _sla_violated, _fabric_name):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)

    if _sla_violated:
        mo.callout(mo.md(
            f"**Bisection bandwidth bottleneck — AllReduce SLA violated.** "
            f"AllReduce requires **{_allreduce_sec:.2f} s** on **{_fabric_name}**. "
            f"This exceeds the 1-second target. "
            f"At this AllReduce time, GPUs spend more time waiting on gradient "
            f"synchronization than computing. Training throughput is severely degraded. "
            f"Reduce the tensor size, increase bisection bandwidth (lower oversubscription "
            f"or switch fabric), or increase cluster link speed to meet the SLA."
        ), kind="danger")
    else:
        mo.callout(mo.md(
            f"**AllReduce SLA met.** "
            f"{_fabric_name} delivers AllReduce in **{_allreduce_sec:.2f} s** "
            f"— within the 1-second target. GPUs remain productive."
        ), kind="success")
    return


# ─── ACT II: PREDICTION FEEDBACK ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction,
      _allreduce_sec, _bw_bisect, _fabric_cost_m):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)

    _act2_correct = act2_prediction.value == "pred_ib"

    _act2_feedback = {
        "pred_torus": (
            "**Not quite.** The 3D-torus is the cheapest option, but its bisection "
            "bandwidth scales as O(N^(2/3)). At 1024 GPUs, the torus bisection "
            "bandwidth is roughly 10× lower than a non-blocking fat-tree. "
            "AllReduce on a 1024-GPU torus will take several seconds per iteration — "
            "far exceeding the 1-second target. The cost saving disappears when "
            "GPU utilization drops below 50%."
        ),
        "pred_eth": (
            "**Not quite.** The 2:1 Ethernet fat-tree halves bisection bandwidth "
            "compared to the IB non-blocking option. For large AllReduce tensors, "
            "this 2× reduction in bisection bandwidth doubles AllReduce time. "
            "At 32 GB tensor size with a 1024-GPU cluster, the Ethernet 2:1 fabric "
            "likely violates the 1-second SLA. The $12M saved on fabric is lost "
            "in reduced GPU utilization and extended training time."
        ),
        "pred_ib": (
            f"**Correct.** For an all-to-all AllReduce workload, bisection bandwidth "
            f"is the bottleneck — not per-link speed or switch count. "
            f"The IB NDR fat-tree with 1:1 non-blocking provides the highest bisection "
            f"bandwidth at scale, completing AllReduce in {_allreduce_sec:.2f} s "
            f"on the current configuration. The $20M fabric cost is justified: "
            f"a cluster with 1024 GPUs at $25,000 each represents $25.6M in compute. "
            f"A fabric that wastes 50% of compute time with slow AllReduce "
            f"effectively discards $12M in GPU capacity."
        ),
        "pred_equal": (
            "**Not quite.** All three topologies perform very differently for AllReduce. "
            "Link speed matters only at the local level — bisection bandwidth determines "
            "the cross-cluster communication ceiling for all-to-all patterns. "
            "A fat-tree and a 3D-torus with the same link speed differ by an order of "
            "magnitude in AllReduce time at 1024 GPU scale because their bisection "
            "bandwidths scale differently with N."
        ),
    }

    mo.callout(
        mo.md(_act2_feedback[act2_prediction.value]),
        kind="success" if _act2_correct else "warn",
    )
    return


# ─── ACT II: MATHPEEK ACCORDION ───────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)

    mo.accordion({
        "The governing equations: AllReduce time and topology scaling": mo.md("""
        **Ring-AllReduce time (Rabenseifner's algorithm):**

        ```
        T_allreduce = 2 × (N - 1) / N × M / BW_per_gpu
                    ≈ 2 × M / BW_per_gpu         for large N

        where BW_per_gpu = BW_bisection / N
        ```

        - **M** — AllReduce message size (gradient tensor, GB)
        - **N** — Number of GPUs
        - **BW_per_gpu** — Each GPU's share of bisection bandwidth
        - Factor **2** — Reduce-scatter phase + all-gather phase

        Substituting bisection bandwidth for each topology:

        ```
        Fat-tree (r oversubscription):
            BW_bisection = N × link_BW / (2 × r)
            T_allreduce  ≈ 2 × M × 2 × r / link_BW
                         = 4 × r × M / link_BW    ← independent of N!

        3D-torus:
            BW_bisection = N^(2/3) × link_BW / 2
            BW_per_gpu   = N^(2/3) × link_BW / (2 × N)  =  link_BW / (2 × N^(1/3))
            T_allreduce  ≈ 4 × M × N^(1/3) / link_BW    ← grows as N^(1/3)!
        ```

        **The critical insight:**

        For fat-tree, AllReduce time is **independent of cluster size N** (assuming
        fixed link_BW and oversubscription). Every GPU added brings its own uplink,
        keeping per-GPU bandwidth constant.

        For 3D-torus, AllReduce time grows as **N^(1/3)** — at 8,000 GPUs, it is
        20× slower than at 1 GPU on the same link speed.

        **Cost-performance trade-off:**

        | Topology         | AllReduce scaling | Cost (1024 GPU) | BW/dollar |
        |------------------|-------------------|-----------------|-----------|
        | IB NDR fat-tree  | O(1)              | $20M            | high      |
        | Eth fat-tree 2:1 | O(1) but 2× slower| $8M             | medium    |
        | 3D-torus         | O(N^(1/3))        | $5M             | low at scale |

        The torus is cost-efficient only when clusters are small enough that
        N^(1/3) × link_BW still meets the latency target.
        """),
    })
    return


# ─── ACT II: STRUCTURED REFLECTION ───────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)

    mo.md("---")
    return


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)

    act2_reflection = mo.ui.radio(
        options={
            "A) 3D-torus has higher latency than fat-tree because of more hops between non-adjacent nodes": "r2_a",
            "B) Torus bisection bandwidth scales as O(N^(2/3)) vs fat-tree O(N) — all-to-all traffic is bottlenecked by the plane boundaries": "r2_b",
            "C) 3D-torus requires special AllReduce algorithms that are less efficient than ring-allreduce": "r2_c",
            "D) 3D-torus topologies cannot support more than 512 GPUs due to diameter limits": "r2_d",
        },
        label="""**Reflection — Act II.**
Why do 3D-torus topologies perform poorly for AllReduce at large cluster scales?""",
    )
    act2_reflection
    return (act2_reflection,)


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection, act2_prediction, act2_reflection):
    mo.stop(act1_prediction.value is None)
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)
    mo.stop(act2_reflection.value is None, mo.callout(
        mo.md("Select your answer to see the explanation."), kind="warn"
    ))

    _r2_correct = act2_reflection.value == "r2_b"

    _r2_feedback = {
        "r2_a": (
            "**Not quite.** Latency (hop count) is a separate property from bandwidth. "
            "A 3D-torus and a fat-tree can have similar average path lengths at the same "
            "cluster size. The AllReduce degradation is purely a bandwidth phenomenon: "
            "all-to-all traffic must cross the bisection, and the torus has fewer links "
            "at the bisection plane than the fat-tree. More hops contribute microseconds; "
            "inadequate bisection bandwidth contributes seconds."
        ),
        "r2_b": (
            "**Correct.** The torus bisection bandwidth grows as O(N^(2/3)) because "
            "the bisection cut passes through a cross-sectional plane of the torus — "
            "a 2D slice whose size grows as the square of the linear dimension. "
            "For a cube-shaped 3D-torus of N GPUs, the bisection has N^(2/3) links. "
            "Fat-tree bisection grows as O(N) because every GPU brings its own uplink "
            "to the spine. At 1024 GPUs, the fat-tree has 10× the bisection bandwidth "
            "of the torus per unit of link speed."
        ),
        "r2_c": (
            "**Not quite.** Standard ring-allreduce operates on any topology. "
            "The algorithm itself is topology-agnostic — it works by routing gradients "
            "around a logical ring of GPUs. The 3D-torus does not require special "
            "AllReduce algorithms. The performance difference is purely a function of "
            "available bisection bandwidth, not algorithmic efficiency."
        ),
        "r2_d": (
            "**Not quite.** 3D-torus topologies are used in some large HPC installations "
            "at thousands of nodes (e.g., early Blue Gene systems used 3D/5D torus). "
            "The problem is not a hard capacity limit but a scaling law: AllReduce time "
            "grows as N^(1/3) on a torus. At 4096 GPUs, this produces AllReduce times "
            "that are 4× slower than at 512 GPUs on the same link speed — making the "
            "topology progressively worse for ML training as clusters grow."
        ),
    }

    mo.callout(
        mo.md(_r2_feedback[act2_reflection.value]),
        kind="success" if _r2_correct else "warn",
    )
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
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. The alpha-beta crossover at ~75 KB determines which optimization lever matters.</strong>
                    Below 75 KB, doubling bandwidth from 200 Gbps to 400 Gbps improves transfer time by
                    only 5% &mdash; startup latency (&alpha; = 1.5 &mu;s) dominates. Above 75 KB,
                    bandwidth upgrades scale linearly. Applying the wrong optimization to the wrong
                    regime wastes engineering budget entirely.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Oversubscription reduces bisection bandwidth proportionally.</strong>
                    A 4:1 oversubscribed spine cuts bisection bandwidth from 25.6 TB/s to 6.4 TB/s,
                    making every global AllReduce 4&times; slower. At 70B-model scale, this pushes
                    AllReduce above 50% of step time, turning the network into the dominant training
                    bottleneck regardless of per-link speed.
                </div>
                <div>
                    <strong>3. Maximum 2:1 oversubscription keeps AllReduce below 30% of step time for 70B models.</strong>
                    At 2:1, bisection bandwidth is 12.8 TB/s and AllReduce consumes approximately 10%
                    of step time &mdash; an acceptable trade-off that cuts switch costs by 50% while
                    preserving training efficiency.
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
                    <strong>Lab V2-04: The Data Gravity Trap</strong> &mdash; This lab showed that
                    the network determines AllReduce throughput. The next lab asks: what happens
                    when the storage pipeline cannot feed GPUs fast enough, and checkpoint writes
                    compete with training data reads for the same NVMe bandwidth?
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
                    <strong>Read:</strong> @sec-network-fabrics-performance-model for the full
                    alpha-beta derivation and @sec-network-fabrics-fat-tree for bisection bandwidth
                    formulas.<br/>
                    <strong>Build:</strong> TinyTorch collective communication module &mdash; implement
                    Ring AllReduce with configurable bandwidth and latency parameters in
                    <code>tinytorch/src/collective/</code>.
                </div>
            </div>

        </div>
        """),

        mo.accordion({


            "Self-Assessment": mo.md("""
**Check your understanding:**

1. The alpha-beta model predicts T = alpha + n/beta. Below ~75 KB, doubling bandwidth from 200 Gbps to 400 Gbps improves transfer time by only ~5%. Why does startup latency dominate for small messages, and at what message size does bandwidth become the binding constraint?
2. A 4:1 oversubscribed spine cuts bisection bandwidth from 25.6 TB/s to 6.4 TB/s. For a 70B-model gradient AllReduce, what maximum oversubscription ratio keeps communication below 30% of step time?
3. Both Pipeline Parallelism (~200 MB activations) and Data Parallelism (~700 GB gradients) operate above the 75 KB crossover. Why do they require different optimization strategies despite both being bandwidth-dominated?

**You're ready to move on if you can:**
- Use the alpha-beta model to determine whether a given message size is latency-dominated or bandwidth-dominated
- Calculate the bisection bandwidth impact of spine oversubscription on AllReduce performance
- Recommend a maximum oversubscription ratio for a given model size and training SLA
""")


        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ─────────────────────────────────────────────────────
# ─── LEDGER SAVE + HUD ────────────────────────────────────────────────────────


@app.cell
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle,
    act1_prediction, act1_reflection,
    act2_prediction, act2_reflection,
    act2_fabric, act2_cluster_n,
    _bw_bisect, _allreduce_sec, _sla_violated, _fabric_name, _fabric_cost_m,
, decision_input, decision_ui):
    # Only save when both acts are complete
    _acts_complete = (
        act1_prediction.value is not None and
        act1_reflection.value is not None and
        act2_prediction.value is not None and
        act2_reflection.value is not None
    )

    if _acts_complete:
        ledger.save(
            chapter="v2_03",
            design={
                "context":           context_toggle.value,
                "fabric_type":       act2_fabric.value,
                "cluster_size":      act2_cluster_n.value,
                "oversubscription":  2.0 if act2_fabric.value == "eth_fat_tree" else (1.0 if act2_fabric.value == "ib_fat_tree" else 0.0),
                "bisection_bw_gbps": round(_bw_bisect, 1),
                "act1_prediction":   act1_prediction.value,
                "act1_correct":      act1_prediction.value == "option_b",
                "act2_result":       round(_allreduce_sec, 3),
                "act2_decision":     act2_fabric.value,
                "constraint_hit":    _sla_violated,
        "student_justification": str(decision_input.value),
            },
        )

    # ── HUD footer ────────────────────────────────────────────────────────────
    _act1_status  = act1_prediction.value is not None
    _act2_status  = act2_prediction.value is not None
    _act1_correct = act1_prediction.value == "option_b"
    _act2_correct = act2_prediction.value == "pred_ib"

    _hud_items = [
        ("Lab",            "Vol II · Lab 03",                             "hud-value"),
        ("Context",        context_toggle.value,                          "hud-value"),
        ("Act I",          "Correct" if _act1_correct else ("Answered" if _act1_status else "Pending"),
         "hud-active" if _act1_correct else ("hud-value" if _act1_status else "hud-none")),
        ("Act II",         "Correct" if _act2_correct else ("Answered" if _act2_status else "Pending"),
         "hud-active" if _act2_correct else ("hud-value" if _act2_status else "hud-none")),
        ("Bisection BW",   f"{_bw_bisect:,.0f} GB/s" if _act2_status else "—",
         "hud-value"),
        ("AllReduce",      f"{_allreduce_sec:.2f} s" if _act2_status else "—",
         "hud-active" if (_act2_status and not _sla_violated) else ("hud-none" if (_act2_status and _sla_violated) else "hud-value")),
        ("SLA",            "Violated" if _sla_violated else ("Met" if _act2_status else "—"),
         "hud-none" if _sla_violated else ("hud-active" if _act2_status else "hud-value")),
        ("Ledger",         "Saved" if _acts_complete else "Incomplete",
         "hud-active" if _acts_complete else "hud-none"),
    ]

    _hud_cells = "".join([
        f"""<div style="display:flex; flex-direction:column; gap:2px;">
                <span class="hud-label">{label}</span>
                <span class="{cls}">{val}</span>
            </div>"""
        for label, val, cls in _hud_items
    ])

    mo.Html(f"""
    <div class="lab-hud" style="margin-top:32px;">
        {_hud_cells}
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
