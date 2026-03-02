import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 06: THE BANDWIDTH INVARIANT
#
# Chapter: Collective Communication (@sec-collective-communication)
# Core Invariant: Ring AllReduce sends 2(N-1)/N × data per GPU — nearly
#                 optimal for large messages. Tree AllReduce sends the same
#                 asymptotic volume but with different bottlenecks. Ring favors
#                 bandwidth; tree favors latency. The algorithm choice determines
#                 whether large-scale gradient sync is feasible.
#
# 2-Act Structure (35-40 minutes):
#   Act I  — The Ring Efficiency Revelation (12-15 min)
#             Naive AllReduce (master-based) at 128 GPUs: master receives 127×M,
#             sends 128×M. Ring: each node sends 2(N-1)/N × M ≈ 2M.
#             The ratio: 255M / 2M = 127.5×. Students must predict this BEFORE
#             running the comparator.
#
#   Act II — Algorithm Selection Under Constraints (20-25 min)
#             Three workloads: 175B gradient (280 GB), 7B gradient (14 GB),
#             small param update (< 1 MB). Students select the optimal algorithm
#             for each using the alpha-beta model crossover formula.
#             Failure state: when AllReduce exceeds 50% of compute step time.
#
# Deployment Contexts (2 comparison toggle):
#   Ring:  Per-node volume = 2(N-1)/N × M — bandwidth-bound, scales perfectly
#   Tree:  Per-node volume = 2(1-1/N) × M — same asymptotic, different bottleneck
#
# Hardware Constants (all commented with source):
#   H100_TFLOPS_FP16  = 1979        # H100 SXM5, NVIDIA spec
#   H100_RAM_GB       = 80          # H100 HBM3e, NVIDIA spec
#   IB_HDR200_BW_GBS  = 400         # InfiniBand NDR, NVIDIA spec (Gbps, ÷8 → GB/s)
#   IB_LATENCY_US     = 1.5         # InfiniBand one-way latency, @sec-collective-communication
#   NVLINK4_BW_GBS    = 900         # NVLink 4.0, NVIDIA spec
#   NVLINK_LATENCY_US = 0.5         # NVLink latency, @sec-collective-communication
#
# Design Ledger: saves chapter="v2_06" with algorithm choice, efficiency,
#                prediction accuracy, failure state trigger.
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
    return COLORS, LAB_CSS, DesignLedger, apply_plotly_theme, go, ledger, math, mo, np


# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, LAB_CSS, mo):
    _c_ring  = COLORS["Cloud"]      # indigo — ring context
    _c_tree  = COLORS["BlueLine"]   # blue — tree context
    _c_s0    = COLORS["Surface0"]
    _c_s1    = COLORS["Surface1"]

    mo.Html(f"""
    {LAB_CSS}
    <div style="background: linear-gradient(135deg, {_c_s0} 0%, {_c_s1} 100%);
                border-radius: 16px; padding: 32px 40px; margin-bottom: 8px;
                border: 1px solid #2d3748;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                            text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;">
                    Vol 2 · Lab 06 · Collective Communication
                </div>
                <div style="font-size: 2.0rem; font-weight: 800; color: #f1f5f9; line-height: 1.15; margin-bottom: 10px;">
                    The Bandwidth Invariant
                </div>
                <div style="font-size: 0.95rem; color: #94a3b8; max-width: 620px; line-height: 1.6;">
                    Ring AllReduce sends 2(N-1)/N per GPU — nearly optimal regardless of cluster size.
                    Naive AllReduce concentrates all traffic on one node. At 128 GPUs, the difference
                    is not 2× or 10×. This lab forces you to confront the exact ratio before you see it.
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;">
                <span class="badge badge-info">Ring vs Tree AllReduce</span>
                <span class="badge badge-info">α-β Model: T = α + n/β</span>
                <span class="badge badge-warn">35-40 minutes · 2 Acts</span>
            </div>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap;">
            <div style="background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_ring}; font-weight: 700;">Ring Context</span>
                <span style="color: #94a3b8;"> — Per-node volume: 2(N-1)/N × M · bandwidth-bound · O(1) per node</span>
            </div>
            <div style="background: rgba(0,99,149,0.15); border: 1px solid rgba(0,99,149,0.4);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_tree}; font-weight: 700;">Tree Context</span>
                <span style="color: #94a3b8;"> — O(log N) steps · latency-efficient · different bottleneck</span>
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 2: RECOMMENDED READING ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-communication-collective-operations-collective-operations-communication-fundamentals-44eb** — Gradient synchronization as a mathematical necessity; why ring AllReduce achieves O(1) per-node cost
    - **@sec-communication-collective-operations-collective-operations-alphabeta-model-f9b4** — The α-β model `T = α + n/β`, the critical message size `n* = α·β`, and the latency vs. bandwidth regimes
    - **@sec-collective-communication** (AllReduce algorithms section) — Ring, tree, and recursive halving algorithms; when each is optimal
    - **@sec-communication-parallelism-patterns** — How parallelism strategy determines which collective primitive runs and at what message size
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE + LEDGER LOAD ──────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _c_ring = COLORS["Cloud"]
    _c_tree = COLORS["BlueLine"]

    context_toggle = mo.ui.radio(
        options={"Ring AllReduce (bandwidth-optimal)": "ring",
                 "Tree AllReduce (latency-optimal)": "tree"},
        value="Ring AllReduce (bandwidth-optimal)",
        label="Algorithm context for this session:",
        inline=True,
    )

    mo.vstack([
        mo.Html(f"""
        <div style="margin: 8px 0 4px 0; font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                    text-transform: uppercase; letter-spacing: 0.1em;">
            Select your comparison context — Ring vs Tree
        </div>
        """),
        context_toggle,
        mo.Html(f"""
        <div style="display: flex; gap: 16px; margin-top: 10px; flex-wrap: wrap; font-size: 0.82rem;">
            <div style="padding: 8px 14px; background: {COLORS['BlueLL']}; border-radius: 8px;
                        border: 1px solid {COLORS['BlueL']}; color: {COLORS['BlueLine']};">
                <strong>Ring:</strong> Each of N nodes sends and receives simultaneously.
                Per-node traffic stays constant as cluster grows.
            </div>
            <div style="padding: 8px 14px; background: {COLORS['BlueLL']}; border-radius: 8px;
                        border: 1px solid {COLORS['BlueL']}; color: {COLORS['BlueLine']};">
                <strong>Tree:</strong> O(log N) reduction steps. Low latency for small messages.
                Root node becomes bottleneck for large data.
            </div>
        </div>
        """),
    ])
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ACT I — THE RING EFFICIENCY REVELATION
# ═══════════════════════════════════════════════════════════════════════════════

# ─── ACT I: STAKEHOLDER MESSAGE ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _color = COLORS["Cloud"]
    _bg    = COLORS["BlueLL"]

    mo.vstack([
        mo.Html(f"""
        <div style="margin: 24px 0 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 4px;">
                Act I · Calibration · 12-15 min
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};">
                The Ring Efficiency Revelation
            </div>
        </div>
        """),
        mo.Html(f"""
        <div style="border-left: 4px solid {_color}; background: {_bg};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 4px 0 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message · Systems Researcher
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "Our naive AllReduce — gather-to-master then broadcast from master — at 128 GPUs
                takes 14× longer than ring AllReduce on the same cluster. We thought our
                implementation was fine because it 'just works.' Can you show me the bandwidth
                ratio calculation so I can explain this to leadership? I need to understand
                exactly how much traffic the master node is absorbing versus what ring distributes
                to every node."
            </div>
        </div>
        """),
        mo.callout(mo.md("""
        **The Setup:** At 128 GPUs, every worker holds a gradient tensor of size M.

        - **Naive AllReduce (master-based):** Master receives 127 × M (all workers send), then
          broadcasts 128 × M back. Total through master: **255 × M**.
        - **Ring AllReduce:** Each node sends exactly 2(N-1)/N × M ≈ 2M per step. Total per node: **~2M**.
        - **Ratio:** 255M / 2M = **127.5×** — the master is 127× more loaded than any ring node.
        - **At N=128:** ring bandwidth efficiency = (N-1)/N = 127/128 ≈ **99.2% of theoretical maximum**.
        """), kind="info"),
    ])
    return


# ─── ACT I: PREDICTION LOCK ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) ~2× difference — naive AllReduce isn't that inefficient": "A",
            "B) ~10× difference — the master bottleneck is real but manageable": "B",
            "C) ~100-130× difference — ring perfectly distributes communication; naive concentrates it on master": "C",
            "D) No difference at small model sizes — bandwidth bottleneck scales with cluster, not model": "D",
        },
        label="At 128 GPUs, what is the bandwidth ratio of naive AllReduce vs. ring AllReduce (naive / ring, per node)?",
    )

    mo.vstack([
        mo.Html("""
        <div style="margin: 16px 0 8px 0; font-size: 0.72rem; font-weight: 700; color: #475569;
                    text-transform: uppercase; letter-spacing: 0.14em;">
            Prediction Lock — Commit before running the simulator
        </div>
        """),
        act1_pred,
    ])
    return (act1_pred,)


# ─── ACT I: GATE ───────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(act1_pred, mo):
    mo.stop(
        act1_pred.value is None,
        mo.callout(mo.md("Select your prediction above to unlock the Act I simulator."), kind="warn"),
    )
    return


# ─── ACT I: SIMULATOR CONTROLS ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    n_gpus_slider = mo.ui.slider(
        start=8, stop=1024, value=128, step=8,
        label="Cluster size N (GPUs)",
    )
    msg_gb_slider = mo.ui.slider(
        start=1, stop=300, value=140, step=1,
        label="Gradient tensor size M (GB)",
    )
    link_bw_dropdown = mo.ui.dropdown(
        options={
            "InfiniBand NDR 400G (50 GB/s per port)": 50,
            "InfiniBand HDR 200G (25 GB/s per port)": 25,
            "NVLink 4.0 (900 GB/s, intra-node only)": 900,
        },
        value="InfiniBand NDR 400G (50 GB/s per port)",
        label="Link bandwidth",
    )

    mo.vstack([
        mo.Html("""
        <div style="margin: 16px 0 4px 0; font-size: 1.1rem; font-weight: 700; color: #0f172a;">
            AllReduce Algorithm Comparator
        </div>
        <div style="font-size: 0.85rem; color: #475569; margin-bottom: 12px;">
            Adjust cluster size, gradient size, and link bandwidth. Watch how naive vs. ring diverge.
        </div>
        """),
        mo.hstack([n_gpus_slider, msg_gb_slider, link_bw_dropdown], justify="start", gap=2),
    ])
    return link_bw_dropdown, msg_gb_slider, n_gpus_slider


# ─── ACT I: SIMULATION PHYSICS ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, go, link_bw_dropdown, math, mo, msg_gb_slider, n_gpus_slider, np):
    # ── Hardware constants ──────────────────────────────────────────────────────
    # IB_LATENCY_US = 1.5  # InfiniBand one-way latency, microseconds
    #                      # Source: @sec-collective-communication, @tbl-interconnect-parameters
    _IB_LATENCY_US = 1.5

    # ── Simulation inputs ───────────────────────────────────────────────────────
    _N   = n_gpus_slider.value          # cluster size
    _M   = msg_gb_slider.value          # gradient size in GB
    _bw  = link_bw_dropdown.value       # link bandwidth in GB/s
    _alpha = _IB_LATENCY_US * 1e-6      # latency in seconds

    # ── Naive AllReduce (master-based) ──────────────────────────────────────────
    # Master receives (N-1)×M from all workers, then sends N×M back.
    # Total through master = (N-1)×M + N×M = (2N-1)×M
    # Time = total_data / bw
    # Source: @sec-collective-communication, Gradient Synchronization definition
    _naive_master_recv_gb = (_N - 1) * _M
    _naive_master_send_gb = _N * _M
    _naive_total_gb = _naive_master_recv_gb + _naive_master_send_gb   # through master
    _naive_time_s   = _naive_total_gb / _bw

    # ── Ring AllReduce ───────────────────────────────────────────────────────────
    # Per-node volume: 2(N-1)/N × M — both scatter-reduce and allgather phases
    # Time = 2(N-1)/N × M / bw  +  2(N-1) × alpha
    # Source: @sec-collective-communication, Ring AllReduce bandwidth formula
    _ring_bw_factor   = 2 * (_N - 1) / _N
    _ring_per_node_gb = _ring_bw_factor * _M
    _ring_bw_time_s   = _ring_per_node_gb / _bw
    _ring_lat_time_s  = 2 * (_N - 1) * _alpha
    _ring_time_s      = _ring_bw_time_s + _ring_lat_time_s

    # ── Tree AllReduce ───────────────────────────────────────────────────────────
    # Binary tree reduction: O(log2 N) steps, each of size M/step_chunk
    # Total volume per node: 2(1 - 1/N) × M  (reduce + broadcast)
    # Number of steps: 2 × log2(N) (reduce tree + broadcast tree)
    # Time = 2(1-1/N) × M / bw  +  2 × log2(N) × alpha
    # Source: @sec-collective-communication
    _log2_N          = math.log2(_N)
    _tree_bw_factor  = 2 * (1 - 1 / _N)
    _tree_per_node_gb = _tree_bw_factor * _M
    _tree_bw_time_s  = _tree_per_node_gb / _bw
    _tree_lat_time_s = 2 * _log2_N * _alpha
    _tree_time_s     = _tree_bw_time_s + _tree_lat_time_s

    # ── Bandwidth ratio naive vs. ring (per-node comparison) ───────────────────
    # Ring per-node: ~2M. Naive per-master: (2N-1)×M
    # ratio = naive_total / ring_per_node
    _ratio_naive_ring = _naive_total_gb / _ring_per_node_gb if _ring_per_node_gb > 0 else 0
    _ring_efficiency  = (_N - 1) / _N * 100    # percentage of theoretical maximum

    # ── Color coding ────────────────────────────────────────────────────────────
    _c_naive = COLORS["RedLine"]
    _c_ring  = COLORS["GreenLine"]
    _c_tree  = COLORS["BlueLine"]

    _naive_ok = _naive_time_s < 60
    _ring_ok  = _ring_time_s  < 60
    _tree_ok  = _tree_time_s  < 60

    def _time_color(t):
        if t < 5:  return COLORS["GreenLine"]
        if t < 30: return COLORS["OrangeLine"]
        return COLORS["RedLine"]

    # ── Bar chart: time comparison ───────────────────────────────────────────────
    _fig = go.Figure()

    _algorithms = ["Naive (master)", "Ring AllReduce", "Tree AllReduce"]
    _times_s     = [_naive_time_s, _ring_time_s, _tree_time_s]
    _bar_colors  = [_c_naive, _c_ring, _c_tree]

    _fig.add_trace(go.Bar(
        x=_algorithms,
        y=[t for t in _times_s],
        marker_color=_bar_colors,
        text=[f"{t:.1f}s" if t >= 1 else f"{t*1000:.0f}ms" for t in _times_s],
        textposition="outside",
        width=0.45,
    ))

    _fig.update_layout(
        height=320,
        yaxis_title="AllReduce Time (seconds)",
        yaxis=dict(
            gridcolor="#f1f5f9",
            linecolor=COLORS["Border"],
            zeroline=True,
            zerolinecolor=COLORS["Border"],
        ),
        xaxis=dict(linecolor=COLORS["Border"]),
        showlegend=False,
        margin=dict(l=50, r=20, t=30, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=COLORS["Text"]),
    )
    apply_plotly_theme(_fig)

    # ── Bandwidth efficiency sweep (ring efficiency vs N) ──────────────────────
    _n_vals = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
    _eff_vals = (_n_vals - 1) / _n_vals * 100

    _fig2 = go.Figure()
    _fig2.add_trace(go.Scatter(
        x=_n_vals, y=_eff_vals,
        mode="lines+markers",
        line=dict(color=COLORS["GreenLine"], width=2),
        marker=dict(size=7, color=COLORS["GreenLine"]),
        name="Ring efficiency",
    ))
    _fig2.add_trace(go.Scatter(
        x=[_N], y=[_ring_efficiency],
        mode="markers",
        marker=dict(size=14, color=COLORS["BlueLine"], symbol="diamond",
                    line=dict(color="white", width=2)),
        name=f"Current N={_N}",
    ))
    _fig2.update_layout(
        height=260,
        xaxis_title="Cluster size N",
        yaxis_title="Ring efficiency (%)",
        xaxis=dict(type="log", gridcolor="#f1f5f9", linecolor=COLORS["Border"]),
        yaxis=dict(range=[90, 101], gridcolor="#f1f5f9", linecolor=COLORS["Border"]),
        showlegend=True,
        margin=dict(l=50, r=20, t=24, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=COLORS["Text"]),
        legend=dict(font=dict(size=10)),
    )
    apply_plotly_theme(_fig2)

    # ── Metric cards ────────────────────────────────────────────────────────────
    def _fmt_time(t):
        if t >= 1:  return f"{t:.2f} s"
        return f"{t*1000:.0f} ms"

    _naive_str = _fmt_time(_naive_time_s)
    _ring_str  = _fmt_time(_ring_time_s)
    _tree_str  = _fmt_time(_tree_time_s)

    mo.vstack([
        mo.Html(f"""
        <div style="margin-top: 8px; padding: 14px 18px; background: {COLORS['Surface2']};
                    border: 1px solid {COLORS['Border']}; border-radius: 10px; font-family: 'SF Mono', monospace;
                    font-size: 0.83rem; line-height: 1.9; color: {COLORS['Text']};">
            <strong>Physics (AllReduce formulas):</strong><br/>
            Naive master total  = (2N-1) × M = (2×{_N}-1) × {_M} GB = <strong>{_naive_total_gb:,.0f} GB</strong><br/>
            Ring per-node       = 2(N-1)/N × M = 2×({_N}-1)/{_N} × {_M} GB = <strong>{_ring_per_node_gb:.2f} GB</strong><br/>
            Tree per-node       = 2(1-1/N) × M = 2×(1-1/{_N}) × {_M} GB = <strong>{_tree_per_node_gb:.2f} GB</strong><br/>
            Bandwidth ratio (naive / ring) = {_naive_total_gb:,.1f} / {_ring_per_node_gb:.2f} = <strong>{_ratio_naive_ring:.1f}×</strong><br/>
            Ring efficiency     = (N-1)/N = ({_N}-1)/{_N} = <strong>{_ring_efficiency:.1f}%</strong>
        </div>
        """),

        mo.Html(f"""
        <div style="display: flex; gap: 14px; margin: 12px 0; flex-wrap: wrap;">
            <div style="padding: 18px 22px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;
                        border-top: 4px solid {COLORS['RedLine']};">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.82rem; margin-bottom: 4px;">Naive (master)</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {_time_color(_naive_time_s)};">
                    {_naive_str}
                </div>
                <div style="font-size: 0.72rem; color: {COLORS['TextMuted']}; margin-top: 4px;">
                    {_naive_total_gb:,.0f} GB through master
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;
                        border-top: 4px solid {COLORS['GreenLine']};">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.82rem; margin-bottom: 4px;">Ring AllReduce</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {_time_color(_ring_time_s)};">
                    {_ring_str}
                </div>
                <div style="font-size: 0.72rem; color: {COLORS['TextMuted']}; margin-top: 4px;">
                    {_ring_per_node_gb:.2f} GB per node · {_ring_efficiency:.1f}% efficient
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;
                        border-top: 4px solid {COLORS['BlueLine']};">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.82rem; margin-bottom: 4px;">Tree AllReduce</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {_time_color(_tree_time_s)};">
                    {_tree_str}
                </div>
                <div style="font-size: 0.72rem; color: {COLORS['TextMuted']}; margin-top: 4px;">
                    {_tree_per_node_gb:.2f} GB per node · {int(_log2_N*2)} tree steps
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;
                        border-top: 4px solid {COLORS['OrangeLine']};">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.82rem; margin-bottom: 4px;">Naive / Ring ratio</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {COLORS['OrangeLine']};">
                    {_ratio_naive_ring:.0f}×
                </div>
                <div style="font-size: 0.72rem; color: {COLORS['TextMuted']}; margin-top: 4px;">
                    master load vs. ring per-node
                </div>
            </div>
        </div>
        """),

        mo.hstack([
            mo.vstack([
                mo.Html(f'<div style="font-size:0.82rem; font-weight:700; color:{COLORS["TextMuted"]}; margin-bottom:4px;">AllReduce Time Comparison (N={_N}, M={_M} GB)</div>'),
                mo.as_html(_fig),
            ]),
            mo.vstack([
                mo.Html(f'<div style="font-size:0.82rem; font-weight:700; color:{COLORS["TextMuted"]}; margin-bottom:4px;">Ring Efficiency vs. Cluster Size</div>'),
                mo.as_html(_fig2),
            ]),
        ], justify="start", gap=2),
    ])
    return


# ─── ACT I: PREDICTION vs. REALITY OVERLAY ─────────────────────────────────────
@app.cell(hide_code=True)
def _(act1_pred, mo, n_gpus_slider):
    _N        = n_gpus_slider.value
    _ring_per = 2 * (_N - 1) / _N   # factor relative to M
    _naive_master = 2 * _N - 1       # factor relative to M at master
    _actual_ratio = _naive_master / _ring_per

    _predicted_ratios = {
        "A": 2.0,
        "B": 10.0,
        "C": 127.5,
        "D": 1.0,
    }
    _selected = act1_pred.value
    if _selected:
        _key = _selected[0]   # "A", "B", "C", or "D"
        _pred_val = _predicted_ratios.get(_key, 0)
        _correct = _key == "C"

        _ratio_off = abs(_actual_ratio / _pred_val - 1) if _pred_val > 0 else float("inf")
        _kind = "success" if _correct else "warn"

        _msg = (
            f"**You predicted ~{_pred_val:.0f}×. The actual bandwidth ratio at N={_N} is {_actual_ratio:.1f}×.** "
        )

        if _correct:
            _explanation = (
                "Correct. Ring AllReduce achieves this because every node sends and receives "
                "simultaneously — the total network traffic is spread evenly across all N links. "
                "Naive AllReduce routes everything through one master node, creating a 2N-1 factor "
                "on the master's bandwidth while ring nodes each see only 2(N-1)/N ≈ 2× their M. "
                "At N=128, that is 255/1.984 ≈ 128× difference."
            )
        elif _key == "A":
            _explanation = (
                "The 2× intuition comes from thinking 'ring does two passes.' That is correct for "
                "the ring itself. The bottleneck shifts: naive concentrates (2N-1)×M on one node. "
                "At N=128, that is 255×M versus 2×M — a 127.5× difference in per-node peak load."
            )
        elif _key == "B":
            _explanation = (
                "10× is in the right direction but underestimates by an order of magnitude. "
                "The master in naive AllReduce must handle (N-1) receive operations and then N "
                "send operations — total (2N-1)×M. At N=128, that is 255 GB for a 1 GB tensor, "
                "not 10 GB. The ratio grows linearly with N."
            )
        else:  # D
            _explanation = (
                "Bandwidth bottleneck scales with model size M, not cluster size N, in ring AllReduce "
                "— that part is correct. But naive AllReduce's master bottleneck absolutely scales with N: "
                "at N=128 it is 127× worse than ring; at N=1024 it would be 1023×. The claim that naive "
                "is equivalent at small models is false because the ratio is independent of M."
            )

        mo.callout(mo.md(_msg + _explanation), kind=_kind)
    return


# ─── ACT I: REFLECTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A) Ring uses a faster network protocol than naive AllReduce": "A",
            "B) Every node sends and receives simultaneously — total network traffic is evenly distributed across all N links": "B",
            "C) Ring compresses gradients during communication to reduce data volume": "C",
            "D) Ring uses asynchronous updates so nodes don't need to synchronize barriers": "D",
        },
        label="Reflection: Why does ring AllReduce achieve near-optimal bandwidth utilization?",
    )

    mo.vstack([
        mo.Html("""
        <div style="margin: 20px 0 8px 0; font-size: 0.72rem; font-weight: 700; color: #475569;
                    text-transform: uppercase; letter-spacing: 0.14em;">
            Act I Reflection — Consolidate the Mechanism
        </div>
        """),
        act1_reflection,
    ])
    return (act1_reflection,)


# ─── ACT I: REFLECTION FEEDBACK ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(act1_reflection, mo):
    if act1_reflection.value is None:
        return

    _key = act1_reflection.value[0]

    if _key == "B":
        mo.callout(mo.md(
            "**Correct.** Ring AllReduce achieves near-optimal bandwidth utilization because every "
            "node is simultaneously a sender and a receiver. In the scatter-reduce phase, node i sends "
            "a chunk to node i+1 while receiving a chunk from node i-1 — all N links fire in parallel. "
            "No single node is a bottleneck. The naive master-based approach violates this property: "
            "one node must absorb all incoming traffic, and its NIC bandwidth becomes the system ceiling."
        ), kind="success")
    elif _key == "A":
        mo.callout(mo.md(
            "**Not quite.** Ring AllReduce uses the same physical network protocol as naive AllReduce. "
            "The difference is algorithmic: ring coordinates N simultaneous point-to-point transfers "
            "rather than one-to-many through a central aggregator. The protocol is identical; the "
            "topology of data movement is not."
        ), kind="warn")
    elif _key == "C":
        mo.callout(mo.md(
            "**Not quite.** Standard ring AllReduce does not compress gradients. Gradient compression "
            "(FP8 quantization, Top-K sparsity) is a separate optimization that can be layered on top "
            "of ring AllReduce. Ring's bandwidth efficiency comes from the algorithmic routing "
            "structure, not from reducing the data volume."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Not quite.** Ring AllReduce is synchronous: all nodes must complete each scatter-reduce "
            "phase before any node can proceed to the allgather phase. The efficiency comes from "
            "simultaneous link utilization, not from eliminating synchronization. Asynchronous gradient "
            "updates (like ASGD) are a different technique that trades convergence guarantees for "
            "reduced synchronization overhead."
        ), kind="warn")
    return


# ─── ACT I: MATHPEEK ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations (Act I)": mo.md("""
        **Ring AllReduce time:**

        ```
        T_ring = 2(N-1)/N × M / β  +  2(N-1) × α
        ```

        - **N** — number of GPUs in the communicator
        - **M** — gradient tensor size (bytes or GB)
        - **β** — link bandwidth (GB/s)
        - **α** — one-way latency per message (seconds)
        - **2(N-1)/N × M** — bandwidth term: nearly 2M for large N (≈ 99.2% of 2M at N=128)
        - **2(N-1) × α** — latency term: 2 phases × (N-1) steps, each costs α

        **Naive (master-based) AllReduce time:**

        ```
        T_naive = (2N-1) × M / β_master
        ```

        - **(N-1) × M** — master receives from N-1 workers (gather phase)
        - **N × M** — master sends back to N workers (broadcast phase)
        - **β_master** — master's NIC bandwidth — the hard ceiling for the entire cluster

        **Bandwidth ratio (naive vs. ring per-node):**

        ```
        ratio = (2N-1)×M / [2(N-1)/N × M] = N(2N-1) / [2(N-1)] ≈ N/2  for large N
        ```

        At N=128: ratio = 128×255 / (2×127) ≈ **127.5×**

        **Ring bandwidth efficiency:**

        ```
        η_ring = (N-1)/N  → 99.2% at N=128,  99.9% at N=1024
        ```

        *Source: @sec-collective-communication, Gradient Synchronization definition.*
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ACT II — ALGORITHM SELECTION UNDER CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── ACT II: STAKEHOLDER MESSAGE ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _color = COLORS["OrangeLine"]
    _bg    = COLORS["OrangeLL"]

    mo.vstack([
        mo.Html(f"""
        <div style="margin: 32px 0 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 4px;">
                Act II · Design Challenge · 20-25 min
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};">
                Algorithm Selection Under Constraints
            </div>
        </div>
        """),
        mo.Html(f"""
        <div style="border-left: 4px solid {_color}; background: {_bg};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 4px 0 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message · MLOps Lead
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "We have three gradient synchronization workloads running on the same InfiniBand cluster:
                (1) a 175B model gradient sync — 280 GB per training step; (2) a 7B model gradient
                sync — 14 GB per step; (3) small parameter server updates for an embedding table —
                under 1 MB per message. Our engineers want to use ring AllReduce for everything
                because 'ring is always fastest.' Design a selection dashboard that shows whether
                that's correct and, if not, what algorithm each workload should use."
            </div>
        </div>
        """),
        mo.callout(mo.md("""
        **The three workloads:**

        | Workload | Gradient size | Expected optimal algorithm |
        |:---------|:-------------|:--------------------------|
        | 175B model gradient sync | 280 GB | Ring (large, bandwidth-bound) |
        | 7B model gradient sync | 14 GB | Ring or tree depending on N |
        | Small param-server updates | < 1 MB | Recursive halving or direct (latency-bound) |

        The α-β crossover point `M* = α/β × (N-1)` determines which regime a message falls in.
        Messages above M* are bandwidth-bound → ring wins.
        Messages below M* are latency-bound → tree or recursive halving win.
        """), kind="info"),
    ])
    return


# ─── ACT II: PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) Ring for all three — ring is always optimal": "A",
            "B) Ring for large (280 GB), tree or ring for medium (14 GB), recursive halving for small (< 1 MB)": "B",
            "C) Tree for all three — tree has lower latency and that always matters": "C",
            "D) Naive (master-based) for small messages — it's simpler and latency doesn't matter under 1 MB": "D",
        },
        label="For the three workloads above, which algorithm assignment is correct?",
    )

    mo.vstack([
        mo.Html("""
        <div style="margin: 16px 0 8px 0; font-size: 0.72rem; font-weight: 700; color: #475569;
                    text-transform: uppercase; letter-spacing: 0.14em;">
            Act II Prediction Lock — Commit before running the algorithm selection dashboard
        </div>
        """),
        act2_pred,
    ])
    return (act2_pred,)


# ─── ACT II: GATE ──────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(act2_pred, mo):
    mo.stop(
        act2_pred.value is None,
        mo.callout(mo.md("Select your Act II prediction above to unlock the algorithm selection dashboard."), kind="warn"),
    )
    return


# ─── ACT II: SIMULATOR CONTROLS ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    workload_radio = mo.ui.radio(
        options={
            "175B model gradient sync (280 GB)": "large",
            "7B model gradient sync (14 GB)": "medium",
            "Small parameter update (< 1 MB)": "small",
        },
        value="175B model gradient sync (280 GB)",
        label="Workload:",
        inline=False,
    )

    msg_size_slider = mo.ui.slider(
        start=0.001, stop=300, value=280, step=0.001,
        label="Message size M (GB)",
    )

    gpu_count_slider = mo.ui.slider(
        start=8, stop=4096, value=128, step=8,
        label="GPU count N",
    )

    bw_radio = mo.ui.radio(
        options={
            "InfiniBand NDR 400G (50 GB/s)": 50,
            "InfiniBand HDR 200G (25 GB/s)": 25,
            "NVLink 4.0 (900 GB/s)": 900,
        },
        value="InfiniBand NDR 400G (50 GB/s)",
        label="Link bandwidth β:",
        inline=True,
    )

    mo.vstack([
        mo.Html("""
        <div style="margin: 8px 0 4px 0; font-size: 1.1rem; font-weight: 700; color: #0f172a;">
            Algorithm Selection Dashboard
        </div>
        <div style="font-size: 0.85rem; color: #475569; margin-bottom: 10px;">
            Select a workload, then adjust N and bandwidth to see crossover behavior.
            Override message size to explore the full latency-bandwidth spectrum.
        </div>
        """),
        workload_radio,
        mo.hstack([msg_size_slider, gpu_count_slider, bw_radio], justify="start", gap=2),
    ])
    return bw_radio, gpu_count_slider, msg_size_slider, workload_radio


# ─── ACT II: WORKLOAD PRESET SYNC ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, msg_size_slider, workload_radio):
    # When user selects a workload preset, show what the canonical message size is
    _workload = workload_radio.value
    _preset_sizes = {"large": 280, "medium": 14, "small": 0.001}
    _preset_names = {
        "large": "175B model (280 GB gradient)",
        "medium": "7B model (14 GB gradient)",
        "small": "Small param update (1 MB = 0.001 GB)",
    }
    _canonical = _preset_sizes.get(_workload, msg_size_slider.value)

    mo.callout(mo.md(
        f"**Selected workload:** {_preset_names.get(_workload, '?')} — "
        f"canonical message size is **{_canonical} GB**. "
        f"The slider currently shows **{msg_size_slider.value:.3f} GB**. "
        f"Adjust the slider to match the workload or explore the crossover point."
    ), kind="info")
    return


# ─── ACT II: SIMULATION PHYSICS ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, bw_radio, go, gpu_count_slider, math, mo, msg_size_slider, np, workload_radio):
    # ── Hardware constants ──────────────────────────────────────────────────────
    # IB_LATENCY_US     = 1.5  # InfiniBand one-way latency, μs
    #                          # Source: @sec-collective-communication, @tbl-interconnect-parameters
    # H100_TFLOPS_FP16  = 1979 # H100 SXM5 FP16 Tensor Core TFLOPS, NVIDIA spec
    # COMPUTE_STEP_REF  = estimated compute time per training step
    _IB_LATENCY_US    = 1.5
    _H100_TFLOPS_FP16 = 1979   # TFLOPS, NVIDIA H100 SXM5 spec
    _H100_RAM_GB      = 80     # GB HBM3e, NVIDIA spec

    # ── Simulation inputs ───────────────────────────────────────────────────────
    _N        = gpu_count_slider.value
    _M_gb     = msg_size_slider.value
    _M_bytes  = _M_gb * 1e9
    _bw       = bw_radio.value         # GB/s
    _bw_bytes = _bw * 1e9
    _alpha    = _IB_LATENCY_US * 1e-6  # seconds

    # ── Crossover message size (α-β model) ─────────────────────────────────────
    # M* = α × β — messages above this are bandwidth-bound (ring wins)
    # M* scaled for N: M*_N = α × β × (N-1) — ring crossover accounting for N steps
    # Source: @sec-collective-communication, α-β Model definition
    _m_star_bytes = _alpha * _bw_bytes
    _m_star_gb    = _m_star_bytes / 1e9
    _m_star_n_gb  = _m_star_gb * (_N - 1)   # ring-adjusted crossover for N nodes

    _is_bw_bound  = _M_gb > _m_star_gb
    _regime       = "bandwidth-bound" if _is_bw_bound else "latency-bound"

    # ── Ring AllReduce ──────────────────────────────────────────────────────────
    # T_ring = 2(N-1)/N × M/β  +  2(N-1)×α
    _ring_bw_s  = 2 * (_N - 1) / _N * _M_gb / _bw
    _ring_lat_s = 2 * (_N - 1) * _alpha
    _ring_s     = _ring_bw_s + _ring_lat_s

    # ── Tree AllReduce ──────────────────────────────────────────────────────────
    # T_tree = 2(1-1/N) × M/β  +  2×log2(N)×α
    _log2_N      = math.log2(_N)
    _tree_bw_s   = 2 * (1 - 1 / _N) * _M_gb / _bw
    _tree_lat_s  = 2 * _log2_N * _alpha
    _tree_s      = _tree_bw_s + _tree_lat_s

    # ── Recursive Halving (small messages) ─────────────────────────────────────
    # T_rh = log2(N) × α  +  (1-1/N) × M/β
    # More efficient than ring for latency-bound regime (fewer steps than ring)
    # Source: @sec-collective-communication
    _rh_lat_s  = _log2_N * _alpha
    _rh_bw_s   = (1 - 1 / _N) * _M_gb / _bw
    _rh_s      = _rh_lat_s + _rh_bw_s

    # ── Optimal algorithm selection ─────────────────────────────────────────────
    _times = {"Ring": _ring_s, "Tree": _tree_s, "Recursive Halving": _rh_s}
    _best_alg = min(_times, key=lambda k: _times[k])

    # ── Compute step time estimate ──────────────────────────────────────────────
    # Estimate compute FLOPs for the workload; use step_time = FLOPs / (N × TFLOPS × utilization)
    # 175B model forward+backward ≈ 2 × 2 × 175e9 × seq_len × batch_size FLOPs
    # For a representative single step with batch=16, seq=2048:
    #   FLOPs ≈ 6 × params × seq × batch (rule of thumb: 6P per token)
    _workload = workload_radio.value
    _params_b = {"large": 175, "medium": 7, "small": 0.1}.get(_workload, 14)
    _seq      = 2048
    _batch    = 16
    _step_flops = 6 * _params_b * 1e9 * _seq * _batch   # total FLOPs for the step
    _util       = 0.50                                    # 50% MFU is typical at scale
    _compute_s  = _step_flops / (_N * _H100_TFLOPS_FP16 * 1e12 * _util)

    # ── Communication overhead percentage ──────────────────────────────────────
    _best_comm_s    = _times[_best_alg]
    _total_step_s   = _compute_s + _best_comm_s
    _comm_overhead_pct = _best_comm_s / _total_step_s * 100 if _total_step_s > 0 else 0

    # ── Failure state: AllReduce exceeds 50% of step time ─────────────────────
    _comm_dominates = _comm_overhead_pct > 50

    # ── Algorithm sweep plot (time vs message size) ─────────────────────────────
    _m_range_gb = np.logspace(-4, math.log10(max(300, _M_gb * 1.5)), 200)
    _ring_curve  = 2 * (_N - 1) / _N * _m_range_gb / _bw + 2 * (_N - 1) * _alpha
    _tree_curve  = 2 * (1 - 1 / _N) * _m_range_gb / _bw + 2 * _log2_N * _alpha
    _rh_curve    = _log2_N * _alpha + (1 - 1 / _N) * _m_range_gb / _bw

    _fig3 = go.Figure()
    _fig3.add_trace(go.Scatter(
        x=_m_range_gb, y=_ring_curve * 1000,
        mode="lines", name="Ring AllReduce",
        line=dict(color=COLORS["GreenLine"], width=2),
    ))
    _fig3.add_trace(go.Scatter(
        x=_m_range_gb, y=_tree_curve * 1000,
        mode="lines", name="Tree AllReduce",
        line=dict(color=COLORS["BlueLine"], width=2),
    ))
    _fig3.add_trace(go.Scatter(
        x=_m_range_gb, y=_rh_curve * 1000,
        mode="lines", name="Recursive Halving",
        line=dict(color=COLORS["OrangeLine"], width=2, dash="dash"),
    ))
    # Current operating point
    _fig3.add_trace(go.Scatter(
        x=[_M_gb], y=[_best_comm_s * 1000],
        mode="markers", name=f"Current: {_best_alg}",
        marker=dict(size=14, color=COLORS["RedLine"], symbol="star",
                    line=dict(color="white", width=2)),
    ))
    # Crossover line
    _fig3.add_vline(
        x=_m_star_gb,
        line_dash="dot", line_color=COLORS["TextMuted"], line_width=1.5,
        annotation_text=f"n*={_m_star_gb*1000:.0f} MB (crossover)",
        annotation_position="top right",
        annotation_font_size=10,
    )

    _fig3.update_layout(
        height=320,
        xaxis=dict(title="Message size M (GB)", type="log", gridcolor="#f1f5f9", linecolor=COLORS["Border"]),
        yaxis=dict(title="AllReduce time (ms)", type="log", gridcolor="#f1f5f9", linecolor=COLORS["Border"]),
        showlegend=True,
        legend=dict(font=dict(size=10)),
        margin=dict(l=50, r=20, t=30, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=COLORS["Text"]),
    )
    apply_plotly_theme(_fig3)

    # ── Format times ────────────────────────────────────────────────────────────
    def _ft(t):
        if t >= 1:   return f"{t:.2f} s"
        if t >= 0.001: return f"{t*1000:.1f} ms"
        return f"{t*1e6:.1f} μs"

    _ring_str = _ft(_ring_s)
    _tree_str = _ft(_tree_s)
    _rh_str   = _ft(_rh_s)
    _comp_str = _ft(_compute_s)
    _best_str = _ft(_best_comm_s)

    _rec_color = {
        "Ring": COLORS["GreenLine"],
        "Tree": COLORS["BlueLine"],
        "Recursive Halving": COLORS["OrangeLine"],
    }.get(_best_alg, COLORS["BlueLine"])

    # ── Output ───────────────────────────────────────────────────────────────────
    _output = mo.vstack([
        mo.Html(f"""
        <div style="margin-top: 8px; padding: 14px 18px; background: {COLORS['Surface2']};
                    border: 1px solid {COLORS['Border']}; border-radius: 10px;
                    font-family: 'SF Mono', monospace; font-size: 0.83rem; line-height: 1.9;">
            <strong>α-β Model Analysis (N={_N}, M={_M_gb:.3f} GB, β={_bw} GB/s):</strong><br/>
            Critical message size   n* = α × β = {_IB_LATENCY_US}μs × {_bw} GB/s = <strong>{_m_star_gb*1000:.1f} MB</strong><br/>
            Current M = {_M_gb:.3f} GB = {_M_gb*1000:.1f} MB → regime: <strong>{_regime}</strong><br/>
            Ring: bw_term={_ring_bw_s*1000:.2f}ms + lat_term={_ring_lat_s*1e6:.1f}μs = <strong>{_ring_str}</strong><br/>
            Tree: bw_term={_tree_bw_s*1000:.2f}ms + lat_term={_tree_lat_s*1e6:.1f}μs = <strong>{_tree_str}</strong><br/>
            Rec.Halving: bw_term={_rh_bw_s*1000:.2f}ms + lat_term={_rh_lat_s*1e6:.1f}μs = <strong>{_rh_str}</strong>
        </div>
        """),

        mo.Html(f"""
        <div style="display: flex; gap: 14px; margin: 12px 0; flex-wrap: wrap;">
            <div style="padding: 18px 22px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        min-width: 155px; text-align: center; background: white;
                        border-top: 4px solid {COLORS['GreenLine']};">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.82rem; margin-bottom: 4px;">Ring AllReduce</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {COLORS['GreenLine']};">{_ring_str}</div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        min-width: 155px; text-align: center; background: white;
                        border-top: 4px solid {COLORS['BlueLine']};">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.82rem; margin-bottom: 4px;">Tree AllReduce</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {COLORS['BlueLine']};">{_tree_str}</div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        min-width: 155px; text-align: center; background: white;
                        border-top: 4px solid {COLORS['OrangeLine']};">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.82rem; margin-bottom: 4px;">Recursive Halving</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {COLORS['OrangeLine']};">{_rh_str}</div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        min-width: 155px; text-align: center; background: white;
                        border-top: 4px solid {_rec_color};">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.82rem; margin-bottom: 4px;">Recommended</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: {_rec_color};">{_best_alg}</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextMuted']}; margin-top: 4px;">{_best_str}</div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        min-width: 155px; text-align: center; background: white;
                        border-top: 4px solid {COLORS['TextMuted']};">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.82rem; margin-bottom: 4px;">Compute step</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {COLORS['Text']};">{_comp_str}</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextMuted']}; margin-top: 4px;">
                    Comm overhead: {_comm_overhead_pct:.0f}%
                </div>
            </div>
        </div>
        """),

        mo.Html(f"""
        <div style="margin: 8px 0 4px 0; font-size: 0.82rem; font-weight: 700; color: {COLORS['TextMuted']};">
            AllReduce Time vs. Message Size (N={_N}, β={_bw} GB/s)
        </div>
        """),
        mo.as_html(_fig3),
    ])

    if _comm_dominates:
        _failure = mo.callout(mo.md(
            f"**Communication overhead: {_comm_overhead_pct:.0f}% of step time.** "
            f"{_best_alg} AllReduce requires **{_best_str}** vs. compute step **{_comp_str}**. "
            f"At this ratio, training efficiency is severely degraded. "
            f"Consider: (1) gradient compression (FP8 or Top-K sparsity) to reduce M; "
            f"(2) communication-computation overlap (pipeline AllReduce with backward pass); "
            f"(3) gradient accumulation to increase compute per AllReduce; "
            f"(4) hierarchical AllReduce to use faster intra-node NVLink bandwidth first."
        ), kind="warn")
        mo.vstack([_output, _failure])
    else:
        _output

    # Export named variables for HUD cell
    best_alg          = _best_alg
    comm_overhead_pct = _comm_overhead_pct
    best_comm_s       = _best_comm_s
    comm_dominates    = _comm_dominates
    act2_N            = _N
    act2_M_gb         = _M_gb

    return best_alg, comm_overhead_pct, best_comm_s, comm_dominates, act2_N, act2_M_gb


# ─── ACT II: PREDICTION FEEDBACK ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(act2_pred, mo):
    if act2_pred.value is None:
        return

    _key = act2_pred.value[0]

    if _key == "B":
        mo.callout(mo.md(
            "**Correct.** The optimal algorithm depends on where each workload falls relative to the "
            "crossover message size `n* = α × β`. At InfiniBand NDR (α=1.5 μs, β=50 GB/s), "
            "n* ≈ 75 KB. The 280 GB gradient is 3.7 million × above n* — firmly bandwidth-bound; "
            "ring wins decisively. The 14 GB gradient is still well above n* — ring remains "
            "strong. The sub-1 MB param updates fall near or below n* — recursive halving or "
            "direct send reduces the per-message latency by avoiding ring's 2(N-1) round-trip steps."
        ), kind="success")
    elif _key == "A":
        mo.callout(mo.md(
            "**Not quite.** Ring is optimal for large messages (above n*) because it maximizes "
            "bandwidth utilization. For small messages (below n*), ring executes 2(N-1) sequential "
            "steps — at N=128, that is 254 latency terms (254 × 1.5 μs = 381 μs) for a message "
            "that only requires ~20 μs to transfer. Recursive halving cuts this to log2(128) = 7 "
            "steps: 7 × 1.5 μs = 10.5 μs of latency. Ring for small messages wastes 36× the "
            "latency budget."
        ), kind="warn")
    elif _key == "C":
        mo.callout(mo.md(
            "**Not quite.** Tree AllReduce has O(log N) steps like recursive halving, so it is "
            "better than ring for latency. But tree AllReduce routes all data through the root node "
            "for large messages — the root receives from all children and re-sends, creating the "
            "same master-bottleneck problem as naive AllReduce for bandwidth-bound messages. "
            "Ring's simultaneous link utilization makes it strictly superior for large gradients."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Not quite.** Naive (master-based) AllReduce is the worst choice for small messages "
            "because latency absolutely matters for small messages. When M < n*, the dominant term "
            "is α — and naive AllReduce requires two sequential phases (gather then broadcast) "
            "through one master, doubling the latency. Recursive halving requires only log2(N) "
            "concurrent steps. Even for 1 MB, naive is significantly slower than ring or halving."
        ), kind="warn")
    return


# ─── ACT II: REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A) The number of GPUs — ring above 64, tree below 64": "A",
            "B) Message size relative to the latency-bandwidth crossover point M* = α × β × (N-1)": "B",
            "C) Whether gradients are in FP16 or FP32 — ring only works with FP32": "C",
            "D) The network topology — ring only works on physical ring-shaped topologies": "D",
        },
        label="Reflection: What is the key metric that determines whether to use ring vs. tree AllReduce?",
    )

    mo.vstack([
        mo.Html("""
        <div style="margin: 20px 0 8px 0; font-size: 0.72rem; font-weight: 700; color: #475569;
                    text-transform: uppercase; letter-spacing: 0.14em;">
            Act II Reflection — Consolidate the Selection Rule
        </div>
        """),
        act2_reflection,
    ])
    return (act2_reflection,)


# ─── ACT II: REFLECTION FEEDBACK ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(act2_reflection, mo):
    if act2_reflection.value is None:
        return

    _key = act2_reflection.value[0]

    if _key == "B":
        mo.callout(mo.md(
            "**Correct.** The crossover point `M* = α × β` (per-link) — scaled by (N-1) for the "
            "multi-step ring — is the single metric that determines which algorithm wins. When "
            "`M >> M*`, the bandwidth term dominates and ring's simultaneous link utilization "
            "gives it the edge. When `M << M*`, the latency term dominates and ring's 2(N-1) "
            "sequential round-trips are wasteful — recursive halving's log2(N) steps win. "
            "The crossover is independent of GPU count but depends entirely on the interconnect "
            "parameters α and β."
        ), kind="success")
    elif _key == "A":
        mo.callout(mo.md(
            "**Not quite.** GPU count N affects the absolute latency term (2(N-1)×α for ring vs. "
            "2×log2(N)×α for tree), but it is not the primary crossover metric. Even at N=8, "
            "a 100 GB gradient is bandwidth-bound and ring wins. Even at N=1024, a 1 KB message "
            "is latency-bound and recursive halving wins. Message size M relative to n* is the "
            "decisive variable."
        ), kind="warn")
    elif _key == "C":
        mo.callout(mo.md(
            "**Not quite.** Ring AllReduce is precision-agnostic — it works with FP32, BF16, FP16, "
            "FP8, or any numeric format. The reduce operation (summation) is the same regardless. "
            "FP16 gradients are half the size (70 GB for a 175B model instead of 140 GB in FP32), "
            "which shifts the workload toward the bandwidth-bound regime more quickly, but does not "
            "change which algorithm is optimal for a given message size."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Not quite.** Ring AllReduce is a logical algorithm, not a physical topology requirement. "
            "NCCL implements ring AllReduce on fat-tree, torus, and fully-connected networks by "
            "constructing a logical ring ordering over the physical nodes. The algorithm is topology-aware "
            "in the sense that NCCL tries to construct a ring that follows high-bandwidth physical paths "
            "(e.g., intra-node NVLink first), but the algorithm itself runs on any network topology."
        ), kind="warn")
    return


# ─── ACT II: MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations (Act II)": mo.md("""
        **α-β Model (Hockney Model):**

        ```
        T(M) = α + M/β
        ```

        - **α** — per-message startup latency (fixed cost, hardware-dependent)
        - **β** — link bandwidth (GB/s, bytes per second)
        - **M** — message size (bytes)

        **Critical message size (crossover point):**

        ```
        n* = α × β
        ```

        At InfiniBand NDR (α=1.5 μs, β=50 GB/s):
        `n* = 1.5×10⁻⁶ × 50×10⁹ = 75,000 bytes = 75 KB`

        Messages below n* are latency-bound; above n* are bandwidth-bound.

        **Algorithm complexity (time complexity):**

        | Algorithm | Bandwidth term | Latency term | Total steps |
        |:----------|:--------------|:-------------|:-----------|
        | Ring AllReduce | 2(N-1)/N × M/β | 2(N-1)×α | 2(N-1) |
        | Tree AllReduce | 2(1-1/N) × M/β | 2×log₂(N)×α | 2×log₂(N) |
        | Recursive Halving | (1-1/N) × M/β | log₂(N)×α | log₂(N) |

        **Crossover message size (ring vs. tree, adjusted for N):**

        ```
        M*_ring_tree = α × β × (N-1) / log₂(N)
        ```

        Above this: ring wins (bandwidth savings dominate).
        Below this: tree or recursive halving wins (fewer latency steps).

        *Source: @sec-collective-communication, α-β Model definition and AllReduce algorithms.*
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN LEDGER SAVE + HUD FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(
    COLORS,
    act1_pred,
    act2_pred,
    act2_N,
    act2_M_gb,
    best_alg,
    best_comm_s,
    comm_dominates,
    comm_overhead_pct,
    context_toggle,
    ledger,
    mo,
):
    # ── Compute prediction correctness ──────────────────────────────────────────
    _a1_correct = (act1_pred.value is not None and act1_pred.value.startswith("C"))
    _a2_correct = (act2_pred.value is not None and act2_pred.value.startswith("B"))

    _a1_key = act1_pred.value[0] if act1_pred.value else "?"
    _a2_key = act2_pred.value[0] if act2_pred.value else "?"

    # ── Ring efficiency at current N ────────────────────────────────────────────
    _bw_efficiency = (act2_N - 1) / act2_N * 100

    # ── Save to Design Ledger ───────────────────────────────────────────────────
    ledger.save(
        chapter="v2_06",
        design={
            "context":              context_toggle.value,
            "algorithm":            best_alg,
            "cluster_size":         act2_N,
            "message_size_gb":      act2_M_gb,
            "bandwidth_efficiency": round(_bw_efficiency, 2),
            "act1_prediction":      _a1_key,
            "act1_correct":         _a1_correct,
            "act2_result":          round(best_comm_s, 4),
            "act2_decision":        best_alg,
            "constraint_hit":       comm_dominates,
            "comm_overhead_pct":    round(comm_overhead_pct, 1),
        },
    )

    # ── HUD display ─────────────────────────────────────────────────────────────
    _tm = COLORS["TextMuted"]

    def _hud_val(v):
        return f'<span class="hud-value">{v}</span>'

    def _hud_bool(b):
        cls = "hud-active" if b else "hud-none"
        txt = "YES" if b else "NO"
        return f'<span class="{cls}">{txt}</span>'

    mo.vstack([
        mo.Html(f"""
        <div style="margin-top: 32px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {_tm};
                        text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;">
                Design Ledger — Lab 06 Summary
            </div>
            <div class="lab-hud">
                <div>
                    <span class="hud-label">CONTEXT </span>
                    {_hud_val(context_toggle.value.upper())}
                </div>
                <div>
                    <span class="hud-label">CLUSTER N </span>
                    {_hud_val(act2_N)}
                </div>
                <div>
                    <span class="hud-label">MSG M </span>
                    {_hud_val(f"{act2_M_gb:.1f} GB")}
                </div>
                <div>
                    <span class="hud-label">BEST ALG </span>
                    {_hud_val(best_alg)}
                </div>
                <div>
                    <span class="hud-label">RING EFF </span>
                    {_hud_val(f"{_bw_efficiency:.1f}%")}
                </div>
                <div>
                    <span class="hud-label">ACT 1 </span>
                    {_hud_bool(_a1_correct)}
                </div>
                <div>
                    <span class="hud-label">ACT 2 </span>
                    {_hud_bool(_a2_correct)}
                </div>
                <div>
                    <span class="hud-label">COMM OVERHEAD </span>
                    {_hud_val(f"{comm_overhead_pct:.0f}%")}
                </div>
                <div>
                    <span class="hud-label">CONSTRAINT HIT </span>
                    {_hud_bool(comm_dominates)}
                </div>
            </div>
        </div>
        """),

        mo.callout(mo.md("""
        **Lab 06 complete.** Your design decisions have been saved to the Design Ledger.

        **Key invariants to carry forward:**

        1. **Ring AllReduce bandwidth efficiency = (N-1)/N** — near-optimal for large gradients at any cluster size above ~8 GPUs. Per-node communication volume is constant at ~2M, independent of N.

        2. **Algorithm selection = message size vs. crossover point n* = α×β.** Large gradients: ring. Sub-kilobyte messages: recursive halving. The crossover is a hardware property, not a configuration choice.

        3. **Communication overhead > 50% of step time is a failure mode** — not a configuration problem. It requires gradient compression, computation-communication overlap, or hierarchical AllReduce across NVLink + InfiniBand.

        Next: **Lab 07** — The Young-Daly optimal checkpoint interval and fault tolerance at scale.
        """), kind="success"),
    ])
    return


if __name__ == "__main__":
    app.run()
