import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly"], keep_going=False)
        await micropip.install(
            "../../wheels/mlsysim-0.1.0-py3-none-any.whl", keep_going=False
        )
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    import mlsysim
    from mlsysim.core.defaults import (
        INFINIBAND_NDR_BW_GBS,
        INFINIBAND_HDR_BW_GBS,
        IB_NDR_LATENCY_US,
        IB_HDR_LATENCY_US,
        DEFAULT_OVERLAP_EFFICIENCY,
    )
    from mlsysim.core.formulas import (
        calc_ring_allreduce_time,
        calc_tree_allreduce_time,
        calc_hierarchical_allreduce_time,
    )
    from mlsysim.core.constants import ureg, NVLINK_H100_BW

    NVLINK_GBS = NVLINK_H100_BW.m_as("GB/s")

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return (
        COLORS, LAB_CSS, apply_plotly_theme, go, math, mo, np, ledger, ureg,
        INFINIBAND_NDR_BW_GBS, INFINIBAND_HDR_BW_GBS,
        IB_NDR_LATENCY_US, IB_HDR_LATENCY_US,
        NVLINK_GBS, DEFAULT_OVERLAP_EFFICIENCY,
        calc_ring_allreduce_time, calc_tree_allreduce_time,
        calc_hierarchical_allreduce_time,
    )


@app.cell(hide_code=True)
def _(LAB_CSS, mo):
    mo.vstack([
        LAB_CSS,
        mo.Html("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 03
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                Communication at Scale
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Alpha-Beta Model &middot; Ring vs Tree &middot; Hierarchy &middot; Compression
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Gradient synchronization can consume half of training step time.
                The alpha-beta model reveals why, algorithm choice depends on a
                crossover formula, and hierarchical strategies exploit the NVLink/IB cliff.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    5 Parts &middot; ~58 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Vol II Ch 3+6: Network Fabrics + Collective Comms
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">T(n) = alpha + n/beta</span>
                <span class="badge badge-warn">Ring vs Tree crossover</span>
                <span class="badge badge-fail">11-second AllReduce for 70B</span>
            </div>
        </div>
        """),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
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
                <div style="margin-bottom: 3px;">1. <strong>Calculate AllReduce time</strong> &mdash; use the alpha-beta
                    model to show that 70B FP32 gradients on 64 GPUs take ~11 seconds via Ring
                    AllReduce on IB NDR, with bandwidth comprising 99.99% of total.</div>
                <div style="margin-bottom: 3px;">2. <strong>Identify the Ring/Tree crossover</strong> &mdash; find the
                    message size where Tree AllReduce beats Ring at 256 GPUs and explain the
                    O(N) vs O(log N) latency trade-off.</div>
                <div style="margin-bottom: 3px;">3. <strong>Build a communication budget</strong> &mdash; stack hierarchical
                    AllReduce, FP16 gradients, overlap, and bucket fusion to reduce communication
                    below 20% of step time for a 70B model.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    V2-01 (Fleet Law) &middot; V2-02 (bandwidth staircase)
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~58 min</strong><br/>
                    Parts A&ndash;E: ~10&ndash;12 min each
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 12px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;A single Ring AllReduce for a 70B model takes 11 seconds. How many
                optimizations must you stack to get communication under 20% of step time
                &mdash; and when does each optimization actually help?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **Vol II Ch 3: Network Fabrics** -- alpha-beta model, fat-tree topology, oversubscription.
    - **Vol II Ch 6: Collective Communication** -- Ring vs Tree AllReduce, hierarchical algorithms,
      gradient compression trade-offs, communication-computation overlap.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(
    COLORS, apply_plotly_theme, go, math, mo, np, ureg,
    INFINIBAND_NDR_BW_GBS, IB_NDR_LATENCY_US, NVLINK_GBS,
    DEFAULT_OVERLAP_EFFICIENCY,
    calc_ring_allreduce_time, calc_tree_allreduce_time,
    calc_hierarchical_allreduce_time,
):
    # ═════════════════════════════════════════════════════════════════════════
    # WIDGETS
    # ═════════════════════════════════════════════════════════════════════════
    pA_pred = mo.ui.radio(
        options={
            "A) ~0.5 ms -- network latency dominates": "0.5",
            "B) ~50 ms -- bandwidth matters, but IB is fast": "50",
            "C) ~1,100 ms (~1 second)": "1100",
            "D) ~11,000 ms (~11 seconds) -- bandwidth completely dominates": "11000",
        },
        label="70B FP32 gradients, 64 GPUs, IB NDR. How long does one Ring AllReduce take?",
    )
    return (pA_pred,)

@app.cell(hide_code=True)
def _(mo, pA_pred):
    pA_model = mo.ui.dropdown(
        options={"1B": 1, "7B": 7, "13B": 13, "70B": 70, "175B": 175},
        value="70B", label="Model params (B)",
    )
    pA_prec = mo.ui.dropdown(
        options={"FP32 (4B)": 4, "BF16 (2B)": 2, "FP8 (1B)": 1},
        value="FP32 (4B)", label="Gradient precision",
    )
    pA_gpus = mo.ui.dropdown(
        options={"8": 8, "16": 16, "32": 32, "64": 64, "128": 128, "256": 256, "512": 512, "1024": 1024},
        value="64", label="GPU count",
    )

    pB_pred = mo.ui.radio(
        options={
            "A) Ring -- it is always bandwidth-optimal": "ring",
            "B) Tree -- at 1 MB and 256 GPUs, Tree's O(log N) wins": "tree",
            "C) They are identical for this message size": "same",
            "D) Neither -- you need hierarchical AllReduce": "hier",
        },
        label="256 GPUs on IB NDR. Which AllReduce is faster for a 1 MB message?",
    )
    return (pB_pred,)

@app.cell(hide_code=True)
def _(mo, pB_pred):
    pB_msg_exp = mo.ui.slider(start=0, stop=10, value=0, step=1, label="Message size (10^x KB)")
    pB_n_gpus = mo.ui.dropdown(
        options={"64": 64, "256": 256, "1024": 1024},
        value="256", label="GPU count",
    )

    pC_pred = mo.ui.radio(
        options={
            "A) ~1.5x -- marginal improvement": "1.5",
            "B) ~2x -- moderate improvement": "2",
            "C) ~5-6x -- dramatic improvement": "5",
            "D) ~18x -- full NVLink/IB ratio": "18",
        },
        label="Hierarchical AllReduce vs flat ring for 64 GPUs (8 nodes x 8). Speedup?",
    )
    return (pC_pred,)

@app.cell(hide_code=True)
def _(mo, pC_pred):
    pC_topo = mo.ui.dropdown(
        options={"Flat Ring": "flat", "Hierarchical 2-level": "hier2"},
        value="Flat Ring", label="Topology",
    )
    pC_gpus_per_node = mo.ui.slider(start=2, stop=8, value=8, step=2, label="GPUs per node")
    pC_oversub = mo.ui.dropdown(
        options={"1:1 (full bisection)": 1, "2:1": 2, "4:1": 4},
        value="1:1 (full bisection)", label="Oversubscription",
    )

    pD_pred = mo.ui.radio(
        options={
            "A) Yes, by ~4x -- 75% bandwidth saved": "4x",
            "B) Yes, by ~2x": "2x",
            "C) It depends -- on IB NDR, compression barely helps": "depends",
            "D) No -- compression always hurts": "no",
        },
        label="INT8 gradient compression (4x BW reduction) for 70B on 64 GPUs with IB NDR. "
              "Does total training time decrease?",
    )
    return (pD_pred,)

@app.cell(hide_code=True)
def _(mo, pD_pred):
    pD_comp = mo.ui.dropdown(
        options={"None": 1.0, "FP16 (2x)": 0.5, "INT8 (4x)": 0.25, "Top-K 1%": 0.01, "1-bit": 0.03125},
        value="None", label="Compression method",
    )
    pD_bw = mo.ui.slider(start=10, stop=100, value=50, step=5, label="Network BW (GB/s)")

    pE_pred = mo.ui.radio(
        options={
            "A) Just one -- hierarchical AllReduce is enough": "1",
            "B) Two -- hierarchical + FP16": "2",
            "C) Three or four -- hier + FP16 + overlap + fusion": "3",
            "D) Impossible on IB NDR": "impossible",
        },
        label="Starting from 11s AllReduce for 70B: how many optimizations "
              "to get under 20% of step time?",
    )
    return (pE_pred,)

@app.cell(hide_code=True)
def _(mo, pE_pred):
    pE_hier = mo.ui.checkbox(label="Hierarchical AllReduce")
    pE_fp16 = mo.ui.checkbox(label="FP16 gradients")
    pE_bucket = mo.ui.checkbox(label="Bucket fusion")
    pE_overlap = mo.ui.checkbox(label="Backward overlap")

    # ═════════════════════════════════════════════════════════════════════════
    # PART A: THE NETWORK TIME BUDGET
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Systems Engineer, DistributedAI Corp
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We need a time budget for gradient AllReduce in our 70B model training.
                InfiniBand latency is measured in microseconds. How long can it possibly take?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md(f"""
        ## The Alpha-Beta Model: T(n) = alpha + n/beta

        For LLM-scale gradients (hundreds of GB), the bandwidth term dominates by four
        to five orders of magnitude. InfiniBand latency (5 us) is irrelevant compared to
        the time to push 280 GB through a 50 GB/s pipe.

        Ring AllReduce transfers **2(N-1)/N x M** bytes total.
        """))

        items.append(pA_pred)
        if pA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pA_model, pA_prec, pA_gpus], justify="start", gap="1rem"))

        _params = pA_model.value * 1e9
        _bpp = pA_prec.value
        _n = pA_gpus.value
        _msg_bytes = _params * _bpp
        _msg_gb = _msg_bytes / 1e9

        _ar = calc_ring_allreduce_time(_msg_bytes, _n, INFINIBAND_NDR_BW_GBS * 1e9, IB_NDR_LATENCY_US * 1e-6)
        _t_total_ms = _ar.m_as(ureg.millisecond)

        # Bandwidth vs latency breakdown
        _bw_coeff = 2 * (_n - 1) / _n
        _t_bw_ms = (_bw_coeff * _msg_gb / INFINIBAND_NDR_BW_GBS) * 1000
        _t_lat_ms = (2 * (_n - 1) * IB_NDR_LATENCY_US * 1e-6) * 1000
        _bw_pct = _t_bw_ms / _t_total_ms * 100 if _t_total_ms > 0 else 0

        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Bandwidth term", x=["AllReduce"], y=[_t_bw_ms],
                              marker_color=COLORS["RedLine"], width=0.4))
        _fig.add_trace(go.Bar(name="Latency term", x=["AllReduce"], y=[_t_lat_ms],
                              marker_color=COLORS["BlueLine"], width=0.4))
        _fig.update_layout(barmode="stack", height=280,
                           yaxis=dict(title="Time (ms)", type="log", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Total AllReduce</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">{_t_total_ms:,.0f} ms</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_t_total_ms/1000:.1f} seconds</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Bandwidth %</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_bw_pct:.2f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Message Size</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_msg_gb:.0f} GB</div>
            </div>
        </div>"""))

        items.append(mo.md(f"""
**Alpha-Beta -- Live Calculation** (`{pA_model.value}B, {_bpp}B/param, {_n} GPUs`)

```
M            = {_params:.0e} params x {_bpp} bytes = {_msg_gb:.0f} GB
T_bandwidth  = 2({_n}-1)/{_n} x {_msg_gb:.0f} GB / {INFINIBAND_NDR_BW_GBS} GB/s = {_t_bw_ms:,.0f} ms
T_latency    = 2({_n}-1) x {IB_NDR_LATENCY_US} us = {_t_lat_ms:.2f} ms
Total        = {_t_total_ms:,.0f} ms ({_t_total_ms/1000:.1f} s)
BW fraction  = {_bw_pct:.4f}%
```
*Source: Vol II Ch 3/6 -- Alpha-Beta Model*
        """))

        _pred = pA_pred.value
        if _pred == "11000":
            _msg = "**Correct.** Bandwidth completely dominates. The latency term is negligible."
            _kind = "success"
        else:
            _msg = (f"**AllReduce takes {_t_total_ms/1000:.1f} seconds.** Students anchor on "
                    "InfiniBand's microsecond latency and underestimate by 100-1000x.")
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B: RING VS TREE CROSSOVER
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; NCCL Engineer, DistributedAI Corp
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We default to Ring AllReduce everywhere. Someone suggested Tree
                might be faster for small messages. When does each win?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Ring vs Tree: Bandwidth-Optimal vs Latency-Optimal

        - **Ring**: BW-optimal (2(N-1)/N x M/beta) but O(N) latency steps
        - **Tree**: O(log N) latency steps but O(log N) bandwidth overhead

        The crossover depends on message size, GPU count, and network parameters.
        """))

        items.append(pB_pred)
        if pB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pB_msg_exp, pB_n_gpus], justify="start", gap="2rem"))

        _n = pB_n_gpus.value
        _msg_kb = 10 ** pB_msg_exp.value
        _msg_bytes = _msg_kb * 1024

        # Sweep message sizes for both algorithms
        _sizes_kb = np.logspace(0, 10, 200)  # 1 KB to 10 GB
        _ring_ms = []
        _tree_ms = []
        for _s in _sizes_kb:
            _sb = _s * 1024
            _r = calc_ring_allreduce_time(_sb, _n, INFINIBAND_NDR_BW_GBS * 1e9, IB_NDR_LATENCY_US * 1e-6)
            _t = calc_tree_allreduce_time(_sb, _n, INFINIBAND_NDR_BW_GBS * 1e9, IB_NDR_LATENCY_US * 1e-6)
            _ring_ms.append(_r.m_as(ureg.millisecond))
            _tree_ms.append(_t.m_as(ureg.millisecond))

        _ring_ms = np.array(_ring_ms)
        _tree_ms = np.array(_tree_ms)

        # Find crossover
        _diff = _ring_ms - _tree_ms
        _cross_idx = np.argmax(_diff < 0)  # Ring becomes faster
        _cross_kb = _sizes_kb[_cross_idx] if _cross_idx > 0 else _sizes_kb[-1]

        # Current point
        _r_cur = calc_ring_allreduce_time(_msg_bytes, _n, INFINIBAND_NDR_BW_GBS * 1e9, IB_NDR_LATENCY_US * 1e-6)
        _t_cur = calc_tree_allreduce_time(_msg_bytes, _n, INFINIBAND_NDR_BW_GBS * 1e9, IB_NDR_LATENCY_US * 1e-6)
        _ring_cur_ms = _r_cur.m_as(ureg.millisecond)
        _tree_cur_ms = _t_cur.m_as(ureg.millisecond)
        _winner = "Ring" if _ring_cur_ms < _tree_cur_ms else "Tree"

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_sizes_kb, y=_ring_ms, mode="lines",
                                  name="Ring AllReduce", line=dict(color=COLORS["BlueLine"], width=2.5)))
        _fig.add_trace(go.Scatter(x=_sizes_kb, y=_tree_ms, mode="lines",
                                  name="Tree AllReduce", line=dict(color=COLORS["OrangeLine"], width=2.5)))
        _fig.add_vline(x=_cross_kb, line_dash="dash", line_color=COLORS["GreenLine"],
                       annotation_text=f"Crossover: {_cross_kb:.0f} KB", annotation_position="top right",
                       annotation_font_size=10)
        _fig.add_trace(go.Scatter(x=[_msg_kb], y=[min(_ring_cur_ms, _tree_cur_ms)],
                                  mode="markers", name=f"Your msg: {_msg_kb:.0f} KB",
                                  marker=dict(color=COLORS["RedLine"], size=14, symbol="star",
                                              line=dict(color="white", width=2))))
        _fig.update_layout(height=380,
                           xaxis=dict(title="Message Size (KB)", type="log", gridcolor="#f1f5f9"),
                           yaxis=dict(title="Time (ms)", type="log", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Ring Time</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_ring_cur_ms:.2f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Tree Time</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_tree_cur_ms:.2f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Winner</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_winner}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['TextMuted']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Crossover</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['TextMuted']};">{_cross_kb:.0f} KB</div>
            </div>
        </div>"""))

        _pred = pB_pred.value
        if _pred == "tree":
            _msg = ("**Correct.** At 1 MB with 256 GPUs, Tree's O(log N) latency advantage wins. "
                    "Ring incurs 510 latency steps vs Tree's 16.")
            _kind = "success"
        else:
            _msg = (f"**Tree wins at 1 MB / 256 GPUs.** Ring is BW-optimal but its O(N) latency "
                    "penalty is catastrophic at large GPU counts for small messages.")
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C: TOPOLOGY AND HIERARCHY EFFECTS
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_c():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Performance Lead, DistributedAI Corp
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our flat Ring AllReduce across 64 GPUs mixes NVLink and IB links.
                A hierarchical approach does local reduce first, then inter-node. How much faster?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md(f"""
        ## Hierarchical AllReduce Exploits the Bandwidth Cliff

        NVLink ({NVLINK_GBS:.0f} GB/s) vs IB NDR ({INFINIBAND_NDR_BW_GBS} GB/s) = {NVLINK_GBS/INFINIBAND_NDR_BW_GBS:.0f}x gap.

        Hierarchical AllReduce: local reduce within NVLink, then global AllReduce over IB.
        Reduces inter-node traffic by a factor of GPUs-per-node.
        """))

        items.append(pC_pred)
        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pC_topo, pC_gpus_per_node], justify="start", gap="2rem"))
        items.append(pC_oversub)

        _gpn = pC_gpus_per_node.value
        _total = 64
        _n_nodes = _total // _gpn
        _oversub = pC_oversub.value
        _msg_bytes = 70e9 * 2  # 70B FP16
        _ib_bw_effective = INFINIBAND_NDR_BW_GBS / _oversub  # oversubscription effect

        # Flat ring
        _flat = calc_ring_allreduce_time(
            _msg_bytes, _total, INFINIBAND_NDR_BW_GBS * 1e9, IB_NDR_LATENCY_US * 1e-6
        )
        _flat_ms = _flat.m_as(ureg.millisecond)

        # Hierarchical
        _hier = calc_hierarchical_allreduce_time(
            _msg_bytes, _n_nodes, _gpn,
            NVLINK_GBS * 1e9 * ureg.byte / ureg.second,
            _ib_bw_effective * 1e9 * ureg.byte / ureg.second,
        )
        _hier_ms = _hier.m_as(ureg.millisecond)
        _speedup = _flat_ms / _hier_ms if _hier_ms > 0 else 0

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Flat Ring</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">{_flat_ms:,.0f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Hierarchical</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_hier_ms:,.0f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Speedup</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_speedup:.1f}x</div>
            </div>
        </div>"""))

        if _oversub > 1:
            items.append(mo.callout(mo.md(
                f"**Oversubscription {_oversub}:1** reduces effective inter-node bandwidth "
                f"from {INFINIBAND_NDR_BW_GBS} to {_ib_bw_effective:.1f} GB/s, "
                "proportionally slowing the inter-node component."
            ), kind="warn"))

        _pred = pC_pred.value
        if _pred == "5":
            _msg = f"**Correct.** Hierarchical achieves {_speedup:.1f}x speedup by confining most traffic to NVLink."
            _kind = "success"
        else:
            _msg = f"**Hierarchical achieves {_speedup:.1f}x speedup.** Reducing inter-node traffic by {_gpn}x is the key."
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D: GRADIENT COMPRESSION
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; ML Researcher, DistributedAI Corp
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We applied INT8 gradient compression. Per-step communication dropped 4x.
                But total training time barely changed. What happened?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Gradient Compression: When Does It Pay Off?

        Compression trades bandwidth savings for convergence slowdown.
        It only helps when the **communication-to-computation ratio is high**.
        On fast networks, the extra convergence steps nearly cancel per-step savings.
        """))

        items.append(pD_pred)
        if pD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pD_comp, pD_bw], justify="start", gap="2rem"))

        _comp_ratio = pD_comp.value
        _bw = pD_bw.value
        _msg_gb = 70 * 2  # 70B FP16 baseline = 140 GB
        _n = 64

        # Convergence penalty
        _penalties = {1.0: 1.0, 0.5: 1.05, 0.25: 1.15, 0.01: 1.3, 0.03125: 1.5}
        _conv_penalty = _penalties.get(_comp_ratio, 1.0)

        _compressed_gb = _msg_gb * _comp_ratio
        _t_comp_ms = 5000  # 5 seconds compute per step (approximate for 70B)
        _t_comm_ms = (2 * (_n - 1) / _n * _compressed_gb / _bw) * 1000
        _t_step_ms = _t_comp_ms + _t_comm_ms
        _comm_pct = _t_comm_ms / _t_step_ms * 100

        # Total training time (relative)
        _base_comm_ms = (2 * (_n - 1) / _n * _msg_gb / _bw) * 1000
        _base_step_ms = _t_comp_ms + _base_comm_ms
        _base_total = _base_step_ms * 1000  # 1000 steps baseline
        _new_total = _t_step_ms * 1000 * _conv_penalty
        _time_ratio = _new_total / _base_total

        _net_benefit = _time_ratio < 1.0

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Comm Time/Step</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_t_comm_ms:,.0f} ms</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Conv. Penalty</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_conv_penalty:.2f}x</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white; border-top:3px solid {COLORS['GreenLine'] if _net_benefit else COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Net Time Change</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine'] if _net_benefit else COLORS['RedLine']};">{(_time_ratio-1)*100:+.0f}%</div>
            </div>
        </div>"""))

        if not _net_benefit and _comp_ratio < 1.0:
            items.append(mo.callout(mo.md(
                "**Compression hurts here.** The convergence penalty outweighs per-step savings. "
                "Try a slower network (lower BW slider) where communication is the dominant term."
            ), kind="danger"))

        _pred = pD_pred.value
        if _pred == "depends":
            _msg = ("**Correct.** On fast networks (IB NDR), communication is only 30-40% of step time. "
                    "The convergence penalty nearly cancels per-step savings. "
                    "On slow networks (100GbE), compression provides substantial benefit.")
            _kind = "success"
        else:
            _msg = ("**It depends on the network.** Compression trades per-step savings for "
                    "extra steps. On IB NDR, the trade-off is marginal. On slow networks, it wins.")
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART E: COMMUNICATION BUDGET OPTIMIZATION
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_e():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid #6366f1; background:#f0f4ff;
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; CTO, DistributedAI Corp
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Raw AllReduce for our 70B model takes 11 seconds. Our training step
                is 30 seconds. Communication is 37% of step time. Get it under 20%.&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Stack Optimizations to Hit the 20% Target

        Starting point: 11-second raw Ring AllReduce for 70B FP32 on 64 GPUs.
        Toggle optimizations one at a time and watch each chip away at the budget.
        """))

        items.append(pE_pred)
        if pE_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pE_hier, pE_fp16, pE_bucket, pE_overlap], justify="start", gap="1rem"))

        _base_ms = 11000  # 11 seconds raw AllReduce
        _compute_ms = 19000  # ~19 seconds compute
        _comm_ms = _base_ms

        # Apply optimizations
        if pE_hier.value:
            _comm_ms *= 0.18  # ~5-6x reduction
        if pE_fp16.value:
            _comm_ms *= 0.5
        if pE_bucket.value:
            _comm_ms *= 0.9  # 10% latency reduction from fusion
        _visible_comm_ms = _comm_ms
        if pE_overlap.value:
            _visible_comm_ms = _comm_ms * (1 - float(DEFAULT_OVERLAP_EFFICIENCY))

        _step_ms = _compute_ms + _visible_comm_ms
        _comm_pct = _visible_comm_ms / _step_ms * 100 if _step_ms > 0 else 0
        _target_met = _comm_pct < 20

        # Progress chart
        _stages = ["Raw"]
        _times = [_base_ms]
        _cur = _base_ms
        if pE_hier.value:
            _cur *= 0.18
            _stages.append("+Hierarchical")
            _times.append(_cur)
        if pE_fp16.value:
            _cur *= 0.5
            _stages.append("+FP16")
            _times.append(_cur)
        if pE_bucket.value:
            _cur *= 0.9
            _stages.append("+Bucket")
            _times.append(_cur)
        if pE_overlap.value:
            _cur *= (1 - float(DEFAULT_OVERLAP_EFFICIENCY))
            _stages.append("+Overlap")
            _times.append(_cur)

        _fig = go.Figure()
        _bar_cols = [COLORS["RedLine"] if t / (_compute_ms + t) > 0.2 else COLORS["GreenLine"] for t in _times]
        _fig.add_trace(go.Bar(x=_stages, y=[t / 1000 for t in _times],
                              marker_color=_bar_cols, width=0.5))
        _target_time_s = 0.2 * _compute_ms / (1 - 0.2) / 1000  # 20% threshold
        _fig.add_hline(y=_target_time_s, line_dash="dash", line_color=COLORS["GreenLine"],
                       annotation_text="20% target", annotation_font_size=10)
        _fig.update_layout(height=300, yaxis=dict(title="Communication (seconds)", gridcolor="#f1f5f9"),
                           margin=dict(l=50, r=20, t=30, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _tc = COLORS["GreenLine"] if _target_met else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {_tc}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Comm % of Step</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_tc};">{_comm_pct:.1f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">Target: &lt; 20%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Effective Comm</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">{_visible_comm_ms/1000:.1f} s</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:140px; text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Reduction from Raw</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{_base_ms/_visible_comm_ms:.0f}x</div>
            </div>
        </div>"""))

        if _target_met:
            items.append(mo.callout(mo.md(
                f"**Target met.** Communication is {_comm_pct:.1f}% of step time. "
                f"You stacked {sum([pE_hier.value, pE_fp16.value, pE_bucket.value, pE_overlap.value])} optimizations."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Not yet.** Communication is still {_comm_pct:.1f}% of step time. "
                "Toggle more optimizations to reach the 20% target."
            ), kind="warn"))

        _pred = pE_pred.value
        if _pred == "3":
            _msg = ("**Correct.** You typically need 3-4 optimizations stacked: hierarchical + "
                    "FP16 + overlap + bucket fusion. No single optimization is sufficient.")
            _kind = "success"
        else:
            _msg = ("**You need 3-4 stacked optimizations.** Hierarchical gives ~5-6x. "
                    "FP16 halves it. Overlap hides 85%. Bucket fusion reduces latency. "
                    "All are required to reach <20%.")
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════════

    def build_synthesis():
        return mo.vstack([
            mo.md("## Key Takeaways"),
            mo.callout(mo.md(
                "**1. AllReduce is a bandwidth problem.** For 70B FP32 gradients on 64 GPUs, "
                "Ring AllReduce takes ~11 seconds. The bandwidth term is 99.99% of total. "
                "InfiniBand latency (5 us) is irrelevant at LLM scale."
            ), kind="info"),
            mo.callout(mo.md(
                "**2. Algorithm choice depends on message size and GPU count.** "
                "Ring is bandwidth-optimal but latency-poor (O(N) steps). "
                "Tree has logarithmic latency but bandwidth overhead. "
                "The crossover formula determines which wins."
            ), kind="info"),
            mo.callout(mo.md(
                "**3. No single optimization is sufficient.** "
                "Reaching <20% communication requires stacking: hierarchical AllReduce (~5-6x), "
                "FP16 gradients (2x), backward overlap (85%), and bucket fusion. "
                "This is the Megatron-LM recipe."
            ), kind="info"),
            mo.md("""
## Connections

**Textbook:** Vol II Ch 3 (Network Fabrics) + Ch 6 (Collective Communication).

**Next Lab:** V2-04 explores the data pipeline wall: the storage-compute chasm,
shard contention, prefetching limits, and checkpoint economics.
            """),
        ])

    # ═════════════════════════════════════════════════════════════════════════
    # COMPOSE TABS
    # ═════════════════════════════════════════════════════════════════════════

    tabs = mo.ui.tabs({
        "Part A -- The Network Time Budget": build_part_a(),
        "Part B -- Ring vs Tree Crossover": build_part_b(),
        "Part C -- Topology and Hierarchy": build_part_c(),
        "Part D -- Gradient Compression": build_part_d(),
        "Part E -- Communication Budget": build_part_e(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return


@app.cell(hide_code=True)
def _(COLORS, ledger, mo):
    _track = ledger._state.track or "not set"
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">V2-03 &middot; Communication at Scale</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">VOL&nbsp;II&nbsp;CH&nbsp;3+6</span>
        <span class="hud-value">Communication at Scale</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
