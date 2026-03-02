import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 07: THE CHECKPOINT RECKONING
#
# Chapter: Fault Tolerance in Distributed Training (@sec-fault-tolerance)
# Core Invariant: Young-Daly formula — optimal checkpoint interval
#                 T* = sqrt(2 × C / λ)
#                 where C = checkpoint cost (hours), λ = cluster failure rate (/hr)
#
# 2-Act Structure (35-40 minutes):
#   Act I  — The Checkpoint Cost Blindspot (12-15 min)
#             Stakeholder: Training Platform Lead with a 70B model on 1024 GPUs.
#             "We checkpoint every 30 minutes. Is that right?"
#             Answer: No — Young-Daly optimal is ~60 min; current overhead is 2× too high.
#
#   Act II — Fleet-Scale Checkpointing (20-25 min)
#             Stakeholder: Meta/Google-scale lead, 1T param model on 16,384 H100s.
#             Design a checkpointing strategy from first principles.
#             Failure state: checkpoint overhead > 20% of training time.
#
# Deployment Contexts:
#   Small Cluster: 1,024-GPU training cluster (realistic research/enterprise scale)
#   Large Cluster: 16,384-GPU fleet (hyperscaler-scale production training)
#
# Design Ledger: saves chapter="v2_07" with context, intervals, overhead, decisions.
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

    # ── Hardware constants (sources documented inline) ──────────────────────
    H100_RAM_GB         = 80       # GB HBM3e; source: NVIDIA H100 SXM5 spec sheet
    H100_MTBF_HOURS     = 200      # hours, GPU MTBF in data center; source: @sec-fault-tolerance
    LUSTRE_WRITE_GBS    = 400      # GB/s Lustre aggregate write BW; source: @sec-fault-tolerance
    IB_HDR200_BW_GBS    = 400      # GB/s InfiniBand HDR 200; source: @sec-network-fabrics

    return (
        COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        ledger,
        H100_RAM_GB, H100_MTBF_HOURS, LUSTRE_WRITE_GBS, IB_HDR200_BW_GBS,
        mo,
    )


# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, LAB_CSS, mo):
    _c_cloud  = COLORS["Cloud"]
    _c_edge   = COLORS["Edge"]
    _c_surf0  = COLORS["Surface0"]
    _c_surf1  = COLORS["Surface1"]

    _header = mo.Html(f"""
    {LAB_CSS}
    <div style="background: linear-gradient(135deg, {_c_surf0} 0%, {_c_surf1} 100%);
                border-radius: 16px; padding: 32px 40px; margin-bottom: 8px;
                border: 1px solid #2d3748;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;
                    flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                            text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;">
                    Vol 2 · Lab 07 · Fault Tolerance
                </div>
                <div style="font-size: 2.0rem; font-weight: 800; color: #f1f5f9;
                            line-height: 1.15; margin-bottom: 10px;">
                    The Checkpoint Reckoning
                </div>
                <div style="font-size: 0.95rem; color: #94a3b8; max-width: 600px; line-height: 1.6;">
                    At 16,384 GPUs, your cluster fails every 44 minutes on average.
                    Checkpoint too rarely and you lose hours of work. Checkpoint too often
                    and 40% of your training time is spent writing to storage.
                    Young-Daly gives you the exact optimum — if you know your numbers.
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;">
                <span class="badge badge-info">Young-Daly: T* = sqrt(2C / λ)</span>
                <span class="badge badge-info">Checkpoint Overhead = C / T</span>
                <span class="badge badge-warn">35-40 minutes · 2 Acts</span>
            </div>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap;">
            <div style="background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_cloud}; font-weight: 700;">Small Cluster</span>
                <span style="color: #94a3b8;">
                    &nbsp;— 1,024 H100s · Cluster MTBF ~8 hrs · 70B param model
                </span>
            </div>
            <div style="background: rgba(203,32,45,0.12); border: 1px solid rgba(203,32,45,0.35);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_edge}; font-weight: 700;">Large Cluster</span>
                <span style="color: #94a3b8;">
                    &nbsp;— 16,384 H100s · Cluster MTBF ~44 min · 1T param model
                </span>
            </div>
        </div>
    </div>
    """)
    _header
    return


# ─── CELL 2: RECOMMENDED READING ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-fault-tolerance-failure-modes** — GPU and node failure rates at scale; how MTBF
      degrades as cluster size grows
    - **@sec-fault-tolerance-checkpointing** — Checkpoint cost model; synchronous vs. asynchronous
      write strategies
    - **@sec-fault-tolerance-young-daly** — Derivation of the Young-Daly formula and its
      assumptions; expected wasted time E[W] = T/2 + C
    - **@sec-fault-tolerance-at-scale** — Multi-level checkpointing; how hyperscalers achieve
      sub-minute effective checkpoint cost via async I/O overlap
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    context_toggle = mo.ui.radio(
        options={
            "Small Cluster (1,024 GPUs)": "small",
            "Large Cluster (16,384 GPUs)": "large",
        },
        value="Small Cluster (1,024 GPUs)",
        label="Deployment context for this session:",
        inline=True,
    )
    mo.hstack([
        mo.Html(f"""
        <div style="font-size:0.78rem; font-weight:700; color:{COLORS['TextMuted']};
                    text-transform:uppercase; letter-spacing:0.08em;
                    margin-right:8px; padding-top:2px;">
            Active Context:
        </div>
        """),
        context_toggle,
    ], justify="start", gap=0)
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ACT I — THE CHECKPOINT COST BLINDSPOT
# ═══════════════════════════════════════════════════════════════════════════════


# ─── ACT I: SECTION HEADER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Act I — The Checkpoint Cost Blindspot
    *Calibration · 12-15 minutes*
    """)
    return


# ─── ACT I: STAKEHOLDER MESSAGE ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _color = COLORS["Cloud"]
    _bg    = COLORS["BlueL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Training Platform Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We're training a 70B parameter model on our 1,024-GPU cluster. We checkpoint
            every 1,000 steps — roughly every 30 minutes. Each checkpoint takes 4 minutes
            to write to NFS. The cluster fails on average once every 8 hours. Our cluster
            utilization has dropped lately. An engineer suggested we're checkpointing too
            often, but I don't want to lose hours of work if we fail. Are we at the right
            frequency?"
        </div>
    </div>
    """)
    return


# ─── ACT I: CONCEPT FRAMING ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    This is not a question about caution — it is a question about optimization.
    The **Young-Daly formula** from @sec-fault-tolerance-young-daly gives the
    mathematically optimal checkpoint interval by minimizing the expected total
    wasted time per hour of training.

    The expected wasted time when a failure occurs is:

    > **E[wasted time] = T/2 + C**

    where **T** is the checkpoint interval and **C** is the checkpoint write time.
    The T/2 term is the expected rework (you fail uniformly within the interval),
    and C is the time to re-load the last checkpoint and resume.

    Minimizing over T, weighted by the failure rate λ, yields:

    > **T\\* = sqrt(2 × C / λ)**

    The key insight is that checkpointing too *frequently* wastes training time
    on I/O overhead (C/T per interval), while checkpointing too *infrequently*
    wastes training time on rework after failures (T/2 × λ per hour).

    Before running the numbers, commit to a prediction.
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
            "A) 30 minutes is too infrequent — checkpoint every 10 minutes for safety": "A",
            "B) 30 minutes is too frequent — Young-Daly optimal is approximately 60 minutes": "B",
            "C) 30 minutes is already optimal for a 1,024-GPU cluster with 8-hour MTBF": "C",
            "D) Checkpoint frequency has minimal impact on training efficiency": "D",
        },
        label="Given C = 4 min and cluster MTBF = 8 hours, where does T = 30 minutes fall relative to Young-Daly optimal?",
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


# ─── ACT I: YOUNG-DALY EXPLORER ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Young-Daly Explorer")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_mtbf_slider = mo.ui.slider(
        start=1, stop=100, value=8, step=1,
        label="Cluster MTBF (hours)",
        show_value=True,
    )
    act1_ckpt_cost_slider = mo.ui.slider(
        start=0.5, stop=30.0, value=4.0, step=0.5,
        label="Checkpoint write time C (minutes)",
        show_value=True,
    )
    act1_current_interval_slider = mo.ui.slider(
        start=5, stop=120, value=30, step=5,
        label="Your current checkpoint interval T (minutes)",
        show_value=True,
    )
    mo.vstack([
        mo.hstack([act1_mtbf_slider, act1_ckpt_cost_slider], justify="start", gap=4),
        act1_current_interval_slider,
    ])
    return (act1_mtbf_slider, act1_ckpt_cost_slider, act1_current_interval_slider)


@app.cell(hide_code=True)
def _(
    COLORS,
    act1_ckpt_cost_slider,
    act1_current_interval_slider,
    act1_mtbf_slider,
    apply_plotly_theme,
    go,
    math,
    mo,
    np,
):
    # ── Pull slider values ─────────────────────────────────────────────────
    _mtbf_hr   = act1_mtbf_slider.value          # hours
    _C_min     = act1_ckpt_cost_slider.value      # minutes
    _T_cur_min = act1_current_interval_slider.value  # minutes

    # ── Convert to hours for all calculations ──────────────────────────────
    _C_hr      = _C_min / 60.0                   # checkpoint cost, hours
    _lambda    = 1.0 / _mtbf_hr                  # failure rate, failures/hour
    _T_cur_hr  = _T_cur_min / 60.0               # current interval, hours

    # ── Young-Daly optimal interval ────────────────────────────────────────
    # T* = sqrt(2 * C / lambda)  [from @sec-fault-tolerance-young-daly]
    _T_opt_hr  = math.sqrt(2.0 * _C_hr / _lambda)
    _T_opt_min = _T_opt_hr * 60.0

    # ── Overhead fractions ─────────────────────────────────────────────────
    # Checkpoint overhead fraction = C / T  (fraction of time spent on I/O)
    _overhead_cur_pct = (_C_hr / _T_cur_hr) * 100.0
    _overhead_opt_pct = (_C_hr / _T_opt_hr) * 100.0

    # ── Expected wasted time per failure (minutes) ─────────────────────────
    # E[wasted] = T/2 + C  [derivation: @sec-fault-tolerance-young-daly]
    _waste_cur_min  = _T_cur_min / 2.0 + _C_min
    _waste_opt_min  = _T_opt_min / 2.0 + _C_min

    # ── Expected wasted time per hour (cost rate) ──────────────────────────
    # Rate = lambda * E[wasted_hr]  (hours wasted / hour of training)
    _waste_rate_cur = _lambda * (_T_cur_hr / 2.0 + _C_hr)
    _waste_rate_opt = _lambda * (_T_opt_hr / 2.0 + _C_hr)

    # ── Total cost fraction (overhead + expected rework per hour) ──────────
    _total_cur = _overhead_cur_pct / 100.0 + _waste_rate_cur
    _total_opt = _overhead_opt_pct / 100.0 + _waste_rate_opt

    # ── Color coding ───────────────────────────────────────────────────────
    _ratio = _T_cur_min / _T_opt_min
    if 0.7 <= _ratio <= 1.4:
        _interval_color = COLORS["GreenLine"]
        _interval_label = "Near-optimal"
    elif _ratio < 0.7:
        _interval_color = COLORS["OrangeLine"]
        _interval_label = "Too frequent (excess I/O overhead)"
    else:
        _interval_color = COLORS["RedLine"]
        _interval_label = "Too infrequent (excess rework risk)"

    _ovh_color = COLORS["GreenLine"] if _overhead_cur_pct < 8 else (
        COLORS["OrangeLine"] if _overhead_cur_pct < 15 else COLORS["RedLine"]
    )

    # ── Overhead vs interval curve ─────────────────────────────────────────
    _T_range_min = np.linspace(1.0, 180.0, 400)
    _T_range_hr  = _T_range_min / 60.0

    # Total cost = overhead fraction + expected rework rate
    # total(T) = C/T + lambda*(T/2 + C)
    _total_curve = (_C_hr / _T_range_hr) + _lambda * (_T_range_hr / 2.0 + _C_hr)

    _fig = go.Figure()

    # Total cost curve
    _fig.add_trace(go.Scatter(
        x=_T_range_min,
        y=_total_curve * 100.0,  # as percentage
        mode="lines",
        name="Total wasted fraction",
        line=dict(color=COLORS["BlueLine"], width=2.5),
    ))

    # Overhead-only curve (C/T)
    _overhead_curve = (_C_hr / _T_range_hr) * 100.0
    _fig.add_trace(go.Scatter(
        x=_T_range_min,
        y=_overhead_curve,
        mode="lines",
        name="Checkpoint I/O overhead",
        line=dict(color=COLORS["OrangeLine"], width=1.5, dash="dot"),
    ))

    # Rework-only curve (lambda * (T/2 + C))
    _rework_curve = _lambda * (_T_range_hr / 2.0 + _C_hr) * 100.0
    _fig.add_trace(go.Scatter(
        x=_T_range_min,
        y=_rework_curve,
        mode="lines",
        name="Expected rework cost",
        line=dict(color=COLORS["RedLine"], width=1.5, dash="dot"),
    ))

    # Optimal marker
    _fig.add_trace(go.Scatter(
        x=[_T_opt_min],
        y=[_total_opt * 100.0],
        mode="markers",
        name=f"Young-Daly T* = {_T_opt_min:.0f} min",
        marker=dict(color=COLORS["GreenLine"], size=14, symbol="diamond",
                    line=dict(color="white", width=2)),
    ))

    # Current interval marker
    _fig.add_trace(go.Scatter(
        x=[_T_cur_min],
        y=[_total_cur * 100.0],
        mode="markers",
        name=f"Your T = {_T_cur_min} min",
        marker=dict(color=_interval_color, size=12, symbol="circle",
                    line=dict(color="white", width=2)),
    ))

    # Vertical line at optimal
    _fig.add_vline(
        x=_T_opt_min,
        line_dash="dash",
        line_color=COLORS["GreenLine"],
        opacity=0.5,
    )

    _fig.update_layout(
        height=340,
        xaxis=dict(title="Checkpoint interval T (minutes)", range=[0, 180]),
        yaxis=dict(title="Wasted fraction of training time (%)", range=[0, None]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=50, l=50, r=20),
    )
    apply_plotly_theme(_fig)

    # ── Physics formula display ────────────────────────────────────────────
    _formula_block = mo.Html(f"""
    <div class="lab-card" style="margin: 8px 0; font-family: 'SF Mono', monospace; font-size: 0.88rem;">
        <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
            Young-Daly Physics
        </div>
        <div style="line-height: 2.0; color: {COLORS['Text']};">
            <div>T* = sqrt(2 &times; C / &lambda;)
                = sqrt(2 &times; {_C_hr:.4f} hr / {_lambda:.4f} /hr)
                = sqrt({2*_C_hr/_lambda:.4f})
                = <strong style="color:{COLORS['GreenLine']};">{_T_opt_min:.1f} min</strong>
            </div>
            <div style="color:{COLORS['TextSec']}; font-size:0.83rem; margin-top:4px;">
                where C = {_C_min:.1f} min = {_C_hr:.4f} hr &nbsp;|&nbsp;
                &lambda; = 1 / {_mtbf_hr} hr = {_lambda:.4f} failures/hr
            </div>
        </div>
        <div style="margin-top:12px; padding-top:12px; border-top:1px solid {COLORS['Border']};
                    display:grid; grid-template-columns:1fr 1fr; gap:12px; font-size:0.85rem;">
            <div>
                <div style="color:{COLORS['TextMuted']}; font-size:0.72rem;
                            font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">
                    Your interval T = {_T_cur_min} min
                </div>
                <div>I/O overhead: <strong style="color:{_ovh_color};">{_overhead_cur_pct:.1f}%</strong></div>
                <div>Expected rework: <strong>{_waste_cur_min:.1f} min</strong> per failure</div>
                <div>Status: <strong style="color:{_interval_color};">{_interval_label}</strong></div>
            </div>
            <div>
                <div style="color:{COLORS['TextMuted']}; font-size:0.72rem;
                            font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">
                    Optimal T* = {_T_opt_min:.1f} min
                </div>
                <div>I/O overhead: <strong style="color:{COLORS['GreenLine']};">{_overhead_opt_pct:.1f}%</strong></div>
                <div>Expected rework: <strong>{_waste_opt_min:.1f} min</strong> per failure</div>
                <div>Status: <strong style="color:{COLORS['GreenLine']};">Young-Daly optimal</strong></div>
            </div>
        </div>
    </div>
    """)

    mo.vstack([_formula_block, mo.ui.plotly(_fig)])
    return (
        _T_opt_min, _T_opt_hr,
        _overhead_cur_pct, _overhead_opt_pct,
        _waste_cur_min, _waste_opt_min,
        _C_hr, _C_min, _lambda, _mtbf_hr,
        _T_cur_min, _T_cur_hr,
        _interval_color, _interval_label,
    )


# ─── ACT I: PREDICTION REVEAL ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS,
    _C_min,
    _T_cur_min,
    _T_opt_min,
    _overhead_cur_pct,
    _overhead_opt_pct,
    act1_pred,
    mo,
):
    _correct = act1_pred.value == "B"

    _overlay_color = COLORS["GreenLine"] if _correct else COLORS["OrangeLine"]
    _ratio_val = _T_cur_min / _T_opt_min

    if _correct:
        _reveal = mo.callout(mo.md(
            f"**Correct.** "
            f"You predicted that 30 minutes is too frequent. "
            f"With C = {_C_min:.0f} min and MTBF = 8 hours, "
            f"Young-Daly gives T\\* = **{_T_opt_min:.0f} minutes** (~1 hour). "
            f"Your current interval of {_T_cur_min} min is {_ratio_val:.1f}× shorter than optimal. "
            f"The checkpoint overhead at T = {_T_cur_min} min is **{_overhead_cur_pct:.1f}%** — "
            f"nearly double the optimal overhead of {_overhead_opt_pct:.1f}%. "
            f"You are spending an extra {_overhead_cur_pct - _overhead_opt_pct:.1f}% of GPU time "
            f"writing checkpoints that provide diminishing protection against failure."
        ), kind="success")
    elif act1_pred.value == "A":
        _reveal = mo.callout(mo.md(
            f"**Not quite.** "
            f"Checkpointing every 10 minutes would increase I/O overhead to "
            f"{(_C_min / 10.0) * 100:.0f}% — paying for {_C_min:.0f} min of writes "
            f"every 10 minutes. Young-Daly T\\* = **{_T_opt_min:.0f} minutes**. "
            f"A 10-minute interval is {_T_opt_min / 10.0:.1f}× too frequent. "
            f"The formula exists precisely to prevent this intuition — aggressively "
            f"short intervals feel safe but waste a large fraction of training time."
        ), kind="warn")
    elif act1_pred.value == "C":
        _reveal = mo.callout(mo.md(
            f"**Not quite.** "
            f"The current T = {_T_cur_min} min is not optimal — it is "
            f"{_ratio_val:.1f}× shorter than Young-Daly T\\* = {_T_opt_min:.0f} min. "
            f"The overhead gap is {_overhead_cur_pct:.1f}% vs. {_overhead_opt_pct:.1f}% optimal. "
            f"With an 8-hour cluster MTBF and a 4-minute checkpoint cost, the formula "
            f"says: 'You are over-insuring. Double the interval to halve the I/O overhead.'"
        ), kind="warn")
    else:  # D
        _reveal = mo.callout(mo.md(
            f"**Not quite.** "
            f"Checkpoint frequency has a direct, quantifiable impact. "
            f"At T = {_T_cur_min} min, your cluster spends {_overhead_cur_pct:.1f}% of GPU-hours "
            f"on checkpoint I/O. Doubling T to {_T_opt_min:.0f} min halves that overhead to "
            f"{_overhead_opt_pct:.1f}%. On a 1,024-GPU cluster running for one week, "
            f"the difference is ({_overhead_cur_pct - _overhead_opt_pct:.1f}%) × 1,024 GPUs × 168 hours "
            f"= {(_overhead_cur_pct - _overhead_opt_pct) / 100.0 * 1024 * 168:.0f} GPU-hours of recovered training time."
        ), kind="warn")

    _reveal
    return


# ─── ACT I: REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Reflection — Scaling Behavior")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) T* stays constant — failure rate doesn't depend on cluster size": "A",
            "B) T* decreases (more frequent checkpoints) — λ grows with N, so T* = sqrt(2C/λ) shrinks": "B",
            "C) T* increases — larger checkpoints mean longer C, which dominates": "C",
            "D) T* is independent of N — only the GPU count in the checkpoint matters": "D",
        },
        label="What happens to the Young-Daly optimal interval T* as cluster size N increases (holding per-GPU MTBF constant)?",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(act1_reflect, mo):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(
            mo.md("Select an answer to see the explanation."),
            kind="warn",
        ),
    )
    mo.md("")
    return


@app.cell(hide_code=True)
def _(act1_reflect, mo):
    if act1_reflect.value == "B":
        _r = mo.callout(mo.md(
            "**Correct.** "
            "Cluster failure rate λ_cluster = N × λ_per_gpu (failures are independent). "
            "As N grows, λ grows proportionally, so T\\* = sqrt(2C / λ) ∝ 1/sqrt(N). "
            "For N = 16,384 instead of 1,024, λ is 16× larger, so T\\* shrinks by "
            "sqrt(16) = **4×**. At hyperscale, you need to checkpoint much more "
            "frequently just to stay near optimal — and checkpoint cost C also grows "
            "because the model is larger, tightening the constraint further."
        ), kind="success")
    elif act1_reflect.value == "A":
        _r = mo.callout(mo.md(
            "**Not quite.** "
            "Cluster failure rate λ_cluster = N × λ_per_gpu. "
            "With 16,384 GPUs each at MTBF = 200 hours, the cluster MTBF = "
            "200 / 16,384 ≈ 0.73 hours = 44 minutes. "
            "λ grows linearly with N, so T\\* = sqrt(2C/λ) ∝ 1/sqrt(N) — "
            "T\\* shrinks as you scale up."
        ), kind="warn")
    elif act1_reflect.value == "C":
        _r = mo.callout(mo.md(
            "**Not quite.** "
            "While C does grow with model size (more parameters = more bytes to write), "
            "λ grows linearly with N while C grows sub-linearly (model size does not "
            "scale 1:1 with GPU count in practice). The net effect: T\\* = sqrt(2C/λ) "
            "shrinks because λ grows faster than C. At 16k GPUs, the failure rate "
            "increase dominates — you must checkpoint more frequently, not less."
        ), kind="warn")
    else:
        _r = mo.callout(mo.md(
            "**Not quite.** "
            "T\\* depends on both C and λ. While the number of GPUs writing the checkpoint "
            "does affect write parallelism (which can reduce C if storage scales), "
            "the dominant effect is the failure rate: λ_cluster = N × λ_per_gpu. "
            "T\\* = sqrt(2C/λ) shrinks as λ grows with N."
        ), kind="warn")
    _r
    return


# ─── ACT I: MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The Young-Daly Derivation": mo.md("""
        **Goal:** minimize the expected total wasted time per hour of training.

        **Definitions:**
        - **T** — checkpoint interval (hours)
        - **C** — checkpoint write + load time (hours); the training stall
        - **λ** — cluster failure rate (failures per hour) = 1 / MTBF
        - **N** — number of GPUs; λ_cluster = N × λ_per_gpu

        **Expected wasted time per failure:**
        ```
        E[wasted per failure] = T/2 + C
        ```
        The T/2 term: failures occur uniformly in [0, T], so on average T/2 of work
        is lost. The C term: after the failure you must re-load the checkpoint and
        resume (this is a fixed cost independent of when the failure occurs).

        **Expected wasted time per hour of training:**
        ```
        E[wasted per hour] = λ × (T/2 + C)
        ```

        **Checkpoint I/O overhead per hour:**
        ```
        overhead = C / T   (one write of cost C every T hours)
        ```

        **Total cost function to minimize:**
        ```
        f(T) = C/T + λ(T/2 + C)
        ```

        **First-order optimality condition:**
        ```
        df/dT = -C/T² + λ/2 = 0
        ⟹  T² = 2C/λ
        ⟹  T* = sqrt(2C/λ)        ← Young-Daly optimal interval
        ```

        **Multi-GPU failure rate model:**
        ```
        λ_cluster = N × λ_per_gpu
        MTBF_cluster = MTBF_per_gpu / N
        ```
        Example: N = 16,384, MTBF_per_gpu = 200 hr ⟹ MTBF_cluster ≈ 0.73 hr (44 min)

        **Effect of async checkpointing:**

        If writes overlap with training (asynchronous I/O), the effective C in the
        formula becomes the *stall* time (near zero for a complete overlap), not the
        total write time. This is the key technique enabling feasible checkpointing
        at hyperscale — see Act II.
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ACT II — FLEET-SCALE CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════════════


# ─── ACT II: SECTION HEADER ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Act II — Fleet-Scale Checkpointing
    *Design Challenge · 20-25 minutes*
    """)
    return


# ─── ACT II: STAKEHOLDER MESSAGE ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _color = COLORS["Edge"]
    _bg    = COLORS["RedL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Fleet Training Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We are training a 1-trillion-parameter model on 16,384 H100s. Each GPU holds 80 GB,
            so the full model checkpoint is roughly 2 TB of optimizer state, weights, and
            gradients. Our Lustre storage fabric writes at 400 GB/s aggregate. A naive
            synchronous checkpoint stalls training for 5 seconds per write. GPU MTBF is
            200 hours. We need to design a checkpointing strategy: how often, and should
            we use synchronous or asynchronous writes? Checkpoint overhead must stay under
            20% of training time."
        </div>
    </div>
    """)
    return


# ─── ACT II: CONCEPT FRAMING ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    At 16,384 GPUs, the cluster MTBF is not 8 hours — it is 44 minutes.
    Young-Daly now operates in a regime where both C and λ are large,
    and the trade-off between checkpoint frequency and checkpoint cost
    becomes a first-order constraint on training efficiency.

    This is a **design problem**, not just an analysis problem. You control:
    - The checkpoint interval (how often)
    - The write strategy (synchronous stall vs. asynchronous overlap)

    Before you run the instruments, commit to a strategy.
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
            "A) Checkpoint every 5 minutes — MTBF of 44 min means aggressive checkpointing is necessary": "A",
            "B) Checkpoint every 10 minutes — balance between rework risk and overhead": "B",
            "C) Young-Daly gives T* ≈ 21 min — checkpoint every 20 min with async writes overlapping training": "C",
            "D) Checkpoint every 60 minutes — MTBF of 44 min means you will almost always lose work anyway": "D",
        },
        label="With N=16,384 GPUs, MTBF_per_gpu=200 hr, C_sync=5 sec (sync stall): what is the optimal checkpointing strategy?",
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


# ─── ACT II: FLEET CHECKPOINT DESIGNER ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Fleet Checkpoint Designer")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_n_gpus_slider = mo.ui.slider(
        start=1024, stop=32768, value=16384, step=1024,
        label="Cluster size N (GPUs)",
        show_value=True,
    )
    act2_gpu_mtbf_slider = mo.ui.slider(
        start=50, stop=500, value=200, step=10,
        label="Per-GPU MTBF (hours)",
        show_value=True,
    )
    act2_model_tb_slider = mo.ui.slider(
        start=0.1, stop=5.0, value=2.0, step=0.1,
        label="Checkpoint size (TB)",
        show_value=True,
    )
    act2_storage_bw_slider = mo.ui.slider(
        start=50, stop=2000, value=400, step=50,
        label="Storage write bandwidth (GB/s)",
        show_value=True,
    )
    act2_step_time_slider = mo.ui.slider(
        start=1, stop=60, value=10, step=1,
        label="Training step time (seconds)",
        show_value=True,
    )
    act2_async_toggle = mo.ui.radio(
        options={"Synchronous (stall training during write)": "sync",
                 "Asynchronous (overlap write with training)": "async"},
        value="Synchronous (stall training during write)",
        label="Checkpoint write strategy:",
    )

    mo.vstack([
        mo.hstack([act2_n_gpus_slider, act2_gpu_mtbf_slider], justify="start", gap=4),
        mo.hstack([act2_model_tb_slider, act2_storage_bw_slider], justify="start", gap=4),
        mo.hstack([act2_step_time_slider, act2_async_toggle], justify="start", gap=4),
    ])
    return (
        act2_n_gpus_slider,
        act2_gpu_mtbf_slider,
        act2_model_tb_slider,
        act2_storage_bw_slider,
        act2_step_time_slider,
        act2_async_toggle,
    )


@app.cell(hide_code=True)
def _(
    COLORS,
    H100_MTBF_HOURS,
    LUSTRE_WRITE_GBS,
    act2_async_toggle,
    act2_gpu_mtbf_slider,
    act2_model_tb_slider,
    act2_n_gpus_slider,
    act2_step_time_slider,
    act2_storage_bw_slider,
    apply_plotly_theme,
    go,
    math,
    mo,
    np,
):
    # ── Pull slider values ─────────────────────────────────────────────────
    _N          = act2_n_gpus_slider.value           # number of GPUs
    _gpu_mtbf   = act2_gpu_mtbf_slider.value         # hours per GPU
    _model_tb   = act2_model_tb_slider.value         # checkpoint size, TB
    _stor_bw    = act2_storage_bw_slider.value       # GB/s
    _step_s     = act2_step_time_slider.value        # seconds per step
    _write_mode = act2_async_toggle.value

    # ── Cluster failure rate ───────────────────────────────────────────────
    # λ_cluster = N × λ_per_gpu  [source: @sec-fault-tolerance-failure-modes]
    _lambda_per_gpu  = 1.0 / _gpu_mtbf              # failures/hour per GPU
    _lambda_cluster  = _N * _lambda_per_gpu          # failures/hour for cluster
    _mtbf_cluster_hr = 1.0 / _lambda_cluster         # hours
    _mtbf_cluster_min= _mtbf_cluster_hr * 60.0       # minutes

    # ── Checkpoint write time ─────────────────────────────────────────────
    # C_write = model_TB × 1024 GB/TB / storage_bw_GBs  → seconds
    _model_gb    = _model_tb * 1024.0                # GB
    _C_write_s   = _model_gb / _stor_bw              # seconds
    _C_write_min = _C_write_s / 60.0                 # minutes
    _C_write_hr  = _C_write_s / 3600.0               # hours

    # ── Effective checkpoint stall depends on write strategy ──────────────
    # Synchronous: C_effective = C_write (full stall)
    # Asynchronous: C_effective ≈ 0 (overlap with training; stall only to
    #   initiate the write and to synchronize at the next checkpoint boundary)
    # We model async effective stall as 2% of C_write (snapshot + barrier cost)
    # [source: @sec-fault-tolerance-young-daly async analysis]
    if _write_mode == "sync":
        _C_eff_hr  = _C_write_hr      # full stall
        _C_eff_min = _C_write_min
        _C_eff_s   = _C_write_s
        _async_benefit_pct = 0.0
    else:
        _C_eff_hr  = _C_write_hr * 0.02   # 2% residual stall for async
        _C_eff_min = _C_write_min * 0.02
        _C_eff_s   = _C_write_s * 0.02
        _async_benefit_pct = (1.0 - 0.02) * 100.0

    # ── Young-Daly optimal interval using effective C ─────────────────────
    _T_opt_hr  = math.sqrt(2.0 * _C_eff_hr / _lambda_cluster)
    _T_opt_min = _T_opt_hr * 60.0
    _T_opt_s   = _T_opt_hr * 3600.0

    # ── Overhead at optimal interval ───────────────────────────────────────
    _overhead_opt_pct = (_C_eff_hr / _T_opt_hr) * 100.0

    # ── If we snap T to nearest step boundary ─────────────────────────────
    # steps per optimal interval
    _steps_per_opt = max(1, round(_T_opt_s / _step_s))
    _T_snapped_s   = _steps_per_opt * _step_s
    _T_snapped_min = _T_snapped_s / 60.0
    _T_snapped_hr  = _T_snapped_s / 3600.0

    _overhead_snapped_pct = (_C_eff_hr / _T_snapped_hr) * 100.0
    _waste_snapped_min    = _T_snapped_min / 2.0 + _C_eff_min

    # ── Net training efficiency ───────────────────────────────────────────
    # Efficiency = 1 - overhead - expected_rework_rate
    _rework_rate = _lambda_cluster * (_T_snapped_hr / 2.0 + _C_eff_hr)
    _efficiency  = max(0.0, (1.0 - _overhead_snapped_pct / 100.0 - _rework_rate)) * 100.0

    # ── Failure state: overhead > 20% ────────────────────────────────────
    _oom_flag = _overhead_snapped_pct > 20.0

    # ── Color coding ──────────────────────────────────────────────────────
    _eff_color = COLORS["GreenLine"] if _efficiency > 80 else (
        COLORS["OrangeLine"] if _efficiency > 60 else COLORS["RedLine"]
    )
    _ovh_color2 = COLORS["GreenLine"] if _overhead_snapped_pct < 10 else (
        COLORS["OrangeLine"] if _overhead_snapped_pct < 20 else COLORS["RedLine"]
    )
    _waste_color = COLORS["GreenLine"] if _waste_snapped_min < 30 else (
        COLORS["OrangeLine"] if _waste_snapped_min < 60 else COLORS["RedLine"]
    )

    # ── Overhead vs interval curve ─────────────────────────────────────────
    _T_range_min2 = np.linspace(0.5, min(120.0, _mtbf_cluster_min * 3), 400)
    _T_range_hr2  = _T_range_min2 / 60.0
    _total_curve2 = (_C_eff_hr / _T_range_hr2) + _lambda_cluster * (_T_range_hr2 / 2.0 + _C_eff_hr)

    _fig2 = go.Figure()

    _fig2.add_trace(go.Scatter(
        x=_T_range_min2,
        y=_total_curve2 * 100.0,
        mode="lines",
        name="Total wasted fraction",
        line=dict(color=COLORS["BlueLine"], width=2.5),
    ))

    # 20% overhead budget line
    _fig2.add_hline(
        y=20.0,
        line_dash="dot",
        line_color=COLORS["RedLine"],
        annotation_text="20% overhead budget",
        annotation_position="top right",
    )

    # Cluster MTBF line
    _fig2.add_vline(
        x=_mtbf_cluster_min,
        line_dash="dash",
        line_color=COLORS["OrangeLine"],
        opacity=0.6,
        annotation_text=f"Cluster MTBF = {_mtbf_cluster_min:.0f} min",
        annotation_position="top left",
    )

    # Optimal marker
    _fig2.add_trace(go.Scatter(
        x=[_T_opt_min],
        y=[_overhead_opt_pct + _lambda_cluster * (_T_opt_hr / 2.0 + _C_eff_hr) * 100.0],
        mode="markers",
        name=f"Young-Daly T* = {_T_opt_min:.1f} min",
        marker=dict(color=COLORS["GreenLine"], size=14, symbol="diamond",
                    line=dict(color="white", width=2)),
    ))

    # Snapped interval marker
    _snapped_total = (_C_eff_hr / _T_snapped_hr + _lambda_cluster * (_T_snapped_hr / 2.0 + _C_eff_hr)) * 100.0
    _fig2.add_trace(go.Scatter(
        x=[_T_snapped_min],
        y=[_snapped_total],
        mode="markers",
        name=f"Step-snapped T = {_T_snapped_min:.0f} min",
        marker=dict(color=COLORS["Cloud"], size=12, symbol="circle",
                    line=dict(color="white", width=2)),
    ))

    _fig2.update_layout(
        height=320,
        xaxis=dict(title="Checkpoint interval T (minutes)"),
        yaxis=dict(title="Wasted fraction of training time (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=50, l=50, r=20),
    )
    apply_plotly_theme(_fig2)

    # ── Metric cards ──────────────────────────────────────────────────────
    _metric_html = mo.Html(f"""
    <div class="lab-card" style="margin: 8px 0;">
        <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
            Fleet Checkpoint Analysis
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
            <div style="text-align: center; padding: 16px; border: 1px solid {COLORS['Border']};
                        border-radius: 8px;">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.78rem; margin-bottom: 6px;">
                    Cluster MTBF
                </div>
                <div style="font-size: 1.6rem; font-weight: 800;
                            color: {COLORS['OrangeLine']}; font-family: monospace;">
                    {_mtbf_cluster_min:.0f} min
                </div>
                <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem;">
                    = {_gpu_mtbf} hr / {_N:,}
                </div>
            </div>
            <div style="text-align: center; padding: 16px; border: 1px solid {COLORS['Border']};
                        border-radius: 8px;">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.78rem; margin-bottom: 6px;">
                    Write time C
                </div>
                <div style="font-size: 1.6rem; font-weight: 800;
                            color: {COLORS['BlueLine']}; font-family: monospace;">
                    {_C_write_s:.1f} s
                </div>
                <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem;">
                    {'sync stall' if _write_mode == 'sync' else f'async: {_C_eff_s:.2f}s stall'}
                </div>
            </div>
            <div style="text-align: center; padding: 16px; border: 1px solid {COLORS['Border']};
                        border-radius: 8px;">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.78rem; margin-bottom: 6px;">
                    Optimal T*
                </div>
                <div style="font-size: 1.6rem; font-weight: 800;
                            color: {COLORS['GreenLine']}; font-family: monospace;">
                    {_T_opt_min:.1f} min
                </div>
                <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem;">
                    Young-Daly
                </div>
            </div>
            <div style="text-align: center; padding: 16px; border: 1px solid {COLORS['Border']};
                        border-radius: 8px;">
                <div style="color: {COLORS['TextMuted']}; font-size: 0.78rem; margin-bottom: 6px;">
                    Net Efficiency
                </div>
                <div style="font-size: 1.6rem; font-weight: 800;
                            color: {_eff_color}; font-family: monospace;">
                    {_efficiency:.1f}%
                </div>
                <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem;">
                    GPU-hours on training
                </div>
            </div>
        </div>
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid {COLORS['Border']};
                    font-family: monospace; font-size: 0.85rem; line-height: 2.0; color: {COLORS['Text']};">
            <div>
                &lambda;_cluster = N &times; &lambda;_gpu = {_N:,} &times; {_lambda_per_gpu:.6f}
                = <strong>{_lambda_cluster:.4f}</strong> failures/hr
                &nbsp;&nbsp;|&nbsp;&nbsp; MTBF_cluster = <strong>{_mtbf_cluster_min:.1f} min</strong>
            </div>
            <div>
                C_write = {_model_tb:.1f} TB &times; 1024 / {_stor_bw} GB/s
                = <strong>{_C_write_s:.1f} s</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp; C_eff ({'sync' if _write_mode == 'sync' else 'async'})
                = <strong>{_C_eff_s:.2f} s</strong>
            </div>
            <div>
                T* = sqrt(2 &times; {_C_eff_hr:.6f} / {_lambda_cluster:.4f})
                = sqrt({2*_C_eff_hr/_lambda_cluster:.6f})
                = <strong style="color:{COLORS['GreenLine']};">{_T_opt_min:.1f} min</strong>
            </div>
            <div>
                Checkpoint overhead at T* = C_eff / T*
                = {_C_eff_s:.2f}s / {_T_opt_s:.0f}s
                = <strong style="color:{_ovh_color2};">{_overhead_opt_pct:.1f}%</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp; Expected rework per failure:
                <strong style="color:{_waste_color};">{_waste_snapped_min:.1f} min</strong>
            </div>
        </div>
    </div>
    """)

    # ── Failure state ─────────────────────────────────────────────────────
    if _oom_flag:
        _failure = mo.callout(mo.md(
            f"**Checkpoint overhead at {_overhead_snapped_pct:.1f}% exceeds the 20% budget.** "
            f"Checkpointing every {_T_snapped_min:.0f} min costs {_C_eff_min:.2f} min per write "
            f"— that is {_overhead_snapped_pct:.1f}% of training time spent on I/O. "
            f"Young-Daly optimal is T\\* = {_T_opt_min:.1f} min with {_overhead_opt_pct:.1f}% overhead. "
            f"**Options:** "
            f"(1) Switch to async checkpointing to reduce effective C to ~0 "
            f"(overhead drops to ~{((_C_write_hr * 0.02) / _T_opt_hr) * 100:.1f}%); "
            f"(2) Increase storage bandwidth (current: {_stor_bw} GB/s — need "
            f"{_stor_bw * (_overhead_snapped_pct / 20.0):.0f} GB/s for sync writes at budget); "
            f"(3) Reduce checkpoint size via model sharding across storage nodes."
        ), kind="danger")
    else:
        _failure = mo.callout(mo.md(
            f"**Checkpoint overhead is {_overhead_snapped_pct:.1f}% — within the 20% budget.** "
            f"T\\* = {_T_opt_min:.1f} min "
            f"({'async: effective stall = {:.2f}s'.format(_C_eff_s) if _write_mode == 'async' else 'sync: full stall = {:.1f}s'.format(_C_eff_s)}). "
            f"Net training efficiency: {_efficiency:.1f}%."
        ), kind="success" if _efficiency > 80 else "warn")

    mo.vstack([_metric_html, mo.ui.plotly(_fig2), _failure])
    return (
        _T_opt_min, _T_opt_s, _T_snapped_min,
        _overhead_snapped_pct, _efficiency,
        _C_write_s, _C_eff_s, _C_eff_min,
        _mtbf_cluster_min, _lambda_cluster,
        _oom_flag, _write_mode,
        _N, _gpu_mtbf, _stor_bw, _model_tb, _step_s,
    )


# ─── ACT II: PREDICTION REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS,
    _C_eff_s,
    _C_write_s,
    _T_opt_min,
    _efficiency,
    _mtbf_cluster_min,
    _overhead_snapped_pct,
    act2_pred,
    mo,
):
    _correct2 = act2_pred.value == "C"

    if _correct2:
        _reveal2 = mo.callout(mo.md(
            f"**Correct.** "
            f"With N = 16,384, MTBF_per_gpu = 200 hr, the cluster MTBF = "
            f"{_mtbf_cluster_min:.0f} minutes. "
            f"C_sync = {_C_write_s:.0f}s = {_C_write_s/60:.3f} hr. "
            f"T\\* = sqrt(2 × {_C_write_s/3600:.5f} / {1/(200/16384):.5f}) "
            f"= **{_T_opt_min:.1f} minutes**. "
            f"With synchronous writes the overhead is "
            f"{_overhead_snapped_pct:.1f}%. "
            f"Async checkpointing reduces the effective stall to {_C_eff_s:.2f}s, "
            f"driving overhead near zero and net training efficiency to "
            f"{_efficiency:.1f}%. "
            f"The key insight: async writes decouple C_write from C_stall — "
            f"the Young-Daly formula uses the *stall* time, not the write time."
        ), kind="success")
    elif act2_pred.value == "A":
        _reveal2 = mo.callout(mo.md(
            f"**Not quite.** "
            f"A 5-minute interval with C = {_C_write_s:.0f}s = {_C_write_s/60:.1f} min "
            f"means overhead = {_C_write_s/60 / 5.0 * 100:.0f}% — "
            f"writing a checkpoint every 5 minutes when each write takes "
            f"{_C_write_s:.0f}s is unsustainable. "
            f"Young-Daly T\\* = {_T_opt_min:.1f} min. "
            f"The 44-minute cluster MTBF does not mean 'checkpoint every 5 minutes' — "
            f"it means 'choose T optimally given the failure rate, not emotionally.'"
        ), kind="warn")
    elif act2_pred.value == "B":
        _reveal2 = mo.callout(mo.md(
            f"**Not quite.** "
            f"A 10-minute interval gives overhead = {_C_write_s/60 / 10.0 * 100:.0f}% "
            f"with synchronous writes (C = {_C_write_s:.0f}s). "
            f"Young-Daly T\\* = {_T_opt_min:.1f} min with {_overhead_snapped_pct:.1f}% overhead. "
            f"At 10 min, you are spending {_C_write_s/60 / 10.0 * 100:.0f}% of training "
            f"on I/O — nearly double the budget threshold. "
            f"The formula gives a precise answer; 'balance' intuition is not sufficient."
        ), kind="warn")
    else:  # D
        _reveal2 = mo.callout(mo.md(
            f"**Not quite.** "
            f"A 60-minute interval with cluster MTBF = {_mtbf_cluster_min:.0f} min "
            f"means failures almost always occur within the interval. "
            f"Expected rework per failure = T/2 + C = 30 + {_C_write_s/60:.1f} = "
            f"{30 + _C_write_s/60:.1f} min. "
            f"Young-Daly T\\* = {_T_opt_min:.1f} min with {_overhead_snapped_pct:.1f}% overhead — "
            f"a much smaller rework exposure. "
            f"When MTBF < T, every training run ends in a failure before the next checkpoint."
        ), kind="warn")

    _reveal2
    return


# ─── ACT II: REFLECTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Reflection — Async Checkpointing")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) Async writes compress the checkpoint data during the write": "A",
            "B) The checkpoint cost C in Young-Daly is the training stall time — async writes reduce stall to near-zero": "B",
            "C) Async writes automatically deduplicate checkpoint tensors": "C",
            "D) Async checkpointing doesn't affect C — it only changes the timing of the write": "D",
        },
        label="Why does asynchronous checkpointing reduce the effective C in the Young-Daly formula?",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(act2_reflect, mo):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(
            mo.md("Select an answer to see the explanation."),
            kind="warn",
        ),
    )
    mo.md("")
    return


@app.cell(hide_code=True)
def _(act2_reflect, mo):
    if act2_reflect.value == "B":
        _r2 = mo.callout(mo.md(
            "**Correct.** "
            "The Young-Daly formula minimizes *training time lost*. "
            "The C term represents how long training is stalled — not how long "
            "the storage write takes. "
            "With synchronous checkpointing, C_stall = C_write (training halts "
            "until the write completes). "
            "With asynchronous checkpointing, training resumes immediately after "
            "snapshotting memory; the write proceeds in parallel. "
            "C_stall ≈ snapshot overhead (nanoseconds to microseconds) rather than "
            "the full write time. "
            "For a 2 TB checkpoint at 400 GB/s, this reduces C from 5 seconds to "
            "essentially zero — changing T\\* by sqrt(C_sync / C_async) ≈ "
            "sqrt(5s / 0.1s) = 7× and reducing overhead from ~40% to <1%."
        ), kind="success")
    elif act2_reflect.value == "A":
        _r2 = mo.callout(mo.md(
            "**Not quite.** "
            "Async checkpointing is about *when* the write happens relative to training, "
            "not about compression. The write size is identical. "
            "The reduction in effective C comes from decoupling the write from the "
            "training compute path — training continues while storage writes proceed "
            "on a separate I/O thread or DMA engine."
        ), kind="warn")
    elif act2_reflect.value == "C":
        _r2 = mo.callout(mo.md(
            "**Not quite.** "
            "Deduplication is a separate optimization (delta checkpointing) that "
            "reduces checkpoint *size*, which reduces C_write. "
            "Async checkpointing addresses a different problem: it reduces "
            "C_stall to near-zero by overlapping the write with training, "
            "even when C_write remains large."
        ), kind="warn")
    else:
        _r2 = mo.callout(mo.md(
            "**Not quite.** "
            "Async checkpointing does change C — specifically, the *effective stall* C. "
            "Young-Daly measures the time training is blocked, not the wall-clock write time. "
            "With synchronous checkpointing, training blocks for C_write seconds. "
            "With asynchronous checkpointing, training blocks only for the snapshot "
            "initiation (near-zero). The formula uses C_stall, so async checkpointing "
            "fundamentally changes the T\\* calculation."
        ), kind="warn")
    _r2
    return


# ─── ACT II: MATHPEEK ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Async Checkpoint Overlap Analysis": mo.md("""
        **Synchronous checkpointing (baseline):**
        ```
        Timeline: [train T] [stall C_write] [train T] [stall C_write] ...
        Overhead fraction = C_write / T
        ```

        **Asynchronous checkpointing (snapshot + background write):**
        ```
        Timeline: [train T][snapshot ε] [train T][snapshot ε] ...
                                ↕ background write (C_write overlaps training)
        Effective stall = ε << C_write  (typically 10-100ms for memory snapshot)
        Overhead fraction = ε / T ≈ 0
        ```

        **Young-Daly with async C:**
        ```
        C_eff = ε ≈ 0  →  T* = sqrt(2ε/λ) → very short intervals become optimal
        ```
        In practice, async checkpoints use double-buffering: while one checkpoint
        is written to storage, training continues accumulating the next snapshot.
        The critical constraint is memory: you need ~2× model memory to hold both
        the live state and the snapshot buffer simultaneously.

        **Expected total training time with failures:**
        ```
        E[T_total] = T_ideal / (1 - overhead - λ·E[wasted_hr])
        ```
        where T_ideal is the GPU-hours of actual training work required.

        **Multi-level checkpointing strategy:**
        - Level 0: In-memory (host RAM) every N steps — near-zero cost, volatile
        - Level 1: NVMe/local SSD every K·N steps — fast, survives GPU failure
        - Level 2: Distributed storage (Lustre/GCS) every M·K·N steps — durable

        Each level uses Young-Daly independently with its own C and λ. The expected
        wasted time reduces dramatically versus a single-level strategy because
        in-memory checkpoints can recover from single-GPU failures in seconds.
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN LEDGER SAVE + HUD FOOTER
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _(
    COLORS,
    _N,
    _T_opt_min,
    _T_snapped_min,
    _efficiency,
    _gpu_mtbf,
    _mtbf_cluster_min,
    _oom_flag,
    _overhead_snapped_pct,
    _write_mode,
    act1_pred,
    act2_pred,
    context_toggle,
    ledger,
    mo,
):
    # ── Determine correctness ─────────────────────────────────────────────
    _act1_correct = act1_pred.value == "B"
    _act2_decision = f"T={_T_snapped_min:.0f}min_{_write_mode}"

    # ── Save to Design Ledger ─────────────────────────────────────────────
    ledger.save(
        chapter="v2_07",
        design={
            "context":                context_toggle.value,
            "cluster_gpus":           _N,
            "cluster_mtbf_hours":     _mtbf_cluster_min / 60.0,
            "checkpoint_interval_min": _T_snapped_min,
            "optimal_interval_min":   _T_opt_min,
            "checkpoint_overhead_pct": _overhead_snapped_pct,
            "act1_prediction":        act1_pred.value,
            "act1_correct":           _act1_correct,
            "act2_result":            _efficiency,
            "act2_decision":          _act2_decision,
            "constraint_hit":         _oom_flag,
        },
    )

    # ── HUD footer ────────────────────────────────────────────────────────
    _act1_status   = "correct" if _act1_correct else "incorrect"
    _act1_color    = COLORS["GreenLine"] if _act1_correct else COLORS["RedLine"]
    _constr_color  = COLORS["RedLine"] if _oom_flag else COLORS["GreenLine"]
    _constr_label  = "HIT" if _oom_flag else "CLEAR"
    _eff_hud_color = COLORS["GreenLine"] if _efficiency > 80 else (
        COLORS["OrangeLine"] if _efficiency > 60 else COLORS["RedLine"]
    )

    mo.Html(f"""
    <div class="lab-hud">
        <div>
            <span class="hud-label">CHAPTER&nbsp;</span>
            <span class="hud-value">Vol 2 · Lab 07 · Fault Tolerance</span>
        </div>
        <div>
            <span class="hud-label">CONTEXT&nbsp;</span>
            <span class="hud-value">{context_toggle.value.upper()}</span>
        </div>
        <div>
            <span class="hud-label">CLUSTER&nbsp;</span>
            <span class="hud-value">{_N:,} GPUs · MTBF {_mtbf_cluster_min:.0f} min</span>
        </div>
        <div>
            <span class="hud-label">ACT I&nbsp;</span>
            <span style="color:{_act1_color}; font-weight:700;">{_act1_status.upper()}</span>
        </div>
        <div>
            <span class="hud-label">T*&nbsp;</span>
            <span class="hud-value">{_T_opt_min:.1f} min ({_write_mode})</span>
        </div>
        <div>
            <span class="hud-label">OVERHEAD&nbsp;</span>
            <span style="color:{_eff_hud_color}; font-weight:700;">{_overhead_snapped_pct:.1f}%</span>
        </div>
        <div>
            <span class="hud-label">EFFICIENCY&nbsp;</span>
            <span style="color:{_eff_hud_color}; font-weight:700;">{_efficiency:.1f}%</span>
        </div>
        <div>
            <span class="hud-label">CONSTRAINT&nbsp;</span>
            <span style="color:{_constr_color}; font-weight:700;">{_constr_label}</span>
        </div>
    </div>
    """)
    return


# ─── KEY TAKEAWAYS ─────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Key Takeaways

    1. **Young-Daly is a physics law, not a guideline:** The optimal checkpoint interval
       T\\* = sqrt(2C/λ) minimizes the provably expected total wasted time. Deviating from
       T\\* in either direction increases costs — more frequent checkpoints burn I/O bandwidth,
       less frequent checkpoints increase expected rework. The formula makes the right
       answer exact, not approximate.

    2. **Cluster failure rate scales linearly with GPU count:**
       λ_cluster = N × λ_per_gpu. At 16,384 GPUs with per-GPU MTBF of 200 hours,
       the cluster fails every 44 minutes. Async checkpointing — where the training
       stall C_stall is decoupled from the write time C_write — is not an optimization
       at this scale; it is a prerequisite. Without it, checkpoint overhead alone
       consumes 30-40% of GPU-hours.
    """)
    return


if __name__ == "__main__":
    app.run()
