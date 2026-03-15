import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-06: WHEN FAILURE IS ROUTINE
#
# Chapter: Fault Tolerance (@sec-fault-tolerance)
# Core Invariant: At fleet scale, failure is statistically certain. The
#                 Young-Daly optimal checkpoint interval (sqrt(2 * T_write * MTBF))
#                 minimizes total wasted work. Checkpoint storms create a
#                 pathological state where checkpointing takes longer than the
#                 optimal interval.
#
# 2-Act Structure (35-40 minutes):
#   Act I  — Reliability Recall + Young-Daly Sweet Spot (12-15 min)
#             Quick recall: 10,000-GPU cluster fails every ~5 hours.
#             Then the Young-Daly U-curve reveals the checkpoint interval.
#
#   Act II — The Checkpoint Storm + Serving Fault Tolerance (20-25 min)
#             175B checkpoint on NFS takes 41 minutes -- longer than the
#             optimal interval. Serving fault tolerance requires stateful
#             KV cache recovery. Reliability budget design challenge.
#
# Hardware Constants:
#   GPU_MTTF_HOURS    = 50,000   (from mlsysim defaults)
#   H100_RAM_GB       = 80       (NVIDIA H100 SXM5 spec)
#   H100_COST_HR      = 3.0      ($3/GPU-hour cloud pricing)
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP ─────────────────────────────────────────────────────────────
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
        await micropip.install("https://mlsysbook.ai/labs/wheels/mlsysim-0.1.0-py3-none-any.whl")
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog
    from mlsysim.core.defaults import GPU_MTTF_HOURS

    GPU_COST_HR = 3.0    # $/GPU-hour cloud pricing
    ledger = DesignLedger()
    return COLORS, LAB_CSS, apply_plotly_theme, go, ledger, math, mo, np, GPU_MTTF_HOURS, GPU_COST_HR, DecisionLog


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, LAB_CSS, mo):
    mo.Html(f"""
    {LAB_CSS}
    <div style="background: linear-gradient(135deg, {COLORS['Surface0']} 0%, {COLORS['Surface1']} 100%);
                border-radius: 16px; padding: 32px 40px; margin-bottom: 8px;
                border: 1px solid #2d3748;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                            text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;">
                    Vol 2 &middot; Lab 06 &middot; Fault Tolerance
                </div>
                <div style="font-size: 2.0rem; font-weight: 800; color: #f1f5f9; line-height: 1.15; margin-bottom: 10px;">
                    When Failure is Routine
                </div>
                <div style="font-size: 0.95rem; color: #94a3b8; max-width: 600px; line-height: 1.6;">
                    A 10,000-GPU cluster fails every 5 hours. Checkpoint too often and you
                    waste compute writing. Checkpoint too rarely and you lose days of work.
                    The Young-Daly formula finds the sweet spot -- but at frontier scale,
                    even checkpointing can become the bottleneck.
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;">
                <span class="badge badge-info">MTBF Scaling</span>
                <span class="badge badge-info">Young-Daly U-Curve</span>
                <span class="badge badge-info">Checkpoint Storm</span>
                <span class="badge badge-warn">35&ndash;40 minutes &middot; 2 Acts</span>
            </div>
        </div>
    </div>
    """)
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
                <div style="margin-bottom: 3px;">1. <strong>Calculate cluster MTBF:</strong> apply MTBF_system = MTBF_component / N to determine failure frequency at scale.</div>
                <div style="margin-bottom: 3px;">2. <strong>Derive the Young-Daly optimal checkpoint interval:</strong> understand why tau_opt = sqrt(2 * T_write * MTBF) minimizes total waste on the U-shaped cost curve.</div>
                <div style="margin-bottom: 3px;">3. <strong>Identify the checkpoint storm pathology:</strong> recognize when checkpoint write time exceeds the optimal interval, making the system spend more time checkpointing than computing.</div>
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
                    Reliability collapse from @sec-scale-illusion &middot;
                    MTBF_system = MTBF_component / N from @sec-fault-tolerance
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35-40 min</strong><br/>
                    Act I: ~15 min &middot; Act II: ~25 min
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
                "If a 10,000-GPU cluster fails every 5 hours, how do you checkpoint often enough
                to avoid losing work but not so often that checkpointing itself becomes the
                bottleneck &mdash; and what happens when even optimal checkpointing cannot keep up?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete before this lab:

    - **@sec-fault-tolerance** -- MTBF scaling, Young-Daly checkpoint model
    - **@sec-scale-illusion** -- Fleet reliability collapse (V2-01 recap)
    - The Checkpoint Storm section -- Storage bandwidth saturation at scale
    - The Serving Fault Tolerance section -- KV cache state and replica failover
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I -- RELIABILITY + YOUNG-DALY SWEET SPOT
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['BlueLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">I</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act I &middot; 12&ndash;15 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            The Young-Daly Sweet Spot
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            You think checkpointing every 10 minutes is safe. The Young-Daly formula will
            reveal that the optimal interval for a 16,000-GPU cluster is ~27 minutes &mdash;
            longer than your instinct suggests &mdash; because the square root law balances
            checkpoint overhead against expected rework.
        </div>
    </div>
    """)
    return


# ─── ACT1: STAKEHOLDER MESSAGE ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']}; background: {COLORS['BlueLL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['BlueLine']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Training Reliability Engineer
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "Our 16,000-GPU cluster has a system MTBF of about 3 hours. Each checkpoint
            write takes 2 minutes. I want to checkpoint every 10 minutes to be safe.
            My manager says that is too aggressive. Who is right?"
        </div>
    </div>
    """)
    return


# ─── ACT1: CONCEPT FRAMING ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Checkpointing has two costs that pull in opposite directions:

    1. **Checkpoint overhead**: Writing checkpoints consumes time that could be used for
       training. Checkpointing every tau seconds wastes T_write/tau fraction of wall time.

    2. **Expected rework**: When a failure occurs, you lose all progress since the last
       checkpoint. On average, you lose tau/2 seconds of compute. The expected rework
       per unit time is tau / (2 * MTBF).

    Total waste = T_write/tau + tau/(2*MTBF). This is a **U-shaped curve** with a unique
    minimum at:

    **tau_opt = sqrt(2 * T_write * MTBF)** (the Young-Daly formula)

    For MTBF = 3 hours (10,800s) and T_write = 2 minutes (120s):
    tau_opt = sqrt(2 * 120 * 10,800) = sqrt(2,592,000) = **~1,610 seconds = ~27 minutes**

    The engineer's 10-minute interval wastes 2/10 = 20% of time on checkpoint overhead alone.
    The optimal 27-minute interval wastes only 2/27 = 7.4% on overhead.
    """)
    return


# ─── ACT1: PREDICTION LOCK ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Your Prediction")
    return


@app.cell(hide_code=True)
def _(mo):
    partA_prediction = mo.ui.radio(
        options={
            "A) Every 2 minutes -- match the write time": "A",
            "B) Every 10 minutes -- frequent saves": "B",
            "C) Every ~27 minutes -- the square-root law": "C",
            "D) Every 90 minutes -- halfway to MTBF": "D",
        },
        label="A 16,000-GPU cluster (MTBF ~3 hours). Checkpoint writes take 2 minutes. What is the optimal checkpoint interval?",
    )
    partA_prediction
    return (partA_prediction,)


@app.cell(hide_code=True)
def _(partA_prediction, mo):
    mo.stop(
        partA_prediction.value is None,
        mo.callout(mo.md("Select your prediction above to unlock the Act I instruments."), kind="warn"),
    )
    mo.md("")
    return


# ─── ACT1: INSTRUMENTS ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Young-Daly Checkpoint Optimizer")
    return


@app.cell(hide_code=True)
def _(mo):
    a1_cluster_gpus = mo.ui.slider(start=1000, stop=25000, value=16000, step=1000, label="Cluster GPUs")
    a1_write_time_s = mo.ui.slider(start=10, stop=300, value=120, step=10, label="Checkpoint write time (seconds)")
    a1_interval_s = mo.ui.slider(start=60, stop=10800, value=600, step=60, label="Your checkpoint interval (seconds)")
    mo.hstack([a1_cluster_gpus, a1_write_time_s, a1_interval_s], justify="center", gap=2)
    return (a1_cluster_gpus, a1_write_time_s, a1_interval_s)


@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, a1_cluster_gpus, a1_write_time_s, a1_interval_s, go, math, mo, np, GPU_MTTF_HOURS, GPU_COST_HR):
    _n_gpus = a1_cluster_gpus.value
    _t_write = a1_write_time_s.value
    _tau = a1_interval_s.value

    # MTBF = component MTTF / N
    _mtbf_s = GPU_MTTF_HOURS * 3600 / _n_gpus
    _mtbf_h = _mtbf_s / 3600

    # Young-Daly optimal
    _tau_opt = math.sqrt(2 * _t_write * _mtbf_s)
    _tau_opt_min = _tau_opt / 60

    # Waste components at current interval
    _ckpt_overhead = _t_write / _tau if _tau > 0 else 1.0
    _rework = _tau / (2 * _mtbf_s) if _mtbf_s > 0 else 1.0
    _total_waste = _ckpt_overhead + _rework
    _total_waste_pct = min(_total_waste * 100, 100)

    # Waste at optimal
    _waste_opt = _t_write / _tau_opt + _tau_opt / (2 * _mtbf_s)
    _waste_opt_pct = _waste_opt * 100

    # Dollar cost of waste per day
    _compute_cost_day = _n_gpus * GPU_COST_HR * 24
    _waste_cost_day = _compute_cost_day * _total_waste
    _waste_cost_opt_day = _compute_cost_day * _waste_opt

    # ── U-curve chart ─────────────────────────────────────────────────────
    _tau_range = np.linspace(60, min(_mtbf_s, 10800), 200)
    _overhead_curve = [_t_write / t * 100 for t in _tau_range]
    _rework_curve = [t / (2 * _mtbf_s) * 100 for t in _tau_range]
    _total_curve = [(_t_write / t + t / (2 * _mtbf_s)) * 100 for t in _tau_range]

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_tau_range / 60, y=_overhead_curve, mode="lines",
                               name="Checkpoint overhead", line=dict(color=COLORS["BlueLine"], width=2)))
    _fig.add_trace(go.Scatter(x=_tau_range / 60, y=_rework_curve, mode="lines",
                               name="Expected rework", line=dict(color=COLORS["RedLine"], width=2)))
    _fig.add_trace(go.Scatter(x=_tau_range / 60, y=_total_curve, mode="lines",
                               name="Total waste", line=dict(color=COLORS["Text"], width=3)))
    # Mark optimal
    _fig.add_trace(go.Scatter(x=[_tau_opt_min], y=[_waste_opt_pct], mode="markers",
                               name="Young-Daly optimum", marker=dict(size=14, color=COLORS["GreenLine"], symbol="star")))
    # Mark current
    _fig.add_trace(go.Scatter(x=[_tau / 60], y=[_total_waste_pct], mode="markers",
                               name="Your interval", marker=dict(size=14, color=COLORS["OrangeLine"], symbol="diamond")))
    _fig.update_layout(
        height=340,
        xaxis=dict(title="Checkpoint Interval (minutes)"),
        yaxis=dict(title="Time Wasted (%)", range=[0, min(max(_total_curve) * 1.1, 100)]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=50, l=50, r=20),
    )
    apply_plotly_theme(_fig)

    # Colors
    _waste_color = COLORS["GreenLine"] if _total_waste_pct < 15 else (COLORS["OrangeLine"] if _total_waste_pct < 30 else COLORS["RedLine"])

    mo.vstack([
        mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Young-Daly Checkpoint Optimization
            </div>
            <div>Cluster: {_n_gpus:,} GPUs &times; MTTF {GPU_MTTF_HOURS:,} hrs = MTBF_system = <strong>{_mtbf_h:.1f} hours ({_mtbf_s:.0f}s)</strong></div>
            <div>tau_opt = sqrt(2 &times; {_t_write}s &times; {_mtbf_s:.0f}s) = <strong>{_tau_opt:.0f}s ({_tau_opt_min:.1f} min)</strong></div>
            <div>Your interval: {_tau}s ({_tau/60:.1f} min) &mdash; overhead: {_ckpt_overhead*100:.1f}% + rework: {_rework*100:.1f}% = <strong style="color:{_waste_color};">{_total_waste_pct:.1f}% waste</strong></div>
            <div>Optimal waste: <strong style="color:{COLORS['GreenLine']};">{_waste_opt_pct:.1f}%</strong> &mdash; savings vs your interval: <strong>${(_waste_cost_day - _waste_cost_opt_day):,.0f}/day</strong></div>
        </div>
        """),
        mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">System MTBF</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']}; font-family:monospace;">{_mtbf_h:.1f}h</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">{_n_gpus:,} GPUs</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Optimal Interval</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['GreenLine']}; font-family:monospace;">{_tau_opt_min:.0f}m</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">Young-Daly</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Total Waste</div>
                <div style="font-size:2rem; font-weight:800; color:{_waste_color}; font-family:monospace;">{_total_waste_pct:.1f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">your interval</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Waste Cost</div>
                <div style="font-size:2rem; font-weight:800; color:{_waste_color}; font-family:monospace;">${_waste_cost_day:,.0f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">per day</div>
            </div>
        </div>
        """),
        mo.ui.plotly(_fig),
    ])
    return


# ─── ACT1: PREDICTION REVEAL ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(partA_prediction, mo):
    if partA_prediction.value == "C":
        mo.callout(mo.md(
            "**Correct.** The Young-Daly formula gives tau_opt = sqrt(2 * 120s * 10,800s) = "
            "~1,610 seconds = ~27 minutes. This is the geometric mean between write time and "
            "MTBF, not the arithmetic mean. The square root law means the optimal interval "
            "is much closer to the write time than to the MTBF."
        ), kind="success")
    elif partA_prediction.value == "A":
        mo.callout(mo.md(
            "**Far too aggressive.** Checkpointing every 2 minutes with a 2-minute write time "
            "means 50% of wall time is spent checkpointing. Only 50% remains for training. "
            "The optimal interval balances overhead against rework, not minimizes rework alone."
        ), kind="warn")
    elif partA_prediction.value == "B":
        mo.callout(mo.md(
            "**Too aggressive.** At 10-minute intervals, checkpoint overhead is 2/10 = 20%, "
            "plus rework of 10/(2*180) = 2.8%, total = 22.8%. The optimal 27-minute interval "
            "achieves only 12.3% waste -- saving nearly 10 percentage points of compute time."
        ), kind="warn")
    elif partA_prediction.value == "D":
        mo.callout(mo.md(
            "**Too conservative.** At 90 minutes with MTBF of 3 hours, expected rework is "
            "90/(2*180) = 25%. Combined with 2/90 = 2.2% overhead, total waste is 27.2%. "
            "The optimal 27-minute interval achieves 12.3% -- half the waste."
        ), kind="warn")
    return


# ─── ACT1: MATHPEEK ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Governing equations -- Young-Daly checkpoint optimization": mo.md("""
        **System MTBF**

        ```
        MTBF_system = MTBF_component / N
        ```

        - N = number of GPUs in the cluster
        - MTBF_component = 50,000 hours (individual GPU MTTF)
        - At N = 16,000: MTBF = 50,000 / 16,000 = 3.125 hours

        **Young-Daly Optimal Interval**

        ```
        tau_opt = sqrt(2 * T_write * MTBF)
        ```

        - T_write = time to write one checkpoint
        - Total waste at optimum = 2 * sqrt(T_write / (2 * MTBF))
        - The square root law means the interval scales with the geometric
          mean of write time and MTBF, not the arithmetic mean

        **U-Shaped Waste Curve**

        ```
        W(tau) = T_write / tau  +  tau / (2 * MTBF)
                 [overhead]        [rework]
        ```

        - dW/d(tau) = 0 yields tau_opt = sqrt(2 * T_write * MTBF)
        """)
    })
    return


# ─── ACT1: REFLECTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    partA_reflection = mo.ui.radio(
        options={
            "A) Double the checkpoint frequency to protect against failures": "A",
            "B) Reduce checkpoint write time through faster storage or async checkpointing": "B",
            "C) Increase GPU MTTF by using higher-quality components": "C",
            "D) Accept the waste -- it is a fixed cost of distributed training": "D",
        },
        label="The Young-Daly formula shows waste is sqrt(T_write / MTBF). Which lever most effectively reduces waste?",
    )
    partA_reflection
    return (partA_reflection,)


@app.cell(hide_code=True)
def _(partA_reflection, mo):
    mo.stop(
        partA_reflection.value is None,
        mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"),
    )
    if partA_reflection.value == "B":
        mo.callout(mo.md(
            "**Correct.** The optimal waste is proportional to sqrt(T_write / MTBF). Reducing "
            "T_write by 4x (e.g., from NFS to parallel FS) reduces waste by 2x. Async "
            "checkpointing effectively makes T_write appear near-zero for the training loop. "
            "This is the most actionable lever because storage is within the team's control."
        ), kind="success")
    else:
        mo.callout(mo.md(
            "**Not the primary lever.** The Young-Daly waste formula is 2 * sqrt(T_write / (2 * MTBF)). "
            "Reducing T_write has the most direct impact because storage technology is within "
            "engineering control, unlike hardware reliability (MTBF)."
        ), kind="warn")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II -- THE CHECKPOINT STORM + RELIABILITY BUDGET
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="margin: 32px 0 12px 0; border-top: 2px solid {COLORS['Border']}; padding-top: 32px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['OrangeLine']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">II</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act II &middot; 20&ndash;25 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            The Checkpoint Storm
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            Act I found the optimal checkpoint interval. Now discover a pathological state:
            when storage bandwidth is too low, checkpoint write time exceeds the Young-Daly
            interval. The system spends more time checkpointing than computing &mdash; a
            checkpoint storm that no scheduling optimization can fix.
        </div>
    </div>
    """)
    return


# ─── ACT2: STAKEHOLDER MESSAGE ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['Cloud']}; background: {COLORS['BlueLL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['Cloud']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Storage Architect
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We have a 175B model training on 1,000 GPUs. Our storage is NFS with
            1 GB/s aggregate write bandwidth. How long does one checkpoint take?
            The training team says it is only a few minutes. I disagree."
        </div>
    </div>
    """)
    return


# ─── ACT2: CONCEPT + PREDICTION ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    A 175B model checkpoint includes weights + optimizer states + gradients:
    - 175B parameters x 14 bytes (FP16 weights + FP32 Adam m1 + m2 + master) = **2.45 TB** per checkpoint
    - At NFS 1 GB/s aggregate write: 2,450 seconds = **~41 minutes**

    With MTBF of ~5 hours (1,000 GPUs), the Young-Daly optimal interval is:
    tau_opt = sqrt(2 * 2450 * 18,000) = sqrt(88,200,000) = ~9,393s = ~2.6 hours

    But wait -- at 1 GB/s NFS, checkpoint write time (41 min) is already a significant
    fraction of the optimal interval (2.6 hours). With faster storage (100 GB/s NVMe RAID),
    the same checkpoint takes only 24.5 seconds.

    What happens at a larger cluster (10,000 GPUs) where MTBF drops to ~5 hours?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    partB_prediction = mo.ui.radio(
        options={
            "A) ~10 seconds -- fast with modern storage": "A",
            "B) ~2 minutes -- manageable": "B",
            "C) ~41 minutes -- longer than the Young-Daly optimal interval at this cluster size": "C",
            "D) ~5 minutes -- within budget": "D",
        },
        label="A 175B model checkpoints on a 1,000-GPU cluster with NFS storage (1 GB/s). How long does one checkpoint take?",
    )
    partB_prediction
    return (partB_prediction,)


@app.cell(hide_code=True)
def _(partB_prediction, mo):
    mo.stop(
        partB_prediction.value is None,
        mo.callout(mo.md("Select your prediction above to unlock the Act II instruments."), kind="warn"),
    )
    mo.md("")
    return


# ─── ACT2: INSTRUMENTS ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Checkpoint Storm Analyzer")
    return


@app.cell(hide_code=True)
def _(mo):
    a2_model_b = mo.ui.slider(start=1, stop=175, value=175, step=1, label="Model size (B params)")
    a2_cluster_gpus = mo.ui.slider(start=100, stop=25000, value=1000, step=100, label="Cluster GPUs")
    a2_storage = mo.ui.dropdown(
        options={"NFS (1 GB/s)": 1.0, "Parallel FS (10 GB/s)": 10.0, "NVMe RAID (100 GB/s)": 100.0},
        value="NFS (1 GB/s)",
        label="Storage type",
    )
    mo.hstack([a2_model_b, a2_cluster_gpus, a2_storage], justify="center", gap=2)
    return (a2_model_b, a2_cluster_gpus, a2_storage)


@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, a2_model_b, a2_cluster_gpus, a2_storage, go, math, mo, np, GPU_MTTF_HOURS, GPU_COST_HR):
    _params_b = a2_model_b.value
    _n_gpus = a2_cluster_gpus.value
    _storage_bw = a2_storage.value

    # Checkpoint size: 14 bytes per param (weights + optimizer)
    _ckpt_bytes = _params_b * 1e9 * 14
    _ckpt_tb = _ckpt_bytes / 1e12
    _ckpt_gb = _ckpt_bytes / 1e9

    # Write time
    _write_time_s = _ckpt_gb / _storage_bw
    _write_time_min = _write_time_s / 60

    # MTBF
    _mtbf_s = GPU_MTTF_HOURS * 3600 / _n_gpus
    _mtbf_h = _mtbf_s / 3600

    # Young-Daly optimal
    _tau_opt_s = math.sqrt(2 * _write_time_s * _mtbf_s)
    _tau_opt_min = _tau_opt_s / 60

    # Pathological: write time > optimal interval
    _pathological = _write_time_s > _tau_opt_s
    _ckpt_fraction = _write_time_s / _tau_opt_s if _tau_opt_s > 0 else 999

    # Dollar cost
    _compute_cost_day = _n_gpus * GPU_COST_HR * 24
    _ckpt_cost_per = _n_gpus * GPU_COST_HR * (_write_time_s / 3600)
    _daily_ckpts = 86400 / _tau_opt_s if _tau_opt_s > 0 else 0
    _daily_ckpt_cost = _ckpt_cost_per * _daily_ckpts

    # ── Storage comparison chart ──────────────────────────────────────────
    _storages = [("NFS 1 GB/s", 1.0), ("Parallel FS 10 GB/s", 10.0), ("NVMe RAID 100 GB/s", 100.0)]
    _write_times = [_ckpt_gb / bw for _, bw in _storages]
    _bar_colors = [COLORS["RedLine"] if wt > _tau_opt_s else COLORS["GreenLine"] for wt in _write_times]

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        x=[n for n, _ in _storages], y=[wt / 60 for wt in _write_times],
        marker_color=_bar_colors,
        text=[f"{wt/60:.1f} min" for wt in _write_times],
        textposition="auto",
        hovertemplate="%{x}<br>Write time: %{y:.1f} min<extra></extra>",
    ))
    _fig.add_hline(y=_tau_opt_min, line=dict(color=COLORS["OrangeLine"], width=2, dash="dash"),
                   annotation_text=f"Young-Daly optimal: {_tau_opt_min:.1f} min",
                   annotation_position="top right")
    _fig.update_layout(
        height=300,
        xaxis=dict(title="Storage System"),
        yaxis=dict(title="Checkpoint Write Time (minutes)"),
        margin=dict(t=30, b=50, l=50, r=20),
        showlegend=False,
    )
    apply_plotly_theme(_fig)

    # ── Failure banner ────────────────────────────────────────────────────
    _storm_banner = ""
    if _pathological:
        _storm_banner = f"""
        <div style="background:{COLORS['RedLL']}; border:2px solid {COLORS['RedLine']};
                    border-radius:10px; padding:14px 18px; margin:10px 0;">
            <div style="font-size:0.88rem; font-weight:800; color:{COLORS['RedLine']}; margin-bottom:4px;">
                CHECKPOINT STORM &mdash; System Pathological
            </div>
            <div style="font-size:0.85rem; color:#7f1d1d; line-height:1.6;">
                Checkpoint write time ({_write_time_min:.1f} min) <strong>exceeds</strong> the
                Young-Daly optimal interval ({_tau_opt_min:.1f} min).<br>
                The system would spend more time writing checkpoints than computing.
                <strong>Upgrade storage bandwidth or reduce checkpoint size.</strong>
            </div>
        </div>
        """

    mo.vstack([
        mo.Html(f"""
        {_storm_banner}
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Checkpoint Storm Analysis
            </div>
            <div>Model: {_params_b}B params &times; 14 bytes = <strong>{_ckpt_tb:.2f} TB</strong> per checkpoint</div>
            <div>Write time = {_ckpt_gb:.0f} GB / {_storage_bw} GB/s = <strong>{_write_time_min:.1f} minutes</strong></div>
            <div>System MTBF = {GPU_MTTF_HOURS:,}h / {_n_gpus:,} = <strong>{_mtbf_h:.1f} hours</strong></div>
            <div>Young-Daly optimal = sqrt(2 &times; {_write_time_s:.0f}s &times; {_mtbf_s:.0f}s) = <strong>{_tau_opt_min:.1f} minutes</strong></div>
            <div>Write/Optimal ratio = <strong style="color:{COLORS['RedLine'] if _pathological else COLORS['GreenLine']};">{_ckpt_fraction:.2f}x</strong>
                 {'(PATHOLOGICAL)' if _pathological else '(healthy)'}</div>
        </div>
        """),
        mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Ckpt Size</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']}; font-family:monospace;">{_ckpt_tb:.1f}TB</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Write Time</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['RedLine'] if _pathological else COLORS['GreenLine']}; font-family:monospace;">{_write_time_min:.0f}m</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Daily Ckpt Cost</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['OrangeLine']}; font-family:monospace;">${_daily_ckpt_cost:,.0f}</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">System MTBF</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']}; font-family:monospace;">{_mtbf_h:.1f}h</div>
            </div>
        </div>
        """),
        mo.ui.plotly(_fig),
    ])
    return (_pathological,)


# ─── ACT2: PREDICTION REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(partB_prediction, mo):
    if partB_prediction.value == "C":
        mo.callout(mo.md(
            "**Correct.** 175B x 14 bytes = 2.45 TB per checkpoint. At 1 GB/s NFS, that is "
            "2,450 seconds = ~41 minutes. With MTBF of ~50 hours for a 1,000-GPU cluster, "
            "the Young-Daly optimal interval is ~2.6 hours, so 41 minutes is within budget. "
            "But at 10,000 GPUs (MTBF ~5 hours), optimal interval drops to ~27 minutes "
            "-- now the write time *exceeds* the optimal interval. Checkpoint storm."
        ), kind="success")
    elif partB_prediction.value == "A":
        mo.callout(mo.md(
            "**Off by orders of magnitude.** 175B x 14 bytes = 2.45 TB. At 1 GB/s NFS, "
            "writing 2,450 GB takes 2,450 seconds -- not 10 seconds. Even NVMe RAID at "
            "100 GB/s takes ~24.5 seconds. NFS cannot handle frontier-scale checkpoints."
        ), kind="warn")
    elif partB_prediction.value == "B":
        mo.callout(mo.md(
            "**Too optimistic by 20x.** 2.45 TB at 1 GB/s = 2,450 seconds = ~41 minutes, "
            "not 2 minutes. 2 minutes would require 2,450 / 120 = 20 GB/s sustained write -- "
            "faster than most parallel file systems."
        ), kind="warn")
    elif partB_prediction.value == "D":
        mo.callout(mo.md(
            "**Still 8x too optimistic.** 5 minutes = 300 seconds. But 2,450 GB / 300s = "
            "8.2 GB/s required. NFS provides 1 GB/s. Even parallel FS at 10 GB/s "
            "takes 4 minutes. NFS: 41 minutes."
        ), kind="warn")
    return


# ─── ACT2: MATHPEEK ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Governing equations -- checkpoint storm and storage requirements": mo.md("""
        **Checkpoint Size**

        ```
        C = P * (2 + 4 + 4 + 4) = P * 14 bytes
        ```

        - FP16 weights: 2 bytes/param
        - FP32 optimizer momentum (m1): 4 bytes/param
        - FP32 optimizer variance (m2): 4 bytes/param
        - FP32 master weights: 4 bytes/param
        - 175B params: C = 175e9 * 14 = 2.45 TB

        **Checkpoint Storm Condition**

        ```
        T_write > tau_opt  =>  C/BW > sqrt(2 * C/BW * MTBF)
        ```

        Simplifying: C/BW > 2 * MTBF (approximately)
        Storm occurs when storage bandwidth is insufficient relative to
        model size and cluster MTBF.

        **Required Storage Bandwidth to Avoid Storm**

        ```
        BW_min > C / tau_opt = C / sqrt(2 * T_write * MTBF)
        ```
        """)
    })
    return


# ─── ACT2: REFLECTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    partB_reflection = mo.ui.radio(
        options={
            "A) Asynchronous checkpointing -- overlap checkpoint writes with training compute": "A",
            "B) Reduce model size so checkpoints are smaller": "B",
            "C) Increase cluster MTBF by adding redundant GPUs": "C",
            "D) Checkpoint only every hour regardless of the Young-Daly formula": "D",
        },
        label="What is the most practical solution to the checkpoint storm?",
    )
    partB_reflection
    return (partB_reflection,)


@app.cell(hide_code=True)
def _(partB_reflection, mo):
    mo.stop(
        partB_reflection.value is None,
        mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"),
    )
    if partB_reflection.value == "A":
        mo.callout(mo.md(
            "**Correct.** Async checkpointing writes to fast local NVMe while training continues, "
            "then drains to durable storage in the background. This makes T_write effectively "
            "near-zero for the training loop, breaking the checkpoint storm. Combined with "
            "parallel file systems (10-100 GB/s), this is the standard production solution."
        ), kind="success")
    else:
        mo.callout(mo.md(
            "**Not the primary solution.** Asynchronous checkpointing decouples checkpoint "
            "writes from the training loop by writing to fast local NVMe first, then draining "
            "to durable storage in the background. This makes T_write effectively zero for "
            "the training loop, eliminating the checkpoint storm regardless of storage speed."
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
        mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. System MTBF = Component_MTTF / N scales inversely with cluster size.</strong>
                    A 10,000-GPU cluster with 50,000-hour component MTTF experiences failures
                    every 5 hours. At 25,000 GPUs (GPT-4 scale), failures occur every 2 hours.
                    Failure is the common case, not the exception.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. The Young-Daly sweet spot is tau = sqrt(2 * T_write * MTBF).</strong>
                    For a 16,000-GPU cluster (MTBF ~3h) with 2-minute writes, the optimal interval
                    is ~27 minutes -- not 10 minutes (too aggressive) or 90 minutes (too conservative).
                    The U-shaped waste curve has a sharp minimum.
                </div>
                <div>
                    <strong>3. Checkpoint storms occur when T_write > tau_opt.</strong>
                    A 175B model on NFS (1 GB/s) takes ~41 minutes to checkpoint. If the
                    Young-Daly optimal interval is shorter than 41 minutes, the system is
                    pathological. Async checkpointing to local NVMe is the standard fix.
                </div>
            </div>
        </div>
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab V2-07: The Scheduling Trap</strong> &mdash; You designed fault tolerance
                    for a cluster. But scheduling GPU jobs across that cluster is harder than scheduling
                    CPU jobs: heavy-tailed durations, fragmentation, and the utilization paradox.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-fault-tolerance for the Young-Daly derivation and
                    checkpoint storm analysis.<br/>
                    <strong>Build:</strong> TinyTorch checkpoint module &mdash; implement async
                    checkpoint write and restore in <code>tinytorch/src/checkpoint/</code>.
                </div>
            </div>
        </div>
        """),
        mo.accordion({
            "Self-Assessment": mo.md("""
1. What is the system MTBF for a 10,000-GPU cluster with 50,000-hour component MTTF?
2. Using the Young-Daly formula, what is the optimal checkpoint interval for MTBF=3h, T_write=2min?
3. At what storage bandwidth does a 175B model checkpoint create a checkpoint storm on a 10,000-GPU cluster?

*If you cannot answer all three from memory, revisit Acts I and II.*
""")
        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return (decision_input, decision_ui)


@app.cell(hide_code=True)
def _(COLORS, _pathological, partA_prediction, partB_prediction, partA_reflection, partB_reflection,
      ledger, mo, decision_input, decision_ui):
    ledger.save(
        chapter="v2_06",
        design={
            "partA_prediction": partA_prediction.value or "no_selection",
            "partA_correct": partA_prediction.value == "C",
            "partA_reflection": partA_reflection.value or "no_selection",
            "partB_prediction": partB_prediction.value or "no_selection",
            "partB_correct": partB_prediction.value == "C",
            "partB_reflection": partB_reflection.value or "no_selection",
            "student_justification": str(decision_input.value),
            "checkpoint_storm_hit": _pathological,
        },
    )

    _a1_ok = partA_prediction.value == "C"
    _a2_ok = partB_prediction.value == "C"
    _tier = "Optimal" if (_a1_ok and _a2_ok) else ("Partial" if (_a1_ok or _a2_ok) else "Developing")
    _tier_color = COLORS["GreenLine"] if _tier == "Optimal" else (COLORS["OrangeLine"] if _tier == "Partial" else COLORS["TextMuted"])

    decision_ui
    mo.Html(f"""
    <div class="lab-hud">
        <div><span class="hud-label">LAB</span> <span class="hud-value">Vol2 &middot; Lab 06</span></div>
        <div><span class="hud-label">CHAPTER</span> <span class="hud-value">v2_06 &middot; Fault Tolerance</span></div>
        <div><span class="hud-label">ACT I</span> <span class="{'hud-active' if _a1_ok else 'hud-none'}">{"CORRECT" if _a1_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">ACT II</span> <span class="{'hud-active' if _a2_ok else 'hud-none'}">{"CORRECT" if _a2_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">TIER</span> <span style="color:{_tier_color}; font-family:var(--font-mono);">{_tier.upper()}</span></div>
        <div><span class="hud-label">STORM</span> <span class="{'hud-none' if _pathological else 'hud-active'}">{"YES" if _pathological else "NO"}</span></div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
