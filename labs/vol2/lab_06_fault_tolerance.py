import marimo

__generated_with = "0.23.1"
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
# Structure (35-40 minutes):
#   Part A  — Young-Daly Sweet Spot (12-15 min)
#             Quick recall: 10,000-GPU cluster fails every ~5 hours.
#             Then the Young-Daly U-curve reveals the checkpoint interval.
#
#   Part B  — The Checkpoint Storm (20-25 min)
#             175B checkpoint on NFS takes 41 minutes -- longer than the
#             optimal interval. Checkpoint storms at frontier scale.
#
#   Synthesis — Key Takeaways + Decision Log
#
# Hardware Constants:
#   GPU_MTTF_HOURS    = 50,000   (from mlsysim defaults)
#   H100_RAM_GB       = 80       (NVIDIA H100 SXM5 spec)
#   H100_COST_HR      = 3.0      ($3/GPU-hour cloud pricing)
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

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install(
            "../../wheels/mlsysim-0.1.0-py3-none-any.whl", keep_going=False
        )
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog
    from mlsysim import Hardware
    from mlsysim.core.defaults import GPU_MTTF_HOURS

    # ── Hardware registry ─────────────────────────────────────────────────
    H100 = Hardware.Cloud.H100
    A100 = Hardware.Cloud.A100
    EDGE = Hardware.Edge.JetsonOrinNX
    H100_RAM_GB = H100.memory.capacity.m_as("GB")
    EDGE_RAM_GB = EDGE.memory.capacity.m_as("GB")
    GPU_COST_HR = 3.0     # $/GPU-hour cloud pricing
    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return COLORS, LAB_CSS, apply_plotly_theme, go, ledger, math, mo, np, GPU_MTTF_HOURS, GPU_COST_HR, DecisionLog, Hardware, H100, A100, EDGE, H100_RAM_GB, EDGE_RAM_GB


# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, LAB_CSS, mo):
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
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
                <span class="badge badge-warn">45&ndash;55 minutes &middot; 4 Parts + Synthesis</span>
            </div>
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
                    Reliability collapse from the Scale chapter &middot;
                    MTBF_system = MTBF_component / N from the Fault Tolerance chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35-40 min</strong><br/>
                    Part A: ~15 min &middot; Part B: ~25 min
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

    - **The Fault Tolerance chapter** -- MTBF scaling, Young-Daly checkpoint model
    - **The Scale chapter** -- Fleet reliability collapse (V2-01 recap)
    - The Checkpoint Storm section -- Storage bandwidth saturation at scale
    - The Serving Fault Tolerance section -- KV cache state and replica failover
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: WIDGET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 4: PART A WIDGETS ──────────────────────────────────────────────────
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
    a1_cluster_gpus = mo.ui.slider(start=1000, stop=25000, value=16000, step=1000, label="Cluster GPUs")
    a1_write_time_s = mo.ui.slider(start=10, stop=300, value=120, step=10, label="Checkpoint write time (seconds)")
    a1_interval_s = mo.ui.slider(start=60, stop=10800, value=600, step=60, label="Your checkpoint interval (seconds)")
    return (partA_prediction, a1_cluster_gpus, a1_write_time_s, a1_interval_s)


# ─── CELL 5: PART A REFLECTION + PART B PREDICTION WIDGETS ──────────────────
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

    partB_prediction = mo.ui.radio(
        options={
            "A) ~10 seconds -- fast with modern storage": "A",
            "B) ~2 minutes -- manageable": "B",
            "C) ~41 minutes -- longer than the Young-Daly optimal interval at this cluster size": "C",
            "D) ~5 minutes -- within budget": "D",
        },
        label="A 175B model checkpoints on a 1,000-GPU cluster with NFS storage (1 GB/s). How long does one checkpoint take?",
    )
    return (partA_reflection, partB_prediction)


# ─── CELL 6: PART B CONTROLS + SYNTHESIS WIDGETS ────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    a2_model_b = mo.ui.slider(start=1, stop=175, value=175, step=1, label="Model size (B params)")
    a2_cluster_gpus = mo.ui.slider(start=100, stop=25000, value=1000, step=100, label="Cluster GPUs")
    a2_storage = mo.ui.dropdown(
        options={"NFS (1 GB/s)": 1.0, "Parallel FS (10 GB/s)": 10.0, "NVMe RAID (100 GB/s)": 100.0},
        value="NFS (1 GB/s)",
        label="Storage type",
    )

    partB_reflection = mo.ui.radio(
        options={
            "A) Asynchronous checkpointing -- overlap checkpoint writes with training compute": "A",
            "B) Reduce model size so checkpoints are smaller": "B",
            "C) Increase cluster MTBF by adding redundant GPUs": "C",
            "D) Checkpoint only every hour regardless of the Young-Daly formula": "D",
        },
        label="What is the most practical solution to the checkpoint storm?",
    )
    return (a2_model_b, a2_cluster_gpus, a2_storage, partB_reflection)


# ─── CELL 6b: PART C WIDGETS ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    partC_prediction = mo.ui.radio(
        options={
            "A) 10 seconds -- NVMe is fast": "A",
            "B) ~24 seconds -- local NVMe first, then drain": "B",
            "C) ~1 second -- async hides all write time": "C",
            "D) Same as NFS -- 41 minutes": "D",
        },
        label="With async checkpointing (write to local NVMe RAID at 100 GB/s, then drain), how long does the training loop pause for a 175B checkpoint?",
    )
    c1_nvme_bw = mo.ui.slider(start=10, stop=200, value=100, step=10, label="Local NVMe BW (GB/s)")
    c1_drain_bw = mo.ui.slider(start=1, stop=50, value=10, step=1, label="Background drain BW to durable storage (GB/s)")
    c1_cluster_gpus = mo.ui.slider(start=1000, stop=25000, value=10000, step=1000, label="Cluster GPUs")
    partC_reflection = mo.ui.radio(
        options={
            "A) Async checkpointing eliminates all checkpoint overhead": "A",
            "B) Async reduces training-loop pause to NVMe write time but requires enough local NVMe capacity for at least 2 checkpoints": "B",
            "C) Async is free -- no additional hardware cost": "C",
            "D) Async is unnecessary if you use a fast parallel file system": "D",
        },
        label="What is the key requirement for async checkpointing to work?",
    )
    return (partC_prediction, c1_nvme_bw, c1_drain_bw, c1_cluster_gpus, partC_reflection)


# ─── CELL 6c: PART D WIDGETS ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    partD_prediction = mo.ui.radio(
        options={
            "A) 1 replica -- just restart the failed one": "A",
            "B) N+1 replicas -- one cold spare per service": "B",
            "C) It depends: replica count = ceil(QPS * recovery_time / max_queue_depth)": "C",
            "D) 2x replicas -- always double for safety": "D",
        },
        label="Your inference service runs 10 replicas at 1000 QPS total. One replica fails (MTBF ~30 days). Recovery takes 90 seconds (model reload). How many spare replicas do you need to maintain the SLO?",
    )
    d1_replicas = mo.ui.slider(start=4, stop=32, value=10, step=1, label="Active replicas")
    d1_qps = mo.ui.slider(start=100, stop=5000, value=1000, step=100, label="Total QPS")
    d1_recovery_s = mo.ui.slider(start=10, stop=300, value=90, step=10, label="Recovery time (seconds)")
    d1_slo_p99_ms = mo.ui.slider(start=100, stop=2000, value=500, step=50, label="P99 latency SLO (ms)")
    partD_reflection = mo.ui.radio(
        options={
            "A) Serving fault tolerance is simpler than training -- just add replicas": "A",
            "B) Serving fault tolerance is harder because KV cache state is lost on failure and must be rebuilt per-request": "B",
            "C) Serving fault tolerance is the same problem as training fault tolerance": "C",
            "D) Serving does not need fault tolerance -- requests are stateless": "D",
        },
        label="How does serving fault tolerance differ from training fault tolerance?",
    )
    return (partD_prediction, d1_replicas, d1_qps, d1_recovery_s, d1_slo_p99_ms, partD_reflection)


# ─── CELL 7: DECISION LOG WIDGET ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(DecisionLog, mo, partD_reflection):
    synth_decision_input, synth_decision_ui = DecisionLog(
        placeholder="Based on what I learned in this lab, the most important insight about "
                    "fault tolerance at scale is..."
    )
    return (synth_decision_input, synth_decision_ui)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: SINGLE TABS CELL
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 8: TABS ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS, apply_plotly_theme, go, math, mo, np,
    GPU_MTTF_HOURS, GPU_COST_HR, H100_RAM_GB,
    partA_prediction, a1_cluster_gpus, a1_write_time_s, a1_interval_s,
    partA_reflection,
    partB_prediction, a2_model_b, a2_cluster_gpus, a2_storage,
    partB_reflection,
    partC_prediction, c1_nvme_bw, c1_drain_bw, c1_cluster_gpus,
    partC_reflection,
    partD_prediction, d1_replicas, d1_qps, d1_recovery_s, d1_slo_p99_ms,
    partD_reflection,
    synth_decision_input, synth_decision_ui,
    ledger,
):
    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER -- The Young-Daly Sweet Spot
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        # ── Stakeholder message ────────────────────────────────────────
        items.append(mo.Html(f"""
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
    """))

        # ── Concept framing ───────────────────────────────────────────
        items.append(mo.md("""
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
    """))

        items.append(mo.callout(mo.md(
            "**Caveat:** The MTBF formula assumes independent failures. In practice, failures are often "
            "correlated: a power supply failure takes down 8 GPUs, a switch failure isolates an entire "
            "rack (64-128 GPUs). Correlated failures can increase effective failure rates by 2-5x over "
            "the independent model."
        ), kind="info"))

        # ── Prediction lock ───────────────────────────────────────────
        items.append(mo.md("### Your Prediction"))
        items.append(partA_prediction)

        if partA_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part A instruments."), kind="warn"))
            return mo.vstack(items)

        # ── Instruments ───────────────────────────────────────────────
        items.append(mo.md("### Young-Daly Checkpoint Optimizer"))
        items.append(mo.hstack([a1_cluster_gpus, a1_write_time_s, a1_interval_s], justify="center", gap=2))

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

        # ── U-curve chart ─────────────────────────────────────────────
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

        items.append(mo.Html(f"""
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
        """))

        items.append(mo.Html(f"""
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
        """))

        items.append(mo.ui.plotly(_fig))

        # ── Prediction reveal ─────────────────────────────────────────
        if partA_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Correct.** The Young-Daly formula gives tau_opt = sqrt(2 * 120s * 10,800s) = "
                "~1,610 seconds = ~27 minutes. This is the geometric mean between write time and "
                "MTBF, not the arithmetic mean. The square root law means the optimal interval "
                "is much closer to the write time than to the MTBF."
            ), kind="success"))
        elif partA_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Far too aggressive.** Checkpointing every 2 minutes with a 2-minute write time "
                "means 50% of wall time is spent checkpointing. Only 50% remains for training. "
                "The optimal interval balances overhead against rework, not minimizes rework alone."
            ), kind="warn"))
        elif partA_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Too aggressive.** At 10-minute intervals, checkpoint overhead is 2/10 = 20%, "
                "plus rework of 10/(2*180) = 2.8%, total = 22.8%. The optimal 27-minute interval "
                "achieves only 12.3% waste -- saving nearly 10 percentage points of compute time."
            ), kind="warn"))
        elif partA_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Too conservative.** At 90 minutes with MTBF of 3 hours, expected rework is "
                "90/(2*180) = 25%. Combined with 2/90 = 2.2% overhead, total waste is 27.2%. "
                "The optimal 27-minute interval achieves 12.3% -- half the waste."
            ), kind="warn"))

        # ── MathPeek ──────────────────────────────────────────────────
        items.append(mo.accordion({
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
        }))

        # ── Reflection ────────────────────────────────────────────────
        items.append(mo.md("### Reflection"))
        items.append(partA_reflection)

        if partA_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"))
        elif partA_reflection.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** The optimal waste is proportional to sqrt(T_write / MTBF). Reducing "
                "T_write by 4x (e.g., from NFS to parallel FS) reduces waste by 2x. Async "
                "checkpointing effectively makes T_write appear near-zero for the training loop. "
                "This is the most actionable lever because storage is within the team's control."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Not the primary lever.** The Young-Daly waste formula is 2 * sqrt(T_write / (2 * MTBF)). "
                "Reducing T_write has the most direct impact because storage technology is within "
                "engineering control, unlike hardware reliability (MTBF)."
            ), kind="warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER -- The Checkpoint Storm
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        # ── Stakeholder message ────────────────────────────────────────
        items.append(mo.Html(f"""
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
    """))

        # ── Concept framing ───────────────────────────────────────────
        items.append(mo.md("""
    A 175B model checkpoint includes weights + optimizer states + gradients:
    - 175B parameters x 14 bytes (FP16 weights + FP32 Adam m1 + m2 + master) = **2.45 TB** per checkpoint
    - At NFS 1 GB/s aggregate write: 2,450 seconds = **~41 minutes**

    With MTBF of ~5 hours (1,000 GPUs), the Young-Daly optimal interval is:
    tau_opt = sqrt(2 * 2450 * 18,000) = sqrt(88,200,000) = ~9,393s = ~2.6 hours

    But wait -- at 1 GB/s NFS, checkpoint write time (41 min) is already a significant
    fraction of the optimal interval (2.6 hours). With faster storage (100 GB/s NVMe RAID),
    the same checkpoint takes only 24.5 seconds.

    What happens at a larger cluster (10,000 GPUs) where MTBF drops to ~5 hours?
    """))

        # ── Prediction lock ───────────────────────────────────────────
        items.append(mo.md("### Your Prediction"))
        items.append(partB_prediction)

        if partB_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part B instruments."), kind="warn"))
            return mo.vstack(items)

        # ── Instruments ───────────────────────────────────────────────
        items.append(mo.md("### Checkpoint Storm Analyzer"))
        items.append(mo.hstack([a2_model_b, a2_cluster_gpus, a2_storage], justify="center", gap=2))

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

        # ── Storage comparison chart ──────────────────────────────────
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

        # ── Failure banner ────────────────────────────────────────────
        if _pathological:
            items.append(mo.Html(f"""
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
        """))

        items.append(mo.Html(f"""
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
        """))

        items.append(mo.Html(f"""
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
        """))

        items.append(mo.ui.plotly(_fig))

        # ── Prediction reveal ─────────────────────────────────────────
        if partB_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Correct.** 175B x 14 bytes = 2.45 TB per checkpoint. At 1 GB/s NFS, that is "
                "2,450 seconds = ~41 minutes. With MTBF of ~50 hours for a 1,000-GPU cluster, "
                "the Young-Daly optimal interval is ~2.6 hours, so 41 minutes is within budget. "
                "But at 10,000 GPUs (MTBF ~5 hours), optimal interval drops to ~27 minutes "
                "-- now the write time *exceeds* the optimal interval. Checkpoint storm."
            ), kind="success"))
        elif partB_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Off by orders of magnitude.** 175B x 14 bytes = 2.45 TB. At 1 GB/s NFS, "
                "writing 2,450 GB takes 2,450 seconds -- not 10 seconds. Even NVMe RAID at "
                "100 GB/s takes ~24.5 seconds. NFS cannot handle frontier-scale checkpoints."
            ), kind="warn"))
        elif partB_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Too optimistic by 20x.** 2.45 TB at 1 GB/s = 2,450 seconds = ~41 minutes, "
                "not 2 minutes. 2 minutes would require 2,450 / 120 = 20 GB/s sustained write -- "
                "faster than most parallel file systems."
            ), kind="warn"))
        elif partB_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Still 8x too optimistic.** 5 minutes = 300 seconds. But 2,450 GB / 300s = "
                "8.2 GB/s required. NFS provides 1 GB/s. Even parallel FS at 10 GB/s "
                "takes 4 minutes. NFS: 41 minutes."
            ), kind="warn"))

        # ── MathPeek ──────────────────────────────────────────────────
        items.append(mo.accordion({
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
        }))

        # ── Reflection ────────────────────────────────────────────────
        items.append(mo.md("### Reflection"))
        items.append(partB_reflection)

        if partB_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"))
        elif partB_reflection.value == "A":
            items.append(mo.callout(mo.md(
                "**Correct.** Async checkpointing writes to fast local NVMe while training continues, "
                "then drains to durable storage in the background. This makes T_write effectively "
                "near-zero for the training loop, breaking the checkpoint storm. Combined with "
                "parallel file systems (10-100 GB/s), this is the standard production solution."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Not the primary solution.** Asynchronous checkpointing decouples checkpoint "
                "writes from the training loop by writing to fast local NVMe first, then draining "
                "to durable storage in the background. This makes T_write effectively zero for "
                "the training loop, eliminating the checkpoint storm regardless of storage speed."
            ), kind="warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER -- Async Checkpointing
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['OrangeLine']}; background: {COLORS['OrangeLL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['OrangeLine']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Systems Architect
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "The checkpoint storm from Part B is blocking our 175B training run.
            Someone suggested async checkpointing -- write to local NVMe first, then drain
            to durable storage in the background. How much does this actually help?"
        </div>
    </div>
    """))

        # Concept framing
        items.append(mo.md("""
    Asynchronous checkpointing splits the write into two phases:

    1. **Snapshot to local NVMe**: Each GPU writes its shard to local NVMe RAID.
       At 100 GB/s per node, 8 GPUs write their shards in parallel.

    2. **Background drain**: A background thread copies the snapshot from NVMe
       to durable storage (parallel FS or object store) while training resumes.

    The training loop pauses only for Phase 1. For a 175B model:
    - Checkpoint per node = 2.45 TB / (N_nodes) -- sharded across nodes
    - At 100 GB/s local NVMe: pause = shard_size / 100 GB/s

    For 256 GPUs (32 nodes): per-node shard = 2.45 TB / 32 = ~77 GB.
    NVMe pause = 77 GB / 100 GB/s = **~0.8 seconds** (vs 41 min on NFS).

    The catch: local NVMe must hold at least 2 checkpoints (current + previous)
    for rollback safety. At 77 GB per checkpoint, that requires ~154 GB local NVMe per node.
        """))

        # Prediction lock
        items.append(mo.md("### Your Prediction"))
        items.append(partC_prediction)

        if partC_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part C instruments."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.md("### Async Checkpoint Analyzer"))
        items.append(mo.hstack([c1_nvme_bw, c1_drain_bw, c1_cluster_gpus], justify="center", gap=2))

        # Physics
        _params_b = 175
        _ckpt_gb = _params_b * 1e9 * 14 / 1e9  # 2450 GB
        _n_gpus = c1_cluster_gpus.value
        _n_nodes = max(_n_gpus // 8, 1)
        _nvme_bw = c1_nvme_bw.value
        _drain_bw = c1_drain_bw.value

        _shard_gb = _ckpt_gb / _n_nodes
        _nvme_pause_s = _shard_gb / _nvme_bw
        _drain_time_s = _ckpt_gb / _drain_bw  # total drain time
        _nfs_time_s = _ckpt_gb / 1.0  # NFS baseline
        _nfs_time_min = _nfs_time_s / 60

        _mtbf_s = GPU_MTTF_HOURS * 3600 / _n_gpus
        _tau_opt = math.sqrt(2 * _nvme_pause_s * _mtbf_s)
        _tau_opt_min = _tau_opt / 60

        _waste_async = _nvme_pause_s / _tau_opt + _tau_opt / (2 * _mtbf_s) if _tau_opt > 0 else 1.0
        _waste_nfs = _nfs_time_s / math.sqrt(2 * _nfs_time_s * _mtbf_s) + math.sqrt(2 * _nfs_time_s * _mtbf_s) / (2 * _mtbf_s) if _mtbf_s > 0 else 1.0

        _speedup = _nfs_time_s / max(_nvme_pause_s, 0.001)
        _nvme_storage_needed_gb = _shard_gb * 2  # 2 checkpoints

        # Comparison chart
        _methods = ["NFS (1 GB/s)", "Parallel FS (10 GB/s)", f"Async NVMe ({_nvme_bw} GB/s)"]
        _pause_times = [_nfs_time_s, _ckpt_gb / 10.0, _nvme_pause_s]
        _bar_colors = [COLORS["RedLine"], COLORS["OrangeLine"], COLORS["GreenLine"]]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=_methods, y=_pause_times,
            marker_color=_bar_colors,
            text=[f"{t:.1f}s" if t < 60 else f"{t/60:.1f}m" for t in _pause_times],
            textposition="auto",
        ))
        _fig.update_layout(
            height=300,
            xaxis=dict(title="Checkpoint Method"),
            yaxis=dict(title="Training Loop Pause (seconds)", type="log"),
            margin=dict(t=30, b=50, l=50, r=20),
            showlegend=False,
        )
        apply_plotly_theme(_fig)

        _pause_color = COLORS["GreenLine"] if _nvme_pause_s < 5 else (COLORS["OrangeLine"] if _nvme_pause_s < 60 else COLORS["RedLine"])

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Async Checkpoint Analysis
            </div>
            <div>Checkpoint: 175B &times; 14 bytes = <strong>{_ckpt_gb:.0f} GB</strong></div>
            <div>Nodes: {_n_nodes} &mdash; shard per node: <strong>{_shard_gb:.1f} GB</strong></div>
            <div>NVMe pause = {_shard_gb:.1f} / {_nvme_bw} = <strong style="color:{_pause_color};">{_nvme_pause_s:.2f}s</strong></div>
            <div>NFS baseline: <strong style="color:{COLORS['RedLine']};">{_nfs_time_min:.1f} min</strong></div>
            <div>Speedup: <strong style="color:{COLORS['GreenLine']};">{_speedup:.0f}x</strong></div>
            <div>Local NVMe required: <strong>{_nvme_storage_needed_gb:.0f} GB</strong> per node (2 checkpoints)</div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">NVMe Pause</div>
                <div style="font-size:2rem; font-weight:800; color:{_pause_color}; font-family:monospace;">{_nvme_pause_s:.1f}s</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">vs {_nfs_time_min:.0f}m NFS</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Speedup</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['GreenLine']}; font-family:monospace;">{_speedup:.0f}x</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Waste (async)</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['GreenLine']}; font-family:monospace;">{_waste_async*100:.1f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">vs {_waste_nfs*100:.1f}% NFS</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # Prediction reveal
        if partC_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Correct.** With 256 GPUs across 32 nodes, each node writes ~77 GB to local NVMe "
                "at 100 GB/s = 0.8 seconds. The training loop pauses for less than 1 second. "
                "Background drain to durable storage takes minutes, but training has already resumed. "
                "This reduces Young-Daly waste from ~25% (NFS) to under 2%."
            ), kind="success"))
        elif partC_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Close but wrong phase.** 10 seconds would be the time to drain to NVMe if the "
                "entire checkpoint were written to one node. With sharding across 32 nodes, "
                "each writes only 77 GB -- pause is under 1 second."
            ), kind="warn"))
        elif partC_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**That is the full checkpoint size / NVMe BW.** 2450 GB / 100 GB/s = 24.5 seconds. "
                "But with 32 nodes writing in parallel, each writes only 77 GB / 100 = 0.8 seconds. "
                "The parallelism across nodes is the key insight."
            ), kind="warn"))
        elif partC_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Async does not use NFS for the training pause.** The whole point is to write to "
                "fast local NVMe first (sub-second), then drain to NFS/parallel FS in the background."
            ), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Governing equations -- async checkpointing": mo.md("""
        **Async Checkpoint Pause**

        ```
        T_pause = (C / N_nodes) / BW_nvme
        ```

        - C = total checkpoint size (bytes)
        - N_nodes = cluster nodes (each writes its shard in parallel)
        - BW_nvme = per-node NVMe write bandwidth

        **Background Drain Time**

        ```
        T_drain = C / BW_durable_storage
        ```

        T_drain > T_pause is expected and acceptable -- training continues during drain.

        **Requirement**: Local NVMe must hold >= 2 checkpoints:
        - Current checkpoint being written
        - Previous checkpoint for rollback safety
        - NVMe_capacity >= 2 * C / N_nodes
            """)
        }))

        # Reflection
        items.append(mo.md("### Reflection"))
        items.append(partC_reflection)

        if partC_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"))
        elif partC_reflection.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** Async checkpointing reduces the training pause to NVMe write time, "
                "but requires: (1) enough local NVMe capacity for 2 checkpoints per node, "
                "(2) background drain bandwidth sufficient to drain before the next checkpoint. "
                "For 175B at 32 nodes: 154 GB NVMe per node. Most DGX nodes ship with 2-4 TB NVMe."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Not quite.** Async checkpointing is not free -- it requires local NVMe storage "
                "capacity for at least 2 checkpoints per node. The training pause is reduced to "
                "NVMe write time, but if drain bandwidth is too slow, checkpoints can pile up "
                "and exhaust local storage."
            ), kind="warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER -- Serving Fault Tolerance
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['RedLine']}; background: {COLORS['RedLL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['RedLine']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; SRE Team Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "Our inference service runs 10 replicas serving 1,000 QPS total with a 500ms P99 SLO.
            When a replica fails, it takes 90 seconds to reload the 70B model. During recovery,
            the remaining 9 replicas must absorb the extra traffic. Are we SLO-safe, or do we
            need spare replicas?"
        </div>
    </div>
    """))

        # Concept framing
        items.append(mo.md("""
    Training fault tolerance is about **protecting state** (checkpoint/restore).
    Serving fault tolerance is about **maintaining throughput** during failures.

    Key differences:
    - **Training**: lose GPU -> lose progress since last checkpoint. Restart from checkpoint.
    - **Serving**: lose replica -> lose KV cache state for all in-flight requests. Those
      requests must be re-issued to other replicas. The remaining replicas must absorb
      the extra QPS without violating the latency SLO.

    The critical question: can N-1 replicas handle the full QPS within the SLO
    during the recovery window?

    Per-replica capacity = QPS_total / N. After one failure, load per survivor =
    QPS_total / (N - 1). If this exceeds per-replica max throughput, requests queue
    and P99 latency spikes above the SLO.
        """))

        # Prediction lock
        items.append(mo.md("### Your Prediction"))
        items.append(partD_prediction)

        if partD_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part D instruments."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.md("### Serving Fault Tolerance Analyzer"))
        items.append(mo.hstack([
            mo.vstack([d1_replicas, d1_qps]),
            mo.vstack([d1_recovery_s, d1_slo_p99_ms]),
        ], justify="center", gap=2))

        # Physics
        _replicas = d1_replicas.value
        _qps = d1_qps.value
        _recovery_s = d1_recovery_s.value
        _slo_ms = d1_slo_p99_ms.value

        _qps_per_replica = _qps / _replicas
        _qps_after_failure = _qps / max(_replicas - 1, 1)

        # Simple capacity model: max throughput per replica before SLO violation
        # Assume each replica can handle ~1.3x its normal load before SLO breach
        _max_qps_per_replica = _qps_per_replica * 1.3
        _overloaded = _qps_after_failure > _max_qps_per_replica
        _overflow_qps = max(0, _qps_after_failure - _max_qps_per_replica)

        # Requests dropped during recovery
        _requests_at_risk = _overflow_qps * _recovery_s if _overloaded else 0
        _slo_violation = _overloaded

        # Spare replicas needed
        _spare_needed = 0
        for _s in range(0, 5):
            _test_survivors = _replicas + _s - 1
            if _test_survivors > 0 and _qps / _test_survivors <= _max_qps_per_replica:
                _spare_needed = _s
                break
        else:
            _spare_needed = 5

        # Cost of spares
        _spare_cost_day = _spare_needed * 8 * GPU_COST_HR * 24  # 8 GPUs per replica

        # Chart: latency vs replica count during failure
        _replica_range = list(range(max(_replicas - 3, 2), _replicas + 5))
        _latency_factor = []
        for _r in _replica_range:
            _survivors = _r - 1
            if _survivors <= 0:
                _latency_factor.append(999)
                continue
            _load_ratio = _qps / _survivors / _qps_per_replica
            # Approximate: latency scales as 1/(1 - load_ratio/capacity) for queueing
            _lat = _slo_ms * min(max(_load_ratio, 1.0), 5.0) if _load_ratio < 1.3 else _slo_ms * 5
            _latency_factor.append(_lat)

        _fig = go.Figure()
        _colors_bar = [COLORS["GreenLine"] if lat <= _slo_ms else COLORS["RedLine"] for lat in _latency_factor]
        _fig.add_trace(go.Bar(
            x=[str(r) for r in _replica_range], y=_latency_factor,
            marker_color=_colors_bar,
            text=[f"{l:.0f}ms" for l in _latency_factor],
            textposition="auto",
        ))
        _fig.add_hline(y=_slo_ms, line=dict(color=COLORS["OrangeLine"], width=2, dash="dash"),
                       annotation_text=f"P99 SLO: {_slo_ms}ms", annotation_position="top right")
        _fig.update_layout(
            height=300,
            xaxis=dict(title="Total Replicas (one fails)"),
            yaxis=dict(title="Estimated P99 Latency (ms)", range=[0, max(max(_latency_factor) * 1.2, _slo_ms * 2)]),
            margin=dict(t=30, b=50, l=50, r=20),
            showlegend=False,
        )
        apply_plotly_theme(_fig)

        _status_color = COLORS["RedLine"] if _slo_violation else COLORS["GreenLine"]
        _status_text = "SLO VIOLATED" if _slo_violation else "SLO MAINTAINED"

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Serving Fault Tolerance
            </div>
            <div>Replicas: {_replicas} &mdash; QPS/replica: {_qps_per_replica:.0f} &mdash; Total QPS: {_qps}</div>
            <div>After 1 failure: QPS/survivor = {_qps_after_failure:.0f} (max safe: {_max_qps_per_replica:.0f})</div>
            <div>Recovery time: {_recovery_s}s &mdash; Requests at risk: <strong>{_requests_at_risk:.0f}</strong></div>
            <div>Status: <strong style="color:{_status_color};">{_status_text}</strong></div>
            <div>Spare replicas needed: <strong>{_spare_needed}</strong> (cost: ${_spare_cost_day:,.0f}/day)</div>
        </div>
        """))

        # SLO violation banner
        if _slo_violation:
            items.append(mo.Html(f"""
        <div style="background:{COLORS['RedLL']}; border:2px solid {COLORS['RedLine']};
                    border-radius:10px; padding:14px 18px; margin:10px 0;">
            <div style="font-size:0.88rem; font-weight:800; color:{COLORS['RedLine']}; margin-bottom:4px;">
                SLO VIOLATION &mdash; Insufficient Replicas
            </div>
            <div style="font-size:0.85rem; color:#7f1d1d; line-height:1.6;">
                After 1 replica failure, remaining {_replicas - 1} replicas receive {_qps_after_failure:.0f} QPS each
                (max safe: {_max_qps_per_replica:.0f}). ~{_requests_at_risk:.0f} requests will breach the {_slo_ms}ms SLO
                during the {_recovery_s}s recovery window. <strong>Add {_spare_needed} spare replica(s).</strong>
            </div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Status</div>
                <div style="font-size:1.4rem; font-weight:800; color:{_status_color}; font-family:monospace;">{_status_text.split()[0]}</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Spares Needed</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['OrangeLine']}; font-family:monospace;">{_spare_needed}</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Recovery</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']}; font-family:monospace;">{_recovery_s}s</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # Prediction reveal
        if partD_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Correct.** The required spare count depends on the specific numbers. With 10 "
                "replicas at 100 QPS each, losing 1 pushes survivors to 111 QPS each. If max "
                "safe capacity is ~130 QPS/replica, 9 survivors can absorb the load. But with "
                "fewer replicas or higher QPS, you need N+1 or N+2 provisioning."
            ), kind="success"))
        elif partD_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Dangerous assumption.** During the 90-second recovery, the remaining 9 replicas "
                "must handle 111% of their normal load. If they were already at 80% capacity, "
                "the extra 11% pushes them past the SLO threshold. 'Just restart' works only "
                "if you have headroom."
            ), kind="warn"))
        elif partD_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Conservative but not always necessary.** N+1 guarantees one failure tolerance "
                "but costs 8 extra GPUs ($576/day). If the existing replicas have >30% headroom, "
                "they can absorb the load without a spare."
            ), kind="warn"))
        elif partD_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Wasteful.** 2x replicas provides excellent fault tolerance but doubles GPU cost. "
                "The right answer depends on the specific QPS, SLO, and recovery time. "
                "Most production systems use N+1 or N+2, not N*2."
            ), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Governing equations -- serving fault tolerance": mo.md("""
        **Load After Failure**

        ```
        QPS_per_survivor = QPS_total / (N - k)
        ```

        - N = total replicas, k = failed replicas
        - Each survivor must handle more traffic

        **SLO Violation Condition**

        ```
        QPS_per_survivor > max_capacity_per_replica  =>  SLO violated
        requests_at_risk = (QPS_per_survivor - max_capacity) * recovery_time
        ```

        **Serving vs Training Fault Tolerance**

        | Property | Training | Serving |
        |----------|----------|---------|
        | State | Checkpoint (GB-TB) | KV cache (GB, per-request) |
        | Recovery | Restore from storage | Reload model weights |
        | Cost of failure | Lost compute since checkpoint | Dropped/delayed requests |
        | Mitigation | Checkpoint frequency | Spare replicas |
            """)
        }))

        # Reflection
        items.append(mo.md("### Reflection"))
        items.append(partD_reflection)

        if partD_reflection.value is None:
            items.append(mo.callout(mo.md("Select an answer to see the explanation."), kind="warn"))
        elif partD_reflection.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** Training fault tolerance preserves long-lived state (checkpoint/restore). "
                "Serving fault tolerance maintains throughput. When a serving replica fails, its KV cache "
                "(active request state) is lost entirely. Those requests must restart from scratch on "
                "another replica, and the replica must reload ~140 GB of model weights before serving again."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Not quite.** Serving fault tolerance differs fundamentally because KV cache state "
                "is per-request and ephemeral. When a replica fails, all in-flight request state is "
                "lost. Recovery requires reloading model weights (90+ seconds for 70B), during which "
                "remaining replicas must absorb the extra traffic."
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
                    <strong>Read:</strong> the Fault Tolerance chapter for the Young-Daly derivation and
                    checkpoint storm analysis.<br/>
                    <strong>Build:</strong> TinyTorch checkpoint module &mdash; implement async
                    checkpoint write and restore in <code>tinytorch/src/checkpoint/</code>.
                </div>
            </div>
        </div>
        """))

        items.append(mo.accordion({
            "Self-Assessment": mo.md("""
1. What is the system MTBF for a 10,000-GPU cluster with 50,000-hour component MTTF?
2. Using the Young-Daly formula, what is the optimal checkpoint interval for MTBF=3h, T_write=2min?
3. At what storage bandwidth does a 175B model checkpoint create a checkpoint storm on a 10,000-GPU cluster?

*If you cannot answer all three from memory, revisit Parts A and B.*
""")
        }))

        items.append(mo.md("---"))
        items.append(mo.md("### Decision Log"))
        items.append(mo.md("Record the single most important insight from this lab. "
                           "This entry carries forward to future labs via the Design Ledger."))
        items.append(synth_decision_ui)

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    # Save ledger
    ledger.save(
        chapter="v2_06",
        design={
            "partA_prediction": partA_prediction.value or "no_selection",
            "partA_correct": partA_prediction.value == "C",
            "partA_reflection": partA_reflection.value or "no_selection",
            "partB_prediction": partB_prediction.value or "no_selection",
            "partB_correct": partB_prediction.value == "C",
            "partB_reflection": partB_reflection.value or "no_selection",
            "partC_prediction": partC_prediction.value or "no_selection",
            "partC_correct": partC_prediction.value == "C",
            "partC_reflection": partC_reflection.value or "no_selection",
            "partD_prediction": partD_prediction.value or "no_selection",
            "partD_correct": partD_prediction.value == "C",
            "partD_reflection": partD_reflection.value or "no_selection",
            "student_justification": str(synth_decision_input.value),
        },
    )

    tabs = mo.ui.tabs({
        "Part A -- The Young-Daly Sweet Spot":  build_part_a(),
        "Part B -- The Checkpoint Storm":       build_part_b(),
        "Part C -- Async Checkpointing":        build_part_c(),
        "Part D -- Serving Fault Tolerance":    build_part_d(),
        "Synthesis":                            build_synthesis(),
    })
    tabs
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 9: LEDGER_HUD ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, partA_prediction, partB_prediction, mo):
    mo.stop(partA_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    _a1_ok = partA_prediction.value == "C"
    _a2_ok = partB_prediction.value == "C"
    _tier = "Optimal" if (_a1_ok and _a2_ok) else ("Partial" if (_a1_ok or _a2_ok) else "Developing")
    _tier_color = COLORS["GreenLine"] if _tier == "Optimal" else (COLORS["OrangeLine"] if _tier == "Partial" else COLORS["TextMuted"])

    mo.Html(f"""
    <div class="lab-hud">
        <div><span class="hud-label">LAB</span> <span class="hud-value">Vol2 &middot; Lab 06</span></div>
        <div><span class="hud-label">CHAPTER</span> <span class="hud-value">v2_06 &middot; Fault Tolerance</span></div>
        <div><span class="hud-label">PART A</span> <span class="{'hud-active' if _a1_ok else 'hud-none'}">{"CORRECT" if _a1_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">PART B</span> <span class="{'hud-active' if _a2_ok else 'hud-none'}">{"CORRECT" if _a2_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">TIER</span> <span style="color:{_tier_color}; font-family:var(--font-mono);">{_tier.upper()}</span></div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
