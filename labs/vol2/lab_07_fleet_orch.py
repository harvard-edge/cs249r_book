import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-07: THE SCHEDULING TRAP
#
# Chapter: Fleet Orchestration (@sec-fleet-orchestration)
# Core Invariant: GPU scheduling is fundamentally harder than CPU scheduling
#                 because of heavy-tailed job distributions, multi-dimensional
#                 packing, topology sensitivity, and the impossibility of
#                 simultaneously optimizing utilization, fairness, and latency.
#
# 2-Part Structure (35-40 minutes):
#   Part A — The Queuing Wall (12-15 min)
#             Heavy-tailed ML workloads (C_s=3-5) make queue wait times 5x
#             worse than uniform workloads at 80% utilization.
#
#   Part B — Fragmentation + The Utilization Paradox (20-25 min)
#             77 idle GPUs cannot schedule a 64-GPU job due to fragmentation.
#             Maximizing utilization, fairness, and latency simultaneously
#             is impossible.
#
# Hardware Constants:
#   H100_COST_HR = 3.0    ($3/GPU-hour cloud pricing)
#   GPUS_PER_NODE = 8     (DGX H100)
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING (4 cells)
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

    # ── Hardware registry ─────────────────────────────────────────────────
    H100 = Hardware.Cloud.H100
    T4 = Hardware.Cloud.T4
    EDGE = Hardware.Edge.JetsonOrinNX
    GPUS_PER_NODE = 8
    GPU_COST_HR = 3.0
    EDGE_TFLOPS = EDGE.compute.peak_flops.m_as("TFLOPs/s")
    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return COLORS, LAB_CSS, apply_plotly_theme, go, ledger, math, mo, np, GPUS_PER_NODE, GPU_COST_HR, DecisionLog, Hardware, H100, T4, EDGE, EDGE_TFLOPS


# ─── CELL 1: HEADER ──────────────────────────────────────────────────────────
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
                    Vol 2 &middot; Lab 07 &middot; Fleet Orchestration
                </div>
                <div style="font-size: 2.0rem; font-weight: 800; color: #f1f5f9; line-height: 1.15; margin-bottom: 10px;">
                    The Scheduling Trap
                </div>
                <div style="font-size: 0.95rem; color: #94a3b8; max-width: 600px; line-height: 1.6;">
                    GPU scheduling is not CPU scheduling. Heavy-tailed job distributions,
                    multi-dimensional packing, and topology constraints create a world
                    where 80% utilization feels like gridlock and free GPUs cannot run
                    pending jobs. You cannot optimize everything simultaneously.
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;">
                <span class="badge badge-info">Pollaczek-Khinchine Formula</span>
                <span class="badge badge-info">GPU Fragmentation</span>
                <span class="badge badge-info">Utilization Paradox</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the queuing wall:</strong> apply the Pollaczek-Khinchine formula to show that ML heavy-tailed workloads (C_s=3-5) cause 5x worse queue wait at 80% utilization compared to uniform workloads.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose GPU fragmentation:</strong> explain why 77 idle GPUs scattered across 12 nodes cannot schedule a 64-GPU gang-scheduled job.</div>
                <div style="margin-bottom: 3px;">3. <strong>Recognize the utilization paradox:</strong> demonstrate that maximizing utilization, fairness, and latency simultaneously is impossible.</div>
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
                    Queuing theory basics from the Fleet Orchestration chapter &middot;
                    Bandwidth hierarchy from the Communication chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35-40 min</strong><br/>
                    Part A: ~12 min &middot; Part B: ~25 min
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
                "Your GPU cluster is 80% utilized and web engineers say that is comfortable.
                Why do ML researchers experience 25-minute queue waits, and why can a cluster
                with 77 free GPUs not schedule a 64-GPU job?"
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

    - **The Fleet Orchestration chapter** -- GPU scheduling, heavy-tailed distributions
    - The Queuing Theory section -- Pollaczek-Khinchine formula, coefficient of variation
    - The Fragmentation section -- Gang scheduling, multi-dimensional bin packing
    - The Scheduling Policy section -- Utilization vs fairness vs latency trade-off
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: WIDGET DEFINITIONS (separate cells for Marimo dataflow)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 4: Part A prediction + controls ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    partA_prediction = mo.ui.radio(
        options={
            "A) ~5 minutes -- similar to web service queuing": "A",
            "B) ~25 minutes -- 5x worse than uniform workloads": "B",
            "C) ~1 hour -- significant delay": "C",
            "D) ~2 minutes -- GPUs are fast": "D",
        },
        label="Your GPU cluster runs at 80% utilization. ML workloads have C_s=3 (heavy tail). What is the average queue wait?",
    )
    a1_utilization = mo.ui.slider(start=0.10, stop=0.99, value=0.80, step=0.01, label="Cluster utilization (rho)")
    a1_workload = mo.ui.dropdown(
        options={"Uniform (C_s=1)": 1.0, "ML Mixed (C_s=3)": 3.0, "Research (C_s=5)": 5.0},
        value="ML Mixed (C_s=3)",
        label="Workload type",
    )
    a1_service_min = mo.ui.slider(start=5, stop=120, value=30, label="Mean service time (minutes)")
    partA_reflection = mo.ui.radio(
        options={
            "A) Run the cluster at lower utilization -- 50-60% keeps wait times manageable": "A",
            "B) Preempt long-running jobs to serve short experiments faster": "B",
            "C) Add more GPUs until utilization drops below 70%": "C",
            "D) Use priority queues to separate long and short jobs": "D",
        },
        label="What is the most effective way to reduce queue wait for ML workloads?",
    )
    return (partA_prediction, a1_utilization, a1_workload, a1_service_min, partA_reflection)


# ─── CELL 5: Part B prediction + controls ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partA_prediction):
    mo.stop(partA_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partB_prediction = mo.ui.radio(
        options={
            "A) Yes -- a good scheduler can achieve all three simultaneously": "A",
            "B) No -- these goals are fundamentally in conflict; improving one degrades another": "B",
            "C) Yes, but only with preemption enabled": "C",
            "D) Yes, but only at 50% utilization": "D",
        },
        label="Can you achieve >90% utilization AND <10 min wait AND fair access across 5 teams?",
    )
    a2_w_throughput = mo.ui.slider(start=0, stop=100, value=40, step=5, label="Priority: Throughput (%)")
    a2_w_fairness = mo.ui.slider(start=0, stop=100, value=30, step=5, label="Priority: Fairness (%)")
    a2_w_latency = mo.ui.slider(start=0, stop=100, value=30, step=5, label="Priority: Low Latency (%)")
    a2_n_teams = mo.ui.slider(start=2, stop=10, value=5, step=1, label="Number of teams")
    partB_reflection = mo.ui.radio(
        options={
            "A) Implement a single 'optimal' scheduling algorithm that maximizes all metrics": "A",
            "B) Accept the trade-off and provide transparency -- let stakeholders choose which metric to sacrifice": "B",
            "C) Use AI to predict optimal scheduling decisions": "C",
            "D) Increase cluster size until all metrics are satisfied": "D",
        },
        label="Given the impossibility of simultaneous optimization, what is the best operational approach?",
    )
    return (partB_prediction, a2_w_throughput, a2_w_fairness, a2_w_latency, a2_n_teams, partB_reflection)


# ─── CELL 5b: Part C prediction + controls ────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partB_reflection):
    mo.stop(partB_reflection.value is None, mo.md("**Complete Part B reflection to unlock Part C.**"))

    partC_prediction = mo.ui.radio(
        options={
            "A) Zero cost -- preemption just moves jobs around": "A",
            "B) ~2 hours of GPU-time wasted per preemption (rework since last checkpoint)": "B",
            "C) Minimal -- just the context switch overhead": "C",
            "D) The preempted job fails permanently": "D",
        },
        label="A 64-GPU training job has been running for 2 hours since its last checkpoint. You preempt it to schedule a higher-priority job. What is the cost?",
    )
    c1_preempt_interval_h = mo.ui.slider(start=0.5, stop=8, value=2.0, step=0.5, label="Hours since last checkpoint")
    c1_job_gpus = mo.ui.slider(start=8, stop=128, value=64, step=8, label="Preempted job GPUs")
    c1_preemptions_day = mo.ui.slider(start=1, stop=20, value=5, step=1, label="Preemptions per day")
    partC_reflection = mo.ui.radio(
        options={
            "A) Never preempt -- let all jobs run to completion": "A",
            "B) Checkpoint before preemption to minimize rework, but accept the checkpoint write cost": "B",
            "C) Preempt freely -- rework cost is negligible": "C",
            "D) Only preempt jobs smaller than 8 GPUs": "D",
        },
        label="What is the correct preemption strategy?",
    )
    return (partC_prediction, c1_preempt_interval_h, c1_job_gpus, c1_preemptions_day, partC_reflection)


# ─── CELL 5c: Part D prediction + controls ────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partC_reflection):
    mo.stop(partC_reflection.value is None, mo.md("**Complete Part C reflection to unlock Part D.**"))

    partD_prediction = mo.ui.radio(
        options={
            "A) 30% -- same as single cluster": "A",
            "B) 10-15% -- heterogeneous packing fills gaps left by gang scheduling": "B",
            "C) 0% -- heterogeneous clusters eliminate fragmentation entirely": "C",
            "D) 50% -- mixing GPU types makes scheduling harder": "D",
        },
        label="You add a pool of T4 GPUs ($0.35/hr) for inference/small jobs alongside your H100 training cluster. What happens to the H100 fragmentation rate?",
    )
    d1_h100_count = mo.ui.slider(start=64, stop=512, value=256, step=8, label="H100 GPUs (training)")
    d1_t4_count = mo.ui.slider(start=32, stop=256, value=128, step=8, label="T4 GPUs (inference/small jobs)")
    d1_small_job_pct = mo.ui.slider(start=10, stop=60, value=30, step=5, label="Small job fraction (%)")
    partD_reflection = mo.ui.radio(
        options={
            "A) Use one homogeneous cluster type for simplicity": "A",
            "B) Use heterogeneous pools: route large training to H100s, inference/small jobs to T4s, reducing H100 fragmentation": "B",
            "C) Buy only T4s -- they are cheaper per GPU": "C",
            "D) Hardware type does not affect scheduling efficiency": "D",
        },
        label="What is the correct fleet composition strategy?",
    )
    return (partD_prediction, d1_h100_count, d1_t4_count, d1_small_job_pct, partD_reflection)


# ─── CELL 5d: DECISION LOG WIDGET ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(DecisionLog, mo, partD_reflection):
    decision_input, decision_ui = DecisionLog()
    return (decision_input, decision_ui)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: SINGLE TABS CELL (all build_part_X functions + mo.ui.tabs)
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    math,
    mo,
    np,
    GPU_COST_HR,
    partA_prediction,
    a1_utilization,
    a1_workload,
    a1_service_min,
    partA_reflection,
    partB_prediction,
    a2_w_throughput,
    a2_w_fairness,
    a2_w_latency,
    a2_n_teams,
    partB_reflection,
    partC_prediction,
    c1_preempt_interval_h,
    c1_job_gpus,
    c1_preemptions_day,
    partC_reflection,
    partD_prediction,
    d1_h100_count,
    d1_t4_count,
    d1_small_job_pct,
    partD_reflection,
):

    # ═════════════════════════════════════════════════════════════════════════
    # PART A: THE QUEUING WALL
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_a():
        items = []

        # ── Stakeholder message ──────────────────────────────────────────
        items.append(mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']}; background: {COLORS['BlueLL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['BlueLine']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Cluster Operations Manager
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "Our GPU cluster runs at 80% utilization. We used to run web services at 80%
            with no complaints. But ML researchers say they wait 25 minutes for a job to
            start. Our web service engineers say 80% is comfortable. Who is right?"
        </div>
    </div>
        """))

        # ── Concept framing ──────────────────────────────────────────────
        items.append(mo.md("""
    The Pollaczek-Khinchine (P-K) formula for average wait time in an M/G/1 queue:

    **W_q = (rho / (1 - rho)) * ((1 + C_s^2) / (2 * mu))**

    Where:
    - rho = utilization (0 to 1)
    - C_s = coefficient of variation (std_dev / mean) of service time
    - mu = service rate (1 / mean service time)

    For web services: C_s = 1 (roughly exponential, well-behaved)
    For ML workloads: C_s = 3-5 (heavy-tailed: rare month-long training runs
    coexist with thousands of 1-hour experiments)

    The amplification factor (1 + C_s^2) / 2:
    - C_s = 1: factor = 1.0 (baseline)
    - C_s = 3: factor = 5.0 (5x worse wait)
    - C_s = 5: factor = 13.0 (13x worse wait)

    At rho = 0.80, this turns a 5-minute web wait into a 25-minute ML wait.
        """))

        # ── Prediction ───────────────────────────────────────────────────
        items.append(partA_prediction)

        if partA_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part A instruments."), kind="warn"))
            return mo.vstack(items)

        # ── Controls ─────────────────────────────────────────────────────
        items.append(mo.md("### Queuing Wall Explorer"))
        items.append(mo.hstack([a1_utilization, a1_workload, a1_service_min], justify="center", gap=2))

        # ── Instruments ──────────────────────────────────────────────────
        _rho = a1_utilization.value
        _cs = a1_workload.value
        _service_min = a1_service_min.value
        _mu = 1.0 / _service_min  # jobs per minute

        # P-K formula: W_q = (rho/(1-rho)) * ((1+Cs^2)/(2*mu))
        _amplification = (1 + _cs ** 2) / 2
        _wait_min = (_rho / (1 - _rho)) * (_amplification / _mu) if _rho < 1.0 else float('inf')
        _wait_uniform = (_rho / (1 - _rho)) * (1.0 / _mu) if _rho < 1.0 else float('inf')
        _ratio = _amplification

        # ── Wait vs utilization curves ────────────────────────────────────
        _rho_range = np.linspace(0.1, 0.98, 100)
        _fig = go.Figure()
        for _csv, _label, _clr in [(1.0, "Uniform (C_s=1)", COLORS["GreenLine"]),
                                     (3.0, "ML Mixed (C_s=3)", COLORS["OrangeLine"]),
                                     (5.0, "Research (C_s=5)", COLORS["RedLine"])]:
            _amp = (1 + _csv ** 2) / 2
            _waits = [(_r / (1 - _r)) * (_amp / _mu) for _r in _rho_range]
            _fig.add_trace(go.Scatter(x=_rho_range * 100, y=_waits, mode="lines", name=_label,
                                       line=dict(color=_clr, width=2.5 if _csv == _cs else 1.5)))
        # Mark current
        _fig.add_trace(go.Scatter(x=[_rho * 100], y=[min(_wait_min, 500)], mode="markers",
                                   name="Current", marker=dict(size=14, color=COLORS["RedLine"], symbol="diamond")))
        _fig.add_hline(y=10, line=dict(color=COLORS["TextMuted"], width=1, dash="dot"),
                       annotation_text="10 min threshold", annotation_position="top right")
        _fig.update_layout(
            height=340,
            xaxis=dict(title="Cluster Utilization (%)", range=[10, 100]),
            yaxis=dict(title="Average Wait Time (minutes)", range=[0, min(max(_wait_min * 1.5, 60), 500)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=50, r=20),
        )
        apply_plotly_theme(_fig)

        _wait_color = COLORS["GreenLine"] if _wait_min < 10 else (COLORS["OrangeLine"] if _wait_min < 30 else COLORS["RedLine"])

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Pollaczek-Khinchine Queuing Model
            </div>
            <div>Utilization: rho = {_rho:.2f} &mdash; C_s = {_cs:.0f} &mdash; mean service = {_service_min} min</div>
            <div>Amplification factor = (1 + {_cs:.0f}^2) / 2 = <strong>{_amplification:.1f}x</strong></div>
            <div>W_q = ({_rho:.2f} / {1-_rho:.2f}) &times; ({_amplification:.1f} / {_mu:.4f}) = <strong style="color:{_wait_color};">{min(_wait_min, 999):.1f} min</strong></div>
            <div>Uniform wait at same rho: {min(_wait_uniform, 999):.1f} min &mdash; ML is <strong>{_ratio:.1f}x worse</strong></div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Wait Time</div>
                <div style="font-size:2rem; font-weight:800; color:{_wait_color}; font-family:monospace;">{min(_wait_min, 999):.0f}m</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">ML workload</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Amplification</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['OrangeLine']}; font-family:monospace;">{_ratio:.0f}x</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">vs uniform</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Utilization</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']}; font-family:monospace;">{_rho*100:.0f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">cluster</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # ── Reveal ───────────────────────────────────────────────────────
        if partA_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** At C_s=3, the amplification factor is (1+9)/2 = 5. This turns "
                "a 5-minute uniform wait into a 25-minute ML wait at 80% utilization. The "
                "heavy tail means rare but massive training jobs (months-long) coexist with "
                "thousands of 1-hour experiments, creating extreme variance."
            ), kind="success"))
        elif partA_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**That is the web service answer.** 5 minutes is correct for C_s=1 (uniform). "
                "But ML workloads have C_s=3-5, which amplifies wait by 5-13x. The heavy tail "
                "-- rare large jobs blocking many small ones -- is what makes ML scheduling "
                "fundamentally harder than web service scheduling."
            ), kind="warn"))
        elif partA_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Possible at C_s=5, but C_s=3 gives ~25 min.** At C_s=5, the amplification "
                "is 13x, and wait time at 80% utilization would be ~65 minutes. But typical "
                "ML mixed workloads have C_s=3, giving ~25 minutes."
            ), kind="warn"))
        elif partA_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Far too optimistic.** 2 minutes would require very low utilization (~30%) "
                "for ML workloads. At 80% with C_s=3, the P-K formula gives ~25 minutes."
            ), kind="warn"))

        # ── MathPeek ─────────────────────────────────────────────────────
        items.append(mo.accordion({
            "Governing equations -- Pollaczek-Khinchine formula": mo.md("""
        **Pollaczek-Khinchine Mean Wait (M/G/1)**

        ```
        W_q = (rho / (1 - rho)) * ((1 + C_s^2) / (2 * mu))
        ```

        - rho: server utilization (arrival_rate / service_rate)
        - C_s: coefficient of variation of service time = sigma / mu_service
        - mu: service rate = 1 / mean_service_time
        - The term (1 + C_s^2) / 2 is the "heavy-tail amplification factor"

        **Why ML Workloads Have Heavy Tails**

        A research cluster runs jobs spanning 6 orders of magnitude:
        - Quick experiments: 1-10 GPU-minutes
        - Hyperparameter sweeps: 10-100 GPU-hours
        - Full training runs: 1,000-100,000 GPU-hours

        This creates C_s = 3-5, amplifying queue wait by 5-13x vs uniform.
            """)
        }))

        # ── Reflection ───────────────────────────────────────────────────
        items.append(partA_reflection)

        if partA_reflection.value is not None:
            if partA_reflection.value == "A":
                items.append(mo.callout(mo.md(
                    "**Correct, but expensive.** The P-K formula shows that wait time is rho/(1-rho), "
                    "which diverges as rho -> 1. Dropping from 80% to 60% reduces the rho/(1-rho) "
                    "factor from 4 to 1.5 -- a 2.7x reduction. But you are paying for 20% more idle "
                    "GPUs. This is the utilization paradox: you must choose between GPU efficiency "
                    "and researcher productivity."
                ), kind="success"))
            elif partA_reflection.value == "B":
                items.append(mo.callout(mo.md(
                    "**Effective but costly.** Preempting a long training job means killing it and "
                    "restarting from the last checkpoint. If the job has been running for 2 hours "
                    "since the last checkpoint, you lose 2 hours x N GPUs of compute. Preemption "
                    "reduces wait time but increases total compute waste."
                ), kind="warn"))
            elif partA_reflection.value == "C":
                items.append(mo.callout(mo.md(
                    "**Correct in principle, expensive in practice.** Adding GPUs reduces utilization, "
                    "which reduces wait. But each GPU costs ~$3/hour. Adding 100 GPUs = $7,200/day. "
                    "The real question is whether researcher time saved justifies GPU cost -- and "
                    "that depends on researcher salary vs GPU cost."
                ), kind="warn"))
            elif partA_reflection.value == "D":
                items.append(mo.callout(mo.md(
                    "**Helpful but does not solve the fundamental problem.** Priority queues can "
                    "reduce wait for high-priority short jobs, but they increase wait for low-priority "
                    "jobs. The total wait across all jobs is still governed by the P-K formula. "
                    "You are redistributing pain, not eliminating it."
                ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B: FRAGMENTATION + THE UTILIZATION PARADOX
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_b():
        items = []

        # ── Stakeholder message ──────────────────────────────────────────
        items.append(mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['Cloud']}; background: {COLORS['BlueLL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['Cloud']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; ML Platform Team Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "The dashboard shows 30% free capacity -- 77 GPUs idle across the cluster.
            A researcher submitted a 64-GPU training job 30 minutes ago. It still has not
            started. The dashboard is green. The researcher is furious. What is happening?"
        </div>
    </div>
        """))

        # ── Concept framing ──────────────────────────────────────────────
        items.append(mo.md("""
    The 77 idle GPUs are scattered across 12 nodes in fragments of 1-4 GPUs each.
    Gang scheduling requires all 64 GPUs to be allocated **simultaneously** in
    **contiguous** 8-GPU nodes. With fragments scattered across the cluster,
    no contiguous block of 8 nodes (64 GPUs) exists.

    This is the **fragmentation tax**: physical capacity != effective capacity.

    The second challenge is the **utilization paradox**: you operate a shared cluster
    and must satisfy three stakeholders simultaneously:
    - **Operations**: maximize GPU utilization (> 90%)
    - **Researchers**: minimize queue wait time (< 10 min)
    - **Management**: ensure fair access across 5 teams

    These goals are **fundamentally in conflict**. Improving one necessarily degrades another.
        """))

        # ── Prediction ───────────────────────────────────────────────────
        items.append(partB_prediction)

        if partB_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part B instruments."), kind="warn"))
            return mo.vstack(items)

        # ── Controls ─────────────────────────────────────────────────────
        items.append(mo.md("### Scheduling Policy Simulator"))
        items.append(mo.hstack([
            mo.vstack([a2_w_throughput, a2_w_fairness]),
            mo.vstack([a2_w_latency, a2_n_teams]),
        ], justify="center", gap=2))

        # ── Instruments ──────────────────────────────────────────────────
        # Normalize weights
        _wt = a2_w_throughput.value
        _wf = a2_w_fairness.value
        _wl = a2_w_latency.value
        _total_w = max(_wt + _wf + _wl, 1)
        _nt = _wt / _total_w
        _nf = _wf / _total_w
        _nl = _wl / _total_w
        _teams = a2_n_teams.value

        # Simulate metrics based on policy weights
        # Throughput-biased: high utilization, long waits, unfair (big jobs favored)
        # Fairness-biased: equal shares, moderate util, long waits for productive teams
        # Latency-biased: short jobs first, low util, unfair to large jobs

        _utilization = 65 + 30 * _nt - 10 * _nl  # throughput boosts util, latency hurts it
        _utilization = max(40, min(98, _utilization))

        _avg_wait = 5 + 40 * _nt - 25 * _nl + 10 * _nf  # throughput hurts wait, latency helps
        _avg_wait = max(2, min(120, _avg_wait))

        _max_wait = _avg_wait * (2.5 + 3 * _nt)  # throughput creates extreme max waits
        _max_wait = max(5, min(480, _max_wait))

        # Jain's fairness index: 1.0 = perfectly fair, 1/N = maximally unfair
        _fairness = 0.5 + 0.45 * _nf - 0.3 * _nt - 0.1 * _nl
        _fairness = max(1.0 / _teams, min(1.0, _fairness))

        # Fragmentation: percentage of GPUs stranded
        _fragmentation = 15 + 20 * _nt - 10 * _nl  # throughput causes more fragmentation
        _fragmentation = max(5, min(45, _fragmentation))

        # Check if all metrics meet targets
        _util_ok = _utilization > 90
        _wait_ok = _avg_wait < 10
        _fair_ok = _fairness > 0.85
        _all_green = _util_ok and _wait_ok and _fair_ok

        # Colors
        _util_color = COLORS["GreenLine"] if _util_ok else (COLORS["OrangeLine"] if _utilization > 75 else COLORS["RedLine"])
        _wait_color = COLORS["GreenLine"] if _wait_ok else (COLORS["OrangeLine"] if _avg_wait < 30 else COLORS["RedLine"])
        _fair_color = COLORS["GreenLine"] if _fair_ok else (COLORS["OrangeLine"] if _fairness > 0.7 else COLORS["RedLine"])
        _frag_color = COLORS["GreenLine"] if _fragmentation < 15 else (COLORS["OrangeLine"] if _fragmentation < 30 else COLORS["RedLine"])

        # ── Radar chart ───────────────────────────────────────────────────
        _categories = ["Utilization", "Low Wait", "Fairness", "Low Fragmentation"]
        _values = [
            _utilization / 100,                          # normalize to 0-1
            max(0, 1 - _avg_wait / 60),                  # invert: lower wait = better
            _fairness,
            max(0, 1 - _fragmentation / 50),             # invert: lower frag = better
        ]
        _values.append(_values[0])  # close the polygon
        _categories.append(_categories[0])

        _fig = go.Figure()
        _fig.add_trace(go.Scatterpolar(
            r=_values, theta=_categories, fill="toself",
            line=dict(color=COLORS["BlueLine"], width=2),
            fillcolor="rgba(0,99,149,0.15)",
            name="Current policy",
        ))
        # Target overlay
        _fig.add_trace(go.Scatterpolar(
            r=[0.9, 0.83, 0.85, 0.7, 0.9], theta=_categories, fill="none",
            line=dict(color=COLORS["GreenLine"], width=1.5, dash="dash"),
            name="Target",
        ))
        _fig.update_layout(
            height=320,
            polar=dict(radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0])),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=30, b=60, l=60, r=60),
        )
        apply_plotly_theme(_fig)

        # ── Impossibility banner ──────────────────────────────────────────
        if not _all_green:
            _failing = []
            if not _util_ok:
                _failing.append(f"Utilization ({_utilization:.0f}% < 90%)")
            if not _wait_ok:
                _failing.append(f"Wait ({_avg_wait:.0f}m > 10m)")
            if not _fair_ok:
                _failing.append(f"Fairness ({_fairness:.2f} < 0.85)")
            _impossible_banner = f"""
        <div style="background:{COLORS['OrangeLL']}; border:1px solid {COLORS['OrangeLine']};
                    border-radius:10px; padding:14px 18px; margin:10px 0;">
            <div style="font-size:0.88rem; font-weight:800; color:{COLORS['OrangeLine']}; margin-bottom:4px;">
                TRADE-OFF ACTIVE &mdash; Not All Targets Met
            </div>
            <div style="font-size:0.85rem; color:#7c2d12; line-height:1.6;">
                Failing metrics: {', '.join(_failing)}<br>
                <strong>Adjust priority weights to explore the trade-off space.</strong>
                You will find that turning all four metrics green simultaneously is impossible.
            </div>
        </div>
            """
        else:
            _impossible_banner = f"""
        <div style="background:{COLORS['GreenLL']}; border:1px solid {COLORS['GreenLine']};
                    border-radius:10px; padding:14px 18px; margin:10px 0;">
            <div style="font-size:0.88rem; font-weight:800; color:{COLORS['GreenLine']};">
                All metrics green! But verify: is this sustainable, or did the model reach
                a fragile equilibrium? Try increasing teams or throughput priority.
            </div>
        </div>
            """

        items.append(mo.Html(f"""
        {_impossible_banner}
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Scheduling Policy Trade-off
            </div>
            <div>Weights: Throughput={_wt}% Fairness={_wf}% Latency={_wl}% (normalized: {_nt:.2f}/{_nf:.2f}/{_nl:.2f})</div>
            <div>Teams: {_teams}</div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:150px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Utilization</div>
                <div style="font-size:2rem; font-weight:800; color:{_util_color}; font-family:monospace;">{_utilization:.0f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">target: &gt;90%</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:150px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Avg Wait</div>
                <div style="font-size:2rem; font-weight:800; color:{_wait_color}; font-family:monospace;">{_avg_wait:.0f}m</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">target: &lt;10m</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:150px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Fairness</div>
                <div style="font-size:2rem; font-weight:800; color:{_fair_color}; font-family:monospace;">{_fairness:.2f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">target: &gt;0.85</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:150px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Fragmentation</div>
                <div style="font-size:2rem; font-weight:800; color:{_frag_color}; font-family:monospace;">{_fragmentation:.0f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">stranded GPUs</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # ── Reveal ───────────────────────────────────────────────────────
        if partB_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** High utilization requires keeping GPUs busy with large jobs, "
                "which blocks queues for small jobs (high wait). Low latency requires serving "
                "small jobs first, which fragments the cluster (low utilization). Fairness "
                "requires equal access, which may starve the most productive teams. "
                "Every scheduling policy is a point in this three-dimensional trade-off space."
            ), kind="success"))
        elif partB_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**This is the trap.** Students from OS courses believe scheduling is solved. "
                "But ML scheduling has unique properties: gang scheduling (all-or-nothing "
                "allocation), topology sensitivity (NVLink boundaries), and heavy-tailed "
                "durations. These create conflicts that no single policy resolves."
            ), kind="warn"))
        elif partB_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Preemption helps but introduces new costs.** Preempting a job that has "
                "been running for 2 hours since its last checkpoint loses 2 hours of compute. "
                "Preemption reduces wait for short jobs but increases total waste and hurts "
                "long-job throughput. It shifts the trade-off, not eliminates it."
            ), kind="warn"))
        elif partB_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**50% utilization eases wait times but fails the utilization target.** "
                "At $3/GPU-hour, 50% utilization on a 256-GPU cluster wastes "
                "$3 x 128 x 24 = $9,216/day in idle GPUs. Operations will not accept this."
            ), kind="warn"))

        # ── MathPeek ─────────────────────────────────────────────────────
        items.append(mo.accordion({
            "Governing equations -- fragmentation and the scheduling impossibility": mo.md("""
        **Fragmentation Ratio**

        ```
        F = stranded_GPUs / total_GPUs
        ```

        - "Stranded" GPUs are idle but not schedulable due to topology constraints
        - Gang scheduling requires contiguous 8-GPU nodes
        - Fragments of 1-4 GPUs per node cannot form complete blocks
        - F can exceed 30% even at 70% physical utilization

        **Effective Capacity**

        ```
        C_effective = (1 - F) * C_physical
        ```

        - A 256-GPU cluster with F=30% has effective capacity of only 179 GPUs
        - This explains why 77 "free" GPUs cannot schedule a 64-GPU job

        **Jain's Fairness Index**

        ```
        J = (sum(x_i))^2 / (N * sum(x_i^2))
        ```

        - J = 1.0: perfectly fair (all teams get equal share)
        - J = 1/N: maximally unfair (one team gets everything)
        - Throughput-optimal policies push J toward 1/N
            """)
        }))

        # ── Reflection ───────────────────────────────────────────────────
        items.append(partB_reflection)

        if partB_reflection.value is not None:
            if partB_reflection.value == "B":
                items.append(mo.callout(mo.md(
                    "**Correct.** The impossibility is mathematical, not engineering. The best "
                    "approach is to make the trade-off explicit: provide dashboards showing all "
                    "three metrics, let stakeholders decide which to sacrifice, and implement "
                    "policy knobs (preemption thresholds, fairshare weights, backfill aggressiveness) "
                    "that map to understandable trade-offs."
                ), kind="success"))
            else:
                items.append(mo.callout(mo.md(
                    "**The impossibility is fundamental.** No algorithm -- AI or otherwise -- can "
                    "simultaneously maximize utilization, minimize wait, and ensure fairness. "
                    "The best approach is transparency: make the trade-off visible and let "
                    "stakeholders choose which metric to sacrifice."
                ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C: PREEMPTION COST
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_c():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['OrangeLine']}; background: {COLORS['OrangeLL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['OrangeLine']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Scheduling Policy Engineer
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We want to implement preemption to reduce queue wait times. The idea is simple:
            when a high-priority job arrives, we preempt a lower-priority large job. But our
            training team says this wastes compute. How much does preemption actually cost?"
        </div>
    </div>
        """))

        # Concept framing
        items.append(mo.md("""
    Preemption kills a running job and restarts it later. The cost is **rework**:
    all compute since the last checkpoint is lost.

    For a 64-GPU job running 2 hours since its last checkpoint:
    - Lost compute = 64 GPUs x 2 hours = **128 GPU-hours = $384** (at $3/GPU-hr)

    If the scheduler preempts 5 jobs per day with similar rework:
    - Daily waste = 5 x $384 = **$1,920/day = $57,600/month**

    The preemption paradox: preempting reduces **wait time** for new jobs
    but increases **total compute waste**. The net benefit depends on whether
    saved researcher time (from shorter queues) exceeds the rework cost.

    The fix: **checkpoint-before-preempt** reduces rework to just the checkpoint
    write time. But this adds 2-40 minutes of delay before the slot is freed.
        """))

        # Prediction lock
        items.append(partC_prediction)

        if partC_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part C instruments."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.md("### Preemption Cost Calculator"))
        items.append(mo.hstack([c1_preempt_interval_h, c1_job_gpus, c1_preemptions_day], justify="center", gap=2))

        # Physics
        _hours_since_ckpt = c1_preempt_interval_h.value
        _gpus = c1_job_gpus.value
        _preemptions_day = c1_preemptions_day.value

        _gpu_hours_lost = _gpus * _hours_since_ckpt
        _cost_per_preempt = _gpu_hours_lost * GPU_COST_HR
        _daily_waste = _cost_per_preempt * _preemptions_day
        _monthly_waste = _daily_waste * 30

        # Compare: preemption savings (reduced wait time -> researcher productivity)
        # Assume each preemption saves 30 minutes of researcher wait x downstream team
        _researcher_hr_saved = _preemptions_day * 0.5  # 30 min per preemption
        _researcher_value = _researcher_hr_saved * 150  # $150/hr loaded cost
        _net_daily = _researcher_value - _daily_waste

        # Chart: cost vs rework hours
        _hours_range = np.linspace(0.25, 8, 20)
        _costs = [_gpus * h * GPU_COST_HR for h in _hours_range]
        _daily_costs = [c * _preemptions_day for c in _costs]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_hours_range, y=_daily_costs, mode="lines+markers",
            line=dict(color=COLORS["RedLine"], width=2.5),
            name="Daily preemption waste ($)",
        ))
        _fig.add_hline(y=_researcher_value, line=dict(color=COLORS["GreenLine"], width=2, dash="dash"),
                       annotation_text=f"Researcher value saved: ${_researcher_value:.0f}/day",
                       annotation_position="top right")
        _fig.add_trace(go.Scatter(
            x=[_hours_since_ckpt], y=[_daily_waste],
            mode="markers", marker=dict(size=14, color=COLORS["OrangeLine"], symbol="diamond"),
            name="Current config",
        ))
        _fig.update_layout(
            height=300,
            xaxis=dict(title="Hours Since Last Checkpoint"),
            yaxis=dict(title="Daily Cost ($)", tickformat="$,.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=50, l=70, r=20),
        )
        apply_plotly_theme(_fig)

        _waste_color = COLORS["GreenLine"] if _daily_waste < _researcher_value else COLORS["RedLine"]
        _net_color = COLORS["GreenLine"] if _net_daily > 0 else COLORS["RedLine"]

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Preemption Cost Model
            </div>
            <div>Rework per preemption: {_gpus} GPUs &times; {_hours_since_ckpt}h = <strong>{_gpu_hours_lost:.0f} GPU-hours = ${_cost_per_preempt:,.0f}</strong></div>
            <div>Daily waste: {_preemptions_day} preemptions &times; ${_cost_per_preempt:,.0f} = <strong style="color:{_waste_color};">${_daily_waste:,.0f}/day</strong></div>
            <div>Researcher value saved: <strong style="color:{COLORS['GreenLine']};">${_researcher_value:,.0f}/day</strong></div>
            <div>Net daily: <strong style="color:{_net_color};">${_net_daily:+,.0f}/day</strong></div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Per Preempt</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['RedLine']}; font-family:monospace;">${_cost_per_preempt:,.0f}</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Daily Waste</div>
                <div style="font-size:2rem; font-weight:800; color:{_waste_color}; font-family:monospace;">${_daily_waste/1000:.1f}K</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Net/Day</div>
                <div style="font-size:2rem; font-weight:800; color:{_net_color}; font-family:monospace;">${_net_daily:+,.0f}</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # Prediction reveal
        if partC_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** The preempted job loses 64 GPUs x 2 hours = 128 GPU-hours of compute. "
                "At $3/GPU-hour, that is $384 per preemption. This is pure waste -- the compute "
                "produced no useful training progress. Checkpoint-before-preempt reduces this to "
                "just the checkpoint write time (minutes), but delays freeing the slot."
            ), kind="success"))
        elif partC_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Wrong.** Preemption kills the running process. All compute since the last "
                "checkpoint is lost and must be redone. 'Moving jobs around' implies migration, "
                "which is not preemption."
            ), kind="warn"))
        elif partC_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Underestimates dramatically.** Context switch overhead is milliseconds. The real "
                "cost is the lost compute: hours of GPU-time that produced no checkpoint."
            ), kind="warn"))
        elif partC_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Not true.** Preempted jobs restart from their last checkpoint. They do not fail "
                "permanently, but they lose all progress since the last save point."
            ), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Governing equations -- preemption cost": mo.md("""
        **Rework Cost**

        ```
        C_rework = N_gpus * T_since_checkpoint * cost_per_GPU_hour
        ```

        **Daily Preemption Waste**

        ```
        W_daily = n_preemptions * C_rework
        ```

        **Checkpoint-Before-Preempt**

        ```
        C_rework_reduced = N_gpus * T_checkpoint_write * cost_per_GPU_hour
        ```

        With async checkpointing: T_checkpoint_write = ~1 second
        Rework drops from hours to seconds of GPU-time.
            """)
        }))

        # Reflection
        items.append(partC_reflection)
        if partC_reflection.value is not None:
            if partC_reflection.value == "B":
                items.append(mo.callout(mo.md(
                    "**Correct.** Checkpoint-before-preempt triggers a checkpoint write before killing "
                    "the job, reducing rework to near-zero. The trade-off: the preempted slot is not "
                    "freed until the checkpoint completes (2-40 minutes depending on storage). "
                    "With async checkpointing, this delay drops to under 1 second."
                ), kind="success"))
            else:
                items.append(mo.callout(mo.md(
                    "**Not the best strategy.** The optimal approach is checkpoint-before-preempt: "
                    "trigger a checkpoint write, then kill the job. This minimizes rework while "
                    "still enabling preemption-based scheduling."
                ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D: HETEROGENEOUS FLEET COMPOSITION
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_d():
        items = []

        # Stakeholder message
        items.append(mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['GreenLine']}; background: {COLORS['GreenLL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['GreenLine']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Fleet Capacity Planner
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "Our H100 cluster runs at 80% utilization but 30% of GPUs are stranded by fragmentation.
            Meanwhile, 30% of our workloads are small inference jobs and experiments that do not
            need H100s. What if we added a pool of cheaper T4 GPUs ($0.35/hr vs $3/hr) to
            offload small jobs and reduce H100 fragmentation?"
        </div>
    </div>
        """))

        # Concept framing
        items.append(mo.md("""
    A homogeneous H100 cluster forces ALL jobs -- from 1-GPU inference tests to 256-GPU
    training runs -- onto the same expensive hardware. Small jobs create fragmentation
    by occupying partial nodes and blocking gang-scheduled large jobs.

    A **heterogeneous fleet** routes jobs to the cheapest hardware that meets their requirements:
    - **H100 pool**: Large training jobs (64+ GPUs, NVLink required)
    - **T4 pool**: Inference, small experiments, hyperparameter sweeps (1-8 GPUs)

    Benefits:
    1. Small jobs leave H100 nodes, reducing fragmentation
    2. Small jobs run on 8.6x cheaper hardware (T4: $0.35/hr vs H100: $3/hr)
    3. H100 fragmentation drops because nodes have fewer partial allocations
        """))

        # Prediction lock
        items.append(partD_prediction)

        if partD_prediction.value is None:
            items.append(mo.callout(mo.md("Select your prediction above to unlock the Part D instruments."), kind="warn"))
            return mo.vstack(items)

        # Controls
        items.append(mo.md("### Heterogeneous Fleet Simulator"))
        items.append(mo.hstack([
            mo.vstack([d1_h100_count, d1_t4_count]),
            mo.vstack([d1_small_job_pct]),
        ], justify="center", gap=2))

        # Physics
        _h100s = d1_h100_count.value
        _t4s = d1_t4_count.value
        _small_pct = d1_small_job_pct.value / 100

        # Baseline: all on H100
        _baseline_frag = 30.0  # 30% fragmentation baseline
        _baseline_cost_day = _h100s * GPU_COST_HR * 24

        # Heterogeneous: small jobs offloaded to T4
        _h100_frag_reduction = _small_pct * 0.7  # 70% of small-job fragmentation removed
        _hetero_frag = max(5, _baseline_frag * (1 - _h100_frag_reduction))
        _t4_cost_hr = 0.35
        _hetero_cost_day = (_h100s * GPU_COST_HR + _t4s * _t4_cost_hr) * 24
        _savings_day = _baseline_cost_day - _hetero_cost_day + (_baseline_frag - _hetero_frag) / 100 * _h100s * GPU_COST_HR * 24

        # Effective capacity comparison
        _baseline_effective = _h100s * (1 - _baseline_frag / 100)
        _hetero_h100_effective = _h100s * (1 - _hetero_frag / 100)
        _hetero_total_effective = _hetero_h100_effective + _t4s * 0.9  # T4s have less fragmentation

        # Chart: fragmentation comparison
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=["Homogeneous H100"], y=[_baseline_frag],
            marker_color=COLORS["RedLine"], text=[f"{_baseline_frag:.0f}%"], textposition="auto",
            name="H100 only",
        ))
        _fig.add_trace(go.Bar(
            x=["Heterogeneous (H100+T4)"], y=[_hetero_frag],
            marker_color=COLORS["GreenLine"], text=[f"{_hetero_frag:.1f}%"], textposition="auto",
            name="H100+T4",
        ))
        _fig.update_layout(
            height=280,
            yaxis=dict(title="H100 Fragmentation (%)", range=[0, 50]),
            margin=dict(t=30, b=50, l=50, r=20),
            showlegend=False,
        )
        apply_plotly_theme(_fig)

        _frag_color = COLORS["GreenLine"] if _hetero_frag < 15 else (COLORS["OrangeLine"] if _hetero_frag < 25 else COLORS["RedLine"])

        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:12px; padding:16px 20px; margin:8px 0; font-family:monospace;
                    font-size:0.83rem; line-height:1.8;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; font-family:sans-serif;">
                Physics &mdash; Heterogeneous Fleet Analysis
            </div>
            <div>H100 pool: {_h100s} GPUs &mdash; T4 pool: {_t4s} GPUs</div>
            <div>Small job fraction offloaded: {_small_pct*100:.0f}%</div>
            <div>H100 fragmentation: <strong>{_baseline_frag:.0f}%</strong> (homogeneous) &rarr; <strong style="color:{_frag_color};">{_hetero_frag:.1f}%</strong> (heterogeneous)</div>
            <div>Effective H100 capacity: {_baseline_effective:.0f} &rarr; <strong>{_hetero_h100_effective:.0f}</strong> GPUs</div>
            <div>Daily fleet cost: ${_baseline_cost_day:,.0f} (homo) vs ${_hetero_cost_day:,.0f} (hetero)</div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:16px; justify-content:center; margin:8px 0; flex-wrap:wrap;">
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">H100 Frag</div>
                <div style="font-size:2rem; font-weight:800; color:{_frag_color}; font-family:monospace;">{_hetero_frag:.0f}%</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">was {_baseline_frag:.0f}%</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">T4 Cost</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['GreenLine']}; font-family:monospace;">${_t4_cost_hr}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">vs $3/hr H100</div>
            </div>
            <div style="padding:18px 24px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        width:160px; text-align:center; background:white;">
                <div style="color:{COLORS['TextMuted']}; font-size:0.82rem; font-weight:600; text-transform:uppercase;">Effective Cap</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']}; font-family:monospace;">{_hetero_h100_effective:.0f}</div>
                <div style="font-size:0.72rem; color:{COLORS['TextMuted']};">H100 effective GPUs</div>
            </div>
        </div>
        """))

        items.append(mo.ui.plotly(_fig))

        # Prediction reveal
        if partD_prediction.value == "B":
            items.append(mo.callout(mo.md(
                "**Correct.** Offloading small jobs to T4s reduces H100 fragmentation from ~30% "
                "to ~10-15%. Small jobs no longer occupy partial H100 nodes, leaving contiguous "
                "blocks available for large gang-scheduled training runs. The T4 pool runs small "
                "jobs at 8.6x lower cost per GPU-hour."
            ), kind="success"))
        elif partD_prediction.value == "A":
            items.append(mo.callout(mo.md(
                "**Not if you offload small jobs.** Fragmentation on H100s is caused partly by "
                "small jobs occupying partial nodes. Removing them to a T4 pool frees those "
                "partial allocations, reducing H100 fragmentation significantly."
            ), kind="warn"))
        elif partD_prediction.value == "C":
            items.append(mo.callout(mo.md(
                "**Too optimistic.** Some fragmentation remains because training jobs themselves "
                "create partial-node allocations (e.g., 48-GPU jobs leave 2 GPUs stranded per "
                "6-node block). But removing small-job fragmentation is the largest single lever."
            ), kind="warn"))
        elif partD_prediction.value == "D":
            items.append(mo.callout(mo.md(
                "**Opposite of the truth.** Heterogeneous scheduling is more complex, but it "
                "reduces fragmentation because each job class runs on its natural hardware. "
                "Mixing all workloads onto one tier is what causes high fragmentation."
            ), kind="warn"))

        # MathPeek
        items.append(mo.accordion({
            "Governing equations -- heterogeneous fleet": mo.md("""
        **Fragmentation Reduction**

        ```
        F_hetero = F_baseline * (1 - small_job_fraction * offload_efficiency)
        ```

        - offload_efficiency = ~70% (not all small jobs can be offloaded)
        - 30% small jobs offloaded at 70% efficiency: F reduces by 21%

        **Cost Comparison**

        ```
        C_homo = N_h100 * $3/hr * 24
        C_hetero = N_h100 * $3/hr * 24 + N_t4 * $0.35/hr * 24
        ```

        The T4 pool costs 8.6x less per GPU-hour. If it handles 30% of
        workloads, the total fleet cost may decrease even though you added GPUs.

        **Hardware-Workload Matching**

        | Workload | Best Tier | Why |
        |----------|-----------|-----|
        | 64+ GPU training | H100 | NVLink, high bandwidth |
        | 1-8 GPU inference | T4 | INT8 inference, low cost |
        | HP sweeps | T4 | Short, parallelizable |
            """)
        }))

        # Reflection
        items.append(partD_reflection)
        if partD_reflection.value is not None:
            if partD_reflection.value == "B":
                items.append(mo.callout(mo.md(
                    "**Correct.** Heterogeneous pools route each workload to its cheapest viable "
                    "hardware. Large training needs H100 (NVLink, high bandwidth). Inference and "
                    "small experiments need only T4 (INT8, low cost). This reduces H100 fragmentation "
                    "AND total fleet cost simultaneously."
                ), kind="success"))
            else:
                items.append(mo.callout(mo.md(
                    "**Not the optimal strategy.** Heterogeneous pools with workload-aware routing "
                    "reduce fragmentation on expensive hardware and lower total cost by running "
                    "small jobs on cheaper GPUs."
                ), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════════

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
                    <strong>1. The heavy-tail amplification factor (1+C_s^2)/2 makes ML queue wait 5x worse than web services.</strong>
                    At 80% utilization with C_s=3, average wait is ~25 minutes vs 5 minutes for uniform workloads.
                    The rare but massive training jobs create extreme variance that the P-K formula captures precisely.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Physical capacity != effective capacity due to fragmentation.</strong>
                    Gang scheduling requires contiguous 8-GPU blocks. Fragments of 1-4 GPUs per node
                    create 30%+ stranded capacity. 77 free GPUs scattered across nodes cannot schedule
                    a 64-GPU job because no contiguous block exists.
                </div>
                <div>
                    <strong>3. Utilization, fairness, and latency form an impossible triangle.</strong>
                    High utilization requires large jobs (high wait). Low latency requires small jobs first
                    (low utilization). Fairness requires equal shares (reduced throughput for productive teams).
                    Every scheduling policy is a trade-off point, never an optimum.
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
                    <strong>Lab V2-08: The Inference Economy</strong> &mdash; You scheduled training
                    jobs on your cluster. Now discover that serving those trained models costs more
                    than training them -- and the KV cache memory wall, not compute, determines
                    how many users you can serve.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> the Fleet Orchestration chapter for queuing theory, fragmentation analysis,
                    and scheduling policy trade-offs.<br/>
                    <strong>Build:</strong> TinyTorch scheduler module &mdash; implement a basic
                    gang scheduler with backfill in <code>tinytorch/src/scheduler/</code>.
                </div>
            </div>
        </div>
        """))

        items.append(mo.accordion({
            "Self-Assessment": mo.md("""
1. At 80% utilization, how much worse is ML queue wait (C_s=3) vs web service wait (C_s=1)?
2. Why can 77 idle GPUs scattered across 12 nodes not schedule a 64-GPU gang-scheduled job?
3. Why is it impossible to simultaneously maximize utilization AND minimize wait AND ensure fairness?

*If you cannot answer all three from memory, revisit Parts A and B.*
""")
        }))

        return mo.vstack(items)

    # ── Tab composition ──────────────────────────────────────────────────
    tabs = mo.ui.tabs({
        "Part A -- The Queuing Wall": build_part_a(),
        "Part B -- Fragmentation & Utilization Paradox": build_part_b(),
        "Part C -- Preemption Cost": build_part_c(),
        "Part D -- Heterogeneous Fleet": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: LEDGER_HUD
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(COLORS, partA_prediction, partB_prediction, partC_prediction, partD_prediction,
      partA_reflection, partB_reflection, partC_reflection, partD_reflection,
      ledger, mo, decision_input, decision_ui):
    ledger.save(
        chapter="v2_07",
        design={
            "partA_prediction": partA_prediction.value or "no_selection",
            "partA_correct": partA_prediction.value == "B",
            "partB_prediction": partB_prediction.value or "no_selection",
            "partB_correct": partB_prediction.value == "B",
            "partC_prediction": partC_prediction.value or "no_selection",
            "partC_correct": partC_prediction.value == "B",
            "partD_prediction": partD_prediction.value or "no_selection",
            "partD_correct": partD_prediction.value == "B",
            "partA_reflection": partA_reflection.value or "no_selection",
            "partB_reflection": partB_reflection.value or "no_selection",
            "partC_reflection": partC_reflection.value or "no_selection",
            "partD_reflection": partD_reflection.value or "no_selection",
            "student_justification": str(decision_input.value),
        },
    )

    _a1_ok = partA_prediction.value == "B"
    _a2_ok = partB_prediction.value == "B"
    _tier = "Optimal" if (_a1_ok and _a2_ok) else ("Partial" if (_a1_ok or _a2_ok) else "Developing")
    _tier_color = COLORS["GreenLine"] if _tier == "Optimal" else (COLORS["OrangeLine"] if _tier == "Partial" else COLORS["TextMuted"])

    decision_ui
    mo.Html(f"""
    <div class="lab-hud">
        <div><span class="hud-label">LAB</span> <span class="hud-value">Vol2 &middot; Lab 07</span></div>
        <div><span class="hud-label">CHAPTER</span> <span class="hud-value">v2_07 &middot; Fleet Orchestration</span></div>
        <div><span class="hud-label">PART A</span> <span class="{'hud-active' if _a1_ok else 'hud-none'}">{"CORRECT" if _a1_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">PART B</span> <span class="{'hud-active' if _a2_ok else 'hud-none'}">{"CORRECT" if _a2_ok else "REVIEW"}</span></div>
        <div><span class="hud-label">TIER</span> <span style="color:{_tier_color}; font-family:var(--font-mono);">{_tier.upper()}</span></div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
