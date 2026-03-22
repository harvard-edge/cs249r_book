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
            "../../../wheels/mlsysim-0.1.0-py3-none-any.whl", keep_going=False
        )
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog

    GPUS_PER_NODE = 8
    GPU_COST_HR = 3.0
    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return COLORS, LAB_CSS, apply_plotly_theme, go, ledger, math, mo, np, GPUS_PER_NODE, GPU_COST_HR, DecisionLog


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
                <span class="badge badge-warn">35&ndash;40 minutes &middot; 2 Parts</span>
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
                    Queuing theory basics from @sec-fleet-orchestration &middot;
                    Bandwidth hierarchy from @sec-collective-communication
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

    - **@sec-fleet-orchestration** -- GPU scheduling, heavy-tailed distributions
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
                    <strong>Read:</strong> @sec-fleet-orchestration for queuing theory, fragmentation analysis,
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
        "Synthesis": build_synthesis(),
    })
    tabs
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: LEDGER_HUD
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, DecisionLog):
    decision_input, decision_ui = DecisionLog()
    return (decision_input, decision_ui)


@app.cell(hide_code=True)
def _(COLORS, partA_prediction, partB_prediction, partA_reflection, partB_reflection,
      ledger, mo, decision_input, decision_ui):
    ledger.save(
        chapter="v2_07",
        design={
            "partA_prediction": partA_prediction.value or "no_selection",
            "partA_correct": partA_prediction.value == "B",
            "partB_prediction": partB_prediction.value or "no_selection",
            "partB_correct": partB_prediction.value == "B",
            "partA_reflection": partA_reflection.value or "no_selection",
            "partB_reflection": partB_reflection.value or "no_selection",
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
