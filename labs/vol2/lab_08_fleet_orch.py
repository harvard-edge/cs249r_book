import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 08: THE UTILIZATION TRAP
#
# Chapter: Fleet Orchestration (@sec-fleet-orchestration)
# Core Invariant: Utilization vs queue latency (Little's Law for job schedulers)
#   L = λW  — average queue depth = arrival rate × average wait time.
#   At 95% cluster utilization, M/M/1 queuing theory predicts W = 20× mean job
#   runtime. The "utilization trap": operators chase high utilization while
#   users complain about queue waits. The fix is not more hardware — it is
#   understanding where the knee of the utilization-wait curve falls (~80-85%).
#
# 2 Contexts: FIFO scheduling (default Slurm behavior) vs
#             Priority scheduling with backfill (advanced Slurm policy)
#
# Act I  (12–15 min): The Utilization Trap
#   Wrong prior: "95% utilization means the cluster is almost always computing"
#   Reality: At 95% utilization, M/M/1 queue theory predicts wait = 20×
#   mean job duration. A 1-hour test job waits 20 hours. This is not a bug —
#   it is the mathematics of queuing theory.
#   Prediction: which answer explains the 4-hour wait for 1-hour jobs?
#
# Act II (20–25 min): Scheduling Policy Design
#   Wrong prior: "priority queues mean interactive jobs always go first"
#   Reality: Simple priority without backfill still creates gaps when a large
#   job reserves nodes it cannot yet start. Backfill fills those gaps with
#   short jobs — higher utilization AND lower interactive wait.
#   Failure state: interactive wait > 5 min → danger callout
#   Starvation state: training wait > 48h → warn callout
#
# Hardware Constants:
#   H100_BW_GBS         = 3350   H100 SXM5 HBM3e, NVIDIA spec
#   H100_TFLOPS_FP16    = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
#   H100_RAM_GB         = 80     H100 SXM5 HBM3e capacity, NVIDIA spec
#   CLUSTER_SIZE_GPUS   = 1024   reference cluster for examples
#   SLURM_SCHED_LAT_MS  = 100    Slurm scheduling decision latency, Slurm docs
#
# Design Ledger: chapter="v2_08"
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible) ─────────────────────────
@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np
    import math

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    ledger = DesignLedger()

    # ── Hardware constants (all sourced from fleet_orchestration.qmd) ─────────

    # H100 SXM5 HBM3e bandwidth — NVIDIA datasheet, fleet_orchestration.qmd
    H100_BW_GBS         = 3350   # GB/s HBM3e memory bandwidth
    # H100 FP16 Tensor Core peak — NVIDIA H100 spec sheet
    H100_TFLOPS_FP16    = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    # H100 HBM3e capacity — NVIDIA H100 SXM5 datasheet
    H100_RAM_GB         = 80     # GB HBM3e
    # Reference cluster scale — fleet_orchestration.qmd FleetSetup.cluster_size
    CLUSTER_SIZE_GPUS   = 1024   # GPUs in reference Slurm cluster for this lab
    # Slurm scheduling decision latency — Slurm docs (typical Backfill cycle)
    SLURM_SCHED_LAT_MS  = 100    # ms per scheduling decision cycle

    # ── Queuing theory physics constants ──────────────────────────────────────
    # M/M/1 queue: W = service_time / (1 - rho) where rho = utilization
    # At rho = 0.95: W = service_time / 0.05 = 20 * service_time
    # At rho = 0.80: W = service_time / 0.20 = 5 * service_time (more tractable)
    # At rho = 0.75: W = service_time / 0.25 = 4 * service_time
    # Sources: Little (1961), fleet_orchestration.qmd Purpose section

    # ── Job class definitions (fleet_orchestration.qmd, Act II scenario) ──────
    # Interactive experiments: 8 GPUs, 30 min, 50 jobs/day
    INTERACTIVE_GPUS    = 8      # GPUs per interactive experiment
    INTERACTIVE_DUR_H   = 0.5    # hours (30 minutes)
    INTERACTIVE_JOBS_DAY = 50    # jobs per day
    # Model training: 64–512 GPUs, 12–24 hours, 10 jobs/day
    TRAINING_GPUS_MIN   = 64     # minimum GPUs per training job
    TRAINING_GPUS_MAX   = 512    # maximum GPUs per training job
    TRAINING_DUR_H      = 18     # hours (midpoint of 12–24 range)
    TRAINING_JOBS_DAY   = 10     # jobs per day
    # Hyperparameter search: 8–32 GPUs, 2 hours, 20 jobs/day
    SEARCH_GPUS_MIN     = 8      # minimum GPUs per search job
    SEARCH_GPUS_MAX     = 32     # maximum GPUs per search job
    SEARCH_DUR_H        = 2.0    # hours
    SEARCH_JOBS_DAY     = 20     # jobs per day

    # ── SLA target (Act II failure state) ─────────────────────────────────────
    INTERACTIVE_SLA_MIN = 5.0    # minutes — interactive job wait SLA

    return (
        mo, go, np, math,
        ledger, COLORS, LAB_CSS, apply_plotly_theme,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB,
        CLUSTER_SIZE_GPUS, SLURM_SCHED_LAT_MS,
        INTERACTIVE_GPUS, INTERACTIVE_DUR_H, INTERACTIVE_JOBS_DAY,
        TRAINING_GPUS_MIN, TRAINING_GPUS_MAX, TRAINING_DUR_H, TRAINING_JOBS_DAY,
        SEARCH_GPUS_MIN, SEARCH_GPUS_MAX, SEARCH_DUR_H, SEARCH_JOBS_DAY,
        INTERACTIVE_SLA_MIN,
    )


# ─── CELL 1: HEADER ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _fifo_c   = COLORS["BlueLine"]
    _prio_c   = COLORS["OrangeLine"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1a2744 50%, #0f172a 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 08
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Utilization Trap
            </h1>
            <p style="margin: 0 0 22px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 680px; line-height: 1.65;">
                Your 1024-GPU cluster runs at 95% utilization. Users complain their
                1-hour test jobs wait 4 hours in the queue. You added more GPUs —
                waits barely changed. Little&rsquo;s Law explains why utilization,
                not capacity, is the binding constraint.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 22px;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    2 Acts &middot; 35&ndash;40 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    Chapter 8: Fleet Orchestration
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Read @sec-fleet-orchestration first
                </span>
            </div>
            <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 12px 18px; min-width: 165px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: #64748b;
                                text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                        Core Invariant
                    </div>
                    <div style="font-size: 0.93rem; color: #e2e8f0; font-weight: 600;">
                        L = &lambda;W (Little&rsquo;s Law)
                    </div>
                </div>
                <div style="background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
                            border-radius: 10px; padding: 12px 18px; min-width: 165px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: #6366f1;
                                text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                        Context A
                    </div>
                    <div style="font-size: 0.93rem; color: #a5b4fc; font-weight: 600;">
                        FIFO Scheduling
                    </div>
                </div>
                <div style="background: rgba(204,85,0,0.1); border: 1px solid rgba(204,85,0,0.25);
                            border-radius: 10px; padding: 12px 18px; min-width: 165px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {_prio_c};
                                text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                        Context B
                    </div>
                    <div style="font-size: 0.93rem; color: #fb923c; font-weight: 600;">
                        Priority + Backfill
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 12px 18px; min-width: 165px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: #64748b;
                                text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                        Cluster Size
                    </div>
                    <div style="font-size: 0.93rem; color: #e2e8f0; font-weight: 600;">
                        1,024 GPUs (H100)
                    </div>
                </div>
            </div>
        </div>
        """),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 2: BRIEFING ───────────────────────────────────────────────────────────
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the variance multiplier in the Pollaczek-Khinchine formula</strong> &mdash; at 80% utilization, ML workloads (C<sub>s</sub> = 3) produce 20x mean-job-duration wait, not the 4x expected for uniform jobs.</div>
                <div style="margin-bottom: 3px;">2. <strong>Identify the optimal utilization range (60&ndash;70%) for ML clusters</strong> and calculate the annual cost of running at 90%: 45x queue multiplier &asymp; 180-hour waits for 4-hour average jobs.</div>
                <div style="margin-bottom: 3px;">3. <strong>Compare FIFO, Priority, and Backfill scheduling</strong> on the joint objective of effective utilization &gt; 60% and P50 queue wait &lt; 2 hours simultaneously.</div>
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
                    Little&rsquo;s Law (L = &lambda;W) from @sec-fleet-orchestration-introduction &middot;
                    Checkpoint overhead and MTBF scaling from Lab 07 (@sec-fault-tolerance-reliability)
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
                "Your cluster runs at 95% utilization and users wait 4 hours for
                1-hour jobs &mdash; adding GPUs barely helps. Is this a capacity
                problem, or is high utilization itself the cause of the catastrophic
                queue wait?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-fleet-orchestration-introduction** — The Scheduling Problem: why
      orchestration becomes the bottleneck when hardware is plentiful
    - **@sec-fleet-orchestration-scheduling-algorithms** — FIFO, priority queues,
      backfill scheduling, and the trade-offs between them
    - **@sec-fleet-orchestration-multi-tenant-cluster-management** — Quota systems,
      fair-share scheduling, and hierarchical resource allocation
    """), kind="info")
    return


# ─── CELL 4: CONTEXT TOGGLE ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "FIFO Scheduling (default Slurm)": "fifo",
            "Priority + Backfill (advanced policy)": "priority",
        },
        value="FIFO Scheduling (default Slurm)",
        label="Scheduling context:",
        inline=True,
    )
    mo.vstack([
        mo.md("### Scheduling Context"),
        mo.md("Select the scheduling policy you will explore. Both Acts use the context "
              "you choose — you can switch and re-run any time."),
        context_toggle,
    ])
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I -- CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "I"
    _act_color    = COLORS["BlueLine"]
    _act_title    = "The Utilization Trap"
    _act_duration = "12-15 min"
    _act_why      = ("You expect that 95% utilization means the cluster is nearly always computing. "
                     "The data will show that at 95% utilization, M/M/1 queuing theory predicts "
                     "mean wait = 20x mean job duration &mdash; a 1-hour job waits 20 hours, "
                     "not because of bugs but because of mathematics.")

    mo.Html(f"""
    <div style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {_act_color}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">{_act_num}</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act {_act_num} &middot; {_act_duration}</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            {_act_title}
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            {_act_why}
        </div>
    </div>
    """)
    return


# ─── CELL: ACT I STAKEHOLDER MESSAGE ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _c  = COLORS["BlueLine"]
    _bg = COLORS["BlueL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_c}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_c};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; HPC Center Director
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "Our 1,024-GPU Slurm cluster runs at 95% utilization. Users are furious —
            their 1-hour test jobs wait 4 hours before they even start. We benchmarked
            the hardware, there are no bottlenecks. We requested budget to add 256 more
            GPUs. The VP asked why we can't just fix the scheduler. We don't understand
            why the wait is so long when the cluster is barely idle."
        </div>
    </div>
    """)
    return


# ─── CELL: ACT I CONCEPT SETUP ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The director's cluster is not broken. The cluster is *too busy*. This is the
    utilization trap: the same property that maximizes hardware ROI — high utilization —
    mathematically guarantees long queue waits.

    **Little's Law** (John Little, 1961) governs every queuing system, from airport
    security to Slurm job schedulers:

    > L = λ × W

    where **L** is mean queue depth (jobs waiting), **λ** is arrival rate (jobs/hour),
    and **W** is mean wait time (hours). For an M/M/1 queue with utilization ρ = λ/μ
    (where μ is service rate), the mean wait time is:

    > W = service_time / (1 − ρ)

    At ρ = 0.95: W = service_time / 0.05 = **20 × service_time**.

    A 1-hour job waits 20 hours at 95% utilization. The director's 4-hour wait for
    1-hour jobs corresponds to approximately 75–80% effective utilization — entirely
    consistent with queuing theory. Adding more GPUs does not change the physics
    unless it meaningfully lowers utilization.
    """)
    return


# ─── CELL: ACT I PREDICTION ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Your Prediction

    *Before exploring the simulator, commit to your hypothesis.*

    The HPC Center Director reports: 1,024-GPU cluster, 95% utilization, 1-hour mean
    job duration, 4-hour mean queue wait. Which explanation is correct?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A — Buy more GPUs: 95% utilization means we are out of capacity. Adding 256 GPUs will cut waits proportionally.": "a",
            "B — Scheduler bug: 95% utilization should produce only slightly longer waits. Something is wrong with the Slurm configuration.": "b",
            "C — Physics of queuing: at 95% utilization, M/M/1 theory predicts 20× mean job runtime wait. A 4-hour wait for 1-hour jobs is mathematically expected.": "c",
            "D — Job mix problem: large jobs are hoarding GPUs unfairly. The scheduler should enforce per-user GPU limits.": "d",
        },
        label="The 4-hour queue wait for 1-hour jobs is caused by:",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(mo.md("Select your prediction to continue to the simulator."), kind="warn"),
    )
    mo.callout(mo.md(
        f"**Prediction locked:** Option {act1_pred.value.upper()}. "
        "Now run the utilization explorer to test your hypothesis."
    ), kind="info")
    return


# ─── CELL: ACT I INSTRUMENTS ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Utilization Explorer")
    return


@app.cell(hide_code=True)
def _(mo):
    util_slider = mo.ui.slider(
        start=50, stop=99, value=95, step=1,
        label="Cluster utilization ρ (%)",
    )
    mean_job_h_slider = mo.ui.slider(
        start=1, stop=24, value=1, step=1,
        label="Mean job duration (hours)",
    )
    arrival_slider = mo.ui.slider(
        start=5, stop=100, value=40, step=5,
        label="Job arrival rate (jobs/hour)",
    )
    mo.vstack([
        mo.hstack([util_slider, mean_job_h_slider, arrival_slider], justify="start", gap="2rem"),
    ])
    return (arrival_slider, mean_job_h_slider, util_slider,)


@app.cell(hide_code=True)
def _(mo, util_slider, mean_job_h_slider, arrival_slider, COLORS, go, np, apply_plotly_theme, math):
    # ── M/M/1 queue physics ───────────────────────────────────────────────────
    rho       = util_slider.value / 100.0          # utilization fraction
    svc_h     = mean_job_h_slider.value            # mean service time (hours)
    lam       = arrival_slider.value               # arrival rate (jobs/hour)

    # M/M/1 mean wait time: W = svc / (1 - rho)
    # Guard against division by zero at rho → 1
    _eps = 1e-6
    wait_h    = svc_h / max(1.0 - rho, _eps)      # hours
    wait_min  = wait_h * 60.0                      # minutes
    wait_min_p99 = wait_h * 60.0 * 4.6            # P99 approximation (M/M/1: ~4.6× mean)

    # Little's Law: L = λ × W
    queue_depth = lam * wait_h                     # mean jobs waiting

    # Color coding
    _ok    = COLORS["GreenLine"]
    _warn  = COLORS["OrangeLine"]
    _fail  = COLORS["RedLine"]

    def _wait_color(w_min):
        if w_min < 30:
            return _ok
        elif w_min < 120:
            return _warn
        return _fail

    _wc = _wait_color(wait_min)
    _p99c = _wait_color(wait_min_p99)

    # ── Utilization vs wait curve (for chart) ────────────────────────────────
    _rho_range = np.linspace(0.50, 0.99, 200)
    _wait_curve = (svc_h / (1.0 - _rho_range)) * 60.0   # minutes
    # Knee region: highlight 80–85%
    _knee_lo, _knee_hi = 0.80, 0.85

    _fig = go.Figure()
    # Fill danger zone (>85%)
    _fig.add_vrect(x0=85, x1=99, fillcolor="rgba(203,32,45,0.06)",
                   layer="below", line_width=0, annotation_text="Danger zone",
                   annotation_position="top left",
                   annotation_font=dict(color=_fail, size=11))
    # Fill knee zone (80–85%)
    _fig.add_vrect(x0=80, x1=85, fillcolor="rgba(204,85,0,0.08)",
                   layer="below", line_width=0, annotation_text="Knee region",
                   annotation_position="top left",
                   annotation_font=dict(color=_warn, size=11))
    # Wait curve
    _fig.add_trace(go.Scatter(
        x=_rho_range * 100,
        y=_wait_curve,
        mode="lines",
        line=dict(color=COLORS["BlueLine"], width=2.5),
        name="Mean wait (min)",
    ))
    # Current operating point
    _fig.add_trace(go.Scatter(
        x=[util_slider.value],
        y=[wait_min],
        mode="markers",
        marker=dict(color=_wc, size=14, symbol="circle",
                    line=dict(color="white", width=2)),
        name=f"Current: {util_slider.value}% util → {wait_min:.0f} min wait",
    ))
    # 30-minute SLO line
    _fig.add_hline(y=30, line_dash="dot", line_color=_ok,
                   annotation_text="30 min SLO", annotation_position="bottom right",
                   annotation_font=dict(color=_ok, size=11))

    _fig.update_layout(
        title=None,
        xaxis=dict(title="Cluster Utilization ρ (%)", range=[50, 100]),
        yaxis=dict(title="Mean Queue Wait (minutes)", range=[0, max(400, wait_min * 1.15)]),
        height=320,
        legend=dict(x=0.02, y=0.95, bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(l=50, r=20, t=20, b=50),
    )
    apply_plotly_theme(_fig)

    # ── Formula display ───────────────────────────────────────────────────────
    _formula_html = f"""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
                padding: 16px 20px; margin: 12px 0; font-family: 'SF Mono', monospace;
                font-size: 0.87rem; color: #334155; line-height: 1.9;">
        <div style="font-weight: 700; color: #0f172a; margin-bottom: 8px; font-size: 0.8rem;
                    text-transform: uppercase; letter-spacing: 0.08em;">
            M/M/1 Queue Physics
        </div>
        ρ = {rho:.2f} (utilization fraction)<br>
        W = service_time / (1 − ρ)
          = {svc_h:.1f}h / (1 − {rho:.2f})
          = {svc_h:.1f}h / {(1 - rho):.2f}
          = <strong style="color:{_wc};">{wait_h:.1f} hours ({wait_min:.0f} minutes)</strong><br>
        L = λ × W = {lam} jobs/h × {wait_h:.1f}h
          = <strong>{queue_depth:.1f} jobs waiting</strong><br>
        P99 ≈ 4.6 × W = <strong style="color:{_p99c};">{wait_min_p99:.0f} minutes</strong>
    </div>
    """

    # ── Metric cards ─────────────────────────────────────────────────────────
    _metrics_html = f"""
    <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0;">
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 160px; text-align: center; background: white;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                Mean Wait
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: {_wc}; line-height: 1;">
                {wait_min:.0f} min
            </div>
            <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                {wait_h:.2f} hours
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 160px; text-align: center; background: white;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                P99 Wait
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: {_p99c}; line-height: 1;">
                {wait_min_p99:.0f} min
            </div>
            <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                ~4.6× mean wait
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 160px; text-align: center; background: white;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                Queue Depth L
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: {COLORS['BlueLine']}; line-height: 1;">
                {queue_depth:.1f}
            </div>
            <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                jobs waiting (L = λW)
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 160px; text-align: center; background: white;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                Wait / Job Duration
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: {_wc}; line-height: 1;">
                {wait_h / svc_h:.1f}×
            </div>
            <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                multiplier (M/M/1: {1/(1-rho):.1f}×)
            </div>
        </div>
    </div>
    """

    mo.vstack([
        mo.Html(_formula_html),
        mo.Html(_metrics_html),
        mo.Html("<div style='margin-top:8px;'></div>"),
        mo.ui.plotly(_fig),
    ])
    return (rho, svc_h, lam, wait_h, wait_min, wait_min_p99, queue_depth,)


# ─── CELL: ACT I PREDICTION VS REALITY OVERLAY ────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, wait_h, svc_h, rho):
    _correct_answer = "c"
    _is_correct = act1_pred.value == _correct_answer

    # At 95% util and 1-hour mean job: theory predicts exactly 20h wait
    _theory_at_95pct = 1.0 / (1.0 - 0.95)   # = 20 hours
    _ratio = wait_h / svc_h                   # multiplier at current setting

    if _is_correct:
        _msg = mo.callout(mo.md(
            f"**Correct.** Option C is the only explanation consistent with the physics. "
            f"At {rho*100:.0f}% utilization, M/M/1 theory predicts wait = "
            f"**{_ratio:.1f}× mean job duration** = {wait_h:.1f} hours. "
            f"At 95% specifically: {_theory_at_95pct:.0f}× mean job duration. "
            f"The director's 4-hour wait for 1-hour jobs corresponds to ~80% effective "
            f"utilization — perfectly consistent. Adding GPUs only helps if it brings "
            f"utilization below the knee (~80–85%)."
        ), kind="success")
    elif act1_pred.value == "a":
        _msg = mo.callout(mo.md(
            f"**Not quite.** Adding GPUs is only effective if it reduces utilization "
            f"substantially below 85%. At {rho*100:.0f}% util, wait = **{_ratio:.1f}×** "
            f"mean job duration. Adding 25% more GPUs lowers utilization by only ~4 percentage "
            f"points (from 95% to ~76%). That cuts wait from 20× to ~4× — a real improvement, "
            f"but the root cause is utilization physics, not raw capacity. Option C is the "
            f"precise explanation."
        ), kind="warn")
    elif act1_pred.value == "b":
        _msg = mo.callout(mo.md(
            f"**Not quite.** There is no scheduler bug. The M/M/1 queue model makes an "
            f"explicit prediction: at {rho*100:.0f}% utilization, wait = **{_ratio:.1f}×** "
            f"mean job runtime. At 95%, this is 20× — a 4-hour wait for 1-hour jobs is "
            f"exactly what queuing theory forecasts. The scheduler is working as designed; "
            f"the design runs the cluster too hot."
        ), kind="warn")
    else:
        _msg = mo.callout(mo.md(
            f"**Not quite.** Per-user GPU limits affect fairness but not the aggregate "
            f"wait time driven by cluster-wide utilization. At {rho*100:.0f}% utilization, "
            f"M/M/1 theory predicts wait = **{_ratio:.1f}×** mean job duration regardless "
            f"of how GPUs are distributed among users. The utilization level itself is the "
            f"primary driver — Option C is the physically correct explanation."
        ), kind="warn")
    _msg
    return


# ─── CELL: ACT I REFLECTION ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Reflection: The Safe Utilization Ceiling

    Set the mean job duration to **1 hour** and find the utilization level where
    mean queue wait falls below 30 minutes. What is the practical maximum?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A — 95%: modern schedulers handle high utilization fine.": "a",
            "B — ~80%: above this, queuing theory predicts wait grows faster than linearly.": "b",
            "C — 50%: utilization above 50% always causes unacceptable waits.": "c",
            "D — Queue wait depends only on job mix, not utilization level.": "d",
        },
        label="The practical maximum cluster utilization to keep wait under 30 min for 1-hour jobs is:",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select your answer to continue."), kind="warn"),
    )
    if act1_reflect.value == "b":
        mo.callout(mo.md(
            "**Correct.** At 83% utilization with a 1-hour mean job: "
            "W = 1h / (1 − 0.83) = 5.9 hours. Wait is still substantial, but the "
            "*rate of change* is the key insight. From 50% to 80%, doubling utilization "
            "roughly doubles wait. From 80% to 95%, a 15-percentage-point increase "
            "multiplies wait by 5×. The knee of the curve sits at ~80–85%. HPC centers "
            "that target 80% utilization as a ceiling preserve headroom against burst "
            "arrivals and keep mean wait under 5× mean service time."
        ), kind="success")
    elif act1_reflect.value == "c":
        mo.callout(mo.md(
            "**Too conservative.** At 50% utilization, W = 2× mean job duration — "
            "perfectly reasonable (2-hour wait for a 1-hour job). The wait at 50% is "
            "manageable; the problem only becomes acute above ~80–85%, where the curve "
            "turns sharply upward. Targeting 50% wastes half your expensive GPU capacity."
        ), kind="warn")
    elif act1_reflect.value == "a":
        mo.callout(mo.md(
            "**Incorrect.** No scheduler — however advanced — overcomes the M/M/1 "
            "physics. At 95% utilization, wait = 20× mean job duration. This is a "
            "mathematical consequence of utilization, not a software limitation. A "
            "better scheduler redistributes wait across job classes; it does not "
            "eliminate wait caused by high utilization."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Incorrect.** Queue wait is determined by utilization level, not just "
            "job mix. Little's Law L = λW holds for any queuing system. For fixed "
            "mean service time, W = service_time / (1 − ρ) depends directly on "
            "utilization ρ. Job mix affects the variance of service times (and thus "
            "P99 relative to mean), but the mean wait is driven primarily by ρ."
        ), kind="warn")
    return


# ─── CELL: ACT I MATH PEEK ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — Little's Law and M/M/1 Queue Theory": mo.md("""
        **Little's Law** (John Little, 1961 — verified empirically for all stable queuing systems):

        > **L = λ × W**

        - **L** — mean number of jobs in system (queue + running)
        - **λ** — mean job arrival rate (jobs per unit time)
        - **W** — mean time a job spends in the system (queue wait + service time)

        ---

        **M/M/1 Queue** (Markovian arrivals, Markovian service, single server):

        Service rate μ, arrival rate λ, utilization **ρ = λ/μ**.

        Mean queue wait (time waiting, not including service):

        > **W_q = ρ × service_time / (1 − ρ)**

        Total mean time in system (wait + service):

        > **W = service_time / (1 − ρ)**

        **The heavy-traffic approximation**: as ρ → 1, W → ∞. The queue becomes
        unboundedly long as utilization approaches 100%.

        ---

        **Practical numbers** (mean job duration = 1 hour):

        | Utilization ρ | 1 − ρ | W (hours) | W (minutes) | Wait multiplier |
        |:---:|:---:|:---:|:---:|:---:|
        | 50% | 0.50 | 2.0 h | 120 min | 2× |
        | 75% | 0.25 | 4.0 h | 240 min | 4× |
        | 80% | 0.20 | 5.0 h | 300 min | 5× |
        | 85% | 0.15 | 6.7 h | 400 min | 6.7× |
        | 90% | 0.10 | 10.0 h | 600 min | 10× |
        | 95% | 0.05 | 20.0 h | 1200 min | **20×** |
        | 99% | 0.01 | 100.0 h | 6000 min | **100×** |

        The knee of the curve falls at ρ ≈ 80–85%. Above 85%, each additional
        percentage point of utilization multiplies wait disproportionately.

        ---

        **P99 approximation for M/M/1**: P99 ≈ 4.6 × mean wait (from the exponential
        tail of the M/M/1 sojourn distribution). This means at 95% utilization:

        > P99 ≈ 4.6 × 20h ≈ **92 hours** for a 1-hour mean job.
        """)
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II -- DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "II"
    _act_color    = COLORS["OrangeLine"]
    _act_title    = "Scheduling Policy Design"
    _act_duration = "20-25 min"
    _act_why      = ("Act I showed that high utilization is the root cause of long waits. "
                     "Now discover that no single scheduling policy can simultaneously "
                     "satisfy all objectives: FIFO blocks interactive jobs, priority queues "
                     "starve training runs, and only backfill &mdash; filling temporal gaps "
                     "between reservations &mdash; achieves both low interactive wait and "
                     "high effective utilization.")

    mo.Html(f"""
    <div style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {_act_color}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">{_act_num}</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act {_act_num} &middot; {_act_duration}</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            {_act_title}
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            {_act_why}
        </div>
    </div>
    """)
    return


# ─── CELL: ACT II STAKEHOLDER MESSAGE ────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _c  = COLORS["OrangeLine"]
    _bg = COLORS["OrangeL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_c}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_c};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; ML Platform Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We run three job classes on our 1,024-GPU cluster: (1) interactive
            experiments — 8 GPUs, 30 minutes, roughly 50 jobs per day; (2) full
            model training — 64 to 512 GPUs, 12 to 24 hours, 10 jobs per day;
            and (3) hyperparameter search — 8 to 32 GPUs, 2 hours, 20 jobs per day.
            Our SLA says interactive jobs must start within 5 minutes. Right now on
            FIFO, interactive jobs wait behind large training runs for 2 to 4 hours.
            Design a scheduling policy that keeps interactive wait under 5 minutes
            without starving the training jobs."
        </div>
    </div>
    """)
    return


# ─── CELL: ACT II SCENARIO SETUP ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, INTERACTIVE_GPUS, INTERACTIVE_DUR_H, INTERACTIVE_JOBS_DAY,
       TRAINING_GPUS_MIN, TRAINING_GPUS_MAX, TRAINING_DUR_H, TRAINING_JOBS_DAY,
       SEARCH_GPUS_MIN, SEARCH_GPUS_MAX, SEARCH_DUR_H, SEARCH_JOBS_DAY,
       CLUSTER_SIZE_GPUS):
    # Compute GPU-hours demand per day for each class
    _inter_gpuh_day  = INTERACTIVE_GPUS * INTERACTIVE_DUR_H * INTERACTIVE_JOBS_DAY
    _train_gpuh_day  = ((TRAINING_GPUS_MIN + TRAINING_GPUS_MAX) / 2) * TRAINING_DUR_H * TRAINING_JOBS_DAY
    _search_gpuh_day = ((SEARCH_GPUS_MIN + SEARCH_GPUS_MAX) / 2) * SEARCH_DUR_H * SEARCH_JOBS_DAY
    _total_demand    = _inter_gpuh_day + _train_gpuh_day + _search_gpuh_day
    _cluster_cap_day = CLUSTER_SIZE_GPUS * 24   # GPU-hours available per day
    _util_pct        = (_total_demand / _cluster_cap_day) * 100

    mo.md(f"""
    The three job classes create the following daily load profile on the
    {CLUSTER_SIZE_GPUS:,}-GPU cluster:

    | Job Class | GPUs | Duration | Jobs/Day | GPU-hours/Day |
    |:---|:---:|:---:|:---:|:---:|
    | Interactive | {INTERACTIVE_GPUS} | {INTERACTIVE_DUR_H*60:.0f} min | {INTERACTIVE_JOBS_DAY} | {_inter_gpuh_day:,.0f} |
    | Model Training | {TRAINING_GPUS_MIN}–{TRAINING_GPUS_MAX} | {TRAINING_DUR_H:.0f}h | {TRAINING_JOBS_DAY} | {_train_gpuh_day:,.0f} |
    | HP Search | {SEARCH_GPUS_MIN}–{SEARCH_GPUS_MAX} | {SEARCH_DUR_H:.0f}h | {SEARCH_JOBS_DAY} | {_search_gpuh_day:,.0f} |
    | **Total demand** | | | | **{_total_demand:,.0f}** |
    | Cluster capacity | | 24h | | {_cluster_cap_day:,} |

    Aggregate utilization: **{_util_pct:.1f}%** — firmly in the danger zone
    from Act I. Scheduling policy determines *which jobs* wait, but the total
    load is set by arrival rates and job sizes.
    """)
    return


# ─── CELL: ACT II PREDICTION ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Your Prediction

    *Before configuring the scheduler, commit to your hypothesis.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A — FIFO: fairest for all users; jobs start in arrival order.": "a",
            "B — Priority (interactive > search > training) with preemption: highest-priority class always runs first.": "b",
            "C — Priority queues with backfill: short jobs fill idle gaps left by large reserved jobs — higher utilization and lower interactive wait.": "c",
            "D — Separate partitions for each class: each class gets dedicated GPUs, no interference.": "d",
        },
        label="Which scheduling policy keeps interactive wait < 5 min without starving training jobs?",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(mo.md("Select your prediction to unlock the scheduler simulator."), kind="warn"),
    )
    mo.callout(mo.md(
        f"**Prediction locked:** Option {act2_pred.value.upper()}. "
        "Configure the scheduler below and observe the per-class wait times."
    ), kind="info")
    return


# ─── CELL: ACT II INSTRUMENTS ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Scheduler Policy Simulator")
    return


@app.cell(hide_code=True)
def _(mo):
    policy_drop = mo.ui.dropdown(
        options={
            "FIFO (first-in, first-out)": "fifo",
            "Priority (interactive > search > training)": "priority",
            "Priority + Backfill": "backfill",
            "Fair-Share (weighted by class)": "fairshare",
        },
        value="FIFO (first-in, first-out)",
        label="Scheduling policy",
    )
    inter_arrival_slider = mo.ui.slider(
        start=1, stop=20, value=2, step=1,
        label="Interactive arrival rate (jobs/hour)",
    )
    train_size_slider = mo.ui.slider(
        start=64, stop=512, value=256, step=64,
        label="Training job size (GPUs)",
    )
    inter_weight_slider = mo.ui.slider(
        start=1, stop=10, value=5, step=1,
        label="Interactive priority weight (fair-share only)",
    )

    mo.vstack([
        mo.hstack([policy_drop, inter_arrival_slider], justify="start", gap="2rem"),
        mo.hstack([train_size_slider, inter_weight_slider], justify="start", gap="2rem"),
    ])
    return (policy_drop, inter_arrival_slider, train_size_slider, inter_weight_slider,)


@app.cell(hide_code=True)
def _(
    mo, policy_drop, inter_arrival_slider, train_size_slider, inter_weight_slider,
    COLORS, go, np, apply_plotly_theme, math,
    CLUSTER_SIZE_GPUS, INTERACTIVE_GPUS, INTERACTIVE_DUR_H, INTERACTIVE_SLA_MIN,
    TRAINING_DUR_H, SEARCH_DUR_H, SEARCH_GPUS_MIN, SEARCH_GPUS_MAX,
    TRAINING_GPUS_MIN, TRAINING_GPUS_MAX,
):
    # ── Scheduling simulation physics ─────────────────────────────────────────
    # This is a queueing-theory-based approximation, not a full discrete-event sim.
    # Each job class modeled as M/M/1 queue with policy-dependent effective service rate.
    #
    # Policy effects on effective utilization per class:
    #   FIFO:       all classes compete for same queue; large jobs block small ones
    #   Priority:   interactive/search jump queue; training waits behind all
    #   Backfill:   interactive fills gaps; training wait reduced by ~20% vs priority
    #   Fair-share: utilization allocated by weight; starvation capped by decay function
    #
    # Effective utilization per class: ρ_class = (λ_class × S_class) / C_class
    # where C_class is the share of cluster capacity assigned to that class.

    policy     = policy_drop.value
    lam_inter  = inter_arrival_slider.value       # jobs/hour (interactive)
    train_gpus = train_size_slider.value          # GPUs per training job
    w_inter    = inter_weight_slider.value        # fair-share weight for interactive
    cluster    = CLUSTER_SIZE_GPUS               # total GPUs

    # Job parameters
    s_inter    = INTERACTIVE_DUR_H               # 0.5 hours service time (interactive)
    s_train    = TRAINING_DUR_H                  # 18 hours service time (training)
    s_search   = SEARCH_DUR_H                    # 2 hours service time (search)
    g_inter    = INTERACTIVE_GPUS               # 8 GPUs (interactive)
    g_train    = train_gpus                      # variable GPUs (training)
    g_search   = (SEARCH_GPUS_MIN + SEARCH_GPUS_MAX) / 2   # 20 GPUs avg (search)

    # Arrival rates (jobs/hour): interactive is variable; training and search from spec
    lam_train  = 10 / 24.0                       # 10 jobs/day → 0.417 jobs/hour
    lam_search = 20 / 24.0                       # 20 jobs/day → 0.833 jobs/hour

    # GPU-hours per job per class
    gh_inter  = g_inter * s_inter               # 4 GPU-hours/job
    gh_train  = g_train * s_train               # variable (depends on slider)
    gh_search = g_search * s_search             # 40 GPU-hours/job

    # Aggregate GPU utilization fraction: sum(λ * GPU-hours/job) / cluster_capacity_per_hour
    rho_total = (lam_inter * gh_inter + lam_train * gh_train + lam_search * gh_search) / cluster
    rho_total = min(rho_total, 0.99)            # cap for numerical stability

    # ── Policy-dependent wait time model ─────────────────────────────────────
    # Base M/M/1 wait per class given effective utilization seen by that class.
    # Policy modifies the utilization "seen" by each class via priority and
    # capacity allocation factors derived from the chapter's discussion:
    #
    # FIFO: all classes share a single queue in arrival order. A large training job
    #   arriving just before an interactive job means the interactive job waits
    #   the full training duration. We model this as each class seeing rho_total.
    #
    # Priority: interactive sees only its own offered load; search sees its load
    #   plus some bleed from interactive; training sees near-full cluster load.
    #   Approximation: rho_interactive = lam_inter*gh_inter/cluster; rho_train ≈ rho_total.
    #
    # Backfill: training reserves nodes for its large allocations, but the reserved
    #   idle slots are filled with small interactive/search jobs. Interactive wait
    #   falls because it can fill any gap large enough for 8 GPUs.
    #   Approximation: interactive rho ≈ 0.5 * rho_inter_only; training rho ≈ rho_total - 0.15.
    #
    # Fair-share: weighted allocation proportional to weight ratios. Each class
    #   sees a capped utilization based on its GPU allocation fraction.

    _rho_inter_own  = min(lam_inter * gh_inter / cluster, 0.99)
    _rho_train_full = rho_total
    _rho_search_own = min(lam_search * gh_search / cluster, 0.99)

    _eps = 1e-6  # prevent division by zero

    if policy == "fifo":
        # All classes share single queue; large jobs cause head-of-line blocking.
        # Interactive effectively sees full cluster utilization (worst case for small jobs).
        rho_i = rho_total
        rho_t = rho_total
        rho_s = rho_total
        # FIFO interactive wait: W = s / (1 - rho_total) — blocked by large training jobs
        wait_inter_h   = s_inter / max(1.0 - rho_i, _eps)
        wait_train_h   = s_train / max(1.0 - rho_t, _eps)
        wait_search_h  = s_search / max(1.0 - rho_s, _eps)

    elif policy == "priority":
        # Interactive jumps to front of queue — sees only its own offered load.
        # Training is preempted by higher-priority arrivals; sees near full utilization.
        rho_i = _rho_inter_own
        rho_t = rho_total
        rho_s = min(_rho_inter_own + _rho_search_own, 0.99)
        wait_inter_h   = s_inter / max(1.0 - rho_i, _eps)
        wait_train_h   = s_train / max(1.0 - rho_t, _eps)
        wait_search_h  = s_search / max(1.0 - rho_s, _eps)
        # Priority preemption overhead: checkpoint/restart adds ~10 min per preemption
        # (negligible for training duration but significant for search jobs)
        # We add 0.1h overhead to search wait to account for preemption cost
        wait_search_h  += 0.1

    elif policy == "backfill":
        # Backfill: large training jobs reserve nodes but don't block small jobs
        # from filling gaps. Interactive sees dramatically reduced effective ρ.
        # Key insight: a 512-GPU training job waiting for nodes leaves many 8-GPU
        # gaps that interactive jobs can fill immediately.
        _gap_fill_factor = min(g_train / cluster, 0.6)  # fraction of nodes in gap
        rho_i = max(_rho_inter_own * (1.0 - _gap_fill_factor), 0.05)
        rho_t = max(rho_total - 0.12, _rho_train_full * 0.85)   # slightly better due to gap fill
        rho_s = min(_rho_search_own + _rho_inter_own * 0.3, 0.90)
        wait_inter_h   = s_inter / max(1.0 - rho_i, _eps)
        wait_train_h   = s_train / max(1.0 - rho_t, _eps)
        wait_search_h  = s_search / max(1.0 - rho_s, _eps)
        # No preemption overhead for backfill (non-preemptive)

    else:  # fairshare
        # Allocate GPU capacity proportionally to weights.
        # Interactive weight = w_inter; training weight = 5 (fixed); search weight = 3 (fixed)
        _w_train  = 5
        _w_search = 3
        _w_total  = w_inter + _w_train + _w_search
        _frac_i   = w_inter / _w_total
        _frac_t   = _w_train / _w_total
        _frac_s   = _w_search / _w_total
        # Each class's utilization within its allocated fraction:
        rho_i = min(lam_inter * gh_inter / (cluster * _frac_i), 0.99)
        rho_t = min(lam_train * gh_train / (cluster * _frac_t), 0.99)
        rho_s = min(lam_search * gh_search / (cluster * _frac_s), 0.99)
        wait_inter_h   = s_inter / max(1.0 - rho_i, _eps)
        wait_train_h   = s_train / max(1.0 - rho_t, _eps)
        wait_search_h  = s_search / max(1.0 - rho_s, _eps)

    # Convert to minutes
    wait_inter_min  = wait_inter_h * 60.0
    wait_train_min  = wait_train_h * 60.0
    wait_search_min = wait_search_h * 60.0

    # Starvation risk: fraction of training jobs waiting > 48 hours
    # Modeled as P(W > 48h) = exp(-48h / wait_train_h) for exponential sojourn
    _p_starve = math.exp(-48.0 / max(wait_train_h, 0.01))
    starvation_pct = _p_starve * 100.0

    # SLA check
    sla_met = wait_inter_min <= INTERACTIVE_SLA_MIN

    # ── Effective cluster utilization (weighted by class GPU demand) ──────────
    # After scheduling, utilization stays at rho_total — policy doesn't change demand.
    eff_util_pct = rho_total * 100.0

    # ── Color helpers ─────────────────────────────────────────────────────────
    _ok   = COLORS["GreenLine"]
    _warn = COLORS["OrangeLine"]
    _fail = COLORS["RedLine"]
    _blue = COLORS["BlueLine"]

    def _wcolor(w_min, sla_min=5.0):
        if w_min <= sla_min:
            return _ok
        elif w_min <= sla_min * 3:
            return _warn
        return _fail

    _ic = _wcolor(wait_inter_min, INTERACTIVE_SLA_MIN)
    _tc = _wcolor(wait_train_min, 60 * 12)   # training SLA = 12h for coloring
    _sc = _wcolor(wait_search_min, 30)       # search SLA = 30 min for coloring

    # ── Gantt chart simulation (30 time steps, illustrative) ─────────────────
    # Simulate simplified 30-step schedule using the chosen policy.
    # Each step = 1 hour. Show which jobs are running on a 32-GPU slice.
    _n_steps  = 30
    _n_gpus_vis = 32   # show a 32-GPU slice for clarity
    _rng      = np.random.default_rng(seed=42)

    # Generate job arrivals for the 30-hour window
    _inter_arrivals  = _rng.poisson(lam_inter * _n_steps)
    _train_arrivals  = max(1, int(lam_train * _n_steps))
    _search_arrivals = max(1, int(lam_search * _n_steps))

    # Build simplified schedule blocks for Gantt
    _gantt_bars = []
    _t_cursor = 0.0

    # Training blocks (large, long)
    for _i in range(min(_train_arrivals, 3)):
        _start = _i * (s_train * 0.8)
        if policy == "fifo":
            _start_adj = _start + wait_train_h * 0.3
        else:
            _start_adj = _start + wait_train_h * 0.1
        _start_adj = min(_start_adj, _n_steps - s_train)
        if _start_adj < _n_steps:
            _gantt_bars.append({
                "label": f"Training {_i+1}",
                "start": _start_adj,
                "dur": min(s_train, _n_steps - _start_adj),
                "gpu_start": 0,
                "gpu_span": min(g_train, _n_gpus_vis),
                "color": "rgba(99,102,241,0.7)",
                "class": "training",
            })

    # Search blocks (medium)
    for _i in range(min(_search_arrivals, 6)):
        _start = _i * (s_search + 1.0)
        if policy in ("priority", "backfill"):
            _start_adj = _start + wait_search_h * 0.2
        else:
            _start_adj = _start + wait_search_h * 0.5
        _start_adj = min(_start_adj, _n_steps - s_search)
        _g_s = min(int(g_search), _n_gpus_vis - int(g_inter))
        if _start_adj < _n_steps and _g_s > 0:
            _gantt_bars.append({
                "label": f"Search {_i+1}",
                "start": _start_adj,
                "dur": min(s_search, _n_steps - _start_adj),
                "gpu_start": _n_gpus_vis - _g_s,
                "gpu_span": _g_s,
                "color": "rgba(204,85,0,0.65)",
                "class": "search",
            })

    # Interactive blocks (small, short)
    for _i in range(min(_inter_arrivals, 12)):
        _start = _i * 2.0
        if policy in ("backfill", "priority"):
            _start_adj = _start + wait_inter_h * 0.5
        else:
            _start_adj = _start + wait_inter_h
        _start_adj = min(_start_adj, _n_steps - s_inter)
        if _start_adj < _n_steps:
            _gantt_bars.append({
                "label": f"Interactive {_i+1}" if _i < 4 else "",
                "start": _start_adj,
                "dur": min(s_inter, _n_steps - _start_adj),
                "gpu_start": _n_gpus_vis - g_inter,
                "gpu_span": g_inter,
                "color": "rgba(0,143,69,0.7)",
                "class": "interactive",
            })

    _fig2 = go.Figure()

    for _bar in _gantt_bars:
        _fig2.add_trace(go.Bar(
            x=[_bar["dur"]],
            y=[f"GPU {_bar['gpu_start']}-{_bar['gpu_start'] + _bar['gpu_span'] - 1}"],
            base=[_bar["start"]],
            orientation="h",
            marker=dict(color=_bar["color"], line=dict(color="white", width=0.5)),
            name=_bar["class"].capitalize(),
            showlegend=False,
            text=_bar["label"],
            textposition="inside",
            insidetextanchor="start",
            hovertemplate=(
                f"<b>{_bar['label']}</b><br>"
                f"Start: {_bar['start']:.1f}h<br>"
                f"Duration: {_bar['dur']:.1f}h<br>"
                f"GPUs: {_bar['gpu_span']}<extra></extra>"
            ),
        ))

    # Legend traces (invisible, just for legend)
    for _cls, _col, _lbl in [
        ("training", "rgba(99,102,241,0.7)", "Training"),
        ("search",   "rgba(204,85,0,0.65)",  "Search"),
        ("interactive", "rgba(0,143,69,0.7)", "Interactive"),
    ]:
        _fig2.add_trace(go.Bar(
            x=[0], y=["Legend"], base=[0],
            orientation="h",
            marker=dict(color=_col),
            name=_lbl,
            showlegend=True,
        ))

    _fig2.update_layout(
        barmode="overlay",
        xaxis=dict(title="Simulation Time (hours)", range=[0, _n_steps]),
        yaxis=dict(title="GPU Slice (32 of 1,024 shown)", autorange="reversed"),
        height=280,
        legend=dict(orientation="h", x=0, y=1.12, bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(l=60, r=20, t=40, b=50),
    )
    apply_plotly_theme(_fig2)

    # ── Metrics HTML ──────────────────────────────────────────────────────────
    _metrics2 = f"""
    <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0;">
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 150px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                Interactive Wait
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: {_ic}; line-height: 1;">
                {wait_inter_min:.1f} min
            </div>
            <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                SLA: &lt;{INTERACTIVE_SLA_MIN:.0f} min
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 150px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                Training Wait
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: {_tc}; line-height: 1;">
                {wait_train_h:.1f} h
            </div>
            <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                {wait_train_min:.0f} min
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 150px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                Search Wait
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: {_sc}; line-height: 1;">
                {wait_search_min:.1f} min
            </div>
            <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                {wait_search_h:.2f} h
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 150px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                Cluster Util.
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: {_blue}; line-height: 1;">
                {eff_util_pct:.1f}%
            </div>
            <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                aggregate ρ
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 150px; text-align: center; background: white;">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                Starvation Risk
            </div>
            <div style="font-size: 2rem; font-weight: 800;
                        color: {"#CB202D" if starvation_pct > 20 else "#CC5500" if starvation_pct > 5 else "#008F45"};
                        line-height: 1;">
                {starvation_pct:.1f}%
            </div>
            <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">
                training jobs &gt;48h wait
            </div>
        </div>
    </div>
    """

    mo.vstack([
        mo.Html(_metrics2),
        mo.Html("<div style='margin-top:8px;'></div>"),
        mo.md(f"**30-hour schedule sample** (32-GPU slice, policy: `{policy}`)"),
        mo.ui.plotly(_fig2),
    ])
    return (
        wait_inter_min, wait_train_min, wait_search_min,
        starvation_pct, eff_util_pct, sla_met,
        policy, rho_total,
    )


# ─── CELL: ACT II FAILURE STATES ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, wait_inter_min, starvation_pct, sla_met, INTERACTIVE_SLA_MIN):
    _items = []

    # SLA violation: interactive wait > 5 min
    if not sla_met:
        _items.append(mo.callout(mo.md(
            f"**SLA Violated.** Interactive jobs waiting "
            f"**{wait_inter_min:.0f} min**. Target: < {INTERACTIVE_SLA_MIN:.0f} min. "
            f"Switch to `Priority + Backfill` to fill GPU gaps with short jobs, "
            f"or reduce cluster utilization below 85% to lower queue depth."
        ), kind="danger"))

    # Starvation risk: training jobs waiting > 48h more than 20% of the time
    if starvation_pct > 20:
        _items.append(mo.callout(mo.md(
            f"**Training Job Starvation Risk.** {starvation_pct:.0f}% of large training "
            f"jobs are estimated to wait more than 48 hours. Consider lowering interactive "
            f"priority weight, adding a maximum wait time guarantee (backfill reservation), "
            f"or reducing training job size to improve scheduling flexibility."
        ), kind="warn"))

    # Success state: both SLA met and no starvation
    if sla_met and starvation_pct <= 20:
        _items.append(mo.callout(mo.md(
            f"**Policy working.** Interactive wait: {wait_inter_min:.1f} min "
            f"(SLA: < {INTERACTIVE_SLA_MIN:.0f} min). "
            f"Training starvation risk: {starvation_pct:.1f}% (target: < 20%). "
            f"This policy balances responsiveness against fairness at the current "
            f"arrival rates and job sizes."
        ), kind="success"))

    mo.vstack(_items) if _items else mo.md("")
    return


# ─── CELL: ACT II PREDICTION REVEAL ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred, wait_inter_min, sla_met, starvation_pct, INTERACTIVE_SLA_MIN):
    _correct = "c"
    _is_correct = act2_pred.value == _correct

    if _is_correct:
        _reveal = mo.callout(mo.md(
            f"**Correct prediction.** Priority + Backfill achieves the best balance. "
            f"Backfill fills idle GPU gaps — the nodes reserved for large training jobs "
            f"but not yet started — with short interactive jobs. Interactive wait drops "
            f"dramatically without preempting training runs. Your current configuration "
            f"shows interactive wait = **{wait_inter_min:.1f} min** vs SLA of "
            f"{INTERACTIVE_SLA_MIN:.0f} min."
        ), kind="success")
    elif act2_pred.value == "a":
        _reveal = mo.callout(mo.md(
            f"**FIFO is the least fair for mixed workloads.** In FIFO ordering, a "
            f"512-GPU training job arriving just before an interactive experiment blocks "
            f"it for the entire training duration (18+ hours). FIFO maximizes simplicity "
            f"and provides strong sequential fairness, but SLA guarantees for short jobs "
            f"require either very low utilization or explicit priority mechanisms."
        ), kind="warn")
    elif act2_pred.value == "b":
        _reveal = mo.callout(mo.md(
            f"**Priority with preemption creates its own problem.** Preempting training "
            f"jobs wastes checkpointing overhead (saving and restoring hundreds of GB of "
            f"model state). At high interactive arrival rates, training jobs are "
            f"repeatedly interrupted. Priority without preemption (i.e. backfill) solves "
            f"the problem more efficiently: interactive jobs fill idle gaps without "
            f"disturbing running training jobs."
        ), kind="warn")
    else:
        _reveal = mo.callout(mo.md(
            f"**Separate partitions sacrifice cluster efficiency.** Dedicating fixed GPU "
            f"slices to each class means interactive partition GPUs sit idle when no "
            f"interactive jobs are queued, while training jobs wait in their overloaded "
            f"partition. Cluster utilization drops significantly. Shared pools with "
            f"policy-based allocation (backfill, fair-share) achieve both high "
            f"utilization and SLA compliance."
        ), kind="warn")
    _reveal
    return


# ─── CELL: ACT II REFLECTION ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Reflection: Why Backfill Outperforms Simple Priority

    Set the policy to **Priority + Backfill** and then to **Priority (without
    backfill)**. Compare interactive wait times, then answer:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A — Backfill uses a better sorting algorithm than simple priority.": "a",
            "B — Backfill fills idle GPU-time gaps with small jobs without delaying reserved-resource jobs — higher utilization and lower interactive wait simultaneously.": "b",
            "C — Backfill automatically checkpoints and migrates jobs to free up space.": "c",
            "D — Backfill is faster because it requires fewer Slurm scheduling cycles.": "d",
        },
        label="Why does backfill scheduling outperform simple priority scheduling for ML workloads?",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_reflect):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select your answer to continue."), kind="warn"),
    )
    if act2_reflect.value == "b":
        mo.callout(mo.md(
            "**Correct.** The key insight is that backfill exploits *temporal gaps* — "
            "intervals during which a large job has reserved nodes but cannot yet start "
            "because it is still waiting for all its nodes to become free simultaneously. "
            "Simple priority gives interactive jobs queue position but makes them wait "
            "for running jobs to finish. Backfill gives interactive jobs *immediate* "
            "access to otherwise idle resources, decoupling their wait from training "
            "job duration. This is why Slurm's `backfill` plugin typically reduces "
            "interactive wait by 5–10× compared to strict priority, while maintaining "
            "or improving cluster utilization."
        ), kind="success")
    elif act2_reflect.value == "c":
        mo.callout(mo.md(
            "**Incorrect.** Backfill is non-preemptive — it never checkpoints or "
            "migrates running jobs. That is the key advantage over preemptive priority. "
            "Checkpointing a 512-GPU training job requires writing hundreds of GB to "
            "storage and reloading it on restart. Backfill avoids this entirely by only "
            "scheduling new jobs into idle gaps, never interrupting running ones."
        ), kind="warn")
    elif act2_reflect.value == "a":
        mo.callout(mo.md(
            "**Incorrect.** Both priority and backfill use similar priority ranking for "
            "job ordering. The algorithmic difference is that backfill adds a constraint: "
            "a job can only run in a gap if it will *finish* before the reserved job "
            "needs those resources. This is a temporal feasibility check, not a "
            "sorting improvement. The benefit is structural — idle resources get used "
            "without delaying high-priority reservations."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Incorrect.** Backfill actually requires *more* scheduling computation per "
            "cycle because it must check whether candidate jobs fit within the temporal "
            "gap before the reserved job starts. Simple FIFO or priority scheduling "
            "is computationally cheaper. The benefit of backfill is cluster efficiency "
            "and lower interactive latency, not reduced scheduling overhead."
        ), kind="warn")
    return


# ─── CELL: ACT II MATH PEEK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Scheduling policy equations — priority queues, backfill, and fair-share": mo.md("""
        **Priority Queue Wait Time** (M/M/1/Priority, non-preemptive):

        For a job class with priority p and offered load ρ_p, the mean wait is:

        > W_p = W_0 / ((1 − σ_{p-1})(1 − σ_p))

        where σ_k = sum of offered loads for all classes with priority ≥ k, and
        W_0 is the mean residual service time (sum over all classes).

        In the limit where high-priority class offered load ρ_1 << 1, the
        high-priority class sees near-zero queue wait regardless of total ρ.

        ---

        **Backfill Algorithm** (Slurm `backfill` plugin, simplified):

        ```
        For each time slot t in [now, horizon]:
            reserved_jobs = jobs with reservations starting before t
            idle_gpus(t) = total_gpus - running_jobs(t) - reserved_jobs(t)

            For each waiting job j (sorted by priority):
                if j.gpus <= idle_gpus(t) AND j.duration <= reservation_gap(t):
                    schedule j at time t
                    break
        ```

        The `duration <= reservation_gap` constraint ensures backfill jobs
        complete before the reserved high-priority job needs those resources.

        ---

        **Fair-Share Decay Function** (Slurm FairShare algorithm):

        > Priority_i(t) = base_weight_i × decay^(past_usage_i / target_share_i)

        where `decay` ∈ (0.5, 1.0) and `past_usage_i` is the GPU-hours consumed
        in the last N days. Users who have used less than their fair share get
        a priority boost; heavy users get a penalty.

        **Starvation prevention**: Slurm's `priority/multifactor` plugin adds a
        *job age* term that grows monotonically with queue time, ensuring no job
        waits indefinitely regardless of fair-share score.

        ---

        **Gang scheduling constraint** (distributed training jobs):

        A distributed training job requiring N nodes can only start when all
        N nodes are simultaneously free. If each node has probability p of being
        free, the probability of all N nodes being free simultaneously is p^N.

        For p = 0.2 (80% utilization): p^4 = 0.0016, p^64 = 10^{-46}.

        This is why large training jobs exhibit superlinear waiting relative to
        the M/M/1 model — the "gang scheduling problem" is a coordination cost
        that worsens exponentially with job size at high utilization.
        """)
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS ─────────────────────────────────────────────────────────
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
                    <strong>1. The utilization trap is mathematical, not operational.</strong>
                    At 95% cluster utilization, M/M/1 queuing theory predicts mean wait
                    = 20x mean job duration. This is not a scheduler bug &mdash; it is
                    physics. Adding GPUs reduces wait only if it substantially lowers
                    utilization below the 80&ndash;85% knee of the curve.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. ML workload variance (C<sub>s</sub> = 3) inflates wait by 5x versus uniform jobs.</strong>
                    The Pollaczek-Khinchine (1 + C<sub>s</sub><sup>2</sup>)/2 term amplifies the
                    base M/M/1 wait from 4x to 20x at 80% utilization. Running an ML cluster
                    at 80% produces the same queue conditions as a uniform-job cluster at 95%.
                    The safe operating point for ML clusters is 60&ndash;70%.
                </div>
                <div>
                    <strong>3. Backfill decouples interactive wait from training job duration.</strong>
                    Simple priority gives short jobs queue position but still forces them to wait
                    for running jobs to finish. Backfill fills the temporal gaps between large
                    reservations with short jobs &mdash; achieving both higher cluster utilization
                    and lower interactive wait simultaneously.
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
                    <strong>Lab 09: Performance Engineering</strong> &mdash; This lab quantified
                    scheduling waste. The next lab asks: once a job is running, what fraction of
                    its GPU-hours are actually doing useful computation? The roofline model and
                    Model FLOP Utilization (MFU) expose how memory bandwidth, not compute,
                    limits real training throughput.
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
                    <strong>Read:</strong> @sec-fleet-orchestration-scheduling-algorithms for
                    the full derivation of the Pollaczek-Khinchine formula and backfill
                    scheduling correctness guarantees.<br/>
                    <strong>Build:</strong> TinyTorch Module 18 &mdash; implement a priority
                    queue scheduler with backfill that passes the interactive-SLA test suite.
                </div>
            </div>

        </div>
        """),

        mo.accordion({


            "Self-Assessment": mo.md("""
**Check your understanding:**

1. At 80% utilization, M/M/1 queuing predicts mean wait of 4x mean job duration for uniform jobs. Why does ML workload variance (Cs = 3) inflate this to 20x via the Pollaczek-Khinchine formula, and what safe utilization range does this imply?
2. An operator pushes cluster utilization to 95% to maximize GPU-hours billed. For a cluster where average jobs take 4 hours, what is the expected queue wait time, and why does this make the cluster effectively unusable for interactive jobs?
3. How does backfill scheduling fill temporal gaps between large reservations with short jobs, and why does this achieve both higher cluster utilization and lower interactive wait simultaneously?

**You're ready to move on if you can:**
- Calculate expected queue wait using the Pollaczek-Khinchine formula for ML workloads with high variance
- Explain why the utilization-latency tradeoff is a mathematical property, not a software limitation
- Describe how backfill scheduling decouples interactive job latency from training job duration
""")


        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# DESIGN LEDGER SAVE + HUD FOOTER
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle, act1_pred, act2_pred,
    wait_inter_min, starvation_pct, eff_util_pct, sla_met,
    rho_total, policy,
):
    _ctx  = context_toggle.value or "fifo"

    # Determine if Act I prediction was correct
    _a1_correct = (act1_pred.value == "c") if act1_pred.value is not None else False

    ledger.save(
        chapter="v2_08",
        design={
            "context":               _ctx,
            "scheduling_policy":     policy if act2_pred.value is not None else "not_set",
            "cluster_utilization":   eff_util_pct,
            "interactive_wait_min":  wait_inter_min,
            "training_starvation_pct": starvation_pct,
            "act1_prediction":       act1_pred.value or "not_set",
            "act1_correct":          _a1_correct,
            "act2_result":           wait_inter_min,
            "act2_decision":         act2_pred.value or "not_set",
            "constraint_hit":        not sla_met or starvation_pct > 20,
            "sla_met":               sla_met,
        },
    )

    # HUD footer
    _ok     = "#4ade80"
    _danger = "#f87171"
    _muted  = "#94a3b8"

    _sla_status  = "MET" if sla_met else "VIOLATED"
    _sla_color   = _ok if sla_met else _danger
    _a1_color    = _ok if _a1_correct else _danger
    _a1_status   = "CORRECT" if _a1_correct else ("WRONG" if act1_pred.value else "---")
    _star_color  = _ok if starvation_pct <= 20 else _danger

    mo.Html(f"""
    <div style="display: flex; gap: 28px; align-items: center; flex-wrap: wrap;
                padding: 14px 24px; background: {COLORS['Surface0']};
                border-radius: 12px; margin-top: 32px; font-family: 'SF Mono', monospace;
                font-size: 0.8rem; border: 1px solid {COLORS['Surface1']};">
        <span style="color: {_muted}; font-weight: 700; letter-spacing: 0.06em;">LEDGER</span>
        <span>
            <span style="color: {_muted};">Lab </span>
            <span style="color: #e2e8f0;">v2_08</span>
        </span>
        <span>
            <span style="color: {_muted};">Context </span>
            <span style="color: #e2e8f0;">{_ctx.upper()}</span>
        </span>
        <span>
            <span style="color: {_muted};">Act I </span>
            <span style="color: {_a1_color};">{_a1_status}</span>
        </span>
        <span>
            <span style="color: {_muted};">Policy </span>
            <span style="color: #e2e8f0;">{policy.upper()}</span>
        </span>
        <span>
            <span style="color: {_muted};">Util </span>
            <span style="color: #e2e8f0;">{eff_util_pct:.1f}%</span>
        </span>
        <span>
            <span style="color: {_muted};">Inter.Wait </span>
            <span style="color: {_sla_color};">{wait_inter_min:.1f}min</span>
        </span>
        <span>
            <span style="color: {_muted};">SLA </span>
            <span style="color: {_sla_color};">{_sla_status}</span>
        </span>
        <span>
            <span style="color: {_muted};">Starvation </span>
            <span style="color: {_star_color};">{starvation_pct:.1f}%</span>
        </span>
    </div>
    """)
    return




if __name__ == "__main__":
    app.run()
