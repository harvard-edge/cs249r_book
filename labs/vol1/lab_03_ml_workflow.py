import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 03: THE SILENT DEGRADATION LOOP
#
# Chapter: ml_workflow.qmd  (@sec-ml-workflow)
# Core Invariant: The Constraint Propagation Principle (2^(N-1) cost escalation)
#                 + Iteration velocity race (fast cycles outcompete slow ones)
#
# Act I  — The Constraint Tax (12-15 min)
#   Stakeholder message → prediction lock → lifecycle timeline instrument →
#   prediction-vs-reality reveal → structured reflection → MathPeek
#
# Act II — The Iteration Velocity Race (20-25 min)
#   Context toggle → prediction lock → dual-model accuracy plot →
#   constraint gate designer → failure state → reflection → MathPeek
#
# Design Ledger: saves chapter=3 with model_size, cycle_hours,
#                crossover_week, constraint_gates, discovery_stage.
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP ────────────────────────────────────────────────────────────
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
    from labs.core.components import StakeholderMessage, MathPeek

    ledger = DesignLedger()

    # ── Chapter-sourced constants (ml_workflow.qmd) ───────────────────────────
    # 2^(N-1) Constraint Propagation Principle — @sec-ml-workflow line 73
    # "the Constraint Propagation Principle (2^(N-1) cost escalation)"
    STAGES = ["Requirements", "Data", "Modeling", "Evaluation", "Deployment", "Monitoring"]
    STAGE_DAYS = [5, 30, 60, 40, 10, 5]   # Rural Clinic timeline (150 day total)
    # Rural Clinic: 95% accuracy Day 90, 96% Day 120, failure Day 151
    # Source: ml_workflow.qmd line 84
    RURAL_CLINIC_TOTAL_DAYS = 150
    # Large model: 95% start, +0.15%/iter, 1-week cycle — lines 312-314
    LARGE_START_ACC  = 95.0
    LARGE_GAIN_ITER  = 0.15
    LARGE_CYCLE_HRS  = 168    # 1 week = 168 hours
    # Small model: 90% start, +0.1%/iter, 1-hour cycle — lines 317-319
    SMALL_START_ACC  = 90.0
    SMALL_GAIN_ITER  = 0.10
    SMALL_CYCLE_HRS  = 1
    # 26-week project window — line 308
    WEEKS_TOTAL      = 26
    # Accuracy ceiling at 99% — lines 325, 329
    ACC_CEILING      = 99.0
    # 60-80% time on data activities — line 201
    DATA_TIME_LOW    = 60
    DATA_TIME_HIGH   = 80
    # 10-20% on model development — line 201
    MODEL_TIME_LOW   = 10
    MODEL_TIME_HIGH  = 20

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        StakeholderMessage, MathPeek,
        go, np, math,
        STAGES, STAGE_DAYS, RURAL_CLINIC_TOTAL_DAYS,
        LARGE_START_ACC, LARGE_GAIN_ITER, LARGE_CYCLE_HRS,
        SMALL_START_ACC, SMALL_GAIN_ITER, SMALL_CYCLE_HRS,
        WEEKS_TOTAL, ACC_CEILING,
        DATA_TIME_LOW, DATA_TIME_HIGH,
        MODEL_TIME_LOW, MODEL_TIME_HIGH,
    )


# ─── CELL 1: HEADER ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1a2744 60%, #0f172a 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume I · Lab 03
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.3rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Constraint Tax
            </h1>
            <p style="margin: 0 0 22px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 640px; line-height: 1.65;">
                A team achieves 96% accuracy on Day 120. On Day 151, they discover the
                deployment target has 512 MB of memory and the model needs 4 GB.
                Five months of work is discarded. This lab quantifies why — and how to
                prevent it.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Chapter: ML Workflow
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min · 2 Acts
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Prerequisite: @sec-ml-workflow
                </span>
            </div>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span class="badge badge-info">Constraint Propagation: 2^(N-1)</span>
                <span class="badge badge-info">Iteration Tax</span>
                <span class="badge badge-warn">Cloud vs Mobile Deployment</span>
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

    - **@sec-ml-workflow-understanding-ml-lifecycle-ca87** — The six-stage ML lifecycle
      and the Rural Clinic case study (Day 90 through Day 153)
    - **@sec-ml-workflow-quantifying-ml-lifecycle-bd69** — Time allocation data: why
      60–80% of project time goes to data activities, not modeling
    - **@sec-ml-workflow-integrating-systems-thinking-principles-24c0** — The Constraint
      Propagation Principle ($2^{N-1}$ cost escalation) and the Iteration Tax
    """), kind="info")
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT I: THE CONSTRAINT TAX
# Pedagogical Goal: Students overestimate the cost of modeling and underestimate
# the cost of late constraint discovery. The 2^(N-1) formula is the surprise.
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.vstack([
        mo.Html(f"""
        <div style="display: flex; align-items: center; gap: 12px; margin: 28px 0 8px 0;">
            <div style="background: {COLORS['BlueLine']}; color: white; border-radius: 50%;
                        width: 28px; height: 28px; display: flex; align-items: center;
                        justify-content: center; font-weight: 800; font-size: 0.9rem;
                        flex-shrink: 0;">I</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #0f172a;
                        letter-spacing: -0.01em;">Act I — The Constraint Tax</div>
            <div style="flex: 1; height: 1px; background: #e2e8f0;"></div>
            <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;
                        white-space: nowrap;">12–15 min</div>
        </div>
        """),
        mo.Html(f"""
        <div style="border-left: 4px solid {COLORS['BlueLine']};
                    background: {COLORS['BlueLL']};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 4px 0 16px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message · ML Project Lead, Rural Health Initiative
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "Day 90: we hit 95% accuracy. Day 120: 96% — better than any published
                benchmark on this dataset. Day 151: we handed the model to deployment.
                Day 152: they checked the tablets in the field clinics. 512 MB of RAM.
                Our model needs 4 GB. We have to start over. Five months. Gone."
            </div>
        </div>
        """),
    ])
    return


# ─── ACT I PREDICTION LOCK ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A) About 5× more expensive — roughly proportional to project progress": "A",
            "B) About 10× more expensive — late changes always cost more, but double-digits cap it": "B",
            "C) About 16× more expensive — each stage compounds the rework required": "C",
            "D) About 100× more expensive — exponential blowup dominates by deployment": "D",
        },
        label=(
            "**Before touching any instrument, commit to your prediction.**\n\n"
            "An ML pipeline has 6 stages: Requirements (1), Data (2), Modeling (3), "
            "Evaluation (4), Deployment (5), Monitoring (6). The Rural Clinic team "
            "discovers a memory constraint at Stage 5 (Deployment) — 150 days into the "
            "project. Compared to discovering it at Stage 1 (Requirements), how much "
            "more expensive is it to fix?"
        ),
    )
    act1_prediction
    return (act1_prediction,)


@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(
        act1_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the constraint timeline instrument."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(
            "**Prediction recorded.** Now use the instrument below to explore "
            "how constraint discovery stage affects total rework cost. "
            "Then compare your prediction to the actual formula."
        ),
        kind="info",
    )
    return


# ─── ACT I INSTRUMENT: CONSTRAINT PROPAGATION TIMELINE ───────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(act1_prediction.value is None)

    act1_discovery_stage = mo.ui.slider(
        start=1, stop=6, value=5, step=1,
        label="Constraint Discovery Stage (drag left to discover earlier)",
        show_value=True,
    )
    act1_data_guess = mo.ui.slider(
        start=5, stop=95, value=40, step=5,
        label="Your estimate: % of ML project time spent on DATA activities",
        show_value=True,
    )
    mo.vstack([
        mo.md("### Constraint Propagation Timeline"),
        mo.md(
            "The Rural Clinic team discovered their 512 MB memory constraint at Stage 5 "
            "(Deployment). Drag the slider left to see how the cost changes if they had "
            "discovered it earlier. Each stage crossed compounds the rework required."
        ),
        act1_discovery_stage,
        mo.md("---"),
        mo.md("### Time Allocation Reality Check"),
        mo.md(
            "Before seeing the data: what percentage of a typical ML project's time "
            "do you think goes to data activities (collection, cleaning, labeling, validation)?"
        ),
        act1_data_guess,
    ])
    return (act1_discovery_stage, act1_data_guess)


@app.cell(hide_code=True)
def _(
    mo, go, np, act1_prediction, act1_discovery_stage, act1_data_guess,
    COLORS, apply_plotly_theme,
    STAGES, STAGE_DAYS, RURAL_CLINIC_TOTAL_DAYS,
    DATA_TIME_LOW, DATA_TIME_HIGH,
):
    mo.stop(act1_prediction.value is None)

    _disc_stage = act1_discovery_stage.value   # 1-indexed
    _cost_mult = 2 ** (_disc_stage - 1)        # 2^(N-1) formula
    _wasted_days = sum(STAGE_DAYS[_disc_stage - 1:])

    # ── Lifecycle Bar Chart ────────────────────────────────────────────────────
    _bar_colors = []
    for _i in range(len(STAGES)):
        if _i < _disc_stage - 1:
            _bar_colors.append(COLORS["GreenLine"])    # completed and kept
        elif _i == _disc_stage - 1:
            _bar_colors.append(COLORS["OrangeLine"])   # discovery point
        else:
            _bar_colors.append(COLORS["RedLine"])      # wasted / must redo

    _fig1 = go.Figure()
    _fig1.add_trace(go.Bar(
        x=STAGES,
        y=STAGE_DAYS,
        marker_color=_bar_colors,
        text=[f"{d}d" for d in STAGE_DAYS],
        textposition="outside",
        name="Stage Duration (days)",
        width=0.55,
    ))
    _fig1.update_layout(
        title=dict(
            text=(
                f"Rural Clinic Timeline — Constraint discovered at Stage {_disc_stage}: "
                f"{STAGES[_disc_stage - 1]}"
            ),
            font=dict(size=13),
        ),
        yaxis=dict(title="Duration (person-days)", range=[0, 75]),
        xaxis=dict(title="Lifecycle Stage"),
        height=340,
        showlegend=False,
        margin=dict(l=50, r=20, t=55, b=50),
        annotations=[
            dict(
                x=STAGES[_disc_stage - 1],
                y=STAGE_DAYS[_disc_stage - 1] + 6,
                text=f"Constraint<br>found here",
                showarrow=True,
                arrowhead=2,
                arrowcolor=COLORS["OrangeLine"],
                font=dict(size=10, color=COLORS["OrangeLine"]),
                ax=0, ay=-30,
            )
        ] if _disc_stage <= len(STAGES) else [],
    )
    apply_plotly_theme(_fig1)

    # ── Time Allocation Pie Chart ──────────────────────────────────────────────
    _student_data_pct  = act1_data_guess.value
    _student_model_pct = max(0, 100 - _student_data_pct - 10)
    _student_other_pct = 100 - _student_data_pct - _student_model_pct

    _actual_data_pct  = 70   # midpoint of 60-80%
    _actual_model_pct = 15   # midpoint of 10-20%
    _actual_other_pct = 15

    _fig2 = go.Figure()
    _fig2.add_trace(go.Pie(
        labels=["Data activities", "Model development", "Deployment & monitoring"],
        values=[_actual_data_pct, _actual_model_pct, _actual_other_pct],
        marker=dict(colors=[COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"]]),
        hole=0.38,
        textinfo="label+percent",
        textfont=dict(size=11),
        name="Actual",
        domain={"x": [0.5, 1.0]},
        title=dict(text="Actual", font=dict(size=11)),
    ))
    _fig2.add_trace(go.Pie(
        labels=["Data activities", "Model development", "Deployment & monitoring"],
        values=[_student_data_pct, _student_model_pct, _student_other_pct],
        marker=dict(colors=[COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"]]),
        hole=0.38,
        textinfo="label+percent",
        textfont=dict(size=11),
        name="Your estimate",
        domain={"x": [0.0, 0.5]},
        title=dict(text="Your Estimate", font=dict(size=11)),
    ))
    _fig2.update_layout(
        title=dict(text="Time Allocation: Your Estimate vs. Chapter Data", font=dict(size=13)),
        height=320,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    apply_plotly_theme(_fig2)

    # ── Metric cards ──────────────────────────────────────────────────────────
    _cost_color = (
        COLORS["GreenLine"] if _cost_mult <= 2 else
        COLORS["OrangeLine"] if _cost_mult <= 8 else
        COLORS["RedLine"]
    )
    _waste_pct = int(100 * _wasted_days / RURAL_CLINIC_TOTAL_DAYS)

    _cards_html = f"""
    <div style="display: flex; gap: 16px; justify-content: flex-start;
                margin: 16px 0; flex-wrap: wrap;">
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0;
                    border-radius: 10px; min-width: 170px; background: white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
            <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em;">
                Cost Multiplier
            </div>
            <div style="font-size: 2.4rem; font-weight: 800;
                        color: {_cost_color}; line-height: 1.1; margin-top: 4px;">
                {_cost_mult}×
            </div>
            <div style="color: #64748b; font-size: 0.78rem; margin-top: 4px;">
                2^({_disc_stage}−1) = 2^{_disc_stage - 1}
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0;
                    border-radius: 10px; min-width: 170px; background: white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
            <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em;">
                Wasted Days
            </div>
            <div style="font-size: 2.4rem; font-weight: 800;
                        color: {'#CB202D' if _wasted_days > 60 else '#CC5500' if _wasted_days > 20 else '#008F45'};
                        line-height: 1.1; margin-top: 4px;">
                {_wasted_days}d
            </div>
            <div style="color: #64748b; font-size: 0.78rem; margin-top: 4px;">
                {_waste_pct}% of {RURAL_CLINIC_TOTAL_DAYS}-day project
            </div>
        </div>
        <div style="padding: 18px 22px; border: 1px solid #e2e8f0;
                    border-radius: 10px; min-width: 170px; background: white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
            <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em;">
                Stages to Redo
            </div>
            <div style="font-size: 2.4rem; font-weight: 800;
                        color: {'#CB202D' if _disc_stage >= 4 else '#CC5500' if _disc_stage >= 3 else '#008F45'};
                        line-height: 1.1; margin-top: 4px;">
                {max(0, len(STAGES) - _disc_stage + 1)}
            </div>
            <div style="color: #64748b; font-size: 0.78rem; margin-top: 4px;">
                Stages {_disc_stage}–{len(STAGES)} must be revisited
            </div>
        </div>
    </div>
    """

    _physics_text = f"""
    **Constraint Propagation Physics:**

    ```
    Stage of discovery (N) = {_disc_stage}  ({STAGES[_disc_stage - 1]})
    Cost multiplier        = 2^(N-1) = 2^({_disc_stage}-1) = {_cost_mult}×
    Wasted person-days     = {_wasted_days}  ({_waste_pct}% of total project)
    ```

    At Stage 1 (Requirements): 2^0 = **1× cost** — the constraint shapes all work before it begins.
    At Stage {_disc_stage} ({STAGES[_disc_stage - 1]}): 2^{_disc_stage - 1} = **{_cost_mult}× cost** —
    {_wasted_days} person-days must be redone or discarded.
    At Stage 6 (Monitoring): 2^5 = **32× cost** — post-deployment remediation.
    """

    mo.vstack([
        mo.Html(_cards_html),
        mo.md(_physics_text),
        mo.as_html(_fig1),
        mo.md("---"),
        mo.as_html(_fig2),
        mo.md(
            f"The chapter reports that data activities consume **{DATA_TIME_LOW}–{DATA_TIME_HIGH}%** "
            "of ML project time. Model development is **10–20%**. "
            f"You estimated **{_student_data_pct}%** for data."
        ),
    ])
    return


# ─── ACT I REVEAL ─────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_discovery_stage):
    mo.stop(act1_prediction.value is None)

    _predicted_map = {"A": 5, "B": 10, "C": 16, "D": 100}
    _predicted = _predicted_map[act1_prediction.value]
    _actual = 16  # Stage 5 discovery → 2^(5-1) = 16

    _ratio = _actual / _predicted
    _is_correct = act1_prediction.value == "C"

    if _is_correct:
        _reveal_msg = (
            f"**You predicted {_predicted}×. The actual value for Stage 5 discovery is "
            f"16× (2^{{5−1}} = 2^4 = 16). Correct.** "
            "The formula is 2^(N−1): each stage crossed means its output becomes input to "
            "the next stage. Discarding at Stage 5 invalidates all modeling, evaluation, "
            "and deployment work — not just the stage where the failure was found. "
            "The Rural Clinic team spent 150 person-days before discovering the 512 MB constraint. "
            "A 1-day requirements checklist at Stage 1 would have cost 1×."
        )
        _kind = "success"
    elif act1_prediction.value == "D":
        _reveal_msg = (
            f"**You predicted {_predicted}×. The actual Stage 5 value is 16×.** "
            f"You were off by {_predicted / _actual:.1f}× in the other direction — "
            "the cost is large but not 100×. The 100× figure is Boehm's original estimate "
            "for post-deployment bugs in *traditional* software. In ML, Stage 6 (Monitoring/post-deploy) "
            "discovery reaches ~32× (2^5). The 16× figure applies specifically to Stage 5 (Deployment). "
            "The principle is right; the stage assignment matters."
        )
        _kind = "warn"
    else:
        _gap = _actual / _predicted
        _reveal_msg = (
            f"**You predicted {_predicted}×. The actual value is 16× (2^{{5−1}}).** "
            f"You were off by {_gap:.1f}×. "
            "The compounding is what surprises most engineers: it is not that late fixes cost "
            "more in absolute terms — it is that each stage's output becomes the *input* "
            "to the next. Invalidating Stage 5 requires revisiting Stages 3, 4, and 5 at minimum. "
            "At Stage 5, that is 4 compounding factors: 2^4 = 16×."
        )
        _kind = "warn"

    mo.callout(mo.md(_reveal_msg), kind=_kind)
    return


# ─── ACT I REFLECTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(act1_prediction.value is None)

    act1_reflection = mo.ui.radio(
        options={
            "A) 96% accuracy is insufficient for medical AI — they should have aimed for 99%": "A",
            "B) The team used the wrong model architecture and needed to retrain from scratch": "B",
            "C) The memory constraint was never recorded at Stage 1, so modeling aimed at the wrong target": "C",
            "D) The data labeling was incomplete, causing distribution shift at deployment": "D",
        },
        label=(
            "**Reflection.** The Rural Clinic team achieved 96% accuracy. "
            "Why was the project a failure?"
        ),
    )
    act1_reflection
    return (act1_reflection,)


@app.cell(hide_code=True)
def _(mo, act1_prediction, act1_reflection):
    mo.stop(act1_prediction.value is None)
    mo.stop(
        act1_reflection.value is None,
        mo.callout(mo.md("Select an answer to continue to Act II."), kind="warn"),
    )

    _correct = act1_reflection.value == "C"
    _feedback_map = {
        "A": (
            "**Not quite.** The failure had nothing to do with accuracy. "
            "The model achieved 96% — well above typical clinical benchmarks. "
            "The problem was that 150 days of engineering effort optimized a metric "
            "that was irrelevant to whether the model could actually run on the deployment hardware. "
            "Accuracy only matters after the physical constraints are satisfied."
        ),
        "B": (
            "**Not quite.** Architecture was not the failure mode. "
            "The team's modeling decisions were technically competent — they improved accuracy "
            "from 95% to 96% over the final month. The failure was upstream: "
            "the deployment target's memory budget was never recorded as a Stage 1 constraint. "
            "No architecture change can fix a workflow failure."
        ),
        "C": (
            "**Correct.** The memory constraint (512 MB tablet) should have been the first "
            "entry in the requirements document. Instead it was discovered 150 days later — "
            "after every downstream decision was already committed to a 4 GB model. "
            "This is what the Constraint Propagation Principle (2^(N-1)) formalizes: "
            "late constraint discovery multiplies rework across every preceding stage. "
            "High accuracy is not a substitute for correct problem definition."
        ),
        "D": (
            "**Not quite.** There is no evidence of data quality issues in this case study. "
            "The model performed well at 96% — suggesting the data was sufficient. "
            "Distribution shift is a real failure mode (the focus of Lab 14, @sec-ml-operations), "
            "but it was not the mechanism here. The failure was an absent hardware constraint "
            "at Stage 1."
        ),
    }

    mo.vstack([
        act1_reflection,
        mo.callout(
            mo.md(_feedback_map[act1_reflection.value]),
            kind="success" if _correct else "warn",
        ),
    ])
    return


# ─── ACT I MATHPEEK ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(act1_reflection.value is None)

    mo.accordion({
        "The Constraint Propagation Principle — governing equation": mo.md("""
        **Formula:**

        $$\\text{Cost}(N) = 2^{N-1} \\quad N = \\text{stage of constraint discovery (1–6)}$$

        **Stage values:**

        | Stage | Name | Cost Multiplier | Rural Clinic |
        |-------|------|-----------------|-------------|
        | 1 | Requirements | 2⁰ = **1×** | 1 day to record the 512 MB constraint |
        | 2 | Data | 2¹ = **2×** | Data re-collection may be needed |
        | 3 | Modeling | 2² = **4×** | Architecture + training must restart |
        | 4 | Evaluation | 2³ = **8×** | Eval + modeling restart |
        | 5 | Deployment | 2⁴ = **16×** | 150 days wasted (the Rural Clinic case) |
        | 6 | Monitoring | 2⁵ = **32×** | Post-launch remediation |

        **Note on the formula:** The 2^(N-1) model is a pedagogical approximation, not
        a precise physical law. Boehm's original empirical data for traditional software
        showed post-deployment fixes cost up to **100×** what requirements-phase fixes cost.
        For ML, late-stage constraint violations are often *worse* than traditional software
        because they invalidate learned model weights — there is no "undo" for 150 days of
        gradient descent.

        **Stage interface contracts** (from @tbl-stage-interface in the chapter) are the
        engineering mechanism that enforces this: each stage's output contract must satisfy
        the next stage's input contract. A violated contract discovered at Stage N costs
        $2^{N-1}$ times what a Stage 1 violation costs.
        """),
    })
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II: THE ITERATION VELOCITY RACE
# Pedagogical Goal: Students believe starting with the highest-accuracy model
# is the right strategy. The chapter's calculation shows that a model starting
# 5 percentage points behind but cycling 168× faster overtakes the leader
# within the 26-week project window.
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act1_reflection, COLORS):
    mo.stop(act1_reflection.value is None)

    mo.vstack([
        mo.Html(f"""
        <div style="display: flex; align-items: center; gap: 12px; margin: 28px 0 8px 0;">
            <div style="background: {COLORS['OrangeLine']}; color: white; border-radius: 50%;
                        width: 28px; height: 28px; display: flex; align-items: center;
                        justify-content: center; font-weight: 800; font-size: 0.9rem;
                        flex-shrink: 0;">II</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #0f172a;
                        letter-spacing: -0.01em;">Act II — The Iteration Velocity Race</div>
            <div style="flex: 1; height: 1px; background: #e2e8f0;"></div>
            <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;
                        white-space: nowrap;">20–25 min</div>
        </div>
        """),
        mo.md(
            "The Rural Clinic project failed not just because of a missing constraint — "
            "it also suffered from a strategic mistake: the team chose the largest, "
            "highest-accuracy model and iterated slowly. "
            "The chapter introduces the **Iteration Tax**: a fast, smaller model can "
            "accumulate thousands of learning cycles in the time a large model completes "
            "a handful. The question is when — and whether — the fast model catches up."
        ),
    ])
    return


# ─── CONTEXT TOGGLE ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflection, COLORS):
    mo.stop(act1_reflection.value is None)

    act2_context = mo.ui.radio(
        options={
            "Cloud Training Node (H100, 700W, $3/hr compute)": "cloud",
            "Mobile On-Device (Smartphone NPU, 5W sustained, 2hr battery)": "mobile",
        },
        value="Cloud Training Node (H100, 700W, $3/hr compute)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.Html(f"""
        <div style="background: white; border: 1px solid {COLORS['Border']};
                    border-radius: 10px; padding: 16px 20px; margin: 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">
                Deployment Context
            </div>
        """),
        act2_context,
        mo.Html("</div>"),
    ])
    return (act2_context,)


# ─── ACT II PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(act1_reflection.value is None)

    act2_prediction = mo.ui.radio(
        options={
            "A) Always choose the large model — higher starting accuracy means better final accuracy": "A",
            "B) Always choose the small model — faster cycles always win over any window": "B",
            "C) It depends on the project window and drift rate — there is a crossover point": "C",
            "D) Neither — deploy once and never retrain to minimize compute cost": "D",
        },
        label=(
            "**Before using the instruments, commit to your prediction.**\n\n"
            "A large model starts at 95% accuracy and gains 0.15% per iteration, "
            "with a 1-week training cycle. A small model starts at 90% accuracy "
            "and gains 0.1% per iteration, with a 1-hour training cycle. "
            "In a 26-week project window, which strategy produces the better final model?"
        ),
    )
    act2_prediction
    return (act2_prediction,)


@app.cell(hide_code=True)
def _(mo, act1_reflection, act2_prediction):
    mo.stop(act1_reflection.value is None)
    mo.stop(
        act2_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the iteration velocity instruments."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(
            "**Prediction recorded.** Adjust the sliders to explore the velocity race. "
            "Find the crossover week — and watch what happens when you push the small model "
            "to hourly cycles in the mobile context."
        ),
        kind="info",
    )
    return


# ─── ACT II INSTRUMENTS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction, act1_reflection):
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)

    act2_large_cycle = mo.ui.dropdown(
        options={"1 hour": 1, "24 hours (1 day)": 24, "168 hours (1 week)": 168, "336 hours (2 weeks)": 336},
        value="168 hours (1 week)",
        label="Large model cycle time",
    )
    act2_small_cycle = mo.ui.dropdown(
        options={"1 hour": 1, "4 hours": 4, "24 hours (1 day)": 24, "168 hours (1 week)": 168},
        value="1 hour",
        label="Small model cycle time",
    )
    act2_decay_rate = mo.ui.dropdown(
        options={"Fast saturation (r=0.90)": 0.90, "Moderate saturation (r=0.95)": 0.95, "Slow saturation (r=0.99)": 0.99},
        value="Moderate saturation (r=0.95)",
        label="Gain decay rate per iteration",
    )
    act2_project_weeks = mo.ui.slider(
        start=4, stop=52, value=26, step=2,
        label="Project window (weeks)",
        show_value=True,
    )

    # Constraint gate toggles
    act2_gate_memory  = mo.ui.checkbox(value=False, label="Stage 1: Hardware memory budget recorded")
    act2_gate_latency = mo.ui.checkbox(value=False, label="Stage 1: Latency budget recorded")
    act2_gate_schema  = mo.ui.checkbox(value=False, label="Stage 2: Data schema validation active")
    act2_gate_profile = mo.ui.checkbox(value=False, label="Stage 3: Memory profiling at batch=1")
    act2_gate_ondev   = mo.ui.checkbox(value=False, label="Stage 4: On-device test with production hardware")
    act2_gate_rollout = mo.ui.checkbox(value=False, label="Stage 5: Staged rollout with monitoring")

    mo.vstack([
        mo.md("### Iteration Velocity Controls"),
        mo.hstack([
            mo.vstack([act2_large_cycle, act2_small_cycle]),
            mo.vstack([act2_decay_rate, act2_project_weeks]),
        ], justify="start", gap=3),
        mo.md("---"),
        mo.md("### Constraint Gate Designer"),
        mo.md(
            "Toggle these gates on or off to see how the Rural Clinic failure "
            "appears (or disappears) on the project timeline. Each gate costs "
            "~1 person-day to implement at the listed stage."
        ),
        mo.vstack([
            act2_gate_memory,
            act2_gate_latency,
            act2_gate_schema,
            act2_gate_profile,
            act2_gate_ondev,
            act2_gate_rollout,
        ]),
    ])
    return (
        act2_large_cycle, act2_small_cycle, act2_decay_rate, act2_project_weeks,
        act2_gate_memory, act2_gate_latency, act2_gate_schema,
        act2_gate_profile, act2_gate_ondev, act2_gate_rollout,
    )


@app.cell(hide_code=True)
def _(
    mo, go, np, act2_prediction, act1_reflection, act2_context,
    act2_large_cycle, act2_small_cycle, act2_decay_rate, act2_project_weeks,
    act2_gate_memory, act2_gate_latency, act2_gate_schema,
    act2_gate_profile, act2_gate_ondev, act2_gate_rollout,
    COLORS, apply_plotly_theme,
    LARGE_START_ACC, LARGE_GAIN_ITER,
    SMALL_START_ACC, SMALL_GAIN_ITER,
    ACC_CEILING,
):
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)

    _ctx       = act2_context.value
    _lc_hrs    = act2_large_cycle.value
    _sc_hrs    = act2_small_cycle.value
    _r         = act2_decay_rate.value
    _weeks     = act2_project_weeks.value
    _total_hrs = _weeks * 168   # 168 hours per week

    # ── Mobile context: feasibility check ─────────────────────────────────────
    # Mobile: 5W sustained, 2hr battery = 10 Wh total capacity
    # On-device retraining: 4.5W average power during training (90% of 5W)
    # Source: PROTOCOL.md Mobile context definition
    MOBILE_BATTERY_WH  = 10.0   # 2 hr × 5 W = 10 Wh
    MOBILE_RETRAIN_W   = 4.5    # 90% of 5W sustained for retraining
    _retrain_hrs_day   = 24 / _sc_hrs  # retraining sessions per day
    _power_wh_day      = _retrain_hrs_day * MOBILE_RETRAIN_W * (_sc_hrs / 1.0)
    # Each session uses (_sc_hrs * MOBILE_RETRAIN_W) Wh, but a full retrain
    # on mobile takes at least 0.5 hr minimum regardless of cycle time requested
    _session_wh        = max(0.5, _sc_hrs) * MOBILE_RETRAIN_W
    _sessions_per_day  = 24 / max(0.5, _sc_hrs)
    _total_wh_per_day  = _session_wh * _sessions_per_day

    _mobile_infeasible = (_ctx == "mobile" and _sc_hrs == 1)

    # ── Compute iteration counts ───────────────────────────────────────────────
    _large_iters = int(_total_hrs / _lc_hrs)
    _small_iters = int(_total_hrs / max(1, _sc_hrs))

    # ── Saturating accuracy curves ─────────────────────────────────────────────
    # Gain decays: Delta_a_n = Delta_a_0 * r^n
    # Model accuracy at iteration k: A_k = A_0 + sum_{n=0}^{k-1} gain * r^n
    # = A_0 + gain * (1 - r^k) / (1 - r)   [geometric series]
    # Capped at ACC_CEILING = 99.0
    def _saturating_acc(start, gain_per_iter, r, n_iters):
        if r >= 1.0 or n_iters == 0:
            return min(start + gain_per_iter * n_iters, ACC_CEILING)
        total_gain = gain_per_iter * (1 - r**n_iters) / (1 - r)
        return min(start + total_gain, ACC_CEILING)

    # Weekly accuracy points for both models
    _week_arr = np.arange(0, _weeks + 1, dtype=float)
    _large_acc_arr = np.array([
        _saturating_acc(
            LARGE_START_ACC, LARGE_GAIN_ITER, _r,
            int(w * 168 / _lc_hrs)
        ) for w in _week_arr
    ])
    _small_acc_arr = np.array([
        _saturating_acc(
            SMALL_START_ACC, SMALL_GAIN_ITER, _r,
            int(w * 168 / max(1, _sc_hrs))
        ) for w in _week_arr
    ])

    # Final accuracies
    _large_final = _large_acc_arr[-1]
    _small_final = _small_acc_arr[-1]

    # Find crossover week (first week where small >= large)
    _crossover_week = None
    for _wi in range(len(_week_arr)):
        if _small_acc_arr[_wi] >= _large_acc_arr[_wi]:
            _crossover_week = _week_arr[_wi]
            break

    # ── Dual-line accuracy plot ────────────────────────────────────────────────
    _fig = go.Figure()

    _fig.add_trace(go.Scatter(
        x=_week_arr, y=_large_acc_arr,
        mode="lines", name=f"Large model ({_lc_hrs}h cycle, starts {LARGE_START_ACC}%)",
        line=dict(color=COLORS["BlueLine"], width=2.5),
    ))
    _fig.add_trace(go.Scatter(
        x=_week_arr, y=_small_acc_arr,
        mode="lines", name=f"Small model ({_sc_hrs}h cycle, starts {SMALL_START_ACC}%)",
        line=dict(color=COLORS["GreenLine"], width=2.5, dash="dash"),
    ))

    # Week 20 deadline reference
    if _weeks >= 20:
        _fig.add_vline(
            x=20,
            line_color=COLORS["OrangeLine"],
            line_dash="dot",
            annotation_text="Week 20 target",
            annotation_font_color=COLORS["OrangeLine"],
        )

    # Crossover annotation
    if _crossover_week is not None:
        _fig.add_vline(
            x=_crossover_week,
            line_color=COLORS["GreenLine"],
            line_dash="longdash",
            annotation_text=f"Crossover: Week {_crossover_week:.0f}",
            annotation_font_color=COLORS["GreenLine"],
            annotation_position="top left",
        )

    _fig.update_layout(
        title=dict(
            text=(
                f"Iteration Velocity Race — {_large_iters} large-model iterations "
                f"vs. {_small_iters:,} small-model iterations over {_weeks} weeks"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title="Project Week", range=[0, _weeks]),
        yaxis=dict(title="Model Accuracy (%)", range=[85, 100]),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=70, b=50),
    )
    apply_plotly_theme(_fig)

    # ── Gate status summary ────────────────────────────────────────────────────
    _gates_active = [
        act2_gate_memory.value,
        act2_gate_latency.value,
        act2_gate_schema.value,
        act2_gate_profile.value,
        act2_gate_ondev.value,
        act2_gate_rollout.value,
    ]
    _n_gates = sum(_gates_active)
    _memory_gate_on = act2_gate_memory.value

    # Rural Clinic failure is prevented only if memory gate is active
    _failure_prevented = _memory_gate_on
    _failure_cost_mult = 16 if not _failure_prevented else 1
    _failure_days_lost = 150 if not _failure_prevented else 0

    _gate_names = [
        "Memory budget", "Latency budget", "Schema validation",
        "Memory profiling", "On-device test", "Staged rollout",
    ]
    _gate_rows = "".join([
        f"""<tr>
            <td style="padding: 5px 10px; font-size:0.85rem; color:#475569;">{_gate_names[i]}</td>
            <td style="padding: 5px 10px; text-align:center;">
                {"<span style='color:#008F45; font-weight:700;'>ACTIVE</span>"
                 if _gates_active[i] else
                 "<span style='color:#CB202D; font-weight:700;'>OFF</span>"}
            </td>
        </tr>"""
        for i in range(len(_gate_names))
    ])

    _gate_html = f"""
    <div style="border: 1px solid {'#008F45' if _failure_prevented else '#CB202D'};
                border-radius: 10px; padding: 14px 18px; margin: 10px 0;
                background: {'#ECFDF5' if _failure_prevented else '#FEF2F2'};">
        <div style="font-weight:800; font-size:0.9rem;
                    color:{'#008F45' if _failure_prevented else '#CB202D'}; margin-bottom:8px;">
            {'Rural Clinic failure PREVENTED' if _failure_prevented
             else 'Rural Clinic failure IN PROGRESS (Day 150 risk)'}
        </div>
        <div style="font-size:0.82rem; color:#475569; margin-bottom:10px;">
            {'Memory budget gate is ACTIVE. The 512 MB constraint propagates to Stage 1. '
             'No wasted work.'
             if _failure_prevented else
             'Memory budget gate is OFF. The 512 MB constraint will not be discovered '
             'until Stage 5 (Day 150). Cost: 16×.'}
        </div>
        <table style="width:100%; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #e2e8f0;">
                <th style="text-align:left; padding:5px 10px; font-size:0.78rem;
                           color:#94a3b8; text-transform:uppercase;">Gate</th>
                <th style="text-align:center; padding:5px 10px; font-size:0.78rem;
                           color:#94a3b8; text-transform:uppercase;">Status</th>
            </tr>
            {_gate_rows}
        </table>
        <div style="margin-top:10px; font-size:0.82rem; font-weight:600;
                    color:{'#008F45' if _failure_prevented else '#CB202D'};">
            Gates active: {_n_gates}/6 &nbsp;|&nbsp;
            {'Cost multiplier: 1× (Stage 1 discovery)' if _failure_prevented
             else f'Cost multiplier: {_failure_cost_mult}× (Stage 5 discovery)'}
            &nbsp;|&nbsp;
            {'Days at risk: 0' if _failure_prevented
             else f'Days at risk: {_failure_days_lost}'}
        </div>
    </div>
    """

    # ── Metric cards ──────────────────────────────────────────────────────────
    _crossover_str = f"Week {_crossover_week:.0f}" if _crossover_week is not None else "Never"
    _card_color_large = COLORS["GreenLine"] if _large_final >= 97 else COLORS["OrangeLine"]
    _card_color_small = COLORS["GreenLine"] if _small_final >= 97 else COLORS["OrangeLine"]

    _metrics_html = f"""
    <div style="display: flex; gap: 14px; flex-wrap: wrap; margin: 12px 0;">
        <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 150px; background: white;">
            <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase;">Large Model Final</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {_card_color_large};
                        line-height: 1.1; margin-top: 4px;">{_large_final:.1f}%</div>
            <div style="color: #64748b; font-size: 0.75rem;">{_large_iters} iterations</div>
        </div>
        <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 150px; background: white;">
            <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase;">Small Model Final</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {_card_color_small};
                        line-height: 1.1; margin-top: 4px;">{_small_final:.1f}%</div>
            <div style="color: #64748b; font-size: 0.75rem;">{_small_iters:,} iterations</div>
        </div>
        <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 150px; background: white;">
            <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase;">Crossover</div>
            <div style="font-size: 2.2rem; font-weight: 800;
                        color: {'#008F45' if _crossover_week is not None else '#94a3b8'};
                        line-height: 1.1; margin-top: 4px;">{_crossover_str}</div>
            <div style="color: #64748b; font-size: 0.75rem;">
                {'small model overtakes large' if _crossover_week else 'large model stays ahead'}
            </div>
        </div>
        <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: 150px; background: white;">
            <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase;">Cycle Ratio</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {COLORS['BlueLine']};
                        line-height: 1.1; margin-top: 4px;">
                {int(_lc_hrs / max(1, _sc_hrs))}×
            </div>
            <div style="color: #64748b; font-size: 0.75rem;">small cycles faster</div>
        </div>
    </div>
    """

    # ── Physics text ──────────────────────────────────────────────────────────
    _physics = f"""
    **Iteration physics (chapter lines 308–329):**

    ```
    Project window        = {_weeks} weeks × 168 hrs/week = {_weeks * 168:,} hours
    Large model cycles    = {_weeks * 168:,} ÷ {_lc_hrs} = {_large_iters} iterations
    Small model cycles    = {_weeks * 168:,} ÷ {max(1, _sc_hrs)} = {_small_iters:,} iterations
    Cycle ratio           = {_lc_hrs} ÷ {max(1, _sc_hrs)} = {int(_lc_hrs / max(1, _sc_hrs))}×
    Accuracy (large, end) = {LARGE_START_ACC}% + {LARGE_GAIN_ITER}% × (1−{_r}^{_large_iters})/(1−{_r}) = {_large_final:.1f}%
    Accuracy (small, end) = {SMALL_START_ACC}% + {SMALL_GAIN_ITER}% × (1−{_r}^{min(_small_iters, 999)})/(1−{_r}) ≈ {_small_final:.1f}%
    ```
    """

    _output_parts = [
        mo.Html(_metrics_html),
        mo.md(_physics),
        mo.as_html(_fig),
        mo.md("---"),
        mo.md("### Constraint Gate Status"),
        mo.Html(_gate_html),
    ]

    mo.vstack(_output_parts)
    return


# ─── ACT II FAILURE STATE ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, act2_prediction, act1_reflection, act2_context,
    act2_small_cycle,
    COLORS,
):
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)

    _ctx    = act2_context.value
    _sc_hrs = act2_small_cycle.value

    # Mobile failure state: hourly retraining exceeds battery capacity
    # Mobile: 5W sustained, battery = 2hr × 5W = 10 Wh
    # Each 1-hour retrain cycle = 5W × 1hr = 5 Wh
    # 24 retrains/day × 5 Wh = 120 Wh/day >> 10 Wh battery
    MOBILE_BATTERY_WH = 10.0
    _session_wh       = max(0.5, _sc_hrs) * 5.0  # 5W per session
    _sessions_per_day = 24.0 / max(0.5, _sc_hrs)
    _total_wh_per_day = _session_wh * _sessions_per_day

    _failure = (_ctx == "mobile" and _sc_hrs == 1)

    if _failure:
        mo.callout(
            mo.md(
                f"**OOM — Infeasible on Mobile.**  "
                f"On-device retraining at {_sc_hrs}-hour cycles requires "
                f"**{_total_wh_per_day:.0f} Wh/day** of battery capacity.  "
                f"Mobile sustained budget: **{MOBILE_BATTERY_WH:.0f} Wh** (2 hr × 5 W).  "
                f"This design exceeds the power budget by "
                f"**{_total_wh_per_day / MOBILE_BATTERY_WH:.0f}×**.  "
                f"Reduce retraining frequency or switch to cloud retraining."
            ),
            kind="danger",
        )
    elif _ctx == "mobile" and _sc_hrs <= 4:
        mo.callout(
            mo.md(
                f"**Power Warning.**  "
                f"{_sc_hrs}-hour cycles on mobile require {_total_wh_per_day:.1f} Wh/day.  "
                f"Battery capacity: {MOBILE_BATTERY_WH:.0f} Wh.  "
                f"Margin: {MOBILE_BATTERY_WH - _total_wh_per_day:.1f} Wh — tight but feasible "
                "if the device charges between sessions."
            ),
            kind="warn",
        )
    else:
        mo.callout(
            mo.md(
                f"**Design is feasible** in the {_ctx} context with {_sc_hrs}-hour cycles.  "
                "No power or compute constraint violation detected. "
                "Explore what happens when you select Mobile context + 1-hour cycles."
            ),
            kind="success",
        )
    return


# ─── ACT II REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction, act1_reflection):
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)

    act2_reflection = mo.ui.radio(
        options={
            "A) Model starting accuracy — a 5% head start is worth more than faster cycles": "A",
            "B) The deployment context constraint — hardware limits override strategy": "B",
            "C) The project window — with enough time, the large model always wins": "C",
            "D) The number of users — more users means longer cycles to gather feedback": "D",
        },
        label=(
            "**Reflection.** In Act II you saw the small model (90% start, 1-hour cycle) "
            "overtake the large model (95% start, 1-week cycle) within the project window. "
            "What determines whether that crossover happens early, late, or never?"
        ),
    )
    act2_reflection
    return (act2_reflection,)


@app.cell(hide_code=True)
def _(mo, act2_prediction, act1_reflection, act2_reflection):
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)
    mo.stop(
        act2_reflection.value is None,
        mo.callout(mo.md("Select an answer to complete Act II."), kind="warn"),
    )

    _correct = act2_reflection.value == "B"
    _feedback = {
        "A": (
            "**Not the primary driver.** The starting accuracy gap (90% vs 95%) affects "
            "when the small model catches up — but not whether it does. With 4,368 iterations "
            "in 26 weeks versus 26, the small model accumulates learning signal that "
            "eventually saturates the accuracy ceiling regardless of where it started. "
            "Starting accuracy sets the initial position; cycle time determines the velocity."
        ),
        "B": (
            "**Correct.** The deployment context is the binding constraint. "
            "A Cloud H100 can run 168× cycles per week at negligible marginal cost, making "
            "the fast iteration strategy dominant. A Mobile NPU at 5W sustained cannot "
            "physically run hourly retraining — the power budget collapses at 1-hour cycles. "
            "Strategy choice is not separable from hardware constraint. "
            "This is the chapter's central lesson: the constraint (Stage 1, hardware budget) "
            "must shape every downstream decision, including iteration strategy."
        ),
        "C": (
            "**Not quite.** With enough time, the large model does not necessarily win — "
            "it saturates. Both models follow a diminishing-returns curve toward the 99% ceiling. "
            "The small model, with thousands more iterations, reaches the saturation ceiling "
            "first. Beyond that point, additional iterations of either model produce negligible gain. "
            "The crossover timing depends on cycle ratio and gain decay rate, not time alone."
        ),
        "D": (
            "**Not directly.** User count affects dataset size and feedback loop quality, "
            "but it does not directly determine the crossover between fast and slow iteration "
            "strategies. The chapter's calculation uses a fixed gain-per-iteration parameter — "
            "increasing the dataset can change that parameter, but the fundamental trade-off "
            "between cycle time and starting accuracy is determined by the physics of the "
            "deployment hardware, not the user base."
        ),
    }

    mo.vstack([
        act2_reflection,
        mo.callout(
            mo.md(_feedback[act2_reflection.value]),
            kind="success" if _correct else "warn",
        ),
    ])
    return


# ─── ACT II PREDICTION vs REALITY OVERLAY ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction, act1_reflection, act2_reflection):
    mo.stop(act1_reflection.value is None)
    mo.stop(act2_prediction.value is None)
    mo.stop(act2_reflection.value is None)

    _is_correct_pred = act2_prediction.value == "C"
    _pred_labels = {
        "A": "large model always wins (starting accuracy dominates)",
        "B": "small model always wins (fast cycles always dominate)",
        "C": "it depends on the window and constraints — there is a crossover",
        "D": "never retrain — deploy once",
    }
    _pred_txt = _pred_labels[act2_prediction.value]

    if _is_correct_pred:
        _msg = (
            f"**You predicted: \"{_pred_txt}\" — Correct.** "
            "The crossover exists because the small model's velocity advantage compounds "
            "only when the deployment context permits high-frequency retraining. "
            "In the cloud context, the small model dominates. In the mobile context with "
            "hourly cycles, the power budget makes the strategy infeasible — the constraint "
            "changes the answer. This is the chapter's core claim: strategy is inseparable "
            "from physical constraints."
        )
        _kind = "success"
    elif act2_prediction.value == "A":
        _msg = (
            f"**You predicted: \"{_pred_txt}.\"** "
            "The instruments showed that the 5% starting accuracy gap is erased by "
            "the velocity advantage — in the cloud context, the small model accumulates "
            "thousands of learning cycles while the large model completes 26. "
            "Starting position matters; cycle time matters more."
        )
        _kind = "warn"
    elif act2_prediction.value == "B":
        _msg = (
            f"**You predicted: \"{_pred_txt}.\"** "
            "Almost — but the mobile failure state shows that fast cycles are not always "
            "physically achievable. In the mobile context with hourly cycles, the power "
            "budget collapses: 120 Wh/day required vs. 10 Wh available. Fast cycles only "
            "win when the hardware can sustain them."
        )
        _kind = "warn"
    else:
        _msg = (
            f"**You predicted: \"{_pred_txt}.\"** "
            "The instruments showed that both models improve substantially over 26 weeks. "
            "Deploying once without retraining leaves significant accuracy on the table "
            "as the small model accumulates 4,368 iterations vs. 26 for the large model. "
            "The chapter's Degradation Equation shows that model accuracy decreases over "
            "time even without retraining — 'deploy once' is not a stable strategy."
        )
        _kind = "warn"

    mo.callout(mo.md(_msg), kind=_kind)
    return


# ─── ACT II MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_reflection):
    mo.stop(act2_reflection.value is None)

    mo.accordion({
        "The Iteration Tax — governing equations": mo.md("""
        **Iteration counts over a project window:**

        $$\\text{Iterations}_{\\text{large}} = \\frac{T_{\\text{project}}}{t_{\\text{cycle,large}}}
        = \\frac{26 \\times 168\\text{ hrs}}{168\\text{ hrs}} = 26$$

        $$\\text{Iterations}_{\\text{small}} = \\frac{T_{\\text{project}}}{t_{\\text{cycle,small}}}
        = \\frac{26 \\times 168\\text{ hrs}}{1\\text{ hr}} = 4{,}368$$

        **Saturating accuracy (diminishing-returns model):**

        $$A_k = A_0 + \\Delta a_0 \\cdot \\frac{1 - r^k}{1 - r}$$

        where:
        - $A_0$ — starting accuracy (95% large, 90% small)
        - $\\Delta a_0$ — gain per first iteration (0.15% large, 0.10% small)
        - $r$ — decay rate per iteration (0.90–0.99); controls saturation speed
        - $k$ — number of iterations completed
        - Accuracy is capped at $A_{\\text{max}} = 99\\%$

        **Mobile power feasibility:**

        $$\\text{Power}_{\\text{day}} = \\frac{24\\text{ hrs}}{t_{\\text{cycle}}} \\times P_{\\text{train}} \\times t_{\\text{cycle}}
        = 24 \\times P_{\\text{train}}$$

        For mobile sustained $P_{\\text{train}} = 5\\text{ W}$ and battery capacity $= 10\\text{ Wh}$:
        - Hourly cycles: $24 \\times 5\\text{ W} \\times 1\\text{ hr} = 120\\text{ Wh/day} >> 10\\text{ Wh}$ — infeasible
        - 4-hour cycles: $6 \\times 5\\text{ W} \\times 4\\text{ hr} = 120\\text{ Wh/day}$ — still infeasible
        - Daily cycles: $1 \\times 5\\text{ W} \\times 2\\text{ hr typical} = 10\\text{ Wh/day}$ — marginal
        """),
    })
    return


# ─── DESIGN LEDGER SAVE + HUD ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    act2_reflection, act1_reflection,
    act2_context, act2_large_cycle, act2_small_cycle,
    act2_project_weeks, act1_discovery_stage,
    act2_gate_memory, act2_gate_latency, act2_gate_schema,
    act2_gate_profile, act2_gate_ondev, act2_gate_rollout,
    LARGE_START_ACC, LARGE_GAIN_ITER, SMALL_START_ACC, SMALL_GAIN_ITER,
    ACC_CEILING,
    np,
):
    mo.stop(act2_reflection.value is None)

    # Compute crossover week for ledger
    _lc_hrs = act2_large_cycle.value
    _sc_hrs = act2_small_cycle.value
    _r_val  = 0.95  # default, not directly accessible from dropdown value name
    _weeks  = act2_project_weeks.value

    def _sat_acc(start, gain, r, k):
        if r >= 1.0 or k == 0:
            return min(start + gain * k, ACC_CEILING)
        return min(start + gain * (1 - r**k) / (1 - r), ACC_CEILING)

    _crossover = None
    for _w in range(_weeks + 1):
        _lg = _sat_acc(LARGE_START_ACC, LARGE_GAIN_ITER, _r_val, int(_w * 168 / _lc_hrs))
        _sm = _sat_acc(SMALL_START_ACC, SMALL_GAIN_ITER, _r_val, int(_w * 168 / max(1, _sc_hrs)))
        if _sm >= _lg:
            _crossover = _w
            break

    _gates = [
        act2_gate_memory.value, act2_gate_latency.value, act2_gate_schema.value,
        act2_gate_profile.value, act2_gate_ondev.value, act2_gate_rollout.value,
    ]
    _gate_names_short = ["memory_budget", "latency_budget", "schema_validation",
                         "memory_profiling", "on_device_test", "staged_rollout"]
    _active_gate_names = [_gate_names_short[i] for i, g in enumerate(_gates) if g]

    _design = {
        "context":               act2_context.value,
        "act1_prediction":       act1_reflection.value,
        "act1_correct":          act1_reflection.value == "C",
        "model_size_chosen":     "small" if _sc_hrs <= _lc_hrs else "large",
        "iteration_cycle_hours": _sc_hrs,
        "crossover_week":        _crossover if _crossover is not None else -1,
        "constraint_gates":      _active_gate_names,
        "constraint_discovery_stage": act1_discovery_stage.value,
        "cost_multiplier":       2 ** (act1_discovery_stage.value - 1),
        "constraint_hit":        (act2_context.value == "mobile" and _sc_hrs == 1),
    }

    ledger.save(chapter=3, design=_design)

    # ── HUD Footer ─────────────────────────────────────────────────────────────
    _ctx_label   = "Cloud Training Node" if act2_context.value == "cloud" else "Mobile On-Device"
    _ctx_color   = COLORS["Cloud"] if act2_context.value == "cloud" else COLORS["Mobile"]
    _gate_count  = sum(_gates)
    _gates_color = COLORS["GreenLine"] if _gate_count >= 4 else COLORS["OrangeLine"] if _gate_count >= 2 else COLORS["RedLine"]

    mo.vstack([
        mo.md("---"),
        mo.callout(
            mo.md(
                "**Lab 03 complete.** Your choices have been recorded in the Design Ledger. "
                "The `iteration_cycle_hours` and `model_size_chosen` values carry forward to "
                "**Lab 05 (Neural Computation)** and **Lab 08 (Training)**, where the "
                "memory footprint of your chosen model size becomes the baseline for "
                "the memory wall and training memory analysis."
            ),
            kind="success",
        ),
        mo.Html(f"""
        <div style="display: flex; gap: 20px; align-items: center; padding: 14px 22px;
                    background: {COLORS['Surface0']}; border-radius: 12px; margin-top: 8px;
                    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem;
                    border: 1px solid {COLORS['Surface1']}; flex-wrap: wrap;">
            <div>
                <span style="color:{COLORS['TextMuted']}; font-weight:600; letter-spacing:0.06em;">
                    LAB ·
                </span>
                <span style="color:#e2e8f0;"> 03 — ML Workflow</span>
            </div>
            <div>
                <span style="color:{COLORS['TextMuted']}; font-weight:600;">CONTEXT · </span>
                <span style="color:{_ctx_color};">{_ctx_label}</span>
            </div>
            <div>
                <span style="color:{COLORS['TextMuted']}; font-weight:600;">CYCLE · </span>
                <span style="color:#e2e8f0;">{_sc_hrs}h small / {_lc_hrs}h large</span>
            </div>
            <div>
                <span style="color:{COLORS['TextMuted']}; font-weight:600;">CROSSOVER · </span>
                <span style="color:#e2e8f0;">
                    {'Week ' + str(_crossover) if _crossover is not None else 'None'}
                </span>
            </div>
            <div>
                <span style="color:{COLORS['TextMuted']}; font-weight:600;">GATES · </span>
                <span style="color:{_gates_color};">{_gate_count}/6 active</span>
            </div>
            <div>
                <span style="color:{COLORS['TextMuted']}; font-weight:600;">STAGE · </span>
                <span style="color:#e2e8f0;">
                    {act1_discovery_stage.value} ({2 ** (act1_discovery_stage.value - 1)}×)
                </span>
            </div>
        </div>
        """),
    ])
    return


if __name__ == "__main__":
    app.run()
