import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ===========================================================================
# ZONE A: OPENING
# ===========================================================================


@app.cell
async def _():
    import marimo as mo
    import sys
    from pathlib import Path

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    import mlsysim
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        action_box,
        build_lab_report,
        diagnose_triad,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        intervention_frontier,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        triad_track_profile,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        action_box,
        COLORS,
        LAB_CSS,
        apply_plotly_theme,
        build_lab_report,
        diagnose_triad,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        intervention_frontier,
        ledger,
        mlsysim,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        triad_track_profile,
    )


@app.cell
def _(get_lab_metadata):
    v1_01_metadata = get_lab_metadata("vol1/lab_01_ml_intro.py")
    return (v1_01_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_01_track_picker = track_selector(default=_default_track)
    v1_01_track_picker
    return (v1_01_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    triad_track_profile,
    v1_01_track_picker,
):
    v1_01_track_id = v1_01_track_picker.value
    v1_01_profile = get_track_profile(v1_01_track_id)
    v1_01_variant = get_lab_track_variant("v1_01_ai_triad", v1_01_profile.track_id)
    v1_01_hardware = resolve_mlsysim_ref(v1_01_variant.hardware_ref)
    v1_01_model = resolve_mlsysim_ref(v1_01_variant.model_ref)
    v1_01_triad = triad_track_profile(
        v1_01_profile,
        v1_01_variant,
        v1_01_hardware,
        v1_01_model,
    )
    return (
        v1_01_hardware,
        v1_01_model,
        v1_01_profile,
        v1_01_track_id,
        v1_01_triad,
        v1_01_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
        track_arc_context,
    v1_01_metadata,
    v1_01_profile,
    v1_01_triad,
    v1_01_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #94a3b8; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 01
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                The AI Triad
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Data &middot; Algorithm &middot; Machine
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 780px; line-height: 1.65;">
                {v1_01_variant.workload_summary} The first engineering decision
                is not which knob to improve; it is which axis is actually binding.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    3 Parts + Memo &middot; ~45 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_01_profile.label}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Binding Axis</span>
                <span class="badge badge-warn">Intervention Frontier</span>
                <span class="badge badge-fail">Rejected Alternatives</span>
            </div>
        </div>
        """),
        track_context(v1_01_profile),
        track_arc_context(v1_01_profile, v1_01_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_01_triad):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
            Learning Objectives
        </div>
        <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
            <div style="margin-bottom: 3px;">1. <strong>Diagnose the binding axis:</strong>
                separate data coverage, algorithm design, and machine envelope.</div>
            <div style="margin-bottom: 3px;">2. <strong>Compare interventions:</strong>
                allocate a fixed budget and see which axis actually improves feasibility.</div>
            <div style="margin-bottom: 3px;">3. <strong>Defend the first fix:</strong>
                select an intervention, reject alternatives, and name validation evidence.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                "The system is failing. Is the first fix Data, Algorithm, or Machine?"
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** - Complete the Introduction chapter's discussion of
    production ML as a system before starting this lab.
    """), kind="info")
    return


# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================


@app.cell(hide_code=True)
def _(action_box, mo, v1_01_triad):
    partA_pred = mo.ui.radio(
        options={
            "A) Data will bind": "Data",
            "B) Algorithm will bind": "Algorithm",
            "C) Machine will bind": "Machine",
            "D) I need evidence before deciding": "Depends",
        },
        label=f"Step 1 - predict before evidence: for {v1_01_triad.label}, which axis do you expect to bind?",
    )
    return (partA_pred,)


@app.cell(hide_code=True)
def _(mo, v1_01_triad):
    partA_data = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v1_01_triad.default_data_pct),
        step=5,
        label="Data readiness (%)",
    )
    partA_algorithm = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v1_01_triad.default_algorithm_pct),
        step=5,
        label="Algorithm readiness (%)",
    )
    partA_machine = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v1_01_triad.default_machine_pct),
        step=5,
        label="Machine readiness (%)",
    )
    partA_decision = action_box(
        mo.ui.radio(
            options={
                "Data": "Data",
                "Algorithm": "Algorithm",
                "Machine": "Machine",
            },
            label="",
        ),
        title="Step 3 - Final Diagnosis",
        body="Choose the answer you would defend after reading the chart and table.",
        name="diagnosis",
    )

    partB_pred = mo.ui.radio(
        options={
            "A) Spend the whole budget on the weakest axis": "weakest",
            "B) Spend evenly across all three axes": "even",
            "C) Spend on hardware first": "hardware",
            "D) Spend on model architecture first": "model",
        },
        label="How should a fixed engineering budget be allocated?",
    )
    return (partA_algorithm, partA_data, partA_decision, partA_machine, partB_pred)


@app.cell(hide_code=True)
def _(mo):
    partB_data_budget = mo.ui.slider(start=0, stop=100, value=40, step=5, label="Data budget (%)")
    partB_algorithm_budget = mo.ui.slider(start=0, stop=100, value=30, step=5, label="Algorithm budget (%)")
    partB_machine_budget = mo.ui.slider(start=0, stop=100, value=30, step=5, label="Machine budget (%)")
    partB_selected = mo.ui.dropdown(
        options={"Data": "Data", "Algorithm": "Algorithm", "Machine": "Machine"},
        value="Data",
        label="Intervention to defend",
    )

    partC_pred = mo.ui.radio(
        options={
            "A) Defend the selected fix and explain rejected alternatives": "defend",
            "B) Say all axes matter equally and stop": "equal",
            "C) Choose the cheapest fix regardless of evidence": "cheap",
            "D) Skip validation until later labs": "skip_validation",
        },
        label="What belongs in the triad diagnosis memo?",
    )
    return (partB_algorithm_budget, partB_data_budget, partB_machine_budget, partB_selected, partC_pred)


@app.cell(hide_code=True)
def _(mo, v1_01_triad):
    _tests = {test: test for test in v1_01_triad.validation_tests}
    partC_validation = mo.ui.dropdown(
        options=_tests,
        value=v1_01_triad.validation_tests[0],
        label="Validation evidence",
    )
    return (partC_validation,)


# ===========================================================================
# ZONE C: MAIN LAB
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    diagnose_triad,
    go,
    intervention_frontier,
    mo,
    partA_algorithm,
    partA_data,
    partA_decision,
    partA_machine,
    partA_pred,
    partB_algorithm_budget,
    partB_data_budget,
    partB_machine_budget,
    partB_pred,
    partB_selected,
    partC_pred,
    partC_validation,
    v1_01_profile,
    v1_01_triad,
    v1_01_variant,
):
    def _metric_card(label, value, detail, color, border=False):
        border_style = f"2px solid {color}" if border else "1px solid #e2e8f0"
        return f"""
        <div style="padding:16px; border:{border_style}; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {color}; flex:1;">
            <div style="color:#64748b; font-size:0.78rem; font-weight:700;">{label}</div>
            <div style="font-size:1.45rem; font-weight:800; color:{color};">{value}</div>
            <div style="font-size:0.72rem; color:#64748b;">{detail}</div>
        </div>
        """

    def build_part_a():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Incoming Message &middot; {v1_01_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "{v1_01_triad.failure_story}"
                </div>
            </div>
            """),
            mo.md(f"""
## Part A: Diagnose Data, Algorithm, Machine

**What you need to know.** The same symptom can come from different axes.

- **Data:** {v1_01_triad.data_axis}
- **Algorithm:** {v1_01_triad.algorithm_axis}
- **Machine:** {v1_01_triad.machine_axis}

**How this part works.**

1. Make a prediction. This records what you expect; it does not change the plot.
2. Move the readiness sliders. These are the experiment controls, so they update the plot and evidence.
3. After reading the evidence, choose a final diagnosis.
            """),
            partA_pred,
        ]
        if partA_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select a prediction first. Nothing is supposed to change yet; this is your hypothesis before the evidence."
            ), kind="warn"))
            return mo.vstack(items)

        items.append(mo.callout(mo.md(
            "**Now explore the evidence.** The sliders below change the simulated readiness scores. "
            "The prediction above stays fixed so you can compare expectation against evidence."
        ), kind="info"))
        items.append(mo.hstack([partA_data, partA_algorithm, partA_machine], widths="equal"))
        _diag = diagnose_triad(
            v1_01_triad,
            data_score_pct=partA_data.value,
            algorithm_score_pct=partA_algorithm.value,
            machine_score_pct=partA_machine.value,
        )
        _axes = ["Data", "Algorithm", "Machine"]
        _scores = [_diag.data_score_pct, _diag.algorithm_score_pct, _diag.machine_score_pct]
        _thresholds = [_diag.data_threshold_pct, _diag.algorithm_threshold_pct, _diag.machine_threshold_pct]
        _colors = [
            COLORS["RedLine"] if axis == _diag.binding_axis else COLORS["BlueLine"]
            for axis in _axes
        ]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=_axes, y=_scores, name="Readiness", marker_color=_colors, opacity=0.9))
        _fig.add_trace(go.Scatter(x=_axes, y=_thresholds, name="Track threshold", mode="markers+lines", line=dict(color=COLORS["OrangeLine"], dash="dash")))
        _fig.update_layout(
            height=340,
            yaxis=dict(title="Readiness (%)", gridcolor="#f1f5f9", range=[0, 105]),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=60, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))
        items.append(mo.callout(mo.md(
            f"**Read the plot:** the binding axis is the axis with the weakest margin to its track threshold. "
            f"Under the current settings, the evidence points to **{_diag.binding_axis}**."
        ), kind="info"))

        _status_color = COLORS["GreenLine"] if _diag.feasible else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Binding Axis", _diag.binding_axis, "lowest margin to threshold", COLORS["RedLine"], True)}
            {_metric_card("Primary Metric", _diag.primary_metric, v1_01_triad.label, COLORS["BlueLine"])}
            {_metric_card("Guardrail", _diag.guardrail_metric, "must stay protected", COLORS["OrangeLine"])}
            {_metric_card("Status", "PASS" if _diag.feasible else "FAIL", ", ".join(_diag.violations) or "no violations", _status_color, True)}
        </div>
        """))

        items.append(mo.md(f"""
**Diagnosis Table**

| Axis | Score | Threshold | Meaning |
|---|---:|---:|---|
| Data | {_diag.data_score_pct:.0f}% | {_diag.data_threshold_pct:.0f}% | {v1_01_triad.data_axis} |
| Algorithm | {_diag.algorithm_score_pct:.0f}% | {_diag.algorithm_threshold_pct:.0f}% | {v1_01_triad.algorithm_axis} |
| Machine | {_diag.machine_score_pct:.0f}% | {_diag.machine_threshold_pct:.0f}% | {v1_01_triad.machine_axis} |
        """))

        items.append(partA_decision)
        _decision_value = (partA_decision.value or {}).get("diagnosis")
        if _decision_value is None:
            return mo.vstack(items)

        if _decision_value == _diag.binding_axis:
            items.append(mo.callout(mo.md(
                "**Good diagnosis.** Your final answer matches the evidence. The binding axis comes from the track thresholds, not from a default rule."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**Re-check the evidence.** Your final answer is {_decision_value}, but the current binding axis is {_diag.binding_axis}. "
                "The first fix should target the weakest margin to threshold."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_b():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Budget Review &middot; {v1_01_triad.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "We have one engineering budget. What first intervention is worth defending?"
                </div>
            </div>
            """),
            mo.md("""
## Part B: Intervention Frontier

**What you need to know.** The first fix should move the binding axis enough to
change feasibility. Improving a healthy axis has weak marginal return.
            """),
            partB_pred,
        ]
        if partB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the intervention frontier."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_data_budget, partB_algorithm_budget, partB_machine_budget, partB_selected], widths="equal"))
        _frontier = intervention_frontier(
            v1_01_triad,
            data_budget_pct=partB_data_budget.value,
            algorithm_budget_pct=partB_algorithm_budget.value,
            machine_budget_pct=partB_machine_budget.value,
            selected_intervention=partB_selected.value,
        )
        _axes = ["Data", "Algorithm", "Machine"]
        _scores = [_frontier.data_score_pct, _frontier.algorithm_score_pct, _frontier.machine_score_pct]
        _colors = [
            COLORS["GreenLine"] if axis == _frontier.best_intervention else COLORS["BlueLine"]
            for axis in _axes
        ]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=_axes, y=_scores, marker_color=_colors, opacity=0.9))
        _fig.update_layout(
            height=330,
            yaxis=dict(title="Post-intervention readiness (%)", gridcolor="#f1f5f9", range=[0, 105]),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _status_color = COLORS["GreenLine"] if _frontier.feasible else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Selected", _frontier.selected_intervention, f"margin {_frontier.selected_score_pct:+.1f} pp", COLORS["OrangeLine"], True)}
            {_metric_card("Best Axis", _frontier.best_intervention, f"margin {_frontier.best_score_pct:+.1f} pp", COLORS["GreenLine"])}
            {_metric_card("Binding After Fix", _frontier.binding_axis, "remaining weakest axis", COLORS["RedLine"])}
            {_metric_card("Status", "PASS" if _frontier.feasible else "FAIL", "all thresholds met" if _frontier.feasible else "constraint remains", _status_color, True)}
        </div>
        """))

        items.append(mo.md(f"""
**Frontier Table**

| Axis | Budget | Post-score |
|---|---:|---:|
| Data | {_frontier.data_budget_pct:.0f}% | {_frontier.data_score_pct:.1f}% |
| Algorithm | {_frontier.algorithm_budget_pct:.0f}% | {_frontier.algorithm_score_pct:.1f}% |
| Machine | {_frontier.machine_budget_pct:.0f}% | {_frontier.machine_score_pct:.1f}% |

Rejected alternatives for the selected intervention: {", ".join(_frontier.rejected_alternatives)}.
        """))

        if partB_pred.value == "weakest":
            items.append(mo.callout(mo.md("**Correct principle.** Spend first where the constraint is binding."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Budget should follow diagnosis.** Even spending and hardware-first spending can leave the binding axis untouched."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_c():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenLL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Diagnosis Memo &middot; {v1_01_triad.report_artifact}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Write the first fix you would defend, the alternatives you reject,
                    and the evidence that would invalidate your choice."
                </div>
            </div>
            """),
            mo.md("""
## Part C: Defensible Fix

**What you need to know.** A triad diagnosis memo must defend one first move and
explain why the other two axes are weaker first investments.
            """),
            partC_pred,
        ]
        if partC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the memo builder."), kind="warn"))
            return mo.vstack(items)

        items.append(partC_validation)
        _frontier = intervention_frontier(
            v1_01_triad,
            data_budget_pct=partB_data_budget.value,
            algorithm_budget_pct=partB_algorithm_budget.value,
            machine_budget_pct=partB_machine_budget.value,
            selected_intervention=partB_selected.value,
        )
        items.append(mo.callout(mo.md(
            f"**Memo decision:** Defend **{_frontier.selected_intervention}** first for {v1_01_triad.label}. "
            f"The remaining binding axis is **{_frontier.binding_axis}**. Attach **{partC_validation.value}** as validation evidence."
        ), kind="info"))
        items.append(mo.md(f"""
**Rejected Alternatives**

{chr(10).join(f"- {axis}: weaker first move under the current evidence scores." for axis in _frontier.rejected_alternatives)}
        """))

        if partC_pred.value == "defend":
            items.append(mo.callout(mo.md("**Correct.** The memo must be a defensible first-fix decision, not a generic list."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**The memo needs a defended first fix.** Name the intervention, rejected alternatives, validation evidence, and residual risk."
            ), kind="warn"))
        return mo.vstack(items)

    def build_synthesis():
        return mo.vstack([
            mo.md("## Key Takeaways"),
            mo.callout(mo.md(
                "**1. Data, algorithm, and machine form a diagnosis tool.** Do not spend on an axis until evidence says it binds."
            ), kind="info"),
            mo.callout(mo.md(
                f"**2. Track context changes the answer.** For {v1_01_triad.label}, the same words mean different engineering constraints and different stakeholder risks."
            ), kind="info"),
            mo.callout(mo.md(
                "**3. The report artifact is a first-fix memo.** It must include the selected intervention, rejected alternatives, validation evidence, and residual risk."
            ), kind="info"),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Diagnosis": build_part_a(),
        "Part B: Intervention Frontier": build_part_b(),
        "Part C: Defensible Fix": build_part_c(),
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


# ===========================================================================
# ZONE D: LEDGER HUD AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    partA_decision,
    partA_pred,
    partB_pred,
    partC_pred,
    v1_01_profile,
    v1_01_triad,
    v1_01_variant,
):
    _decision_value = (partA_decision.value or {}).get("diagnosis")
    if (
        partA_pred.value is not None
        and _decision_value is not None
        and partB_pred.value is not None
        and partC_pred.value is not None
    ):
        ledger.save(chapter=1, design={
            "chapter": "v1_01",
            "track_id": v1_01_profile.track_id,
            "scenario_id": v1_01_variant.scenario_id,
            "hardware_ref": v1_01_triad.hardware_ref,
            "model_ref": v1_01_triad.model_ref,
            "completed": True,
            "triad_prediction": partA_pred.value,
            "triad_final_diagnosis": _decision_value,
            "budget_prediction": partB_pred.value,
            "memo_prediction": partC_pred.value,
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">01 &middot; AI Triad</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_01_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">ARTIFACT</span>
        <span class="hud-value">{v1_01_triad.report_artifact}</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    diagnose_triad,
    intervention_frontier,
    mo,
    partA_algorithm,
    partA_data,
    partA_decision,
    partA_machine,
    partA_pred,
    partB_algorithm_budget,
    partB_data_budget,
    partB_machine_budget,
    partB_pred,
    partB_selected,
    partC_pred,
    partC_validation,
    report_export_panel,
    v1_01_metadata,
    v1_01_profile,
    v1_01_triad,
    v1_01_variant,
):
    _diag = diagnose_triad(
        v1_01_triad,
        data_score_pct=partA_data.value,
        algorithm_score_pct=partA_algorithm.value,
        machine_score_pct=partA_machine.value,
    )
    _frontier = intervention_frontier(
        v1_01_triad,
        data_budget_pct=partB_data_budget.value,
        algorithm_budget_pct=partB_algorithm_budget.value,
        machine_budget_pct=partB_machine_budget.value,
        selected_intervention=partB_selected.value,
    )
    _incomplete = []
    if partA_pred.value is None:
        _incomplete.append("Part A pre-evidence prediction")
    _decision_value = (partA_decision.value or {}).get("diagnosis")
    if _decision_value is None:
        _incomplete.append("Part A final diagnosis")
    if partB_pred.value is None:
        _incomplete.append("Part B intervention prediction")
    if partC_pred.value is None:
        _incomplete.append("Part C memo prediction")

    _report = build_lab_report(
        v1_01_metadata,
        track=v1_01_profile.label,
        scenario=v1_01_variant.workload_summary,
        learning_objectives=(
            "Diagnose which data, algorithm, or machine axis binds for the selected track.",
            "Compare fixed-budget interventions across Data, Algorithm, and Machine.",
            "Write a first-fix diagnosis memo with rejected alternatives and validation evidence.",
        ),
        predictions={
            "pre_evidence_binding_axis": partA_pred.value,
            "final_binding_axis_diagnosis": _decision_value,
            "budget_strategy": partB_pred.value,
            "memo_structure": partC_pred.value,
        },
        knob_settings={
            "data_score_pct": partA_data.value,
            "algorithm_score_pct": partA_algorithm.value,
            "machine_score_pct": partA_machine.value,
            "data_budget_pct": partB_data_budget.value,
            "algorithm_budget_pct": partB_algorithm_budget.value,
            "machine_budget_pct": partB_machine_budget.value,
            "selected_intervention": partB_selected.value,
            "validation_evidence": partC_validation.value,
        },
        evidence_summary={
            "hardware_ref": v1_01_triad.hardware_ref,
            "model_ref": v1_01_triad.model_ref,
            "binding_axis": _diag.binding_axis,
            "diagnosis_feasible": _diag.feasible,
            "violations": _diag.violations,
            "frontier_binding_axis": _frontier.binding_axis,
            "selected_intervention": _frontier.selected_intervention,
            "best_intervention": _frontier.best_intervention,
            "rejected_alternatives": _frontier.rejected_alternatives,
        },
        final_decision=(
            f"Defend {_frontier.selected_intervention} first for {v1_01_triad.label}; "
            f"validate with {partC_validation.value}; reject {', '.join(_frontier.rejected_alternatives)} as weaker first moves."
        ),
        big_takeaways=(
            "Data, algorithm, and machine are a diagnosis tool, not a vocabulary list.",
            "Track context changes the meaning of each axis.",
            "A first-fix memo should explain rejected alternatives and validation evidence.",
        ),
        reflections={
            "report_artifact": v1_01_triad.report_artifact,
            "data_axis": v1_01_triad.data_axis,
            "algorithm_axis": v1_01_triad.algorithm_axis,
            "machine_axis": v1_01_triad.machine_axis,
            "validation_tests": v1_01_triad.validation_tests,
        },
        residual_risk=(
            f"The selected first fix may be invalidated if {partC_validation.value} fails "
            f"or if {_frontier.binding_axis} remains below threshold after implementation."
        ),
        source_trace={
            "track_id": v1_01_profile.track_id,
            "scenario_id": v1_01_variant.scenario_id,
            "hardware_ref": v1_01_variant.hardware_ref,
            "model_ref": v1_01_variant.model_ref,
            "shared_helper": "mlsysbook_labs.triad",
            "source_policy": v1_01_profile.source_policy,
        },
        result_snapshot={
            "triad_profile": v1_01_triad,
            "diagnosis": _diag,
            "intervention_frontier": _frontier,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-01 diagnosis memo is generated locally from the selected track, "
                "your inputs, and the computed evidence."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
