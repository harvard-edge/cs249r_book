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
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        constraint_tax,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        iteration_frontier,
        part_workflow,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        workflow_policy,
        workflow_track_profile,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        COLORS,
        LAB_CSS,
        apply_plotly_theme,
        build_lab_report,
        constraint_tax,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        iteration_frontier,
        ledger,
        mo,
        part_workflow,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        workflow_policy,
        workflow_track_profile,
    )


@app.cell
def _(get_lab_metadata):
    v1_03_metadata = get_lab_metadata("vol1/lab_03_ml_workflow.py")
    return (v1_03_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_03_track_picker = track_selector(default=_default_track)
    v1_03_track_picker
    return (v1_03_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_03_track_picker,
    workflow_track_profile,
):
    v1_03_track_id = v1_03_track_picker.value
    v1_03_profile = get_track_profile(v1_03_track_id)
    v1_03_variant = get_lab_track_variant("v1_03_constraint_tax", v1_03_profile.track_id)
    v1_03_hardware = resolve_mlsysim_ref(v1_03_variant.hardware_ref)
    v1_03_model = resolve_mlsysim_ref(v1_03_variant.model_ref)
    v1_03_workflow = workflow_track_profile(
        v1_03_profile,
        v1_03_variant,
        v1_03_hardware,
        v1_03_model,
    )
    return (
        v1_03_hardware,
        v1_03_model,
        v1_03_profile,
        v1_03_track_id,
        v1_03_variant,
        v1_03_workflow,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
    track_arc_context,
    v1_03_metadata,
    v1_03_profile,
    v1_03_variant,
    v1_03_workflow,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 03
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                The Constraint Tax
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Workflow Gates &middot; Iteration Risk &middot; Release Policy
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 820px; line-height: 1.65;">
                {v1_03_variant.workload_summary} This lab traces how the selected
                track's deployment constraint propagates backward through data,
                model design, validation, release, and monitoring.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~45 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_03_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_03_workflow.constraint_name}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Constraint Propagation</span>
                <span class="badge badge-warn">Iteration Tax</span>
                <span class="badge badge-info">Gate Confidence</span>
                <span class="badge badge-fail">Release Policy</span>
            </div>
        </div>
        """),
        track_context(v1_03_profile),
        track_arc_context(v1_03_profile, v1_03_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, part_workflow, v1_03_workflow):
    mo.vstack([
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
            <div style="margin-bottom: 3px;">1. <strong>Trace propagation:</strong>
                follow {v1_03_workflow.constraint_name} across data, model, validation, release, and monitoring.</div>
            <div style="margin-bottom: 3px;">2. <strong>Measure iteration tax:</strong>
                compute the rework created by late discovery.</div>
            <div style="margin-bottom: 3px;">3. <strong>Balance gates:</strong>
                trade iteration speed against deployment confidence.</div>
            <div style="margin-bottom: 3px;">4. <strong>Write policy:</strong>
                choose gates, release rules, rollback rules, and residual blind spot.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                When should {v1_03_workflow.label} test the deployment constraint so
                the team avoids expensive rework without making every iteration too slow?
            </div>
        </div>
    </div>
    """),
    part_workflow(
        "Constraint Tax Workflow",
        (
            {
                "part": "Part A",
                "concept": "Constraints Propagate Through The Workflow",
                "prediction": "Predict when the deployment constraint should be tested.",
                "controls": "Move the discovery stage and inspect which assumptions harden.",
                "evidence": "Read the stage table across data, model, validation, release, and monitoring.",
                "decision": "Name the first gate that should block bad assumptions.",
            },
            {
                "part": "Part B",
                "concept": "Late Discovery Creates Iteration Tax",
                "prediction": "Predict the cost shape for discovering the constraint late.",
                "controls": "Move the discovery stage and compare current rework with the recommended gate.",
                "evidence": "Read multiplier, rework days, avoidable rework, and artifacts to rebuild.",
                "decision": "Decide whether to pay the tax or move the gate earlier.",
            },
            {
                "part": "Part C",
                "concept": "Evaluation Gates Trade Speed For Confidence",
                "prediction": "Predict the weakest validation dimension for this track.",
                "controls": "Tune validation depth, automation, hardware realism, and data scale.",
                "evidence": "Compare iteration days, confidence, residual risk, and risk budget.",
                "decision": "Choose the validation stance before release pressure arrives.",
            },
            {
                "part": "Part D",
                "concept": "Workflow Policy Is System Design",
                "prediction": "Predict which release gate should become non-negotiable.",
                "controls": "Select the gate, release policy, and rollback rule.",
                "evidence": "Compare policy summary, residual risk, and the remaining blind spot.",
                "decision": "Write the workflow memo and name the risk you still carry.",
            },
        ),
        scenario=(
            f"{v1_03_workflow.label} needs a workflow that discovers "
            f"{v1_03_workflow.constraint_name} before the team builds on a bad assumption."
        ),
        reflection="Carry one gate, one evidence requirement, and one residual blind spot into the report.",
    ),
    ])
    return


# ===========================================================================
# ZONE B: CONTROLS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_03_workflow):
    v1_03_gate_prediction = mo.ui.radio(
        options={
            "Before data and model assumptions harden": "early",
            "During model design": "model_design",
            "During release hardening": "release",
            "After launch in monitoring": "monitoring",
        },
        label=f"When should {v1_03_workflow.constraint_name} be tested?",
    )
    v1_03_gate_prediction
    return (v1_03_gate_prediction,)


@app.cell(hide_code=True)
def _(mo, v1_03_workflow):
    v1_03_discovery_stage = mo.ui.slider(
        start=1,
        stop=len(v1_03_workflow.stage_names),
        value=v1_03_workflow.default_discovery_stage,
        step=1,
        label="Discovery stage",
    )
    v1_03_discovery_stage
    return (v1_03_discovery_stage,)


@app.cell(hide_code=True)
def _(mo, v1_03_workflow):
    v1_03_tax_prediction = mo.ui.radio(
        options={
            "It stays roughly constant across stages": "constant",
            "It grows linearly with the number of stages": "linear",
            "It doubles at each later stage": "exponential",
            "It is mostly documentation overhead": "paperwork",
        },
        label=f"How does the cost change if {v1_03_workflow.constraint_name} is found later?",
    )
    v1_03_tax_prediction
    return (v1_03_tax_prediction,)


@app.cell(hide_code=True)
def _(mo, v1_03_workflow):
    v1_03_frontier_prediction = mo.ui.radio(
        options={
            "Validation depth": "validation depth",
            "Automation": "automation",
            "Hardware realism": "hardware realism",
            "Data scale": "data scale",
        },
        label="Which validation dimension is most likely to be the current bottleneck?",
    )
    v1_03_validation_depth = mo.ui.slider(
        start=0,
        stop=100,
        value=v1_03_workflow.default_validation_depth_pct,
        step=5,
        label="Validation depth (%)",
    )
    v1_03_automation = mo.ui.slider(
        start=0,
        stop=100,
        value=v1_03_workflow.default_automation_pct,
        step=5,
        label="Automation (%)",
    )
    v1_03_realism = mo.ui.slider(
        start=0,
        stop=100,
        value=v1_03_workflow.default_hardware_realism_pct,
        step=5,
        label="Hardware realism (%)",
    )
    v1_03_data_scale = mo.ui.slider(
        start=0,
        stop=100,
        value=v1_03_workflow.default_data_scale_pct,
        step=5,
        label="Data scale coverage (%)",
    )
    return (
        v1_03_automation,
        v1_03_data_scale,
        v1_03_frontier_prediction,
        v1_03_realism,
        v1_03_validation_depth,
    )


@app.cell(hide_code=True)
def _(mo, v1_03_workflow):
    _policy_prediction_options = {gate.label: gate.gate_id for gate in v1_03_workflow.gate_options}
    v1_03_policy_prediction = mo.ui.radio(
        options=_policy_prediction_options,
        label="Which gate should become non-negotiable in the workflow policy?",
    )
    _gate_options = {gate.label: gate.gate_id for gate in v1_03_workflow.gate_options}
    v1_03_gate_choice = mo.ui.dropdown(
        options=_gate_options,
        value=v1_03_workflow.gate_options[0].label,
        label="Workflow gate",
    )
    _release_options = {policy: policy for policy in v1_03_workflow.release_policies}
    v1_03_release_policy = mo.ui.dropdown(
        options=_release_options,
        value=v1_03_workflow.release_policies[0],
        label="Release policy",
    )
    _rollback_options = {rule: rule for rule in v1_03_workflow.rollback_rules}
    v1_03_rollback_rule = mo.ui.dropdown(
        options=_rollback_options,
        value=v1_03_workflow.rollback_rules[0],
        label="Rollback rule",
    )
    return (
        v1_03_gate_choice,
        v1_03_policy_prediction,
        v1_03_release_policy,
        v1_03_rollback_rule,
    )


@app.cell(hide_code=True)
def _(mo):
    _text_area = getattr(mo.ui, "text_area")
    v1_03_reflection = _text_area(
        label="Reflection",
        placeholder="Name the gate you would defend and the blind spot the workflow still has.",
        full_width=True,
    )
    return (v1_03_reflection,)


@app.cell
def _(
    constraint_tax,
    iteration_frontier,
    v1_03_automation,
    v1_03_data_scale,
    v1_03_discovery_stage,
    v1_03_gate_choice,
    v1_03_realism,
    v1_03_release_policy,
    v1_03_rollback_rule,
    v1_03_validation_depth,
    v1_03_workflow,
    workflow_policy,
):
    v1_03_tax = constraint_tax(v1_03_workflow, discovery_stage=v1_03_discovery_stage.value)
    v1_03_frontier = iteration_frontier(
        v1_03_workflow,
        validation_depth_pct=v1_03_validation_depth.value,
        automation_pct=v1_03_automation.value,
        hardware_realism_pct=v1_03_realism.value,
        data_scale_pct=v1_03_data_scale.value,
    )
    v1_03_policy = workflow_policy(
        v1_03_workflow,
        v1_03_frontier,
        gate_id=v1_03_gate_choice.value,
        release_policy=v1_03_release_policy.value,
        rollback_rule=v1_03_rollback_rule.value,
    )
    v1_03_risk_budget_pct = (
        v1_03_workflow.min_residual_risk_pct
        + (v1_03_workflow.base_residual_risk_pct - v1_03_workflow.min_residual_risk_pct) * 0.35
    )
    v1_03_cycle_budget_days = v1_03_workflow.base_cycle_days * 1.5
    return (
        v1_03_cycle_budget_days,
        v1_03_frontier,
        v1_03_policy,
        v1_03_risk_budget_pct,
        v1_03_tax,
    )


# ===========================================================================
# ZONE C: PARTS
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    constraint_tax,
    go,
    iteration_frontier,
    mo,
    source_trace,
    v1_03_automation,
    v1_03_cycle_budget_days,
    v1_03_data_scale,
    v1_03_discovery_stage,
    v1_03_frontier,
    v1_03_frontier_prediction,
    v1_03_gate_choice,
    v1_03_gate_prediction,
    v1_03_policy,
    v1_03_policy_prediction,
    v1_03_realism,
    v1_03_reflection,
    v1_03_release_policy,
    v1_03_risk_budget_pct,
    v1_03_rollback_rule,
    v1_03_tax,
    v1_03_tax_prediction,
    v1_03_validation_depth,
    v1_03_workflow,
):
    def v1_03_part_header(part, concept, question, color):
        return mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget" style="border-left:4px solid {color};">
          <div class="mlsysbook-part-title"><h2>{part}: {concept}</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong> {question}</div>
        </div>
        """)

    def v1_03_prediction_feedback(value, correct_value, correct_text, miss_text):
        if value is None:
            return mo.callout(mo.md("Commit to a prediction to unlock the instrument."), kind="warn")
        _reveal = f"You predicted `{value}`; actual evidence points to `{correct_value}`."
        if value == correct_value:
            return mo.callout(mo.md(f"{_reveal} {correct_text}"), kind="success")
        return mo.callout(mo.md(f"{_reveal} {miss_text}"), kind="warn")

    def v1_03_recommended_gate():
        return min(
            v1_03_workflow.gate_options,
            key=lambda gate: abs(gate.stage - v1_03_workflow.recommended_gate_stage),
        )

    def v1_03_build_part_a():
        _items = [
            v1_03_part_header(
                "Part A",
                "Constraints Propagate Through The Workflow",
                f"Where should {v1_03_workflow.constraint_name} first block the workflow?",
                COLORS["BlueLine"],
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <p>{v1_03_workflow.failure_story}</p>
              <p>The stakeholder is the <strong>{v1_03_workflow.stakeholder}</strong>. The decision is not just when to test;
                 it is which downstream assumptions are allowed to harden before the track proves the deployment wall.</p>
            </div>
            """),
            mo.Html("<div class=\"mlsysbook-panel\"><h2>Prediction</h2></div>"),
            v1_03_gate_prediction,
        ]
        if v1_03_gate_prediction.value is None:
            _items.append(mo.callout(mo.md("Pick a gate timing before inspecting the stage evidence."), kind="warn"))
            return mo.vstack(_items)

        _assumptions = (
            "success metric, guardrail metric, and deployment envelope",
            "data or signal contract and collection assumptions",
            "model size, runtime, feature, and preprocessing choices",
            "production-condition validation evidence",
            "release package, rollout commitment, and rollback surface",
            "monitoring threshold, retraining trigger, and incident playbook",
        )
        _stage_rows = []
        for idx, stage in enumerate(v1_03_workflow.stage_names, start=1):
            _tax = constraint_tax(v1_03_workflow, discovery_stage=idx)
            _gate = next((gate.label for gate in v1_03_workflow.gate_options if gate.stage == idx), "stage contract")
            _status = "recommended or earlier" if idx <= v1_03_workflow.recommended_gate_stage else "late evidence debt"
            _color = COLORS["GreenLine"] if idx <= v1_03_workflow.recommended_gate_stage else COLORS["RedLine"]
            _stage_rows.append(
                f"""
                <tr>
                  <td>{idx}</td>
                  <td>{stage}</td>
                  <td>{_assumptions[idx - 1]}</td>
                  <td>{_gate}</td>
                  <td style="color:{_color}; font-weight:800;">{_status}</td>
                  <td style="text-align:right;">{_tax.cost_multiplier:.0f}x</td>
                </tr>
                """
            )

        _items.extend([
            v1_03_prediction_feedback(
                v1_03_gate_prediction.value,
                "early",
                "**Correct direction.** The deployment constraint must be tested before data and model assumptions harden.",
                "**The instrument will show why later checks are expensive.** A deployment constraint that survives into release or monitoring propagates backward through the already-built workflow.",
            ),
            mo.Html("<div class=\"mlsysbook-panel\"><h2>Manipulation</h2></div>"),
            v1_03_discovery_stage,
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Evidence Table</h2>
              <table style="width:100%; border-collapse:collapse; font-size:0.84rem;">
                <thead>
                  <tr style="border-bottom:1px solid {COLORS['Border']}; color:{COLORS['TextMuted']}; text-align:left;">
                    <th>Stage</th><th>Name</th><th>Assumption that hardens</th><th>Evidence gate</th><th>Status</th><th style="text-align:right;">Cost shape</th>
                  </tr>
                </thead>
                <tbody>{''.join(_stage_rows)}</tbody>
              </table>
            </div>
            """),
        ])
        if v1_03_tax.late_discovery:
            _items.append(mo.callout(
                mo.md(
                    f"**Boundary crossed.** Discovery at **{v1_03_tax.discovery_stage_name}** means "
                    f"the workflow must revisit **{', '.join(v1_03_tax.artifacts_to_rebuild)}** for "
                    f"**{v1_03_workflow.constraint_name}**."
                ),
                kind="danger",
            ))
        else:
            _items.append(mo.callout(
                mo.md(f"**Gate is early enough.** {v1_03_tax.discovery_stage_name} catches the constraint before late artifacts harden."),
                kind="success",
            ))
        _items.extend([
            mo.accordion({
                "Math Peek / Source Model - workflow iron law": mo.md(f"""
The chapter's workflow view maps lifecycle stages onto the iron law:

$$
T = \\frac{{D_{{vol}}}}{{BW}} + \\frac{{O}}{{R_{{peak}} \\cdot \\eta_{{hw}}}} + L_{{lat}}
$$

For **{v1_03_workflow.label}**, the deployment constraint is **{v1_03_workflow.constraint_name}**.
If that constraint changes $L_{{lat}}$, $R_{{peak}}$, or feasible efficiency $\\eta_{{hw}}$,
then data assumptions, model operations, validation evidence, and monitoring thresholds all move.
""")
            }),
            source_trace({
                "chapter_anchor": "ML Workflow - Lifecycle Stages and Constraint Propagation Principle",
                "profile_helper": "mlsysbook_labs.workflow_track_profile",
                "hardware_ref": v1_03_workflow.hardware_ref,
                "model_ref": v1_03_workflow.model_ref,
            }, summary="Stage names and track constraints come from the selected V1-03 workflow profile."),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <div class="mlsysbook-callout"><strong>Report decision:</strong>
                The first blocking gate should be at stage {v1_03_tax.recommended_stage},
                {v1_03_tax.recommended_stage_name}, before {v1_03_workflow.constraint_name}
                becomes release debt.</div>
            </div>
            """),
        ])
        return mo.vstack(_items)

    def v1_03_build_part_b():
        _items = [
            v1_03_part_header(
                "Part B",
                "Late Discovery Creates A Measurable Iteration Tax",
                "How large is the rework tax when the deployment wall is found late?",
                COLORS["RedLine"],
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <p>The team asks whether late discovery is just a schedule issue. Move the discovery stage and measure
                 how many person-days become avoidable rework for <strong>{v1_03_workflow.label}</strong>.</p>
            </div>
            """),
            mo.Html("<div class=\"mlsysbook-panel\"><h2>Prediction</h2></div>"),
            v1_03_tax_prediction,
        ]
        if v1_03_tax_prediction.value is None:
            _items.append(mo.callout(mo.md("Predict the cost shape before opening the rework chart."), kind="warn"))
            return mo.vstack(_items)

        _stage_numbers = list(range(1, len(v1_03_workflow.stage_names) + 1))
        _stage_taxes = [constraint_tax(v1_03_workflow, discovery_stage=idx) for idx in _stage_numbers]
        _bar_colors = [
            COLORS["GreenLine"] if tax.discovery_stage <= v1_03_workflow.recommended_gate_stage else COLORS["RedLine"]
            for tax in _stage_taxes
        ]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=[tax.discovery_stage_name for tax in _stage_taxes],
            y=[tax.rework_days for tax in _stage_taxes],
            marker=dict(color=_bar_colors),
            name="Rework days",
        ))
        _fig.add_trace(go.Scatter(
            x=[v1_03_tax.discovery_stage_name],
            y=[v1_03_tax.rework_days],
            mode="markers",
            marker=dict(color=COLORS["BlueLine"], size=16, line=dict(color="white", width=2)),
            name="Current discovery",
        ))
        _fig.update_layout(
            height=340,
            xaxis=dict(title="Discovery stage"),
            yaxis=dict(title="Person-days of rework"),
            margin=dict(l=60, r=20, t=35, b=90),
        )
        apply_plotly_theme(_fig)

        _stage_rows = []
        for tax in _stage_taxes:
            _color = COLORS["GreenLine"] if tax.discovery_stage <= v1_03_workflow.recommended_gate_stage else COLORS["RedLine"]
            _stage_rows.append(
                f"""
                <tr>
                  <td>{tax.discovery_stage}</td>
                  <td>{tax.discovery_stage_name}</td>
                  <td style="text-align:right;">{tax.cost_multiplier:.0f}x</td>
                  <td style="text-align:right; color:{_color}; font-weight:800;">{tax.rework_days:.0f}</td>
                  <td style="text-align:right;">{tax.avoidable_rework_days:.0f}</td>
                </tr>
                """
            )

        _items.extend([
            v1_03_prediction_feedback(
                v1_03_tax_prediction.value,
                "exponential",
                "**Correct.** The chapter model doubles the correction cost at each later stage.",
                "**The chart shows exponential escalation.** A late constraint is not a one-stage fix; it propagates backward through every artifact that assumed the wrong envelope.",
            ),
            mo.Html("<div class=\"mlsysbook-panel\"><h2>Manipulation</h2></div>"),
            v1_03_discovery_stage,
            mo.as_html(_fig),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Computed Evidence</h2>
              <div class="mlsysbook-grid">
                <div class="mlsysbook-field"><strong>Discovery stage</strong>{v1_03_tax.discovery_stage}: {v1_03_tax.discovery_stage_name}</div>
                <div class="mlsysbook-field"><strong>Recommended gate</strong>{v1_03_tax.recommended_stage}: {v1_03_tax.recommended_stage_name}</div>
                <div class="mlsysbook-field"><strong>Cost multiplier</strong>{v1_03_tax.cost_multiplier:.0f}x</div>
                <div class="mlsysbook-field"><strong>Avoidable rework</strong>{v1_03_tax.avoidable_rework_days:.0f} person-days</div>
              </div>
              <table style="width:100%; border-collapse:collapse; margin-top:14px; font-size:0.84rem;">
                <thead>
                  <tr style="border-bottom:1px solid {COLORS['Border']}; color:{COLORS['TextMuted']}; text-align:left;">
                    <th>Stage</th><th>Name</th><th style="text-align:right;">Multiplier</th><th style="text-align:right;">Rework days</th><th style="text-align:right;">Avoidable days</th>
                  </tr>
                </thead>
                <tbody>{''.join(_stage_rows)}</tbody>
              </table>
            </div>
            """),
        ])
        if v1_03_tax.late_discovery:
            _items.append(mo.callout(
                mo.md(
                    f"**Failure state: iteration tax is active.** The current stage creates "
                    f"**{v1_03_tax.avoidable_rework_days:.0f} avoidable person-days** and reopens: "
                    f"{', '.join(v1_03_tax.artifacts_to_rebuild)}."
                ),
                kind="danger",
            ))
        else:
            _items.append(mo.callout(mo.md("**Recovered.** Moving the gate to the recommended stage removes the avoidable tax."), kind="success"))
        _items.extend([
            mo.accordion({
                "Math Peek / Source Model - constraint propagation cost": mo.md(f"""
The notebook uses the chapter's simplified correction model:

$$
\\text{{rework}} = \\text{{base effort}} \\times 2^{{\\text{{stage}} - 1}}
$$

For this track:

$$
{v1_03_workflow.base_rework_days:.1f} \\times 2^{{{v1_03_tax.discovery_stage - 1}}}
= {v1_03_tax.rework_days:.1f}\\;\\text{{person-days}}
$$

Stage 5 produces a 16x multiplier and stage 6 produces a 32x multiplier in the chapter framing.
""")
            }),
            source_trace({
                "api": "mlsysbook_labs.constraint_tax",
                "base_rework_days": v1_03_workflow.base_rework_days,
                "recommended_gate_stage": v1_03_workflow.recommended_gate_stage,
                "stage_count": len(v1_03_workflow.stage_names),
            }, summary="Constraint-tax evidence uses the shared workflow helper and selected profile."),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <div class="mlsysbook-callout"><strong>Report decision:</strong>
                Move the gate earlier if the avoidable tax ({v1_03_tax.avoidable_rework_days:.0f} days)
                is larger than the cost of running the gate before release pressure.</div>
            </div>
            """),
        ])
        return mo.vstack(_items)

    def v1_03_build_part_c():
        _items = [
            v1_03_part_header(
                "Part C",
                "Evaluation Gates Trade Speed For Confidence",
                "How much evidence should the workflow buy before deployment confidence is credible?",
                COLORS["OrangeLine"],
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <p>{v1_03_workflow.stakeholder} must decide how realistic the gate should be before
                 {v1_03_workflow.constraint_name} can block release. A shallow gate is fast; a realistic
                 gate makes weaker assumptions.</p>
            </div>
            """),
            mo.Html("<div class=\"mlsysbook-panel\"><h2>Prediction</h2></div>"),
            v1_03_frontier_prediction,
        ]
        if v1_03_frontier_prediction.value is None:
            _items.append(mo.callout(mo.md("Predict the bottleneck before opening the frontier chart."), kind="warn"))
            return mo.vstack(_items)

        _depth_values = list(range(5, 101, 5))
        _points = [
            iteration_frontier(
                v1_03_workflow,
                validation_depth_pct=depth,
                automation_pct=v1_03_automation.value,
                hardware_realism_pct=v1_03_realism.value,
                data_scale_pct=v1_03_data_scale.value,
            )
            for depth in _depth_values
        ]
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=[point.iteration_days for point in _points],
            y=[point.residual_risk_pct for point in _points],
            mode="lines+markers",
            marker=dict(color=COLORS["BlueLine"], size=7),
            line=dict(color=COLORS["BlueLine"], width=2.5),
            name="Validation depth sweep",
        ))
        _fig.add_trace(go.Scatter(
            x=[v1_03_frontier.iteration_days],
            y=[v1_03_frontier.residual_risk_pct],
            mode="markers",
            marker=dict(color=COLORS["RedLine"], size=14, line=dict(color="white", width=2)),
            name="Current workflow",
        ))
        _fig.add_hline(
            y=v1_03_risk_budget_pct,
            line_dash="dash",
            line_color=COLORS["RedLine"],
            annotation_text="risk budget",
        )
        _fig.add_vline(
            x=v1_03_cycle_budget_days,
            line_dash="dot",
            line_color=COLORS["TextMuted"],
            annotation_text="cycle budget",
        )
        _fig.update_layout(
            height=360,
            xaxis=dict(title="Iteration time (days)", gridcolor="#f1f5f9"),
            yaxis=dict(title="Residual deployment risk (%)", gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=40, b=55),
        )
        apply_plotly_theme(_fig)

        _risk_over = v1_03_frontier.residual_risk_pct > v1_03_risk_budget_pct
        _cycle_over = v1_03_frontier.iteration_days > v1_03_cycle_budget_days
        if _risk_over:
            _consequence = mo.callout(
                mo.md(
                    f"**Fast but blind.** Residual risk is **{v1_03_frontier.residual_risk_pct:.1f}%**, "
                    f"above the {v1_03_risk_budget_pct:.1f}% budget. Add realism, data scale, or validation depth."
                ),
                kind="danger",
            )
        elif _cycle_over:
            _consequence = mo.callout(
                mo.md(
                    f"**Confident but slow.** Iteration time is **{v1_03_frontier.iteration_days:.1f} days**, "
                    f"above the {v1_03_cycle_budget_days:.1f}-day cycle budget. Add automation or reduce scope."
                ),
                kind="warn",
            )
        else:
            _consequence = mo.callout(
                mo.md("**Balanced gate.** Current settings stay inside the risk and cycle budgets."),
                kind="success",
            )

        _items.extend([
            v1_03_prediction_feedback(
                v1_03_frontier_prediction.value,
                v1_03_frontier.bottleneck,
                f"**Correct.** The current bottleneck is **{v1_03_frontier.bottleneck}**.",
                f"**Measured bottleneck: {v1_03_frontier.bottleneck}.** The weakest validation dimension controls the residual risk.",
            ),
            mo.Html("<div class=\"mlsysbook-panel\"><h2>Manipulation</h2></div>"),
            mo.hstack([v1_03_validation_depth, v1_03_automation], justify="start", gap="2rem"),
            mo.hstack([v1_03_realism, v1_03_data_scale], justify="start", gap="2rem"),
            mo.as_html(_fig),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Computed Evidence</h2>
              <div class="mlsysbook-grid">
                <div class="mlsysbook-field"><strong>Iteration time</strong>{v1_03_frontier.iteration_days:.1f} days</div>
                <div class="mlsysbook-field"><strong>Cycle budget</strong>{v1_03_cycle_budget_days:.1f} days</div>
                <div class="mlsysbook-field"><strong>Confidence</strong>{v1_03_frontier.confidence_pct:.1f}%</div>
                <div class="mlsysbook-field"><strong>Residual risk</strong>{v1_03_frontier.residual_risk_pct:.1f}%</div>
                <div class="mlsysbook-field"><strong>Risk budget</strong>{v1_03_risk_budget_pct:.1f}%</div>
                <div class="mlsysbook-field"><strong>Bottleneck</strong>{v1_03_frontier.bottleneck}</div>
              </div>
            </div>
            """),
            _consequence,
            mo.accordion({
                "Math Peek / Source Model - validation frontier": mo.md(f"""
The source model estimates confidence from four gate dimensions:

$$
\\text{{confidence}} = 18 + 0.28d + 0.27r + 0.22s + 0.12a
$$

Residual risk falls toward a track-specific floor:

$$
\\text{{risk}} = \\max(\\text{{floor}}, \\text{{base risk}} - 0.62 \\cdot \\text{{confidence}})
$$

Current values: depth={v1_03_frontier.validation_depth_pct:.0f}%, realism={v1_03_frontier.hardware_realism_pct:.0f}%,
data scale={v1_03_frontier.data_scale_pct:.0f}%, automation={v1_03_frontier.automation_pct:.0f}%.
""")
            }),
            source_trace({
                "api": "mlsysbook_labs.iteration_frontier",
                "base_cycle_days": v1_03_workflow.base_cycle_days,
                "base_residual_risk_pct": v1_03_workflow.base_residual_risk_pct,
                "min_residual_risk_pct": v1_03_workflow.min_residual_risk_pct,
                "derived_risk_budget_pct": round(v1_03_risk_budget_pct, 2),
            }, summary="Frontier evidence uses the shared helper plus a notebook-local derived risk budget."),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <div class="mlsysbook-callout"><strong>Report decision:</strong>
                The gate should focus on {v1_03_frontier.bottleneck} until residual risk is below
                {v1_03_risk_budget_pct:.1f}% without pushing cycle time beyond {v1_03_cycle_budget_days:.1f} days.</div>
            </div>
            """),
        ])
        return mo.vstack(_items)

    def v1_03_build_part_d():
        _recommended_gate = v1_03_recommended_gate()
        _items = [
            v1_03_part_header(
                "Part D",
                "Workflow Policy Is System Design",
                f"Which gate and release rule should {v1_03_workflow.label} adopt?",
                COLORS["GreenLine"],
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Scenario</h2>
              <p>The policy decides when evidence can block release. For this track, that means the policy must
                 name the gate, release rule, rollback rule, and the residual blind spot the team still accepts.</p>
            </div>
            """),
            mo.Html("<div class=\"mlsysbook-panel\"><h2>Prediction</h2></div>"),
            v1_03_policy_prediction,
        ]
        if v1_03_policy_prediction.value is None:
            _items.append(mo.callout(mo.md("Predict the non-negotiable gate before comparing policies."), kind="warn"))
            return mo.vstack(_items)

        _gate_rows = []
        for gate in v1_03_workflow.gate_options:
            _tax = constraint_tax(v1_03_workflow, discovery_stage=gate.stage)
            _color = COLORS["GreenLine"] if gate.gate_id == v1_03_policy.gate_id else COLORS["TextSec"]
            _gate_rows.append(
                f"""
                <tr>
                  <td style="color:{_color}; font-weight:800;">{gate.label}</td>
                  <td>{_tax.discovery_stage_name}</td>
                  <td style="text-align:right;">{_tax.rework_days:.0f} days</td>
                  <td>{gate.validation_focus}</td>
                  <td>{gate.residual_risk}</td>
                </tr>
                """
            )

        _late_policy = v1_03_policy.gate_stage > v1_03_workflow.recommended_gate_stage
        _items.extend([
            v1_03_prediction_feedback(
                v1_03_policy_prediction.value,
                _recommended_gate.gate_id,
                f"**Correct direction.** {_recommended_gate.label} is the earliest policy gate aligned to the recommended stage.",
                f"**Compare against {_recommended_gate.label}.** Later gates can still be useful, but they let more assumptions harden before evidence can block release.",
            ),
            mo.Html("<div class=\"mlsysbook-panel\"><h2>Manipulation</h2></div>"),
            mo.hstack([v1_03_gate_choice, v1_03_release_policy], justify="start", gap="2rem"),
            v1_03_rollback_rule,
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Policy Evidence</h2>
              <div class="mlsysbook-grid">
                <div class="mlsysbook-field"><strong>Selected gate</strong>{v1_03_policy.gate_label}</div>
                <div class="mlsysbook-field"><strong>Gate stage</strong>{v1_03_policy.gate_stage_name}</div>
                <div class="mlsysbook-field"><strong>Rework at gate</strong>{v1_03_policy.rework_days_at_gate:.0f} days</div>
                <div class="mlsysbook-field"><strong>Residual risk</strong>{v1_03_policy.residual_risk_pct:.1f}%</div>
                <div class="mlsysbook-field"><strong>Release policy</strong>{v1_03_policy.release_policy}</div>
                <div class="mlsysbook-field"><strong>Rollback rule</strong>{v1_03_policy.rollback_rule}</div>
              </div>
              <div class="mlsysbook-callout"><strong>Policy summary:</strong> {v1_03_policy.policy_summary}</div>
              <table style="width:100%; border-collapse:collapse; margin-top:14px; font-size:0.84rem;">
                <thead>
                  <tr style="border-bottom:1px solid {COLORS['Border']}; text-align:left; color:{COLORS['TextMuted']};">
                    <th>Gate</th><th>Stage</th><th style="text-align:right;">Rework</th><th>Validation focus</th><th>Blind spot</th>
                  </tr>
                </thead>
                <tbody>{''.join(_gate_rows)}</tbody>
              </table>
            </div>
            """),
        ])
        if _late_policy:
            _items.append(mo.callout(
                mo.md(
                    f"**Policy boundary crossed.** {v1_03_policy.gate_label} is after the recommended gate, "
                    f"so the policy accepts {v1_03_policy.rework_days_at_gate:.0f} days of rework before evidence can block release."
                ),
                kind="danger",
            ))
        else:
            _items.append(mo.callout(
                mo.md("**Policy blocks early enough.** Evidence can stop the workflow before release debt compounds."),
                kind="success",
            ))
        _items.extend([
            mo.accordion({
                "Math Peek / Source Model - policy tuple": mo.md(f"""
The policy is a system design tuple, not paperwork:

$$
\\text{{policy}} =
(\\text{{gate timing}}, \\text{{evidence requirement}}, \\text{{rollout}}, \\text{{rollback}}, \\text{{blind spot}})
$$

Current tuple:

- Gate timing: **{v1_03_policy.gate_label}** at **{v1_03_policy.gate_stage_name}**
- Evidence requirement: **{v1_03_policy.release_policy}**
- Rollback: **{v1_03_policy.rollback_rule}**
- Residual blind spot: **{v1_03_policy.blind_spot}**
""")
            }),
            source_trace({
                "api": "mlsysbook_labs.workflow_policy",
                "gate_id": v1_03_policy.gate_id,
                "release_policy": v1_03_policy.release_policy,
                "rollback_rule": v1_03_policy.rollback_rule,
                "blind_spot_source": "WorkflowGate.residual_risk",
            }, summary="Policy evidence packages selected gate metadata and frontier risk."),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Checkpoint</h2>
              <div class="mlsysbook-callout"><strong>Report decision:</strong>
                Defend this policy only if the release evidence can block {v1_03_workflow.constraint_name}
                before {v1_03_policy.blind_spot} becomes the next unknown.</div>
            </div>
            """),
        ])
        return mo.vstack(_items)

    def v1_03_build_synthesis():
        _complete = (
            v1_03_gate_prediction.value is not None
            and v1_03_tax_prediction.value is not None
            and v1_03_frontier_prediction.value is not None
            and v1_03_policy_prediction.value is not None
            and bool(str(v1_03_reflection.value or "").strip())
        )
        _status = "READY" if _complete else "IN PROGRESS"
        return mo.vstack([
            v1_03_part_header(
                "Synthesis",
                "Release Memo With Residual Blind Spot",
                "What policy will this track carry forward?",
                COLORS["BlueLine"],
            ),
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Release Policy Memo</h2>
              <div class="mlsysbook-grid">
                <div class="mlsysbook-field"><strong>Track</strong>{v1_03_workflow.label}</div>
                <div class="mlsysbook-field"><strong>Constraint</strong>{v1_03_workflow.constraint_name}</div>
                <div class="mlsysbook-field"><strong>Discovery stage</strong>{v1_03_tax.discovery_stage_name}</div>
                <div class="mlsysbook-field"><strong>Avoidable rework</strong>{v1_03_tax.avoidable_rework_days:.0f} person-days</div>
                <div class="mlsysbook-field"><strong>Iteration time</strong>{v1_03_frontier.iteration_days:.1f} days</div>
                <div class="mlsysbook-field"><strong>Residual risk</strong>{v1_03_frontier.residual_risk_pct:.1f}%</div>
                <div class="mlsysbook-field"><strong>Policy</strong>{v1_03_policy.policy_summary}</div>
                <div class="mlsysbook-field"><strong>Blind spot</strong>{v1_03_policy.blind_spot}</div>
              </div>
            </div>
            """),
            mo.Html("<div class=\"mlsysbook-panel\"><h2>Final Checkpoint</h2></div>"),
            v1_03_reflection,
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Big Takeaways</h2>
              <ul class="mlsysbook-list">
                <li><strong>Constraints propagate backward.</strong> A late deployment failure invalidates upstream data, model, validation, release, and monitoring artifacts.</li>
                <li><strong>The iteration tax is measurable.</strong> The same stage move changes the cost multiplier and the artifacts to rebuild.</li>
                <li><strong>Evaluation gates buy evidence.</strong> Speed without realism leaves residual deployment risk; realism without automation slows learning.</li>
                <li><strong>Workflow policy is system design.</strong> The selected gate determines when evidence can block the system.</li>
              </ul>
            </div>
            """),
            mo.Html(f"""
            <div class="lab-hud">
                <span class="hud-label">LAB</span>
                <span class="hud-value">03 &middot; Constraint Tax</span>
                <span class="hud-label">TRACK</span>
                <span class="hud-value">{v1_03_workflow.label}</span>
                <span style="flex:1;"></span>
                <span class="hud-label">ARTIFACT</span>
                <span class="hud-value">{v1_03_workflow.report_artifact}</span>
                <span class="hud-label">STATUS</span>
                <span class="hud-active">{_status}</span>
            </div>
            """),
        ])

    def build_synthesis():
        return v1_03_build_synthesis()

    _tabs = mo.ui.tabs({
        "Part A: Propagation": v1_03_build_part_a(),
        "Part B: Iteration Tax": v1_03_build_part_b(),
        "Part C: Gate Confidence": v1_03_build_part_c(),
        "Part D: Workflow Policy": v1_03_build_part_d(),
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


# ===========================================================================
# ZONE D: SYNTHESIS AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    v1_03_frontier,
    v1_03_frontier_prediction,
    v1_03_gate_prediction,
    v1_03_policy,
    v1_03_policy_prediction,
    v1_03_profile,
    v1_03_reflection,
    v1_03_risk_budget_pct,
    v1_03_tax,
    v1_03_tax_prediction,
    v1_03_variant,
    v1_03_workflow,
):
    _ledger_ready = (
        v1_03_gate_prediction.value is not None
        and v1_03_tax_prediction.value is not None
        and v1_03_frontier_prediction.value is not None
        and v1_03_policy_prediction.value is not None
        and bool(str(v1_03_reflection.value or "").strip())
    )
    if _ledger_ready:
        ledger.save(chapter=3, design={
            "chapter": "v1_03",
            "track_id": v1_03_profile.track_id,
            "scenario_id": v1_03_variant.scenario_id,
            "hardware_ref": v1_03_workflow.hardware_ref,
            "model_ref": v1_03_workflow.model_ref,
            "completed": True,
            "gate_prediction": v1_03_gate_prediction.value,
            "tax_prediction": v1_03_tax_prediction.value,
            "frontier_prediction": v1_03_frontier_prediction.value,
            "policy_prediction": v1_03_policy_prediction.value,
            "constraint_name": v1_03_workflow.constraint_name,
            "discovery_stage": v1_03_tax.discovery_stage_name,
            "selected_gate_id": v1_03_policy.gate_id,
            "avoidable_rework_days": v1_03_tax.avoidable_rework_days,
            "iteration_days": v1_03_frontier.iteration_days,
            "confidence_pct": v1_03_frontier.confidence_pct,
            "residual_risk_pct": v1_03_frontier.residual_risk_pct,
            "risk_budget_pct": v1_03_risk_budget_pct,
            "release_policy": v1_03_policy.release_policy,
            "rollback_rule": v1_03_policy.rollback_rule,
            "policy_summary": v1_03_policy.policy_summary,
            "blind_spot": v1_03_policy.blind_spot,
        })

    _status = "SAVED" if _ledger_ready else "ACTIVE"
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">03 &middot; Constraint Tax</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_03_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">ARTIFACT</span>
        <span class="hud-value">{v1_03_workflow.report_artifact}</span>
        <span class="hud-label">LEDGER</span>
        <span class="hud-active">{_status}</span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    report_export_panel,
    v1_03_automation,
    v1_03_cycle_budget_days,
    v1_03_data_scale,
    v1_03_frontier,
    v1_03_frontier_prediction,
    v1_03_gate_prediction,
    v1_03_metadata,
    v1_03_policy,
    v1_03_policy_prediction,
    v1_03_profile,
    v1_03_realism,
    v1_03_reflection,
    v1_03_risk_budget_pct,
    v1_03_tax,
    v1_03_tax_prediction,
    v1_03_variant,
    v1_03_validation_depth,
    v1_03_workflow,
):
    _incomplete = []
    if v1_03_gate_prediction.value is None:
        _incomplete.append("Part A gate timing prediction")
    if v1_03_tax_prediction.value is None:
        _incomplete.append("Part B iteration-tax prediction")
    if v1_03_frontier_prediction.value is None:
        _incomplete.append("Part C validation-bottleneck prediction")
    if v1_03_policy_prediction.value is None:
        _incomplete.append("Part D policy-gate prediction")
    if not str(v1_03_reflection.value or "").strip():
        _incomplete.append("Synthesis release memo blind spot")

    _report = build_lab_report(
        v1_03_metadata,
        track=v1_03_profile.label,
        scenario=v1_03_variant.workload_summary,
        learning_objectives=(
            "Trace how deployment constraints propagate through workflow stages.",
            "Measure the iteration tax created by late constraint discovery.",
            "Compare validation realism, automation, iteration time, and residual risk.",
            "Choose a track-specific workflow gate, release policy, rollback rule, and blind spot.",
        ),
        predictions={
            "gate_timing": v1_03_gate_prediction.value,
            "iteration_tax_shape": v1_03_tax_prediction.value,
            "validation_bottleneck": v1_03_frontier_prediction.value,
            "policy_gate": v1_03_policy_prediction.value,
        },
        knob_settings={
            "discovery_stage": v1_03_tax.discovery_stage,
            "validation_depth_pct": v1_03_validation_depth.value,
            "automation_pct": v1_03_automation.value,
            "hardware_realism_pct": v1_03_realism.value,
            "data_scale_pct": v1_03_data_scale.value,
            "selected_gate": v1_03_policy.gate_id,
            "release_policy": v1_03_policy.release_policy,
            "rollback_rule": v1_03_policy.rollback_rule,
        },
        evidence_summary={
            "hardware_ref": v1_03_workflow.hardware_ref,
            "model_ref": v1_03_workflow.model_ref,
            "constraint_name": v1_03_workflow.constraint_name,
            "cost_multiplier": v1_03_tax.cost_multiplier,
            "rework_days": v1_03_tax.rework_days,
            "avoidable_rework_days": v1_03_tax.avoidable_rework_days,
            "iteration_days": v1_03_frontier.iteration_days,
            "cycle_budget_days": v1_03_cycle_budget_days,
            "confidence_pct": v1_03_frontier.confidence_pct,
            "residual_risk_pct": v1_03_frontier.residual_risk_pct,
            "risk_budget_pct": v1_03_risk_budget_pct,
            "bottleneck": v1_03_frontier.bottleneck,
            "selected_gate": v1_03_policy.gate_label,
            "release_policy": v1_03_policy.release_policy,
            "rollback_rule": v1_03_policy.rollback_rule,
            "blind_spot": v1_03_policy.blind_spot,
        },
        final_decision=v1_03_policy.policy_summary,
        big_takeaways=(
            "Deployment constraints propagate backward through the whole workflow.",
            "Late discovery creates a measurable iteration tax.",
            "Evaluation gates trade iteration speed for deployment confidence.",
            "A workflow policy must name both release evidence and residual blind spot.",
        ),
        reflections={
            "student_reflection": v1_03_reflection.value,
            "blind_spot": v1_03_policy.blind_spot,
            "report_artifact": v1_03_workflow.report_artifact,
        },
        residual_risk=v1_03_policy.blind_spot,
        source_trace={
            "track_id": v1_03_profile.track_id,
            "scenario_id": v1_03_variant.scenario_id,
            "hardware_ref": v1_03_variant.hardware_ref,
            "model_ref": v1_03_variant.model_ref,
            "shared_helpers": "workflow_track_profile, constraint_tax, iteration_frontier, workflow_policy",
            "source_policy": v1_03_profile.source_policy,
        },
        result_snapshot={
            "workflow_profile": v1_03_workflow,
            "constraint_tax": v1_03_tax,
            "iteration_frontier": v1_03_frontier,
            "workflow_policy": v1_03_policy,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-03 workflow memo is generated locally from the selected track, "
                "your inputs, and the computed evidence."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
