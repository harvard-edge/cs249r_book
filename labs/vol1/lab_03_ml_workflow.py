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
                    3 Parts + Memo &middot; ~45 min
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
                <span class="badge badge-info">Late Discovery</span>
                <span class="badge badge-warn">Iteration Frontier</span>
                <span class="badge badge-fail">Release Gate</span>
            </div>
        </div>
        """),
        track_context(v1_03_profile),
        track_arc_context(v1_03_profile, v1_03_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_03_workflow):
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
            <div style="margin-bottom: 3px;">1. <strong>Quantify late discovery:</strong>
                compute rework cost when {v1_03_workflow.constraint_name} is tested late.</div>
            <div style="margin-bottom: 3px;">2. <strong>Balance speed and confidence:</strong>
                compare iteration time with residual deployment risk.</div>
            <div style="margin-bottom: 3px;">3. <strong>Write a workflow policy:</strong>
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
    """)
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
        v1_03_realism,
        v1_03_validation_depth,
    )


@app.cell(hide_code=True)
def _(mo, v1_03_workflow):
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
    return (v1_03_gate_choice, v1_03_release_policy, v1_03_rollback_rule)


@app.cell(hide_code=True)
def _(mo):
    v1_03_reflection = mo.ui.text_area(
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
    return (v1_03_frontier, v1_03_policy, v1_03_tax)


# ===========================================================================
# ZONE C: PARTS
# ===========================================================================


@app.cell(hide_code=True)
def _(COLORS, constraint_tax, mo, v1_03_gate_prediction, v1_03_tax, v1_03_workflow):
    _stage_rows = []
    for idx, stage in enumerate(v1_03_workflow.stage_names, start=1):
        _tax = constraint_tax(v1_03_workflow, discovery_stage=idx)
        _color = COLORS["GreenLine"] if idx <= v1_03_workflow.recommended_gate_stage else COLORS["RedLine"]
        _stage_rows.append(
            f"""
            <tr>
              <td>{idx}</td>
              <td>{stage}</td>
              <td style="text-align:right;">{_tax.cost_multiplier:.0f}x</td>
              <td style="text-align:right; color:{_color}; font-weight:800;">{_tax.rework_days:.0f} days</td>
            </tr>
            """
        )
    _artifacts = "".join(f"<li>{item}</li>" for item in v1_03_tax.artifacts_to_rebuild)
    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Constraint Propagation</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            What does it cost to discover {v1_03_workflow.constraint_name} late?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>A deployment constraint discovered late invalidates every artifact built on the wrong assumption.</li>
            <li>The teaching cost model doubles rework at each later stage: cost = base * 2^(stage - 1).</li>
            <li>For {v1_03_workflow.label}, the constraint is {v1_03_workflow.constraint_name}.</li>
          </ul>
          <div class="mlsysbook-callout"><strong>Scenario:</strong> {v1_03_workflow.failure_story}</div>
        </div>
        """),
        v1_03_gate_prediction,
        mo.Html('<div class="mlsysbook-panel"><h2>Try It</h2></div>'),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Discovery stage</strong>{v1_03_tax.discovery_stage}: {v1_03_tax.discovery_stage_name}</div>
            <div class="mlsysbook-field"><strong>Recommended gate</strong>{v1_03_tax.recommended_stage}: {v1_03_tax.recommended_stage_name}</div>
            <div class="mlsysbook-field"><strong>Rework cost</strong>{v1_03_tax.rework_days:.0f} person-days</div>
            <div class="mlsysbook-field"><strong>Avoidable rework</strong>{v1_03_tax.avoidable_rework_days:.0f} person-days</div>
          </div>
          <div class="mlsysbook-callout"><strong>Artifacts to rebuild:</strong><ul class="mlsysbook-list">{_artifacts}</ul></div>
          <table style="width:100%; border-collapse:collapse; margin-top:14px; font-size:0.88rem;">
            <thead>
              <tr style="border-bottom:1px solid {COLORS['Border']}; color:{COLORS['TextMuted']}; text-align:left;">
                <th>Stage</th><th>Name</th><th style="text-align:right;">Multiplier</th><th style="text-align:right;">Rework</th>
              </tr>
            </thead>
            <tbody>{''.join(_stage_rows)}</tbody>
          </table>
        </div>
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    iteration_frontier,
    mo,
    v1_03_automation,
    v1_03_data_scale,
    v1_03_discovery_stage,
    v1_03_frontier,
    v1_03_realism,
    v1_03_validation_depth,
    v1_03_workflow,
):
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
    _fig.update_layout(
        height=340,
        xaxis=dict(title="Iteration time (days)", gridcolor="#f1f5f9"),
        yaxis=dict(title="Residual deployment risk (%)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=40, b=50),
    )
    apply_plotly_theme(_fig)
    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Iteration Frontier</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            How much validation should the team buy before iteration gets too slow?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>More validation usually lowers residual risk but increases iteration time.</li>
            <li>Automation can buy confidence without slowing every cycle as much.</li>
            <li>The bottleneck is the weakest validation dimension in the current workflow.</li>
          </ul>
        </div>
        """),
        mo.hstack([v1_03_validation_depth, v1_03_automation], justify="start", gap="2rem"),
        mo.hstack([v1_03_realism, v1_03_data_scale], justify="start", gap="2rem"),
        v1_03_discovery_stage,
        mo.as_html(_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Iteration time</strong>{v1_03_frontier.iteration_days:.1f} days</div>
            <div class="mlsysbook-field"><strong>Confidence</strong>{v1_03_frontier.confidence_pct:.1f}%</div>
            <div class="mlsysbook-field"><strong>Residual risk</strong>{v1_03_frontier.residual_risk_pct:.1f}%</div>
            <div class="mlsysbook-field"><strong>Bottleneck</strong>{v1_03_frontier.bottleneck}</div>
          </div>
        </div>
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    constraint_tax,
    mo,
    source_trace,
    v1_03_gate_choice,
    v1_03_policy,
    v1_03_reflection,
    v1_03_release_policy,
    v1_03_rollback_rule,
    v1_03_workflow,
):
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
    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Workflow Policy</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which gate and release policy should {v1_03_workflow.label} adopt?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>A policy is not just a checklist; it determines when evidence can block release.</li>
            <li>Every gate has a residual blind spot even when it lowers rework.</li>
            <li>The report should connect gate timing, release policy, rollback, and residual risk.</li>
          </ul>
        </div>
        """),
        mo.hstack([v1_03_gate_choice, v1_03_release_policy], justify="start", gap="2rem"),
        v1_03_rollback_rule,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
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
        mo.Html('<div class="mlsysbook-panel"><h2>Reflection</h2></div>'),
        v1_03_reflection,
    ])
    return


# ===========================================================================
# ZONE D: SYNTHESIS AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    ledger,
    mo,
    v1_03_frontier,
    v1_03_gate_prediction,
    v1_03_policy,
    v1_03_profile,
    v1_03_tax,
    v1_03_variant,
    v1_03_workflow,
):
    if v1_03_gate_prediction.value is not None:
        ledger.save(chapter=3, design={
            "chapter": "v1_03",
            "track_id": v1_03_profile.track_id,
            "scenario_id": v1_03_variant.scenario_id,
            "hardware_ref": v1_03_workflow.hardware_ref,
            "model_ref": v1_03_workflow.model_ref,
            "completed": True,
            "gate_prediction": v1_03_gate_prediction.value,
            "constraint_name": v1_03_workflow.constraint_name,
            "discovery_stage": v1_03_tax.discovery_stage_name,
            "avoidable_rework_days": v1_03_tax.avoidable_rework_days,
            "iteration_days": v1_03_frontier.iteration_days,
            "residual_risk_pct": v1_03_frontier.residual_risk_pct,
            "policy_summary": v1_03_policy.policy_summary,
        })

    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Synthesis</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Track</strong>{v1_03_workflow.label}</div>
            <div class="mlsysbook-field"><strong>Constraint</strong>{v1_03_workflow.constraint_name}</div>
            <div class="mlsysbook-field"><strong>Avoidable rework</strong>{v1_03_tax.avoidable_rework_days:.0f} person-days</div>
            <div class="mlsysbook-field"><strong>Policy</strong>{v1_03_policy.policy_summary}</div>
          </div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Big Takeaways</h2>
          <ul class="mlsysbook-list">
            <li><strong>Constraints propagate backward.</strong> A late deployment failure invalidates upstream data, model, validation, and release artifacts.</li>
            <li><strong>Workflow policy is system design.</strong> The right gate depends on the selected track's physical constraint.</li>
            <li><strong>Fast iteration still needs realism.</strong> Automation only helps if it tests the actual deployment wall.</li>
          </ul>
        </div>
        """),
        mo.Html(f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">03 &middot; Constraint Tax</span>
            <span class="hud-label">TRACK</span>
            <span class="hud-value">{v1_03_profile.label}</span>
            <span style="flex:1;"></span>
            <span class="hud-label">ARTIFACT</span>
            <span class="hud-value">{v1_03_workflow.report_artifact}</span>
            <span class="hud-label">STATUS</span>
            <span class="hud-active">ACTIVE</span>
        </div>
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    report_export_panel,
    v1_03_frontier,
    v1_03_gate_prediction,
    v1_03_metadata,
    v1_03_policy,
    v1_03_profile,
    v1_03_reflection,
    v1_03_tax,
    v1_03_variant,
    v1_03_workflow,
):
    _incomplete = []
    if v1_03_gate_prediction.value is None:
        _incomplete.append("Part A gate timing prediction")
    if not str(v1_03_reflection.value or "").strip():
        _incomplete.append("Part C reflection")

    _report = build_lab_report(
        v1_03_metadata,
        track=v1_03_profile.label,
        scenario=v1_03_variant.workload_summary,
        learning_objectives=(
            "Quantify the rework cost of discovering deployment constraints late.",
            "Compare validation realism, automation, iteration time, and residual risk.",
            "Choose a track-specific workflow gate, release policy, and rollback rule.",
        ),
        predictions={
            "gate_timing": v1_03_gate_prediction.value,
        },
        knob_settings={
            "discovery_stage": v1_03_tax.discovery_stage,
            "selected_gate": v1_03_policy.gate_id,
            "release_policy": v1_03_policy.release_policy,
            "rollback_rule": v1_03_policy.rollback_rule,
        },
        evidence_summary={
            "hardware_ref": v1_03_workflow.hardware_ref,
            "model_ref": v1_03_workflow.model_ref,
            "constraint_name": v1_03_workflow.constraint_name,
            "rework_days": v1_03_tax.rework_days,
            "avoidable_rework_days": v1_03_tax.avoidable_rework_days,
            "iteration_days": v1_03_frontier.iteration_days,
            "residual_risk_pct": v1_03_frontier.residual_risk_pct,
            "bottleneck": v1_03_frontier.bottleneck,
        },
        final_decision=v1_03_policy.policy_summary,
        big_takeaways=(
            "Deployment constraints propagate backward through the whole workflow.",
            "The selected track changes which gate must become non-negotiable.",
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
            "shared_helper": "mlsysbook_labs.workflow",
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
