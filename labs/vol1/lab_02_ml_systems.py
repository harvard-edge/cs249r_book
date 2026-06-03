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
        deployment_mitigation,
        deployment_track_profile,
        evaluate_deployment_envelope,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        sweep_deployment_knob,
        track_context,
        track_selector,
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
        deployment_mitigation,
        deployment_track_profile,
        evaluate_deployment_envelope,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        sweep_deployment_knob,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_02_metadata = get_lab_metadata("vol1/lab_02_ml_systems.py")
    return (v1_02_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_02_track_picker = track_selector(default=_default_track)
    v1_02_track_picker
    return (v1_02_track_picker,)


@app.cell
def _(
    deployment_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_02_track_picker,
):
    v1_02_track_id = v1_02_track_picker.value
    v1_02_profile = get_track_profile(v1_02_track_id)
    v1_02_variant = get_lab_track_variant("v1_02_physics_of_deployment", v1_02_profile.track_id)
    v1_02_hardware = resolve_mlsysim_ref(v1_02_variant.hardware_ref)
    v1_02_model = resolve_mlsysim_ref(v1_02_variant.model_ref)
    v1_02_deployment = deployment_track_profile(
        v1_02_profile,
        v1_02_variant,
        v1_02_hardware,
        v1_02_model,
    )
    return (
        v1_02_deployment,
        v1_02_hardware,
        v1_02_model,
        v1_02_profile,
        v1_02_track_id,
        v1_02_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
    v1_02_deployment,
    v1_02_metadata,
    v1_02_profile,
    v1_02_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 02
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                The Physics of Deployment
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Memory &middot; Latency &middot; Energy &middot; Power &middot; Bandwidth &middot; Cost
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 820px; line-height: 1.65;">
                {v1_02_variant.workload_summary} This lab asks which physical wall
                appears first when the selected track's workload is pushed beyond demo scale.
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
                    {v1_02_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_02_deployment.hardware_ref}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">First Wall</span>
                <span class="badge badge-warn">Envelope Sweep</span>
                <span class="badge badge-fail">Placement Risk</span>
            </div>
        </div>
        """),
        track_context(v1_02_profile),
        source_trace(
            {
                "lab_id": v1_02_metadata.lab_id,
                "track_id": v1_02_profile.track_id,
                "hardware_ref": v1_02_variant.hardware_ref,
                "model_ref": v1_02_variant.model_ref,
                "system_ref": v1_02_variant.system_ref or "single-device profile",
                "shared_helper": "mlsysbook_labs.deployment",
                "source_policy": v1_02_profile.source_policy,
            },
            summary="V1-02 computes envelope evidence through MLSysIM refs and mlsysbook_labs.deployment.",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_02_deployment):
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
            <div style="margin-bottom: 3px;">1. <strong>Diagnose a physical wall:</strong>
                compare memory, flash/OTA, latency, energy, power, bandwidth, and cost with units.</div>
            <div style="margin-bottom: 3px;">2. <strong>Sweep a workload knob:</strong>
                find the threshold where feasibility breaks for {v1_02_deployment.workload_knob}.</div>
            <div style="margin-bottom: 3px;">3. <strong>Choose a placement:</strong>
                explain which wall is avoided and which residual risk appears instead.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                Which physical constraint becomes non-negotiable first for {v1_02_deployment.label},
                and what deployment choice changes the trade-off?
            </div>
        </div>
    </div>
    """)
    return


# ===========================================================================
# ZONE B: CONTROLS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_02_deployment):
    v1_02_first_wall_pred = mo.ui.radio(
        options={
            "Memory or flash/OTA fit binds first": "memory_or_flash",
            "Latency deadline binds first": "latency",
            "Energy or thermal power binds first": "energy_or_power",
            "Bandwidth or cost binds first": "bandwidth_or_cost",
        },
        label=f"{v1_02_deployment.label}: which wall do you expect to bind first?",
    )
    v1_02_first_wall_pred
    return (v1_02_first_wall_pred,)


@app.cell(hide_code=True)
def _(mo, v1_02_deployment):
    v1_02_workload = mo.ui.slider(
        start=v1_02_deployment.knob_min,
        stop=v1_02_deployment.knob_max,
        value=v1_02_deployment.default_knob,
        step=v1_02_deployment.knob_step,
        label=f"{v1_02_deployment.workload_knob} ({v1_02_deployment.workload_unit})",
    )
    v1_02_workload
    return (v1_02_workload,)


@app.cell(hide_code=True)
def _(mo, v1_02_deployment):
    _options = {option.label: option.placement_id for option in v1_02_deployment.placement_options}
    v1_02_placement = mo.ui.dropdown(
        options=_options,
        value=v1_02_deployment.placement_options[0].label,
        label="Placement or mitigation path",
    )
    v1_02_placement
    return (v1_02_placement,)


@app.cell(hide_code=True)
def _(mo):
    v1_02_reflection = mo.ui.text_area(
        label="Reflection",
        placeholder="Name the wall you would defend in the memo and the residual risk you would still test.",
        full_width=True,
    )
    return (v1_02_reflection,)


@app.cell
def _(
    deployment_mitigation,
    evaluate_deployment_envelope,
    sweep_deployment_knob,
    v1_02_deployment,
    v1_02_placement,
    v1_02_workload,
):
    v1_02_result = evaluate_deployment_envelope(
        v1_02_deployment,
        workload_value=v1_02_workload.value,
        placement_id=v1_02_placement.value,
    )
    v1_02_sweep = sweep_deployment_knob(
        v1_02_deployment,
        placement_id=v1_02_placement.value,
        samples=40,
    )
    v1_02_mitigation = deployment_mitigation(
        v1_02_deployment,
        v1_02_result,
        placement_id=v1_02_placement.value,
    )
    return (v1_02_mitigation, v1_02_result, v1_02_sweep)


# ===========================================================================
# ZONE C: PARTS
# ===========================================================================


@app.cell(hide_code=True)
def _(COLORS, mo, v1_02_deployment, v1_02_first_wall_pred, v1_02_result, v1_02_variant):
    def _checks_table(checks):
        rows = []
        for check in checks:
            status = "PASS" if check.feasible else "WALL"
            color = COLORS["GreenLine"] if check.feasible else COLORS["RedLine"]
            rows.append(
                f"""
                <tr>
                    <td>{check.name}</td>
                    <td style="text-align:right;">{check.value:.3g} {check.unit}</td>
                    <td style="text-align:right;">{check.limit:.3g} {check.unit}</td>
                    <td style="text-align:right; color:{color}; font-weight:800;">{check.headroom_pct:.1f}%</td>
                    <td style="color:{color}; font-weight:800;">{status}</td>
                </tr>
                """
            )
        return "".join(rows)

    _pred = v1_02_first_wall_pred.value or "not selected"
    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: First Wall</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which physical resource becomes the first hard limit for {v1_02_deployment.label}?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Deployment feasibility is a vector of unitful constraints, not a single model score.</li>
            <li>The first wall is the constraint with the least headroom after comparing value to limit.</li>
            <li>For this track, the same model family is interpreted through {v1_02_deployment.hardware_ref} and {v1_02_deployment.model_ref}.</li>
          </ul>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Equation</strong>headroom = (limit - value) / limit</div>
            <div class="mlsysbook-field"><strong>Track scenario</strong>{v1_02_variant.workload_summary}</div>
          </div>
        </div>
        """),
        v1_02_first_wall_pred,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Prediction</strong>{_pred}</div>
            <div class="mlsysbook-field"><strong>Actual first wall</strong>{v1_02_result.first_wall}</div>
            <div class="mlsysbook-field"><strong>Feasible</strong>{'yes' if v1_02_result.feasible else 'no'}</div>
            <div class="mlsysbook-field"><strong>Placement</strong>{v1_02_result.placement_label}</div>
          </div>
          <table style="width:100%; border-collapse:collapse; margin-top:14px; font-size:0.88rem;">
            <thead>
              <tr style="border-bottom:1px solid {COLORS['Border']}; color:{COLORS['TextMuted']}; text-align:left;">
                <th>Constraint</th><th style="text-align:right;">Value</th><th style="text-align:right;">Limit</th>
                <th style="text-align:right;">Headroom</th><th>Status</th>
              </tr>
            </thead>
            <tbody>{_checks_table(v1_02_result.checks)}</tbody>
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
    mo,
    v1_02_deployment,
    v1_02_result,
    v1_02_sweep,
    v1_02_workload,
):
    _colors = [COLORS["GreenLine"] if ok else COLORS["RedLine"] for ok in v1_02_sweep.feasible]
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=list(v1_02_sweep.knob_values),
        y=list(v1_02_sweep.worst_headroom_pct),
        mode="lines+markers",
        marker=dict(color=_colors, size=7),
        line=dict(color=COLORS["BlueLine"], width=2.5),
        name="Worst headroom",
    ))
    _fig.add_hline(y=0, line_dash="dash", line_color=COLORS["RedLine"], line_width=1.5)
    _fig.add_vline(
        x=v1_02_workload.value,
        line_dash="dot",
        line_color=COLORS["OrangeLine"],
        line_width=2,
        annotation_text="current setting",
        annotation_font_color=COLORS["OrangeLine"],
    )
    if v1_02_sweep.threshold_crossing is not None:
        _fig.add_vline(
            x=v1_02_sweep.threshold_crossing,
            line_dash="dash",
            line_color=COLORS["RedLine"],
            line_width=2,
            annotation_text=f"first wall: {v1_02_sweep.threshold_wall}",
            annotation_font_color=COLORS["RedLine"],
        )
    _fig.update_layout(
        height=340,
        xaxis=dict(title=f"{v1_02_deployment.workload_knob} ({v1_02_deployment.workload_unit})", gridcolor="#f1f5f9"),
        yaxis=dict(title="Worst headroom (%)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=50),
    )
    apply_plotly_theme(_fig)

    _crossing = (
        f"{v1_02_sweep.threshold_crossing:.1f} {v1_02_deployment.workload_unit}"
        if v1_02_sweep.threshold_crossing is not None
        else "not reached in sweep"
    )
    _sample_rows = []
    _stride = max(1, len(v1_02_sweep.knob_values) // 6)
    for idx in range(0, len(v1_02_sweep.knob_values), _stride):
        _sample_rows.append(
            f"<tr><td>{v1_02_sweep.knob_values[idx]:.1f}</td>"
            f"<td>{v1_02_sweep.first_walls[idx]}</td>"
            f"<td>{v1_02_sweep.worst_headroom_pct[idx]:.1f}%</td>"
            f"<td>{'yes' if v1_02_sweep.feasible[idx] else 'no'}</td></tr>"
        )

    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Physics Curve</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Where does {v1_02_deployment.workload_knob} cross from feasible to physically blocked?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>A sweep turns a qualitative wall into a threshold value with units.</li>
            <li>The zero-headroom line is the feasibility boundary.</li>
            <li>The current setting is {v1_02_result.workload_value:.1f} {v1_02_deployment.workload_unit}; the first sweep crossing is {_crossing}.</li>
          </ul>
        </div>
        """),
        v1_02_workload,
        mo.as_html(_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Table Fallback</h2>
          <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
            <thead>
              <tr style="border-bottom:1px solid {COLORS['Border']}; text-align:left; color:{COLORS['TextMuted']};">
                <th>{v1_02_deployment.workload_unit}</th><th>First wall</th><th>Worst headroom</th><th>Feasible</th>
              </tr>
            </thead>
            <tbody>{''.join(_sample_rows)}</tbody>
          </table>
        </div>
        """),
    ])
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    evaluate_deployment_envelope,
    mo,
    source_trace,
    v1_02_deployment,
    v1_02_mitigation,
    v1_02_placement,
    v1_02_reflection,
    v1_02_result,
    v1_02_workload,
):
    _placement_rows = []
    for option in v1_02_deployment.placement_options:
        _result = evaluate_deployment_envelope(
            v1_02_deployment,
            workload_value=v1_02_workload.value,
            placement_id=option.placement_id,
        )
        _color = COLORS["GreenLine"] if _result.feasible else COLORS["RedLine"]
        _placement_rows.append(
            f"""
            <tr>
              <td>{option.label}</td>
              <td>{_result.first_wall}</td>
              <td style="color:{_color}; font-weight:800;">{'yes' if _result.feasible else 'no'}</td>
              <td>{option.risk}</td>
            </tr>
            """
        )

    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Deployment Choice</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which placement changes the wall, and what new risk does it introduce?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Moving computation changes the physical constraint; it does not remove constraints.</li>
            <li>Local placement usually preserves latency and privacy but spends device power, memory, or energy.</li>
            <li>Offload can reduce local pressure while adding network latency, availability, privacy, or cost risk.</li>
          </ul>
        </div>
        """),
        v1_02_placement,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Selected placement</strong>{v1_02_mitigation.placement_label}</div>
            <div class="mlsysbook-field"><strong>Binding constraint</strong>{v1_02_mitigation.binding_constraint}</div>
            <div class="mlsysbook-field"><strong>Mitigation</strong>{v1_02_mitigation.mitigation}</div>
            <div class="mlsysbook-field"><strong>New risk</strong>{v1_02_mitigation.new_risk}</div>
          </div>
          <table style="width:100%; border-collapse:collapse; margin-top:14px; font-size:0.88rem;">
            <thead>
              <tr style="border-bottom:1px solid {COLORS['Border']}; text-align:left; color:{COLORS['TextMuted']};">
                <th>Placement</th><th>First wall</th><th>Feasible</th><th>Residual risk</th>
              </tr>
            </thead>
            <tbody>{''.join(_placement_rows)}</tbody>
          </table>
        </div>
        """),
        source_trace(
            {
                "workload_value": f"{v1_02_result.workload_value:.1f} {v1_02_deployment.workload_unit}",
                "placement_id": v1_02_placement.value,
                "helper": "evaluate_deployment_envelope + deployment_mitigation",
                "source_refs": ", ".join(v1_02_deployment.source_refs),
            },
            summary="Placement evidence is computed from the selected workload, MLSysIM refs, and V1-02 variant budgets.",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Reflection</h2></div>'),
        v1_02_reflection,
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
    v1_02_deployment,
    v1_02_first_wall_pred,
    v1_02_mitigation,
    v1_02_profile,
    v1_02_result,
    v1_02_sweep,
    v1_02_variant,
    v1_02_workload,
):
    if v1_02_first_wall_pred.value is not None:
        ledger.save(chapter=2, design={
            "chapter": "v1_02",
            "track_id": v1_02_profile.track_id,
            "scenario_id": v1_02_variant.scenario_id,
            "hardware_ref": v1_02_deployment.hardware_ref,
            "model_ref": v1_02_deployment.model_ref,
            "completed": True,
            "first_wall_prediction": v1_02_first_wall_pred.value,
            "actual_first_wall": v1_02_result.first_wall,
            "workload_value": v1_02_workload.value,
            "threshold_crossing": v1_02_sweep.threshold_crossing,
            "mitigation": v1_02_mitigation.mitigation,
        })

    _crossing = (
        f"{v1_02_sweep.threshold_crossing:.1f} {v1_02_deployment.workload_unit}"
        if v1_02_sweep.threshold_crossing is not None
        else "not reached in sweep"
    )
    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Synthesis</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Track</strong>{v1_02_deployment.label}</div>
            <div class="mlsysbook-field"><strong>First wall</strong>{v1_02_result.first_wall}</div>
            <div class="mlsysbook-field"><strong>Threshold crossing</strong>{_crossing}</div>
            <div class="mlsysbook-field"><strong>Selected mitigation</strong>{v1_02_mitigation.mitigation}</div>
          </div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Big Takeaways</h2>
          <ul class="mlsysbook-list">
            <li><strong>Deployment is physical.</strong> A feasible model must fit memory, flash, latency, energy, power, bandwidth, and cost at the same time.</li>
            <li><strong>Track choice changes the wall.</strong> {v1_02_deployment.label} turns the chapter idea into a specific hardware and stakeholder constraint.</li>
            <li><strong>Placement trades one wall for another.</strong> The report must name the avoided wall and the residual risk.</li>
          </ul>
        </div>
        """),
        mo.Html(f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">02 &middot; Physics of Deployment</span>
            <span class="hud-label">TRACK</span>
            <span class="hud-value">{v1_02_profile.label}</span>
            <span style="flex:1;"></span>
            <span class="hud-label">ARTIFACT</span>
            <span class="hud-value">{v1_02_deployment.report_artifact}</span>
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
    v1_02_deployment,
    v1_02_first_wall_pred,
    v1_02_metadata,
    v1_02_mitigation,
    v1_02_placement,
    v1_02_profile,
    v1_02_reflection,
    v1_02_result,
    v1_02_sweep,
    v1_02_variant,
    v1_02_workload,
):
    _incomplete = []
    if v1_02_first_wall_pred.value is None:
        _incomplete.append("Part A first-wall prediction")
    if not str(v1_02_reflection.value or "").strip():
        _incomplete.append("Part C reflection")

    _crossing = (
        f"{v1_02_sweep.threshold_crossing:.1f} {v1_02_deployment.workload_unit}"
        if v1_02_sweep.threshold_crossing is not None
        else "not reached in sweep"
    )
    _report = build_lab_report(
        v1_02_metadata,
        track=v1_02_profile.label,
        scenario=v1_02_variant.workload_summary,
        learning_objectives=(
            "Diagnose the first physical wall in a track-specific deployment envelope.",
            "Sweep a workload knob and identify the feasibility threshold.",
            "Choose a placement or mitigation and name the residual risk.",
        ),
        predictions={
            "first_wall": v1_02_first_wall_pred.value,
        },
        knob_settings={
            "workload_knob": v1_02_deployment.workload_knob,
            "workload_value": v1_02_workload.value,
            "workload_unit": v1_02_deployment.workload_unit,
            "placement_id": v1_02_placement.value,
        },
        evidence_summary={
            "hardware_ref": v1_02_deployment.hardware_ref,
            "model_ref": v1_02_deployment.model_ref,
            "actual_first_wall": v1_02_result.first_wall,
            "feasible": v1_02_result.feasible,
            "violations": v1_02_result.violations,
            "threshold_crossing": _crossing,
            "mitigation": v1_02_mitigation.mitigation,
        },
        final_decision=(
            f"For {v1_02_deployment.label}, use {v1_02_mitigation.placement_label} and "
            f"{v1_02_mitigation.mitigation}"
        ),
        big_takeaways=(
            "Deployment feasibility is a vector of physical constraints with units.",
            "The selected track determines which wall appears first.",
            "A mitigation should name the wall it avoids and the new risk it creates.",
        ),
        reflections={
            "student_reflection": v1_02_reflection.value,
            "placement_risk": v1_02_mitigation.new_risk,
            "report_artifact": v1_02_deployment.report_artifact,
        },
        residual_risk=v1_02_mitigation.new_risk,
        source_trace={
            "track_id": v1_02_profile.track_id,
            "scenario_id": v1_02_variant.scenario_id,
            "hardware_ref": v1_02_variant.hardware_ref,
            "model_ref": v1_02_variant.model_ref,
            "shared_helper": "mlsysbook_labs.deployment",
            "source_policy": v1_02_profile.source_policy,
        },
        result_snapshot={
            "deployment_profile": v1_02_deployment,
            "envelope_result": v1_02_result,
            "sweep": v1_02_sweep,
            "mitigation": v1_02_mitigation,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-02 memo is generated locally from the selected track, MLSysIM refs, "
                "and shared `mlsysbook_labs.deployment` calculations."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
