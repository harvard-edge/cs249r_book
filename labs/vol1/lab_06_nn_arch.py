import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ===========================================================================
# ZONE A: SETUP
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
        architecture_decision,
        architecture_scaling_curve,
        architecture_signature,
        architecture_track_profile,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
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
        architecture_decision,
        architecture_scaling_curve,
        architecture_signature,
        architecture_track_profile,
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_06_metadata = get_lab_metadata("vol1/lab_06_nn_arch.py")
    return (v1_06_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_06_track_picker = track_selector(default=_default_track)
    v1_06_track_picker
    return (v1_06_track_picker,)


@app.cell
def _(
    architecture_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_06_track_picker,
):
    v1_06_track_id = v1_06_track_picker.value
    v1_06_profile = get_track_profile(v1_06_track_id)
    v1_06_variant = get_lab_track_variant("v1_06_architecture_tax", v1_06_profile.track_id)
    v1_06_hardware = resolve_mlsysim_ref(v1_06_variant.hardware_ref)
    v1_06_model = resolve_mlsysim_ref(v1_06_variant.model_ref)
    v1_06_architecture = architecture_track_profile(
        v1_06_profile,
        v1_06_variant,
        v1_06_hardware,
        v1_06_model,
    )
    return (
        v1_06_architecture,
        v1_06_hardware,
        v1_06_model,
        v1_06_profile,
        v1_06_track_id,
        v1_06_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
    v1_06_architecture,
    v1_06_metadata,
    v1_06_profile,
    v1_06_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 06
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Architecture Tax
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Inductive Bias &middot; Scaling Shape &middot; Kernel Support &middot; Guardrails
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 860px; line-height: 1.65;">
                {v1_06_variant.workload_summary} The goal is not to crown one universal
                architecture; it is to choose the family whose resource signature matches
                the selected track.
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
                    {v1_06_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_06_architecture.scaling_variable}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Architecture Signature</span>
                <span class="badge badge-warn">Scaling Shape</span>
                <span class="badge badge-fail">Decision Memo</span>
            </div>
        </div>
        """),
        track_context(v1_06_profile),
        source_trace(
            {
                "lab_id": v1_06_metadata.lab_id,
                "track_id": v1_06_profile.track_id,
                "hardware_ref": v1_06_variant.hardware_ref,
                "model_ref": v1_06_variant.model_ref,
                "shared_helper": "mlsysbook_labs.architecture",
                "source_policy": v1_06_profile.source_policy,
            },
            summary="V1-06 evaluates architecture signatures, scaling curves, and recommendations through mlsysbook_labs.architecture.",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_06_architecture):
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
            <div style="margin-bottom: 3px;">1. <strong>Compare architecture signatures:</strong>
                parameters, operations, activation memory, latency, power, quality, and kernel support.</div>
            <div style="margin-bottom: 3px;">2. <strong>Predict scaling failure:</strong>
                sweep {v1_06_architecture.scaling_variable} and identify the first architecture family that breaks.</div>
            <div style="margin-bottom: 3px;">3. <strong>Defend an architecture:</strong>
                recommend one family, reject alternatives, and state the validation requirement.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                Which architecture family fits {v1_06_architecture.label}, and what failure
                appears next as {v1_06_architecture.scaling_variable} grows?
            </div>
        </div>
    </div>
    """)
    return


# ===========================================================================
# ZONE B: CONTROLS AND COMPUTATION
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_06_architecture):
    v1_06_failure_prediction = mo.ui.radio(
        options={
            "CNN-style locality fails first": "cnn",
            "Transformer-style token mixing fails first": "transformer",
            "Kernel support or dispatch fails first": "kernel",
            "Quality guardrail fails before resource budgets": "quality",
        },
        label=f"{v1_06_architecture.label}: which architecture risk do you expect to bind first?",
    )
    v1_06_failure_prediction
    return (v1_06_failure_prediction,)


@app.cell(hide_code=True)
def _(mo, v1_06_architecture):
    v1_06_scale = mo.ui.slider(
        start=v1_06_architecture.scale_min,
        stop=v1_06_architecture.scale_max,
        value=v1_06_architecture.default_scale,
        step=v1_06_architecture.scale_step,
        label=f"{v1_06_architecture.scaling_variable} ({v1_06_architecture.scaling_unit})",
    )
    v1_06_scale
    return (v1_06_scale,)


@app.cell(hide_code=True)
def _(mo, v1_06_architecture):
    _architecture_options = {
        candidate.label: candidate.architecture_id
        for candidate in v1_06_architecture.candidates
    }
    v1_06_arch_choice = mo.ui.dropdown(
        options=_architecture_options,
        value=v1_06_architecture.candidates[0].label,
        label="Architecture recommendation",
    )
    v1_06_reflection = mo.ui.text_area(
        label="Reflection",
        placeholder="Name the architecture you recommend, the alternatives you reject, and the validation risk that remains.",
        full_width=True,
    )
    return (v1_06_arch_choice, v1_06_reflection)


@app.cell
def _(
    architecture_decision,
    architecture_scaling_curve,
    architecture_signature,
    v1_06_arch_choice,
    v1_06_architecture,
    v1_06_scale,
):
    v1_06_signature = architecture_signature(
        v1_06_architecture,
        scale_value=v1_06_scale.value,
    )
    v1_06_curve = architecture_scaling_curve(v1_06_architecture, samples=36)
    v1_06_decision = architecture_decision(
        v1_06_architecture,
        architecture_id=v1_06_arch_choice.value,
        scale_value=v1_06_scale.value,
    )
    v1_06_selected_eval = next(
        item for item in v1_06_signature
        if item.architecture_id == v1_06_decision.selected_id
    )
    return (
        v1_06_curve,
        v1_06_decision,
        v1_06_selected_eval,
        v1_06_signature,
    )


# ===========================================================================
# ZONE C: PARTS
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    source_trace,
    v1_06_arch_choice,
    v1_06_architecture,
    v1_06_curve,
    v1_06_decision,
    v1_06_failure_prediction,
    v1_06_reflection,
    v1_06_scale,
    v1_06_selected_eval,
    v1_06_signature,
):
    _labels = [item.label for item in v1_06_signature]
    _bar_fig = go.Figure()
    _bar_fig.add_trace(go.Bar(
        x=_labels,
        y=[item.activation_mb for item in v1_06_signature],
        name="Activation MB",
        marker_color=COLORS["BlueLine"],
        text=[f"{item.activation_mb:.1f}" for item in v1_06_signature],
        textposition="outside",
    ))
    _bar_fig.add_trace(go.Bar(
        x=_labels,
        y=[item.latency_ms for item in v1_06_signature],
        name="Latency ms",
        marker_color=COLORS["OrangeLine"],
        text=[f"{item.latency_ms:.1f}" for item in v1_06_signature],
        textposition="outside",
    ))
    _bar_fig.update_layout(
        barmode="group",
        height=360,
        xaxis=dict(title="Candidate architecture", gridcolor="#f1f5f9"),
        yaxis=dict(title="Value", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=80),
    )
    apply_plotly_theme(_bar_fig)

    _rows = "".join(
        f"""
        <tr>
          <td>{item.label}</td>
          <td>{item.family}</td>
          <td>{item.params_m:.2f}M</td>
          <td>{item.ops_gmac:.2f}</td>
          <td>{item.activation_mb:.2f} MB</td>
          <td>{item.latency_ms:.2f} ms</td>
          <td>{item.quality_pct:.1f}%</td>
          <td>{item.kernel_support_pct:.1f}%</td>
          <td>{'yes' if item.feasible else 'no - violation'}</td>
          <td>{item.dominant_constraint}</td>
        </tr>
        """
        for item in v1_06_signature
    )

    _scale_fig = go.Figure()
    _palette = [COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"], COLORS["RedLine"]]
    for idx, candidate in enumerate(v1_06_architecture.candidates):
        _points = v1_06_curve.points_by_candidate[candidate.architecture_id]
        _scale_fig.add_trace(go.Scatter(
            x=[point.scale_value for point in _points],
            y=[point.latency_ms for point in _points],
            mode="lines+markers",
            name=candidate.label,
            line=dict(color=_palette[idx % len(_palette)], width=2.5),
            marker=dict(size=6),
        ))
    _scale_fig.add_hline(
        y=v1_06_architecture.latency_budget_ms,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        line_width=1.5,
        annotation_text="latency budget",
        annotation_font_color=COLORS["RedLine"],
    )
    _scale_fig.update_layout(
        height=360,
        xaxis=dict(
            title=f"{v1_06_architecture.scaling_variable} ({v1_06_architecture.scaling_unit})",
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(title="Latency (ms)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=50),
    )
    apply_plotly_theme(_scale_fig)

    _failure_rows = "".join(
        f"""
        <tr>
          <td>{candidate.label}</td>
          <td>{candidate.scaling_law}</td>
          <td>{v1_06_curve.first_failure_by_candidate[candidate.architecture_id] or 'not reached'}</td>
        </tr>
        """
        for candidate in v1_06_architecture.candidates
    )
    _validation_items = "".join(f"<li>{test}</li>" for test in v1_06_architecture.validation_tests)
    _rejections = "".join(f"<li>{item}</li>" for item in v1_06_decision.rejected_alternatives)

    _part_a = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Architecture Signature</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which candidate has the resource signature that matches {v1_06_architecture.label}?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Architecture families encode assumptions: locality, sequence mixing, recurrence, sparsity, or expert routing.</li>
            <li>The same quality number can hide different parameters, activations, latency, power, and kernel support.</li>
            <li>Feasibility is checked against track-specific memory, latency, power, quality, and supported-kernel budgets.</li>
          </ul>
          <div class="mlsysbook-callout"><strong>Track architecture story:</strong> {v1_06_architecture.architecture_story}</div>
        </div>
        """),
        v1_06_failure_prediction,
        v1_06_scale,
        mo.as_html(_bar_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Table Fallback</h2>
          <table class="mlsysbook-table">
            <thead>
              <tr>
                <th>Architecture</th><th>Family</th><th>Params</th><th>GMAC</th>
                <th>Activations</th><th>Latency</th><th>Quality</th><th>Kernels</th>
                <th>Feasible</th><th>Dominant constraint</th>
              </tr>
            </thead>
            <tbody>{_rows}</tbody>
          </table>
        </div>
        """),
    ])

    _part_b = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Scaling Shape</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            What happens as {v1_06_architecture.scaling_variable} grows?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Scaling laws differ: convolution often tracks resolution, attention tracks tokens, and routing adds tail risk.</li>
            <li>The first failure point should be named with a value, limit, unit, and mitigation.</li>
            <li>A candidate that is feasible at the default scale can still be a poor choice if the next scaling step is fragile.</li>
          </ul>
        </div>
        """),
        mo.as_html(_scale_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Failure Boundary</h2>
          <table class="mlsysbook-table">
            <thead><tr><th>Architecture</th><th>Scaling shape</th><th>First infeasible scale</th></tr></thead>
            <tbody>{_failure_rows}</tbody>
          </table>
        </div>
        """),
    ])

    _part_c = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Architecture Choice</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which architecture would you defend in an engineering review?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>The recommendation should not be only the highest-quality architecture.</li>
            <li>Reject alternatives explicitly: one may fail memory, another kernels, another quality.</li>
            <li>The validation requirement is part of the choice because architecture assumptions can fail in production.</li>
          </ul>
        </div>
        """),
        v1_06_arch_choice,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Selected architecture</strong>{v1_06_decision.selected_label}</div>
            <div class="mlsysbook-field"><strong>Feasible</strong>{'yes' if v1_06_decision.feasible else 'no - violation'}</div>
            <div class="mlsysbook-field"><strong>Dominant constraint</strong>{v1_06_decision.dominant_constraint}</div>
            <div class="mlsysbook-field"><strong>Next failure</strong>{v1_06_decision.next_failure}</div>
            <div class="mlsysbook-field"><strong>Quality</strong>{v1_06_selected_eval.quality_pct:.1f}% / floor {v1_06_architecture.quality_floor_pct:.1f}%</div>
            <div class="mlsysbook-field"><strong>Kernel support</strong>{v1_06_selected_eval.kernel_support_pct:.1f}% / floor {v1_06_architecture.kernel_support_floor_pct:.1f}%</div>
          </div>
          <div class="mlsysbook-callout"><strong>Memo decision:</strong> {v1_06_decision.memo_summary}</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Rejected Alternatives</h2>
          <ul class="mlsysbook-list">{_rejections}</ul>
          <h2>Validation Tests</h2>
          <ul class="mlsysbook-list">{_validation_items}</ul>
        </div>
        """),
        source_trace(
            {
                "helper": "architecture_decision",
                "selected_id": v1_06_decision.selected_id,
                "hardware_ref": v1_06_architecture.hardware_ref,
                "model_ref": v1_06_architecture.model_ref,
            },
            summary="Architecture decision evidence is computed from the selected track budgets and candidate registry.",
        ),
        mo.Html('<div class="mlsysbook-panel"><h2>Reflection</h2></div>'),
        v1_06_reflection,
    ])

    mo.ui.tabs({
        "Part A · Signature": _part_a,
        "Part B · Scaling": _part_b,
        "Part C · Choice": _part_c,
    })
    return


# ===========================================================================
# ZONE D: SYNTHESIS AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    v1_06_architecture,
    v1_06_decision,
    v1_06_failure_prediction,
    v1_06_profile,
    v1_06_selected_eval,
    v1_06_variant,
):
    if v1_06_failure_prediction.value is not None:
        ledger.save(chapter=6, design={
            "chapter": "v1_06",
            "track_id": v1_06_profile.track_id,
            "scenario_id": v1_06_variant.scenario_id,
            "hardware_ref": v1_06_architecture.hardware_ref,
            "model_ref": v1_06_architecture.model_ref,
            "completed": True,
            "failure_prediction": v1_06_failure_prediction.value,
            "selected_architecture": v1_06_decision.selected_id,
            "dominant_constraint": v1_06_decision.dominant_constraint,
            "quality_pct": v1_06_selected_eval.quality_pct,
            "kernel_support_pct": v1_06_selected_eval.kernel_support_pct,
        })

    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Synthesis</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Track</strong>{v1_06_architecture.label}</div>
            <div class="mlsysbook-field"><strong>Selected architecture</strong>{v1_06_decision.selected_label}</div>
            <div class="mlsysbook-field"><strong>Dominant constraint</strong>{v1_06_decision.dominant_constraint}</div>
            <div class="mlsysbook-field"><strong>Residual risk</strong>{v1_06_decision.residual_risk}</div>
          </div>
        </div>
        """),
        mo.Html("""
        <div class="mlsysbook-panel">
          <h2>Big Takeaways</h2>
          <ul class="mlsysbook-list">
            <li><strong>Architectures are systems choices.</strong> Inductive bias changes operations, activations, kernels, and validation risk.</li>
            <li><strong>Scaling shape matters.</strong> A family that fits at one scale may hit a nonlinear wall at the next product requirement.</li>
            <li><strong>The track determines the defensible answer.</strong> The same architecture can be right for Cloud Fleet and wrong for Oura Ring.</li>
          </ul>
        </div>
        """),
        mo.Html(f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">06 &middot; Architecture Tax</span>
            <span class="hud-label">TRACK</span>
            <span class="hud-value">{v1_06_profile.label}</span>
            <span style="flex:1;"></span>
            <span class="hud-label">ARTIFACT</span>
            <span class="hud-value">{v1_06_architecture.report_artifact}</span>
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
    v1_06_architecture,
    v1_06_curve,
    v1_06_decision,
    v1_06_failure_prediction,
    v1_06_metadata,
    v1_06_profile,
    v1_06_reflection,
    v1_06_scale,
    v1_06_signature,
    v1_06_variant,
):
    _incomplete = []
    if v1_06_failure_prediction.value is None:
        _incomplete.append("Part A architecture-risk prediction")
    if not str(v1_06_reflection.value or "").strip():
        _incomplete.append("Part C reflection")

    _report = build_lab_report(
        v1_06_metadata,
        track=v1_06_profile.label,
        scenario=v1_06_variant.workload_summary,
        learning_objectives=(
            "Compare architecture signatures across parameters, operations, activations, latency, power, quality, and kernel support.",
            "Sweep the track-specific scaling variable and identify the first failure boundary.",
            "Recommend one architecture family and state rejected alternatives plus validation requirements.",
        ),
        predictions={
            "first_architecture_risk": v1_06_failure_prediction.value,
        },
        knob_settings={
            "scale_value": v1_06_scale.value,
            "scaling_variable": v1_06_architecture.scaling_variable,
            "selected_architecture": v1_06_decision.selected_id,
        },
        evidence_summary={
            "hardware_ref": v1_06_architecture.hardware_ref,
            "model_ref": v1_06_architecture.model_ref,
            "memory_budget_mb": v1_06_architecture.memory_budget_mb,
            "latency_budget_ms": v1_06_architecture.latency_budget_ms,
            "power_budget_w": v1_06_architecture.power_budget_w,
            "selected_architecture": v1_06_decision.selected_label,
            "dominant_constraint": v1_06_decision.dominant_constraint,
            "next_failure": v1_06_decision.next_failure,
        },
        final_decision=v1_06_decision.memo_summary,
        big_takeaways=(
            "Architecture families carry different resource signatures, not just different accuracy numbers.",
            "Scaling shape must be tested against the selected track's memory, latency, power, quality, and kernel budgets.",
            "A defensible architecture memo names rejected alternatives and the validation risk that remains.",
        ),
        reflections={
            "student_reflection": v1_06_reflection.value,
            "rejected_alternatives": v1_06_decision.rejected_alternatives,
            "validation_requirement": v1_06_decision.validation_requirement,
            "report_artifact": v1_06_architecture.report_artifact,
        },
        residual_risk=v1_06_decision.residual_risk,
        source_trace={
            "track_id": v1_06_profile.track_id,
            "scenario_id": v1_06_variant.scenario_id,
            "hardware_ref": v1_06_variant.hardware_ref,
            "model_ref": v1_06_variant.model_ref,
            "shared_helper": "mlsysbook_labs.architecture",
            "source_policy": v1_06_profile.source_policy,
        },
        result_snapshot={
            "architecture_profile": v1_06_architecture,
            "signature": v1_06_signature,
            "scaling_curve": v1_06_curve,
            "decision": v1_06_decision,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-06 architecture recommendation memo is generated locally from "
                "the selected track, MLSysIM refs, and shared `mlsysbook_labs.architecture` calculations."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
