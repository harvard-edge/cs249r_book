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
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        memory_cliff,
        neural_compute_profile,
        operation_ledger,
        operator_design,
        part_workflow,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
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
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        memory_cliff,
        mo,
        neural_compute_profile,
        operation_ledger,
        operator_design,
        part_workflow,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_05_metadata = get_lab_metadata("vol1/lab_05_nn_compute.py")
    return (v1_05_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_05_track_picker = track_selector(default=_default_track)
    v1_05_track_picker
    return (v1_05_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    neural_compute_profile,
    resolve_mlsysim_ref,
    v1_05_track_picker,
):
    v1_05_track_id = v1_05_track_picker.value
    v1_05_profile = get_track_profile(v1_05_track_id)
    v1_05_variant = get_lab_track_variant("v1_05_neural_computation", v1_05_profile.track_id)
    v1_05_hardware = resolve_mlsysim_ref(v1_05_variant.hardware_ref)
    v1_05_model = resolve_mlsysim_ref(v1_05_variant.model_ref)
    v1_05_compute = neural_compute_profile(
        v1_05_profile,
        v1_05_variant,
        v1_05_hardware,
        v1_05_model,
    )
    return (
        v1_05_compute,
        v1_05_hardware,
        v1_05_model,
        v1_05_profile,
        v1_05_track_id,
        v1_05_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
        track_arc_context,
    v1_05_compute,
    v1_05_metadata,
    v1_05_profile,
    v1_05_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 05
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Activation Tax
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Weights &middot; Activations &middot; Operations &middot; Bytes Moved
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 820px; line-height: 1.65;">
                {v1_05_variant.workload_summary} This lab shows why activations
                and bytes moved can dominate even when parameter counts look small.
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
                    {v1_05_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_05_compute.tensor_label}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Operation Ledger</span>
                <span class="badge badge-warn">Memory Cliff</span>
                <span class="badge badge-fail">Operator Design</span>
            </div>
        </div>
        """),
        track_context(v1_05_profile),
        track_arc_context(v1_05_profile, v1_05_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, part_workflow, v1_05_compute):
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
            <div style="margin-bottom: 3px;">1. <strong>Build an operation ledger:</strong>
                compare weights, activations, operations, bytes moved, and arithmetic intensity.</div>
            <div style="margin-bottom: 3px;">2. <strong>Find the memory cliff:</strong>
                sweep the shape multiplier until activation memory crosses the track budget.</div>
            <div style="margin-bottom: 3px;">3. <strong>Choose a layer design:</strong>
                select precision, tiling, fusion, or streaming and name the residual quality risk.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                Which operator resource dominates for {v1_05_compute.label}, and which
                design fits without hiding a new quality risk?
            </div>
        </div>
    </div>
    """),
    part_workflow(
        "Neural Compute Workflow",
        (
            {
                "part": "Part A",
                "concept": "Operation Ledger",
                "prediction": "Predict whether weights, activations, operations, or bytes moved dominate.",
                "controls": "Adjust the shape multiplier for the active operator.",
                "evidence": "Compare activation memory, bytes moved, arithmetic intensity, latency, and feasibility.",
                "decision": "Decide which resource must be reduced first for the selected track.",
            },
            {
                "part": "Part B",
                "concept": "Memory Cliff",
                "prediction": "Predict where the activation budget will fail.",
                "controls": "Sweep the shape variable and watch the feasible/infeasible boundary.",
                "evidence": "Inspect the activation curve and threshold multiplier.",
                "decision": "Choose the largest shape you can justify before the cliff.",
            },
            {
                "part": "Part C",
                "concept": "Layer Design",
                "prediction": "Predict which precision, tiling, fusion, or streaming design fits.",
                "controls": "Select the operator design option.",
                "evidence": "Compare activation memory, latency, bandwidth, feasibility, and quality risk.",
                "decision": "Write the design memo with the residual validation risk.",
            },
        ),
        scenario=(
            f"{v1_05_compute.label} is constrained by {v1_05_compute.tensor_label}; "
            "the lab asks which tensor budget actually controls deployment."
        ),
        reflection="Carry one dominant resource, one feasible design, and one quality risk into the report.",
    ),
    ])
    return


# ===========================================================================
# ZONE B: CONTROLS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_05_compute):
    v1_05_resource_prediction = mo.ui.radio(
        options={
            "Weights dominate because parameters are persistent": "weights",
            "Activations dominate because intermediate tensors are large": "activations",
            "Operations dominate because FLOPs set latency": "operations",
            "Bytes moved dominate because memory traffic sets the wall": "bytes",
        },
        label=f"{v1_05_compute.label}: which resource do you expect to dominate?",
    )
    v1_05_resource_prediction
    return (v1_05_resource_prediction,)


@app.cell(hide_code=True)
def _(mo, v1_05_compute):
    v1_05_shape_multiplier = mo.ui.slider(
        start=v1_05_compute.shape_min,
        stop=v1_05_compute.shape_max,
        value=v1_05_compute.default_shape_multiplier,
        step=v1_05_compute.shape_step,
        label="Shape multiplier",
    )
    v1_05_shape_multiplier
    return (v1_05_shape_multiplier,)


@app.cell(hide_code=True)
def _(mo, v1_05_compute):
    _design_options = {design.label: design.design_id for design in v1_05_compute.design_options}
    v1_05_design = mo.ui.dropdown(
        options=_design_options,
        value=v1_05_compute.design_options[0].label,
        label="Operator design",
    )
    v1_05_reflection = mo.ui.text_area(
        label="Reflection",
        placeholder="Name the resource you reduced and the quality or validation risk you accepted.",
        full_width=True,
    )
    return (v1_05_design, v1_05_reflection)


@app.cell
def _(
    memory_cliff,
    operation_ledger,
    operator_design,
    v1_05_compute,
    v1_05_design,
    v1_05_shape_multiplier,
):
    v1_05_ledger = operation_ledger(
        v1_05_compute,
        shape_multiplier=v1_05_shape_multiplier.value,
    )
    v1_05_cliff = memory_cliff(v1_05_compute, samples=40)
    v1_05_design_result = operator_design(
        v1_05_compute,
        design_id=v1_05_design.value,
        shape_multiplier=v1_05_shape_multiplier.value,
    )
    return (v1_05_cliff, v1_05_design_result, v1_05_ledger)


# ===========================================================================
# ZONE C: PARTS
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    v1_05_compute,
    v1_05_ledger,
    v1_05_resource_prediction,
    v1_05_shape_multiplier,
):
    _components = {
        "weights MB": v1_05_ledger.weights_mb,
        "activations MB": v1_05_ledger.activations_mb,
        "ops GMAC": v1_05_ledger.ops_gmac,
        "bytes moved MB": v1_05_ledger.bytes_moved_mb,
    }
    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        x=list(_components.keys()),
        y=list(_components.values()),
        marker_color=[COLORS["BlueLine"], COLORS["RedLine"], COLORS["GreenLine"], COLORS["OrangeLine"]],
        text=[f"{value:.2f}" for value in _components.values()],
        textposition="outside",
    ))
    _fig.update_layout(
        height=320,
        xaxis=dict(title="Ledger item", gridcolor="#f1f5f9"),
        yaxis=dict(title="Value (mixed units)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=60),
    )
    apply_plotly_theme(_fig)
    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Operation Ledger</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which tensor or resource dominates {v1_05_compute.tensor_label}?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Parameter count is not enough: activations and bytes moved can dominate deployment memory and latency.</li>
            <li>Arithmetic intensity connects operations to bytes moved; low intensity is memory-bound.</li>
            <li>Feasibility is checked against track-specific activation, bandwidth, latency, and power budgets.</li>
          </ul>
          <div class="mlsysbook-callout"><strong>Track operator story:</strong> {v1_05_compute.operator_story}</div>
        </div>
        """),
        v1_05_resource_prediction,
        v1_05_shape_multiplier,
        mo.as_html(_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Dominant resource</strong>{v1_05_ledger.dominant_resource}</div>
            <div class="mlsysbook-field"><strong>Activation memory</strong>{v1_05_ledger.activations_mb:.2f} MB / {v1_05_compute.activation_budget_mb:.2f} MB</div>
            <div class="mlsysbook-field"><strong>Bytes moved</strong>{v1_05_ledger.bytes_moved_mb:.2f} MB</div>
            <div class="mlsysbook-field"><strong>Arithmetic intensity</strong>{v1_05_ledger.arithmetic_intensity:.2f} ops/byte</div>
            <div class="mlsysbook-field"><strong>Latency estimate</strong>{v1_05_ledger.estimated_latency_ms:.3f} ms / {v1_05_compute.latency_budget_ms:.1f} ms</div>
            <div class="mlsysbook-field"><strong>Feasible</strong>{'yes' if v1_05_ledger.feasible else 'no'}</div>
          </div>
        </div>
        """),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, apply_plotly_theme, go, mo, v1_05_cliff, v1_05_compute):
    _colors = [COLORS["GreenLine"] if ok else COLORS["RedLine"] for ok in v1_05_cliff.feasible]
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=list(v1_05_cliff.shape_values),
        y=list(v1_05_cliff.activation_mb),
        mode="lines+markers",
        marker=dict(color=_colors, size=7),
        line=dict(color=COLORS["BlueLine"], width=2.5),
        name="Activation memory",
    ))
    _fig.add_hline(y=v1_05_compute.activation_budget_mb, line_dash="dash", line_color=COLORS["RedLine"], line_width=1.5)
    if v1_05_cliff.threshold_multiplier is not None:
        _fig.add_vline(
            x=v1_05_cliff.threshold_multiplier,
            line_dash="dash",
            line_color=COLORS["RedLine"],
            line_width=1.5,
            annotation_text="activation cliff",
            annotation_font_color=COLORS["RedLine"],
        )
    _fig.update_layout(
        height=340,
        xaxis=dict(title="Shape multiplier", gridcolor="#f1f5f9"),
        yaxis=dict(title="Activation memory (MB)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=50),
    )
    apply_plotly_theme(_fig)
    _threshold = (
        f"{v1_05_cliff.threshold_multiplier:.2f}x"
        if v1_05_cliff.threshold_multiplier is not None
        else "not reached"
    )
    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Memory Cliff</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Where does activation memory cross the {v1_05_compute.label} budget?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Activation memory often scales with area, sequence length, or batch, not just parameter count.</li>
            <li>The cliff is the first shape where activations exceed the budget.</li>
            <li>The threshold for this track is {_threshold}.</li>
          </ul>
        </div>
        """),
        mo.as_html(_fig),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, source_trace, v1_05_compute, v1_05_design, v1_05_design_result, v1_05_reflection):
    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Layer Design</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which operator design fits the track, and what quality risk does it accept?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Precision, tiling, streaming, fusion, and batching reduce different resources.</li>
            <li>A design that fits memory can still create quality or validation risk.</li>
            <li>The final memo should defend the resource reduction and the accepted sacrifice.</li>
          </ul>
        </div>
        """),
        v1_05_design,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Design</strong>{v1_05_design_result.design_label}</div>
            <div class="mlsysbook-field"><strong>Activation memory</strong>{v1_05_design_result.activation_mb:.2f} MB</div>
            <div class="mlsysbook-field"><strong>Latency</strong>{v1_05_design_result.latency_ms:.3f} ms</div>
            <div class="mlsysbook-field"><strong>Bandwidth</strong>{v1_05_design_result.bandwidth_gbs:.2f} GB/s</div>
            <div class="mlsysbook-field"><strong>Feasible</strong>{'yes' if v1_05_design_result.feasible else 'no'}</div>
            <div class="mlsysbook-field"><strong>Quality risk</strong>{v1_05_design_result.quality_risk}</div>
          </div>
          <div class="mlsysbook-callout"><strong>Memo decision:</strong> {v1_05_design_result.memo_summary}</div>
        </div>
        """),
        mo.Html('<div class="mlsysbook-panel"><h2>Reflection</h2></div>'),
        v1_05_reflection,
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
    v1_05_compute,
    v1_05_design_result,
    v1_05_ledger,
    v1_05_profile,
    v1_05_resource_prediction,
    v1_05_variant,
):
    if v1_05_resource_prediction.value is not None:
        ledger.save(chapter=5, design={
            "chapter": "v1_05",
            "track_id": v1_05_profile.track_id,
            "scenario_id": v1_05_variant.scenario_id,
            "hardware_ref": v1_05_compute.hardware_ref,
            "model_ref": v1_05_compute.model_ref,
            "completed": True,
            "resource_prediction": v1_05_resource_prediction.value,
            "dominant_resource": v1_05_ledger.dominant_resource,
            "operator_design": v1_05_design_result.design_id,
            "quality_risk": v1_05_design_result.quality_risk,
        })

    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Synthesis</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Track</strong>{v1_05_compute.label}</div>
            <div class="mlsysbook-field"><strong>Dominant resource</strong>{v1_05_ledger.dominant_resource}</div>
            <div class="mlsysbook-field"><strong>Selected design</strong>{v1_05_design_result.design_label}</div>
            <div class="mlsysbook-field"><strong>Residual risk</strong>{v1_05_design_result.residual_risk}</div>
          </div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Big Takeaways</h2>
          <ul class="mlsysbook-list">
            <li><strong>Activations are a first-class budget.</strong> Temporary tensors can dominate SRAM, DRAM, HBM, and bandwidth.</li>
            <li><strong>Shape changes are nonlinear.</strong> Resolution, sequence length, and batch can cross cliffs quickly.</li>
            <li><strong>Operator design is a trade.</strong> Precision, tiling, streaming, and fusion reduce resources while creating validation risk.</li>
          </ul>
        </div>
        """),
        mo.Html(f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">05 &middot; Activation Tax</span>
            <span class="hud-label">TRACK</span>
            <span class="hud-value">{v1_05_profile.label}</span>
            <span style="flex:1;"></span>
            <span class="hud-label">ARTIFACT</span>
            <span class="hud-value">{v1_05_compute.report_artifact}</span>
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
    v1_05_cliff,
    v1_05_compute,
    v1_05_design_result,
    v1_05_ledger,
    v1_05_metadata,
    v1_05_profile,
    v1_05_reflection,
    v1_05_resource_prediction,
    v1_05_shape_multiplier,
    v1_05_variant,
):
    _incomplete = []
    if v1_05_resource_prediction.value is None:
        _incomplete.append("Part A dominant-resource prediction")
    if not str(v1_05_reflection.value or "").strip():
        _incomplete.append("Part C reflection")

    _report = build_lab_report(
        v1_05_metadata,
        track=v1_05_profile.label,
        scenario=v1_05_variant.workload_summary,
        learning_objectives=(
            "Build an operation ledger for weights, activations, operations, bytes moved, and intensity.",
            "Sweep a shape variable until activation memory crosses the track budget.",
            "Choose an operator design and name the accepted quality or validation risk.",
        ),
        predictions={
            "dominant_resource": v1_05_resource_prediction.value,
        },
        knob_settings={
            "shape_multiplier": v1_05_shape_multiplier.value,
            "operator_design": v1_05_design_result.design_id,
        },
        evidence_summary={
            "hardware_ref": v1_05_compute.hardware_ref,
            "model_ref": v1_05_compute.model_ref,
            "dominant_resource": v1_05_ledger.dominant_resource,
            "activation_memory_mb": v1_05_ledger.activations_mb,
            "bytes_moved_mb": v1_05_ledger.bytes_moved_mb,
            "arithmetic_intensity": v1_05_ledger.arithmetic_intensity,
            "memory_cliff_multiplier": v1_05_cliff.threshold_multiplier,
            "selected_design": v1_05_design_result.design_label,
            "quality_risk": v1_05_design_result.quality_risk,
        },
        final_decision=v1_05_design_result.memo_summary,
        big_takeaways=(
            "Activations and bytes moved can dominate parameter memory.",
            "The selected track determines which memory or bandwidth cliff matters.",
            "A fitting operator design must still explain quality and validation risk.",
        ),
        reflections={
            "student_reflection": v1_05_reflection.value,
            "residual_risk": v1_05_design_result.residual_risk,
            "report_artifact": v1_05_compute.report_artifact,
        },
        residual_risk=v1_05_design_result.residual_risk,
        source_trace={
            "track_id": v1_05_profile.track_id,
            "scenario_id": v1_05_variant.scenario_id,
            "hardware_ref": v1_05_variant.hardware_ref,
            "model_ref": v1_05_variant.model_ref,
            "shared_helper": "mlsysbook_labs.neural_compute",
            "source_policy": v1_05_profile.source_policy,
        },
        result_snapshot={
            "compute_profile": v1_05_compute,
            "operation_ledger": v1_05_ledger,
            "memory_cliff": v1_05_cliff,
            "operator_design": v1_05_design_result,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-05 operator budget note is generated locally from the selected track, "
                "your inputs, and the computed evidence."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
