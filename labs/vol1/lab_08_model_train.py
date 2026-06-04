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
        build_lab_report,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        training_frontier,
        training_memory_stack,
        training_plan,
        training_track_profile,
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
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        source_trace,
        track_context,
        track_arc_context,
        track_selector,
        training_frontier,
        training_memory_stack,
        training_plan,
        training_track_profile,
    )


@app.cell
def _(get_lab_metadata):
    v1_08_metadata = get_lab_metadata("vol1/lab_08_model_train.py")
    return (v1_08_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_08_track_picker = track_selector(default=_default_track)
    v1_08_track_picker
    return (v1_08_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    training_track_profile,
    v1_08_track_picker,
):
    v1_08_track_id = v1_08_track_picker.value
    v1_08_profile = get_track_profile(v1_08_track_id)
    v1_08_variant = get_lab_track_variant("v1_08_training_gauntlet", v1_08_profile.track_id)
    v1_08_hardware = resolve_mlsysim_ref(v1_08_variant.hardware_ref)
    v1_08_model = resolve_mlsysim_ref(v1_08_variant.model_ref)
    v1_08_training = training_track_profile(
        v1_08_profile,
        v1_08_variant,
        v1_08_hardware,
        v1_08_model,
    )
    return (
        v1_08_hardware,
        v1_08_model,
        v1_08_profile,
        v1_08_track_id,
        v1_08_training,
        v1_08_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
        track_arc_context,
    v1_08_metadata,
    v1_08_profile,
    v1_08_training,
    v1_08_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 08
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                Training Gauntlet
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Weights &middot; Gradients &middot; Optimizer State &middot; Activations &middot; Validation
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 860px; line-height: 1.65;">
                {v1_08_variant.workload_summary} This lab asks where training,
                adaptation, or calibration should happen, then checks whether the memory
                stack and validation plan match the selected track.
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
                    {v1_08_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_08_training.workload_label}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Memory Stack</span>
                <span class="badge badge-warn">Batch Frontier</span>
                <span class="badge badge-fail">Training Plan</span>
            </div>
        </div>
        """),
        track_context(v1_08_profile),
        track_arc_context(v1_08_profile, v1_08_metadata.lab_id),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_08_training):
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
            <div style="margin-bottom: 3px;">1. <strong>Build the training memory stack:</strong>
                compare weights, gradients, optimizer state, activations, and data batch memory.</div>
            <div style="margin-bottom: 3px;">2. <strong>Sweep feasibility knobs:</strong>
                move batch size and observe when memory or throughput fails.</div>
            <div style="margin-bottom: 3px;">3. <strong>Choose a training plan:</strong>
                decide where training/adaptation happens and where validation must happen.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                If inference fits on {v1_08_training.label}, does training fit too,
                and if not, what adaptation or centralized plan is defensible?
            </div>
        </div>
    </div>
    """)
    return


# ===========================================================================
# ZONE B: CONTROLS AND COMPUTATION
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_08_training):
    v1_08_memory_prediction = mo.ui.radio(
        options={
            "Weights dominate": "weights",
            "Gradients dominate": "gradients",
            "Optimizer state dominates": "optimizer state",
            "Activations dominate": "activations",
            "Data batch dominates": "data batch",
        },
        label=f"{v1_08_training.label}: which training memory component do you expect to dominate?",
    )
    v1_08_memory_prediction
    return (v1_08_memory_prediction,)


@app.cell(hide_code=True)
def _(mo, v1_08_training):
    v1_08_batch_size = mo.ui.slider(
        start=v1_08_training.batch_min,
        stop=v1_08_training.batch_max,
        value=v1_08_training.default_batch_size,
        step=v1_08_training.batch_step,
        label="Batch size",
    )
    v1_08_batch_size
    return (v1_08_batch_size,)


@app.cell(hide_code=True)
def _(mo, v1_08_training):
    _strategy_options = {
        strategy.label: strategy.strategy_id
        for strategy in v1_08_training.strategy_options
    }
    v1_08_strategy_choice = mo.ui.dropdown(
        options=_strategy_options,
        value=v1_08_training.strategy_options[0].label,
        label="Training/adaptation strategy",
    )
    v1_08_reflection = mo.ui.text_area(
        label="Reflection",
        placeholder="Name where training happens, where validation happens, and the hidden cost or convergence risk you accept.",
        full_width=True,
    )
    return (v1_08_reflection, v1_08_strategy_choice)


@app.cell
def _(
    training_frontier,
    training_memory_stack,
    training_plan,
    v1_08_batch_size,
    v1_08_strategy_choice,
    v1_08_training,
):
    v1_08_memory_rows = tuple(
        training_memory_stack(
            v1_08_training,
            strategy_id=strategy.strategy_id,
            batch_size=v1_08_batch_size.value,
        )
        for strategy in v1_08_training.strategy_options
    )
    v1_08_frontier = training_frontier(
        v1_08_training,
        strategy_id=v1_08_strategy_choice.value,
    )
    v1_08_plan = training_plan(
        v1_08_training,
        strategy_id=v1_08_strategy_choice.value,
        batch_size=v1_08_batch_size.value,
    )
    v1_08_selected_stack = next(
        row for row in v1_08_memory_rows
        if row.strategy_id == v1_08_plan.selected_id
    )
    return (
        v1_08_frontier,
        v1_08_memory_rows,
        v1_08_plan,
        v1_08_selected_stack,
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
    v1_08_batch_size,
    v1_08_frontier,
    v1_08_memory_prediction,
    v1_08_memory_rows,
    v1_08_plan,
    v1_08_reflection,
    v1_08_selected_stack,
    v1_08_strategy_choice,
    v1_08_training,
):
    _component_fig = go.Figure()
    _component_fig.add_trace(go.Bar(
        x=["Weights", "Gradients", "Optimizer", "Activations", "Data batch"],
        y=[
            v1_08_selected_stack.weights_mb,
            v1_08_selected_stack.gradients_mb,
            v1_08_selected_stack.optimizer_mb,
            v1_08_selected_stack.activations_mb,
            v1_08_selected_stack.data_batch_mb,
        ],
        marker_color=[
            COLORS["BlueLine"],
            COLORS["OrangeLine"],
            COLORS["RedLine"],
            COLORS["GreenLine"],
            COLORS["Cloud"],
        ],
        text=[
            f"{v1_08_selected_stack.weights_mb:.1f}",
            f"{v1_08_selected_stack.gradients_mb:.1f}",
            f"{v1_08_selected_stack.optimizer_mb:.1f}",
            f"{v1_08_selected_stack.activations_mb:.1f}",
            f"{v1_08_selected_stack.data_batch_mb:.1f}",
        ],
        textposition="outside",
    ))
    _component_fig.add_hline(
        y=v1_08_selected_stack.budget_mb,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        annotation_text="memory budget",
        annotation_font_color=COLORS["RedLine"],
    )
    _component_fig.update_layout(
        height=350,
        xaxis=dict(title="Memory component", gridcolor="#f1f5f9"),
        yaxis=dict(title="Memory (MB)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=60),
    )
    apply_plotly_theme(_component_fig)

    _rows = "".join(
        f"""
        <tr>
          <td>{row.strategy_label}</td>
          <td>{row.total_mb:.2f} MB</td>
          <td>{row.budget_mb:.2f} MB</td>
          <td>{row.memory_utilization_pct:.1f}%</td>
          <td>{row.throughput_samples_s:.2f}</td>
          <td>{'yes' if row.feasible else 'no - violation'}</td>
          <td>{row.dominant_component}</td>
        </tr>
        """
        for row in v1_08_memory_rows
    )

    _frontier_fig = go.Figure()
    _frontier_fig.add_trace(go.Scatter(
        x=[point.batch_size for point in v1_08_frontier.points],
        y=[point.total_mb for point in v1_08_frontier.points],
        mode="lines+markers",
        marker=dict(
            color=[
                COLORS["GreenLine"] if point.feasible else COLORS["RedLine"]
                for point in v1_08_frontier.points
            ],
            size=7,
        ),
        line=dict(color=COLORS["BlueLine"], width=2.5),
        name="Training memory",
    ))
    _frontier_fig.add_hline(
        y=v1_08_selected_stack.budget_mb,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        annotation_text="memory budget",
        annotation_font_color=COLORS["RedLine"],
    )
    _frontier_fig.update_layout(
        height=360,
        xaxis=dict(title="Batch size", gridcolor="#f1f5f9"),
        yaxis=dict(title="Training memory (MB)", gridcolor="#f1f5f9"),
        margin=dict(l=70, r=20, t=35, b=50),
    )
    apply_plotly_theme(_frontier_fig)

    _frontier_rows = "".join(
        f"""
        <tr>
          <td>{point.batch_size}</td>
          <td>{point.total_mb:.2f} MB</td>
          <td>{point.throughput_samples_s:.2f}</td>
          <td>{'yes' if point.feasible else 'no - violation'}</td>
        </tr>
        """
        for point in v1_08_frontier.points
    )
    _validation_items = "".join(f"<li>{test}</li>" for test in v1_08_training.validation_tests)
    _rejections = "".join(f"<li>{item}</li>" for item in v1_08_plan.rejected_alternatives)

    _part_a = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Training Memory Stack</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which training component dominates {v1_08_training.workload_label}?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Training adds gradients, optimizer state, activations, and data batch memory on top of inference weights.</li>
            <li>Fine-tuning and adaptation reduce the trainable fraction, but they still need validation and rollback plans.</li>
            <li>Full local training can be the wrong activity even when local inference fits.</li>
          </ul>
          <div class="mlsysbook-callout"><strong>Track training story:</strong> {v1_08_training.training_story}</div>
        </div>
        """),
        v1_08_memory_prediction,
        v1_08_batch_size,
        mo.as_html(_component_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Table Fallback</h2>
          <table class="mlsysbook-table">
            <thead>
              <tr>
                <th>Strategy</th><th>Total</th><th>Budget</th><th>Utilization</th>
                <th>Throughput</th><th>Feasible</th><th>Dominant component</th>
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
          <div class="mlsysbook-part-title"><h2>Part B: Feasibility Frontier</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Which batch sizes fit the selected training or adaptation strategy?</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>Batch size changes both memory and throughput, so the first feasible point may not be the largest one.</li>
            <li>Checkpointing and adaptation reduce activations or trainable state but add hidden cost.</li>
            <li>The selected strategy max feasible batch is {v1_08_frontier.max_feasible_batch or 'none'}.</li>
          </ul>
        </div>
        """),
        mo.as_html(_frontier_fig),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Frontier Table</h2>
          <table class="mlsysbook-table">
            <thead><tr><th>Batch</th><th>Total memory</th><th>Throughput</th><th>Feasible</th></tr></thead>
            <tbody>{_frontier_rows}</tbody>
          </table>
        </div>
        """),
    ])

    _part_c = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Training Plan</h2></div>
          <div class="mlsysbook-callout"><strong>Systems question:</strong>
            Where should training happen, and where must validation happen?</div>
        </div>
        """),
        mo.Html("""
        <div class="mlsysbook-panel">
          <h2>What You Need To Know</h2>
          <ul class="mlsysbook-list">
            <li>A training plan is incomplete unless it names the training location and validation location.</li>
            <li>Centralized training, local adaptation, calibration, and validation are different activities.</li>
            <li>The deployment handoff is part of the risk: a trained checkpoint must still become a safe product artifact.</li>
          </ul>
        </div>
        """),
        v1_08_strategy_choice,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Computed Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Selected plan</strong>{v1_08_plan.selected_label}</div>
            <div class="mlsysbook-field"><strong>Feasible</strong>{'yes' if v1_08_plan.feasible else 'no - violation'}</div>
            <div class="mlsysbook-field"><strong>Training location</strong>{v1_08_plan.training_location}</div>
            <div class="mlsysbook-field"><strong>Validation location</strong>{v1_08_plan.validation_location}</div>
            <div class="mlsysbook-field"><strong>Total memory</strong>{v1_08_plan.total_memory_mb:.2f} MB</div>
            <div class="mlsysbook-field"><strong>Dominant component</strong>{v1_08_plan.dominant_component}</div>
          </div>
          <div class="mlsysbook-callout"><strong>Hidden cost:</strong> {v1_08_plan.hidden_cost}</div>
          <div class="mlsysbook-callout"><strong>Memo decision:</strong> {v1_08_plan.memo_summary}</div>
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
        mo.Html('<div class="mlsysbook-panel"><h2>Reflection</h2></div>'),
        v1_08_reflection,
    ])

    mo.ui.tabs({
        "Part A · Memory": _part_a,
        "Part B · Frontier": _part_b,
        "Part C · Plan": _part_c,
    })
    return


# ===========================================================================
# ZONE D: SYNTHESIS AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    v1_08_memory_prediction,
    v1_08_plan,
    v1_08_profile,
    v1_08_selected_stack,
    v1_08_training,
    v1_08_variant,
):
    if v1_08_memory_prediction.value is not None:
        ledger.save(chapter=8, design={
            "chapter": "v1_08",
            "track_id": v1_08_profile.track_id,
            "scenario_id": v1_08_variant.scenario_id,
            "hardware_ref": v1_08_training.hardware_ref,
            "model_ref": v1_08_training.model_ref,
            "completed": True,
            "memory_prediction": v1_08_memory_prediction.value,
            "selected_training_plan": v1_08_plan.selected_id,
            "training_location": v1_08_plan.training_location,
            "validation_location": v1_08_plan.validation_location,
            "dominant_component": v1_08_plan.dominant_component,
            "total_memory_mb": v1_08_selected_stack.total_mb,
        })

    mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Synthesis</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Track</strong>{v1_08_training.label}</div>
            <div class="mlsysbook-field"><strong>Selected plan</strong>{v1_08_plan.selected_label}</div>
            <div class="mlsysbook-field"><strong>Training location</strong>{v1_08_plan.training_location}</div>
            <div class="mlsysbook-field"><strong>Residual risk</strong>{v1_08_plan.residual_risk}</div>
          </div>
        </div>
        """),
        mo.Html("""
        <div class="mlsysbook-panel">
          <h2>Big Takeaways</h2>
          <ul class="mlsysbook-list">
            <li><strong>Training is not inference with a larger batch.</strong> Gradients, optimizer state, activations, and data movement change the feasibility problem.</li>
            <li><strong>Location is a design decision.</strong> Device tracks often validate or adapt locally while training centrally.</li>
            <li><strong>Feasibility is not enough.</strong> A plan must also name convergence risk, hidden cost, and deployment handoff.</li>
          </ul>
        </div>
        """),
        mo.Html(f"""
        <div class="lab-hud">
            <span class="hud-label">LAB</span>
            <span class="hud-value">08 &middot; Training Gauntlet</span>
            <span class="hud-label">TRACK</span>
            <span class="hud-value">{v1_08_profile.label}</span>
            <span style="flex:1;"></span>
            <span class="hud-label">ARTIFACT</span>
            <span class="hud-value">{v1_08_training.report_artifact}</span>
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
    v1_08_batch_size,
    v1_08_frontier,
    v1_08_memory_prediction,
    v1_08_memory_rows,
    v1_08_metadata,
    v1_08_plan,
    v1_08_profile,
    v1_08_reflection,
    v1_08_selected_stack,
    v1_08_training,
    v1_08_variant,
):
    _incomplete = []
    if v1_08_memory_prediction.value is None:
        _incomplete.append("Part A dominant-memory prediction")
    if not str(v1_08_reflection.value or "").strip():
        _incomplete.append("Part C reflection")

    _report = build_lab_report(
        v1_08_metadata,
        track=v1_08_profile.label,
        scenario=v1_08_variant.workload_summary,
        learning_objectives=(
            "Build a training memory stack for weights, gradients, optimizer state, activations, and data batches.",
            "Sweep batch size to find memory and throughput feasibility boundaries.",
            "Choose a training, adaptation, or calibration plan with validation location and deployment handoff risk.",
        ),
        predictions={
            "dominant_training_memory": v1_08_memory_prediction.value,
        },
        knob_settings={
            "batch_size": v1_08_batch_size.value,
            "selected_strategy": v1_08_plan.selected_id,
        },
        evidence_summary={
            "hardware_ref": v1_08_training.hardware_ref,
            "model_ref": v1_08_training.model_ref,
            "model_params_m": v1_08_training.model_params_m,
            "training_budget_mb": v1_08_selected_stack.budget_mb,
            "total_memory_mb": v1_08_selected_stack.total_mb,
            "dominant_component": v1_08_plan.dominant_component,
            "max_feasible_batch": v1_08_frontier.max_feasible_batch,
            "training_location": v1_08_plan.training_location,
            "validation_location": v1_08_plan.validation_location,
        },
        final_decision=v1_08_plan.memo_summary,
        big_takeaways=(
            "Training adds gradients, optimizer state, activations, data batches, and validation obligations.",
            "The selected track determines whether full training, adaptation, calibration, or centralized retraining is defensible.",
            "A training plan must include deployment handoff and residual convergence or validation risk.",
        ),
        reflections={
            "student_reflection": v1_08_reflection.value,
            "hidden_cost": v1_08_plan.hidden_cost,
            "deployment_handoff": v1_08_plan.deployment_handoff,
            "report_artifact": v1_08_training.report_artifact,
        },
        residual_risk=v1_08_plan.residual_risk,
        source_trace={
            "track_id": v1_08_profile.track_id,
            "scenario_id": v1_08_variant.scenario_id,
            "hardware_ref": v1_08_variant.hardware_ref,
            "model_ref": v1_08_variant.model_ref,
            "shared_helper": "mlsysbook_labs.training",
            "source_policy": v1_08_profile.source_policy,
        },
        result_snapshot={
            "training_profile": v1_08_training,
            "memory_rows": v1_08_memory_rows,
            "frontier": v1_08_frontier,
            "selected_stack": v1_08_selected_stack,
            "plan": v1_08_plan,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-08 training feasibility plan is generated locally from "
                "the selected track, your inputs, and the computed evidence."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
