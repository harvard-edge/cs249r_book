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
        compile_break_even,
        dispatch_stack,
        framework_track_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        report_export_panel,
        resolve_mlsysim_ref,
        runtime_decision,
        source_trace,
        track_arc_context,
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
        compile_break_even,
        dispatch_stack,
        framework_track_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        mo,
        report_export_panel,
        resolve_mlsysim_ref,
        runtime_decision,
        source_trace,
        track_arc_context,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_07_metadata = get_lab_metadata("vol1/lab_07_ml_frameworks.py")
    return (v1_07_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_07_track_picker = track_selector(default=_default_track)
    v1_07_track_picker
    return (v1_07_track_picker,)


@app.cell
def _(
    framework_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_07_track_picker,
):
    v1_07_track_id = v1_07_track_picker.value
    v1_07_profile = get_track_profile(v1_07_track_id)
    v1_07_variant = get_lab_track_variant("v1_07_framework_tax", v1_07_profile.track_id)
    v1_07_hardware = resolve_mlsysim_ref(v1_07_variant.hardware_ref)
    v1_07_model = resolve_mlsysim_ref(v1_07_variant.model_ref)
    v1_07_framework = framework_track_profile(
        v1_07_profile,
        v1_07_variant,
        v1_07_hardware,
        v1_07_model,
    )
    return (
        v1_07_framework,
        v1_07_hardware,
        v1_07_model,
        v1_07_profile,
        v1_07_track_id,
        v1_07_variant,
    )


@app.cell
def _(COLORS, mo):
    import html
    import math

    def v1_07_escape(value):
        return html.escape(str(value))

    def v1_07_fmt_float(value, digits=2):
        if value is None:
            return "not available"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    def v1_07_fmt_int(value):
        if value is None:
            return "no payback"
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)

    def v1_07_track_amount_story(profile):
        stories = {
            "iphone": {
                "runtime": "Core ML or TFLite-style local runtime with accelerator delegates",
                "amounts": "latency, memory footprint, operator support, thermal and battery evidence",
                "failure": "unsupported operators fall back off delegate and turn responsiveness into a battery or thermal problem",
                "validation": "delegate coverage, thermal soak, and battery replay",
            },
            "oura_ring": {
                "runtime": "TFLite Micro-like fixed kernels and a firmware memory arena",
                "amounts": "SRAM/flash footprint, wake time, operator resolver size, and OTA payload",
                "failure": "the runtime or custom op set exceeds the arena and cannot be recovered at inference time",
                "validation": "SRAM trace, flash image check, OTA payload check, and battery regression",
            },
            "robotaxi": {
                "runtime": "deterministic perception runtime with replayable p99 and p999 evidence",
                "amounts": "tail latency, supported plugins, fallback determinism, power, and safety validation",
                "failure": "portable fallback or uncertified plugins inject jitter into the safety loop",
                "validation": "p99/p999 replay, plugin audit, provider partition report, and fallback drill",
            },
            "cloud_fleet": {
                "runtime": "graph compiler and graph-capture path reused across high-volume requests",
                "amounts": "reuse count, graph-break rate, throughput, p99 latency, utilization, and cost/request",
                "failure": "dynamic shapes or graph breaks prevent compile amortization and raise SLA or cost risk",
                "validation": "load/SLA test, graph-break audit, cost/request canary, and rollback drill",
            },
        }
        return stories.get(profile.track_id, stories["iphone"])

    def v1_07_metric_card(title, value, subtitle="", color=None):
        _color = color or COLORS["BlueLine"]
        return mo.Html(f"""
        <div class="mlsysbook-field" style="border-top: 3px solid {_color}; min-width: 180px;">
          <strong>{v1_07_escape(title)}</strong>
          <div style="font-size:1.25rem; font-weight:800; color:{COLORS['Text']}; margin-top:4px;">
            {v1_07_escape(value)}
          </div>
          <div style="font-size:0.78rem; color:{COLORS['TextMuted']}; margin-top:2px;">
            {v1_07_escape(subtitle)}
          </div>
        </div>
        """)

    def v1_07_table(title, headers, rows):
        _headers = "".join(f"<th>{v1_07_escape(header)}</th>" for header in headers)
        _rows = []
        for row in rows:
            _cells = "".join(f"<td>{v1_07_escape(cell)}</td>" for cell in row)
            _rows.append(f"<tr>{_cells}</tr>")
        return mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>{v1_07_escape(title)}</h2>
          <table class="mlsysbook-table">
            <thead><tr>{_headers}</tr></thead>
            <tbody>{''.join(_rows)}</tbody>
          </table>
        </div>
        """)

    def v1_07_prediction_feedback(predicted, actual, labels, aligned, correction):
        if predicted is None:
            return mo.callout(
                mo.md("**Prediction checkpoint:** choose an option before treating the evidence as a decision."),
                kind="warn",
            )
        _predicted = labels.get(predicted, predicted)
        _actual = labels.get(actual, actual)
        _kind = "success" if predicted == actual else "warn"
        _body = aligned if predicted == actual else correction
        return mo.callout(
            mo.md(f"**Prediction vs actual:** you picked **{_predicted}**; the instrument shows **{_actual}**. {_body}"),
            kind=_kind,
        )

    def v1_07_math_peek(title, body):
        return mo.accordion({title: mo.md(body)})

    def v1_07_overhead_category(stack, support_row):
        if support_row and support_row["support_headroom_pct"] < 0:
            return "unsupported ops"
        if stack.dominant_overhead in ("memory traffic", "transfer"):
            return "memory traffic"
        if stack.dominant_overhead in ("hardware dispatch", "synchronization"):
            return "hardware dispatch"
        return "runtime dispatch"

    def v1_07_shape_penalty(runtime, profile, shape_dynamism_pct):
        dynamic_delta = max(0.0, float(shape_dynamism_pct) - float(profile.shape_dynamism_pct))
        mode = runtime.execution_mode.lower()
        if any(term in mode for term in ("static", "compiled", "captured", "generated", "ahead")):
            multiplier = 0.75
        elif any(term in mode for term in ("portable", "interpreter", "micro")):
            multiplier = 0.45
        else:
            multiplier = 0.20
        return dynamic_delta * multiplier

    def v1_07_support_rows(profile, dispatch_rows, break_evens, shape_dynamism_pct):
        break_even_by_runtime = {item.runtime_id: item for item in break_evens}
        runtime_by_id = {runtime.runtime_id: runtime for runtime in profile.runtime_options}
        rows = []
        for stack in dispatch_rows:
            runtime = runtime_by_id[stack.runtime_id]
            break_even = break_even_by_runtime[stack.runtime_id]
            adjusted_support = max(
                0.0,
                min(100.0, stack.kernel_support_pct - v1_07_shape_penalty(runtime, profile, shape_dynamism_pct)),
            )
            latency_headroom = profile.latency_budget_ms - stack.total_latency_ms
            footprint_headroom = profile.memory_budget_mb - stack.footprint_mb
            support_headroom = adjusted_support - profile.kernel_support_floor_pct
            compile_issue = (
                break_even.compile_cost_s > 0
                and (break_even.break_even_inferences is None or not break_even.pays_back)
            )
            violations = list(stack.violations)
            if support_headroom < 0:
                violations.append(
                    f"shape-adjusted support {adjusted_support:.1f}% < {profile.kernel_support_floor_pct:.1f}%"
                )
            if compile_issue:
                _be = v1_07_fmt_int(break_even.break_even_inferences)
                violations.append(f"compile/delegate payback not met at selected reuse; break-even {_be}")
            compatibility_score = max(0.0, latency_headroom / max(profile.latency_budget_ms, 1e-9))
            compatibility_score += max(0.0, support_headroom / max(100.0 - profile.kernel_support_floor_pct, 1e-9))
            compatibility_score += max(0.0, footprint_headroom / max(profile.memory_budget_mb, 1e-9))
            compatibility_score = 100.0 * compatibility_score / 3.0
            rows.append({
                "runtime_id": stack.runtime_id,
                "runtime_label": stack.runtime_label,
                "execution_mode": runtime.execution_mode,
                "latency_headroom_ms": latency_headroom,
                "footprint_headroom_mb": footprint_headroom,
                "adjusted_support_pct": adjusted_support,
                "support_headroom_pct": support_headroom,
                "compile_pays_back": break_even.pays_back,
                "break_even_inferences": break_even.break_even_inferences,
                "compatibility_score": compatibility_score,
                "feasible_with_shape": stack.feasible and support_headroom >= 0,
                "violations": tuple(violations),
                "portability_risk": runtime.portability_risk,
                "validation_requirement": runtime.validation_requirement,
                "residual_risk": runtime.residual_risk,
            })
        return tuple(rows)

    def v1_07_break_even_category(break_even, support_row):
        if support_row["support_headroom_pct"] < 0:
            return "shape_support_limit"
        if break_even.break_even_inferences is None or not break_even.pays_back:
            return "no_payback"
        return "pays_back"

    def v1_07_portability_cost_category(support_row):
        if support_row["support_headroom_pct"] < 0 or support_row["support_headroom_pct"] <= 5:
            return "operator_support"
        if support_row["footprint_headroom_mb"] < 0:
            return "memory_footprint"
        if support_row["latency_headroom_ms"] < 0 or support_row["latency_headroom_ms"] <= 5:
            return "latency_headroom"
        return "validation_evidence"

    def v1_07_validation_focus_actual(track_id):
        return {
            "iphone": "delegate_coverage",
            "oura_ring": "memory_trace",
            "robotaxi": "p99_replay",
            "cloud_fleet": "load_canary",
        }.get(track_id, "delegate_coverage")

    def v1_07_release_gate(decision, selected_support, selected_break_even, release_posture):
        issues = []
        if not selected_support["feasible_with_shape"]:
            issues.extend(selected_support["violations"] or ("selected runtime violates a deployment constraint",))
        elif selected_break_even.compile_cost_s > 0 and not selected_break_even.pays_back:
            issues.append("compile/delegate cost does not pay back at the selected reuse count")

        if issues:
            status = "Rework required"
            kind = "danger"
            action = "recover by changing runtime, reducing dynamism, increasing reuse, or choosing a narrower supported graph"
        elif release_posture == "ship":
            status = "Ready after validation"
            kind = "success"
            action = "run the required validation suite before shipment"
        elif release_posture == "canary":
            status = "Canary-ready"
            kind = "success"
            action = "ship behind rollback and compare the runtime path against the baseline"
        elif release_posture == "research":
            status = "Research only"
            kind = "warn"
            action = "keep this runtime out of the deployment path until the evidence packet is complete"
        else:
            status = "Awaiting release posture"
            kind = "warn"
            action = "choose a release posture in Part D"

        return {
            "status": status,
            "kind": kind,
            "issues": tuple(issues),
            "action": action,
        }

    def v1_07_constraint_callout(title, ok, detail, mitigation):
        return mo.callout(
            mo.md(f"**{title}:** {detail} **Mitigation:** {mitigation}"),
            kind="success" if ok else "danger",
        )

    return (
        v1_07_break_even_category,
        v1_07_constraint_callout,
        v1_07_fmt_float,
        v1_07_fmt_int,
        v1_07_math_peek,
        v1_07_metric_card,
        v1_07_overhead_category,
        v1_07_portability_cost_category,
        v1_07_prediction_feedback,
        v1_07_release_gate,
        v1_07_support_rows,
        v1_07_table,
        v1_07_track_amount_story,
        v1_07_validation_focus_actual,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_arc_context,
    track_context,
    v1_07_framework,
    v1_07_metadata,
    v1_07_profile,
    v1_07_track_amount_story,
    v1_07_variant,
):
    _amount_story = v1_07_track_amount_story(v1_07_profile)
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #94a3b8; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 07
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                ML Frameworks: Runtime Consequences
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Dispatch &middot; Graph Shape &middot; Fusion &middot; Portability &middot; Validation
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 900px; line-height: 1.65;">
                Chapter invariant: framework abstractions carry runtime consequences.
                Graph shape, dispatch, portability, and kernel support change the deployed
                system even when the model math is unchanged. Every track follows the same
                Part A-D concept sequence; the selected track changes persona, thresholds,
                evidence emphasis, failure mode, and report framing.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Concept Modules + Synthesis &middot; ~55 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_07_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_07_framework.workload_label}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Part A: Eager vs Graph Overhead</span>
                <span class="badge badge-warn">Part B: Fusion Boundary</span>
                <span class="badge badge-info">Part C: Portability Amounts</span>
                <span class="badge badge-fail">Part D: Release Evidence</span>
            </div>
        </div>
        """),
        track_context(v1_07_profile),
        track_arc_context(v1_07_profile, v1_07_metadata.lab_id),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Track Amount System</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Stakeholder</strong>{v1_07_variant.stakeholder}</div>
            <div class="mlsysbook-field"><strong>Runtime lens</strong>{_amount_story['runtime']}</div>
            <div class="mlsysbook-field"><strong>Measured amounts</strong>{_amount_story['amounts']}</div>
            <div class="mlsysbook-field"><strong>Natural failure</strong>{_amount_story['failure']}</div>
          </div>
        </div>
        """),
        source_trace({
            "chapter_invariant": "Framework abstractions carry runtime consequences.",
            "chapter_anchor": "book/quarto/contents/vol1/frameworks/frameworks.qmd",
            "concept_map": "book/quarto/contents/vol1/frameworks/frameworks_concepts.yml",
            "track_profile": v1_07_profile.track_id,
            "hardware_ref": v1_07_variant.hardware_ref,
            "model_ref": v1_07_variant.model_ref,
        }, summary="Opening assumptions come from the chapter anchors, selected track profile, and MLSysIM references."),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_07_framework):
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
            <div style="margin-bottom: 3px;">1. <strong>Diagnose execution overhead:</strong>
                separate useful compute from dispatch, transfer, sync, memory traffic, and unsupported-op fallback.</div>
            <div style="margin-bottom: 3px;">2. <strong>Find a fusion boundary:</strong>
                compare compile/delegate setup cost with reuse, shape stability, and supported kernels.</div>
            <div style="margin-bottom: 3px;">3. <strong>Reason in track amounts:</strong>
                explain how compatibility can cost latency, footprint, support, battery, safety, or cost.</div>
            <div style="margin-bottom: 3px;">4. <strong>Release with evidence:</strong>
                recommend a runtime only when deployment constraints and validation evidence line up.</div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 14px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Runtime Decision
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                Which framework/runtime path fits {v1_07_framework.label}, and what evidence
                proves that its graph shape, dispatch cost, and operator support are safe to deploy?
            </div>
        </div>
    </div>
    """)
    return


# ===========================================================================
# ZONE B: CONTROLS AND COMPUTATION
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_07_framework):
    v1_07_overhead_prediction = mo.ui.radio(
        options={
            "Runtime dispatch dominates because each op launches separately": "runtime dispatch",
            "Hardware dispatch or synchronization dominates": "hardware dispatch",
            "Memory traffic or transfer dominates": "memory traffic",
            "Unsupported-op fallback dominates": "unsupported ops",
        },
        label=f"Part A prediction for {v1_07_framework.label}: which framework overhead binds first?",
    )
    v1_07_op_count = mo.ui.slider(
        start=max(10, int(v1_07_framework.op_count / 4)),
        stop=max(20, int(v1_07_framework.op_count * 4)),
        value=v1_07_framework.op_count,
        step=max(1, int(v1_07_framework.op_count / 20)),
        label="Operation count in the hot path",
    )
    return (v1_07_op_count, v1_07_overhead_prediction)


@app.cell(hide_code=True)
def _(mo, v1_07_framework):
    v1_07_break_even_prediction = mo.ui.radio(
        options={
            "The compiled/delegate path pays back before expected reuse": "pays_back",
            "Compile/delegate setup does not pay back": "no_payback",
            "Shape or operator support becomes the real boundary": "shape_support_limit",
        },
        label="Part B prediction: what will decide the fusion/compile path?",
    )
    v1_07_reuse_count = mo.ui.slider(
        start=v1_07_framework.reuse_min,
        stop=v1_07_framework.reuse_max,
        value=v1_07_framework.default_reuse_count,
        step=v1_07_framework.reuse_step,
        label="Expected reuse count",
    )
    v1_07_shape_dynamism = mo.ui.slider(
        start=0,
        stop=max(60, int(v1_07_framework.shape_dynamism_pct * 3 + 12)),
        value=int(round(v1_07_framework.shape_dynamism_pct)),
        step=1,
        label="Shape dynamism / guard pressure (%)",
    )
    v1_07_part_b_checkpoint = mo.ui.radio(
        options={
            "Compile/delegate now": "compile",
            "Bucket or pad shapes before compiling": "bucket",
            "Stay eager or portable until reuse is proven": "defer",
        },
        label="Part B checkpoint: what should the team do next?",
    )
    return (
        v1_07_break_even_prediction,
        v1_07_part_b_checkpoint,
        v1_07_reuse_count,
        v1_07_shape_dynamism,
    )


@app.cell(hide_code=True)
def _(mo, v1_07_framework):
    _runtime_options = {
        runtime.label: runtime.runtime_id
        for runtime in v1_07_framework.runtime_options
    }
    v1_07_portability_prediction = mo.ui.radio(
        options={
            "Latency headroom": "latency_headroom",
            "Memory or firmware footprint": "memory_footprint",
            "Kernel/operator support": "operator_support",
            "Validation evidence and rollback": "validation_evidence",
        },
        label="Part C prediction: which amount will portability cost most?",
    )
    v1_07_runtime_choice = mo.ui.dropdown(
        options=_runtime_options,
        value=v1_07_framework.runtime_options[0].label,
        label="Runtime path to evaluate",
    )
    v1_07_part_c_checkpoint = mo.ui.radio(
        options={
            "Choose the target-native optimized runtime": "native",
            "Choose the portable interchange/runtime path": "portable",
            "Choose generated or fixed kernels": "fixed",
            "Keep this only as rollback/debug baseline": "rollback",
        },
        label="Part C checkpoint: which portability posture matches the evidence?",
    )
    return (
        v1_07_part_c_checkpoint,
        v1_07_portability_prediction,
        v1_07_runtime_choice,
    )


@app.cell(hide_code=True)
def _(mo):
    v1_07_validation_prediction = mo.ui.radio(
        options={
            "Delegate/operator coverage plus thermal or battery replay": "delegate_coverage",
            "SRAM/flash arena trace plus OTA payload check": "memory_trace",
            "p99/p999 replay plus plugin or provider partition audit": "p99_replay",
            "Load/SLA graph-break canary plus rollback drill": "load_canary",
        },
        label="Part D prediction: which validation evidence is non-negotiable for the selected track?",
    )
    v1_07_release_posture = mo.ui.radio(
        options={
            "Ship primary runtime after validation": "ship",
            "Canary with rollback baseline": "canary",
            "Keep as research/runtime prototype": "research",
        },
        label="Part D checkpoint: release posture",
    )
    v1_07_recommendation = mo.ui.text_area(
        label="Final runtime recommendation",
        placeholder=(
            "Name the runtime you recommend, the constraint that ruled out the naive choice, "
            "the evidence number, and the validation test that must run before deployment."
        ),
        full_width=True,
    )
    return (
        v1_07_recommendation,
        v1_07_release_posture,
        v1_07_validation_prediction,
    )


@app.cell
def _(
    compile_break_even,
    dispatch_stack,
    runtime_decision,
    v1_07_break_even_category,
    v1_07_framework,
    v1_07_op_count,
    v1_07_overhead_category,
    v1_07_portability_cost_category,
    v1_07_release_gate,
    v1_07_release_posture,
    v1_07_reuse_count,
    v1_07_runtime_choice,
    v1_07_shape_dynamism,
    v1_07_support_rows,
    v1_07_validation_focus_actual,
    v1_07_profile,
):
    v1_07_dispatch_rows = tuple(
        dispatch_stack(
            v1_07_framework,
            runtime_id=runtime.runtime_id,
            op_count=v1_07_op_count.value,
        )
        for runtime in v1_07_framework.runtime_options
    )
    v1_07_break_evens = tuple(
        compile_break_even(
            v1_07_framework,
            runtime_id=runtime.runtime_id,
            reuse_count=v1_07_reuse_count.value,
            op_count=v1_07_op_count.value,
        )
        for runtime in v1_07_framework.runtime_options
    )
    v1_07_support_adjusted_rows = v1_07_support_rows(
        v1_07_framework,
        v1_07_dispatch_rows,
        v1_07_break_evens,
        v1_07_shape_dynamism.value,
    )
    v1_07_decision = runtime_decision(
        v1_07_framework,
        runtime_id=v1_07_runtime_choice.value,
        reuse_count=v1_07_reuse_count.value,
        op_count=v1_07_op_count.value,
    )
    v1_07_selected_stack = next(
        item for item in v1_07_dispatch_rows
        if item.runtime_id == v1_07_decision.selected_id
    )
    v1_07_selected_break_even = next(
        item for item in v1_07_break_evens
        if item.runtime_id == v1_07_decision.selected_id
    )
    v1_07_selected_support = next(
        item for item in v1_07_support_adjusted_rows
        if item["runtime_id"] == v1_07_decision.selected_id
    )
    v1_07_actual_overhead_category = v1_07_overhead_category(
        v1_07_selected_stack,
        v1_07_selected_support,
    )
    v1_07_actual_break_even_category = v1_07_break_even_category(
        v1_07_selected_break_even,
        v1_07_selected_support,
    )
    v1_07_actual_portability_cost = v1_07_portability_cost_category(v1_07_selected_support)
    v1_07_actual_validation_focus = v1_07_validation_focus_actual(v1_07_profile.track_id)
    v1_07_release_result = v1_07_release_gate(
        v1_07_decision,
        v1_07_selected_support,
        v1_07_selected_break_even,
        v1_07_release_posture.value,
    )
    return (
        v1_07_actual_break_even_category,
        v1_07_actual_overhead_category,
        v1_07_actual_portability_cost,
        v1_07_actual_validation_focus,
        v1_07_break_evens,
        v1_07_decision,
        v1_07_dispatch_rows,
        v1_07_release_result,
        v1_07_selected_break_even,
        v1_07_selected_stack,
        v1_07_selected_support,
        v1_07_support_adjusted_rows,
    )


# ===========================================================================
# ZONE C: CONCEPT MODULES
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    go,
    mo,
    source_trace,
    v1_07_actual_break_even_category,
    v1_07_actual_overhead_category,
    v1_07_actual_portability_cost,
    v1_07_actual_validation_focus,
    v1_07_break_even_prediction,
    v1_07_break_evens,
    v1_07_constraint_callout,
    v1_07_decision,
    v1_07_dispatch_rows,
    v1_07_fmt_float,
    v1_07_fmt_int,
    v1_07_framework,
    v1_07_math_peek,
    v1_07_metric_card,
    v1_07_op_count,
    v1_07_overhead_prediction,
    v1_07_part_b_checkpoint,
    v1_07_part_c_checkpoint,
    v1_07_portability_prediction,
    v1_07_prediction_feedback,
    v1_07_profile,
    v1_07_recommendation,
    v1_07_release_posture,
    v1_07_release_result,
    v1_07_reuse_count,
    v1_07_runtime_choice,
    v1_07_selected_break_even,
    v1_07_selected_stack,
    v1_07_selected_support,
    v1_07_shape_dynamism,
    v1_07_support_adjusted_rows,
    v1_07_table,
    v1_07_track_amount_story,
    v1_07_validation_prediction,
    v1_07_variant,
):
    _amount_story = v1_07_track_amount_story(v1_07_profile)

    _labels = [row.runtime_label for row in v1_07_dispatch_rows]
    _latency_fig = go.Figure()
    _latency_fig.add_trace(go.Bar(
        x=_labels,
        y=[row.useful_compute_ms for row in v1_07_dispatch_rows],
        name="Useful compute",
        marker_color=COLORS["GreenLine"],
    ))
    _latency_fig.add_trace(go.Bar(
        x=_labels,
        y=[row.runtime_dispatch_ms for row in v1_07_dispatch_rows],
        name="Runtime dispatch",
        marker_color=COLORS["OrangeLine"],
    ))
    _latency_fig.add_trace(go.Bar(
        x=_labels,
        y=[row.hardware_dispatch_ms for row in v1_07_dispatch_rows],
        name="Hardware dispatch",
        marker_color=COLORS["RedLine"],
    ))
    _latency_fig.add_trace(go.Bar(
        x=_labels,
        y=[row.transfer_ms + row.sync_ms + row.memory_ms for row in v1_07_dispatch_rows],
        name="Transfer, sync, memory",
        marker_color=COLORS["BlueLine"],
    ))
    _latency_fig.add_hline(
        y=v1_07_framework.latency_budget_ms,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        annotation_text="latency budget",
        annotation_font_color=COLORS["RedLine"],
    )
    _latency_fig.update_layout(
        barmode="stack",
        height=390,
        xaxis=dict(title="Runtime path", gridcolor="#f1f5f9"),
        yaxis=dict(title="Latency stack (ms)", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=90),
    )
    apply_plotly_theme(_latency_fig)

    _break_fig = go.Figure()
    _break_fig.add_trace(go.Bar(
        x=[item.runtime_label for item in v1_07_break_evens],
        y=[item.break_even_inferences or 0 for item in v1_07_break_evens],
        marker_color=[
            COLORS["GreenLine"] if item.pays_back else COLORS["RedLine"]
            for item in v1_07_break_evens
        ],
        text=[v1_07_fmt_int(item.break_even_inferences) for item in v1_07_break_evens],
        textposition="outside",
    ))
    _break_fig.add_hline(
        y=v1_07_reuse_count.value,
        line_dash="dash",
        line_color=COLORS["BlueLine"],
        annotation_text="expected reuse",
        annotation_font_color=COLORS["BlueLine"],
    )
    _break_fig.update_layout(
        height=360,
        xaxis=dict(title="Runtime path", gridcolor="#f1f5f9"),
        yaxis=dict(title="Break-even inferences", gridcolor="#f1f5f9"),
        margin=dict(l=70, r=20, t=35, b=90),
    )
    apply_plotly_theme(_break_fig)

    _dispatch_table_rows = [
        (
            row.runtime_label,
            f"{row.total_latency_ms:.2f} ms",
            f"{row.overhead_pct:.1f}%",
            f"{row.footprint_mb:.3f} MB",
            f"{row.kernel_support_pct:.1f}%",
            "yes" if row.feasible else "no",
            row.dominant_overhead,
            "; ".join(row.violations) or "none",
        )
        for row in v1_07_dispatch_rows
    ]
    _support_table_rows = [
        (
            row["runtime_label"],
            row["execution_mode"],
            f"{row['adjusted_support_pct']:.1f}%",
            f"{row['support_headroom_pct']:.1f} pp",
            v1_07_fmt_int(row["break_even_inferences"]),
            "yes" if row["compile_pays_back"] else "no",
            "yes" if row["feasible_with_shape"] else "no",
            "; ".join(row["violations"]) or "none",
        )
        for row in v1_07_support_adjusted_rows
    ]
    _portability_rows = [
        (
            row["runtime_label"],
            f"{row['latency_headroom_ms']:.2f} ms",
            f"{row['footprint_headroom_mb']:.3f} MB",
            f"{row['adjusted_support_pct']:.1f}%",
            f"{row['compatibility_score']:.1f}",
            row["portability_risk"],
        )
        for row in v1_07_support_adjusted_rows
    ]
    _readiness_rows = [
        (
            row["runtime_label"],
            "yes" if row["feasible_with_shape"] else "no",
            "yes" if row["compile_pays_back"] else "no",
            row["validation_requirement"],
            row["residual_risk"],
        )
        for row in v1_07_support_adjusted_rows
    ]

    _overhead_labels = {
        "runtime dispatch": "runtime dispatch",
        "hardware dispatch": "hardware dispatch or synchronization",
        "memory traffic": "memory traffic or transfer",
        "unsupported ops": "unsupported-op fallback",
    }
    _break_even_labels = {
        "pays_back": "compiled/delegate path pays back",
        "no_payback": "compile/delegate setup does not pay back",
        "shape_support_limit": "shape or operator support is the boundary",
    }
    _portability_labels = {
        "latency_headroom": "latency headroom",
        "memory_footprint": "memory or firmware footprint",
        "operator_support": "kernel/operator support",
        "validation_evidence": "validation evidence and rollback",
    }
    _validation_labels = {
        "delegate_coverage": "delegate/operator coverage plus thermal or battery replay",
        "memory_trace": "SRAM/flash arena trace plus OTA payload check",
        "p99_replay": "p99/p999 replay plus plugin or provider audit",
        "load_canary": "load/SLA graph-break canary plus rollback drill",
    }

    _part_a_ok = v1_07_selected_stack.feasible and v1_07_selected_support["support_headroom_pct"] >= 0
    _part_a_detail = (
        f"{v1_07_decision.selected_label} totals {v1_07_selected_stack.total_latency_ms:.2f} ms "
        f"against a {v1_07_framework.latency_budget_ms:.2f} ms budget; dominant overhead is "
        f"{v1_07_selected_stack.dominant_overhead}."
    )
    _part_a_mitigation = "reduce op count, capture a longer graph, or select a runtime with better support for this track"

    _part_b_ok = (
        v1_07_selected_support["support_headroom_pct"] >= 0
        and (v1_07_selected_break_even.compile_cost_s == 0 or v1_07_selected_break_even.pays_back)
    )
    _part_b_detail = (
        f"Break-even is {v1_07_fmt_int(v1_07_selected_break_even.break_even_inferences)} inferences; "
        f"selected reuse is {v1_07_fmt_int(v1_07_reuse_count.value)} and shape-adjusted support is "
        f"{v1_07_selected_support['adjusted_support_pct']:.1f}%."
    )
    _part_b_mitigation = "increase reuse, bucket/pad shapes, or choose a runtime with a wider supported operator set"

    _part_c_ok = v1_07_selected_support["feasible_with_shape"]
    _part_c_detail = (
        f"Compatibility score is {v1_07_selected_support['compatibility_score']:.1f}; "
        f"portability risk is {v1_07_selected_support['portability_risk']}."
    )
    _part_c_mitigation = "make the target runtime explicit and test unsupported-op fallback before preserving portability"

    _release_issue_text = "; ".join(v1_07_release_result["issues"]) or "no blocking issue in the current scenario model"

    _part_a = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part A: Concept Module - Execution Overhead Depends On Reuse And Dynamism</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            {v1_07_variant.stakeholder} must decide whether {v1_07_framework.workload_label}
            can ship on {v1_07_framework.label}. The model is mathematically valid; the question is
            whether the framework execution stack changes the deployed system.</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Prediction</h2>
          <p>Commit to the overhead source before looking at the stack. The chapter's trap is assuming
          that useful compute is the only amount that matters.</p>
        </div>
        """),
        v1_07_overhead_prediction,
        mo.Html("<div class=\"mlsysbook-panel\"><h2>Manipulation</h2><p>Change the hot-path operation count. Small dynamic graphs pay dispatch repeatedly; larger stable graphs give the runtime more room to amortize overhead.</p></div>"),
        v1_07_op_count,
        mo.hstack([
            v1_07_metric_card("Selected runtime", v1_07_decision.selected_label, "changes in Part C", COLORS["BlueLine"]),
            v1_07_metric_card("Total latency", f"{v1_07_selected_stack.total_latency_ms:.2f} ms", f"budget {v1_07_framework.latency_budget_ms:.1f} ms", COLORS["OrangeLine"]),
            v1_07_metric_card("Dominant overhead", v1_07_selected_stack.dominant_overhead, "dispatch stack result", COLORS["RedLine"]),
        ], justify="start", gap=1),
        mo.as_html(_latency_fig),
        v1_07_table(
            "Evidence Table - Dispatch Stack",
            ("Runtime", "Total latency", "Overhead", "Footprint", "Kernel support", "Feasible", "Dominant overhead", "Violation"),
            _dispatch_table_rows,
        ),
        v1_07_prediction_feedback(
            v1_07_overhead_prediction.value,
            v1_07_actual_overhead_category,
            _overhead_labels,
            "That is the amount system the selected runtime exposes first.",
            "The stack shows why framework overhead is not a constant; the binding term changes with runtime path and track constraints.",
        ),
        v1_07_constraint_callout(
            "Consequence boundary",
            _part_a_ok,
            _part_a_detail,
            _part_a_mitigation,
        ),
        v1_07_math_peek(
            "Math Peek / Source Model - dispatch tax",
            f"""
The chapter defines dispatch tax as host/runtime orchestration relative to useful work:

$$
\\text{{Overhead Ratio}} =
\\frac{{N_{{ops}} \\cdot t_{{dispatch}}}}{{T_{{compute}} + T_{{memory}}}}
$$

For **{v1_07_decision.selected_label}**, the notebook-local scenario uses
`dispatch_stack()` with `op_count = {v1_07_op_count.value}` and the selected track profile.
Useful compute is {v1_07_selected_stack.useful_compute_ms:.2f} ms; non-compute overhead is
{v1_07_selected_stack.total_latency_ms - v1_07_selected_stack.useful_compute_ms:.2f} ms.
""",
        ),
        source_trace({
            "chapter_anchor": "Execution Problem / The dispatch tax",
            "formula": "Overhead Ratio = N_ops * t_dispatch / (T_compute + T_memory)",
            "helper": "mlsysbook_labs.frameworks.dispatch_stack",
            "hardware_ref": v1_07_framework.hardware_ref,
            "model_ref": v1_07_framework.model_ref,
        }, summary="Part A evidence is computed from dispatch_stack() and the selected track profile."),
    ])

    _part_b = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part B: Concept Module - Fusion Helps Only Inside Supported Shapes</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            {v1_07_variant.stakeholder} wants the runtime to remove dispatch and memory traffic through
            graph capture, delegate setup, or fusion. The boundary is whether compile cost, reuse, and
            supported shapes line up.</div>
        </div>
        """),
        mo.Html("<div class=\"mlsysbook-panel\"><h2>Prediction</h2><p>Choose whether the compiled/delegate path pays back, fails to amortize, or is blocked by shape/operator support.</p></div>"),
        v1_07_break_even_prediction,
        mo.Html("<div class=\"mlsysbook-panel\"><h2>Manipulation</h2><p>Move both reuse and shape dynamism. Reuse amortizes compile cost; dynamism increases guard pressure and can reduce supported graph coverage.</p></div>"),
        mo.hstack([v1_07_reuse_count, v1_07_shape_dynamism], justify="start", gap=2),
        mo.as_html(_break_fig),
        v1_07_table(
            "Evidence Table - Break-Even And Supported Shapes",
            ("Runtime", "Execution mode", "Shape-adjusted support", "Support headroom", "Break-even", "Pays back", "Feasible with shape", "Violation"),
            _support_table_rows,
        ),
        v1_07_prediction_feedback(
            v1_07_break_even_prediction.value,
            v1_07_actual_break_even_category,
            _break_even_labels,
            "That matches the boundary shown by the reuse and support model.",
            "Compilation is not a switch. The payback depends on reuse, and the fusion benefit only exists inside supported graph regions.",
        ),
        v1_07_constraint_callout(
            "Fusion boundary",
            _part_b_ok,
            _part_b_detail,
            _part_b_mitigation,
        ),
        v1_07_math_peek(
            "Math Peek / Source Model - compile break-even",
            f"""
The chapter's compile decision rule is:

$$
N_{{breakeven}} =
\\frac{{T_{{compile}}}}{{T_{{eager}} - T_{{compiled}}}}
$$

The selected runtime has compile/delegate cost {v1_07_selected_break_even.compile_cost_s:.1f} s and
per-inference savings {v1_07_selected_break_even.per_inference_savings_ms:.2f} ms against the baseline.
The selected shape-dynamism pressure is {v1_07_shape_dynamism.value}% and the adjusted support floor is
{v1_07_framework.kernel_support_floor_pct:.1f}%.
""",
        ),
        source_trace({
            "chapter_anchor": "Kernel fusion / Hybrid JIT and compilation",
            "formula": "N_breakeven = T_compile / (T_eager - T_compiled)",
            "helper": "mlsysbook_labs.frameworks.compile_break_even",
            "local_model": "v1_07_support_rows adjusts support by shape dynamism without editing shared helpers",
            "runtime_id": v1_07_decision.selected_id,
        }, summary="Part B combines compile_break_even() with a notebook-local supported-shape boundary."),
        mo.Html("<div class=\"mlsysbook-panel\"><h2>Checkpoint</h2><p>Record the action you would take before asking the team to optimize this graph.</p></div>"),
        v1_07_part_b_checkpoint,
    ])

    _part_c = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part C: Concept Module - Portability Is An Amount-System Trade</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            The team can keep a portable path or choose a narrower target runtime. For {v1_07_framework.label},
            portability is paid in {_amount_story['amounts']}.</div>
        </div>
        """),
        mo.Html("<div class=\"mlsysbook-panel\"><h2>Prediction</h2><p>Before choosing a runtime, predict which amount compatibility will consume first.</p></div>"),
        v1_07_portability_prediction,
        mo.Html("<div class=\"mlsysbook-panel\"><h2>Manipulation</h2><p>Select a runtime path. The table compares portability risk to latency headroom, footprint headroom, and supported operator coverage.</p></div>"),
        v1_07_runtime_choice,
        v1_07_table(
            "Evidence Table - Portability Amounts",
            ("Runtime", "Latency headroom", "Footprint headroom", "Adjusted support", "Compatibility score", "Portability risk"),
            _portability_rows,
        ),
        mo.hstack([
            v1_07_metric_card("Selected runtime", v1_07_decision.selected_label, v1_07_selected_support["execution_mode"], COLORS["BlueLine"]),
            v1_07_metric_card("Support headroom", f"{v1_07_selected_support['support_headroom_pct']:.1f} pp", f"floor {v1_07_framework.kernel_support_floor_pct:.1f}%", COLORS["GreenLine"]),
            v1_07_metric_card("Compatibility score", f"{v1_07_selected_support['compatibility_score']:.1f}", "0-100 scenario score", COLORS["OrangeLine"]),
        ], justify="start", gap=1),
        v1_07_prediction_feedback(
            v1_07_portability_prediction.value,
            v1_07_actual_portability_cost,
            _portability_labels,
            "That is the amount compatibility consumes in this selected runtime.",
            "The selected runtime shows that portability is not free: the cost appears in the track's binding amount system.",
        ),
        v1_07_constraint_callout(
            "Portability trade",
            _part_c_ok,
            _part_c_detail,
            _part_c_mitigation,
        ),
        v1_07_math_peek(
            "Math Peek / Source Model - compatibility score",
            f"""
The notebook converts portability into normalized headroom amounts:

```
compatibility_score =
  mean(latency_headroom / latency_budget,
       support_headroom / available_support_headroom,
       footprint_headroom / memory_budget) * 100
```

This is not a framework leaderboard. It is a scenario model that forces the same runtime choice to pass
{v1_07_framework.primary_metric} while respecting {v1_07_framework.guardrail_metric}.
""",
        ),
        source_trace({
            "chapter_anchor": "Deployment Targets / Framework Selection / ONNX portability",
            "source_claim": "compatibility can lose target-specific optimizations or custom operators",
            "profile_primary_metric": v1_07_framework.primary_metric,
            "profile_guardrail_metric": v1_07_framework.guardrail_metric,
            "runtime_id": v1_07_decision.selected_id,
        }, summary="Part C interprets portability through the selected track's amount system."),
        mo.Html("<div class=\"mlsysbook-panel\"><h2>Checkpoint</h2><p>Choose the portability posture you would defend in the design memo.</p></div>"),
        v1_07_part_c_checkpoint,
    ])

    _part_d = mo.vstack([
        mo.Html(f"""
        <div class="mlsysbook-panel mlsysbook-nugget">
          <div class="mlsysbook-part-title"><h2>Part D: Concept Module - Framework Selection Requires Validation Evidence</h2></div>
          <div class="mlsysbook-callout"><strong>Scenario:</strong>
            {v1_07_variant.stakeholder} needs a release recommendation, not just a chart. The runtime must satisfy
            deployment constraints and produce validation evidence that survives the selected track.</div>
        </div>
        """),
        mo.Html("<div class=\"mlsysbook-panel\"><h2>Prediction</h2><p>Choose the evidence that would make the recommendation credible for this track.</p></div>"),
        v1_07_validation_prediction,
        mo.Html("<div class=\"mlsysbook-panel\"><h2>Manipulation</h2><p>Set the release posture after inspecting feasibility, payback, rejected alternatives, and validation requirements.</p></div>"),
        v1_07_release_posture,
        v1_07_table(
            "Evidence Table - Release Readiness",
            ("Runtime", "Feasible with shape", "Compile pays back", "Validation requirement", "Residual risk"),
            _readiness_rows,
        ),
        v1_07_prediction_feedback(
            v1_07_validation_prediction.value,
            v1_07_actual_validation_focus,
            _validation_labels,
            "That evidence matches the selected track's deployment risk.",
            "The selected track changes what evidence is credible. A runtime recommendation without the right validation test is incomplete.",
        ),
        mo.callout(
            mo.md(
                f"**Release gate:** {v1_07_release_result['status']}. "
                f"Blocking issue: {_release_issue_text}. "
                f"Next action: {v1_07_release_result['action']}."
            ),
            kind=v1_07_release_result["kind"],
        ),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Selected Runtime Evidence</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Selected runtime</strong>{v1_07_decision.selected_label}</div>
            <div class="mlsysbook-field"><strong>Feasible</strong>{'yes' if v1_07_selected_support['feasible_with_shape'] else 'no'}</div>
            <div class="mlsysbook-field"><strong>Total latency</strong>{v1_07_selected_stack.total_latency_ms:.2f} ms / {v1_07_framework.latency_budget_ms:.1f} ms</div>
            <div class="mlsysbook-field"><strong>Break-even</strong>{v1_07_fmt_int(v1_07_selected_break_even.break_even_inferences)}</div>
            <div class="mlsysbook-field"><strong>Adjusted support</strong>{v1_07_selected_support['adjusted_support_pct']:.1f}% / {v1_07_framework.kernel_support_floor_pct:.1f}%</div>
            <div class="mlsysbook-field"><strong>Validation</strong>{v1_07_decision.validation_requirement}</div>
          </div>
          <div class="mlsysbook-callout"><strong>Unsupported-op warning:</strong> {v1_07_decision.unsupported_op_warning}</div>
          <div class="mlsysbook-callout"><strong>Memo decision:</strong> {v1_07_decision.memo_summary}</div>
        </div>
        """),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Rejected Alternatives</h2>
          <ul class="mlsysbook-list">{''.join(f'<li>{item}</li>' for item in v1_07_decision.rejected_alternatives)}</ul>
          <h2>Track Validation Tests</h2>
          <ul class="mlsysbook-list">{''.join(f'<li>{test}</li>' for test in v1_07_framework.validation_tests)}</ul>
        </div>
        """),
        v1_07_math_peek(
            "Math Peek / Source Model - deployment feasibility rule",
            f"""
The release decision is an amount-system predicate:

```
latency_ms <= {v1_07_framework.latency_budget_ms:.1f}
footprint_mb <= {v1_07_framework.memory_budget_mb:.3f}
support_pct >= {v1_07_framework.kernel_support_floor_pct:.1f}
reuse_count >= break_even_inferences  # when compile/delegate cost is nonzero
validation_evidence matches selected track
```

Framework selection is therefore constrained optimization, not a framework popularity contest.
""",
        ),
        source_trace({
            "chapter_anchor": "Framework Selection / Fallacies and Pitfalls",
            "helper": "mlsysbook_labs.frameworks.runtime_decision",
            "runtime_id": v1_07_decision.selected_id,
            "validation_requirement": v1_07_decision.validation_requirement,
            "residual_risk": v1_07_decision.residual_risk,
        }, summary="Part D turns runtime evidence into a release recommendation."),
        mo.Html("<div class=\"mlsysbook-panel\"><h2>Checkpoint Report Decision</h2><p>Write the final runtime recommendation with the assumption and evidence you would sign.</p></div>"),
        v1_07_recommendation,
    ])

    mo.ui.tabs({
        "Part A - Execution Overhead": _part_a,
        "Part B - Fusion Boundary": _part_b,
        "Part C - Portability Trade": _part_c,
        "Part D - Release Evidence": _part_d,
        "Synthesis": mo.md("Use the synthesis recommendation below after completing Parts A-D."),
    })
    return


# ===========================================================================
# ZONE D: SYNTHESIS AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    source_trace,
    v1_07_actual_break_even_category,
    v1_07_actual_overhead_category,
    v1_07_actual_portability_cost,
    v1_07_actual_validation_focus,
    v1_07_break_even_prediction,
    v1_07_decision,
    v1_07_framework,
    v1_07_op_count,
    v1_07_overhead_prediction,
    v1_07_part_b_checkpoint,
    v1_07_part_c_checkpoint,
    v1_07_portability_prediction,
    v1_07_profile,
    v1_07_recommendation,
    v1_07_release_posture,
    v1_07_release_result,
    v1_07_reuse_count,
    v1_07_runtime_choice,
    v1_07_selected_break_even,
    v1_07_selected_stack,
    v1_07_selected_support,
    v1_07_shape_dynamism,
    v1_07_validation_prediction,
    v1_07_variant,
):
    _recommendation_text = str(v1_07_recommendation.value or "").strip()
    if v1_07_overhead_prediction.value is not None:
        ledger.save(chapter=7, design={
            "chapter": "v1_07",
            "track_id": v1_07_profile.track_id,
            "scenario_id": v1_07_variant.scenario_id,
            "hardware_ref": v1_07_framework.hardware_ref,
            "model_ref": v1_07_framework.model_ref,
            "completed": bool(_recommendation_text),
            "part_a_prediction": v1_07_overhead_prediction.value,
            "part_a_actual_dominant_overhead": v1_07_actual_overhead_category,
            "part_b_prediction": v1_07_break_even_prediction.value,
            "part_b_actual_boundary": v1_07_actual_break_even_category,
            "part_b_checkpoint": v1_07_part_b_checkpoint.value,
            "part_c_prediction": v1_07_portability_prediction.value,
            "part_c_actual_portability_cost": v1_07_actual_portability_cost,
            "part_c_checkpoint": v1_07_part_c_checkpoint.value,
            "part_d_validation_prediction": v1_07_validation_prediction.value,
            "part_d_actual_validation_focus": v1_07_actual_validation_focus,
            "release_posture": v1_07_release_posture.value,
            "release_status": v1_07_release_result["status"],
            "operation_count": v1_07_op_count.value,
            "reuse_count": v1_07_reuse_count.value,
            "shape_dynamism_pct": v1_07_shape_dynamism.value,
            "selected_runtime": v1_07_runtime_choice.value,
            "dominant_overhead": v1_07_decision.dominant_overhead,
            "break_even_inferences": v1_07_selected_break_even.break_even_inferences,
            "total_latency_ms": v1_07_selected_stack.total_latency_ms,
            "kernel_support_pct": v1_07_selected_support["adjusted_support_pct"],
            "runtime_feasible": v1_07_selected_support["feasible_with_shape"],
            "validation_requirement": v1_07_decision.validation_requirement,
            "residual_risk": v1_07_decision.residual_risk,
            "final_recommendation": _recommendation_text,
        })

    def build_synthesis():
        return mo.vstack([
            mo.Html(f"""
            <div class="mlsysbook-panel">
              <h2>Synthesis: Runtime Deployment Recommendation</h2>
              <p>The chapter invariant is now a deployment recommendation: a framework abstraction is
              acceptable only when its graph shape, dispatch cost, operator support, portability risk,
              and validation evidence survive the selected track.</p>
              <div class="mlsysbook-grid">
                <div class="mlsysbook-field"><strong>Track</strong>{v1_07_framework.label}</div>
                <div class="mlsysbook-field"><strong>Selected runtime</strong>{v1_07_decision.selected_label}</div>
                <div class="mlsysbook-field"><strong>Release status</strong>{v1_07_release_result['status']}</div>
                <div class="mlsysbook-field"><strong>Dominant overhead</strong>{v1_07_decision.dominant_overhead}</div>
                <div class="mlsysbook-field"><strong>Break-even</strong>{v1_07_selected_break_even.break_even_inferences or 'no payback'}</div>
                <div class="mlsysbook-field"><strong>Residual risk</strong>{v1_07_decision.residual_risk}</div>
              </div>
            </div>
            """),
            mo.Html("""
            <div class="mlsysbook-panel">
              <h2>Big Takeaways</h2>
              <ul class="mlsysbook-list">
                <li><strong>Execution mode is physical.</strong> Eager and graph execution pay different dispatch and memory costs depending on reuse and dynamism.</li>
                <li><strong>Fusion is conditional.</strong> It removes dispatch and memory traffic only inside supported, stable graph regions.</li>
                <li><strong>Portability has a price.</strong> Compatibility can consume latency, memory, supported operators, validation evidence, or rollback simplicity.</li>
                <li><strong>Selection needs evidence.</strong> The runtime decision is valid only inside a source-traced operating envelope.</li>
              </ul>
            </div>
            """),
            source_trace({
                "track_id": v1_07_profile.track_id,
                "scenario_id": v1_07_variant.scenario_id,
                "hardware_ref": v1_07_variant.hardware_ref,
                "model_ref": v1_07_variant.model_ref,
                "shared_helper": "mlsysbook_labs.frameworks",
                "chapter_anchor": "ML Frameworks / Framework Selection",
                "source_policy": v1_07_profile.source_policy,
            }, summary="The synthesis recommendation is source-traced to chapter anchors, track metadata, and helper calculations."),
            mo.Html(f"""
            <div class="lab-hud">
                <span class="hud-label">LAB</span>
                <span class="hud-value">07 &middot; ML Frameworks</span>
                <span class="hud-label">TRACK</span>
                <span class="hud-value">{v1_07_profile.label}</span>
                <span style="flex:1;"></span>
                <span class="hud-label">ARTIFACT</span>
                <span class="hud-value">{v1_07_framework.report_artifact}</span>
                <span class="hud-label">STATUS</span>
                <span class="hud-active">{v1_07_release_result['status']}</span>
            </div>
            """),
        ])

    build_synthesis()
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    mo,
    report_export_panel,
    v1_07_actual_break_even_category,
    v1_07_actual_overhead_category,
    v1_07_actual_portability_cost,
    v1_07_actual_validation_focus,
    v1_07_break_even_prediction,
    v1_07_break_evens,
    v1_07_decision,
    v1_07_dispatch_rows,
    v1_07_framework,
    v1_07_metadata,
    v1_07_op_count,
    v1_07_overhead_prediction,
    v1_07_part_b_checkpoint,
    v1_07_part_c_checkpoint,
    v1_07_portability_prediction,
    v1_07_profile,
    v1_07_recommendation,
    v1_07_release_posture,
    v1_07_release_result,
    v1_07_reuse_count,
    v1_07_runtime_choice,
    v1_07_selected_break_even,
    v1_07_selected_stack,
    v1_07_selected_support,
    v1_07_shape_dynamism,
    v1_07_support_adjusted_rows,
    v1_07_validation_prediction,
    v1_07_variant,
):
    _incomplete = []
    if v1_07_overhead_prediction.value is None:
        _incomplete.append("Part A overhead prediction")
    if v1_07_break_even_prediction.value is None:
        _incomplete.append("Part B break-even prediction")
    if v1_07_part_b_checkpoint.value is None:
        _incomplete.append("Part B checkpoint")
    if v1_07_portability_prediction.value is None:
        _incomplete.append("Part C portability prediction")
    if v1_07_part_c_checkpoint.value is None:
        _incomplete.append("Part C checkpoint")
    if v1_07_validation_prediction.value is None:
        _incomplete.append("Part D validation prediction")
    if v1_07_release_posture.value is None:
        _incomplete.append("Part D release posture")
    if not str(v1_07_recommendation.value or "").strip():
        _incomplete.append("Final runtime recommendation")

    _report = build_lab_report(
        v1_07_metadata,
        track=v1_07_profile.label,
        scenario=v1_07_variant.workload_summary,
        learning_objectives=(
            "Diagnose how eager and graph execution pay different dispatch and memory overheads.",
            "Calculate compile or delegate break-even while checking shape and operator support.",
            "Explain portability as a track-specific amount-system trade.",
            "Make a runtime deployment recommendation backed by validation evidence.",
        ),
        predictions={
            "part_a_overhead": v1_07_overhead_prediction.value,
            "part_b_boundary": v1_07_break_even_prediction.value,
            "part_c_portability_cost": v1_07_portability_prediction.value,
            "part_d_validation_focus": v1_07_validation_prediction.value,
        },
        knob_settings={
            "operation_count": v1_07_op_count.value,
            "reuse_count": v1_07_reuse_count.value,
            "shape_dynamism_pct": v1_07_shape_dynamism.value,
            "selected_runtime": v1_07_runtime_choice.value,
            "part_b_checkpoint": v1_07_part_b_checkpoint.value,
            "part_c_checkpoint": v1_07_part_c_checkpoint.value,
            "release_posture": v1_07_release_posture.value,
        },
        binding_constraints={
            "actual_overhead_category": v1_07_actual_overhead_category,
            "actual_break_even_boundary": v1_07_actual_break_even_category,
            "actual_portability_cost": v1_07_actual_portability_cost,
            "actual_validation_focus": v1_07_actual_validation_focus,
            "release_status": v1_07_release_result["status"],
        },
        evidence_summary={
            "hardware_ref": v1_07_framework.hardware_ref,
            "model_ref": v1_07_framework.model_ref,
            "latency_budget_ms": v1_07_framework.latency_budget_ms,
            "memory_budget_mb": v1_07_framework.memory_budget_mb,
            "kernel_support_floor_pct": v1_07_framework.kernel_support_floor_pct,
            "selected_runtime": v1_07_decision.selected_label,
            "total_latency_ms": v1_07_selected_stack.total_latency_ms,
            "dominant_overhead": v1_07_decision.dominant_overhead,
            "break_even_inferences": v1_07_selected_break_even.break_even_inferences,
            "shape_adjusted_support_pct": v1_07_selected_support["adjusted_support_pct"],
            "runtime_feasible": v1_07_selected_support["feasible_with_shape"],
            "unsupported_op_warning": v1_07_decision.unsupported_op_warning,
            "validation_requirement": v1_07_decision.validation_requirement,
        },
        final_decision={
            "memo_summary": v1_07_decision.memo_summary,
            "release_status": v1_07_release_result["status"],
            "student_recommendation": v1_07_recommendation.value,
        },
        big_takeaways=(
            "Framework abstractions change deployed runtime behavior, not just source-code style.",
            "Compile and fusion pay back only when graph reuse and supported shapes are real.",
            "Portability can cost performance or capability, so unsupported-op evidence belongs in the release memo.",
            "A runtime recommendation is valid only with track-specific validation evidence.",
        ),
        reflections={
            "student_recommendation": v1_07_recommendation.value,
            "rejected_alternatives": v1_07_decision.rejected_alternatives,
            "validation_requirement": v1_07_decision.validation_requirement,
            "release_gate_issues": v1_07_release_result["issues"],
            "report_artifact": v1_07_framework.report_artifact,
        },
        residual_risk=v1_07_decision.residual_risk,
        source_trace={
            "track_id": v1_07_profile.track_id,
            "scenario_id": v1_07_variant.scenario_id,
            "hardware_ref": v1_07_variant.hardware_ref,
            "model_ref": v1_07_variant.model_ref,
            "shared_helper": "mlsysbook_labs.frameworks",
            "notebook_local_helpers": "v1_07_* formatting, support-boundary, release-gate helpers",
            "source_policy": v1_07_profile.source_policy,
        },
        result_snapshot={
            "dispatch_rows": v1_07_dispatch_rows,
            "break_even_rows": v1_07_break_evens,
            "support_adjusted_rows": v1_07_support_adjusted_rows,
            "selected_stack": v1_07_selected_stack,
            "selected_break_even": v1_07_selected_break_even,
            "selected_support": v1_07_selected_support,
            "decision": v1_07_decision,
            "release_result": v1_07_release_result,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-07 runtime deployment recommendation is generated locally from the "
                "selected track, structured predictions, manipulation controls, and computed evidence."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
