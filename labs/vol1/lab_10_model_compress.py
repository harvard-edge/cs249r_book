import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 10: Compression That Executes · MLSysBook")


@app.cell
async def _():
    import sys
    from pathlib import Path
    import marimo as mo

    if sys.platform == "emscripten":
        import micropip

        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        labs_dir = Path(__file__).resolve().parents[1]
        if str(labs_dir) not in sys.path:
            sys.path.insert(0, str(labs_dir))
        from bootstrap import native_bootstrap

        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.engine.v1_10_experiments import (
        MODEL_ID, MODEL_KEYS, RECIPE_SEQUENCES, SPARSITY_CASES,
        evaluate_distillation, evaluate_precision, evaluate_recipe_id,
        evaluate_resource, evaluate_sparsity_case, get_scenario, snapshot_result,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata,
        get_track_profile, report_export_panel, track_context,
    )
    from mlsysbook_labs.experiment_evidence import capture_evidence, audit_evidence

    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, MODEL_ID, MODEL_KEYS,
        RECIPE_SEQUENCES, SPARSITY_CASES,
        apply_plotly_theme, audit_evidence, build_lab_report, capture_evidence,
        evaluate_distillation, evaluate_precision, evaluate_recipe_id,
        evaluate_resource, evaluate_sparsity_case, get_lab_metadata, get_scenario,
        get_track_profile, go, ledger, mo, report_export_panel,
        snapshot_result, track_context,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Deployment track", on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(mo, scenario, track_id):
    _track_key = track_id
    a_storage = mo.ui.dropdown(
        {"INT8 weights": 8, "INT4 weights": 4}, value="INT8 weights", label="Stored weight precision"
    )
    a_compute = mo.ui.dropdown(
        {"INT8 execution": 8, "FP16 execution": 16}, value="INT8 execution", label="Execution precision"
    )
    a_state = mo.ui.dropdown(
        {"FP16 runtime state": 16, "INT8 runtime state": 8}, value="FP16 runtime state", label="Runtime-state precision"
    )
    b_bits = mo.ui.dropdown(
        {"INT8": 8, "6-bit": 6, "INT4": 4}, value="INT8", label="Stored weight precision"
    )
    b_calibration = mo.ui.dropdown(
        {"Post-training calibration": "ptq", "Quantization-aware training": "qat"},
        value="Post-training calibration", label="Quality-recovery method",
    )
    c_sparse = mo.ui.dropdown(
        {
            "50% masked dense tensor": "dense_mask_50",
            "25% indexed sparse": "indexed_25",
            "50% indexed sparse": "indexed_50",
            "25% structured pruning": "structured_25",
            "50% structured pruning": "structured_50",
            "50% N:M pruning": "n_m_50",
        },
        value="25% structured pruning", label="Sparse representation",
    )
    c_index = mo.ui.dropdown(
        {"16-bit indices": 16, "32-bit indices": 32}, value="16-bit indices", label="Sparse index width"
    )
    d_student = mo.ui.dropdown(
        {"Small dense student": "small_dense", "Tiny dense student": "tiny_dense"},
        value="Small dense student", label="Student candidate",
    )
    d_volume = mo.ui.dropdown(
        scenario.lifetime_inferences,
        value=scenario.default_inferences_key, label="Lifetime inferences",
    )
    e_recipe = mo.ui.dropdown(
        {
            "Distill → prune → INT8": "distill_prune_quant",
            "Prune → distill → INT8": "prune_distill_quant",
            "Prune → INT8": "prune_quant",
            "INT8 → prune": "quant_prune",
            "INT8 → distill (outcome unavailable)": "quant_distill",
        },
        value="Distill → prune → INT8", label="Ordered recipe",
    )
    e_release = mo.ui.radio(
        {"Release selected recipe": "release", "No change / hold for more evidence": "none"},
        label="Recipe conclusion",
    )
    return (
        a_compute, a_state, a_storage, b_bits, b_calibration, c_index,
        c_sparse, d_student, d_volume, e_recipe, e_release,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Package only": "package", "Package and working set": "memory", "Memory and latency": "memory_latency", "No useful change": "none"},
        label="Which deployment result will improve?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Quality passes with PTQ": "pass", "PTQ fails, but QAT recovers": "qat_recovers", "Quality fails under both PTQ and QAT": "both_fail"},
        label="What will the quality evidence reveal?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Smaller and faster": "both", "Smaller only": "size", "Larger after metadata": "metadata", "Quality fails": "quality"},
        label="What will the sparse artifact deliver?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Worth training at this volume": "worth", "Deployment passes but training does not amortize": "not_amortized", "Student quality fails": "quality", "Deployment constraint fails": "deployment"},
        label="Is this dense student worth training?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Naive multiplication is optimistic": "optimistic", "Naive multiplication matches": "matches", "Ordering changes the final representation": "ordering", "No quality evidence exists": "unknown"},
        label="What will the ordered recipe reveal?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Distill → prune → INT8": "distill_prune_quant", "Prune → distill → INT8": "prune_distill_quant", "Prune → INT8": "prune_quant", "INT8 → prune": "quant_prune", "No change / hold for more evidence": "none"},
        label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {"Distill → prune → INT8": "distill_prune_quant", "Prune → distill → INT8": "prune_distill_quant", "Prune → INT8": "prune_quant", "INT8 → prune": "quant_prune"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"Task quality below floor": "quality", "Target execution path disappears": "support", "Latency exceeds budget": "latency", "Working set exceeds budget": "working_set"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Unobserved population shift": "population", "Runtime-state growth": "state_growth", "Sparse-kernel portability": "kernel_portability", "Distillation data coverage": "distillation_coverage"},
        label="Residual risk",
    )
    rationale = mo.ui.text_area(
        label="Concise rationale",
        placeholder="Cite the chosen result, one rejected result, the remaining limitation, and the trigger.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(get_scenario, get_track_profile, track):
    track_id = track.value
    scenario = get_scenario(track_id)
    profile = get_track_profile(track_id)
    return profile, scenario, track_id


@app.cell
def _(
    a_compute, a_state, a_storage, b_bits, b_calibration, c_index, c_sparse,
    RECIPE_SEQUENCES, SPARSITY_CASES, d_student, d_volume, e_recipe,
    evaluate_distillation, evaluate_precision, evaluate_recipe_id,
    evaluate_resource, evaluate_sparsity_case, track_id,
):
    baseline = evaluate_precision(
        track_id, storage_bits=16, compute_bits=16, runtime_state_bits=16, calibration="none"
    )
    a_result = evaluate_resource(
        track_id, storage_bits=a_storage.value, compute_bits=a_compute.value,
        runtime_state_bits=a_state.value, calibration="ptq",
    )
    b_result = evaluate_precision(
        track_id, storage_bits=b_bits.value, compute_bits=8,
        runtime_state_bits=16, calibration=b_calibration.value,
    )
    b_alternatives = {
        method: evaluate_precision(
            track_id, storage_bits=b_bits.value, compute_bits=8,
            runtime_state_bits=16, calibration=method,
        )
        for method in ("ptq", "qat")
    }
    c_result = evaluate_sparsity_case(
        track_id, c_sparse.value, storage_bits=16,
        index_bits=c_index.value, runtime_state_bits=16,
    )
    c_alternatives = {
        key: evaluate_sparsity_case(
            track_id, key, storage_bits=16,
            index_bits=c_index.value, runtime_state_bits=16,
        )
        for key in SPARSITY_CASES
    }
    d_result = evaluate_distillation(
        track_id, candidate_id=d_student.value,
        deployment_inferences=d_volume.value, runtime_state_bits=16,
    )
    d_alternatives = {
        candidate: evaluate_distillation(
            track_id, candidate_id=candidate,
            deployment_inferences=d_volume.value, runtime_state_bits=16,
        )
        for candidate in ("small_dense", "tiny_dense")
    }
    e_alternatives = {
        key: evaluate_recipe_id(track_id, key, runtime_state_bits=16)
        for key in RECIPE_SEQUENCES
    }
    e_result = e_alternatives[e_recipe.value]
    return (
        a_result, b_alternatives, b_result, baseline, c_alternatives, c_result,
        d_alternatives, d_result, e_alternatives, e_result,
    )


@app.cell
def _(
    MODEL_KEYS, a_compute, a_prediction, a_result, a_state, a_storage,
    b_alternatives, b_bits, b_calibration, b_prediction, b_result, baseline,
    c_alternatives, c_index, c_prediction, c_result, c_sparse, capture_evidence,
    d_alternatives, d_prediction, d_result, d_student, d_volume,
    e_alternatives, e_prediction, e_recipe, e_release, e_result, mo,
    set_evidence, snapshot_result, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    def arm(model_key, result):
        return snapshot_result(model_key, result).to_dict()

    baseline_arm = arm(MODEL_KEYS["precision"], baseline)
    a_arm = arm(MODEL_KEYS["resource"], a_result)
    b_arm = arm(MODEL_KEYS["precision"], b_result)
    c_arm = arm(MODEL_KEYS["sparsity"], c_result)
    d_arm = arm(MODEL_KEYS["distillation"], d_result)
    e_arm = arm(MODEL_KEYS["recipe"], e_result)
    a_upstream = {"storage_bits": a_storage.value, "compute_bits": a_compute.value, "runtime_state_bits": a_state.value}
    b_upstream = {"storage_bits": b_bits.value, "calibration": b_calibration.value}
    c_upstream = {"sparse_case": c_sparse.value, "index_bits": c_index.value}
    d_upstream = {"candidate_id": d_student.value, "deployment_inferences": d_volume.value}
    e_upstream = {"recipe_id": e_recipe.value, "release_conclusion": e_release.value}

    a_capture = mo.ui.button(
        label="Capture resource contrast", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value, inputs=a_upstream,
            baseline=baseline_arm, result=a_arm, upstream_inputs=a_upstream,
            alternatives=(baseline_arm, a_arm), model_key=MODEL_KEYS["resource"],
        )),
    )
    b_capture = mo.ui.button(
        label="Capture precision evidence", kind="success", disabled=b_prediction.value is None,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value, inputs=b_upstream,
            baseline=baseline_arm, result=b_arm, upstream_inputs=b_upstream,
            alternatives=tuple(arm(MODEL_KEYS["precision"], result) for result in b_alternatives.values()),
            model_key=MODEL_KEYS["precision"],
        )),
    )
    c_capture = mo.ui.button(
        label="Capture sparsity evidence", kind="success", disabled=c_prediction.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value, inputs=c_upstream,
            baseline=baseline_arm, result=c_arm, upstream_inputs=c_upstream,
            alternatives=tuple(arm(MODEL_KEYS["sparsity"], result) for result in c_alternatives.values()),
            model_key=MODEL_KEYS["sparsity"],
        )),
    )
    d_capture = mo.ui.button(
        label="Capture student evidence", kind="success", disabled=d_prediction.value is None,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value, inputs=d_upstream,
            baseline=baseline_arm, result=d_arm, upstream_inputs=d_upstream,
            alternatives=tuple(arm(MODEL_KEYS["distillation"], result) for result in d_alternatives.values()),
            decision=d_student.value, model_key=MODEL_KEYS["distillation"],
        )),
    )
    _e_decision = e_recipe.value if e_release.value == "release" else "none"
    e_capture = mo.ui.button(
        label="Capture ordered recipe", kind="success",
        disabled=e_prediction.value is None or e_release.value is None,
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value, inputs=e_upstream,
            baseline=baseline_arm, result=e_arm, upstream_inputs=e_upstream,
            alternatives=tuple(arm(MODEL_KEYS["recipe"], result) for result in e_alternatives.values()),
            decision=_e_decision, model_key=MODEL_KEYS["recipe"],
            chosen_result=baseline_arm if _e_decision == "none" else e_arm,
            result_role="rejected alternative" if _e_decision == "none" else "chosen candidate",
        )),
    )
    return (
        a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream,
        d_capture, d_upstream, e_capture, e_upstream,
    )


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, scenario, track, track_context):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:780px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}
    .lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.table-wrap{max-width:100%;overflow-x:auto}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 10</span><span>ABOUT 50–55 MIN</span></div><h1>Compression That Executes</h1><p>When does a smaller representation improve the deployed system, and when does it only move the bottleneck?</p><div class="pilot-meta"><div><b>Track</b><br>{profile.label}</div><div><b>Task evidence</b><br>{scenario.quality_metric}</div><div><b>Deliverable</b><br>Compression release recipe</div></div></section>""")
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, css, header, track, track_context(profile),
        mo.Html('<p class="pilot-note">All task outcomes and effective execution paths are illustrative scenario evidence. Physical sizes, metadata, state, latency, and ordered composition are calculated in MLSysIM. Open Calculation Notes in each part for assumptions.</p>'),
    ], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_capture, a_compute, a_prediction, a_result, a_state, a_storage,
    a_upstream, apply_plotly_theme, audit_evidence, b_alternatives, b_bits,
    b_calibration, b_capture, b_prediction, b_result, b_upstream, baseline,
    c_alternatives, c_capture, c_index, c_prediction, c_result, c_sparse,
    c_upstream, d_alternatives, d_capture, d_prediction, d_result, d_student,
    d_upstream, d_volume, e_alternatives, e_capture, e_prediction, e_recipe,
    e_release, e_result, e_upstream, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, go, mo, rationale, scenario, track_id,
):
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=_upstream, contrast_required_parts=tuple("ABCDE"),
    )

    _mem_unit = "KiB" if track_id == "tinyml" else "MiB"

    def mem(value):
        return value.to(_mem_unit).magnitude

    def fmt_mem(value):
        return f"{value.to(_mem_unit).magnitude:.1f} {_mem_unit}" if _mem_unit == "KiB" else f"{value.to(_mem_unit).magnitude:.2f} {_mem_unit}"

    def mb(value):
        return value.to("MiB").magnitude

    def ms(value):
        return value.to("ms").magnitude

    def verdict(result):
        return "PASS" if result.feasible else "FAIL · " + ", ".join(result.violations)

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def saved(part):
        _capture = _captures.get(part)
        if _capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Live inputs changed, or the saved arms are identical. Recapture."), kind="danger")
        _data = _capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {_data["prediction"]}<br><small>Track {_data["track"]}; later controls cannot rewrite this evidence.</small></div>')

    def part_a():
        _context = mo.md(f"### A · Which resource must compression save? (8 min)\nThe baseline uses FP16 weights and FP16 runtime state. Package budget: **{fmt_mem(scenario.package_budget)}**; working-set budget: **{fmt_mem(scenario.working_set_budget)}**; latency budget: **{ms(scenario.latency_budget):.1f} ms**.")
        if a_prediction.value is None:
            return mo.vstack([_context, mo.hstack([a_storage, a_compute, a_state], widths="equal", wrap=True), a_prediction])
        _figure = go.Figure()
        for _name, _result in (("FP16 baseline", baseline), ("Changed", a_result)):
            _figure.add_bar(name="Artifact", x=[_name], y=[mem(_result.artifact_size)], marker_color=COLORS["BlueLine"], legendgroup="artifact", showlegend=_name == "FP16 baseline")
            _figure.add_bar(name="Runtime state", x=[_name], y=[mem(_result.runtime_state_size)], marker_color=COLORS["OrangeLine"], legendgroup="state", showlegend=_name == "FP16 baseline")
            _figure.add_bar(name="Application", x=[_name], y=[mem(scenario.application_memory)], marker_color=COLORS["GreenLine"], legendgroup="application", showlegend=_name == "FP16 baseline")
        _figure.update_layout(barmode="stack", height=270, margin=dict(l=20, r=20, t=20, b=20), yaxis_title=f"Peak working set ({_mem_unit})", legend_orientation="h")
        _rows = [
            {"Run": "FP16 baseline", "Artifact": fmt_mem(baseline.artifact_size), "Working set": fmt_mem(baseline.working_set_size), "Latency": f"{ms(baseline.end_to_end_latency):.2f} ms", "Outcome": verdict(baseline)},
            {"Run": "Changed", "Artifact": fmt_mem(a_result.artifact_size), "Working set": fmt_mem(a_result.working_set_size), "Latency": f"{ms(a_result.end_to_end_latency):.2f} ms", "Outcome": verdict(a_result)},
        ]
        if a_state.value < 16:
            _state_note = "Quantizing runtime state reduced activation memory toward the floor, but fixed application memory remains."
        else:
            _state_note = "Smaller stored weights compress the artifact, but leave unquantized runtime state and application memory at their floor."
        return mo.vstack([_context, mo.hstack([a_storage, a_compute, a_state], widths="equal", wrap=True), a_prediction, apply_plotly_theme(_figure), table(_rows), mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. **Observed analytical result:** artifact {fmt_mem(baseline.artifact_size)} → {fmt_mem(a_result.artifact_size)}; working set {fmt_mem(baseline.working_set_size)} → {fmt_mem(a_result.working_set_size)}; latency {ms(baseline.end_to_end_latency):.2f} → {ms(a_result.end_to_end_latency):.2f} ms. {_state_note} Storage precision changes weight bytes ({a_storage.value}-bit), while execution precision ({a_compute.value}-bit) determines compute speedup."), kind="success" if a_result.feasible else "danger"), a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("Artifact bytes include per-group scale and zero-point metadata below 16 bits. Peak working set adds runtime state and application memory. Whole-path latency adds weight movement, executed model work, and fixed application time.")})])

    def part_b():
        _context = mo.md(f"### B · How much precision can the task lose? (10 min)\nThe supplied **{scenario.quality_metric}** baseline is **{scenario.baseline_quality * 100:.2f}%**; the release floor is **{scenario.quality_floor * 100:.2f}%**.")
        if b_prediction.value is None:
            return mo.vstack([_context, mo.hstack([b_bits, b_calibration], widths="equal", wrap=True), b_prediction])
        _rows = [{"Recovery": _method.upper(), "Quality": "unavailable" if _result.task_quality is None else f"{_result.task_quality * 100:.2f}%", "Preparation": f"{_result.preparation_time.to('minute').magnitude:.1f} min", "Latency": f"{ms(_result.end_to_end_latency):.2f} ms", "Outcome": verdict(_result)} for _method, _result in b_alternatives.items()]
        return mo.vstack([_context, mo.hstack([b_bits, b_calibration], widths="equal", wrap=True), b_prediction, table(_rows), mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. **Selected evidence:** {verdict(b_result)}. Quality is {'unavailable for this exact choice' if b_result.task_quality is None else f'{b_result.task_quality * 100:.2f}%'}; preparation takes {b_result.preparation_time.to('minute').magnitude:.1f} minutes. Storage precision changes the artifact; execution precision separately determines whether acceleration exists."), kind="success" if b_result.feasible else "danger"), b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md("Quality values are supplied task outcomes for exact bit-width and recovery-method pairs. PTQ and QAT time are one-time preparation costs. No bit-width-to-quality formula is used.")})])

    def part_c():
        _context = mo.md("### C · When do fewer weights execute faster? (10 min)\nChoose a physical representation before predicting. A mask, indices, or structured layout changes stored bytes and executable work differently.")
        if c_prediction.value is None:
            return mo.vstack([_context, mo.hstack([c_sparse, c_index], widths="equal", wrap=True), c_prediction])
        _rows = [{"Case": _name.replace("_", " "), "Artifact": fmt_mem(_result.artifact_size), "Model time": f"{ms(_result.model_execution_time):.2f} ms", "Fast path": "YES" if _result.execution_supported else "NO", "Outcome": verdict(_result)} for _name, _result in c_alternatives.items()]
        return mo.vstack([_context, mo.hstack([c_sparse, c_index], widths="equal", wrap=True), c_prediction, table(_rows), mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. **Selected evidence:** artifact {fmt_mem(c_result.artifact_size)}, model execution {ms(c_result.model_execution_time):.2f} ms, {verdict(c_result)}. Index metadata can erase storage savings; unsupported zeros still execute as dense work."), kind="success" if c_result.feasible else "danger"), c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md("Masked tensors store every value plus a mask. Indexed layouts store each surviving value, its index, and row pointers. Structured layouts remove aligned work only when the scenario supplies an execution path.")})])

    def part_d():
        _context = mo.md("### D · Is a dense student worth training? (10 min)\nA student pays teacher-generation and training cost once, then deploys as a smaller dense graph. Predict whether the selected lifetime repays that cost.")
        if d_prediction.value is None:
            return mo.vstack([_context, mo.hstack([d_student, d_volume], widths="equal", wrap=True), d_prediction])
        _rows = [{"Student": _name.replace("_", " "), "Quality": f"{_result.deployment.task_quality * 100:.2f}%", "Artifact": fmt_mem(_result.deployment.artifact_size), "Latency": f"{ms(_result.deployment.end_to_end_latency):.2f} ms", "Break-even": f"{_result.break_even_inferences:,}" if _result.break_even_inferences else "never", "Outcome": verdict(_result.deployment)} for _name, _result in d_alternatives.items()]
        return mo.vstack([_context, mo.hstack([d_student, d_volume], widths="equal", wrap=True), d_prediction, table(_rows), mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. **Selected evidence:** {verdict(d_result.deployment)}; one-time training {d_result.training_time.to('hour').magnitude:.1f} h; break-even {d_result.break_even_inferences:,} inferences; training is {'amortized' if d_result.training_amortized else 'not amortized'} at the selected lifetime. A smaller student can still fail its supplied task outcome."), kind="success" if d_result.deployment.feasible and d_result.training_amortized else "danger"), d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md("Break-even divides one-time training duration by saved whole-path time per inference. Student quality, parameter fraction, operation fraction, and state fraction are supplied candidate outcomes rather than reciprocal-size rules.")})])

    def part_e():
        _context = mo.md("### E · Do successful techniques compose? (9 min)\nApply the transformations in order. The instrument recomputes the remaining representation after every stage and compares it with an intentionally naive product of standalone speedups.")
        if e_prediction.value is None:
            return mo.vstack([_context, e_recipe, e_prediction])
        _figure = go.Figure([go.Bar(x=["Recomputed path", "Naive product"], y=[ms(e_result.deployment.end_to_end_latency), ms(e_result.naive_end_to_end_latency)], marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]])])
        _figure.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="End-to-end latency (ms)", showlegend=False)
        _rows = [{"Stage": _stage.transformation.replace("_", " "), "Remaining parameters": f"{_stage.logical_parameters:,}", "Stored bits": _stage.storage_bits, "Artifact": fmt_mem(_stage.artifact_size)} for _stage in e_result.stages]
        return mo.vstack([_context, e_recipe, e_prediction, apply_plotly_theme(_figure), table(_rows), e_release, mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. **Recomputed result:** {ms(e_result.deployment.end_to_end_latency):.2f} ms versus {ms(e_result.naive_end_to_end_latency):.2f} ms from multiplying standalone speedups. Exact-sequence quality is {'available' if e_result.outcome_available else 'unavailable'}; {verdict(e_result.deployment)}. Unknown sequences remain physical comparisons, never inferred quality claims."), kind="success" if e_result.deployment.feasible else "danger"), e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md("Each stage mutates the remaining parameter count, structure, and stored precision. Distillation creates a new dense representation; a later quantization stage is therefore different from quantizing the teacher first. Whole-path time is recomputed once.")})])

    def build_synthesis():
        _rows = []
        for _part in "ABCDE":
            _capture = _captures.get(_part)
            _rows.append({"Part": _part, "Original prediction": _capture.to_dict()["prediction"] if _capture else "—", "Evidence": "CURRENT" if _capture and _part not in audit.stale and (_part, _part) not in audit.identical_pairs else ("STALE" if _capture else "MISSING")})
        _saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
        _complete = audit.complete and all(_widget.value is not None for _widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved_decision
        return mo.vstack([mo.md("### Synthesis · Defend one release decision (5 min)\nUse saved evidence to name the chosen recipe, quantify one tested rejected alternative, state the remaining limitation, and name the condition that forces reevaluation."), table(_rows), mo.callout(mo.md("A chosen recipe must compare with another tested recipe. A hold/no-change conclusion must still name the tested alternative that came closest or failed most instructively."), kind="info"), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale, mo.callout(mo.md("**Ready for the local report.**" if _complete else "Capture five current contrasts, match the recommendation to the saved Part E conclusion, select a different tested alternative, and add the rationale."), kind="success" if _complete else "warn")])

    tabs = mo.ui.tabs({"Part A": part_a(), "Part B": part_b(), "Part C": part_c(), "Part D": part_d(), "Part E": part_e(), "Synthesis": build_synthesis()})
    tabs
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures = get_evidence()
    _saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and all(_widget.value is not None for _widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved_decision
    mo.stop(not _ready)
    _snapshots = {_part: _captures[_part].to_dict() for _part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_10_model_compress.py"), track=track_id,
        scenario=profile.narrative,
        learning_objectives=["Separate stored weights, runtime state, and executable precision", "Evaluate quantization, sparse layouts, and dense students with task evidence", "Compose an ordered compression recipe against deployment constraints"],
        predictions={_part: _snapshots[_part]["prediction"] for _part in "ABCDE"},
        knob_settings={_part: _snapshots[_part]["inputs"] for _part in "ABCDE"},
        evidence_summary={_part: {"baseline": _snapshots[_part]["baseline"], "result": _snapshots[_part]["result"], "alternatives": _snapshots[_part]["alternatives"]} for _part in "ABCDE"},
        binding_constraints={_part: (_snapshots[_part].get("chosen_result") or _snapshots[_part]["result"])["result"].get("violations", (_snapshots[_part].get("chosen_result") or _snapshots[_part]["result"])["result"].get("deployment", {}).get("violations", [])) for _part in "ABCDE"},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["Compression helps when it relieves a binding deployment resource.", "Stored precision and sparse zeros accelerate only on an executable path.", "Ordered recipes require recomputed whole-path evidence and exact task outcomes."],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value}, residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": _snapshots, "recommendation": final_choice.value, "rejected": final_rejected.value, "trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": "Illustrative matched-task outcomes and execution assumptions.", "calculations": "MLSysIM Chapter 10 compression experiments."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    MODEL_ID, audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and all(_widget.value is not None for _widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved_decision
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
        try:
            ledger.save(chapter=10, design={
                "schema_version": 1, "lab_id": "v1_10", "track_id": track_id,
                "model_id": MODEL_ID,
                "evidence": {_part: _capture.to_dict() for _part, _capture in _captures.items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value, "rationale": rationale.value,
            })
            await ledger.flush()
        except Exception:
            _status = "LOCAL SAVE FAILED · DOWNLOAD THE REPORT TO KEEP YOUR EVIDENCE"
        else:
            _status = "SAVED"
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">10 · Compression That Executes · STATUS: </span><span class="hud-active">{_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
