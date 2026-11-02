import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 05: Tensor Costs · MLSysBook")


@app.cell
async def _():
    import sys
    from pathlib import Path
    import marimo as mo

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.3.0-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        labs_dir = Path(__file__).resolve().parents[1]
        if str(labs_dir) not in sys.path:
            sys.path.insert(0, str(labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.labs.experiments.v1_05_experiments import (
        NUMERICAL_CASES,
        TRACKS,
        default_batch,
        evaluate_track,
        growth_comparison,
        repair_comparison,
        run_numerical_case,
        snapshot,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        build_lab_report,
        get_lab_metadata,
        report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence

    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        COLORS,
        LAB_CSS,
        NUMERICAL_CASES,
        TRACKS,
        apply_plotly_theme,
        audit_evidence,
        build_lab_report,
        capture_evidence,
        default_batch,
        evaluate_track,
        get_lab_metadata,
        go,
        growth_comparison,
        ledger,
        mo,
        repair_comparison,
        report_export_panel,
        run_numerical_case,
        snapshot,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence, ledger):
    from mlsysbook_labs.contexts import preferred_context as _preferred_context
    _track_labels = {"tinyml": "TinyML", "mobile": "Mobile", "edge": "Edge", "cloud": "Cloud"}
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value=_track_labels[_preferred_context(ledger)], label="Deployment track", on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(TRACKS, track):
    track_id = track.value
    profile = TRACKS[track_id]
    return profile, track_id


@app.cell
def _(default_batch, mo, track_id):
    _track_key = track_id
    a_target = mo.ui.dropdown(
        {"Input dimension": "input", "Hidden width": "hidden", "Batch": "batch"},
        value="Hidden width", label="Dimension to grow",
    )
    a_factor = mo.ui.dropdown({"2×": 2, "4×": 4}, value="2×", label="Growth factor")
    b_optimizer = mo.ui.dropdown(
        {"SGD": "sgd", "Momentum": "momentum", "Adam": "adam"},
        value="Adam", label="Training optimizer",
    )
    b_precision = mo.ui.dropdown(
        {"FP32": "fp32", "FP16": "fp16", "INT8 storage": "int8"},
        value="FP32", label="Tensor format",
    )
    c_batch = mo.ui.dropdown(
        {"2": 2, "4": 4, "8": 8, "16": 16, "32": 32, "64": 64},
        value=str(default_batch(track_id)),
        label="Batch size",
    )
    d_case = mo.ui.dropdown(
        {"Decimal rounding": "rounding", "Floating-point overflow": "float_overflow", "Integer clipping": "integer_clipping"},
        value="Floating-point overflow", label="Numerical case",
    )
    d_format = mo.ui.dropdown({"FP16": "fp16", "INT8": "int8"}, value="FP16", label="Test format")
    e_choice = mo.ui.radio(
        {"Reduce hidden tensors": "tensor", "Reduce batch": "batch", "Use FP16 within tested limits": "format", "No change / hold for more evidence": "none"},
        label="Decision for the held-out failure",
    )
    e_rejected = mo.ui.radio(
        {"Reduce hidden tensors": "tensor", "Reduce batch": "batch", "Use FP16 within tested limits": "format"},
        label="Quantified rejected alternative",
    )
    return a_factor, a_target, b_optimizer, b_precision, c_batch, d_case, d_format, e_choice, e_rejected


@app.cell
def _(mo):
    nonlinearity_choice = mo.ui.dropdown({"ReLU between layers": "relu", "Affine layers only": "linear"}, value="ReLU between layers", label="Fixed network representation")
    discarded_operand = mo.ui.dropdown({"Keep all forward operands": "keep", "Discard the input": "x", "Discard the preactivation": "z1", "Discard the hidden activation": "h", "Discard the prediction": "prediction"}, value="Keep all forward operands", label="Forward value to discard")
    dependency_input = mo.ui.dropdown({"Active ReLU input": 2.0, "Inactive ReLU input": -1.0}, value="Active ReLU input", label="Chain-rule input case")
    return nonlinearity_choice, discarded_operand, dependency_input


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Parameters grow most": "parameters", "Batch MACs grow most": "macs", "Activation memory grows most": "activations", "They grow by the same factor": "same"},
        label="Which reported quantity changes most proportionally?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Weights": "weights", "Retained activations": "activations", "Parameter gradients": "gradients", "Optimizer state": "optimizer"},
        label="Which added state will dominate training memory?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Throughput rises; batch latency rises": "tradeoff", "Throughput rises; batch latency falls": "both_improve", "Throughput falls; batch latency rises": "both_worsen", "Neither changes": "unchanged", "Batch exceeds memory; no executable result": "infeasible"},
        label="What happens when the batch grows from one?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Same output": "same", "Finite rounded output": "rounded", "Input clipping changes output": "clipped", "Nonfinite overflow": "overflow"},
        label="What numerical behavior will the test format show?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Reduce hidden tensors": "tensor", "Reduce batch": "batch", "Use FP16 within tested limits": "format", "None will fit": "none"},
        label="Which intervention will repair the failed scenario?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Reduce hidden tensors": "tensor", "Reduce batch": "batch", "Use FP16 within tested limits": "format", "No change / hold for more evidence": "none"}, label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {"Reduce hidden tensors": "tensor", "Reduce batch": "batch", "Use FP16 within tested limits": "format"}, label="Rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"Tensor dimensions change": "dimensions", "Batch demand changes": "batch", "Numerical case changes": "numerics", "Memory envelope changes": "memory"}, label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Unmodeled runtime workspace": "workspace", "Format behavior on new values": "numeric_range", "Arrival and queue delay": "queueing"}, label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Concise rationale", placeholder="Use saved quantities to defend the choice and reject one tested alternative.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_factor, a_target, b_optimizer, b_precision, c_batch, d_case, d_format,
    evaluate_track, growth_comparison, repair_comparison, run_numerical_case,
    snapshot, track_id,
):
    a_comparison = growth_comparison(track_id, a_target.value, a_factor.value)
    b_inference = evaluate_track(track_id, precision=b_precision.value, phase="inference")
    b_training = evaluate_track(track_id, precision=b_precision.value, phase="training", optimizer=b_optimizer.value)
    c_baseline = evaluate_track(track_id, batch_size=1)
    c_result = evaluate_track(track_id, batch_size=c_batch.value)
    d_baseline = run_numerical_case(d_case.value, "fp32")
    d_result = run_numerical_case(d_case.value, d_format.value)
    e_results = {action: repair_comparison(track_id, action) for action in ("tensor", "batch", "format")}
    serial = snapshot
    return a_comparison, b_inference, b_training, c_baseline, c_result, d_baseline, d_result, e_results, serial


@app.cell
def _(
    a_comparison, a_factor, a_prediction, a_target, b_inference, b_optimizer,
    b_precision, b_prediction, b_training, c_baseline, c_batch, c_prediction,
    c_result, capture_evidence, d_baseline, d_case, d_format, d_prediction,
    d_result, e_choice, e_prediction, e_rejected, e_results, mo, serial,
    set_evidence, track_id, nonlinearity_choice, discarded_operand, dependency_input,
):
    from mlsysim.labs.experiments.v1_05_experiments import run_dependency_trace as _saved_dependency, run_nonlinearity_case as _saved_nonlinearity
    _nonlinear_ref = _saved_nonlinearity(nonlinearity_choice.value)
    _dependency_ref = _saved_dependency(discarded=() if discarded_operand.value == "keep" else (discarded_operand.value,), x=dependency_input.value)

    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"target": a_target.value, "factor": a_factor.value, "reference_activation": nonlinearity_choice.value}
    b_upstream = {"precision": b_precision.value, "optimizer": b_optimizer.value, "reference_discarded": discarded_operand.value, "reference_input": dependency_input.value}
    c_upstream = {"baseline_batch": 1, "result_batch": c_batch.value}
    d_upstream = {"case": d_case.value, "format": d_format.value}
    e_upstream = {"choice": e_choice.value, "rejected": e_rejected.value}

    a_capture = mo.ui.button(
        label="Capture tensor-growth contrast", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _value: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value, inputs=a_upstream,
            baseline=serial(a_comparison["baseline"]), result={**serial(a_comparison["result"]), "supplemental_reference": _nonlinear_ref},
            upstream_inputs=a_upstream, model_key="v1_05.evaluate_track",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture phase contrast", kind="success", disabled=b_prediction.value is None,
        on_click=lambda _value: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value, inputs=b_upstream,
            baseline=serial(b_inference), result={**serial(b_training), "supplemental_reference": _dependency_ref},
            upstream_inputs=b_upstream, model_key="v1_05.evaluate_track",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture batching tradeoff", kind="success", disabled=c_prediction.value is None or c_batch.value == 1,
        on_click=lambda _value: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value, inputs=c_upstream,
            baseline=serial(c_baseline), result=serial(c_result),
            upstream_inputs=c_upstream, model_key="v1_05.evaluate_track",
        )),
    )
    d_capture = mo.ui.button(
        label="Capture numerical contrast", kind="success", disabled=d_prediction.value is None,
        on_click=lambda _value: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value, inputs=d_upstream,
            baseline=serial(d_baseline), result=serial(d_result),
            upstream_inputs=d_upstream, model_key="v1_05.run_numerical_case",
        )),
    )
    e_invalid = e_prediction.value is None or e_choice.value is None or e_rejected.value is None or e_choice.value == e_rejected.value
    if e_choice.value in e_results:
        e_selected = e_results[e_choice.value]["result"]
        e_chosen = e_selected
        e_result_role = "chosen intervention"
    elif e_rejected.value in e_results:
        e_selected = e_results[e_rejected.value]["result"]
        e_chosen = e_results["tensor"]["baseline"]
        e_result_role = "rejected alternative"
    else:
        e_selected = e_results["batch"]["result"]
        e_chosen = e_results["tensor"]["baseline"]
        e_result_role = "rejected alternative"
    e_capture = mo.ui.button(
        label="Capture repair decision", kind="success", disabled=e_invalid,
        on_click=lambda _value: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value, inputs=e_upstream,
            baseline=serial(e_results["tensor"]["baseline"]), result=serial(e_selected),
            alternatives=tuple(serial(e_results[action]["result"]) for action in e_results),
            decision=e_choice.value, upstream_inputs=e_upstream, model_key="v1_05.evaluate_track",
            chosen_result=serial(e_chosen), result_role=e_result_role,
        )),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream, e_capture, e_upstream


@app.cell
def _(get_lab_metadata, mo, track, track_id, profile):
    from mlsysbook_labs.notebook import notebook_scenario as _scenario
    from mlsysim import Scenarios as _Scenarios
    from mlsysbook_labs.learning_paths import get_learning_path as _get_learning_path
    from mlsysbook_labs.notebook import (
        notebook_bridge as _bridge, notebook_header as _header,
        notebook_learning_map as _learning_map, notebook_shell as _opening_shell,
    )

    learning_contract = _get_learning_path("vol1", 5)
    _journey = _Scenarios.DeploymentJourneys.get(track_id)
    _episode = _journey.episode(5)
    learning_contract["narrative"] = {
        "project": _journey.mission,
        "scene": _episode.scene,
        "carry_forward": _episode.handoff,
        "scope": _journey.boundary,
    }
    _opening_shell([
        _header(get_lab_metadata("vol1/lab_05_nn_compute.py"), _episode.question,
                topics=[part["label"] for part in learning_contract["parts"]]),
        track,
        _scenario(learning_contract, track=track_id, scene=_episode.scene),
        _learning_map(learning_contract),
        mo.md(f"**Reference used in today’s instruments:** {profile['scenario']}. This controlled case tests the chapter’s mechanism; transfer the reasoning to your project without treating the result as project validation."),
        mo.callout(mo.md("These are supplied, controlled reference studies. Their results do not validate the deployment project. Choose a prediction in Part A to open its live experiment; a wrong prediction still opens the evidence."), kind="info"),
    ])
    return (learning_contract,)


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, NUMERICAL_CASES, a_capture, a_comparison, a_factor, a_prediction,
    a_target, a_upstream, apply_plotly_theme, audit_evidence, b_capture,
    b_inference, b_optimizer, b_precision, b_prediction, b_training, b_upstream,
    c_baseline, c_batch, c_capture, c_prediction, c_result, c_upstream,
    d_baseline, d_capture, d_case, d_format, d_prediction, d_result, d_upstream,
    e_capture, e_choice, e_prediction, e_rejected, e_results, e_upstream,
    final_choice, final_rejected, final_risk, final_trigger, get_evidence, go,
    mo, profile, rationale, track_id,
    nonlinearity_choice, discarded_operand, dependency_input,
    learning_contract,
):
    from mlsysbook_labs.notebook import notebook_shell as _shell, notebook_learning_part as _learning_part, notebook_tab_hint as _tab_hint
    from mlsysim.labs.experiments.v1_05_experiments import run_dependency_trace as _dependency, run_nonlinearity_case as _nonlinearity

    def _nonlinearity_view():
        _result = _nonlinearity(nonlinearity_choice.value)
        return mo.vstack([
            mo.md("#### What does the activation add?\nThis exact mathematical reference uses the same fixed weights for both choices. Change the activation, inspect each input, and explain whether stacking affine layers preserved the required mapping."),
            nonlinearity_choice,
            table([{"Input": str(row["input"]), "Hidden values": str(row["hidden"]), "Output": row["output"], "Required output": row["target"], "Matches": row["matches"]} for row in _result["rows"]]),
            mo.md(_result["scope_note"]),
        ])

    def _dependency_view():
        _discarded = () if discarded_operand.value == "keep" else (discarded_operand.value,)
        _result = _dependency(discarded=_discarded, x=dependency_input.value)
        return mo.vstack([
            mo.md("#### Why retain forward values?\nCompare keeping and discarding one operand. Then try the inactive input. Which derivative is unavailable, and which is actually zero? No recomputation is allowed in this reference."),
            mo.hstack([discarded_operand, dependency_input], wrap=True),
            table([{"Derivative": row["derivative"], "Needs": ", ".join(row["required_operands"]), "Value": "Unavailable" if row["value"] is None else row["value"], "Missing": ", ".join(row["missing_operands"]) or "None"} for row in _result["backward_trace"]]),
            mo.md(_result["scope_note"]),
        ])

    _route = {part["id"]: part for part in learning_contract["parts"]}


    captures = get_evidence()
    upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def memory_kib(result, key="total_state_memory"):
        return result[key].to("KiB").magnitude

    def milliseconds(result, key):
        return result[key].to("millisecond").magnitude

    def per_second(result):
        return result["throughput"].to("1 / second").magnitude

    def table(rows):
        from mlsysbook_labs.notebook import notebook_table as _compact_table
        return _compact_table(rows)

    def saved(part):
        capture = captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** A dependency changed, or the saved runs are identical. Recapture this part."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later control changes do not rewrite this evidence.</small></div>')

    def part_a():
        intro = mo.md("Choose one dimension and predict the largest proportional change before opening the operation ledger.")
        if a_prediction.value is None:
            return mo.vstack([intro, mo.hstack([a_target, a_factor], wrap=True), a_prediction])
        base, changed = a_comparison["baseline"], a_comparison["result"]
        layer_rows = [{
            "Layer": index, "Shape before": f"{left['input_dim']} → {left['output_dim']}",
            "Shape after": f"{right['input_dim']} → {right['output_dim']}",
            "Parameters after": f"{right['parameters']:,}", "Batch MACs after": f"{right['batch_macs']:,}",
        } for index, (left, right) in enumerate(zip(base["layers"], changed["layers"]), start=1)]
        summary = [
            {"Run": "Baseline", "Parameters": f"{base['parameters']:,}", "Batch MACs": f"{base['batch_macs']:,}", "Activation KiB": f"{memory_kib(base, 'activation_memory'):.2f}", "Traffic KiB": f"{memory_kib(base, 'minimum_forward_traffic'):.2f}"},
            {"Run": "Changed", "Parameters": f"{changed['parameters']:,}", "Batch MACs": f"{changed['batch_macs']:,}", "Activation KiB": f"{memory_kib(changed, 'activation_memory'):.2f}", "Traffic KiB": f"{memory_kib(changed, 'minimum_forward_traffic'):.2f}"},
        ]
        return mo.vstack([
            intro, mo.hstack([a_target, a_factor], wrap=True), a_prediction, table(layer_rows), table(summary),
            mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. The intervention changed only **{a_target.value}**; parameter count, batch work, activation state, and traffic follow their tensor dimensions."), kind="info"),
            a_capture, saved("A"), _nonlinearity_view(), mo.accordion({"Calculation Notes": mo.md("Each dense layer counts its weight matrix and bias vector. One weight application is one MAC; the ledger uses two FLOPs per MAC and reports bias additions separately. Minimum traffic reads each layer input, weights, and biases once and writes its output once.")}),
        ])

    def part_b():
        intro = mo.md("Choose a tensor format and optimizer. Predict which added state dominates before comparing phases.")
        if b_prediction.value is None:
            return mo.vstack([intro, mo.hstack([b_precision, b_optimizer], wrap=True), b_prediction])
        rows = []
        for label, result in (("Inference", b_inference), ("Training", b_training)):
            rows.append({
                "Phase": label, "Weights KiB": f"{memory_kib(result, 'weight_memory'):.1f}",
                "Activations KiB": f"{memory_kib(result, 'activation_memory'):.1f}",
                "Gradients KiB": f"{memory_kib(result, 'gradient_memory'):.1f}",
                "Optimizer KiB": f"{memory_kib(result, 'optimizer_memory'):.1f}",
                "Total KiB": f"{memory_kib(result):.1f}", "Fits": "PASS" if result["fits_memory"] else "FAIL · memory",
            })
        host_context = " On device tracks (TinyML and Mobile), exceeding the memory envelope illustrates why models are trained on a development host and only inference runs on-device." if track_id in ("tinyml", "mobile") else ""
        return mo.vstack([
            intro, mo.hstack([b_precision, b_optimizer], wrap=True), b_prediction, table(rows),
            mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. Inference uses {memory_kib(b_inference):.1f} KiB; training uses {memory_kib(b_training):.1f} KiB and **{'fits' if b_training['fits_memory'] else 'crosses'}** the scenario memory envelope.{host_context}"), kind="success" if b_training["fits_memory"] else "danger"),
            b_capture, saved("B"), _dependency_view(), mo.accordion({"Calculation Notes": mo.md(f"Inference uses an exact two-buffer activation schedule. Training retains the batch input and each layer output, adds parameter gradients, and adds zero, one, or two optimizer slots per parameter. For microcontroller and mobile tracks, optimizer state and retained activations can exceed embedded memory, clarifying why training is hosted on a development workstation while only the inference runtime fits on-device. The selected tensors use **{b_training['bytes_per_element']} byte(s) per element**; optimizer slots use **{b_training['optimizer_bytes_per_element']} bytes per element**. The FP16 case keeps gradients in FP16 and two Adam slots in FP32; it does **not** include a separate FP32 master copy of the weights. Transient activation-gradient scratch and runtime workspaces are outside this analytical state inventory.")}),
        ])

    def part_c():
        intro = mo.md(f"Compare a fixed batch-one baseline with a larger batch under the same network and execution assumptions (track-native batch is {profile['batch_size']}).")
        if c_prediction.value is None:
            return mo.vstack([intro, c_batch, c_prediction])
        figure = go.Figure()
        _results = (c_baseline, c_result)
        figure.add_bar(
            name="Modeled batch latency (ms)",
            x=[f"Batch {result['batch_size']}" for result in _results],
            y=[milliseconds(result, "batch_latency") if result["fits_memory"] else None for result in _results],
            marker_color=COLORS["OrangeLine"],
        )
        figure.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Modeled batch latency (ms)", legend_orientation="h")
        rows = [{
            "Batch": result["batch_size"],
            "Modeled batch ms": f"{milliseconds(result, 'batch_latency'):.4f}" if result["fits_memory"] else "Unavailable · memory failure",
            "Modeled samples/s": f"{per_second(result):,.0f}" if result["fits_memory"] else "Unavailable · memory failure",
            "Minimum traffic/sample KiB": f"{memory_kib(result, 'minimum_forward_traffic_per_sample'):.2f}",
            "Activation KiB": f"{memory_kib(result, 'activation_memory'):.2f}",
            "Fits": "PASS" if result["fits_memory"] else "FAIL · memory",
        } for result in _results]
        if all(result["fits_memory"] for result in _results):
            feedback = f"Modeled throughput changes from {per_second(c_baseline):,.0f} to {per_second(c_result):,.0f} samples/s, while modeled batch latency changes from {milliseconds(c_baseline, 'batch_latency'):.4f} to {milliseconds(c_result, 'batch_latency'):.4f} ms. These are analytical estimates, not measurements."
        else:
            feedback = f"Batch {c_result['batch_size']} exceeds the {profile['display']} memory envelope ({memory_kib(c_result):.1f} KiB required; {memory_kib(c_result, 'memory_limit'):.1f} KiB available). It cannot execute under these assumptions, so it has no modeled service latency or throughput and cannot win a performance comparison. The minimum tensor-traffic count remains informative; it does not establish an executable design."
        return mo.vstack([
            intro, c_batch, c_prediction, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. {feedback}"), kind="warn" if not c_result["fits_memory"] else "info"),
            c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md(f"For memory-feasible batches, sequential analytical time adds batch FLOPs divided by an illustrative effective compute rate of **{c_baseline['effective_compute_rate']:~P}**, counted tensor bytes divided by **{c_baseline['effective_bandwidth']:~P}**, and **{c_baseline['launch_overhead']:~P}** launch overhead per layer. This compares batch one against the selected batch; the track-native batch is {profile['batch_size']}. Service metrics are unavailable when state exceeds memory. A streaming or memory-expansion repair would require its own traffic and timing model. Weights and launch overhead are amortized across a feasible batch; work and activations scale with batch. The model excludes overlap, arrivals, queue delay, and latency percentiles.")}),
        ])

    def part_d():
        intro = mo.md("Choose a curated numerical case and predict the observed low-precision behavior.")
        if d_prediction.value is None:
            return mo.vstack([intro, mo.hstack([d_case, d_format], wrap=True), d_prediction])
        rows = []
        for label, result in (("FP32 reference path", d_baseline), (d_format.value.upper(), d_result)):
            rows.append({
                "Run": label, "Inputs": str(result["represented_inputs"]), "Products": str(result["products"]),
                "Output": str(result["output"]), "Rounded": "YES" if result["representation_rounded"] else "NO",
                "Clipped": "YES" if result["input_clipped"] else "NO", "Overflow": "YES" if result["arithmetic_overflow"] else "NO",
            })
        flags = []
        if d_result["input_clipped"]:
            flags.append("input clipping")
        if d_result["arithmetic_overflow"]:
            flags.append("arithmetic overflow")
        if d_result["representation_rounded"]:
            flags.append("representation rounding")
        observed = ", ".join(flags) if flags else "no flagged change"
        return mo.vstack([
            intro, mo.hstack([d_case, d_format], wrap=True), d_prediction,
            mo.md(f"**Case assumption:** {NUMERICAL_CASES[d_case.value]['description']}"), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. **Observed analytical execution:** {observed}; reference output {d_result['reference_output']}, test-format output {d_result['output']}."), kind="danger" if d_result["output_changed"] else "success"),
            d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md("FP32 and FP16 execute the displayed arrays in their stated formats. INT8 uses a unit scale, round-to-nearest conversion, saturation to the signed INT8 range, and an INT32 accumulator. These deterministic cases establish behavior only for the tested values; they do not validate learned-model accuracy, training convergence, or untested numerical ranges.")}),
        ])

    def part_e():
        intro = mo.md(f"The held-out scenario is a fixed-batch baseline (FP32 Adam training at batch eight, held constant across tracks rather than track-native batch {profile['batch_size']}). Predict a repair, then compare all three actions from that same failed baseline.")
        if e_prediction.value is None:
            return mo.vstack([intro, e_prediction])
        baseline = e_results["tensor"]["baseline"]
        rows = [{"Design": "Held-out baseline", "Parameters": f"{baseline['parameters']:,}", "Batch": baseline["batch_size"], "Format": baseline["precision"].upper(), "State KiB": f"{memory_kib(baseline):.1f}", "Outcome": "PASS" if baseline["fits_memory"] else "FAIL · memory"}]
        labels = {"tensor": "Reduce hidden tensors", "batch": "Reduce batch", "format": "FP16 within tested limits"}
        for action, comparison in e_results.items():
            result = comparison["result"]
            rows.append({"Design": labels[action], "Parameters": f"{result['parameters']:,}", "Batch": result["batch_size"], "Format": result["precision"].upper(), "State KiB": f"{memory_kib(result):.1f}", "Outcome": "PASS" if result["fits_memory"] else "FAIL · memory"})
        valid = e_choice.value is not None and e_rejected.value is not None and e_choice.value != e_rejected.value
        selected = e_results[e_choice.value]["result"] if e_choice.value in e_results else baseline
        rejected = e_results[e_rejected.value]["result"] if e_rejected.value in e_results else None
        active_rows = [{"Role": "Selected", "Design": labels.get(e_choice.value, "Hold failed baseline" if e_choice.value == "none" else "No decision yet"),
                        "State KiB": f"{memory_kib(selected):.1f}",
                        "Memory fit": "PASS" if selected["fits_memory"] else "FAIL · memory"}]
        if rejected is not None:
            active_rows.append({"Role": "Rejected", "Design": labels[e_rejected.value],
                                "State KiB": f"{memory_kib(rejected):.1f}",
                                "Memory fit": "PASS" if rejected["fits_memory"] else "FAIL · memory"})
        return mo.vstack([
            intro, e_prediction, table(rows),
            mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. Choose a decision and a different, quantified rejected alternative. A no-change decision still requires one tested alternative as evidence. On constrained device tracks, all tested single repairs may fail the memory envelope; a failed repair is a valid outcome showing why training remains on a development host."), kind="info"),
            mo.hstack([e_choice, e_rejected], widths="equal", wrap=True),
            table(active_rows),
            mo.callout(mo.md(
                ("The selected repair fits the memory envelope under the stated assumptions. " if selected["fits_memory"]
                 else "The selected design still fails the memory envelope; it does not establish a repaired training run. A combined change, development host, or explicit hold remains necessary. ")
                + ("The distinct comparison is ready to capture." if valid else "Select a decision and a different rejected alternative.")
            ), kind="success" if valid and selected["fits_memory"] else "warn"),
            e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md(f"All repairs start from the identical failed training state (fixed batch eight). The tensor action halves hidden widths, the batch action reduces batch eight to one, and the format action uses FP16 weights and gradients with two FP32 Adam slots and no separate FP32 master-weight copy. A repair that still exceeds the memory envelope remains a failed repair; combined changes or a development host may be needed. Passing the resource envelope establishes memory fit under those assumptions; it does not validate training accuracy, convergence, or numerical behavior beyond the tested cases. The comparison reports consequences rather than an aggregate score.")}),
        ])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = captures.get(part)
            rows.append({"Part": part, "Original prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING")})
        saved_decision = captures["E"].to_dict()["decision"] if "E" in captures else None
        ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == saved_decision
        return mo.vstack([
            mo.md("### Synthesis · Defend one tensor decision (5 min)\nUse saved evidence to name the chosen intervention, quantify a rejected tested alternative, state the remaining limitation, and identify the trigger that would reopen the decision."),
            table(rows), mo.callout(mo.md("Saved snapshots remain fixed when live controls move. Recapture stale evidence before generating the report."), kind="info"),
            mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale,
            mo.callout(mo.md("**Ready for the local report.**" if ready else "Complete five current contrasts, match the recommendation to saved Part E, choose a different rejected alternative, and add the rationale."), kind="success" if ready else "warn"),
        ])

    _shell([_tab_hint(), mo.ui.tabs({"Part A": _learning_part(learning_contract, "A", part_a()), "Part B": _learning_part(learning_contract, "B", part_b()), "Part C": _learning_part(learning_contract, "C", part_c()), "Part D": _learning_part(learning_contract, "D", part_d()), "Part E": _learning_part(learning_contract, "E", part_e()), "Synthesis": build_synthesis()})])
    return (audit,)


@app.cell(hide_code=True)
def _():
    from mlsysbook_labs.learning_paths import get_learning_path as _submission_path
    from mlsysbook_labs.notebook import notebook_shell as _submission_shell, notebook_submission as _submission
    _submission_shell([_submission(_submission_path(1, 5))])
    return


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures = get_evidence()
    _saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved_decision
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not _ready, _download_readiness(volume=1))
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_05_nn_compute.py"), track=track_id, scenario=profile["scenario"],
        learning_objectives=["Count dense-network parameters, MACs, FLOPs, activations, and tensor traffic", "Explain why training state can cross a memory boundary that inference does not", "Compare batching and numerical-format interventions from preserved baselines"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "chosen_result": snapshots[part].get("chosen_result"), "result_role": snapshots[part].get("result_role"), "alternatives": snapshots[part]["alternatives"]} for part in "ABCDE"},
        binding_constraints={part: (snapshots[part].get("chosen_result") or snapshots[part]["result"]).get("violations", []) for part in "ABCE"},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["Tensor dimensions determine both work and retained state.", "Batching amortizes fixed traffic and launch overhead while increasing batch latency and activations.", "A smaller numerical format requires value-range evidence, not a quality proxy."],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value}, residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": snapshots, "recommendation": final_choice.value, "rejected": final_rejected.value, "trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": "Illustrative analytical scenarios, not device measurements.", "calculations": "MLSysIM v1_05_experiments evaluator."},
    )
    from mlsysbook_labs.notebook import notebook_shell as _report_shell
    _report_shell([report_export_panel(report)])
    return (report,)




@app.cell
def _(mo, get_evidence, final_choice, final_rejected, final_risk, final_trigger, rationale, audit):
    from mlsysbook_labs.notebook import notebook_shell as _save_shell
    _draft_identity = (audit.to_dict(), get_evidence(), final_choice.value, final_rejected.value, final_risk.value, final_trigger.value, rationale.value)
    save_discussion = mo.ui.run_button(label="Save discussion")
    _save_shell([
        save_discussion,
        mo.md("Save your completed discussion locally. Editing a decision, rationale, or saved comparison requires saving again. Download remains available independently of local storage."),
    ])
    return (save_discussion,)


@app.cell
async def _(
    audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
    save_discussion,
):
    _captures = get_evidence()
    _saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved_decision
    _status = "DRAFT READY · SAVE DISCUSSION TO RECORD" if _ready else "EVIDENCE IN PROGRESS"
    if _ready and save_discussion.value:
        try:
            ledger.save(chapter=5, design={
                "schema_version": 1, "lab_id": "v1_05", "track_id": track_id,
                "model_id": "v1_05_experiments", "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "recommendation": final_choice.value, "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value, "residual_risk": final_risk.value,
                "rationale": rationale.value,
            })
            await ledger.flush()
        except Exception:
            _status = "LOCAL SAVE FAILED · DOWNLOAD THE REPORT TO KEEP YOUR EVIDENCE"
        else:
            _status = "SAVED"
    from mlsysbook_labs.notebook import notebook_shell as _status_shell, notebook_status as _status_view
    _status_shell([_status_view("Lab 05", _status)])
    return


if __name__ == "__main__":
    app.run()
