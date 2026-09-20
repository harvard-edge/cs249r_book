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
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        labs_dir = Path(__file__).resolve().parents[1]
        if str(labs_dir) not in sys.path:
            sys.path.insert(0, str(labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.engine.v1_05_experiments import (
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
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Deployment track", on_change=lambda _value: set_evidence({}),
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
        {"Throughput rises; batch latency rises": "tradeoff", "Throughput rises; batch latency falls": "both_improve", "Throughput falls; batch latency rises": "both_worsen", "Neither changes": "unchanged"},
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
    set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"target": a_target.value, "factor": a_factor.value}
    b_upstream = {"precision": b_precision.value, "optimizer": b_optimizer.value}
    c_upstream = {"baseline_batch": 1, "result_batch": c_batch.value}
    d_upstream = {"case": d_case.value, "format": d_format.value}
    e_upstream = {"choice": e_choice.value, "rejected": e_rejected.value}

    a_capture = mo.ui.button(
        label="Capture tensor-growth contrast", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _value: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value, inputs=a_upstream,
            baseline=serial(a_comparison["baseline"]), result=serial(a_comparison["result"]),
            upstream_inputs=a_upstream, model_key="v1_05.evaluate_track",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture phase contrast", kind="success", disabled=b_prediction.value is None,
        on_click=lambda _value: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value, inputs=b_upstream,
            baseline=serial(b_inference), result=serial(b_training),
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
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#174e63);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:780px}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .table-wrap{max-width:100%;overflow-x:auto}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 05</span><span>ABOUT 50 MIN</span></div><h1>Tensor Costs</h1><p>Which tensor choice causes the workload to cross a compute, memory, or numerical boundary?</p><div class="pilot-meta"><div><b>Track</b><br>{profile['display']}</div><div><b>Context</b><br>{profile['scenario']}</div><div><b>Output</b><br>Evidence-backed tensor decision</div></div></section>""")
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, css, header, track,
        mo.Html('<p class="pilot-note">The workloads and execution rates are illustrative analytical scenarios, not device measurements. Each Calculation Notes panel states the counted tensors and exclusions.</p>'),
    ], gap=0.5)
    return


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
):
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
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def saved(part):
        capture = captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** A dependency changed, or the saved runs are identical. Recapture this part."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later control changes do not rewrite this evidence.</small></div>')

    def part_a():
        intro = mo.md("### A · What becomes expensive when a tensor dimension grows? (9 min)\nChoose one dimension and predict the largest proportional change before opening the operation ledger.")
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
            a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("Each dense layer counts its weight matrix and bias vector. One weight application is one MAC; the ledger uses two FLOPs per MAC and reports bias additions separately. Minimum traffic reads each layer input, weights, and biases once and writes its output once.")}),
        ])

    def part_b():
        intro = mo.md("### B · Why can inference fit while training fails? (9 min)\nChoose a tensor format and optimizer. Predict which added state dominates before comparing phases.")
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
            b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md(f"Inference uses an exact two-buffer activation schedule. Training retains the batch input and each layer output, adds parameter gradients, and adds zero, one, or two optimizer slots per parameter. For microcontroller and mobile tracks, optimizer state and retained activations can exceed embedded memory, clarifying why training is hosted on a development workstation while only the inference runtime fits on-device. The selected tensors use **{b_training['bytes_per_element']} byte(s) per element**; optimizer slots use **{b_training['optimizer_bytes_per_element']} bytes per element**. The FP16 case keeps gradients in FP16 and two Adam slots in FP32; it does **not** include a separate FP32 master copy of the weights. Transient activation-gradient scratch and runtime workspaces are outside this analytical state inventory.")}),
        ])

    def part_c():
        intro = mo.md(f"### C · Does batching make every request faster? (10 min)\nCompare a fixed batch-one baseline with a larger batch under the same network and execution assumptions (track-native batch is {profile['batch_size']}).")
        if c_prediction.value is None:
            return mo.vstack([intro, c_batch, c_prediction])
        figure = go.Figure()
        figure.add_bar(name="Batch latency (ms)", x=["Batch 1", f"Batch {c_batch.value}"], y=[milliseconds(c_baseline, "batch_latency"), milliseconds(c_result, "batch_latency")], marker_color=COLORS["OrangeLine"])
        figure.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Batch latency (ms)", legend_orientation="h")
        rows = [
            {"Batch": c_baseline["batch_size"], "Batch latency ms": f"{milliseconds(c_baseline, 'batch_latency'):.4f}", "Samples/s": f"{per_second(c_baseline):,.0f}", "Traffic/sample KiB": f"{memory_kib(c_baseline, 'minimum_forward_traffic_per_sample'):.2f}", "Activation KiB": f"{memory_kib(c_baseline, 'activation_memory'):.2f}", "Fits": "PASS" if c_baseline["fits_memory"] else "FAIL · memory"},
            {"Batch": c_result["batch_size"], "Batch latency ms": f"{milliseconds(c_result, 'batch_latency'):.4f}", "Samples/s": f"{per_second(c_result):,.0f}", "Traffic/sample KiB": f"{memory_kib(c_result, 'minimum_forward_traffic_per_sample'):.2f}", "Activation KiB": f"{memory_kib(c_result, 'activation_memory'):.2f}", "Fits": "PASS" if c_result["fits_memory"] else "FAIL · memory"},
        ]
        mem_status = f" Warning: Batch {c_result['batch_size']} exceeds the {profile['display']} memory limit ({memory_kib(c_result):.1f} KiB vs {memory_kib(c_result, 'memory_limit'):.1f} KiB limit)." if not c_result["fits_memory"] else ""
        return mo.vstack([
            intro, c_batch, c_prediction, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. Throughput changes from {per_second(c_baseline):,.0f} to {per_second(c_result):,.0f} samples/s, while the complete batch takes {milliseconds(c_baseline, 'batch_latency'):.4f} to {milliseconds(c_result, 'batch_latency'):.4f} ms.{mem_status}"), kind="warn" if not c_result["fits_memory"] else "info"),
            c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md(f"The sequential analytical time adds batch FLOPs divided by an illustrative effective compute rate of **{c_baseline['effective_compute_rate']:~P}**, counted tensor bytes divided by **{c_baseline['effective_bandwidth']:~P}**, and **{c_baseline['launch_overhead']:~P}** launch overhead per layer. The experiment compares a fixed batch-one baseline against the chosen batch to evaluate latency and throughput amortization, alongside the track-native operating batch ({profile['batch_size']}). If activation state exceeds available memory, the batch is physically infeasible without streaming or memory expansion. Weights and launch overhead are amortized across the batch; work and activations scale with batch. The model excludes overlap, arrivals, queue delay, and latency percentiles.")}),
        ])

    def part_d():
        intro = mo.md("### D · Can fewer bytes produce an invalid answer? (9 min)\nChoose a curated numerical case and predict the observed low-precision behavior.")
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
        intro = mo.md(f"### E · Which intervention fixes the observed failure? (8 min)\nThe held-out scenario is a fixed-batch baseline (FP32 Adam training at batch eight, held constant across tracks rather than track-native batch {profile['batch_size']}). Predict a repair, then compare all three actions from that same failed baseline.")
        if e_prediction.value is None:
            return mo.vstack([intro, e_prediction])
        baseline = e_results["tensor"]["baseline"]
        rows = [{"Design": "Held-out baseline", "Parameters": f"{baseline['parameters']:,}", "Batch": baseline["batch_size"], "Format": baseline["precision"].upper(), "State KiB": f"{memory_kib(baseline):.1f}", "Outcome": "PASS" if baseline["fits_memory"] else "FAIL · memory"}]
        labels = {"tensor": "Reduce hidden tensors", "batch": "Reduce batch", "format": "FP16 within tested limits"}
        for action, comparison in e_results.items():
            result = comparison["result"]
            rows.append({"Design": labels[action], "Parameters": f"{result['parameters']:,}", "Batch": result["batch_size"], "Format": result["precision"].upper(), "State KiB": f"{memory_kib(result):.1f}", "Outcome": "PASS" if result["fits_memory"] else "FAIL · memory"})
        valid = e_choice.value is not None and e_rejected.value is not None and e_choice.value != e_rejected.value
        return mo.vstack([
            intro, e_prediction, table(rows),
            mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. Choose a decision and a different, quantified rejected alternative. A no-change decision still requires one tested alternative as evidence. On constrained device tracks, all tested single repairs may fail the memory envelope; a failed repair is a valid outcome showing why training remains on a development host."), kind="info"),
            mo.hstack([e_choice, e_rejected], widths="equal", wrap=True),
            mo.callout(mo.md("Decision contrast is complete." if valid else "Select a decision and a different rejected alternative."), kind="success" if valid else "warn"),
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

    mo.ui.tabs({"Part A": part_a(), "Part B": part_b(), "Part C": part_c(), "Part D": part_d(), "Part E": part_e(), "Synthesis": build_synthesis()})
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures = get_evidence()
    _saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved_decision
    mo.stop(not _ready)
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
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved_decision
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
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
    mo.Html(f'<div class="lab-hud"><span>LAB 05 · Tensor Costs · STATUS: {_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
