import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 07: The Framework Tax · MLSysBook")


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
    from mlsysim.engine.v1_07_experiments import (
        MODEL_KEY, OPERATION_GRAPH, TRACKS, compare_dispatch,
        evaluate_compilation, evaluate_fusion, evaluate_operator_support,
        evaluate_recomputation,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata,
        report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence

    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, MODEL_KEY, OPERATION_GRAPH,
        TRACKS, apply_plotly_theme, audit_evidence, build_lab_report,
        capture_evidence, compare_dispatch, evaluate_compilation,
        evaluate_fusion, evaluate_operator_support, evaluate_recomputation,
        get_lab_metadata, go, ledger, mo, report_export_panel,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Deployment track",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(TRACKS, track):
    track_id = track.value
    profile = TRACKS[track_id]
    return profile, track_id


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_dispatches = mo.ui.dropdown(
        {"One launch": 1, "Two launches": 2, "Three launches": 3},
        value="One launch", label="Intervention granularity",
    )
    b_executions = mo.ui.slider(10, 200, value=50, step=10, label="Repeated executions")
    b_recompiles_label = (
        "Host rebuilds / redeployments"
        if track_id == "tinyml"
        else "Guard-triggered recompilations"
    )
    b_recompiles = mo.ui.slider(0, 8, value=0, step=1, label=b_recompiles_label)
    c_breaks = mo.ui.slider(0, 4, value=0, step=1, label="Graph breaks")
    d_policy = mo.ui.dropdown(
        {"Retain alternate activations": "alternate", "Retain input only": "input_only"},
        value="Retain alternate activations", label="Recomputation policy",
    )
    d_batch = mo.ui.slider(1, 64, value=8, step=1, label="Training batch")
    e_operator = mo.ui.dropdown(
        {
            "ReLU": "relu", "Layer normalization": "layer_norm",
            "Dynamic slice": "dynamic_slice", "Custom attention": "custom_attention",
            "Host callback": "host_callback",
        },
        value="Host callback", label="Inserted operator",
    )
    e_shape = mo.ui.dropdown(
        {"Static": "static", "Bounded": "bounded", "Dynamic": "dynamic"},
        value="Static", label="Shape behavior",
    )
    e_decision = mo.ui.radio(
        {
            "Use selected path": "selected",
            "Hold native baseline for more evidence": "hold",
            "No feasible target path": "none",
        },
        label="Runtime decision",
    )
    return a_dispatches, b_executions, b_recompiles, c_breaks, d_batch, d_policy, e_decision, e_operator, e_shape


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Below 25%": "below_25", "25–50%": "25_50", "50–75%": "50_75", "Above 75%": "above_75"},
        label="What fraction of fine-grained execution time is dispatch?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Eager remains faster": "eager", "Compilation repays setup": "compiled", "They are equal": "equal"},
        label="Which path has lower total time at these settings?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"No traffic": "none", "Some traffic, no launches": "traffic_only", "Traffic and launches": "both", "Arithmetic operations": "arithmetic"},
        label="What can fusion eliminate from this graph?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Less memory, more step time": "memory_for_time", "Less memory only": "memory_only", "More step time only": "time_only", "Neither changes": "neither"},
        label="What does recomputation change?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Native target path": "native", "Portable fallback path": "fallback", "Cannot execute": "unsupported"},
        label="How will the selected operator and shape execute?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {
            "Eager execution": "eager",
            "Compiled execution": "compiled",
            "Compiled with fusion": "compiled_fused",
            "Hold current path for more evidence": "hold",
            "No feasible target path": "none",
        },
        label="Recommended execution plan",
    )
    final_rejected = mo.ui.radio(
        {"Eager execution": "eager", "Compiled execution": "compiled", "Compiled with fusion": "compiled_fused"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {
            "Execution count changes": "execution_count",
            "Deployment shapes change" if track_id == "tinyml" else "Shape guards change": "shape_guards",
            "Operator set changes": "operator_set",
            "Activation memory changes": "activation_memory",
        },
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Illustrative timing assumptions": "timing_assumptions", "Unmodeled operator interactions": "operator_interactions", "Unmeasured target behavior": "target_measurement"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Decision rationale",
        placeholder="Use saved values to compare the recommendation with the rejected alternative, then name the remaining limitation.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    OPERATION_GRAPH, a_dispatches, b_executions, b_recompiles, c_breaks,
    compare_dispatch, d_batch, d_policy, e_operator, e_shape,
    evaluate_compilation, evaluate_fusion, evaluate_operator_support,
    evaluate_recomputation, track_id,
):
    operation_count = len(OPERATION_GRAPH)
    a_comparison = compare_dispatch(track_id, operation_count, a_dispatches.value)
    b_baseline = evaluate_compilation(track_id, 1, recompilations=0)
    b_result = evaluate_compilation(track_id, b_executions.value, recompilations=b_recompiles.value)
    c_baseline = evaluate_fusion(track_id, graph_breaks=operation_count - 1)
    c_result = evaluate_fusion(track_id, graph_breaks=c_breaks.value)
    d_baseline = evaluate_recomputation(track_id, "retain_all", batch_size=d_batch.value)
    d_result = evaluate_recomputation(track_id, d_policy.value, batch_size=d_batch.value)
    e_baseline = evaluate_operator_support(track_id, "relu", "static")
    e_result = evaluate_operator_support(track_id, e_operator.value, e_shape.value)
    return a_comparison, b_baseline, b_result, c_baseline, c_result, d_baseline, d_result, e_baseline, e_result, operation_count


@app.cell
def _(
    MODEL_KEY, a_comparison, a_dispatches, a_prediction, b_baseline,
    b_executions, b_prediction, b_recompiles, b_result, c_baseline,
    c_breaks, c_prediction, c_result, capture_evidence, d_baseline,
    d_batch, d_policy, d_prediction, d_result, e_baseline, e_decision, e_operator,
    e_prediction, e_result, e_shape, mo, operation_count, set_evidence,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"baseline_dispatches": operation_count, "intervention_dispatches": a_dispatches.value}
    b_upstream = {"executions": b_executions.value, "recompilations": b_recompiles.value}
    c_upstream = {"baseline_breaks": operation_count - 1, "intervention_breaks": c_breaks.value}
    d_upstream = {"batch_size": d_batch.value, "policy": d_policy.value}
    e_upstream = {"operator": e_operator.value, "shape_mode": e_shape.value, "decision": e_decision.value}

    a_capture = mo.ui.button(
        label="Capture dispatch contrast", kind="success",
        disabled=a_prediction.value is None,
        on_click=lambda _value: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs=a_upstream, baseline=a_comparison["baseline"],
            result=a_comparison["intervention"], upstream_inputs=a_upstream,
            alternatives=(a_comparison,), model_key=MODEL_KEY,
        )),
    )
    b_capture = mo.ui.button(
        label="Capture compilation contrast", kind="success",
        disabled=b_prediction.value is None,
        on_click=lambda _value: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs=b_upstream, baseline=b_baseline, result=b_result,
            upstream_inputs=b_upstream, model_key=MODEL_KEY,
        )),
    )
    c_capture = mo.ui.button(
        label="Capture fusion contrast", kind="success",
        disabled=c_prediction.value is None,
        on_click=lambda _value: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs=c_upstream, baseline=c_baseline, result=c_result,
            upstream_inputs=c_upstream, model_key=MODEL_KEY,
        )),
    )
    d_capture = mo.ui.button(
        label="Capture activation contrast", kind="success",
        disabled=d_prediction.value is None,
        on_click=lambda _value: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs=d_upstream, baseline=d_baseline, result=d_result,
            upstream_inputs=d_upstream, model_key=MODEL_KEY,
        )),
    )
    e_same = e_operator.value == "relu" and e_shape.value == "static"
    e_invalid_decision = e_decision.value is None or (e_decision.value == "none" and e_result["executable"])
    e_chosen = e_baseline if e_decision.value == "hold" else e_result
    e_result_role = "rejected alternative" if e_decision.value == "hold" else "tested intervention"
    e_capture = mo.ui.button(
        label="Capture runtime-path contrast", kind="success",
        disabled=e_prediction.value is None or e_same or e_invalid_decision,
        on_click=lambda _value: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs=e_upstream, baseline=e_baseline, result=e_result,
            upstream_inputs=e_upstream, decision=e_decision.value,
            model_key=MODEL_KEY, chosen_result=e_chosen,
            result_role=e_result_role,
        )),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream, e_capture, e_upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = """
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:780px}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    .table-wrap{max-width:100%;overflow-x:auto}@media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>"""
    header = mo.Html(
        f"""{css}<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 07</span><span>ABOUT 50–55 MIN</span></div><h1>The Framework Tax</h1><p>When does an execution path earn its setup and memory costs, and when does target support make the decision for us?</p><div class="pilot-meta"><div><b>Track</b><br>{profile.display}</div><div><b>Workload</b><br>{profile.workload}</div><div><b>Output</b><br>Runtime-path recommendation</div></div></section>"""
    )
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, header, track,
        mo.Html('<p class="pilot-note">Graph shape, tensor scale, compilation work, and runtime support are illustrative scenario assumptions. Hardware limits and runtime overhead anchors come from MLSysIM.</p>'),
    ], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_capture, a_comparison, a_dispatches, a_prediction, a_upstream,
    apply_plotly_theme, audit_evidence, b_baseline, b_capture, b_executions,
    b_prediction, b_recompiles, b_result, b_upstream, c_baseline, c_breaks,
    c_capture, c_prediction, c_result, c_upstream, d_baseline, d_batch,
    d_capture, d_policy, d_prediction, d_result, d_upstream, e_baseline,
    e_capture, e_decision, e_operator, e_prediction, e_result, e_shape, e_upstream,
    final_choice, final_rejected, final_risk, final_trigger, get_evidence,
    go, mo, operation_count, profile, rationale, track_id,
):
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=_upstream,
        contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style(
            {"max-width": "100%", "overflow-x": "auto"}
        )

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(
                mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture this comparison."),
                kind="danger",
            )
        snapshot = capture.to_dict()
        return mo.Html(
            f'<div class="saved"><b>Saved snapshot</b> · original prediction: {snapshot["prediction"]}<br><small>Track {snapshot["track"]}; later controls cannot rewrite this result.</small></div>'
        )

    def part_a():
        intro = mo.md(
            f"### A · Why can small operations waste the machine? (9 min)\nMaya, the performance engineer, keeps arithmetic and payload fixed on **{profile.target.name}** while changing only the number of dispatches."
        )
        if a_prediction.value is None:
            return mo.vstack([intro, a_prediction])
        baseline = a_comparison["baseline"]
        result = a_comparison["intervention"]
        figure = go.Figure()
        figure.add_bar(name="Useful work", x=["Fine", "Coarse"], y=[baseline["useful_time_us"], result["useful_time_us"]], marker_color=COLORS["BlueLine"])
        figure.add_bar(name="Dispatch", x=["Fine", "Coarse"], y=[baseline["dispatch_time_us"], result["dispatch_time_us"]], marker_color=COLORS["OrangeLine"])
        figure.update_layout(barmode="stack", height=270, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Analytical time (µs)", legend_orientation="h")
        rows = [
            {"Path": "Fine", "Dispatches": baseline["dispatches"], "Total": f"{baseline['total_time_us']:.2f} µs", "Dispatch share": f"{baseline['dispatch_fraction']:.1%}"},
            {"Path": "Coarse", "Dispatches": result["dispatches"], "Total": f"{result['total_time_us']:.2f} µs", "Dispatch share": f"{result['dispatch_fraction']:.1%}"},
        ]
        return mo.vstack([
            intro, a_prediction, a_dispatches, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Prediction:** {a_prediction.value}. **Observed analytically:** the fine path spends **{baseline['dispatch_fraction']:.1%}** of total time in dispatch. Coarsening to {result['dispatches']} launch(es) produces **{a_comparison['speedup']:.2f}×** speedup with identical FLOPs and payload."), kind="info"),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md("Useful time is the larger of arithmetic time and fixed input/output movement time. Total time adds one registry-backed dispatch cost per launch. The two runs use identical operations and bytes.")}),
        ])

    def part_b():
        recompile_context = (
            "offline host rebuild and redeployment count (TinyML compiles ahead-of-time on the host, not via on-device JIT guards)"
            if track_id == "tinyml"
            else "guard-triggered recompilation count"
        )
        intro = mo.md(
            f"### B · When does compilation repay setup? (9 min)\n"
            f"Ishan, the compiler engineer, separates repeated execution count from {recompile_context}."
        )
        if b_prediction.value is None:
            return mo.vstack([intro, mo.hstack([b_executions, b_recompiles], widths="equal", wrap=True), b_prediction])
        figure = go.Figure()
        figure.add_bar(name="Eager total", x=["One execution", "Selected repetitions"], y=[b_baseline["eager_total_ms"], b_result["eager_total_ms"]], marker_color=COLORS["BlueLine"])
        figure.add_bar(name="Compiled total", x=["One execution", "Selected repetitions"], y=[b_baseline["compiled_total_ms"], b_result["compiled_total_ms"]], marker_color=COLORS["OrangeLine"])
        figure.update_layout(barmode="group", height=270, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Cumulative analytical time (ms)", legend_orientation="h")
        outcome = "Compilation repays setup" if b_result["compilation_repaid"] else "Eager remains faster"
        recompile_note = (
            "Host rebuilds raise setup without changing the eager baseline."
            if track_id == "tinyml"
            else "Recompilation raises setup without changing the eager baseline."
        )
        return mo.vstack([
            intro, b_prediction, mo.hstack([b_executions, b_recompiles], widths="equal", wrap=True),
            apply_plotly_theme(figure),
            table([{"Executions": b_result["executions"], "Compilations": b_result["compilation_count"], "Break-even": b_result["break_even_executions"], "Eager": f"{b_result['eager_total_ms']:.2f} ms", "Compiled": f"{b_result['compiled_total_ms']:.2f} ms"}]),
            mo.callout(mo.md(f"**Prediction:** {b_prediction.value}. **Observed analytically:** {outcome.lower()} at **{b_result['executions']}** executions and **{b_result['compilation_count']}** compilation(s). {recompile_note}"), kind="success" if b_result["compilation_repaid"] else "warn"),
            b_capture, saved("B"),
            mo.accordion({
                "Calculation Notes": mo.md(
                    "Compiled total = compilation count × setup time + repetitions × fused execution time. "
                    "Eager total = repetitions × unfused execution time. Break-even is the first repetition that recovers all setup work. "
                    "Setup time uses a shared illustrative compiler pass assumption (constant framework dispatch tax across analysis passes, "
                    "not measured on the named host). For TinyML, compilation is ahead-of-time (AOT) host compilation; recompilation represents "
                    "an offline host rebuild and redeployment rather than a dynamic native JIT."
                )
            }),
        ])

    def part_c():
        intro = mo.md(f"### C · What does fusion eliminate? (9 min)\nSofia, the runtime engineer, compares the same {operation_count}-operation graph before and after fusion. Graph breaks split the fused region.")
        if c_prediction.value is None:
            return mo.vstack([intro, c_prediction])
        figure = go.Figure()
        figure.add_bar(name="Traffic", x=["Unfused", "Selected"], y=[c_baseline["total_traffic_mb"], c_result["total_traffic_mb"]], marker_color=COLORS["BlueLine"])
        figure.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Graph traffic (MB)", showlegend=False)
        rows = [
            {"Path": "Unfused", "Launches": c_baseline["launches"], "Breaks": c_baseline["graph_breaks"], "Traffic": f"{c_baseline['total_traffic_mb']:.3f} MB"},
            {"Path": "Selected", "Launches": c_result["launches"], "Breaks": c_result["graph_breaks"], "Traffic": f"{c_result['total_traffic_mb']:.3f} MB"},
        ]
        return mo.vstack([
            intro, c_prediction, c_breaks, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Prediction:** {c_prediction.value}. **Observed analytically:** the selected path removes **{c_result['eliminated_traffic_mb']:.3f} MB** of graph traffic and uses **{c_result['launches']}** launch(es). Arithmetic is preserved; each graph break restores a materialized boundary."), kind="info"),
            c_capture, saved("C"),
            mo.accordion({"Calculation Notes": mo.md("Each unfused operation reads and writes the tensor. A fused region reads once and writes once, so every graph boundary adds another full read/write pair. Fusion changes dispatch and intermediate traffic, not the graph's arithmetic.")}),
        ])

    def part_d():
        intro = mo.md(f"### D · Which activations should remain stored? (9 min)\nNoah, the training engineer, runs this track on **{d_baseline['training_host']}** and compares saved activations against repeated forward work.")
        if d_prediction.value is None:
            return mo.vstack([intro, mo.hstack([d_policy, d_batch], widths="equal", wrap=True), d_prediction])
        figure = go.Figure()
        figure.add_bar(name="Retained activations", x=["Retain all", d_result["policy"]], y=[d_baseline["retained_activation_mb"], d_result["retained_activation_mb"]], marker_color=COLORS["BlueLine"])
        figure.add_bar(name="Step time", x=["Retain all", d_result["policy"]], y=[d_baseline["step_time_ms"], d_result["step_time_ms"]], marker_color=COLORS["OrangeLine"], yaxis="y2")
        figure.update_layout(height=275, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(title="Retained activations (MB)"), yaxis2=dict(title="Analytical step time (ms)", overlaying="y", side="right"), legend_orientation="h")
        feasible = d_result["memory_feasible"]
        return mo.vstack([
            intro, d_prediction, mo.hstack([d_policy, d_batch], widths="equal", wrap=True),
            apply_plotly_theme(figure),
            table([
                {"Policy": "Retain all", "Stored": f"{d_baseline['retained_activation_mb']:.3f} MB", "Repeated ops": d_baseline["recomputed_operation_count"], "Step": f"{d_baseline['step_time_ms']:.3f} ms"},
                {"Policy": d_result["policy"], "Stored": f"{d_result['retained_activation_mb']:.3f} MB", "Repeated ops": d_result["recomputed_operation_count"], "Step": f"{d_result['step_time_ms']:.3f} ms"},
            ]),
            mo.callout(mo.md(f"**Prediction:** {d_prediction.value}. **Observed analytically:** stored activations fall to **{d_result['retained_activation_mb']:.3f} MB**, while **{d_result['recomputed_operation_count']}** operations repeat and step time becomes **{d_result['step_time_ms']:.3f} ms**. Memory status: **{'PASS' if feasible else 'OOM — training infeasible on this host'}**."), kind="success" if feasible else "danger"),
            d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md("Training counts one forward pass and a two-forward-equivalent backward pass. A discarded activation repeats its forward operation during backward. Retained bytes are checked directly against the selected training host's memory capacity.")}),
        ])

    def part_e():
        intro = mo.md("### E · Can the runtime execute the actual graph? (9 min)\nPriya, the deployment engineer, inserts one operator and shape behavior into a known-native baseline. Unsupported work must follow a visible fallback path or fail.")
        if e_prediction.value is None:
            return mo.vstack([intro, mo.hstack([e_operator, e_shape], widths="equal", wrap=True), e_prediction])
        if e_result["status"] == "native":
            status_kind = "success"
            status_interpretation = (
                "Native execution runs directly on the target device without fallback materialization or extra copy penalty."
            )
        elif e_result["status"] == "fallback":
            status_kind = "warn"
            status_interpretation = (
                "A fallback is an executable path, but it fails the native-target requirement and carries explicit copies and execution cost."
            )
        else:
            status_kind = "danger"
            status_interpretation = (
                "The workload cannot execute on the target because the operator or shape behavior is unsupported and no fallback runtime exists."
            )
        total = "No execution" if e_result["total_time_us"] is None else f"{e_result['total_time_us']:.2f} µs"
        return mo.vstack([
            intro, e_prediction, mo.hstack([e_operator, e_shape], widths="equal", wrap=True),
            table([
                {"Path": "Known native", "Operator": e_baseline["operator"], "Shape": e_baseline["shape_mode"], "Status": e_baseline["status"], "Copies": f"{e_baseline['fallback_copy_mb']:.3f} MB"},
                {"Path": "Selected", "Operator": e_result["operator"], "Shape": e_result["shape_mode"], "Status": e_result["status"], "Copies": f"{e_result['fallback_copy_mb']:.3f} MB"},
            ]),
            mo.callout(mo.md(f"**Prediction:** {e_prediction.value}. **Observed analytically:** **{e_result['status'].upper()}**. Target result: **{total}**; fallback materialization: **{e_result['fallback_copy_mb']:.3f} MB**. {status_interpretation}"), kind=status_kind),
            e_decision,
            e_capture, saved("E"),
            mo.accordion({"Calculation Notes": mo.md("Native execution requires both operator and shape support. A fallback materializes one input and one output through target memory, then uses an illustrative portable CPU path capped by the MLSysIM reference CPU rate. TinyML has no fallback runtime in this scenario.")}),
        ])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            rows.append({
                "Part": part,
                "Prediction": capture.to_dict()["prediction"] if capture else "—",
                "Evidence": "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING"),
            })
        e_snapshot = _captures["E"].to_dict() if "E" in _captures else None
        no_path_supported = e_snapshot is not None and e_snapshot["decision"] == "none" and not e_snapshot["result"]["executable"]
        hold_supported = e_snapshot is not None and e_snapshot["decision"] == "hold"
        distinct = final_choice.value is not None and final_rejected.value is not None and final_choice.value != final_rejected.value
        choice_supported = (
            (final_choice.value == "none" and no_path_supported)
            or (final_choice.value == "hold" and hold_supported)
            or final_choice.value in {"eager", "compiled", "compiled_fused"}
        )
        complete = audit.complete and distinct and choice_supported and final_trigger.value is not None and final_risk.value is not None and bool(rationale.value.strip())
        return mo.vstack([
            mo.md("### Synthesis · Defend one execution plan (5 min)\nChoose a tested path, quantify a rejected tested alternative, state the remaining limitation, and name the condition that would reopen the decision."),
            table(rows),
            mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
            mo.hstack([final_trigger, final_risk], widths="equal", wrap=True),
            rationale,
            mo.callout(mo.md("**Ready for the local evidence report.**" if complete else "Complete five current contrasts. Choose two different tested paths; use hold when the baseline remains viable but evidence is insufficient, and select no feasible path only after an unsupported result. Then add a quantified rationale, limitation, and trigger."), kind="success" if complete else "warn"),
        ])

    tabs = mo.ui.tabs({
        "Part A": part_a(), "Part B": part_b(), "Part C": part_c(),
        "Part D": part_d(), "Part E": part_e(), "Synthesis": build_synthesis(),
    })
    tabs
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures = get_evidence()
    _e_snapshot = _captures["E"].to_dict() if "E" in _captures else None
    _no_path_supported = _e_snapshot is not None and _e_snapshot["decision"] == "none" and not _e_snapshot["result"]["executable"]
    _hold_supported = _e_snapshot is not None and _e_snapshot["decision"] == "hold"
    _choice_supported = (
        (final_choice.value == "none" and _no_path_supported)
        or (final_choice.value == "hold" and _hold_supported)
        or final_choice.value in {"eager", "compiled", "compiled_fused"}
    )
    _ready = (
        audit.complete and final_choice.value is not None
        and final_rejected.value is not None
        and final_choice.value != final_rejected.value
        and _choice_supported
        and final_trigger.value is not None and final_risk.value is not None
        and bool(rationale.value.strip())
    )
    mo.stop(not _ready)
    _snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    _chosen_e = _snapshots["E"]["chosen_result"] or _snapshots["E"]["result"]
    report = build_lab_report(
        get_lab_metadata("vol1/lab_07_ml_frameworks.py"),
        track=track_id, scenario=profile.workload,
        learning_objectives=[
            "Quantify dispatch and compilation overhead across repeated execution",
            "Compare graph traffic and activation storage across framework policies",
            "Diagnose native, fallback, and unsupported deployment paths",
        ],
        predictions={part: _snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: _snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": _snapshots[part]["baseline"], "result": _snapshots[part]["result"], "alternatives": _snapshots[part]["alternatives"]} for part in "ABCDE"},
        binding_constraints={"runtime_path": _chosen_e["status"], "activation_memory": "PASS" if _snapshots["D"]["result"]["memory_feasible"] else "OOM"},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=[
            "Fine-grained work can spend more time dispatching than computing.",
            "Compilation and fusion repay costs only when reuse and graph continuity preserve their savings.",
            "Recomputation and fallback move cost between memory, arithmetic, and portability.",
        ],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": _snapshots, "recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": "Illustrative operation graph and support fixture.", "calculations": "MLSysIM v1_07_experiments evaluators."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    import html as _html

    _captures = get_evidence()
    _e_snapshot = _captures["E"].to_dict() if "E" in _captures else None
    _no_path_supported = _e_snapshot is not None and _e_snapshot["decision"] == "none" and not _e_snapshot["result"]["executable"]
    _hold_supported = _e_snapshot is not None and _e_snapshot["decision"] == "hold"
    _choice_supported = (
        (final_choice.value == "none" and _no_path_supported)
        or (final_choice.value == "hold" and _hold_supported)
        or final_choice.value in {"eager", "compiled", "compiled_fused"}
    )
    _ready = (
        audit.complete and final_choice.value is not None
        and final_rejected.value is not None
        and final_choice.value != final_rejected.value
        and _choice_supported
        and final_trigger.value is not None and final_risk.value is not None
        and bool(rationale.value.strip())
    )
    _save_error = None
    _save_succeeded = False
    if _ready:
        try:
            ledger.save(chapter=7, design={
                "schema_version": 1, "lab_id": "v1_07", "track_id": track_id,
                "model_id": "v1_07_experiments",
                "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value, "rationale": rationale.value,
            })
            await ledger.flush()
        except Exception as _exc:
            _save_error = f"{type(_exc).__name__}: {_exc}"
        else:
            _save_succeeded = True
    if _save_succeeded:
        _status = "SAVED"
    elif _save_error is not None:
        _status = f"SAVE FAILED: {_html.escape(_save_error)}"
    else:
        _status = "EVIDENCE IN PROGRESS"
    mo.Html(
        f'<div class="lab-hud"><span>LAB 07 · The Framework Tax · STATUS: {_status}</span></div>'
    )
    return


if __name__ == "__main__":
    app.run()
