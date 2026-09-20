import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 06: Architecture Commitments · MLSysBook")


@app.cell
async def _():
    import sys
    from pathlib import Path
    import marimo as mo

    if sys.platform == "emscripten":
        import micropip

        await micropip.install(
            ["pydantic", "pint", "plotly", "pandas"], keep_going=False
        )
        await micropip.install(
            "../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False
        )
        await micropip.install(
            "../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False
        )
    else:
        labs_dir = Path(__file__).resolve().parents[1]
        if str(labs_dir) not in sys.path:
            sys.path.insert(0, str(labs_dir))
        from bootstrap import native_bootstrap

        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.engine.v1_06_experiments import (
        MODEL_KEY,
        choose_architecture,
        default_scale_candidate,
        evaluate_candidate,
        evaluation_to_dict,
        get_scenario,
        highest_quality_candidate,
        sweep_scale,
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
        MODEL_KEY,
        apply_plotly_theme,
        audit_evidence,
        build_lab_report,
        capture_evidence,
        choose_architecture,
        default_scale_candidate,
        evaluate_candidate,
        evaluation_to_dict,
        get_lab_metadata,
        get_scenario,
        go,
        highest_quality_candidate,
        ledger,
        mo,
        report_export_panel,
        sweep_scale,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML",
        label="Deployment track",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_scenario, track):
    track_id = track.value
    scenario = get_scenario(track_id)
    candidate_labels = {item.label: item.candidate_id for item in scenario.candidates}
    candidate_names = {item.candidate_id: item.label for item in scenario.candidates}
    return candidate_labels, candidate_names, scenario, track_id


@app.cell
def _(
    candidate_labels,
    candidate_names,
    default_scale_candidate,
    mo,
    scenario,
    track_id,
):
    _track_key = track_id
    _examples = {
        f"{p.training_examples:,} examples": p.training_examples
        for p in scenario.candidates[0].quality_curve.observations
    }
    a_examples = mo.ui.dropdown(
        _examples, value=list(_examples)[1], label="Available training examples"
    )
    b_candidate = mo.ui.dropdown(
        candidate_labels,
        value=list(candidate_labels)[0],
        label="Architecture signature",
    )
    b_resource = mo.ui.radio(
        {"Double compute rate": "compute", "Double memory bandwidth": "bandwidth"},
        value="Double compute rate",
        label="System intervention",
    )
    c_candidate = mo.ui.dropdown(
        candidate_labels,
        value=candidate_names[default_scale_candidate(scenario)],
        label="Architecture to scale",
    )
    _scales = {
        f"{value} {scenario.scale_axis} units": value for value in scenario.sweep_values
    }
    c_scale = mo.ui.dropdown(
        _scales, value=list(_scales)[-1], label=f"New {scenario.scale_axis}"
    )
    d_priority = mo.ui.radio(
        {"Task quality": "quality", "Latency": "latency", "Runtime memory": "memory"},
        value="Task quality",
        label="Requirement to prioritize",
    )
    d_choice = mo.ui.radio(
        {
            **candidate_labels,
            "Hold / gather more evidence": "hold",
            "No feasible architecture": "none",
        },
        label="Recommendation",
    )
    d_rejected = mo.ui.dropdown(
        candidate_labels, value=list(candidate_labels)[1], label="Rejected alternative"
    )
    return (
        a_examples,
        b_candidate,
        b_resource,
        c_candidate,
        c_scale,
        d_choice,
        d_priority,
        d_rejected,
    )


@app.cell
def _(candidate_labels, mo, scenario, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        candidate_labels,
        label="Which architecture has the highest quality at this example count?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {
            "Compute time": "compute",
            "Memory traffic": "memory",
            "Serial dependency": "serial dependency",
        },
        label="What will limit execution after the intervention?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {
            "Operations": "operations",
            "Activations": "activations",
            "Resident state": "state",
            "Attention scores": "scores",
            "No requirement": "none",
        },
        label=f"Which commitment grows into the first concern as {scenario.scale_axis} increases?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {
            **candidate_labels,
            "Hold / gather more evidence": "hold",
            "No feasible architecture": "none",
        },
        label="Which architecture will the stated requirement select?",
    ).form(submit_button_label="Lock Part D prediction")
    a_conclusion = mo.ui.radio(
        candidate_labels, label="Conclusion after inspecting the curves"
    )
    return a_conclusion, a_prediction, b_prediction, c_prediction, d_prediction


@app.cell
def _(candidate_labels, mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {
            **candidate_labels,
            "Hold / gather more evidence": "hold",
            "No feasible architecture": "none",
        },
        label="Final recommendation",
    )
    final_rejected = mo.ui.dropdown(
        candidate_labels,
        value=list(candidate_labels)[1],
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {
            "Quality below floor": "quality",
            "Latency above budget": "latency",
            "Runtime memory above budget": "memory",
            "Training-data availability changes": "data",
        },
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {
            "Illustrative quality may not transfer": "quality_fixture",
            "Execution bound omits runtime overhead": "runtime",
            "Input growth exceeds tested sweep": "scale",
        },
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Concise rationale",
        placeholder="Cite saved evidence, quantify the rejected alternative, and explain the trigger.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_examples,
    b_candidate,
    b_resource,
    c_candidate,
    c_scale,
    choose_architecture,
    d_priority,
    evaluate_candidate,
    evaluation_to_dict,
    highest_quality_candidate,
    scenario,
    sweep_scale,
):
    a_records = {
        item.candidate_id: evaluation_to_dict(
            evaluate_candidate(
                scenario, item.candidate_id, training_examples=a_examples.value
            )
        )
        for item in scenario.candidates
    }
    a_quality_winner = highest_quality_candidate(
        scenario, training_examples=a_examples.value
    )
    b_base = evaluation_to_dict(evaluate_candidate(scenario, b_candidate.value))
    b_compute = evaluation_to_dict(
        evaluate_candidate(scenario, b_candidate.value, compute_multiplier=2.0)
    )
    b_bandwidth = evaluation_to_dict(
        evaluate_candidate(scenario, b_candidate.value, bandwidth_multiplier=2.0)
    )
    b_result = b_compute if b_resource.value == "compute" else b_bandwidth
    c_base = evaluation_to_dict(
        evaluate_candidate(
            scenario, c_candidate.value, scale_value=scenario.baseline_scale
        )
    )
    c_result = evaluation_to_dict(
        evaluate_candidate(scenario, c_candidate.value, scale_value=c_scale.value)
    )
    c_sweep = tuple(
        evaluation_to_dict(point.evaluation)
        for point in sweep_scale(scenario, c_candidate.value)
    )
    d_records = {
        item.candidate_id: evaluation_to_dict(
            evaluate_candidate(scenario, item.candidate_id)
        )
        for item in scenario.candidates
    }
    d_model_decision = choose_architecture(scenario, priority=d_priority.value)
    return (
        a_quality_winner,
        a_records,
        b_bandwidth,
        b_base,
        b_compute,
        b_result,
        c_base,
        c_result,
        c_sweep,
        d_model_decision,
        d_records,
    )


@app.cell
def _(
    MODEL_KEY,
    a_conclusion,
    a_examples,
    a_prediction,
    a_records,
    b_bandwidth,
    b_base,
    b_candidate,
    b_compute,
    b_prediction,
    b_resource,
    b_result,
    c_base,
    c_candidate,
    c_prediction,
    c_result,
    c_scale,
    capture_evidence,
    d_choice,
    d_prediction,
    d_priority,
    d_records,
    d_rejected,
    mo,
    scenario,
    set_evidence,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    _rejected_id = d_rejected.value or tuple(d_records)[0]
    a_upstream = {
        "training_examples": a_examples.value,
        "conclusion": a_conclusion.value,
    }
    b_upstream = {
        "candidate_id": b_candidate.value,
        "intervention": b_resource.value,
    }
    c_upstream = {
        "candidate_id": c_candidate.value,
        "scale_value": c_scale.value,
    }
    d_upstream = {
        "priority": d_priority.value,
        "choice": d_choice.value,
        "rejected": _rejected_id,
    }
    _a_ids = tuple(a_records)
    a_capture = mo.ui.button(
        label="Capture sample-efficiency contrast",
        kind="success",
        disabled=a_prediction.value is None or a_conclusion.value is None,
        on_click=lambda _v: store(
            "A",
            capture_evidence(
                track=track_id,
                part="A",
                prediction=a_prediction.value,
                inputs={
                    "training_examples": a_examples.value,
                    "conclusion": a_conclusion.value,
                },
                baseline=a_records[_a_ids[1]],
                result=a_records[_a_ids[0]],
                alternatives=tuple(a_records.values()),
                upstream_inputs=a_upstream,
                decision=a_conclusion.value,
                model_key=MODEL_KEY,
            ),
        ),
    )
    b_capture = mo.ui.button(
        label="Capture execution contrast",
        kind="success",
        disabled=b_prediction.value is None,
        on_click=lambda _v: store(
            "B",
            capture_evidence(
                track=track_id,
                part="B",
                prediction=b_prediction.value,
                inputs={
                    "candidate_id": b_candidate.value,
                    "intervention": b_resource.value,
                },
                baseline=b_base,
                result=b_result,
                alternatives=(b_compute, b_bandwidth),
                upstream_inputs=b_upstream,
                decision=b_result["bottleneck"],
                model_key=MODEL_KEY,
            ),
        ),
    )
    c_capture = mo.ui.button(
        label="Capture scaling contrast",
        kind="success",
        disabled=c_prediction.value is None or c_scale.value == scenario.baseline_scale,
        on_click=lambda _v: store(
            "C",
            capture_evidence(
                track=track_id,
                part="C",
                prediction=c_prediction.value,
                inputs={
                    "candidate_id": c_candidate.value,
                    "scale_value": c_scale.value,
                },
                baseline=c_base,
                result=c_result,
                upstream_inputs=c_upstream,
                decision=c_result["violations"],
                model_key=MODEL_KEY,
            ),
        ),
    )
    _non_candidate_choices = (None, "hold", "none")
    _same = d_choice.value not in _non_candidate_choices and d_choice.value == _rejected_id
    _other = next(key for key in d_records if key != _rejected_id)
    _result_id = _other if d_choice.value in _non_candidate_choices else d_choice.value
    if d_choice.value == "hold":
        _chosen_result = d_records[_rejected_id]
    elif d_choice.value in (None, "none"):
        _chosen_result = None
    else:
        _chosen_result = d_records[_result_id]
    _result_role = (
        "tested alternative"
        if d_choice.value in ("hold", "none")
        else "chosen architecture"
    )
    d_capture = mo.ui.button(
        label="Capture architecture decision",
        kind="success",
        disabled=d_prediction.value is None or d_choice.value is None or _same,
        on_click=lambda _v: store(
            "D",
            capture_evidence(
                track=track_id,
                part="D",
                prediction=d_prediction.value,
                inputs={
                    "priority": d_priority.value,
                    "choice": d_choice.value,
                    "rejected": _rejected_id,
                },
                baseline=d_records[_rejected_id],
                result=d_records[_result_id],
                alternatives=tuple(d_records.values()),
                upstream_inputs=d_upstream,
                decision=d_choice.value,
                model_key=MODEL_KEY,
                chosen_result=_chosen_result,
                result_role=_result_role,
            ),
        ),
    )
    return (
        a_capture,
        a_upstream,
        b_capture,
        b_upstream,
        c_capture,
        c_upstream,
        d_capture,
        d_upstream,
    )


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, scenario, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin:10px 0 14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:780px}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.table-wrap{max-width:100%;overflow-x:auto}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}.lab-hud .hud-failed{color:#fca5a5;font-weight:700}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(
        f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 06</span><span>ABOUT 45–55 MIN</span></div><h1>Architecture Commitments</h1><p>When does a neural architecture match the task, and when does its computation, traffic, dependency chain, or memory shape break the system?</p><div class="pilot-meta"><div><b>Track</b><br>{scenario.label}</div><div><b>Task</b><br>{scenario.task}</div><div><b>Input sweep</b><br>{scenario.scale_axis}</div><div><b>Output</b><br>Architecture recommendation</div></div></section>"""
    )
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, track, header])
    return


@app.cell
def _(
    COLORS,
    a_capture,
    a_conclusion,
    a_examples,
    a_prediction,
    a_quality_winner,
    a_records,
    a_upstream,
    apply_plotly_theme,
    audit_evidence,
    b_bandwidth,
    b_base,
    b_candidate,
    b_capture,
    b_compute,
    b_prediction,
    b_resource,
    b_result,
    b_upstream,
    c_base,
    c_candidate,
    c_capture,
    c_prediction,
    c_result,
    c_scale,
    c_sweep,
    c_upstream,
    candidate_names,
    d_capture,
    d_choice,
    d_model_decision,
    d_prediction,
    d_priority,
    d_records,
    d_rejected,
    d_upstream,
    final_choice,
    final_rejected,
    final_risk,
    final_trigger,
    get_evidence,
    go,
    mo,
    rationale,
    scenario,
    track_id,
):
    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def verdict(record):
        return (
            "FEASIBLE"
            if record["feasible"]
            else "FAILS: " + ", ".join(record["violations"])
        )

    def saved(part):
        return (
            mo.Html(f'<div class="saved">Saved evidence for Part {part}.</div>')
            if part in get_evidence()
            else mo.md("")
        )

    _captures = get_evidence()
    audit = audit_evidence(
        _captures,
        track=track_id,
        required_parts=("A", "B", "C", "D"),
        contrast_required_parts=("A", "B", "C", "D"),
        per_part_upstream_inputs={
            "A": a_upstream,
            "B": b_upstream,
            "C": c_upstream,
            "D": d_upstream,
        },
    )

    def part_a():
        intro = mo.md(
            "### A · When does a structural assumption help? (10 min)\nThe candidates solve the same task and use the same training-example count. Commit to which one will reach the highest supplied validation quality before opening the evidence."
        )
        if a_prediction.value is None:
            return mo.vstack([intro, a_examples, a_prediction])
        _fig = go.Figure()
        for candidate in scenario.candidates:
            _fig.add_trace(
                go.Scatter(
                    x=[
                        point.training_examples
                        for point in candidate.quality_curve.observations
                    ],
                    y=[
                        point.quality_percent
                        for point in candidate.quality_curve.observations
                    ],
                    mode="lines+markers",
                    name=candidate.label,
                )
            )
        _fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Training examples",
            yaxis_title="Illustrative task quality (%)",
        )
        _rows = [
            {
                "Architecture": candidate_names[key],
                "Quality": f"{record['quality_percent']:.1f}%",
                "Evidence": record["quality_evidence_kind"],
            }
            for key, record in a_records.items()
        ]
        return mo.vstack(
            [
                intro,
                a_examples,
                a_prediction,
                apply_plotly_theme(_fig),
                table(_rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {candidate_names.get(a_prediction.value, a_prediction.value)}. **Fixture result:** {candidate_names[a_quality_winner]} is highest at {a_examples.value:,} examples. These curves are scenario assumptions, not benchmark measurements."
                    ),
                    kind="info",
                ),
                a_conclusion,
                a_capture,
                saved("A"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Quality comes only from the matched-task fixture. MLSysIM interpolates between supplied observations. Architecture family and hardware speed do not create a quality bonus."
                        )
                    }
                ),
            ]
        )

    def part_b():
        intro = mo.md(
            "### B · Does more compute capability guarantee a proportional speedup? (10 min)\nHold the task and architecture fixed. Change compute rate or memory bandwidth independently, then predict the remaining execution bound."
        )
        if b_prediction.value is None:
            return mo.vstack([intro, b_candidate, b_resource, b_prediction])
        _rows = [
            {
                "Case": label,
                "Latency": f"{record['latency_ms']:.3f} ms",
                "Parallel bound": f"{record['parallel_time_ms']:.3f} ms",
                "Dependency floor": f"{record['serial_floor_time_ms']:.3f} ms",
                "Bound": record["bottleneck"],
            }
            for label, record in (
                ("Baseline", b_base),
                ("2× compute", b_compute),
                ("2× bandwidth", b_bandwidth),
            )
        ]
        return mo.vstack(
            [
                intro,
                b_candidate,
                b_resource,
                b_prediction,
                table(_rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {b_prediction.value}. **Selected intervention:** {b_result['latency_ms']:.3f} ms, limited by **{b_result['bottleneck']}**. Quality remains {b_result['quality_percent']:.1f}% because resources do not alter the supplied task evidence."
                    ),
                    kind="info",
                ),
                b_capture,
                saved("B"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "MLSysIM takes the larger operation-time and traffic-time bound, then the larger of that result and the dependency floor. Recurrent work is not counted twice."
                        )
                    }
                ),
            ]
        )

    def part_c():
        intro = mo.md(
            f"### C · Which commitment breaks first as {scenario.scale_axis} grows? (10 min)\nKeep the architecture fixed and compare the baseline {scenario.scale_axis} with one changed input."
        )
        if c_prediction.value is None:
            return mo.vstack([intro, c_candidate, c_scale, c_prediction])
        _fig = go.Figure()
        _fig.add_trace(
            go.Scatter(
                x=[point["scale_value"] for point in c_sweep],
                y=[point["activation_memory_mb"] for point in c_sweep],
                mode="lines+markers",
                name="Activations",
                line=dict(color=COLORS["BlueLine"]),
            )
        )
        _fig.add_trace(
            go.Scatter(
                x=[point["scale_value"] for point in c_sweep],
                y=[point["resident_state_memory_mb"] for point in c_sweep],
                mode="lines+markers",
                name="Resident state",
                line=dict(color=COLORS["OrangeLine"]),
            )
        )
        _fig.add_trace(
            go.Scatter(
                x=[point["scale_value"] for point in c_sweep],
                y=[point["attention_score_memory_mb"] for point in c_sweep],
                mode="lines+markers",
                name="Attention scores",
                line=dict(color=COLORS["GreenLine"]),
            )
        )
        _fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title=scenario.scale_axis.title(),
            yaxis_title="Memory (MB)",
        )
        _rows = [
            {
                "Case": label,
                "Scale": record["scale_value"],
                "MACs": f"{record['macs']:,}",
                "Activation": f"{record['activation_memory_mb']:.3f} MB",
                "State": f"{record['resident_state_memory_mb']:.3f} MB",
                "Scores": f"{record['attention_score_memory_mb']:.3f} MB",
                "Outcome": verdict(record),
            }
            for label, record in (("Baseline", c_base), ("Changed input", c_result))
        ]
        return mo.vstack(
            [
                intro,
                c_candidate,
                c_scale,
                c_prediction,
                apply_plotly_theme(_fig),
                table(_rows),
                mo.callout(
                    mo.md(
                        f"**Your prediction:** {c_prediction.value}. The changed input uses **{c_result['total_runtime_memory_mb']:.3f} MB** and **{c_result['latency_ms']:.3f} ms**. {c_result['interpretation']}"
                    ),
                    kind="danger" if not c_result["feasible"] else "success",
                ),
                c_capture,
                saved("C"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "MLSysIM derives each signature from tensor dimensions and changes only the track-compatible axis. Weight sharing keeps convolution parameters fixed; materialized attention scores use the token-pair matrix."
                        )
                    }
                ),
            ]
        )

    def part_d():
        intro = mo.md(
            "### D · Which architecture remains defensible? (10 min)\nEvery candidate must satisfy memory, latency, and matched-task quality requirements. Predict which feasible candidate the selected priority will choose."
        )
        if d_prediction.value is None:
            return mo.vstack([intro, d_priority, d_prediction])
        _rows = [
            {
                "Architecture": candidate_names[key],
                "Quality": f"{record['quality_percent']:.1f}%",
                "Latency": f"{record['latency_ms']:.3f} ms",
                "Runtime memory": f"{record['total_runtime_memory_mb']:.3f} MB",
                "Outcome": verdict(record),
            }
            for key, record in d_records.items()
        ]
        _pick = d_model_decision.selected_id
        _note = (
            "No candidate passes every requirement."
            if _pick is None
            else f"The {d_priority.value} priority selects {candidate_names[_pick]} among feasible candidates."
        )
        return mo.vstack(
            [
                intro,
                d_priority,
                d_prediction,
                table(_rows),
                mo.callout(
                    mo.md(
                        f"**Model result:** {_note} Choose a recommendation and a distinct tested alternative. A no-feasible recommendation still compares two tested candidates."
                    ),
                    kind="info",
                ),
                mo.hstack([d_choice, d_rejected], widths="equal", wrap=True),
                d_capture,
                saved("D"),
                mo.accordion(
                    {
                        "Calculation Notes": mo.md(
                            "Feasibility is a conjunction of named requirements. MLSysIM orders feasible candidates by one observable quantity; it uses no architecture bonus or composite score."
                        )
                    }
                ),
            ]
        )

    def build_synthesis():
        _rows = []
        for part in "ABCD":
            _capture = _captures.get(part)
            _status = (
                "MISSING"
                if not _capture
                else (
                    "STALE"
                    if part in audit.stale or (part, part) in audit.identical_pairs
                    else "CURRENT"
                )
            )
            _rows.append(
                {
                    "Part": part,
                    "Prediction": _capture.to_dict()["prediction"] if _capture else "—",
                    "Evidence": _status,
                }
            )
        _saved = _captures["D"].to_dict()["decision"] if "D" in _captures else None
        _ready = (
            audit.complete
            and all(
                widget.value is not None
                for widget in (final_choice, final_rejected, final_trigger, final_risk)
            )
            and bool(rationale.value.strip())
            and final_choice.value == _saved
            and (
                final_choice.value in ("hold", "none")
                or final_choice.value != final_rejected.value
            )
        )
        return mo.vstack(
            [
                mo.md(
                    "### Synthesis · Defend one architecture (5 min)\nUse saved experiments to state a chosen option, quantify a rejected alternative, name a limitation, and set a reevaluation trigger."
                ),
                table(_rows),
                mo.callout(
                    mo.md(
                        "Saved snapshots remain fixed while live controls move. Recapture stale evidence before generating the report."
                    ),
                    kind="info",
                ),
                mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
                mo.hstack([final_trigger, final_risk], widths="equal", wrap=True),
                rationale,
                mo.callout(
                    mo.md(
                        "**Ready for the local report.**"
                        if _ready
                        else "Complete four current contrasts, match the saved Part D choice, select a tested alternative, and add the rationale."
                    ),
                    kind="success" if _ready else "warn",
                ),
            ]
        )

    mo.ui.tabs(
        {
            "Part A": part_a(),
            "Part B": part_b(),
            "Part C": part_c(),
            "Part D": part_d(),
            "Synthesis": build_synthesis(),
        }
    )
    return (audit,)


@app.cell
def _(
    audit,
    build_lab_report,
    candidate_names,
    final_choice,
    final_rejected,
    final_risk,
    final_trigger,
    get_evidence,
    get_lab_metadata,
    mo,
    rationale,
    report_export_panel,
    scenario,
    track_id,
):
    _captures = get_evidence()
    _saved = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and all(
            widget.value is not None
            for widget in (final_choice, final_rejected, final_trigger, final_risk)
        )
        and bool(rationale.value.strip())
        and final_choice.value == _saved
        and (
            final_choice.value in ("hold", "none")
            or final_choice.value != final_rejected.value
        )
    )
    mo.stop(not _ready)
    _snapshots = {part: _captures[part].to_dict() for part in "ABCD"}
    _choice_label = (
        "Hold / gather more evidence"
        if final_choice.value == "hold"
        else (
            "No feasible architecture"
            if final_choice.value == "none"
            else candidate_names[final_choice.value]
        )
    )
    report = build_lab_report(
        get_lab_metadata("vol1/lab_06_nn_arch.py"),
        track=track_id,
        scenario=scenario.task,
        learning_objectives=[
            "Relate structural assumptions to matched-task sample efficiency",
            "Derive operation, traffic, dependency, and memory signatures",
            "Defend a feasible architecture and reevaluation trigger",
        ],
        predictions={
            part: snapshot["prediction"] for part, snapshot in _snapshots.items()
        },
        knob_settings={
            part: snapshot["inputs"] for part, snapshot in _snapshots.items()
        },
        evidence_summary={
            part: {
                "baseline": snapshot["baseline"],
                "result": snapshot["result"],
                "alternatives": snapshot["alternatives"],
            }
            for part, snapshot in _snapshots.items()
        },
        binding_constraints={
            part: (snapshot.get("chosen_result") or snapshot["result"])["violations"]
            for part, snapshot in _snapshots.items()
        },
        decisions={
            "recommendation": _choice_label,
            "rejected_alternative": candidate_names[final_rejected.value],
            "reevaluation_trigger": final_trigger.value,
        },
        final_decision={
            "recommendation": _choice_label,
            "rejected_alternative": candidate_names[final_rejected.value],
            "rationale": rationale.value,
        },
        big_takeaways=[
            "Task structure can reduce data needs, but quality evidence stays task-specific.",
            "Operation count alone cannot identify a memory or dependency bound.",
            "Input growth exposes different state commitments across architecture families.",
        ],
        reflections={
            "rationale": rationale.value,
            "reevaluation_trigger": final_trigger.value,
        },
        residual_risk=final_risk.value,
        result_snapshot={
            "track": track_id,
            "captures": _snapshots,
            "recommendation": final_choice.value,
            "rejected": final_rejected.value,
            "trigger": final_trigger.value,
            "residual_risk": final_risk.value,
        },
        source_trace={
            "quality": "Illustrative matched-task fixtures; not benchmark measurements.",
            "calculations": "Analytical operation, state, traffic, and critical-path model.",
        },
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    audit,
    final_choice,
    final_rejected,
    final_risk,
    final_trigger,
    get_evidence,
    ledger,
    mo,
    rationale,
    track_id,
):
    _captures = get_evidence()
    _saved = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and all(
            widget.value is not None
            for widget in (final_choice, final_rejected, final_trigger, final_risk)
        )
        and bool(rationale.value.strip())
        and final_choice.value == _saved
        and (
            final_choice.value in ("hold", "none")
            or final_choice.value != final_rejected.value
        )
    )
    _save_status = "EVIDENCE IN PROGRESS"
    _status_class = "hud-active"
    if _ready:
        try:
            ledger.save(
                chapter=6,
                design={
                    "schema_version": 1,
                    "lab_id": "v1_06",
                    "track_id": track_id,
                    "model_id": "v1_06_experiments",
                    "evidence": {
                        part: capture.to_dict()
                        for part, capture in _captures.items()
                    },
                    "recommendation": final_choice.value,
                    "rejected_alternative": final_rejected.value,
                    "reevaluation_trigger": final_trigger.value,
                    "residual_risk": final_risk.value,
                    "rationale": rationale.value,
                },
            )
            await ledger.flush()
            _save_status = "SAVED"
        except Exception:
            _save_status = "SAVE FAILED — REPORT STILL AVAILABLE"
            _status_class = "hud-failed"
    mo.Html(
        f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">06 · Architecture Commitments</span><span aria-hidden="true"> · STATUS: </span><span class="{_status_class}">{_save_status}</span></div>'
    )
    return


if __name__ == "__main__":
    app.run()
