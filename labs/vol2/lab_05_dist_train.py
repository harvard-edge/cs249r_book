import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 05: Scaling Without Illusions · MLSysBook")


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
    from mlsysim import Models, Systems
    from mlsysim.engine.v2_05_experiments import (
        compare_straggler_policy,
        get_capacity_dimensions,
        get_parallel_layout_specs,
        get_pipeline_dimensions,
        get_quality_fixtures,
        get_track_scenario,
        memory_plan_record,
        parallel_layout,
        parallel_layout_record,
        pipeline_schedule,
        pipeline_schedule_record,
        quality_policy_record,
        scaling_point_record,
        scaling_sweep,
        training_memory_plan,
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

    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, Models, Systems,
        apply_plotly_theme, audit_evidence, build_lab_report, capture_evidence,
        compare_straggler_policy, get_capacity_dimensions, get_lab_metadata,
        get_parallel_layout_specs, get_pipeline_dimensions,
        get_quality_fixtures, get_track_scenario, go, ledger,
        memory_plan_record, mo, parallel_layout, parallel_layout_record,
        pipeline_schedule, pipeline_schedule_record, quality_policy_record,
        report_export_panel, scaling_point_record, scaling_sweep, training_memory_plan,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Training context",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_track_scenario, track):
    track_id = track.value
    scenario = get_track_scenario(track_id)
    return scenario, track_id


@app.cell
def _(mo, scenario, track_id):
    _track_key = track_id
    _worker_options = {f"{count} workers": count for count in scenario.worker_counts[1:]}
    a_mode = mo.ui.dropdown(
        {"Fixed global batch": "strong", "Fixed local batch": "weak"},
        value="Fixed global batch", label="Scaling experiment",
    )
    a_workers = mo.ui.dropdown(
        _worker_options, value=next(iter(_worker_options)), label="Larger worker pool",
    )
    a_prediction = mo.ui.radio(
        {"Below half of ideal": "below-half", "Half to three quarters": "middle", "Above three quarters": "above-three-quarters"},
        label="How close will the larger pool come to ideal scaling?",
    ).form(submit_button_label="Lock Part A prediction")
    b_zero_stage = mo.ui.dropdown(
        {"ZeRO-1": 1, "ZeRO-2": 2, "ZeRO-3": 3}, value="ZeRO-1", label="State-sharding stage",
    )
    b_checkpointing = mo.ui.dropdown(
        {"Selective recomputation": "selective", "Full recomputation": "full"},
        value="Full recomputation", label="Activation policy",
    )
    b_prediction = mo.ui.radio(
        {"Still out of memory": "oom", "Fits; optimizer state falls most": "optimizer", "Fits; weights fall most": "weights", "Fits; activations fall most": "activations"},
        label="What will the selected intervention change?",
    ).form(submit_button_label="Lock Part B prediction")
    b_decision = mo.ui.radio(
        {"Carry the sharded plan": "shard", "No feasible plan": "none"}, label="Capacity decision",
    )
    c_alternative = mo.ui.dropdown(
        {"TP=16 across nodes": "cross", "TP=8 with two pipeline stages": "pipeline"},
        value="TP=16 across nodes", label="Alternative layout",
    )
    c_prediction = mo.ui.radio(
        {"Within-node TP is fastest": "within", "Cross-node TP is fastest": "cross", "Pipeline layout is fastest": "pipeline"},
        label="Which equal-device layout will have the shortest step?",
    ).form(submit_button_label="Lock Part C prediction")
    c_decision = mo.ui.radio(
        {"Keep TP within each node": "within", "Use selected alternative": "alternative"}, label="Placement decision",
    )
    d_microbatches = mo.ui.dropdown(
        {"4 microbatches": 4, "8 microbatches": 8, "16 microbatches": 16},
        value="8 microbatches", label="Pipeline microbatch count",
    )
    d_prediction = mo.ui.radio(
        {"Bubble falls and memory fits": "faster-fits", "Bubble falls but activations exceed HBM": "faster-oom", "Bubble and memory both fall": "both-fall"},
        label="What happens with more fixed-size microbatches?",
    ).form(submit_button_label="Lock Part D prediction")
    d_decision = mo.ui.radio(
        {"Keep 2 microbatches": "baseline", "Use selected count": "selected"}, label="Pipeline decision",
    )
    e_prediction = mo.ui.radio(
        {"Strict synchronization": "strict", "Drop slow workers": "drop", "They tie": "tie"},
        label="Which policy reaches the same target first?",
    ).form(submit_button_label="Lock Part E prediction")
    e_decision = mo.ui.radio(
        {"Strict synchronization": "strict", "Drop slow workers": "drop"}, label="Straggler policy",
    )
    return (
        a_mode, a_prediction, a_workers, b_checkpointing, b_decision,
        b_prediction, b_zero_stage, c_alternative, c_decision, c_prediction,
        d_decision, d_microbatches, d_prediction, e_decision, e_prediction,
    )


@app.cell
def _(mo, scenario, track_id):
    _track_key = track_id
    _choice_options = (
        {
            "Data parallel only (sufficient for compact workload)": "data-parallel",
            "Fit-first: shard, local TP, bounded pipeline (model-parallel)": "fit-first",
            "Step-first: cross nodes, deep pipeline, drop stragglers": "step-first",
        }
        if not scenario.model_parallel_applicable
        else {
            "Fit-first: shard, local TP, bounded pipeline": "fit-first",
            "Step-first: cross nodes, deep pipeline, drop stragglers": "step-first",
            "Data parallel only (unsupported for 8B model)": "data-parallel",
        }
    )
    _rejected_options = (
        {
            "Fit-first model-parallel plan": "fit-first",
            "Step-first model-parallel plan": "step-first",
            "Data-parallel-only plan": "data-parallel",
        }
        if not scenario.model_parallel_applicable
        else {"Data-parallel-only plan (fails to fit)": "data-parallel", "Fit-first plan": "fit-first", "Step-first plan": "step-first"}
    )
    final_choice = mo.ui.radio(_choice_options, label="Recommended training plan")
    final_rejected = mo.ui.radio(_rejected_options, label="Rejected alternative")
    final_trigger = mo.ui.radio(
        {"Memory exceeds HBM": "memory", "Inter-node communication exceeds local compute": "communication", "Time to target exceeds strict synchronization": "quality-time"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Illustrative convergence may not transfer": "convergence", "Analytical links omit runtime contention": "contention", "Capacity case differs from track model": "transfer"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Quantified rationale",
        placeholder="Cite a saved result, quantify the rejected alternative, and state when you would reevaluate.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    Models, a_mode, a_workers, b_checkpointing, b_zero_stage, c_alternative,
    compare_straggler_policy, d_microbatches, get_capacity_dimensions,
    get_parallel_layout_specs, get_pipeline_dimensions, get_quality_fixtures,
    memory_plan_record, parallel_layout, parallel_layout_record,
    pipeline_schedule, pipeline_schedule_record, quality_policy_record,
    scaling_point_record, scaling_sweep, scenario, track_id, training_memory_plan,
):
    base_workers = scenario.worker_counts[0]
    selected_workers = int(a_workers.value)
    strong_points = scaling_sweep(
        scenario, (base_workers, selected_workers), mode="strong",
        global_batch=scenario.global_batch, local_batch=scenario.local_batch,
    )
    weak_points = scaling_sweep(
        scenario, (base_workers, selected_workers), mode="weak",
        global_batch=scenario.global_batch, local_batch=scenario.local_batch,
    )
    a_points = strong_points if a_mode.value == "strong" else weak_points
    a_args = {
        "scenario": track_id, "worker_counts": [base_workers, selected_workers],
        "mode": a_mode.value, "global_batch": scenario.global_batch,
        "local_batch": scenario.local_batch, "precision": "fp16", "efficiency": 0.5,
    }
    a_base_record = scaling_point_record(a_points[0], inputs={**a_args, "selected_workers": base_workers})
    a_result_record = scaling_point_record(a_points[1], inputs={**a_args, "selected_workers": selected_workers})

    capacity_model = Models.Language.Llama3_70B
    capacity_fleet = scenario.fleet
    b_dims = get_capacity_dimensions(capacity_fleet)
    b_common = {
        "model": capacity_model.name, "fleet": capacity_fleet.name,
        "seq_len": 1024, "microbatch_count": 8, "gradient_accumulation_steps": 1,
        "precision": "fp16", "activation_checkpointing": b_checkpointing.value,
        **b_dims,
    }
    b_base_args = {**b_common, "zero_stage": 0}
    b_result_args = {**b_common, "zero_stage": int(b_zero_stage.value)}
    b_base_plan = training_memory_plan(
        capacity_model, capacity_fleet, seq_len=1024,
        zero_stage=0, microbatch_count=8, precision="fp16",
        activation_checkpointing=b_checkpointing.value,
        **b_dims,
    )
    b_result_plan = training_memory_plan(
        capacity_model, capacity_fleet, seq_len=1024,
        zero_stage=int(b_zero_stage.value), microbatch_count=8, precision="fp16",
        activation_checkpointing=b_checkpointing.value,
        **b_dims,
    )
    b_base_record = memory_plan_record(b_base_plan, inputs=b_base_args)
    b_result_record = memory_plan_record(b_result_plan, inputs=b_result_args)

    layout_model = Models.Language.Llama3_8B
    layout_fleet = scenario.fleet
    c_specs = get_parallel_layout_specs(layout_fleet)
    c_common = {
        "model": layout_model.name, "fleet": layout_fleet.name,
        "global_batch": 256, "seq_len": 1024, "microbatch_count": 8,
        "zero_stage": 3, "precision": "fp16", "efficiency": 0.5,
        "activation_checkpointing": "selective",
    }
    c_layouts = {}
    c_records = {}
    for _key, _spec in c_specs.items():
        _layout = parallel_layout(
            layout_model, layout_fleet, global_batch=256, seq_len=1024,
            microbatch_count=8, zero_stage=3, precision="fp16", efficiency=0.5,
            activation_checkpointing="selective", **_spec,
        )
        c_layouts[_key] = _layout
        c_records[_key] = parallel_layout_record(_layout, inputs={**c_common, **_spec})
    c_selected_key = c_alternative.value

    pipeline_model = Models.Language.Llama3_8B
    pipeline_fleet = scenario.fleet
    d_dims = get_pipeline_dimensions(pipeline_fleet)
    d_common = {
        "model": pipeline_model.name, "fleet": pipeline_fleet.name,
        "microbatch_size": 4, "seq_len": 2048, "zero_stage": 0,
        "precision": "fp16", "activation_checkpointing": "none", **d_dims,
    }
    d_base_args = {**d_common, "microbatch_count": 2}
    d_result_args = {**d_common, "microbatch_count": int(d_microbatches.value)}
    d_base_schedule = pipeline_schedule(
        pipeline_model, pipeline_fleet, microbatch_size=4, seq_len=2048,
        microbatch_count=2, zero_stage=0, precision="fp16", activation_checkpointing="none",
        **d_dims,
    )
    d_result_schedule = pipeline_schedule(
        pipeline_model, pipeline_fleet, microbatch_size=4, seq_len=2048,
        microbatch_count=int(d_microbatches.value), zero_stage=0, precision="fp16", activation_checkpointing="none",
        **d_dims,
    )
    d_base_record = pipeline_schedule_record(d_base_schedule, inputs=d_base_args)
    d_result_record = pipeline_schedule_record(d_result_schedule, inputs=d_result_args)

    convergence_fixture, straggler_fixture = get_quality_fixtures(track_id)
    quality_comparison = compare_straggler_policy(
        step_time=a_points[1].step_time, global_batch=a_points[1].global_batch,
        fixture=convergence_fixture, policy=straggler_fixture,
    )
    _fixture_record = {
        "name": convergence_fixture.name,
        "minimum_steps": convergence_fixture.minimum_steps,
        "critical_batch": convergence_fixture.critical_batch,
    }
    e_base_args = {
        "name": quality_comparison.baseline.name,
        "step_time": {"value": float(a_points[1].step_time.to("ms").magnitude), "unit": "ms"},
        "global_batch": a_points[1].global_batch, "fixture": _fixture_record,
        "work_multiplier": 1.0,
    }
    e_result_args = {
        "name": quality_comparison.intervention.name,
        "step_time": {"value": float(quality_comparison.intervention.step_time.to("ms").magnitude), "unit": "ms"},
        "global_batch": a_points[1].global_batch, "fixture": _fixture_record,
        "work_multiplier": straggler_fixture.work_multiplier,
    }
    e_base_record = quality_policy_record(quality_comparison.baseline, inputs=e_base_args)
    e_result_record = quality_policy_record(quality_comparison.intervention, inputs=e_result_args)
    return (
        a_base_record, a_result_record, b_base_record, b_result_record,
        c_layouts, c_records, c_selected_key, convergence_fixture,
        d_base_record, d_result_record, e_base_record, e_result_record,
        quality_comparison, straggler_fixture, strong_points, weak_points,
    )


@app.cell
def _(
    a_base_record, a_mode, a_prediction, a_result_record, a_workers,
    b_base_record, b_checkpointing, b_decision, b_prediction, b_result_record,
    b_zero_stage, c_alternative, c_decision, c_prediction, c_records,
    c_selected_key, capture_evidence, d_base_record, d_decision,
    d_microbatches, d_prediction, d_result_record, e_base_record, e_decision,
    e_prediction, e_result_record, mo, scenario, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"scenario": track_id}
    b_upstream = {"scenario": track_id, "fleet": scenario.fleet.name, "checkpointing": b_checkpointing.value}
    c_upstream = {"scenario": track_id, "fleet": scenario.fleet.name}
    d_upstream = {"scenario": track_id, "fleet": scenario.fleet.name}
    e_upstream = {"scenario": track_id, "scaling_mode": a_mode.value, "workers": int(a_workers.value)}

    a_capture = mo.ui.button(
        label="Capture scaling contrast", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"mode": a_mode.value, "workers": int(a_workers.value)},
            baseline=a_base_record, result=a_result_record,
            upstream_inputs=a_upstream, decision=a_mode.value,
            model_key="v2_05_experiments.scaling_sweep",
        )),
    )
    b_keep = b_decision.value == "none"
    b_capture = mo.ui.button(
        label="Capture capacity contrast", kind="success",
        disabled=b_prediction.value is None or b_decision.value is None,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"zero_stage": int(b_zero_stage.value), "decision": b_decision.value},
            baseline=b_base_record, result=b_result_record,
            chosen_result=b_base_record if b_keep else b_result_record,
            result_role="rejected alternative" if b_keep else "chosen intervention",
            upstream_inputs=b_upstream, alternatives=(b_base_record, b_result_record),
            decision=b_decision.value,
            model_key="v2_05_experiments.training_memory_plan",
        )),
    )
    c_keep = c_decision.value == "within"
    c_capture = mo.ui.button(
        label="Capture placement contrast", kind="success",
        disabled=c_prediction.value is None or c_decision.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"alternative": c_alternative.value, "decision": c_decision.value},
            baseline=c_records["within"], result=c_records[c_selected_key],
            chosen_result=c_records["within"] if c_keep else c_records[c_selected_key],
            result_role="rejected alternative" if c_keep else "chosen intervention",
            upstream_inputs=c_upstream, alternatives=tuple(c_records.values()),
            decision=c_decision.value,
            model_key="v2_05_experiments.parallel_layout",
        )),
    )
    d_keep = d_decision.value == "baseline"
    d_capture = mo.ui.button(
        label="Capture pipeline contrast", kind="success",
        disabled=d_prediction.value is None or d_decision.value is None,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={"microbatch_count": int(d_microbatches.value), "decision": d_decision.value},
            baseline=d_base_record, result=d_result_record,
            chosen_result=d_base_record if d_keep else d_result_record,
            result_role="rejected alternative" if d_keep else "chosen intervention",
            upstream_inputs=d_upstream, alternatives=(d_base_record, d_result_record),
            decision=d_decision.value,
            model_key="v2_05_experiments.pipeline_schedule",
        )),
    )
    e_keep = e_decision.value == "strict"
    e_capture = mo.ui.button(
        label="Capture quality-time contrast", kind="success",
        disabled=e_prediction.value is None or e_decision.value is None,
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"policy": e_decision.value}, baseline=e_base_record,
            result=e_result_record, chosen_result=e_base_record if e_keep else e_result_record,
            result_role="rejected alternative" if e_keep else "chosen intervention",
            upstream_inputs=e_upstream, alternatives=(e_base_record, e_result_record),
            decision=e_decision.value,
            model_key="v2_05_experiments.compare_straggler_policy",
        )),
    )
    return (
        a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream,
        d_capture, d_upstream, e_capture, e_upstream,
    )


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, scenario, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#17486e);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin:8px 0 14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.6rem);line-height:1.05;margin:14px 0 8px}.pilot-head p{color:#dbeafe;max-width:800px;line-height:1.5}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:9px;margin-top:16px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.55;margin:2px 2px 12px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME II · LAB 05</span><span>ABOUT 50–55 MIN</span></div><h1>Scaling Without Illusions</h1><p>When do more accelerators stop buying useful progress, and which parallel plan survives memory, network, pipeline, and convergence costs?</p><div class="pilot-meta"><div><b>Deployment fleet</b><br>{scenario.deployment_target.title()}</div><div><b>Training cluster</b><br>{scenario.fleet.name}</div><div><b>Output</b><br>Auditable training recommendation</div></div></section>""")
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, css, track, header,
        mo.Html(f'<p class="pilot-note"><b>Track workload:</b> {scenario.model.name}. {scenario.applicability_note} Parts B–D evaluate sharding, placement, and pipeline mechanisms on the upstream training cluster ({scenario.fleet.name}); model parallelism is an unsupported path on compact deployment devices.</p>'),
    ], gap=0.5).style({"padding-top": "32px"})
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base_record, a_capture, a_mode, a_prediction, a_result_record,
    a_upstream, a_workers, apply_plotly_theme, audit_evidence, b_base_record,
    b_capture, b_checkpointing, b_decision, b_prediction, b_result_record,
    b_upstream, b_zero_stage, c_alternative, c_capture, c_decision,
    c_prediction, c_records, c_selected_key, c_upstream, convergence_fixture,
    d_base_record, d_capture, d_decision, d_microbatches, d_prediction,
    d_result_record, d_upstream, e_base_record, e_capture, e_decision,
    e_prediction, e_result_record, e_upstream, final_choice, final_rejected,
    final_risk, final_trigger, get_evidence, go, mo, rationale, scenario,
    straggler_fixture, strong_points, track_id, weak_points,
):
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=_upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def q(record, key):
        return record[key]["value"]

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture after changing upstream choices."), kind="danger")
        snapshot = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {snapshot["prediction"]}<br><small>The report keeps this result even if live controls move.</small></div>')

    def part_a():
        intro = mo.md(f"### A · When do more workers stop helping? (9 min)\nThe **{scenario.model.name}** upstream job starts with **{scenario.worker_counts[0]} accelerators**. Choose fixed global work or fixed work per worker, then predict the larger pool’s efficiency.")
        if a_prediction.value is None:
            return mo.vstack([intro, mo.hstack([a_mode, a_workers], widths="equal", wrap=True), a_prediction])
        fig = go.Figure()
        for _name, _points, _color in (("Strong", strong_points, COLORS["BlueLine"]), ("Weak", weak_points, COLORS["OrangeLine"])):
            fig.add_scatter(x=[point.workers for point in _points], y=[point.parallel_efficiency for point in _points], mode="lines+markers", name=_name, line=dict(color=_color))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=25, b=20), xaxis_title="Active accelerators", yaxis_title="Parallel efficiency (fraction)")
        rows = [
            {"Run": "Baseline", "Workers": a_base_record["workers"], "Global batch": a_base_record["global_batch"], "Local batch": a_base_record["local_batch"], "Step": f'{q(a_base_record, "step_time"):.2f} ms', "Communication": f'{q(a_base_record, "communication_time"):.2f} ms'},
            {"Run": "Selected", "Workers": a_result_record["workers"], "Global batch": a_result_record["global_batch"], "Local batch": a_result_record["local_batch"], "Step": f'{q(a_result_record, "step_time"):.2f} ms', "Communication": f'{q(a_result_record, "communication_time"):.2f} ms'},
        ]
        return mo.vstack([intro, mo.hstack([a_mode, a_workers], widths="equal", wrap=True), a_prediction, apply_plotly_theme(fig), table(rows), mo.callout(mo.md(f'**Your prediction:** {a_prediction.value}. **Analytical result:** {a_result_record["speedup"]:.2f}× useful-throughput speedup at {a_result_record["parallel_efficiency"]:.2f} parallel efficiency. Communication is **{q(a_result_record, "communication_time"):.2f} ms**.'), kind="info"), a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("Strong scaling holds global batch fixed; weak scaling holds local batch fixed. Step time combines local training with gradient synchronization over the registered fleet fabric. These are analytical estimates, not measurements.")})])

    def part_b():
        intro = mo.md(f"### B · What does making the model fit cost? (10 min)\n**Backend capacity case:** Llama 3 70B on {scenario.fleet.total_accelerators} accelerators ({scenario.fleet.name}). Compare replicated state with one ZeRO stage and activation policy. State sharding is an unsupported path on compact deployment devices.")
        if b_prediction.value is None:
            return mo.vstack([intro, mo.hstack([b_zero_stage, b_checkpointing], widths="equal", wrap=True), b_prediction])
        components = ("weights", "gradients", "optimizer_state", "activations", "communication_buffers")
        fig = go.Figure()
        for _key, _color in zip(components, (COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["TextMuted"], COLORS["RedLine"])):
            fig.add_bar(name=_key.replace("_", " ").title(), x=["Replicated", f"ZeRO-{b_zero_stage.value}"], y=[q(b_base_record, _key), q(b_result_record, _key)], marker_color=_color)
        fig.update_layout(barmode="stack", height=300, margin=dict(l=20, r=20, t=25, b=20), yaxis_title="Per-accelerator memory (GB)", legend_orientation="h")
        rows = [
            {"Plan": "Replicated", "Required": f'{q(b_base_record, "total"):.1f} GB', "Available": f'{q(b_base_record, "available"):.1f} GB', "Outcome": "FIT" if b_base_record["feasible"] else "OOM · exceeds HBM"},
            {"Plan": f"ZeRO-{b_zero_stage.value}", "Required": f'{q(b_result_record, "total"):.1f} GB', "Available": f'{q(b_result_record, "available"):.1f} GB', "Outcome": "FIT" if b_result_record["feasible"] else "OOM · exceeds HBM"},
        ]
        return mo.vstack([intro, mo.hstack([b_zero_stage, b_checkpointing], widths="equal", wrap=True), b_prediction, apply_plotly_theme(fig), table(rows), b_decision, mo.callout(mo.md(f'**Your prediction:** {b_prediction.value}. **Analytical result:** replicated state needs **{q(b_base_record, "total"):.1f} GB** per accelerator; the selected plan needs **{q(b_result_record, "total"):.1f} GB** and **{"fits" if b_result_record["feasible"] else "still fails"}**.'), kind="success" if b_result_record["feasible"] else "danger"), b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md("Total memory is weights + gradients + optimizer state + retained activations + communication buffers. ZeRO stages cumulatively shard optimizer state, gradients, then weights. Recomputation reduces stored activations but adds compute outside this capacity comparison.")})])

    def part_c():
        intro = mo.md(f"### C · Where should each parallel dimension run? (10 min)\n**Backend placement case:** Llama 3 8B on {scenario.fleet.total_accelerators} accelerators ({scenario.fleet.name}). The baseline keeps TP=8 inside each node. Test cross-node TP or a two-stage pipeline. Model parallelism is an unsupported path on compact deployment devices.")
        if c_prediction.value is None:
            return mo.vstack([intro, c_alternative, c_prediction])
        def layout_label(_key):
            rec = c_records[_key]
            return f'TP{rec["tp_size"]} · PP{rec["pp_size"]} · DP{rec["dp_size"]}'
        rows = []
        for _key in ("within", "cross", "pipeline"):
            _record = c_records[_key]
            rows.append({"Layout": layout_label(_key), "TP tier": _record["tensor_parallel_tier"], "DP comm": f'{q(_record, "dp_communication_time"):.2f} ms', "TP comm": f'{q(_record, "tp_communication_time"):.2f} ms', "Bubble": f'{q(_record, "pipeline_bubble_time"):.2f} ms', "Step": f'{q(_record, "step_time"):.2f} ms'})
        fig = go.Figure()
        for _key in ("within", c_selected_key):
            _record = c_records[_key]
            fig.add_bar(name="DP communication", x=[layout_label(_key)], y=[q(_record, "dp_communication_time")], marker_color=COLORS["BlueLine"])
            fig.add_bar(name="TP communication", x=[layout_label(_key)], y=[q(_record, "tp_communication_time")], marker_color=COLORS["OrangeLine"])
            fig.add_bar(name="Pipeline bubble", x=[layout_label(_key)], y=[q(_record, "pipeline_bubble_time")], marker_color=COLORS["GreenLine"])
        fig.update_layout(barmode="stack", height=285, margin=dict(l=20, r=20, t=25, b=20), yaxis_title="Exposed overhead (ms)", legend_orientation="h")
        _selected = c_records[c_selected_key]
        return mo.vstack([intro, c_alternative, c_prediction, apply_plotly_theme(fig), table(rows), c_decision, mo.callout(mo.md(f'**Your prediction:** {c_prediction.value}. **Analytical result:** the selected alternative puts TP on the **{_selected["tensor_parallel_tier"]}** tier and spends **{q(_selected, "tp_communication_time"):.2f} ms** in TP communication versus **{q(c_records["within"], "tp_communication_time"):.2f} ms** for within-node TP.'), kind="info"), c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md(f"Every layout uses exactly {scenario.fleet.total_accelerators} accelerators ({scenario.fleet.name}). TP uses the node link while it fits within eight accelerators, then the registered inter-node fabric. DP synchronization and pipeline fill/drain are recomputed for each factorization.")})])

    def part_d():
        intro = mo.md(f"### D · Can a smaller pipeline bubble cause another failure? (10 min)\n**Backend pipeline schedule:** Llama 3 8B on {scenario.fleet.name}. Hold microbatch size at 4 sequences and increase only the count entering an eight-stage pipeline. Predict bubble and worst-stage activation retention. Pipelining is an unsupported path on compact deployment devices.")
        if d_prediction.value is None:
            return mo.vstack([intro, d_microbatches, d_prediction])
        fig = go.Figure()
        fig.add_scatter(name="Bubble fraction", x=[2, d_result_record["microbatch_count"]], y=[d_base_record["bubble_fraction"], d_result_record["bubble_fraction"]], mode="lines+markers", line=dict(color=COLORS["BlueLine"]), yaxis="y")
        fig.add_scatter(name="Retained activations", x=[2, d_result_record["microbatch_count"]], y=[q(d_base_record, "retained_activation_memory"), q(d_result_record, "retained_activation_memory")], mode="lines+markers", line=dict(color=COLORS["OrangeLine"]), yaxis="y2")
        fig.update_layout(height=285, margin=dict(l=20, r=55, t=25, b=20), xaxis_title="Pipeline microbatch count", yaxis=dict(title="Bubble fraction"), yaxis2=dict(title="Activation memory (GB)", overlaying="y", side="right"), legend_orientation="h")
        rows = [
            {"Schedule": "Baseline", "Count": d_base_record["microbatch_count"], "Size": d_base_record["microbatch_size"], "Bubble": f'{d_base_record["bubble_fraction"]:.3f}', "Retained": d_base_record["retained_microbatches"], "Activations": f'{q(d_base_record, "retained_activation_memory"):.1f} GB', "Outcome": "FIT" if d_base_record["feasible"] else "OOM"},
            {"Schedule": "Selected", "Count": d_result_record["microbatch_count"], "Size": d_result_record["microbatch_size"], "Bubble": f'{d_result_record["bubble_fraction"]:.3f}', "Retained": d_result_record["retained_microbatches"], "Activations": f'{q(d_result_record, "retained_activation_memory"):.1f} GB', "Outcome": "FIT" if d_result_record["feasible"] else "OOM · activations exceed HBM"},
        ]
        return mo.vstack([intro, d_microbatches, d_prediction, apply_plotly_theme(fig), table(rows), d_decision, mo.callout(mo.md(f'**Your prediction:** {d_prediction.value}. **Analytical result:** bubble fraction changes from **{d_base_record["bubble_fraction"]:.3f}** to **{d_result_record["bubble_fraction"]:.3f}**, while retained activations change from **{q(d_base_record, "retained_activation_memory"):.1f} GB** to **{q(d_result_record, "retained_activation_memory"):.1f} GB**.'), kind="danger" if not d_result_record["feasible"] else "success"), d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md("Bubble fraction = (stages − 1) / (microbatches + stages − 1). The worst stage retains up to the smaller of microbatch count and pipeline depth. Microbatch count and size remain separate inputs.")})])

    def part_e():
        intro = mo.md(f"### E · Is the fastest step the best training policy? (8 min)\nReturn to upstream training of **{scenario.model.name}** on **{scenario.fleet.name}** for deployment to **{scenario.deployment_target}**. The convergence and straggler behavior are explicitly illustrative scenario assumptions, not measured quality evidence.")
        if e_prediction.value is None:
            return mo.vstack([intro, e_prediction])
        rows = [
            {"Policy": e_base_record["name"], "Step": f'{q(e_base_record, "step_time"):.2f} ms', "Steps to target": e_base_record["optimizer_steps"], "Time to target": f'{q(e_base_record, "time_to_quality"):.2f} h'},
            {"Policy": e_result_record["name"], "Step": f'{q(e_result_record, "step_time"):.2f} ms', "Steps to target": e_result_record["optimizer_steps"], "Time to target": f'{q(e_result_record, "time_to_quality"):.2f} h'},
        ]
        fig = go.Figure([go.Bar(x=[e_base_record["name"], e_result_record["name"]], y=[q(e_base_record, "time_to_quality"), q(e_result_record, "time_to_quality")], marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]])])
        fig.update_layout(height=270, margin=dict(l=20, r=20, t=25, b=20), yaxis_title="Time to matched quality target (hours)", showlegend=False)
        return mo.vstack([intro, e_prediction, apply_plotly_theme(fig), table(rows), e_decision, mo.callout(mo.md(f'**Your prediction:** {e_prediction.value}. **Illustrative result:** dropping slow workers shortens each step by **{straggler_fixture.step_time_multiplier:.2f}×**, but multiplies optimization work by **{straggler_fixture.work_multiplier:.2f}×**. Time to target changes from **{q(e_base_record, "time_to_quality"):.2f} h** to **{q(e_result_record, "time_to_quality"):.2f} h**.'), kind="info"), e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md(f"Scenario fixture: {convergence_fixture.name}. Optimizer steps follow a diminishing-return batch curve around the stated critical batch. Replace these illustrative policy multipliers with workload evidence before a real decision.")})])

    def build_synthesis():
        rows = []
        for _part in "ABCDE":
            _capture = _captures.get(_part)
            rows.append({"Part": _part, "Original prediction": _capture.to_dict()["prediction"] if _capture else "—", "Evidence": "CURRENT" if _capture and _part not in audit.stale and (_part, _part) not in audit.identical_pairs else ("STALE" if _capture else "MISSING")})
        _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and final_choice.value != final_rejected.value and bool(rationale.value.strip())
        return mo.vstack([mo.md("### Synthesis · Defend one training plan (5 min)\nChoose a plan, quantify a rejected alternative from saved evidence, name the remaining limitation, and state the trigger that would reopen the decision."), table(rows), mo.callout(mo.md("Saved snapshots preserve original predictions and evaluator inputs. Track changes invalidate all five; Part A scaling changes invalidate the quality-time comparison."), kind="info"), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale, mo.callout(mo.md("**Ready for the local evidence report.**" if _ready else "Capture five current contrasts, choose different recommended and rejected plans, and add a quantified rationale."), kind="success" if _ready else "warn")])

    tabs = mo.ui.tabs({"Part A": part_a(), "Part B": part_b(), "Part C": part_c(), "Part D": part_d(), "Part E": part_e(), "Synthesis": build_synthesis()})
    tabs
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, rationale,
    report_export_panel, scenario, track_id,
):
    _captures = get_evidence()
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and final_choice.value != final_rejected.value and bool(rationale.value.strip())
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    binding_constraints = {
        "A": "communication and fixed-work scaling",
        "B": "per-accelerator HBM (unsupported for compact track model)" if not scenario.model_parallel_applicable else "per-accelerator HBM",
        "C": "interconnect tier (unsupported for compact track model)" if not scenario.model_parallel_applicable else "interconnect tier",
        "D": "bubble and retained activations (unsupported for compact track model)" if not scenario.model_parallel_applicable else "bubble and retained activations",
        "E": "time to matched illustrative quality",
    }
    report = build_lab_report(
        get_lab_metadata("vol2/lab_05_dist_train.py"), track=track_id,
        scenario=f"{scenario.model.name} training on {scenario.fleet.name} for {scenario.deployment_target}",
        learning_objectives=["Separate strong and weak scaling and quantify efficiency", "Account state and activations per accelerator", "Choose a layout using memory, link, pipeline, and time-to-quality evidence"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "chosen_result": snapshots[part].get("chosen_result"), "result_role": snapshots[part].get("result_role"), "alternatives": snapshots[part]["alternatives"]} for part in "ABCDE"},
        binding_constraints=binding_constraints,
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["Strong and weak scaling hold different work fixed.", "All training state and activations must fit together.", "Shorter steps need not reach matched quality sooner."],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": snapshots, "recommendation": final_choice.value, "rejected": final_rejected.value, "trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": "Registered model and hardware facts; explicitly illustrative convergence behavior.", "calculations": "Analytical distributed-training, memory, pipeline, and convergence models."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and final_choice.value != final_rejected.value and bool(rationale.value.strip())
    _save_status = "EVIDENCE IN PROGRESS"
    if _ready:
        try:
            ledger.save(chapter=5, design={
                "schema_version": 1, "lab_id": "v2_05", "track_id": track_id,
                "model_id": "v2_05_experiments",
                "evidence": {part: capture.to_dict() for part, capture in get_evidence().items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value, "rationale": rationale.value,
            })
            await ledger.flush()
            _save_status = "SAVED"
        except Exception as _error:
            _save_status = f"SAVE FAILED · {type(_error).__name__}"
    mo.Html(f'<div class="lab-hud" style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:#101827;color:#fff;padding:14px 18px;border-radius:9px;font-family:ui-monospace,monospace"><span>LAB 05 · Scaling Without Illusions</span><span aria-hidden="true">|</span><span>STATUS: {_save_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
