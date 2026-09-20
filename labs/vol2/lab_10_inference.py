import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 10: The Inference Economy · MLSysBook")


# ZONE A · SETUP
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
    from mlsysim.core.units import ureg
    from mlsysim.engine.v2_10_experiments import (
        MODEL_ID, admission_snapshot, compare_equal_budget_placements,
        cost_snapshot, illustrative_track_scenario, lifetime_cost_crossover,
        replay_requests, replay_snapshot, serialize_quantity, state_admission,
        unavailable_snapshot,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata, report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence
    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, MODEL_ID, admission_snapshot,
        apply_plotly_theme, audit_evidence, build_lab_report, capture_evidence,
        compare_equal_budget_placements, cost_snapshot, get_lab_metadata, go,
        illustrative_track_scenario, ledger, lifetime_cost_crossover, mo,
        replay_requests, replay_snapshot, report_export_panel,
        serialize_quantity, state_admission, unavailable_snapshot, ureg,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="Cloud", label="Deployment track",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(illustrative_track_scenario, track):
    track_id = track.value
    scenario = illustrative_track_scenario(track_id)
    return scenario, track_id


# ZONE B · WIDGETS
@app.cell
def _(mo, scenario, track_id):
    _track_key = track_id
    a_rate = mo.ui.dropdown(
        {label.title(): label for label, _value in scenario.request_rate_options},
        value="Growth", label="Request-volume case",
    )
    b_fragmentation = mo.ui.dropdown(
        {f"{value:.0%} unavailable": value for value in scenario.fragmentation_options},
        value="30% unavailable", label="Fragmented free memory",
    )
    c_window = mo.ui.dropdown(
        {f"{value.m_as('millisecond'):g} ms": value.m_as("millisecond") for value in scenario.batch_window_options},
        value="25 ms", label="Batch collection window",
    )
    c_batch = mo.ui.slider(2, 5, value=5, step=1, label="Maximum batch size")
    _choice_options = {}
    if "replicate" in scenario.supported_placements:
        _choice_options["Replicate"] = "replicate"
    if "shard" in scenario.supported_placements:
        _choice_options["Shard"] = "shard"
    if "split_phases" in scenario.supported_placements:
        _choice_options["Split phases"] = "split_phases"
    _choice_options["Hold current replication"] = "none"
    d_choice = mo.ui.radio(
        _choice_options,
        label="Placement decision",
    )
    d_rejected = mo.ui.radio(
        {"Replicate": "replicate", "Shard": "shard", "Split phases": "split_phases"},
        label="Rejected alternative",
    )
    e_lost = mo.ui.slider(0, scenario.baseline_replicas - 1, value=1, step=1, label="Lost replicas")
    e_warmup = mo.ui.dropdown(
        {f"{value.m_as('millisecond'):g} ms": value.m_as("millisecond") for value in scenario.warmup_options},
        value=f"{scenario.warmup_options[-1].m_as('millisecond'):g} ms", label=f"Startup delay ({scenario.startup_kind})",
    )
    return a_rate, b_fragmentation, c_batch, c_window, d_choice, d_rejected, e_lost, e_warmup


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Serving stays below training": "below", "Serving exceeds training": "above"},
        label="Which lifetime cost is larger?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Weights do not fit": "weights", "State admission fails": "state", "Everything fits": "fits"},
        label="What binds after fragmentation?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Lower p99 and more on time": "both", "Lower work span but worse deadlines": "tradeoff", "No meaningful change": "same"},
        label="What will batching do?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Replicate": "replicate", "Shard": "shard", "Split phases": "split_phases"},
        label="Which equal-device placement has the lowest p99?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"All requests stay on time": "all", "Some requests miss": "some", "No request finishes": "none"},
        label="What happens after loss and warmup?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, scenario, track_id):
    _track_key = track_id
    _final_choice_options = {}
    if "replicate" in scenario.supported_placements:
        _final_choice_options["Replicate"] = "replicate"
    if "shard" in scenario.supported_placements:
        _final_choice_options["Shard"] = "shard"
    if "split_phases" in scenario.supported_placements:
        _final_choice_options["Split phases"] = "split_phases"
    _final_choice_options["Hold current replication"] = "none"
    final_choice = mo.ui.radio(
        _final_choice_options,
        label="Final architecture",
    )
    final_rejected = mo.ui.radio(
        {"Replicate": "replicate", "Shard": "shard", "Split phases": "split_phases"},
        label="Quantified rejected architecture",
    )
    final_trigger = mo.ui.radio(
        {"p99 reaches the SLO": "p99", "State rejection appears": "state", "Serving cost reaches training cost": "cost"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Longer requests": "length", "Correlated replica loss": "failure", "Unvalidated quality shift": "quality"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Decision rationale",
        placeholder="Cite saved numbers, the rejected alternative, limitation, and trigger.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_rate, admission_snapshot, b_fragmentation, c_batch, c_window,
    compare_equal_budget_placements, cost_snapshot, e_lost, e_warmup,
    lifetime_cost_crossover, replay_requests, replay_snapshot, scenario,
    serialize_quantity, state_admission, ureg,
):
    def request_args():
        return [request.to_evaluator_args() for request in scenario.requests]
    def service_args():
        return scenario.service.to_evaluator_args()

    _rates = dict(scenario.request_rate_options)
    _normal_rate, _chosen_rate = _rates["normal"], _rates[a_rate.value]
    a_base_inputs = {
        "training_cost": serialize_quantity(scenario.training_cost),
        "request_rate": serialize_quantity(_normal_rate),
        "cost_per_request": serialize_quantity(scenario.cost_per_request),
        "horizon": serialize_quantity(scenario.horizon),
    }
    a_result_inputs = {**a_base_inputs, "request_rate": serialize_quantity(_chosen_rate)}
    a_base = lifetime_cost_crossover(
        training_cost=scenario.training_cost, request_rate=_normal_rate,
        cost_per_request=scenario.cost_per_request, horizon=scenario.horizon,
    )
    a_result = lifetime_cost_crossover(
        training_cost=scenario.training_cost, request_rate=_chosen_rate,
        cost_per_request=scenario.cost_per_request, horizon=scenario.horizon,
    )
    a_base_snapshot = cost_snapshot(a_base, inputs=a_base_inputs)
    a_result_snapshot = cost_snapshot(a_result, inputs=a_result_inputs)

    b_base_inputs = {
        "requests": request_args(), "device_memory": serialize_quantity(scenario.service.device_memory),
        "weight_memory": serialize_quantity(scenario.service.weight_memory),
        "reserved_memory": serialize_quantity(scenario.service.reserved_memory),
        "fragmentation_fraction": 0.0,
    }
    b_result_inputs = {**b_base_inputs, "fragmentation_fraction": b_fragmentation.value}
    b_base = state_admission(
        scenario.requests, device_memory=scenario.service.device_memory,
        weight_memory=scenario.service.weight_memory,
        reserved_memory=scenario.service.reserved_memory,
    )
    b_result = state_admission(
        scenario.requests, device_memory=scenario.service.device_memory,
        weight_memory=scenario.service.weight_memory,
        reserved_memory=scenario.service.reserved_memory,
        fragmentation_fraction=b_fragmentation.value,
    )
    b_base_snapshot = admission_snapshot(b_base, inputs=b_base_inputs)
    b_result_snapshot = admission_snapshot(b_result, inputs=b_result_inputs)

    _zero_ms, _window = 0 * ureg.millisecond, c_window.value * ureg.millisecond
    c_base_inputs = {
        "requests": request_args(), "service": service_args(), "replicas": 2,
        "policy": "immediate", "slo": serialize_quantity(scenario.slo),
        "max_batch": 1, "batch_window": serialize_quantity(_zero_ms),
        "warmup": serialize_quantity(_zero_ms), "lost_replicas": 0,
    }
    c_result_inputs = {**c_base_inputs, "policy": "windowed_batch", "max_batch": c_batch.value, "batch_window": serialize_quantity(_window)}
    c_base = replay_requests(
        scenario.requests, service=scenario.service, replicas=2,
        policy="immediate", slo=scenario.slo,
    )
    c_result = replay_requests(
        scenario.requests, service=scenario.service, replicas=2,
        policy="windowed_batch", slo=scenario.slo, max_batch=c_batch.value,
        batch_window=_window,
    )
    c_base_snapshot = replay_snapshot(c_base, inputs=c_base_inputs)
    c_result_snapshot = replay_snapshot(c_result, inputs=c_result_inputs)

    d_inputs = {
        "requests": request_args(), "service": service_args(),
        "device_budget": scenario.device_budget, "slo": serialize_quantity(scenario.slo),
        "handoff_bytes": serialize_quantity(scenario.handoff_bytes),
        "link_bandwidth": serialize_quantity(scenario.link_bandwidth),
        "link_latency": serialize_quantity(scenario.link_latency),
        "track_id": scenario.track_id,
    }
    d_results = compare_equal_budget_placements(
        scenario.requests, service=scenario.service,
        device_budget=scenario.device_budget, slo=scenario.slo,
        handoff_bytes=scenario.handoff_bytes, link_bandwidth=scenario.link_bandwidth,
        link_latency=scenario.link_latency,
        track_id=scenario.track_id,
    )
    d_by_name = {item.placement: item for item in d_results}
    d_snapshots = {
        name: replay_snapshot(
            item.replay,
            inputs={**d_inputs, "placement": name},
            available=item.available,
            unsupported_reason=item.unsupported_reason,
        )
        for name, item in d_by_name.items()
    }

    _warmup = e_warmup.value * ureg.millisecond
    e_base_inputs = {
        "requests": request_args(), "service": service_args(),
        "replicas": scenario.baseline_replicas, "policy": "immediate",
        "slo": serialize_quantity(scenario.slo), "max_batch": 1,
        "batch_window": serialize_quantity(_zero_ms), "warmup": serialize_quantity(_zero_ms),
        "lost_replicas": 0,
    }
    e_result_inputs = {**e_base_inputs, "warmup": serialize_quantity(_warmup), "lost_replicas": e_lost.value}
    e_base = replay_requests(
        scenario.requests, service=scenario.service, replicas=scenario.baseline_replicas,
        policy="immediate", slo=scenario.slo,
    )
    e_result = replay_requests(
        scenario.requests, service=scenario.service, replicas=scenario.baseline_replicas,
        policy="immediate", slo=scenario.slo, warmup=_warmup, lost_replicas=e_lost.value,
    )
    e_base_snapshot = replay_snapshot(e_base, inputs=e_base_inputs)
    e_result_snapshot = replay_snapshot(e_result, inputs=e_result_inputs)
    return (
        a_base, a_base_snapshot, a_result, a_result_snapshot,
        b_base, b_base_snapshot, b_result, b_result_snapshot,
        c_base, c_base_snapshot, c_result, c_result_snapshot,
        d_by_name, d_snapshots, e_base, e_base_snapshot, e_result, e_result_snapshot,
    )


@app.cell
def _(
    a_base_snapshot, a_prediction, a_rate, a_result_snapshot,
    b_base_snapshot, b_fragmentation, b_prediction, b_result_snapshot,
    c_base_snapshot, c_batch, c_prediction, c_result_snapshot, c_window,
    capture_evidence, d_choice, d_prediction, d_rejected, d_snapshots,
    e_base_snapshot, e_lost, e_prediction, e_result_snapshot, e_warmup,
    mo, scenario, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})
    a_upstream = {"scenario": track_id}
    b_upstream = {"scenario": track_id}
    c_upstream = {"scenario": track_id}
    d_upstream = {"scenario": track_id}
    e_upstream = {"scenario": track_id, "placement_decision": d_choice.value}
    a_capture = mo.ui.button(
        label="Capture lifetime cost contrast", kind="success",
        disabled=a_prediction.value is None or a_rate.value == "normal",
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"request_volume_case": a_rate.value}, baseline=a_base_snapshot,
            result=a_result_snapshot, upstream_inputs=a_upstream,
            model_key="v2_10_experiments.lifetime_cost_crossover",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture state admission contrast", kind="success",
        disabled=b_prediction.value is None or b_fragmentation.value == 0,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"fragmentation_fraction": b_fragmentation.value},
            baseline=b_base_snapshot, result=b_result_snapshot,
            upstream_inputs=b_upstream, model_key="v2_10_experiments.state_admission",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture scheduling replay", kind="success",
        disabled=c_prediction.value is None or c_window.value == 0,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"batch_window_ms": c_window.value, "max_batch": c_batch.value},
            baseline=c_base_snapshot, result=c_result_snapshot,
            upstream_inputs=c_upstream, model_key="v2_10_experiments.replay_requests",
        )),
    )
    _chosen = d_choice.value
    _contrast = d_rejected.value if _chosen in (None, "replicate", "none") else _chosen
    _chosen_snapshot = d_snapshots["replicate"] if _chosen == "none" else d_snapshots.get(_chosen, d_snapshots["replicate"])
    _result_role = "rejected alternative" if _chosen in ("replicate", "none") else "chosen intervention"
    _choice_valid = d_choice.value in scenario.supported_placements or d_choice.value == "none"
    d_capture = mo.ui.button(
        label="Capture equal-budget placement", kind="success",
        disabled=(
            d_prediction.value is None
            or d_choice.value is None
            or d_rejected.value is None
            or d_choice.value == d_rejected.value
            or not _choice_valid
            or _contrast == "replicate"
        ),
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={"choice": d_choice.value, "rejected": d_rejected.value},
            baseline=d_snapshots["replicate"], result=d_snapshots[_contrast],
            chosen_result=_chosen_snapshot, result_role=_result_role,
            alternatives=tuple(d_snapshots.values()), decision=d_choice.value,
            upstream_inputs=d_upstream,
            model_key="v2_10_experiments.compare_equal_budget_placements",
        )),
    )
    e_capture = mo.ui.button(
        label="Capture capacity stress", kind="success",
        disabled=e_prediction.value is None or d_choice.value is None or (e_lost.value == 0 and e_warmup.value == 0),
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"lost_replicas": e_lost.value, "warmup_ms": e_warmup.value},
            baseline=e_base_snapshot, result=e_result_snapshot,
            upstream_inputs=e_upstream, model_key="v2_10_experiments.replay_requests",
        )),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream, e_capture, e_upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, scenario, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}.pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}.pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:760px}.pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}.pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.track-space{margin-left:24px;max-width:calc(100% - 24px)}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;column-gap:14px;row-gap:6px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf;margin-right:2px}.lab-hud .hud-value{color:#fff;margin-right:10px}.lab-hud .hud-active{color:#86efac}@media(max-width:520px){.pilot-head{border-radius:9px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME II · LAB 10</span><span>ABOUT 50–55 MIN</span></div><h1>The Inference Economy</h1><p>Which serving design minimizes cost per successful, on-time request when memory, waiting, and coordination all count?</p><div class="pilot-meta"><div><b>Fleet role</b><br>{scenario.backend_role}</div><div><b>Workload</b><br>{scenario.workload_kind}</div><div><b>State</b><br>{scenario.state_kind}</div><div><b>Startup delay</b><br>{scenario.startup_kind}</div><div><b>Deliverable</b><br>Fleet serving decision</div></div></section>""")
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, track.style({"margin-top": "30px", "margin-left": "24px", "max-width": "calc(100% - 24px)"}), header, mo.Html('<p class="pilot-note">All traces and outcome evidence are illustrative scenarios, not production measurements. Calculation Notes state each model boundary.</p>')], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


# ZONE C · SINGLE TAB SET
@app.cell
def _(
    COLORS, a_base, a_capture, a_prediction, a_rate, a_result, a_upstream,
    apply_plotly_theme, audit_evidence, b_base, b_capture, b_fragmentation,
    b_prediction, b_result, b_upstream, c_base, c_batch, c_capture,
    c_prediction, c_result, c_upstream, c_window, d_by_name, d_capture,
    d_choice, d_prediction, d_rejected, d_upstream, e_base, e_capture,
    e_lost, e_prediction, e_result, e_upstream, e_warmup, final_choice,
    final_rejected, final_risk, final_trigger, get_evidence, go, mo,
    rationale, scenario, track_id,
):
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=_upstream, contrast_required_parts=tuple("ABCDE"),
    )
    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})
    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Review and recapture."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later control changes do not rewrite this record.</small></div>')
    def replay_row(label, result):
        if result is None:
            return {"Run": label, "Submitted": "—", "Completed": "—", "Rejected": "—", "On time": "—", "Complete-duration p99": "Unavailable"}
        _p99 = f"{result.p99_duration.m_as('millisecond'):.1f} ms" if result.completed_count else "Unavailable"
        return {"Run": label, "Submitted": result.request_count, "Completed": result.completed_count, "Rejected": result.rejected_count, "On time": result.deadline_compliant_count, "Complete-duration p99": _p99}

    def build_part_a():
        intro = mo.md("### A · When does serving outweigh training? (8 min)\n**Driving question:** Does recurring serving exceed one-time training cost over the same lifetime?")
        if a_prediction.value is None:
            return mo.vstack([intro, a_rate, a_prediction])
        rows = [
            {"Case": "Normal", "Serving": f"${a_base.serving_cost.m_as('dollar'):,.0f}", "Training": f"${a_base.training_cost.m_as('dollar'):,.0f}", "Crossover": f"{a_base.crossover_time.m_as('week'):.1f} weeks"},
            {"Case": a_rate.value.title(), "Serving": f"${a_result.serving_cost.m_as('dollar'):,.0f}", "Training": f"${a_result.training_cost.m_as('dollar'):,.0f}", "Crossover": f"{a_result.crossover_time.m_as('week'):.1f} weeks"},
        ]
        return mo.vstack([intro, a_rate, a_prediction, table(rows), mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. **Scenario result:** serving/training is **{a_result.serving_to_training_ratio:.2f}×** over {a_result.horizon.m_as('week'):.0f} weeks."), kind="info"), a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("Serving cost is request rate × the same horizon × cost per request. Crossover divides training cost by the serving cost rate. Results are analytical scenario values.")})])

    def build_part_b():
        intro = mo.md(f"### B · Why can a model fit while its service fails? (9 min)\n**Driving question:** How many concurrent {scenario.state_kind} allocations survive fragmentation?")
        if b_prediction.value is None:
            return mo.vstack([intro, b_fragmentation, b_prediction])
        rows = [
            {"Run": "Compact", "Usable": f"{b_base.usable_state_memory.m_as('megabyte'):,.0f} MB", "Max concurrent": b_base.max_concurrent_requests, "Rejected": len(b_base.rejected_request_ids)},
            {"Run": f"{b_fragmentation.value:.0%} fragmented", "Usable": f"{b_result.usable_state_memory.m_as('megabyte'):,.0f} MB", "Max concurrent": b_result.max_concurrent_requests, "Rejected": len(b_result.rejected_request_ids)},
        ]
        return mo.vstack([intro, b_fragmentation, b_prediction, table(rows), mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. **Scenario result:** admitted requests change from **{len(b_base.admitted_request_ids)}** to **{len(b_result.admitted_request_ids)}**."), kind="danger" if b_result.rejected_request_ids else "info"), b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md("Free memory subtracts weights and reserve. Fragmentation reduces usable request-state memory. Requests are admitted in trace order. Device tracks use the named buffers, not KV-cache terminology.")})])

    def build_part_c():
        intro = mo.md("### C · When does batching save work but miss deadlines? (10 min)\n**Driving question:** What changes when identical arrivals wait for a batch window?")
        if c_prediction.value is None:
            return mo.vstack([intro, c_window, c_batch, c_prediction])
        fig = go.Figure([
            go.Bar(name="Immediate", x=[o.request_id for o in c_base.outcomes if o.duration is not None], y=[o.duration.m_as("millisecond") for o in c_base.outcomes if o.duration is not None], marker_color=COLORS["BlueLine"]),
            go.Bar(name="Windowed", x=[o.request_id for o in c_result.outcomes if o.duration is not None], y=[o.duration.m_as("millisecond") for o in c_result.outcomes if o.duration is not None], marker_color=COLORS["OrangeLine"]),
        ])
        fig.update_layout(barmode="group", height=280, margin=dict(l=20, r=20, t=25, b=20), yaxis_title="Complete request duration (ms)", legend_orientation="h")
        return mo.vstack([intro, mo.hstack([c_window, c_batch], widths="equal", wrap=True), c_prediction, apply_plotly_theme(fig), table([replay_row("Immediate", c_base), replay_row("Windowed", c_result)]), mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. **Scenario result:** batching finishes in **{c_result.makespan.m_as('millisecond'):.1f} ms** with **{c_result.deadline_compliant_count}/{c_result.request_count}** on time."), kind="info"), c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md("Both policies replay identical arrivals and lengths. A batch waits for its explicit window, then executes the longest input and output work in the batch. p99 is the empirical nearest-rank percentile of complete arrival-to-finish durations. Rejections remain separate.")})])

    def build_part_d():
        intro = mo.md("### D · Replicate, shard, or split phases? (10 min)\n**Driving question:** Which placement wins with the same device budget?")
        if d_prediction.value is None:
            return mo.vstack([intro, d_prediction])
        rows = []
        for name, item in d_by_name.items():
            row = replay_row(name.replace("_", " ").title(), item.replay)
            handoff = (
                f"{item.transfer_time_per_request.m_as('millisecond'):.2f} ms"
                if item.transfer_time_per_request is not None
                else "Unsupported"
            )
            arch_status = "Supported" if item.available else f"Unsupported: {item.unsupported_reason}"
            rows.append(row | {"Handoff/request": handoff, "Devices": item.device_budget, "Architecture": arch_status})
        return mo.vstack([intro, d_prediction, table(rows), mo.hstack([d_choice, d_rejected], widths="equal", wrap=True), mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. **Decision under review:** {d_choice.value or 'not selected'}. Every row uses the same workload, format, and device count."), kind="info"), d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md("Replication creates one worker per device. Sharding pools memory, divides compute, and adds a transfer per boundary (unsupported on endpoints without multi-device fabrics). Phase splitting routes requests through separate stage queues and one handoff (unsupported for the frame-perception and endpoint scenarios).")})])

    def build_part_e():
        intro = mo.md(
            f"### E · How much spare capacity buys reliability? (8 min)\n"
            f"**Driving question:** Does the {scenario.backend_role} meet its deadline after replica loss and {scenario.startup_kind}?"
        )
        if e_prediction.value is None:
            return mo.vstack([intro, e_lost, e_warmup, e_prediction])
        return mo.vstack([intro, mo.hstack([e_lost, e_warmup], widths="equal", wrap=True), e_prediction, table([replay_row("Healthy pool", e_base), replay_row(f"Loss + {scenario.startup_kind}", e_result)]), mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. **Scenario result:** on-time completions change from **{e_base.deadline_compliant_count}/{e_base.request_count}** to **{e_result.deadline_compliant_count}/{e_result.request_count}**."), kind="danger" if e_result.deadline_compliant_count < e_base.deadline_compliant_count else "success"), e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md(f"Lost replicas are absent for the replay. Surviving replicas accept work only after {scenario.startup_kind}. Device tracks reflect device wake/activation while backend tracks reflect service provisioning. Original arrivals remain fixed, so the timeline exposes capacity loss and queueing.")})])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            rows.append({"Part": part, "Original prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING")})
        _saved = _captures["D"].to_dict()["decision"] if "D" in _captures else None
        _choice_valid = final_choice.value in scenario.supported_placements or final_choice.value == "none"
        _complete = audit.complete and all(w.value is not None for w in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved and _choice_valid
        return mo.vstack([mo.md("### Synthesis · Defend one serving architecture (5 min)\nName the chosen placement, quantify a rejected alternative, state a limitation, and define a reevaluation trigger."), table(rows), mo.callout(mo.md("Snapshots preserve exact evaluator arguments and captured results. Live controls never rewrite saved evidence."), kind="info"), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_risk, final_trigger], widths="equal", wrap=True), rationale, mo.callout(mo.md("**Ready for the ledger and report.**" if _complete else "Complete five current captures, match Part D, reject a different placement, and write the rationale."), kind="success" if _complete else "warn")])

    tabs = mo.ui.tabs({"Part A": build_part_a(), "Part B": build_part_b(), "Part C": build_part_c(), "Part D": build_part_d(), "Part E": build_part_e(), "Synthesis": build_synthesis()})
    tabs
    return (audit,)


@app.cell
def _(
    MODEL_ID, audit, build_lab_report, final_choice, final_rejected,
    final_risk, final_trigger, get_evidence, get_lab_metadata, mo, rationale,
    report_export_panel, scenario, track_id,
):
    _captures = get_evidence()
    _saved = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _choice_valid = final_choice.value in scenario.supported_placements or final_choice.value == "none"
    _ready = audit.complete and all(w.value is not None for w in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved and _choice_valid
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol2/lab_10_inference.py"), track=track_id,
        scenario=scenario.workload_kind,
        learning_objectives=["Compare training and serving cost over one horizon", "Replay admission, scheduling, placement, and failure", "Defend an architecture using complete-request tail latency"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "alternatives": snapshots[part]["alternatives"]} for part in "ABCDE"},
        binding_constraints={"state_rejections": snapshots["B"]["result"]["rejected_request_ids"], "batch_on_time": snapshots["C"]["result"]["deadline_compliant_count"], "stress_on_time": snapshots["E"]["result"]["deadline_compliant_count"]},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["Serving cost compounds across the operating lifetime.", "A fitting model can reject concurrent state.", "Tail latency comes from complete request timelines."],
        reflections={"rationale": rationale.value, "trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"schema_version": 1, "lab_id": "v2_10", "track_id": track_id, "model_id": MODEL_ID, "evidence": snapshots, "recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value, "residual_risk": final_risk.value, "rationale": rationale.value},
        source_trace={"scenario": "Illustrative request traces and outcome fixtures.", "calculations": "MLSysIM v2_10 experiment models."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


# ZONE D · COMPLETION-GATED LEDGER
@app.cell
async def _(
    MODEL_ID, audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, scenario, track_id,
):
    _captures = get_evidence()
    _saved = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _choice_valid = final_choice.value in scenario.supported_placements or final_choice.value == "none"
    _ready = audit.complete and all(w.value is not None for w in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved and _choice_valid
    _saved_ok = False
    _save_error = ""
    if _ready:
        try:
            ledger.save(chapter=10, design={
                "schema_version": 1, "lab_id": "v2_10", "track_id": track_id,
                "model_id": MODEL_ID,
                "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value, "rationale": rationale.value,
            })
            await ledger.flush()
            _saved_ok = True
        except Exception as exc:
            _save_error = f"SAVE FAILED · {type(exc).__name__}: {exc}"
    _status = "SAVED" if _saved_ok else (_save_error or "EVIDENCE IN PROGRESS")
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">10 · The Inference Economy</span><span aria-hidden="true">|</span><span style="flex:1"></span><span class="hud-label">STATUS</span><span class="hud-active">{_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
