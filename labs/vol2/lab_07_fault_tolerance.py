import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 07: When Failure Is Routine · MLSysBook")


@app.cell
async def _():
    import html as html_lib
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
    from mlsysim.labs.experiments.v2_07_experiments import (
        TRACKS, checkpoint_fault_fixture, checkpoint_tradeoff,
        fault_verification_trace, gray_failure_trace, elastic_recovery_trace,
        get_track_scenario, replica_availability, select_restore_checkpoint,
        serving_capacity, track_scenario, training_exposure,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata,
        get_track_profile, report_export_panel, track_context,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence

    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, TRACKS, apply_plotly_theme,
        audit_evidence, build_lab_report, capture_evidence,
        checkpoint_fault_fixture, checkpoint_tradeoff, fault_verification_trace, gray_failure_trace, elastic_recovery_trace, get_lab_metadata, get_track_profile,
        get_track_scenario, go, html_lib, ledger,
        mo, replica_availability, report_export_panel,
        select_restore_checkpoint, serving_capacity, track_context,
        track_scenario, training_exposure,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell(hide_code=True)
def _(ledger, mo, set_evidence):
    from mlsysbook_labs.contexts import preferred_context as _preferred_context
    _track_options = {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"}
    track = mo.ui.dropdown(
        _track_options,
        value=next(label for label, context in _track_options.items() if context == _preferred_context(ledger)), label="Deployment context", on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_track_profile, get_track_scenario, track):
    track_id = track.value
    scenario = get_track_scenario(track_id)
    profile = get_track_profile(track_id)
    return profile, scenario, track_id


@app.cell
def _(mo, scenario, track_id):
    _track_key = track_id
    a_workers = mo.ui.slider(
        scenario["worker_min"], scenario["worker_max"],
        value=scenario["worker_default"], step=scenario["worker_step"],
        label=f"Coupled {scenario['training_unit']}",
    )
    b_interval = mo.ui.slider(
        5, 180, value=scenario["checkpoint_interval_default"], step=5,
        label="Checkpoint interval (min)",
    )
    c_fault = mo.ui.dropdown(
        {"Newest generation is corrupt": "corrupt", "Newest write is incomplete": "incomplete", "One shard belongs to an older step": "mixed_step", "Complete shards, no published manifest": "unpublished", "One required shard is missing": "missing_shard"},
        value="Newest generation is corrupt", label="Injected checkpoint fault",
    )
    d_replicas = mo.ui.slider(2, 4, value=3, step=1, label=f"{scenario['replica_role'].capitalize()} replicas")
    d_decision = mo.ui.radio(
        {"Spread across domains": "spread", "Keep colocated": "colocated"},
        label="Placement decision",
    )
    e_state_scale = mo.ui.slider(5, 20, value=20, step=5, label="Recovery-state multiplier")
    e_bandwidth_scale = mo.ui.slider(2, 12, value=12, step=1, label="Added-path bandwidth multiplier")
    e_decision = mo.ui.radio(
        {"Allocate bandwidth to checkpoint writes": "write", "Allocate bandwidth to restore reads": "restore", "Take no action": "none"},
        label="Recovery decision",
    )
    e_rejected = mo.ui.radio(
        {"Checkpoint-write path": "write", "Restore-read path": "restore"},
        label="Quantified rejected alternative",
    )
    return a_workers, b_interval, c_fault, d_decision, d_replicas, e_bandwidth_scale, e_decision, e_rejected, e_state_scale


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Less than 25%": "under_25", "25% to 75%": "25_to_75", "More than 75%": "over_75"},
        label="Probability that at least one event interrupts the job",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Checkpoint writes": "checkpoint_writes", "Lost work": "lost_work"},
        label="Which interval-dependent cost dominates at your setting?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Step 400": 400, "Step 300": 300, "Step 200": 200, "Step 100": 100, "No restore": 0},
        label="Which generation can recovery safely load?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Same availability": "same", "Spread placement is higher": "spread_higher", "Colocated placement is higher": "colocated_higher"},
        label="How does failure-domain placement change service availability?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Write allocation meets RTO": "write", "Restore allocation meets RTO": "restore", "Both meet RTO": "both", "Neither meets RTO": "neither"},
        label="Which equal-bandwidth allocation meets the recovery objective?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Allocate bandwidth to checkpoint writes": "write", "Allocate bandwidth to restore reads": "restore", "Hold current allocation": "none"},
        label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {"Checkpoint-write path": "write", "Restore-read path": "restore"},
        label="Rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"RPO requirement tightens": "rpo", "Recovery state grows": "state", "Failure-domain rate rises": "domain_rate"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Corruption escapes validation": "corruption", "Recovery phases vary under contention": "recovery_variance", "Failure domains are misspecified": "domain_model"},
        label="Residual limitation",
    )
    rationale = mo.ui.text_area(
        label="Decision rationale",
        placeholder="Use saved numbers: chosen outcome, rejected alternative, remaining limitation, and trigger.",
     debounce=False)
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_workers, b_interval, c_fault, checkpoint_fault_fixture,
    checkpoint_tradeoff, d_replicas, e_bandwidth_scale, e_state_scale, replica_availability, scenario,
    select_restore_checkpoint, serving_capacity, track_id, training_exposure,
):
    a_base = training_exposure(track_id, workers=scenario["worker_sweep_baseline"])
    a_result = training_exposure(track_id, workers=a_workers.value)
    a_sweep = [
        training_exposure(track_id, workers=workers)
        for workers in sorted({scenario["worker_sweep_baseline"], a_workers.value, scenario["worker_max"]})
    ]
    b_base = checkpoint_tradeoff(track_id, interval_min=scenario["checkpoint_interval_min"])
    b_result = checkpoint_tradeoff(track_id, interval_min=b_interval.value)
    b_sweep = [checkpoint_tradeoff(track_id, interval_min=interval) for interval in range(5, 181, 5)]
    c_fixture = checkpoint_fault_fixture(track_id, fault=c_fault.value)
    c_baseline_records = c_fixture["baseline_records"]
    c_result_records = c_fixture["result_records"]
    c_base = c_fixture["baseline"]
    c_result = c_fixture["result"]
    d_capacity = serving_capacity(track_id)
    d_colocated = replica_availability(
        replicas=d_replicas.value, placement_domains=1,
        component_availability=d_capacity["component_availability"],
        domain_mttf_h=scenario["serving_domain_mttf_h"],
        domain_repair_min=scenario["serving_domain_repair_min"],
    )
    d_spread = replica_availability(
        replicas=d_replicas.value, placement_domains=d_replicas.value,
        component_availability=d_capacity["component_availability"],
        domain_mttf_h=scenario["serving_domain_mttf_h"],
        domain_repair_min=scenario["serving_domain_repair_min"],
    )
    e_base = checkpoint_tradeoff(track_id, state_scale=e_state_scale.value)
    e_write = checkpoint_tradeoff(
        track_id, state_scale=e_state_scale.value,
        bandwidth_scale=e_bandwidth_scale.value, bandwidth_allocation="write",
    )
    e_restore = checkpoint_tradeoff(
        track_id, state_scale=e_state_scale.value,
        bandwidth_scale=e_bandwidth_scale.value, bandwidth_allocation="restore",
    )
    return a_base, a_result, a_sweep, b_base, b_result, b_sweep, c_base, c_baseline_records, c_fixture, c_result, c_result_records, d_capacity, d_colocated, d_spread, e_base, e_restore, e_write


@app.cell
def _(
    a_base, a_prediction, a_result, a_workers, b_base, b_interval,
    b_prediction, b_result, c_base, c_fault, c_prediction, c_result,
    capture_evidence, d_colocated, d_decision, d_prediction, d_replicas,
    d_spread, e_bandwidth_scale, e_base, e_decision, e_prediction,
    e_rejected, e_restore, e_state_scale, e_write, mo, set_evidence, track_id,
    f_upstream, f_base, f_result, f_prediction, g_upstream, g_base, g_result, g_prediction,
    h_upstream, h_base, h_result, h_prediction,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"workers": a_workers.value, "fault_trace": f_upstream}
    b_upstream = {"interval_min": b_interval.value}
    c_upstream = {"fault": c_fault.value, "elastic_trace": h_upstream}
    d_upstream = {"replicas": d_replicas.value, "decision": d_decision.value}
    e_upstream = {
        "state_scale": e_state_scale.value, "bandwidth_scale": e_bandwidth_scale.value,
        "decision": e_decision.value, "rejected": e_rejected.value, "gray_trace": g_upstream,
    }
    a_capture = mo.ui.button(
        label="Capture interruption contrast", kind="success",
        disabled=a_prediction.value is None or f_prediction.value is None or a_base["inputs"] == a_result["inputs"],
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction={"exposure": a_prediction.value, "verification": f_prediction.value},
            inputs=a_upstream, baseline={**a_base, "fault_trace": f_base, "inputs": {"exposure": a_base["inputs"], "fault_trace": f_base["inputs"]}}, result={**a_result, "fault_trace": f_result, "inputs": {"exposure": a_result["inputs"], "fault_trace": f_result["inputs"]}},
            upstream_inputs=a_upstream,
            model_key="v2_07_experiments.training_exposure + fault_verification_trace",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture checkpoint contrast", kind="success",
        disabled=b_prediction.value is None or b_base["inputs"] == b_result["inputs"],
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs=b_upstream, baseline=b_base, result=b_result,
            upstream_inputs=b_upstream,
            model_key="v2_07_experiments.checkpoint_tradeoff",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture restore decision", kind="success", disabled=c_prediction.value is None or h_prediction.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction={"restore": c_prediction.value, "elastic": h_prediction.value},
            inputs=c_upstream, baseline={**c_base, "elastic_trace": h_base, "inputs": {"checkpoint": c_base["inputs"], "elastic_trace": h_base["inputs"]}}, result={**c_result, "elastic_trace": h_result, "inputs": {"checkpoint": c_result["inputs"], "elastic_trace": h_result["inputs"]}},
            upstream_inputs=c_upstream, decision=c_prediction.value,
            model_key="v2_07_experiments.select_restore_checkpoint + elastic_recovery_trace",
        )),
    )
    d_capture = mo.ui.button(
        label="Capture placement decision", kind="success",
        disabled=d_prediction.value is None or d_decision.value is None,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs=d_upstream, baseline=d_colocated, result=d_spread,
            upstream_inputs=d_upstream, alternatives=(d_colocated, d_spread),
            decision=d_decision.value,
            model_key="v2_07_experiments.replica_availability",
        )),
    )
    e_invalid = (
        e_prediction.value is None or g_prediction.value is None or e_decision.value is None
        or e_rejected.value is None or e_decision.value == e_rejected.value
    )
    if e_decision.value == "write":
        e_selected = e_write
    elif e_decision.value == "restore":
        e_selected = e_restore
    else:
        e_selected = e_write if e_rejected.value == "write" else e_restore
    e_chosen_result = e_base if e_decision.value == "none" else None
    e_result_role = "rejected alternative" if e_decision.value == "none" else "tested intervention"
    e_capture = mo.ui.button(
        label="Capture recovery allocation", kind="success", disabled=e_invalid,
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction={"bandwidth": e_prediction.value, "gray_detection": g_prediction.value},
            inputs=e_upstream, baseline={**e_base, "gray_trace": g_base, "inputs": {"recovery": e_base["inputs"], "gray_trace": g_base["inputs"]}}, result={**e_selected, "gray_trace": g_result, "inputs": {"recovery": e_selected["inputs"], "gray_trace": g_result["inputs"]}},
            upstream_inputs=e_upstream, alternatives=(e_write, e_restore),
            decision=e_decision.value,
            model_key="v2_07_experiments.checkpoint_tradeoff",
            chosen_result=e_chosen_result, result_role=e_result_role,
        )),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream, e_capture, e_upstream


@app.cell
def _(mo, track_id):
    _track_key = track_id
    f_persistence = mo.ui.dropdown({"Transient": "transient", "Intermittent": "intermittent", "Permanent": "permanent"}, value="Transient", label="Fault persistence")
    f_manifestation = mo.ui.dropdown({"Wrong arithmetic value": "value_corrupting", "Slow but alive": "degraded", "Fail-stop": "fail_stop"}, value="Wrong arithmetic value", label="Fault manifestation")
    f_scope = mo.ui.dropdown({"One rank": "rank", "Shared rack": "rack"}, value="One rank", label="Fault scope")
    f_verification = mo.ui.dropdown({"Recompute rank 0 only": "rank0", "Recompute every rank": "all"}, value="Recompute every rank", label="Arithmetic verification coverage")
    f_prediction = mo.ui.radio({"CRC prevents wrong accepted arithmetic": "crc", "Unchecked arithmetic can escape a valid CRC": "escape"}, label="Will valid transport checks establish numerical correctness?").form(submit_button_label="Lock fault-verification prediction")
    g_incident = mo.ui.dropdown({"Slow but alive": "gray", "Healthy control": "healthy", "Stopped rank": "crash"}, value="Slow but alive", label="Progress incident")
    g_threshold = mo.ui.slider(0.5, 10.0, value=2.0, step=0.5, show_value=True, label="Progress alarm threshold (synthetic time units)")
    g_prediction = mo.ui.radio({"Heartbeats detect every stall": "heartbeat", "A live rank can still slow the barrier": "progress"}, label="What can heartbeat-only monitoring miss?").form(submit_button_label="Lock progress-monitor prediction")
    h_batch = mo.ui.checkbox(value=True, label="Increase accumulation to preserve effective batch")
    h_cursor = mo.ui.dropdown({"Resume at saved sample cursor": "exact", "Rewind cursor": "rewind", "Skip ahead": "skip"}, value="Resume at saved sample cursor", label="Resumed data coverage")
    h_state = mo.ui.checkbox(value=True, label="Reconstruct every lost optimizer shard")
    h_fence = mo.ui.checkbox(value=True, label="Reject old writer epochs")
    h_prediction = mo.ui.radio({"Live survivors are enough": "alive", "Batch, data, state and writer invariants all matter": "invariants"}, label="What permits this resized run to resume safely?").form(submit_button_label="Lock elastic-recovery prediction")
    return f_persistence, f_manifestation, f_scope, f_verification, f_prediction, g_incident, g_threshold, g_prediction, h_batch, h_cursor, h_state, h_fence, h_prediction


@app.cell
def _(fault_verification_trace, gray_failure_trace, elastic_recovery_trace,
       f_persistence, f_manifestation, f_scope, f_verification, g_incident, g_threshold,
       h_batch, h_cursor, h_state, h_fence):
    f_upstream = {"persistence": f_persistence.value, "manifestation": f_manifestation.value, "scope": f_scope.value, "verification": f_verification.value}
    f_base = fault_verification_trace(**{**f_upstream, "verification": "none"})
    f_result = fault_verification_trace(**f_upstream)
    g_upstream = {"detector": "progress", "threshold": g_threshold.value, "incident": g_incident.value}
    g_base = gray_failure_trace(**{**g_upstream, "detector": "heartbeat"})
    g_result = gray_failure_trace(**g_upstream)
    h_upstream = {"preserve_batch": h_batch.value, "data_cursor": h_cursor.value, "reconstruct_state": h_state.value, "fence_epoch": h_fence.value}
    h_base = elastic_recovery_trace(preserve_batch=False, data_cursor="rewind", reconstruct_state=False, fence_epoch=False)
    h_result = elastic_recovery_trace(**h_upstream)
    return f_upstream, f_base, f_result, g_upstream, g_base, g_result, h_upstream, h_base, h_result


@app.cell(hide_code=True)
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, scenario, track):
    from mlsysbook_labs.notebook import (
        notebook_header as _course_header, notebook_shell as _course_shell,
        notebook_learning_map as _course_learning_map,
        notebook_scenario as _course_scenario,
    )
    from mlsysbook_labs.learning_paths import get_learning_path as _course_path
    from mlsysbook_labs import get_lab_metadata as _course_metadata
    css = mo.Html("""
    <style>




    .ft-saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-active{color:#86efac}.lab-hud .hud-error{color:#fca5a5;font-weight:700}

    </style>""")
    header = _course_header(_course_metadata("vol2/lab_07_fault_tolerance.py"), 'How should a fleet preserve useful work when independent faults and shared failure domains are part of normal operation?')
    _course_shell([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        css,
        header,
        track,
        _course_scenario(_course_path('vol2', 7), track=track.value, volume=2),
        mo.callout(mo.md(f"**Scenario assumptions ({scenario['label']}).** Training runs on {scenario['training_backend_role']}. Track sizes, failure domains, state footprints, and service repair times are illustrative fleet-planning inputs. Replace them with observed fleet rates before operational use."), kind="info"),
        _course_learning_map(_course_path('vol2', 7))
    ], volume=2)
    return


@app.cell(hide_code=True)
def _(
    COLORS, a_base, a_capture, a_prediction, a_result, a_sweep,
    a_upstream, a_workers, apply_plotly_theme, audit_evidence, b_base,
    b_capture, b_interval, b_prediction, b_result, b_sweep, b_upstream,
    c_base, c_capture, c_fault, c_prediction, c_result, c_upstream,
    d_capacity, d_capture, d_colocated, d_decision, d_prediction,
    d_replicas, d_spread, d_upstream, e_bandwidth_scale, e_base,
    e_capture, e_decision, e_prediction, e_rejected, e_restore,
    e_state_scale, e_upstream, e_write, final_choice, final_rejected,
    final_risk, final_trigger, get_evidence, go, mo, rationale, scenario,
    track_id, f_persistence, f_manifestation, f_scope, f_verification, f_prediction,
    f_base, f_result, f_upstream, g_incident, g_threshold, g_prediction,
    g_base, g_result, g_upstream, h_batch, h_cursor, h_state, h_fence,
    h_prediction, h_base, h_result, h_upstream,
):
    _saved_prediction_widgets = {'A': a_prediction, 'B': b_prediction, 'C': c_prediction, 'D': d_prediction, 'E': e_prediction}
    from mlsysbook_labs.notebook import notebook_prediction_label as _read_prediction
    from mlsysbook_labs.notebook import notebook_tab_hint as _part_tabs_hint, notebook_shell as _lesson_shell
    from mlsysbook_labs.notebook import notebook_part_reflection as _concept_reflection
    from mlsysbook_labs.learning_paths import get_learning_path as _reflection_path
    _reflection_contract = _reflection_path('vol2', 7)
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=_upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def prediction_label(part, value):
        probe_widgets = {
            "A": {"exposure": a_prediction, "verification": f_prediction},
            "C": {"restore": c_prediction, "elastic": h_prediction},
            "E": {"bandwidth": e_prediction, "gray_detection": g_prediction},
        }
        if isinstance(value, dict):
            return "; ".join(f"{key.replace('_', ' ')}: {_read_prediction(probe_widgets[part][key], value=answer)}" for key, answer in value.items())
        return _read_prediction(_saved_prediction_widgets[part], value=value)

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture after changing this part's conditions."), kind="danger")
        data = capture.to_dict()
        from html import escape
        return mo.Html(f'<div class="ft-saved"><b>Saved snapshot</b> · original prediction: {escape(prediction_label(part, data["prediction"]))}<br><small>Track {data["track"]}; later control changes do not rewrite this record.</small></div>')

    def pass_fail(value):
        return "PASS" if value else "FAIL"

    def part_a():
        context = mo.md(f"### A · When does a rare fault become routine? (9 min)\nA coupled training job runs for **{a_base['duration_h']:.0f} hours**. Any worker event or shared-domain event interrupts the whole job. Choose the fleet size, then commit a probability range.")
        if a_prediction.value is None:
            return mo.vstack([context, a_workers, a_prediction])
        figure = go.Figure(go.Scatter(
            x=[row["workers"] for row in a_sweep],
            y=[row["interruption_probability"] * 100 for row in a_sweep],
            mode="lines+markers", line=dict(color=COLORS["RedLine"], width=3),
        ))
        figure.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), xaxis_title=scenario["training_unit"].title(), yaxis_title="Interruption probability (%)")
        rows = [
            {"Run": "Smaller baseline", "Workers": a_base["workers"], "System MTBF": f"{a_base['system_mtbf_h']:.1f} h", "Expected events": f"{a_base['expected_interruptions']:.2f}", "Any event": f"{a_base['interruption_probability'] * 100:.1f}%"},
            {"Run": "Selected fleet", "Workers": a_result["workers"], "System MTBF": f"{a_result['system_mtbf_h']:.1f} h", "Expected events": f"{a_result['expected_interruptions']:.2f}", "Any event": f"{a_result['interruption_probability'] * 100:.1f}%"},
        ]
        return mo.vstack([mo.vstack([context, a_workers, a_prediction, apply_plotly_theme(figure), table(rows), mo.callout(mo.md(f"**Your prediction:** {_read_prediction(a_prediction)}. The selected coupled job has a **{a_result['interruption_probability'] * 100:.1f}%** interruption probability. Worker and failure-domain event rates are added because either event stops this synchronized job."), kind="danger" if a_result["interruption_probability"] >= 0.5 else "info"), part_f(), a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("For independent exponential event streams, worker rate is N/MTTF and domain rate is D/MTTF_domain. Their sum gives job interruption rate; survival over T hours is exponential. This model applies to a coupled job, not independent deployed devices.")})]), _concept_reflection(_reflection_contract, "A")])

    def part_b():
        context = mo.md(f"### B · How often should progress be saved? (10 min)\nThe current policy checkpoints every **{b_base['interval_min']:.0f} minutes**. Move the interval, then predict whether pause overhead or expected rework dominates.")
        if b_prediction.value is None:
            return mo.vstack([context, b_interval, b_prediction])
        valid = [row for row in b_sweep if row["loss_approximation_valid"]]
        figure = go.Figure()
        figure.add_scatter(x=[row["interval_min"] for row in valid], y=[row["write_tax"] * 100 for row in valid], name="Checkpoint pause", line=dict(color=COLORS["BlueLine"]))
        figure.add_scatter(x=[row["interval_min"] for row in valid], y=[row["rework_tax"] * 100 for row in valid], name="Expected rework", line=dict(color=COLORS["OrangeLine"]))
        figure.add_scatter(x=[row["interval_min"] for row in valid], y=[row["total_waste_fraction"] * 100 for row in valid], name="Total first-order loss", line=dict(color=COLORS["RedLine"], width=3))
        figure.add_vline(x=b_result["requested_rpo_min"], line_dash="dash", annotation_text="Maximum rollback", annotation_position="top right")
        figure.add_vline(x=b_result["interval_min"], line_color=COLORS["BlueLine"], line_dash="dot")
        if b_result["loss_approximation_valid"]:
            figure.add_scatter(
                x=[b_result["interval_min"]], y=[b_result["total_waste_fraction"] * 100],
                name="Selected interval", mode="markers",
                marker=dict(size=14, symbol="diamond", color=COLORS["GreenLine"] if b_result["rpo_ok"] else COLORS["RedLine"]),
                hovertemplate="Interval: %{x:.0f} min<br>Total loss: %{y:.2f}%<extra>Selected interval</extra>",
            )
        figure.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Checkpoint interval (min)", xaxis=dict(range=[0, b_sweep[-1]["interval_min"]]), yaxis_title="Steady-state wall-clock loss (%)", legend=dict(orientation="h", y=-0.3))
        challenge = mo.md("**Find a low-loss interval that respects the rollback limit.** Shorten the interval until checkpoint writes dominate, then lengthen it until rework dominates. The selected point must stay at or left of the maximum rollback line. An unconstrained minimum can still violate the recovery contract.")
        opt_base = f"{b_base['optimal_interval_min']:.1f} min" if b_base["young_approximation_valid"] and b_base["optimal_interval_min"] is not None else "—"
        opt_res = f"{b_result['optimal_interval_min']:.1f} min" if b_result["young_approximation_valid"] and b_result["optimal_interval_min"] is not None else "—"
        rows = [
            {"Run": "Current", "Interval": f"{b_base['interval_min']:.0f} min", "Optimum": opt_base, "Write": f"{b_base['write_tax'] * 100:.2f}%", "Rework": f"{b_base['rework_tax'] * 100:.2f}%", "Recovery": f"{b_base['recovery_tax'] * 100:.2f}%", "RPO": pass_fail(b_base["rpo_ok"])},
            {"Run": "Selected", "Interval": f"{b_result['interval_min']:.0f} min", "Optimum": opt_res, "Write": f"{b_result['write_tax'] * 100:.2f}%", "Rework": f"{b_result['rework_tax'] * 100:.2f}%", "Recovery": f"{b_result['recovery_tax'] * 100:.2f}%", "RPO": pass_fail(b_result["rpo_ok"])},
        ]
        validity = "inside" if b_result["loss_approximation_valid"] else "outside"
        opt_text = (
            f" Engine Young-Daly optimum is **{b_result['optimal_interval_min']:.1f} min**."
            if b_result["young_approximation_valid"] and b_result["optimal_interval_min"] is not None
            else " Engine Young optimum is withheld outside validity domain."
        )
        return mo.vstack([mo.vstack([context, b_interval, b_prediction, challenge, apply_plotly_theme(figure), table(rows), mo.callout(mo.md(f"**Your prediction:** {_read_prediction(b_prediction)}. At **{b_result['interval_min']:.0f} min**, **{b_result['dominant_interval_cost'].replace('_', ' ')}** dominates.{opt_text} This setting is **{validity}** the declared first-order approximation domain. **Rollback limit: {'PASS' if b_result['rpo_ok'] else 'FAIL'}**; at most **{b_result['requested_rpo_min']:.0f} min** may be lost."), kind="info" if b_result["loss_approximation_valid"] and b_result["rpo_ok"] else "danger"), b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md("On a steady-state wall-clock basis: checkpoint pause/interval + interval/(2×MTBF) + recovery/MTBF. RPO uses the maximum rollback window, equal to the full interval. The Young optimum is shown only while checkpoint pause remains small relative to MTBF; invalid settings do not receive a goodput claim. The selected point is green when the rollback limit is satisfied and red otherwise. Outside the approximation domain, only the selected interval line is shown; no loss estimate is plotted.")})]), _concept_reflection(_reflection_contract, "B")])

    def part_c():
        context = mo.md(
            f"### C · Is the newest checkpoint recoverable? (9 min)\n"
            f"**Training backend:** {scenario['training_backend_role']}. {scenario['manifest_context']}\n\n"
            f"A manifest publishes generations at steps 100, 200, 300, and 400. "
            f"Inject one fault and commit the generation recovery may load. Every required shard must name the same training step and the manifest must be published only after all shards are complete."
        )
        if c_prediction.value is None:
            return mo.vstack([context, c_fault, c_prediction])
        rows = [
            {"Timeline": "Before fault", "Newest recorded": c_base["newest_checkpoint_step"], "Restore step": c_base["restore_step"], "Lost steps": c_base["lost_steps"]},
            {"Timeline": "After fault", "Newest recorded": c_result["newest_checkpoint_step"], "Restore step": c_result["restore_step"] if c_result["recoverable"] else "NONE", "Lost steps": c_result["lost_steps"] if c_result["recoverable"] else "UNBOUNDED"},
        ]
        correct = c_prediction.value == (c_result["restore_step"] or 0)
        return mo.vstack([mo.vstack([context, c_fault, c_prediction, table(rows), table(c_result["generation_checks"]), mo.md(f"**Newest shard steps:** {c_result['inputs']['checkpoints'][-1].get('shard_steps')}. **Expected shards:** {c_result['inputs']['checkpoints'][-1].get('expected_shards')}."), mo.callout(mo.md(f"**Your prediction:** step {_read_prediction(c_prediction)}. **Safe restore:** {c_result['restore_step'] if c_result['recoverable'] else 'none'}. Recovery skips incomplete, digest-invalid, mixed-step, missing-shard or unpublished generations even when their step number is newer; this fault discards **{c_result['fallback_generations']}** newer generation(s)."), kind="success" if correct else "danger"), part_h(), c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md("Recovery selects the greatest step at or before failure whose write completed, whose recorded digest validates, whose complete shard set agrees on the step, and whose manifest has been published. A manifest publication flag represents the final commit in this supplied trace; no consensus protocol is implemented. The newest directory name alone is not a commit record. Checkpoint manifests record backend training state generations for restoring interrupted training, not validated deployment-release binaries; deployed endpoints do not write or consume training checkpoint manifests.")})]), _concept_reflection(_reflection_contract, "C")])

    def part_d():
        context = mo.md(
            f"### D · Does redundancy survive a shared domain? (10 min)\n"
            f"Deployed {scenario['serving_unit']} serve independently: intermittent disconnections or restarts temporarily reduce active capacity, but do not terminate a global job (unlike coupled training in Parts A–C). "
            f"Separately, replicated {scenario['replica_role']} instances handle traffic. "
            f"Compare **{d_replicas.value} {scenario['replica_role']} replicas** in one domain with the same replicas spread across domains."
        )
        if d_prediction.value is None:
            return mo.vstack([context, d_replicas, d_prediction])
        rows = [
            {"Placement": "One shared domain", "Replicas/domain": str(d_colocated["replicas_per_domain"]), "Service availability": f"{d_colocated['service_availability'] * 100:.6f}%", "Shared-domain correlation": "YES"},
            {"Placement": "Spread domains", "Replicas/domain": str(d_spread["replicas_per_domain"]), "Service availability": f"{d_spread['service_availability'] * 100:.6f}%", "Shared-domain correlation": "NO"},
        ]
        capacity_rows = [{f"Deployed {scenario['serving_unit']}": d_capacity["fleet_size"], "Expected available": f"{d_capacity['expected_available_devices']:.1f}", "Expected unavailable": f"{d_capacity['expected_unavailable_devices']:.1f}", "Whole fleet terminated": "NO"}]
        return mo.vstack([mo.vstack([context, d_replicas, d_prediction, table(capacity_rows), table(rows), d_decision, mo.callout(mo.md(f"**Your prediction:** {_read_prediction(d_prediction)}. Spread placement raises modeled service availability from **{d_colocated['service_availability'] * 100:.6f}%** to **{d_spread['service_availability'] * 100:.6f}%** because a shared domain event can defeat every colocated replica."), kind="info"), d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md(f"Serving repair time represents an illustrative automated recovery assumption ({scenario['recovery_assumption']}), not a measured manual field repair. Deployed {scenario['serving_unit']} operate independently from coupled training (Parts A–C), whereas the {scenario['replica_role']} tier uses replicated service domains. For one device or component, availability is MTTF/(MTTF+repair). A replica group is unavailable if its domain is down or every replica in that domain is down. Service is unavailable only when every placement domain is unavailable.")})]), _concept_reflection(_reflection_contract, "D")])

    def part_e():
        context = mo.md("### E · Where should recovery bandwidth go? (10 min)\nStress the recovery state, then allocate the same added GB/s to checkpoint writes or restore reads. Both candidates start from the same failing configuration.")
        if e_prediction.value is None:
            return mo.vstack([context, mo.hstack([e_state_scale, e_bandwidth_scale], widths="equal", wrap=True), e_prediction])
        rows = [
            {"Case": "Stressed baseline", "Added BW": "0 GB/s", "Write tax": f"{e_base['write_tax'] * 100:.2f}%", "Recovery": f"{e_base['recovery_s']:.0f}/{e_base['requested_rto_s']:.0f} s", "RTO": pass_fail(e_base["rto_ok"])},
            {"Case": "Allocate to writes", "Added BW": f"{e_write['additional_bandwidth_gbs']:.2f} GB/s", "Write tax": f"{e_write['write_tax'] * 100:.2f}%", "Recovery": f"{e_write['recovery_s']:.0f}/{e_write['requested_rto_s']:.0f} s", "RTO": pass_fail(e_write["rto_ok"])},
            {"Case": "Allocate to restore", "Added BW": f"{e_restore['additional_bandwidth_gbs']:.2f} GB/s", "Write tax": f"{e_restore['write_tax'] * 100:.2f}%", "Recovery": f"{e_restore['recovery_s']:.0f}/{e_restore['requested_rto_s']:.0f} s", "RTO": pass_fail(e_restore["rto_ok"])},
        ]
        return mo.vstack([mo.vstack([context, mo.hstack([e_state_scale, e_bandwidth_scale], widths="equal", wrap=True), e_prediction, table(rows), mo.hstack([e_decision, e_rejected], widths="equal", wrap=True), mo.callout(mo.md(f"**Your prediction:** {_read_prediction(e_prediction)}. Write allocation changes checkpoint pause but leaves recovery at **{e_write['recovery_s']:.0f} s**. Restore allocation changes load time and reaches **{e_restore['recovery_s']:.0f} s**. A no-action decision remains recordable after you identify a tested alternative."), kind="danger" if not e_restore["rto_ok"] else "success"), part_g(), e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md("The added bandwidth quantity is identical in both candidates. Write allocation reduces durable-write time and checkpoint pause. Restore allocation reduces only the load phase. Detection, restart, warmup, interval, RPO, and failure exposure remain fixed.")})]), _concept_reflection(_reflection_contract, "E")])

    def part_f():
        context = mo.md("#### Classify before responding\nA backend rank sends a correctly checksummed contribution. Change persistence, manifestation and scope independently, then compare selective arithmetic recomputation with no recomputation. A digest protects the supplied bytes; it does not establish that the producer computed them correctly.")
        controls = mo.hstack([f_persistence, f_manifestation, f_scope, f_verification], wrap=True)
        rows = [{"Attempt": row["attempt"], "Fault active": row["fault_active"], "All alive": all(row["heartbeat_alive"]), "CRC valid": row["transport_crc_valid"], "Reduced mean": row["reduced_mean"], "Mismatch ranks": str(row["arithmetic_mismatches"]), "Accepted wrong": row["accepted_wrong_value"]} for row in f_result["attempts"]]
        feedback = mo.callout(mo.md(f"**Reference mean:** {f_result['reference_mean']}. **Checked ranks:** {f_result['attempts'][0]['checked_ranks']}. **First attempt accepts a wrong value:** {f_result['attempts'][0]['accepted_wrong_value']}. Recomputations across the supplied attempts: **{f_result['verification_recomputations']}** versus **{f_base['verification_recomputations']}** without verification. Persistence suggests **{f_result['retry_response']}**. For the value-corrupting injection, rank-0-only checks miss the faulty contribution and all-rank checks reject it. Recurrent faults can return after a successful retry. No rate or detection probability is estimated."), kind="info")
        return mo.vstack([context, controls, f_prediction, table(rows), feedback])

    def part_g():
        context = mo.md("#### Alive is not making progress\nThe same synchronized group now has a rank that answers heartbeats. Compare heartbeat-only monitoring with a per-rank progress threshold. Lower the threshold until healthy ranks also trigger an alarm, then test the healthy control and stopped-rank trace.")
        rows = [{"Monitor": label, "All heartbeats alive": all(result["heartbeat_alive"]), "Barrier units": result["barrier_step_units"], "Suspected ranks": str(result["suspected_ranks"]), "Healthy ranks falsely flagged": str(result["false_positive_ranks"]), "First response": result["response"]} for label, result in [("Heartbeat only", g_base), ("Progress and heartbeat", g_result)]]
        return mo.vstack([context, mo.hstack([g_incident, g_threshold], wrap=True), g_prediction, table(rows), mo.callout(mo.md(f"**Selected response:** {g_result['response']}. **Healthy ranks flagged:** {g_result['false_positive_ranks']}. The barrier waits for the slowest rank. Mitigating a live straggler and restoring a stopped rank are different responses. An aggressive threshold can flag healthy work; these exact traces do not establish a field false-alarm rate."), kind="info")])

    def part_h():
        context = mo.md("#### Resume with fewer workers\nTwo workers survive a four-worker job. The naive recovery leaves accumulation unchanged, rewinds data, lacks lost optimizer shards and accepts an old writer. Repair each obligation, then break one at a time. Preserving batch is the tested policy; an alternative learning-rate policy would require separate validation.")
        rows = [{"Gate": gate.replace("_", " "), "Naive recovery": pass_fail(h_base["gates"][gate]), "Selected recovery": pass_fail(value)} for gate, value in h_result["gates"].items()]
        details = [{"Quantity": key.replace("_", " "), "Before resize": h_result["before"][key], "After resize": h_result["after"][key]} for key in ("workers", "microbatch", "accumulation", "effective_batch")]
        state_rows = [{"Required state": shard, "Reconstructable": shard in h_result["available_state"]} for shard in h_result["required_state"]]
        return mo.vstack([context, mo.hstack([h_batch, h_cursor, h_state, h_fence], wrap=True), h_prediction, table(rows), table(details), table(state_rows), table(h_result["writer_attempts"]), mo.md(f"**Duplicate sample IDs:** {h_result['duplicate_samples']}. **Missing sample IDs:** {h_result['missing_samples']}."), mo.callout(mo.md(f"**Safe to resume in this trace: {pass_fail(h_result['safe_to_resume'])}.** Passing these necessary consistency gates does not establish numerical convergence or measured throughput. Old-epoch writes must be fenced even when surviving workers are alive."), kind="success" if h_result["safe_to_resume"] else "danger")])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            state = "MISSING"
            if capture:
                state = "STALE" if part in audit.stale or (part, part) in audit.identical_pairs else "CURRENT"
            rows.append({"Part": part, "Original prediction": prediction_label(part, capture.to_dict()["prediction"]) if capture else "—", "Evidence": state})
        saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
        complete = (
            audit.complete
            and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
            and bool(rationale.value.strip()) and final_choice.value != final_rejected.value
            and final_choice.value == saved_decision
        )
        return mo.vstack([mo.md("### Synthesis · Defend the recovery allocation (5 min)\nChoose one option, quantify the rejected alternative from Part E, state a remaining limitation, and name the condition that forces reevaluation."), table(rows), mo.callout(mo.md("Saved snapshots retain the original prediction, exact simulator arguments, and observed comparison. Recapture stale evidence before producing the report."), kind="info"), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale, mo.callout(mo.md("**Ready for the local report.**" if complete else "Complete five current contrasts, match the recommendation to saved Part E, reject a different tested option, and add the rationale."), kind="success" if complete else "warn")])

    tabs = mo.ui.tabs({"Part A": part_a(), "Part B": part_b(), "Part C": part_c(), "Part D": part_d(), "Part E": part_e(), "Synthesis": build_synthesis()})
    _lesson_shell([_part_tabs_hint(), tabs], volume=2)
    return (audit,)


@app.cell
def _():
    from mlsysbook_labs.notebook import notebook_shell as _submit_shell, notebook_submission as _submit_guide
    from mlsysbook_labs.learning_paths import get_learning_path as _submit_path
    _submit_shell([_submit_guide(_submit_path(2, 7))], volume=2)
    return


@app.cell(hide_code=True)
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, rationale,
    report_export_panel, scenario, track_id,
):
    from mlsysbook_labs.notebook import notebook_shell as _workspace_shell
    _captures = get_evidence()
    _saved_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip()) and final_choice.value != final_rejected.value
        and final_choice.value == _saved_decision
    )
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not _ready, _download_readiness(volume=2))
    _snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    _chosen_e = _snapshots["E"]["chosen_result"] or _snapshots["E"]["result"]
    report = build_lab_report(
        get_lab_metadata("vol2/lab_07_fault_tolerance.py"),
        track=track_id, scenario=scenario["label"],
        learning_objectives=[
            "Calculate interruption exposure for a coupled fleet job",
            "Balance checkpoint pause, rework, and recovery under RPO and RTO",
            "Select valid checkpoint generations and failure-domain placement",
            "Distinguish fault axes, arithmetic validation, gray failures and elastic recovery invariants",
        ],
        predictions={part: _snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: _snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": _snapshots[part]["baseline"], "result": _snapshots[part]["result"], "chosen_result": _snapshots[part]["chosen_result"], "result_role": _snapshots[part]["result_role"], "alternatives": _snapshots[part]["alternatives"]} for part in "ABCDE"},
        binding_constraints={
            "B": {"rpo_ok": _snapshots["B"]["result"]["rpo_ok"], "validity_issues": _snapshots["B"]["result"]["validity_issues"]},
            "C": {"recoverable": _snapshots["C"]["result"]["recoverable"], "restore_step": _snapshots["C"]["result"]["restore_step"]},
            "E": {"rto_ok": _chosen_e["rto_ok"], "recovery_s": _chosen_e["recovery_s"]},
        },
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=[
            "Coupled jobs and independent serving fleets have different failure semantics.",
            "Checkpoint frequency trades write pause against expected rework.",
            "Recovery trusts committed, validated generations and independent failure domains.",
        ],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": _snapshots, "recommendation": final_choice.value, "rejected": final_rejected.value, "trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": "Illustrative fleet-planning assumptions", "calculations": "MLSysIM fault-tolerance experiments"},
    )
    _workspace_shell([mo.vstack([mo.md("### Download your discussion"), report_export_panel(report)])], volume=2)
    return (report,)


@app.cell
def _(
    audit, final_choice, final_rejected, final_risk, final_trigger, get_evidence, mo, rationale, track_id
):
    from mlsysbook_labs.fleet_evidence import discussion_fingerprint as _discussion_fingerprint
    _captures = get_evidence()
    _saved_decision = _captures['E'].to_dict()['decision'] if 'E' in _captures else None
    _ready = audit.complete and all((widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))) and bool(rationale.value.strip()) and (final_choice.value != final_rejected.value) and (final_choice.value == _saved_decision)
    fleet_discussion_ready = _ready
    fleet_discussion_design = {'schema_version': 1, 'lab_id': 'v2_07', 'track_id': track_id, 'model_id': 'v2_07_experiments', 'evidence': {part: capture.to_dict() for part, capture in _captures.items()}, 'recommendation': final_choice.value, 'rejected_alternative': final_rejected.value, 'reevaluation_trigger': final_trigger.value, 'residual_risk': final_risk.value, 'rationale': rationale.value} if _ready else None
    fleet_discussion_fingerprint = _discussion_fingerprint(fleet_discussion_design)
    # Recreated when any evidence, track, or discussion field changes.
    save_discussion = mo.ui.run_button(label="Save discussion", disabled=not fleet_discussion_ready)
    return (fleet_discussion_ready, fleet_discussion_design, fleet_discussion_fingerprint, save_discussion)


@app.cell
async def _(fleet_discussion_design, fleet_discussion_fingerprint, fleet_discussion_ready,
            ledger, mo, save_discussion, track_id):
    from mlsysbook_labs.fleet_evidence import save_completed_discussion as _save_completed_discussion
    from mlsysbook_labs.notebook import notebook_shell as _discussion_shell, notebook_status as _notebook_status
    _outcome = await _save_completed_discussion(
        ledger, chapter=7, track=track_id, design=fleet_discussion_design,
        fingerprint=fleet_discussion_fingerprint, ready=fleet_discussion_ready,
        requested=bool(save_discussion.value),
    )
    _discussion_shell([mo.vstack([save_discussion, _notebook_status(_outcome.status, _outcome.detail)])], volume=2)
    return


if __name__ == "__main__":
    app.run()
