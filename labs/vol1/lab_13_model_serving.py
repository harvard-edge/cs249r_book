import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 13: Serving Under Deadlines · MLSysBook")


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
    from mlsysim.labs.experiments.v1_13_experiments import (
        LANGUAGE_CONTEXT_SCALES, MODEL_ID, REPLAY_METHOD,
        arrival_trace_for_track,
        batching_policy_for_track,
        get_serving_scenario, serving_path_cases,
        policy_for_track,
        simulation_result_to_snapshot,
        simulate_serving,
        state_capacity,
        state_capacity_to_snapshot,
        state_pressure_scenario,
    )
    from mlsysim.labs.experiments.v1_13_experiments import (
        get_recommendation_serving_scenario, get_language_serving_reference,
        language_scheduling_reference, language_scheduling_to_snapshot, language_batching_reference, recommendation_state_breakdown,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata, report_export_panel,
    )
    from mlsysim import Scenarios
    from mlsysbook_labs.learning_paths import get_learning_path
    from mlsysbook_labs.notebook import (
        notebook_bridge, notebook_header, notebook_learning_map, notebook_learning_part, notebook_tab_hint,
        notebook_shell, notebook_status, notebook_table,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence

    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (LANGUAGE_CONTEXT_SCALES, get_recommendation_serving_scenario, get_language_serving_reference,
            language_scheduling_reference, language_scheduling_to_snapshot, language_batching_reference, recommendation_state_breakdown, Scenarios, get_learning_path, notebook_bridge, notebook_header,
            notebook_learning_map, notebook_learning_part, notebook_tab_hint, notebook_shell, notebook_status, notebook_table,
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, MODEL_ID, REPLAY_METHOD, apply_plotly_theme,
        arrival_trace_for_track, audit_evidence, batching_policy_for_track,
        build_lab_report, capture_evidence, get_lab_metadata,
        get_serving_scenario, serving_path_cases, go, ledger, mo, policy_for_track,
        report_export_panel, simulation_result_to_snapshot, simulate_serving,
        state_capacity, state_capacity_to_snapshot, state_pressure_scenario,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(ledger, mo, set_evidence):
    from mlsysbook_labs.contexts import CONTEXT_OPTIONS as _options, get_context as _context, preferred_context as _preferred
    track = mo.ui.dropdown(
        _options,
        value=_context(_preferred(ledger)).label,
        label="Deployment track",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_recommendation_serving_scenario, get_serving_scenario, track):
    track_id = track.value
    scenario = get_recommendation_serving_scenario() if track_id == "cloud" else get_serving_scenario(track_id)
    return scenario, track_id


@app.cell
def _(LANGUAGE_CONTEXT_SCALES, mo, track_id, get_language_serving_reference, language_scheduling_reference,
      state_capacity, state_capacity_to_snapshot):
    _project_identity = track_id
    language_chunks = mo.ui.dropdown({"One prefill chunk": 1, "Two prefill chunks": 2, "Four prefill chunks": 4},
                                    value="Two prefill chunks", label="Language reference: prefill scheduling")
    language_history = mo.ui.dropdown(
        {f"{scale}× baseline history": scale for scale in LANGUAGE_CONTEXT_SCALES},
        value="2× baseline history", label="Retained history",
    )
    language_lengths = mo.ui.dropdown({"Equal responses": "equal", "Mixed short and long responses": "mixed"},
                                      value="Mixed short and long responses", label="Response lengths")
    language_arrivals = mo.ui.dropdown({"All arrive together": 0, "One every 16 ms": 16, "One every 160 ms": 160},
                                       value="All arrive together", label="Prepared-request arrivals")
    return language_chunks, language_history, language_lengths, language_arrivals


@app.cell
def _(language_batching_reference, language_lengths, language_arrivals, language_history):
    language_batching = {
        policy: language_batching_reference(policy, length_mix=language_lengths.value,
            arrival_spacing_ms=language_arrivals.value, context_scale=language_history.value)
        for policy in ("static", "continuous")
    }
    return (language_batching,)


@app.cell
def _(get_language_serving_reference, language_chunks, language_history, language_scheduling_reference,
      language_scheduling_to_snapshot, state_capacity, state_capacity_to_snapshot):
    language_schedule = language_scheduling_to_snapshot(language_scheduling_reference(chunks=language_chunks.value))
    language_base = language_scheduling_to_snapshot(language_scheduling_reference(chunks=1))
    language_capacity = state_capacity_to_snapshot(state_capacity(get_language_serving_reference()), pressure="baseline")
    language_capacity_trial = state_capacity_to_snapshot(
        state_capacity(get_language_serving_reference(context_scale=language_history.value)), pressure="baseline")
    return language_base, language_capacity, language_capacity_trial, language_schedule


@app.cell
def _(mo, track_id, scenario, serving_path_cases):
    _track_key = track_id
    a_cases = serving_path_cases(scenario)
    a_intervention = mo.ui.dropdown(
        {case.label: key for key, case in a_cases.items()},
        value=a_cases["host"].label, label="Trial path",
    )
    b_pattern = mo.ui.dropdown(
        {"Bursty replay": "bursty", "Demand shift": "shift"},
        value="Bursty replay",
        label="Changed traffic",
    )
    b_load = mo.ui.dropdown(
        {"1.0× phase rates": 1.0, "1.5× phase rates": 1.5, "2.0× phase rates": 2.0},
        value="1.5× phase rates",
        label="Offered-load scale",
    )
    c_policy = mo.ui.dropdown(
        {"Pair + short timeout": "pair-short", "Four + patient timeout": "quad-patient"},
        value="Four + patient timeout",
        label="Batch policy",
    )
    d_pressure = mo.ui.dropdown(
        {"Elevated live state": "elevated", "State exceeds memory": "overflow"},
        value="State exceeds memory",
        label="State pressure",
    )
    e_pattern = mo.ui.dropdown(
        {"Demand shift": "shift", "Bursty replay": "bursty"},
        value="Demand shift",
        label="Launch traffic",
    )
    e_candidate = mo.ui.dropdown(
        {"Reserve capacity": "reserve", "Delayed activation": "activate", "Bounded admission": "admit"},
        value="Reserve capacity",
        label="Candidate policy",
    )
    e_rejected = mo.ui.radio(
        {"Reserve capacity": "reserve", "Delayed activation": "activate", "Bounded admission": "admit"},
        label="Tested alternative",
    )
    e_conclusion = mo.ui.radio(
        {"Use the candidate": "candidate", "No change / hold for more evidence": "none"},
        label="Launch conclusion",
    )
    return a_cases, a_intervention, b_load, b_pattern, c_policy, d_pressure, e_candidate, e_conclusion, e_pattern, e_rejected


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {
            "The trial improves complete-path latency without losing useful service": "trial",
            "A contract or memory cost prevents the apparent improvement": "cost",
            "The changed stage leaves the service outcome unchanged": "equal",
        },
        label="Will this intervention improve the whole service?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {
            "P99 stays within the deadline": "pass",
            "P99 crosses the deadline": "tail",
            "Admission loss appears first": "rejection",
        },
        label="What fails first when the traffic pattern changes?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {
            "Waiting improves tail latency": "tail-improves",
            "Waiting reduces execution work but worsens tail latency": "tradeoff",
            "Waiting worsens tail latency without reducing execution work": "wait-only",
            "Batching changes neither schedule nor work": "no-change",
        },
        label="What will the selected batch policy change?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {
            "At least one request still fits": "fits",
            "No request state fits with the fixed memory": "fails",
        },
        label="What happens to the concurrency ceiling?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {
            "Candidate meets every deadline without rejection": "candidate-pass",
            "Alternative meets every deadline without rejection": "alternative-pass",
            "Neither tested policy is feasible": "neither",
        },
        label="Which policy survives the changed demand?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {
            "Reserve capacity": "reserve",
            "Delayed activation": "activate",
            "Bounded admission": "admit",
            "No change / hold for more evidence": "none",
        },
        label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {"Reserve capacity": "reserve", "Delayed activation": "activate", "Bounded admission": "admit"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {
            "Any deadline miss": "deadline",
            "Any admission rejection": "rejection",
            "Live state no longer fits": "state",
        },
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {
            "Illustrative service times need measurement": "service-time",
            "Future arrival shape may differ": "arrival-shape",
            "State size may grow": "state-growth",
        },
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Launch rationale",
        placeholder="Use saved quantities to connect the chosen policy, rejected alternative, limitation, and trigger.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_cases, a_intervention, get_recommendation_serving_scenario, arrival_trace_for_track, b_load, b_pattern, batching_policy_for_track, c_policy,
    d_pressure, e_candidate, e_pattern, e_rejected, policy_for_track,
    scenario, simulation_result_to_snapshot, simulate_serving, state_capacity,
    state_capacity_to_snapshot, state_pressure_scenario, track_id,
):
    _steady = arrival_trace_for_track(track_id, "steady")
    _bursty = arrival_trace_for_track(track_id, "bursty")

    a_base = simulate_serving(scenario, _steady, policy_for_track(track_id, "baseline"), scenario.preprocessing_paths[0])
    a_case = a_cases[a_intervention.value]
    a_result = simulate_serving(a_case.scenario, _steady, policy_for_track(track_id, "baseline"), a_case.preprocessing)
    a_memory_base = state_capacity_to_snapshot(state_capacity(scenario), pressure="baseline")
    a_memory_trial = state_capacity_to_snapshot(state_capacity(a_case.scenario), pressure="baseline")
    a_base_snap = simulation_result_to_snapshot(a_base)
    a_result_snap = simulation_result_to_snapshot(a_result)

    b_base = simulate_serving(scenario, _steady, policy_for_track(track_id, "baseline"))
    b_result = simulate_serving(
        scenario,
        arrival_trace_for_track(track_id, b_pattern.value, rate_scale=b_load.value),
        policy_for_track(track_id, "baseline"),
    )
    b_base_snap = simulation_result_to_snapshot(b_base)
    b_result_snap = simulation_result_to_snapshot(b_result)

    c_base = simulate_serving(scenario, _bursty, batching_policy_for_track(track_id, "single"))
    c_result = simulate_serving(scenario, _bursty, batching_policy_for_track(track_id, c_policy.value))
    c_base_snap = simulation_result_to_snapshot(c_base)
    c_result_snap = simulation_result_to_snapshot(c_result)

    d_base_scenario = get_recommendation_serving_scenario() if track_id == "cloud" else state_pressure_scenario(track_id, "baseline")
    d_result_scenario = get_recommendation_serving_scenario(d_pressure.value) if track_id == "cloud" else state_pressure_scenario(track_id, d_pressure.value)
    d_base_capacity = state_capacity(d_base_scenario)
    d_result_capacity = state_capacity(d_result_scenario)
    d_base_capacity_snap = state_capacity_to_snapshot(d_base_capacity, pressure="baseline")
    d_result_capacity_snap = state_capacity_to_snapshot(d_result_capacity, pressure=d_pressure.value)
    d_base = simulate_serving(d_base_scenario, _steady, policy_for_track(track_id, "baseline"))
    d_result = simulate_serving(d_result_scenario, _steady, policy_for_track(track_id, "baseline"))
    d_base_snap = simulation_result_to_snapshot(d_base)
    d_result_snap = simulation_result_to_snapshot(d_result)

    _launch_arrivals = arrival_trace_for_track(track_id, e_pattern.value)
    e_steady = {
        name: simulate_serving(scenario, _steady, policy_for_track(track_id, name))
        for name in ("reserve", "activate", "admit")
    }
    e_changed = {
        name: simulate_serving(scenario, _launch_arrivals, policy_for_track(track_id, name))
        for name in ("reserve", "activate", "admit")
    }
    e_base = e_steady[e_candidate.value]
    e_result = e_changed[e_candidate.value]
    e_alternative = e_changed[e_rejected.value] if e_rejected.value else None
    e_base_snap = simulation_result_to_snapshot(e_base)
    e_result_snap = simulation_result_to_snapshot(e_result)
    e_changed_snaps = {name: simulation_result_to_snapshot(result) for name, result in e_changed.items()}
    return (
        a_case, a_memory_base, a_memory_trial, a_base, a_base_snap, a_result, a_result_snap, b_base, b_base_snap,
        b_result, b_result_snap, c_base, c_base_snap, c_result, c_result_snap,
        d_base_capacity_snap, d_base_snap, d_result_capacity_snap, d_result_snap,
        e_alternative, e_base, e_base_snap, e_changed_snaps, e_result,
        e_result_snap,
    )


@app.cell
def _(REPLAY_METHOD, language_batching, language_lengths, language_arrivals, language_base, language_schedule, language_capacity, language_capacity_trial, language_history,
    MODEL_ID, a_intervention, a_base_snap, a_prediction, a_result_snap, b_base_snap,
    b_load, b_pattern, b_prediction, b_result_snap, c_base_snap, c_policy,
    c_prediction, c_result_snap, capture_evidence, d_base_capacity_snap,
    d_base_snap, d_prediction, d_pressure, d_result_capacity_snap,
    d_result_snap, e_base_snap, e_candidate, e_changed_snaps, e_conclusion,
    e_pattern, e_prediction, e_rejected, e_result_snap, mo, set_evidence,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    def _capture_feasible(snapshot):
        return (
            snapshot["deadline_missed_count"] == 0
            and snapshot["rejected_count"] == 0
            and snapshot["useful_completion_fraction"] == 1.0
        )

    a_upstream = {"intervention": a_intervention.value}
    b_upstream = {"changed_traffic": b_pattern.value, "rate_scale": b_load.value}
    c_upstream = {"batch_policy": c_policy.value, "traffic": "bursty"}
    d_upstream = {"state_pressure": d_pressure.value, "language_chunks": language_schedule["chunks"], "language_history_scale": language_history.value, "language_length_mix": language_lengths.value, "language_arrival_spacing_ms": language_arrivals.value}
    e_upstream = {
        "traffic": e_pattern.value,
        "candidate": e_candidate.value,
        "rejected": e_rejected.value,
        "conclusion": e_conclusion.value,
    }

    a_capture = mo.ui.button(
        label="Capture latency-path contrast", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"comparison": "complete service path", "intervention": a_intervention.value}, baseline=a_base_snap, result=a_result_snap,
            upstream_inputs=a_upstream, model_key=REPLAY_METHOD,
        )),
    )
    b_capture = mo.ui.button(
        label="Capture traffic replay", kind="success", disabled=b_prediction.value is None,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"changed_traffic": b_pattern.value, "rate_scale": b_load.value}, baseline=b_base_snap, result=b_result_snap,
            upstream_inputs=b_upstream, model_key=REPLAY_METHOD,
        )),
    )
    c_capture = mo.ui.button(
        label="Capture batching contrast", kind="success", disabled=c_prediction.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"batch_policy": c_policy.value}, baseline=c_base_snap, result=c_result_snap,
            upstream_inputs=c_upstream, model_key=REPLAY_METHOD,
        )),
    )
    d_capture = mo.ui.button(
        label="Capture state ceiling", kind="success", disabled=d_prediction.value is None,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs=dict(d_upstream), baseline=d_base_snap, result=d_result_snap,
            upstream_inputs=d_upstream,
            alternatives=(d_base_capacity_snap, d_result_capacity_snap, {"supplemental_reference": {
                "static_batching": {"model_key": "v1_13.language_batching", "inputs": language_batching["static"]["inputs"], "result": language_batching["static"]},
                "continuous_batching": {"model_key": "v1_13.language_batching", "inputs": language_batching["continuous"]["inputs"], "result": language_batching["continuous"]},
                "unchunked": {"model_key": "v1_13.language_scheduling", "inputs": language_base["inputs"], "result": language_base},
                "selected": {"model_key": "v1_13.language_scheduling", "inputs": language_schedule["inputs"], "result": language_schedule},
                "capacity": {"model_key": "v1_13.language_state_capacity", "inputs": {"pressure": "baseline"}, "result": language_capacity},
                "changed_capacity": {"model_key": "v1_13.language_state_capacity", "inputs": {"pressure": "baseline", "context_scale": language_history.value}, "result": language_capacity_trial},
            }}), model_key=REPLAY_METHOD,
        )),
    )
    _candidate_ok = _capture_feasible(e_result_snap)
    _valid_conclusion = (e_conclusion.value == "candidate" and _candidate_ok) or e_conclusion.value == "none"
    _e_disabled = (
        e_prediction.value is None or e_rejected.value is None
        or e_candidate.value == e_rejected.value or not _valid_conclusion
    )
    _decision = e_candidate.value if e_conclusion.value == "candidate" else "none"
    _rejected_snap = e_changed_snaps[e_rejected.value] if e_rejected.value else e_result_snap
    e_capture = mo.ui.button(
        label="Capture launch decision", kind="success", disabled=_e_disabled,
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={
                "traffic": e_pattern.value, "candidate": e_candidate.value,
                "rejected": e_rejected.value, "conclusion": e_conclusion.value,
            },
            baseline=e_result_snap, result=_rejected_snap,
            alternatives=(e_base_snap, *tuple(e_changed_snaps.values())), decision=_decision,
            chosen_result=e_result_snap, result_role="rejected alternative",
            upstream_inputs=e_upstream, model_key=REPLAY_METHOD,
        )),
    )
    return (
        a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream,
        d_capture, d_upstream, e_capture, e_upstream,
    )


@app.cell
def _(Scenarios, get_lab_metadata, get_learning_path, mo, notebook_bridge,
      notebook_header, notebook_learning_map, notebook_shell, track, track_id):
    from mlsysbook_labs.notebook import notebook_scenario as _scenario
    learning_path = get_learning_path(1, 13)
    journey = Scenarios.DeploymentJourneys.get(track_id)
    episode = journey.episode(13)
    learning_path["narrative"] = {
        "project": journey.mission,
        "scene": episode.scene,
        "carry_forward": episode.handoff,
        "scope": 'The supplied serving studies are illustrative fixtures. Cloud uses a native recommendation-state fixture; Part D labels a separate language-serving comparison. Preserve those boundaries in your discussion.',
    }
    _previous = journey.episode(12).handoff
    notebook_shell([
        notebook_header(get_lab_metadata("vol1/lab_13_model_serving.py"),
                        'Which serving policy completes useful requests before their deadlines?',
                        topics=episode.core_concepts),
        track,
        _scenario(learning_path, track=track_id, scene=episode.scene),
        notebook_learning_map(learning_path),
    ], volume=1)
    return (learning_path,)



@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(notebook_tab_hint, notebook_learning_part, language_batching, language_lengths, language_arrivals, language_chunks, language_history, language_base, language_schedule, language_capacity, language_capacity_trial, learning_path, notebook_shell, notebook_status, notebook_table,
    COLORS, a_case, a_intervention, a_memory_base, a_memory_trial, a_base_snap, a_capture, a_prediction, a_result_snap, a_upstream,
    apply_plotly_theme, audit_evidence, b_base_snap, b_capture, b_load, b_pattern,
    b_prediction, b_result_snap, b_upstream, c_base_snap, c_capture, c_policy,
    c_prediction, c_result_snap, c_upstream, d_base_capacity_snap, d_capture,
    d_prediction, d_pressure, d_result_capacity_snap, d_upstream, e_capture,
    e_candidate, e_changed_snaps, e_conclusion, e_pattern, e_prediction,
    e_rejected, e_result_snap, e_upstream, final_choice, final_rejected,
    final_risk, final_trigger, get_evidence, go, mo, rationale, scenario,
    track_id,
):
    _captures = get_evidence()
    upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return notebook_table(rows)

    def _display_feasible(snapshot):
        return snapshot["deadline_missed_count"] == 0 and snapshot["rejected_count"] == 0 and snapshot["useful_completion_fraction"] == 1.0

    def outcome(snapshot):
        if snapshot["rejected_count"]:
            return f"FAIL · {snapshot['rejected_count']} rejected"
        if snapshot["deadline_missed_count"]:
            return f"FAIL · {snapshot['deadline_missed_count']} missed deadline"
        if not snapshot["preprocessing_parity"]:
            return "FAIL · preprocessing mismatch"
        return "PASS"

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Live dependencies changed, or the saved runs are identical. Recapture."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div style="border-left:3px solid #247a4d;padding:8px 12px;background:#f8fafc"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later controls cannot rewrite this evidence.</small></div>')

    def part_a():
        intro = mo.md("The team can prepare inputs faster or accelerate the model. Choose one change, predict its complete-path effect, then inspect both latency and the resident-memory cost. Keep arrivals, admission, deadlines, and task fixed.")
        setup = mo.md(a_case.assumption)
        if a_prediction.value is None:
            return mo.vstack([intro, a_intervention, setup, a_prediction])
        stages = ("ingress_ms", "preprocessing_ms", "formation_wait_ms", "resource_queue_wait_ms", "execution_ms", "postprocess_ms")
        labels = ("Ingress", "Preprocess", "Batch wait", "Queue wait", "Execute", "Postprocess")
        fig = go.Figure()
        for key, label, color in zip(stages, labels, (COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["RedLine"], COLORS["Cloud"], COLORS["Grey"])):
            fig.add_bar(name=label, x=["Starting path", "Trial"], y=[a_base_snap["mean_stage_latency_ms"][key], a_result_snap["mean_stage_latency_ms"][key]], marker_color=color)
        fig.update_layout(barmode="stack", height=320, margin=dict(l=20, r=20, t=20, b=80), yaxis_title="Mean completed-path latency (ms)", legend=dict(orientation="h", y=-0.3))
        rows = []
        for name, snap, memory in (("Starting path", a_base_snap, a_memory_base), (a_case.label, a_result_snap, a_memory_trial)):
            p99 = "Unavailable" if snap["p99_latency_ms"] is None else f"{snap['p99_latency_ms']:.1f} ms"
            rows.append({"Path": name, "P99": p99, "Contract": "PASS" if snap["preprocessing_parity"] else "FAIL",
                         "Fixed memory": memory["fixed_memory_display"], "Live slots": memory["max_requests_per_replica"],
                         "Useful/offered": f"{snap['useful_completion_count']}/{snap['offered_count']}"})
        feedback = mo.md("Inspect the changed stage and the full path separately. A smaller local duration need not increase useful completions; the original path may already meet every deadline. If no requests complete, latency is unavailable and the zero-height stage bars do not mean zero-cost service. Explain the memory or contract cost, then name a condition that would reverse your choice.")
        return mo.vstack([intro, a_intervention, setup, a_prediction, apply_plotly_theme(fig), table(rows), feedback, a_capture, saved("A"),
                          mo.accordion({"Calculation Notes": mo.md("The alternatives are explicit hypothetical assumptions, not measured optimization gains. Complete latency includes ingress, preprocessing, formation, queueing, execution, and postprocessing. The same replay checks memory feasibility and retains every offered request in useful-service accounting. Preprocessing runs independently for each request; CPU contention, loading time, and accuracy are not modeled.")})])

    def part_b():
        intro = mo.md("Keep the single-request policy fixed. Compare steady arrivals with an explicit transient trace.")
        if b_prediction.value is None:
            return mo.vstack([intro, mo.hstack([b_pattern, b_load], widths="equal", wrap=True), b_prediction])
        fig = go.Figure([go.Bar(x=["Steady", b_pattern.value.title()], y=[b_base_snap["p99_latency_ms"], b_result_snap["p99_latency_ms"]], marker_color=[COLORS["BlueLine"], COLORS["RedLine"]])])
        fig.add_hline(y=scenario.deadline.to("ms").magnitude, line_dash="dash", annotation_text="Deadline")
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Empirical complete-path p99 (ms)")
        rows = [
            {"Trace": "Steady", "P50": f"{b_base_snap['p50_latency_ms']:.1f} ms", "P99": f"{b_base_snap['p99_latency_ms']:.1f} ms", "Queue peak": b_base_snap["max_queue_depth"], "Missed": b_base_snap["deadline_missed_count"], "Rejected": b_base_snap["rejected_count"]},
            {"Trace": b_pattern.value.title(), "P50": f"{b_result_snap['p50_latency_ms']:.1f} ms", "P99": f"{b_result_snap['p99_latency_ms']:.1f} ms", "Queue peak": b_result_snap["max_queue_depth"], "Missed": b_result_snap["deadline_missed_count"], "Rejected": b_result_snap["rejected_count"]},
        ]
        return mo.vstack([intro, mo.hstack([b_pattern, b_load], widths="equal", wrap=True), b_prediction, apply_plotly_theme(fig), table(rows), mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. Changed traffic produces p99 **{b_result_snap['p99_latency_ms']:.1f} ms**, **{b_result_snap['deadline_missed_count']}** deadline misses, and **{b_result_snap['rejected_count']}** rejections."), kind="danger" if (b_result_snap["deadline_missed_count"] > 0 or b_result_snap["rejected_count"] > 0) else "success"), b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md("The deterministic trace replays every request through the same bounded server. The offered-load control scales every phase rate before the replay. P99 is the nearest-rank percentile of complete request durations; rejected requests are counted separately and never disappear inside the percentile.")})])

    def part_c():
        intro = mo.md("Use the same burst trace for a single-request baseline and the selected size/timeout policy.")
        if c_prediction.value is None:
            return mo.vstack([intro, c_policy, c_prediction])
        rows = [
            {"Policy": "Single", "Batches": c_base_snap["batch_count"], "Mean batch": f"{c_base_snap['mean_batch_size']:.2f}", "Busy replica time": f"{c_base_snap['busy_replica_time_ms']:.1f} replica-ms", "P99": f"{c_base_snap['p99_latency_ms']:.1f} ms", "Outcome": outcome(c_base_snap)},
            {"Policy": c_policy.value, "Batches": c_result_snap["batch_count"], "Mean batch": f"{c_result_snap['mean_batch_size']:.2f}", "Busy replica time": f"{c_result_snap['busy_replica_time_ms']:.1f} replica-ms", "P99": f"{c_result_snap['p99_latency_ms']:.1f} ms", "Outcome": outcome(c_result_snap)},
        ]
        figure = go.Figure()
        for label, snapshot, symbol in (
            ("Single", c_base_snap, "circle"),
            ("Selected batch policy", c_result_snap, "diamond"),
        ):
            figure.add_scatter(
                x=[snapshot["busy_replica_time_ms"]],
                y=[snapshot["p99_latency_ms"]],
                name=label, mode="markers",
                marker=dict(size=15, symbol=symbol, color=COLORS["GreenLine"] if _display_feasible(snapshot) else COLORS["RedLine"]),
                hovertemplate="Execution work: %{x:.1f} replica-ms<br>Complete-path p99: %{y:.1f} ms<extra>%{fullData.name}</extra>",
            )
        figure.add_hline(y=scenario.deadline.to("ms").magnitude, line_dash="dash", annotation_text="Deadline", annotation_position="top left")
        figure.update_layout(
            height=320, margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Execution work (replica-ms)",
            xaxis=dict(rangemode="tozero"),
            yaxis_title="Complete-path p99 (ms)",
            legend=dict(orientation="h", y=1.18, x=0),
        )
        challenge = mo.md("**Try both batch policies.** Look for less execution work while keeping the selected point below the deadline. If work does not fall, explain why waiting did not earn its cost in this context. Check missed and rejected requests in the table before deciding: a lower p99 alone does not establish feasibility.")
        return mo.vstack([intro, c_policy, c_prediction, challenge, apply_plotly_theme(figure), table(rows), mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. The selected policy uses **{c_result_snap['batch_count']}** batches and **{c_result_snap['busy_replica_time_ms']:.1f} replica-ms**, with complete-path p99 **{c_result_snap['p99_latency_ms']:.1f} ms**."), kind="info"), c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md("A batch dispatches when its size limit fills or its oldest request reaches the timeout. Batch size and timeout therefore change actual dispatch and completion times. Busy replica time reports observable execution work. Green points meet every request deadline without rejection or preprocessing mismatch; red points violate at least one of those requirements.")})])

    def part_d():
        _state_name = "resident embeddings and request feature/ranking buffers" if track_id == "cloud" else "working-state buffers"
        intro = mo.md(f"This track accounts for **{_state_name}**. Predict the capacity result before revealing the selected pressure case.")
        if d_prediction.value is None:
            return mo.vstack([intro, d_pressure, d_prediction])
        rows = [
            {"Case": "Baseline", "Fixed memory": d_base_capacity_snap["fixed_memory_display"], "State/request": d_base_capacity_snap["state_per_request_display"], "KV/request": d_base_capacity_snap["kv_state_per_request_display"], "Concurrent ceiling": d_base_capacity_snap["max_requests_per_replica"], "Outcome": "PASS" if d_base_capacity_snap["feasible"] else "FAIL"},
            {"Case": d_pressure.value.title(), "Fixed memory": d_result_capacity_snap["fixed_memory_display"], "State/request": d_result_capacity_snap["state_per_request_display"], "KV/request": d_result_capacity_snap["kv_state_per_request_display"], "Concurrent ceiling": d_result_capacity_snap["max_requests_per_replica"], "Outcome": "PASS" if d_result_capacity_snap["feasible"] else "FAIL"},
        ]
        return mo.vstack([intro, d_pressure, d_prediction, table(rows), mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. The per-replica concurrency ceiling changes from **{d_base_capacity_snap['max_requests_per_replica']}** to **{d_result_capacity_snap['max_requests_per_replica']}**; the selected case is **{'feasible' if d_result_capacity_snap['feasible'] else 'infeasible'}**."), kind="success" if d_result_capacity_snap["feasible"] else "danger"), mo.md("#### Separate language-serving reference\nThe recommendation endpoint is not a language model. This comparison asks how fixed prompt work competes with an already active token stream."),
            mo.md("**Retained history versus concurrency.** Keep weights, dtype, device memory, and non-KV state fixed. History is relative to the supplied baseline allocation, not an asserted token length. This probe changes storage only; it does not predict context-dependent execution time."), language_history,
            table([{"History": label, "KV/request": result["kv_state_per_request_display"], "Concurrent ceiling": result["max_requests_per_replica"], "Can one request fit?": "Yes" if result["feasible"] else "No"} for label, result in (("Baseline", language_capacity), ("Selected", language_capacity_trial))]),
            mo.md("**Prefill scheduling versus active-stream cadence.** This is a separate fixed-work scheduling probe; its timings are not predictions for the history choice above."), language_chunks,
            table([{"Schedule": label, "Prompt first token (ms)": result["prompt_first_token_ms"], "Maximum active-token gap (ms)": result["max_active_token_gap_ms"], "Busy work (ms)": result["total_busy_time_ms"]} for label, result in (("Unchunked", language_base), ("Selected", language_schedule))]),
            mo.callout(mo.md(language_schedule["assumption_note"]), kind="info"),
            mo.accordion({"Explore iteration-level admission": mo.vstack([
                mo.md("**A long response holds the cohort open.** Predict whether refilling completed slots helps when response lengths vary. Switch to equal lengths and then sparse arrivals to find when the advantage disappears. These requests arrive already prefilled; first-decode waiting here is not end-to-end time to first token."),
                language_lengths, language_arrivals,
                table([{"Policy": policy.title(), "Completed/offered": f"{result['completed_count']}/{result['offered_count']}",
                        "Slots that fit": result["slot_limit"], "Trace tokens/s": "Unavailable" if result["tokens_per_second"] is None else f"{result['tokens_per_second']:.1f}",
                        "p99 completion (ms)": result["p99_completion_latency_ms"],
                        "Mean first decode (ms)": "Unavailable" if result["mean_first_decode_latency_ms"] is None else f"{result['mean_first_decode_latency_ms']:.1f}",
                        "Idle slot iterations": result["idle_slot_iterations"],
                        "Peak reserved memory (GiB)": "Infeasible" if result["peak_memory_bytes"] is None else f"{result['peak_memory_bytes'] / (1024 ** 3):.2f}"}
                       for policy, result in language_batching.items()]),
                table([{"Request": row["request_id"], "Arrival (ms)": row["arrival_ms"], "Output tokens": row["output_tokens"],
                        "Static completes (ms)": language_batching["static"]["requests"][row["request_id"]]["completion_ms"],
                        "Continuous completes (ms)": row["completion_ms"]} for row in language_batching["continuous"]["requests"]]),
                mo.callout(mo.md(language_batching["continuous"]["assumption_note"]), kind="info"),
                mo.md("**Explain:** which arrivals benefit from refilling, and when does memory permit fewer slots? Increasing history does not change the supplied iteration time: it changes admission capacity. A storage ceiling alone cannot establish measured throughput."),
            ])}),
            mo.md("**Explain:** which request waits longer under your selected schedule? Keep that scheduling claim separate from the native live-state capacity result."), d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md("Fixed memory equals resident model memory plus batch workspace. Remaining memory is divided by live state per request. KV state appears only for the compatible language workload; other tracks use their own activation, frame, audio, or image buffers.")})])

    def part_e():
        intro = mo.md("Compare a candidate and a tested alternative under the same changed trace. A launch passes only with zero deadline misses, zero rejections, and preprocessing parity.")
        if e_prediction.value is None:
            return mo.vstack([intro, e_pattern, mo.hstack([e_candidate, e_rejected], widths="equal", wrap=True), e_prediction])
        rows = []
        for name in ("reserve", "activate", "admit"):
            snap = e_changed_snaps[name]
            rows.append({"Policy": name.title(), "P99": f"{snap['p99_latency_ms']:.1f} ms", "Missed": snap["deadline_missed_count"], "Rejected": snap["rejected_count"], "Useful/offered": f"{snap['useful_completion_count']}/{snap['offered_count']}", "Provisioned time": f"{snap['provisioned_replica_time_ms']:.1f} replica-ms", "Outcome": outcome(snap)})
        _alternative = e_changed_snaps[e_rejected.value] if e_rejected.value else None
        _valid = e_rejected.value is not None and e_candidate.value != e_rejected.value
        _candidate_ok = _display_feasible(e_result_snap)
        _conclusion_ok = (e_conclusion.value == "candidate" and _candidate_ok) or e_conclusion.value == "none"
        note = "Choose two different tested policies and a conclusion supported by their outcomes." if not (_valid and _conclusion_ok) else f"Candidate: {outcome(e_result_snap)}. Tested alternative: {outcome(_alternative)}."
        return mo.vstack([intro, e_pattern, mo.hstack([e_candidate, e_rejected], widths="equal", wrap=True), e_prediction, table(rows), e_conclusion, mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. {note}"), kind="success" if _valid and _conclusion_ok else "warn"), e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md("Reserve capacity is available from the start. Delayed activation becomes available only after its activation delay. Bounded admission caps the waiting queue and reports rejected requests separately. Provisioned replica-time exposes the resource consequence of each policy.")})])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            rows.append({"Part": part, "Prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING")})
        _e_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
        _tested_rejected = _captures["E"].to_dict()["inputs"]["rejected"] if "E" in _captures else None
        if "E" in _captures:
            _e_saved = _captures["E"].to_dict()
            _candidate_saved = _e_saved["chosen_result"] or _e_saved["baseline"]
            _rejected_saved = _e_saved["result"]
            _decision_rows = table([
                {"Role": "Recommended" if _e_decision != "none" else "Tested candidate", "Policy": _e_saved["inputs"]["candidate"], "P99": f"{_candidate_saved['p99_latency_ms']:.1f} ms", "Missed": _candidate_saved["deadline_missed_count"], "Rejected": _candidate_saved["rejected_count"], "Provisioned": f"{_candidate_saved['provisioned_replica_time_ms']:.1f} replica-ms"},
                {"Role": "Rejected alternative", "Policy": _tested_rejected, "P99": f"{_rejected_saved['p99_latency_ms']:.1f} ms", "Missed": _rejected_saved["deadline_missed_count"], "Rejected": _rejected_saved["rejected_count"], "Provisioned": f"{_rejected_saved['provisioned_replica_time_ms']:.1f} replica-ms"},
            ])
        else:
            _decision_rows = mo.callout(mo.md("Capture Part E to expose the quantified policy comparison."), kind="warn")
        complete = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value == _e_decision and final_rejected.value == _tested_rejected and final_choice.value != final_rejected.value
        return mo.vstack([mo.md("### Synthesis · Defend the launch decision (5 min)\nUse saved evidence to name the chosen policy or defensible hold, quantify the tested alternative, state one remaining limitation, and set a reevaluation trigger."), table(rows), _decision_rows, mo.callout(mo.md("Saved snapshots remain fixed while live controls move. Recapture stale evidence before generating the report."), kind="info"), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale, mo.callout(mo.md("**Ready for the local report.**" if complete else "Complete five current contrasts, match the Part E decision and tested alternative, then add a limitation, trigger, and quantified rationale."), kind="success" if complete else "warn")])

    notebook_shell([notebook_tab_hint(), mo.ui.tabs({"Part A": notebook_learning_part(learning_path, "A", part_a()), "Part B": notebook_learning_part(learning_path, "B", part_b()), "Part C": notebook_learning_part(learning_path, "C", part_c()), "Part D": notebook_learning_part(learning_path, "D", part_d()), "Part E": notebook_learning_part(learning_path, "E", part_e()), "Synthesis": build_synthesis()})], volume=1)
    return (audit,)


@app.cell(hide_code=True)
def _():
    from mlsysbook_labs.learning_paths import get_learning_path as _submission_path
    from mlsysbook_labs.notebook import notebook_shell as _submission_shell, notebook_submission as _submission
    _submission_shell([_submission(_submission_path(1, 13))])
    return


@app.cell
def _(learning_path, notebook_shell,
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, rationale,
    report_export_panel, scenario, track_id,
):
    _captures = get_evidence()
    _e_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _tested_rejected = _captures["E"].to_dict()["inputs"]["rejected"] if "E" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value == _e_decision and final_rejected.value == _tested_rejected and final_choice.value != final_rejected.value
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not _ready, _download_readiness(volume=1))
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_13_model_serving.py"),
        track=track_id,
        scenario=f"{scenario.serving_role} · {scenario.workload}",
        learning_objectives=learning_path["learning_objectives"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "chosen_result": snapshots[part]["chosen_result"], "result_role": snapshots[part]["result_role"], "alternatives": snapshots[part]["alternatives"]} for part in "ABCDE"},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=[
            "Useful deadline completions require both latency and preprocessing parity.",
            "Batching changes actual schedules, execution work, and tail latency together.",
            "Admission loss must remain visible beside percentiles.",
        ],
        reflections={"rationale": rationale.value, "remaining_limitation": final_risk.value},
        residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": snapshots, "recommendation": final_choice.value, "rejected": final_rejected.value, "trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": scenario.assumption_note, "calculation": "Deterministic request replay with complete-path empirical percentiles."},
    )
    notebook_shell([report_export_panel(report)], volume=1)
    return (report,)




@app.cell
def _(notebook_shell,
    MODEL_ID, audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _e_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _tested_rejected = _captures["E"].to_dict()["inputs"]["rejected"] if "E" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value == _e_decision and final_rejected.value == _tested_rejected and final_choice.value != final_rejected.value
    save_discussion = mo.ui.run_button(label="Save discussion", disabled=not _ready)
    notebook_shell([save_discussion, mo.md("When ready, downloading is separate from saving locally. Editing your evidence or discussion requires saving again.")], volume=1)
    return (save_discussion,)


@app.cell
async def _(notebook_shell, notebook_status, save_discussion,
    MODEL_ID, audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _e_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _tested_rejected = _captures["E"].to_dict()["inputs"]["rejected"] if "E" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value == _e_decision and final_rejected.value == _tested_rejected and final_choice.value != final_rejected.value
    _status = ("Ready to save" if _ready else "Evidence in progress")
    if _ready and save_discussion.value:
        try:
            ledger.save(chapter=13, design={
                "schema_version": 1,
                "lab_id": "v1_13",
                "track_id": track_id,
                "model_id": MODEL_ID,
                "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value,
                "rationale": rationale.value,
            })
            await ledger.flush()
        except Exception:
            _status = "LOCAL SAVE FAILED · DOWNLOAD THE REPORT TO KEEP YOUR EVIDENCE"
        else:
            _status = "SAVED"
    notebook_shell([notebook_status(_status, "Editing requires saving again; the downloaded discussion is separate from local storage.")], volume=1)
    return


if __name__ == "__main__":
    app.run()
