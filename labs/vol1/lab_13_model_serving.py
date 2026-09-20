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
    from mlsysim.engine.v1_13_experiments import (
        MODEL_ID,
        arrival_trace_for_track,
        batching_policy_for_track,
        get_serving_scenario,
        policy_for_track,
        simulation_result_to_snapshot,
        simulate_serving,
        state_capacity,
        state_capacity_to_snapshot,
        state_pressure_scenario,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata, report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence

    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, MODEL_ID, apply_plotly_theme,
        arrival_trace_for_track, audit_evidence, batching_policy_for_track,
        build_lab_report, capture_evidence, get_lab_metadata,
        get_serving_scenario, go, ledger, mo, policy_for_track,
        report_export_panel, simulation_result_to_snapshot, simulate_serving,
        state_capacity, state_capacity_to_snapshot, state_pressure_scenario,
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
def _(get_serving_scenario, track):
    track_id = track.value
    scenario = get_serving_scenario(track_id)
    return scenario, track_id


@app.cell
def _(mo, track_id):
    _track_key = track_id
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
    return b_load, b_pattern, c_policy, d_pressure, e_candidate, e_conclusion, e_pattern, e_rejected


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {
            "Matched path completes more useful requests": "matched",
            "Faster mismatched path completes more useful requests": "mismatched",
            "Both paths are equally useful": "equal",
        },
        label="Which preprocessing path produces more useful deadline completions?",
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
    arrival_trace_for_track, b_load, b_pattern, batching_policy_for_track, c_policy,
    d_pressure, e_candidate, e_pattern, e_rejected, policy_for_track,
    scenario, simulation_result_to_snapshot, simulate_serving, state_capacity,
    state_capacity_to_snapshot, state_pressure_scenario, track_id,
):
    _steady = arrival_trace_for_track(track_id, "steady")
    _bursty = arrival_trace_for_track(track_id, "bursty")

    a_base = simulate_serving(scenario, _steady, policy_for_track(track_id, "baseline"), scenario.preprocessing_paths[0])
    a_result = simulate_serving(scenario, _steady, policy_for_track(track_id, "baseline"), scenario.preprocessing_paths[1])
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

    d_base_scenario = state_pressure_scenario(track_id, "baseline")
    d_result_scenario = state_pressure_scenario(track_id, d_pressure.value)
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
        a_base, a_base_snap, a_result, a_result_snap, b_base, b_base_snap,
        b_result, b_result_snap, c_base, c_base_snap, c_result, c_result_snap,
        d_base_capacity_snap, d_base_snap, d_result_capacity_snap, d_result_snap,
        e_alternative, e_base, e_base_snap, e_changed_snaps, e_result,
        e_result_snap,
    )


@app.cell
def _(
    MODEL_ID, a_base_snap, a_prediction, a_result_snap, b_base_snap,
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

    a_upstream = {"paths": ["matched", "mismatched"]}
    b_upstream = {"changed_traffic": b_pattern.value, "rate_scale": b_load.value}
    c_upstream = {"batch_policy": c_policy.value, "traffic": "bursty"}
    d_upstream = {"state_pressure": d_pressure.value}
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
            inputs={"comparison": "preprocessing parity"}, baseline=a_base_snap, result=a_result_snap,
            upstream_inputs=a_upstream, model_key=MODEL_ID,
        )),
    )
    b_capture = mo.ui.button(
        label="Capture traffic replay", kind="success", disabled=b_prediction.value is None,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"changed_traffic": b_pattern.value, "rate_scale": b_load.value}, baseline=b_base_snap, result=b_result_snap,
            upstream_inputs=b_upstream, model_key=MODEL_ID,
        )),
    )
    c_capture = mo.ui.button(
        label="Capture batching contrast", kind="success", disabled=c_prediction.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"batch_policy": c_policy.value}, baseline=c_base_snap, result=c_result_snap,
            upstream_inputs=c_upstream, model_key=MODEL_ID,
        )),
    )
    d_capture = mo.ui.button(
        label="Capture state ceiling", kind="success", disabled=d_prediction.value is None,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={"state_pressure": d_pressure.value}, baseline=d_base_snap, result=d_result_snap,
            upstream_inputs=d_upstream,
            alternatives=(d_base_capacity_snap, d_result_capacity_snap), model_key=MODEL_ID,
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
            upstream_inputs=e_upstream, model_key=MODEL_ID,
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
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:780px}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 13</span><span>ABOUT 50–55 MIN</span></div><h1>Serving Under Deadlines</h1><p>Which serving policy completes useful requests before their deadlines when demand, batching, and live state interact?</p><div class="pilot-meta"><div><b>Track</b><br>{scenario.label}</div><div><b>Serving role</b><br>{scenario.serving_role}</div><div><b>Workload</b><br>{scenario.workload}</div><div><b>Deliverable</b><br>Serving launch memo</div></div></section>""")
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, css, header, track,
        mo.Html('<p class="pilot-note">Every workload and latency is an illustrative scenario assumption. Each saved result keeps the exact replay inputs used to produce it.</p>'),
    ], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base_snap, a_capture, a_prediction, a_result_snap, a_upstream,
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
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

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
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later controls cannot rewrite this evidence.</small></div>')

    def part_a():
        intro = mo.md("### A · Where should we spend the latency budget? (9 min)\nPredict before comparing two complete request paths. One path preserves the expected preprocessing contract; the other is faster but changes it.")
        if a_prediction.value is None:
            return mo.vstack([intro, a_prediction])
        stages = ("ingress_ms", "preprocessing_ms", "formation_wait_ms", "resource_queue_wait_ms", "execution_ms", "postprocess_ms")
        labels = ("Ingress", "Preprocess", "Batch wait", "Queue wait", "Execute", "Postprocess")
        fig = go.Figure()
        for key, label, color in zip(stages, labels, (COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["RedLine"], COLORS["Cloud"], COLORS["Grey"])):
            fig.add_bar(name=label, x=["Matched", "Mismatched"], y=[a_base_snap["mean_stage_latency_ms"][key], a_result_snap["mean_stage_latency_ms"][key]], marker_color=color)
        fig.update_layout(barmode="stack", height=280, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Mean complete-path latency (ms)", legend_orientation="h")
        rows = [
            {"Path": f"Matched ({scenario.preprocessing_paths[0].name})", "P99": f"{a_base_snap['p99_latency_ms']:.1f} ms", "Parity": "PASS", "Useful/offered": f"{a_base_snap['useful_completion_count']}/{a_base_snap['offered_count']}"},
            {"Path": f"Mismatched ({scenario.preprocessing_paths[1].name})", "P99": f"{a_result_snap['p99_latency_ms']:.1f} ms", "Parity": "FAIL", "Useful/offered": f"{a_result_snap['useful_completion_count']}/{a_result_snap['offered_count']}"},
        ]
        return mo.vstack([intro, a_prediction, apply_plotly_theme(fig), table(rows), mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. The matched path ({scenario.preprocessing_paths[0].name}) yields **{a_base_snap['useful_completion_count']}/{a_base_snap['offered_count']}** useful deadline completions; the mismatched path ({scenario.preprocessing_paths[1].name}) yields **{a_result_snap['useful_completion_count']}/{a_result_snap['offered_count']}**."), kind="info"), a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("Complete latency includes ingress, preprocessing, batch formation, resource queueing, batch execution, and postprocessing. Preprocessing parity is an explicit contract check; the scenario does not invent an accuracy score.")})])

    def part_b():
        intro = mo.md("### B · How much headroom does traffic require? (10 min)\nKeep the single-request policy fixed. Compare steady arrivals with an explicit transient trace.")
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
        intro = mo.md("### C · When is waiting for a batch worthwhile? (10 min)\nUse the same burst trace for a single-request baseline and the selected size/timeout policy.")
        if c_prediction.value is None:
            return mo.vstack([intro, c_policy, c_prediction])
        rows = [
            {"Policy": "Single", "Batches": c_base_snap["batch_count"], "Mean batch": f"{c_base_snap['mean_batch_size']:.2f}", "Busy replica time": f"{c_base_snap['busy_replica_time_ms']:.1f} ms", "P99": f"{c_base_snap['p99_latency_ms']:.1f} ms", "Outcome": outcome(c_base_snap)},
            {"Policy": c_policy.value, "Batches": c_result_snap["batch_count"], "Mean batch": f"{c_result_snap['mean_batch_size']:.2f}", "Busy replica time": f"{c_result_snap['busy_replica_time_ms']:.1f} ms", "P99": f"{c_result_snap['p99_latency_ms']:.1f} ms", "Outcome": outcome(c_result_snap)},
        ]
        return mo.vstack([intro, c_policy, c_prediction, table(rows), mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. The selected policy uses **{c_result_snap['batch_count']}** batches and **{c_result_snap['busy_replica_time_ms']:.1f} replica-ms**, with complete-path p99 **{c_result_snap['p99_latency_ms']:.1f} ms**."), kind="info"), c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md("A batch dispatches when its size limit fills or its oldest request reaches the timeout. Batch size and timeout therefore change actual dispatch and completion times. Busy replica time reports observable execution work.")})])

    def part_d():
        _state_name = "KV + working state" if track_id == "cloud" else "working-state buffers"
        intro = mo.md(f"### D · What state limits concurrency? (9 min)\nThis track accounts for **{_state_name}**. Predict the capacity result before revealing the selected pressure case.")
        if d_prediction.value is None:
            return mo.vstack([intro, d_pressure, d_prediction])
        rows = [
            {"Case": "Baseline", "Fixed memory": d_base_capacity_snap["fixed_memory_display"], "State/request": d_base_capacity_snap["state_per_request_display"], "KV/request": d_base_capacity_snap["kv_state_per_request_display"], "Concurrent ceiling": d_base_capacity_snap["max_requests_per_replica"], "Outcome": "PASS" if d_base_capacity_snap["feasible"] else "FAIL"},
            {"Case": d_pressure.value.title(), "Fixed memory": d_result_capacity_snap["fixed_memory_display"], "State/request": d_result_capacity_snap["state_per_request_display"], "KV/request": d_result_capacity_snap["kv_state_per_request_display"], "Concurrent ceiling": d_result_capacity_snap["max_requests_per_replica"], "Outcome": "PASS" if d_result_capacity_snap["feasible"] else "FAIL"},
        ]
        return mo.vstack([intro, d_pressure, d_prediction, table(rows), mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. The per-replica concurrency ceiling changes from **{d_base_capacity_snap['max_requests_per_replica']}** to **{d_result_capacity_snap['max_requests_per_replica']}**; the selected case is **{'feasible' if d_result_capacity_snap['feasible'] else 'infeasible'}**."), kind="success" if d_result_capacity_snap["feasible"] else "danger"), d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md("Fixed memory equals resident model memory plus batch workspace. Remaining memory is divided by live state per request. KV state appears only for the compatible language workload; other tracks use their own activation, frame, audio, or image buffers.")})])

    def part_e():
        intro = mo.md("### E · Which policy survives demand changes? (10 min)\nCompare a candidate and a tested alternative under the same changed trace. A launch passes only with zero deadline misses, zero rejections, and preprocessing parity.")
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

    mo.ui.tabs({"Part A": part_a(), "Part B": part_b(), "Part C": part_c(), "Part D": part_d(), "Part E": part_e(), "Synthesis": build_synthesis()})
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, rationale,
    report_export_panel, scenario, track_id,
):
    _captures = get_evidence()
    _e_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _tested_rejected = _captures["E"].to_dict()["inputs"]["rejected"] if "E" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value == _e_decision and final_rejected.value == _tested_rejected and final_choice.value != final_rejected.value
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_13_model_serving.py"),
        track=track_id,
        scenario=f"{scenario.serving_role} · {scenario.workload}",
        learning_objectives=[
            "Decompose complete request latency and verify preprocessing parity.",
            "Replay transient arrivals and interpret empirical tail latency and admission loss.",
            "Compare batching, state capacity, and launch policies from controlled baselines.",
        ],
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
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    MODEL_ID, audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _e_decision = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _tested_rejected = _captures["E"].to_dict()["inputs"]["rejected"] if "E" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value == _e_decision and final_rejected.value == _tested_rejected and final_choice.value != final_rejected.value
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
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
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">13 · Serving Under Deadlines</span><span style="flex:1"></span><span class="hud-label"> · STATUS: </span><span class="hud-active">{_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
