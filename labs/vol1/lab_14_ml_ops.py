import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 14: Evidence-Driven Operations · MLSysBook")


@app.cell
async def _():
    import sys
    from pathlib import Path
    import marimo as mo

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.3.1-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        labs_dir = Path(__file__).resolve().parents[1]
        if str(labs_dir) not in sys.path:
            sys.path.insert(0, str(labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.labs.experiments.v1_14_experiments import (
        TRACKS, canary_experiment, experiment_options, get_track_scenario,
        incident_experiment, monitoring_experiment, retraining_experiment,
    )
    from mlsysim.labs.experiments.v1_14_experiments import parity_experiment
    from mlsysim.labs.experiments.v1_14_experiments import rollout_experiment, release_recovery_experiment
    from mlsysim import fmt_memory, fmt_time
    from mlsysim.core.units import Q_
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
    return (rollout_experiment, release_recovery_experiment, fmt_memory, fmt_time, Q_, parity_experiment, Scenarios, get_learning_path, notebook_bridge, notebook_header,
            notebook_learning_map, notebook_learning_part, notebook_tab_hint, notebook_shell, notebook_status, notebook_table,
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, TRACKS, apply_plotly_theme,
        audit_evidence, build_lab_report, canary_experiment, capture_evidence,
        experiment_options, get_lab_metadata, get_track_scenario, go,
        incident_experiment, ledger, mo, monitoring_experiment,
        report_export_panel, retraining_experiment,
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
        value=_context(_preferred(ledger)).label, label="Deployment track",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(TRACKS, experiment_options, get_track_scenario, track):
    track_id = track.value
    scenario = get_track_scenario(track_id)
    options = experiment_options(track_id)
    assert track_id in TRACKS
    return options, scenario, track_id


@app.cell
def _(mo, track_id):
    _selected_project = track_id
    parity_transform = mo.ui.dropdown({"Original fitted transformation": "matched", "Swap request values": "swapped", "Refit preprocessing": "refitted"},
                                     value="Swap request values", label="Serving transformation")
    parity_bundle = mo.ui.dropdown({"Compatible complete bundle": "compatible", "Weights-only rollback": "weights_only", "Missing fitted state": "missing_state"},
                                  value="Compatible complete bundle", label="Bundle available for replay")
    return parity_transform, parity_bundle


@app.cell
def _(parity_experiment, parity_transform, parity_bundle, track_id):
    parity_baseline = parity_experiment(track_id, serving_transform="matched", release_bundle="compatible")
    parity_changed = parity_experiment(track_id, serving_transform=parity_transform.value,
                                       release_bundle=parity_bundle.value)
    return parity_baseline, parity_changed


@app.cell
def _(mo, track_id):
    _project = track_id
    d_reservation = mo.ui.dropdown({"Overlap background reservation": "overlap", "Wait for background reservation to finish": "wait"},
                                   value="Overlap background reservation", label="Candidate staging plan")
    e_restore = mo.ui.dropdown({"Restore weights only": "weights_only", "Restore weights and transform": "weights_transform", "Restore complete release bundle": "complete_bundle"},
                              value="Restore weights only", label="Rollback unit")
    e_inflight = mo.ui.dropdown({"Reuse the mutated global bundle": "reuse_mutated", "Drain requests with their pinned bundle": "drain_pinned", "Confirm cancellation of admitted requests": "cancel_confirmed"},
                               value="Reuse the mutated global bundle", label="Already-admitted requests")
    e_logs = mo.ui.dropdown({"Retain degraded-window logs for training": "retain", "Quarantine degraded-window logs": "quarantine"},
                           value="Retain degraded-window logs for training", label="Degraded data")
    e_cause = mo.ui.dropdown({"Release contract incompatibility": "release_contract", "Population shift after a compatible release": "population_shift"},
                            value="Release contract incompatibility", label="Incident cause")
    return d_reservation, e_restore, e_inflight, e_logs, e_cause


@app.cell
def _(mo, options, track_id):
    _track_key = track_id
    a_threshold = mo.ui.dropdown(
        options["monitor_thresholds"], value="Sensitive · 0.15", label="Proxy alert threshold",
    )
    b_threshold = mo.ui.dropdown(
        options["monitor_thresholds"], value="Insensitive · 0.40", label="Proxy alert threshold",
    )
    b_interval = mo.ui.dropdown(
        options["monitor_intervals"], value=list(options["monitor_intervals"])[1], label="Sampling interval",
    )
    c_interval = mo.ui.dropdown(
        options["scheduled_intervals"], value=list(options["scheduled_intervals"])[1], label="Scheduled cadence",
    )
    c_choice = mo.ui.radio(
        {"Scheduled cadence": "scheduled", "Evidence-triggered policy": "evidence_triggered"},
        value=None, label="Selected retraining policy",
    )
    d_choice = mo.ui.radio(
        {"10% canary": 0.10, "25% canary": 0.25, "50% canary": 0.50, "Hold for more evidence / no feasible promotion": "none"},
        label="Promotion recommendation",
    )
    d_rejected = mo.ui.radio(
        {"10% canary": 0.10, "25% canary": 0.25, "50% canary": 0.50},
        label="Quantified rejected alternative",
    )
    e_rollback = mo.ui.dropdown(
        options["rollback_options"], value="5 minutes", label="Rollback stage duration",
    )
    return (
        a_threshold, b_interval, b_threshold, c_choice, c_interval,
        d_choice, d_rejected, e_rollback,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio({
        "Every proxy alert proves quality failure": "proves_failure",
        "Some proxy alerts precede or outnumber failures": "imperfect_proxy",
        "Delayed labels arrive before proxy alerts": "labels_first",
    }, label="What will the sensitive proxy threshold establish?").form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio({
        "Matching aggregate values establish parity": "aggregate", "Request-keyed comparison is needed": "keyed", "A version label alone establishes replay": "version",
    }, label="Which evidence tests the serving transformation?").form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio({
        "Scheduled always costs less": "scheduled_cheaper",
        "Evidence-triggered may wait longer but launch fewer jobs": "triggered_tradeoff",
        "Retraining changes quality when the job starts": "changes_at_start",
    }, label="How will delayed evidence change retraining?").form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio({
        "A smaller traffic share installs a smaller candidate": "shrinks_copy",
        "Traffic share changes exposure while full copies stay resident": "full_copy",
        "Waiting releases capacity without delaying evidence": "free_wait",
    }, label="What changes when the canary fraction becomes smaller?").form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio({
        "Fast weights-only rollback restores every request": "weights_only",
        "New and already-admitted requests need compatible bundles": "bundle",
        "A complete bundle repairs every possible incident cause": "universal",
    }, label="What establishes recovery after routing is reverted?").form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio({
        "10% canary": 0.10, "25% canary": 0.25, "50% canary": 0.50,
        "Hold for more evidence / no feasible promotion": "none",
    }, label="Final recommendation")
    final_rejected = mo.ui.radio(
        {"10% canary": 0.10, "25% canary": 0.25, "50% canary": 0.50},
        label="Rejected tested alternative",
    )
    final_trigger = mo.ui.radio({
        "A new missed failure episode": "missed_failure",
        "Canary evidence loses cohort coverage": "cohort_coverage",
        "Recovery time exceeds its objective": "recovery_objective",
    }, label="Reevaluation trigger")
    final_risk = mo.ui.radio({
        "Proxy alerts do not prove quality loss": "proxy_ambiguity",
        "Labels remain delayed": "label_delay",
        "Illustrative outcomes require local calibration": "scenario_calibration",
    }, label="Remaining limitation")
    rationale = mo.ui.text_area(
        label="Concise rationale",
        placeholder="Connect saved monitoring, retraining, rollout, and recovery evidence.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(rollout_experiment, release_recovery_experiment, d_reservation, e_restore, e_inflight, e_logs, e_cause,
    a_threshold, b_interval, b_threshold, canary_experiment, c_interval,
    d_choice, d_rejected, e_rollback, incident_experiment,
    monitoring_experiment, options, retraining_experiment, track_id,
):
    a_base = monitoring_experiment(
        track_id, proxy_threshold=0.30, sampling_interval_hours=options["step_hours"],
    )
    a_result = monitoring_experiment(
        track_id, proxy_threshold=a_threshold.value, sampling_interval_hours=options["step_hours"],
    )
    b_base = monitoring_experiment(
        track_id, proxy_threshold=0.30, sampling_interval_hours=options["step_hours"],
    )
    b_result = monitoring_experiment(
        track_id, proxy_threshold=b_threshold.value, sampling_interval_hours=b_interval.value,
    )
    c_scheduled = retraining_experiment(
        track_id, policy="scheduled", scheduled_interval_hours=c_interval.value,
    )
    c_triggered = retraining_experiment(
        track_id, policy="evidence_triggered", scheduled_interval_hours=None,
    )
    d_results = {
        fraction: rollout_experiment(track_id, canary_fraction=fraction, reservation_plan=d_reservation.value)
        for fraction in options["canary_candidates"].values()
    }
    if d_choice.value not in (None, "none") and d_results[d_choice.value]["decision"] == "promote":
        e_affected_fraction = 1.0
    elif d_choice.value not in (None, "none"):
        e_affected_fraction = d_choice.value
    elif d_rejected.value is not None:
        e_affected_fraction = d_rejected.value
    else:
        e_affected_fraction = 0.25
    _comparison_fraction = d_choice.value if d_choice.value not in (None, "none") else (d_rejected.value or 0.25)
    d_capacity_pair = {plan: rollout_experiment(track_id, canary_fraction=_comparison_fraction, reservation_plan=plan)
                       for plan in ("overlap", "wait")}
    e_base = release_recovery_experiment(
        track_id, proxy_threshold=b_threshold.value,
        sampling_interval_hours=b_interval.value, rollback_minutes=e_rollback.value,
        affected_fraction=e_affected_fraction, incident_cause=e_cause.value,
    )
    e_result = release_recovery_experiment(
        track_id, proxy_threshold=b_threshold.value,
        sampling_interval_hours=b_interval.value, rollback_minutes=e_rollback.value,
        affected_fraction=e_affected_fraction, restore=e_restore.value,
        inflight=e_inflight.value, logs=e_logs.value, incident_cause=e_cause.value,
    )

    return a_base, a_result, b_base, b_result, c_scheduled, c_triggered, d_results, d_capacity_pair, e_affected_fraction, e_base, e_result


@app.cell
def _(d_capacity_pair, d_reservation, parity_baseline, parity_changed,
    a_base, a_prediction, a_result, b_base, b_prediction, b_result,
    c_choice, c_prediction, c_scheduled, c_triggered, capture_evidence,
    d_choice, d_prediction, d_rejected, d_results, e_base, e_prediction,
    e_result, mo, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"baseline": a_base["inputs"], "result": a_result["inputs"]}
    b_upstream = {"baseline": parity_baseline["inputs"], "result": parity_changed["inputs"]}
    c_upstream = {"choice": c_choice.value, "scheduled": c_scheduled["inputs"], "evidence_triggered": c_triggered["inputs"]}
    d_upstream = {"choice": d_choice.value, "rejected": d_rejected.value, "reservation_plan": d_reservation.value}
    e_upstream = {"rollout": d_upstream, "baseline": e_base["inputs"], "result": e_result["inputs"]}

    a_capture = mo.ui.button(
        label="Capture proxy comparison", kind="success",
        disabled=a_prediction.value is None or a_base["inputs"] == a_result["inputs"],
        on_click=lambda _value: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs=a_upstream, baseline=a_base, result=a_result,
            upstream_inputs=a_upstream, model_key="v1_14.monitoring",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture keyed replay comparison", kind="success",
        disabled=b_prediction.value is None or parity_baseline["inputs"] == parity_changed["inputs"],
        on_click=lambda _value: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs=b_upstream, baseline=parity_baseline, result=parity_changed,
            upstream_inputs=b_upstream, model_key="v1_14.parity",
        )),
    )
    c_chosen_result = (
        c_scheduled if c_choice.value == "scheduled"
        else (c_triggered if c_choice.value == "evidence_triggered" else None)
    )
    c_capture = mo.ui.button(
        label="Capture retraining comparison", kind="success",
        disabled=c_prediction.value is None or c_choice.value is None,
        on_click=lambda _value: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs=c_upstream, baseline=c_scheduled, result=c_triggered,
            upstream_inputs=c_upstream, alternatives=(c_scheduled, c_triggered),
            decision=c_choice.value, model_key="v1_14.retraining",
            chosen_result=c_chosen_result,
        )),
    )
    d_invalid = (
        d_prediction.value is None or d_choice.value is None or d_rejected.value is None
        or d_choice.value == d_rejected.value
        or (d_choice.value not in (None, "none") and d_results[d_choice.value]["decision"] != "promote")
    )
    d_baseline = d_results[d_rejected.value] if d_rejected.value is not None else d_results[0.10]
    if d_choice.value == "none":
        d_result = d_baseline
        d_baseline = d_results[0.0]
        d_chosen_result = d_baseline
        d_result_role = "rejected alternative"
    elif d_choice.value is None:
        d_result = d_results[0.25]
        d_chosen_result = d_result
        d_result_role = "chosen candidate"
    else:
        d_result = d_results[d_choice.value]
        d_chosen_result = d_result
        d_result_role = "chosen candidate"
    d_capture = mo.ui.button(
        label="Capture rollout decision", kind="success", disabled=d_invalid,
        on_click=lambda _value: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={**d_upstream, "baseline": d_baseline["inputs"], "result": d_result["inputs"], "chosen_result": d_chosen_result["inputs"]},
            baseline=d_baseline, result=d_result, upstream_inputs=d_upstream,
            alternatives=(*tuple(d_results.values()), {"supplemental_reference": {plan: {"model_key": "v1_14.rollout", "inputs": value["inputs"], "result": value} for plan, value in d_capacity_pair.items()}}), decision=d_choice.value,
            model_key="v1_14.rollout", chosen_result=d_chosen_result,
            result_role=d_result_role,
        )),
    )
    e_capture = mo.ui.button(
        label="Capture incident replay", kind="success",
        disabled=(
            e_prediction.value is None or d_choice.value is None or d_rejected.value is None
            or e_base["inputs"] == e_result["inputs"]
        ),
        on_click=lambda _value: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"baseline": e_base["inputs"], "result": e_result["inputs"]},
            baseline=e_base, result=e_result, upstream_inputs=e_upstream,
            alternatives=(e_base, e_result), decision=e_result["met_recovery_objective"],
            model_key="v1_14.release_recovery",
        )),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream, e_capture, e_upstream


@app.cell
def _(Scenarios, get_lab_metadata, get_learning_path, mo, notebook_bridge,
      notebook_header, notebook_learning_map, notebook_shell, track, track_id):
    from mlsysbook_labs.notebook import notebook_scenario as _scenario
    learning_path = get_learning_path(1, 14)
    journey = Scenarios.DeploymentJourneys.get(track_id)
    episode = journey.episode(14)
    learning_path["narrative"] = {
        "project": journey.mission,
        "scene": episode.scene,
        "carry_forward": episode.handoff,
        "scope": 'The supplied outcome traces are controlled operations fixtures. TinyML and MobileML use a companion host for retraining. Finite label gates, declared allocation screens, and request-bundle retests do not certify a deployed release.',
    }
    _previous = journey.episode(13).handoff
    notebook_shell([
        notebook_header(get_lab_metadata("vol1/lab_14_ml_ops.py"),
                        'Which observations justify changing the deployed system?',
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
def _(fmt_memory, fmt_time, Q_, d_reservation, d_capacity_pair, e_restore, e_inflight, e_logs, e_cause, notebook_tab_hint, notebook_learning_part, parity_baseline, parity_changed, parity_transform, parity_bundle, learning_path, notebook_shell, notebook_status, notebook_table,
    COLORS, a_base, a_capture, a_prediction, a_result, a_threshold, a_upstream,
    apply_plotly_theme, audit_evidence, b_base, b_capture, b_interval,
    b_prediction, b_result, b_threshold, b_upstream, c_capture, c_choice,
    c_interval, c_prediction, c_scheduled, c_triggered, c_upstream, d_capture,
    d_choice, d_prediction, d_rejected, d_results, d_upstream,
    e_affected_fraction, e_base, e_capture, e_prediction, e_result,
    e_rollback, e_upstream, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence,
    go, mo, rationale, scenario, track_id,
):
    _captures = get_evidence()
    upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return notebook_table(rows)

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture after changing the live comparison."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div style="border-left:3px solid #247a4d;padding:8px 12px;background:#f8fafc"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; saved values remain fixed when live controls move.</small></div>')

    def part_a():
        intro = mo.md(f"The {scenario.label} proxy rises before delayed outcomes arrive. Choose a threshold, commit a prediction, and compare proxy alerts with eventual labels.")
        if a_prediction.value is None:
            return mo.vstack([intro, a_threshold, a_prediction])
        figure = go.Figure()
        figure.add_scatter(x=a_result["time_hours"], y=a_result["outcome_quality_pct"], name="Outcome quality", line={"color": COLORS["BlueLine"]})
        figure.add_scatter(x=a_result["time_hours"], y=a_result["proxy_values"], name="Proxy", yaxis="y2", line={"color": COLORS["OrangeLine"]})
        figure = apply_plotly_theme(figure)
        figure.update_layout(
            height=350, margin=dict(l=20, r=55, t=20, b=110),
            xaxis_title="Event time (hours)", yaxis_title="Outcome quality (%)",
            yaxis2=dict(title="Proxy value", overlaying="y", side="right"),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.30, yanchor="top"),
        )
        rows = [
            {"Run": "Balanced baseline", "Alerts": a_base["investigations"], "False investigations": a_base["false_investigations"], "Missed failures": a_base["missed_failures"], "Label confirmation": f'{a_base["first_outcome_confirmation_hours"]:g} h'},
            {"Run": "Selected threshold", "Alerts": a_result["investigations"], "False investigations": a_result["false_investigations"], "Missed failures": a_result["missed_failures"], "Label confirmation": f'{a_result["first_outcome_confirmation_hours"]:g} h'},
        ]
        return mo.vstack([
            intro, a_threshold, a_prediction, figure, table(rows),
            mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. **Observed replay:** the selected proxy produced **{a_result['false_investigations']} false investigations** and **{a_result['missed_failures']} missed failure observations**. Outcome confirmation remained delayed until **{a_result['first_outcome_confirmation_hours']:g} h**."), kind="info"),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md("MLSysIM compares each sampled proxy event with its supplied eventual outcome. A proxy alert starts an investigation; it does not change or prove the outcome. Label availability equals event time plus the scenario's label delay.")}),
        ])

    def part_b():
        intro = mo.md("The release owner has a plausible version name and a matching aggregate screen. Predict what establishes per-request parity, then test a transformation and the bundle retained for recovery.")
        controls = mo.hstack([parity_transform, parity_bundle], widths="equal", wrap=True)
        if b_prediction.value is None:
            return mo.vstack([intro, controls, b_prediction])
        rows = [{"Request": row["request_id"], "Raw": row["raw_value"], "Training": row["training_value"], "Serving": row["serving_value"], "Parity": "PASS" if row["matches"] else "FAIL"} for row in parity_changed["rows"]]
        outcomes = table([{"Check": "Aggregate values", "Selected": "match" if parity_changed["aggregate_distribution_matches"] else "differ"},
                          {"Check": "Keyed mismatches", "Selected": parity_changed["mismatch_count"]},
                          {"Check": "Replay available", "Selected": parity_changed["replay_available"]},
                          {"Check": "Compatible retest", "Selected": parity_changed["compatible_retest_passes"]}])
        return mo.vstack([intro, controls, b_prediction, table(rows), outcomes,
            mo.callout(mo.md(f"The selected path has **{parity_changed['mismatch_count']} keyed mismatches**. Aggregate agreement is **{parity_changed['aggregate_distribution_matches']}**; compatible replay passes **{parity_changed['compatible_retest_passes']}**. Those checks answer different questions."), kind="info"),
            b_capture, saved("B"), mo.md("**Explain:** which exact request or missing artifact changes your diagnosis? Try a compatible restoration and a weights-only rollback before deciding what the team can reproduce."),
            mo.accordion({"Monitoring policy for subsequent operations": mo.vstack([b_threshold, b_interval, table([{"Policy": "Baseline", "False investigations": b_base["false_investigations"], "Missed failures": b_base["missed_failures"]}, {"Policy": "Selected", "False investigations": b_result["false_investigations"], "Missed failures": b_result["missed_failures"]}])]),
                          "Calculation Notes": mo.md(parity_changed["limitation"] + " A complete release identity binds code, data, configuration, environment, weights, and fitted state. An identity alone does not establish that a missing artifact can be recovered.")})])

    def part_c():
        host_note = (
            f" Retraining for on-device models executes on a companion development host (1 core for {scenario.retraining.training_duration.to('hour').magnitude:g} hours), not on the microcontroller."
            if track_id in ("tinyml", "mobile") else ""
        )
        intro = mo.md(f"Compare a calendar schedule with a policy that waits for delayed failure evidence.{host_note} Both policies use the same job duration, resources, and supplied candidate outcome traces.")
        controls = mo.hstack([c_interval, c_choice], widths="equal", wrap=True)
        if c_prediction.value is None:
            return mo.vstack([intro, controls, c_prediction])
        figure = go.Figure([
            go.Bar(name="Retraining cost", x=["Scheduled", "Evidence-triggered"], y=[c_scheduled["retraining_cost_usd"], c_triggered["retraining_cost_usd"]], marker_color=COLORS["BlueLine"]),
            go.Bar(name="Stale-outcome loss", x=["Scheduled", "Evidence-triggered"], y=[c_scheduled["stale_outcome_loss_usd"], c_triggered["stale_outcome_loss_usd"]], marker_color=COLORS["OrangeLine"]),
        ])
        figure.update_layout(barmode="stack", height=285, margin=dict(l=20, r=20, t=20, b=30), yaxis_title="Scenario cost (USD)", legend_orientation="h")
        rows = [
            {"Policy": "Scheduled", "Jobs": c_scheduled["job_count"], "Promotions": c_scheduled["promotions"], "Resource-hours": f'{c_scheduled["compute_resource_hours"]:g}', "Total cost": f'${c_scheduled["total_operating_cost_usd"]:,.2f}'},
            {"Policy": "Evidence-triggered", "Jobs": c_triggered["job_count"], "Promotions": c_triggered["promotions"], "Resource-hours": f'{c_triggered["compute_resource_hours"]:g}', "Total cost": f'${c_triggered["total_operating_cost_usd"]:,.2f}'},
        ]
        calc_note = (
            f"The displayed square-root approximation is **{c_scheduled['approximate_optimal_interval_hours']:.1f} hours** for this illustrative quadratic-loss assumption. "
            f"The replay remains authoritative because delayed labels, discrete jobs, validation, and supplied candidate outcomes can violate the approximation's assumptions."
            + (" In on-device tracks, retraining jobs execute on companion host compute rather than on constrained local device hardware." if track_id in ("tinyml", "mobile") else "")
        )
        _c_choice_note = (
            f" Selected policy: **{'Scheduled' if c_choice.value == 'scheduled' else 'Evidence-triggered'}**."
            if c_choice.value is not None
            else " Select a retraining policy above to commit your operational decision."
        )
        return mo.vstack([
            intro, controls, c_prediction, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. **Replay:** scheduled retraining launched **{c_scheduled['job_count']} jobs** with **${c_scheduled['stale_outcome_loss_usd']:,.2f}** stale-outcome loss; evidence-triggered retraining launched **{c_triggered['job_count']}** with **${c_triggered['stale_outcome_loss_usd']:,.2f}** stale-outcome loss.{_c_choice_note}"), kind="info"),
            c_capture, saved("C"),
            mo.accordion({"Calculation Notes": mo.md(calc_note)}),
        ])

    def part_d():
        intro = mo.md("The team proposes a tiny canary while another job owns part of the available allocation. Predict what traffic splitting changes, then test whether two complete artifacts fit. Waiting can recover capacity, but the observation deadline stays fixed. These are declared shared-pool allocations, not measured deployment capacity; the TinyML reference reserves flash banks rather than simultaneous inference SRAM.")
        if d_prediction.value is None:
            return mo.vstack([intro, d_reservation, d_prediction])
        profile = d_capacity_pair["overlap"]["allocation"]
        allocations = table([{"Reservation": "Physical device / bank capacity", "Amount": profile["physical_display"]},
                             {"Reservation": "Protected other work", "Amount": profile["protected_display"]},
                             {"Reservation": "One complete artifact", "Amount": profile["copy_display"]},
                             {"Reservation": "Active background job", "Amount": profile["background_display"]}])
        capacity_rows = [{"Plan": plan, "Full copies": result["resident_copies"], "Required": fmt_memory(Q_(result["required_bytes"], "byte")),
                          "Available": fmt_memory(Q_(result["available_bytes"], "byte")), "Capacity": "PASS" if result["capacity_fits"] else "FAIL",
                          "Launch delay": fmt_time(Q_(result["launch_delay_hours"], "hour"), unit="hour", precision=None), "Mature candidate labels": result["candidate_labeled"]} for plan, result in d_capacity_pair.items()]
        rows = [{"Canary": f"{result['canary_fraction_pct']:g}%", "Full copies": result["resident_copies"], "Capacity": "PASS" if result["capacity_fits"] else "FAIL",
                 "Exposed": result["exposed_requests"], "Candidate labels": result["candidate_labeled"], "Cohort gate": "PASS" if result["evidence_requirements_met"] else "FAIL", "Decision": result["decision"].replace("_", " ")} for result in d_results.values()]
        figure = go.Figure()
        for label, color in (("Required", COLORS["RedLine"]), ("Available", COLORS["BlueLine"])):
            key = "required_bytes" if label == "Required" else "available_bytes"
            figure.add_bar(name=label, x=list(d_capacity_pair), y=[result[key] for result in d_capacity_pair.values()], marker_color=color)
        figure.update_layout(barmode="group", height=260, yaxis_title="Declared allocation (bytes)", margin=dict(l=35,r=15,t=15,b=30),legend_orientation="h")
        _selected = d_results[0.0] if d_choice.value in (None, "none") else d_results[d_choice.value]
        _message = ("Hold the rollout; no candidate is installed for the zero-traffic baseline." if d_choice.value == "none" else
                    "Choose a rollout or a justified hold after comparing capacity and evidence." if d_choice.value is None else
                    f"Selected case: {_selected['decision'].replace('_', ' ')}. Capacity and outcome evidence must both permit promotion.")
        return mo.vstack([intro, d_reservation, d_prediction, mo.md(f"**Allocation reference:** {profile['artifact']} · {profile['resource']}"),
            apply_plotly_theme(figure), table(capacity_rows), table(rows), mo.hstack([d_choice,d_rejected],widths="equal",wrap=True),
            mo.callout(mo.md(_message + " A failed allocation executes no candidate requests; its absent labels cannot justify a release."),kind="info"),
            mo.md("**Explain:** lower the nonzero traffic share without changing the staging plan. Does the candidate copy shrink? Then wait for the reservation: what capacity is recovered, and which labels miss the unchanged observation deadline?"),
            d_capture,saved("D"),mo.accordion({"Allocation and calculation notes":mo.vstack([allocations,mo.md(d_capacity_pair["wait"]["limitation"])])})])

    def part_e():
        intro = mo.md("An incident commander offers a fast weights-only rollback. The release owner asks what a new request and an already-admitted candidate request will actually execute. Keep the rollback duration fixed while repairing the release unit, request handling, and degraded-data obligation. A separate cause change tests whether rollback addresses this incident at all.")
        controls = mo.vstack([mo.hstack([e_restore,e_inflight],widths="equal",wrap=True),mo.hstack([e_logs,e_cause],widths="equal",wrap=True),e_rollback])
        if e_prediction.value is None:
            return mo.vstack([intro,controls,e_prediction])
        requests = []
        for row in e_result["requests"]:
            requests.append({"Request":row["request_id"],"Expected": "/".join(row["expected_bundle"].values()),
                             "Actually selected": "/".join(row["actual_bundle"].values()),
                             "Disposition":"Cancellation confirmed" if row["canceled_confirmed"] else "Completed",
                             "Contract":"Not executed" if row["compatible"] is None else "PASS" if row["compatible"] else "FAIL"})
        checks = table([{"Gate":"Serving request contract","Result":e_result["serving_contract_restored"]},
                        {"Gate":"Cause addressed by rollback","Result":e_result["cause_repaired"]},
                        {"Gate":"Degraded logs quarantined","Result":e_result["degraded_logs_quarantined"]},
                        {"Gate":"Fixture recovery verified","Result":e_result["verified_recovery"]}])
        rows = [{"Policy":label,"Verified recovery":result["verified_recovery"],
                 "Recovery time": "Unavailable" if result["mean_time_to_recovery_hours"] is None else fmt_time(Q_(result["mean_time_to_recovery_hours"],"hour"),unit="hour", precision=None),
                 "Candidate copies retained while draining":result["retained_candidate_copies"],"Confirmed cancellations":result["confirmed_cancellations"]}
                for label,result in (("Weights-only baseline",e_base),("Selected response",e_result))]
        _failed = "; ".join(e_result["failure_explanations"]) or "none in the declared fixture"
        return mo.vstack([intro,controls,e_prediction,table(requests),checks,table(rows),
            mo.callout(mo.md(f"**Observed blockers:** {_failed}. The nominal route-change duration is not proof that request contracts or future training data were repaired."),kind="success" if e_result["verified_recovery"] else "warn"),
            mo.md("**Explain:** restore the complete bundle but mishandle the in-flight request; then compare pinned drain with confirmed cancellation. Which work is retained or lost? Finally change the incident cause: which valid contract still lacks behavioral evidence?"),
            e_capture,saved("E"),mo.accordion({"Calculation notes":mo.md(e_result["limitation"] + " These are component-version compatibility checks, not request-output or cache-content replay. The drain uses the supplied validation duration as an additional declared wait; cancellation is assumed confirmed. End-to-end cancellation protocols and unknown post-cutover harm are not modeled.")})])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            rows.append({
                "Part": part,
                "Original prediction": capture.to_dict()["prediction"] if capture else "—",
                "Evidence": "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING"),
            })
        _d_decision = _captures["D"].to_dict()["decision"] if "D" in _captures else None
        complete = (
            audit.complete
            and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
            and bool(rationale.value.strip()) and final_choice.value != final_rejected.value
            and final_choice.value == _d_decision
        )
        return mo.vstack([
            mo.md("### Synthesis · Defend the operations policy (5 min)\nUse saved evidence to name the selected rollout, quantify one rejected tested alternative, state the remaining limitation, and give a measurable reevaluation trigger."),
            table(rows),
            mo.callout(mo.md("Saved snapshots preserve original predictions and exact simulator inputs. Recapture any stale comparison before generating the report."), kind="info"),
            mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
            mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale,
            mo.callout(mo.md("**Ready for the local report.**" if complete else "Complete five current contrasts, match the final recommendation to Part D, choose a different tested alternative, and add the rationale."), kind="success" if complete else "warn"),
        ])

    tabs = mo.ui.tabs({
        "Part A": notebook_learning_part(learning_path, "A", part_a()), "Part B": notebook_learning_part(learning_path, "B", part_b()), "Part C": notebook_learning_part(learning_path, "C", part_c()),
        "Part D": notebook_learning_part(learning_path, "D", part_d()), "Part E": notebook_learning_part(learning_path, "E", part_e()), "Synthesis": build_synthesis(),
    })
    notebook_shell([notebook_tab_hint(), tabs], volume=1)
    return (audit,)


@app.cell(hide_code=True)
def _():
    from mlsysbook_labs.learning_paths import get_learning_path as _submission_path
    from mlsysbook_labs.notebook import notebook_shell as _submission_shell, notebook_submission as _submission
    _submission_shell([_submission(_submission_path(1, 14))])
    return


@app.cell
def _(learning_path, notebook_shell,
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, rationale,
    report_export_panel, scenario, track_id,
):
    _captures = get_evidence()
    _d_decision = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip()) and final_choice.value != final_rejected.value
        and final_choice.value == _d_decision
    )
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not _ready, _download_readiness(volume=1))
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    c_reported = snapshots["C"]["chosen_result"] or snapshots["C"]["result"]
    report = build_lab_report(
        get_lab_metadata("vol1/lab_14_ml_ops.py"), track=track_id, scenario=scenario.label,
        learning_objectives=learning_path["learning_objectives"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "alternatives": snapshots[part]["alternatives"]} for part in "ABCDE"},
        binding_constraints={
            "A": {"false_investigations": snapshots["A"]["result"]["false_investigations"], "missed_failures": snapshots["A"]["result"]["missed_failures"]},
            "B": {"keyed_mismatches": snapshots["B"]["result"]["mismatch_count"], "compatible_retest_passes": snapshots["B"]["result"]["compatible_retest_passes"]},
            "C": {"jobs": c_reported["job_count"], "stale_outcome_loss_usd": c_reported["stale_outcome_loss_usd"]},
            "D": {"decision": snapshots["D"]["decision"], "cohort_gate": (snapshots["D"]["chosen_result"] or snapshots["D"]["result"])["evidence_requirements_met"]},
            "E": {"verified_recovery": snapshots["E"]["result"]["verified_recovery"], "met_objective": snapshots["E"]["result"]["met_recovery_objective"]},
        },
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=[
            "Proxy drift starts investigation; delayed outcomes establish whether quality failed.",
            "Retraining changes the deployed model only after resource-consuming work and promotion evidence.",
            "Canary evidence and verified rollback bound different parts of release risk.",
        ],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": snapshots, "recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": scenario.scenario_note, "calculations": "MLSysIM outcome-labeled operations replay."},
    )
    notebook_shell([report_export_panel(report)], volume=1)
    return (report,)




@app.cell
def _(notebook_shell,
    audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _d_decision = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip()) and final_choice.value != final_rejected.value
        and final_choice.value == _d_decision
    )
    save_discussion = mo.ui.run_button(label="Save discussion", disabled=not _ready)
    notebook_shell([save_discussion, mo.md("When ready, downloading is separate from saving locally. Editing your evidence or discussion requires saving again.")], volume=1)
    return (save_discussion,)


@app.cell
async def _(notebook_shell, notebook_status, save_discussion,
    audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _d_decision = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip()) and final_choice.value != final_rejected.value
        and final_choice.value == _d_decision
    )
    _save_status = ("Ready to save" if _ready else "Evidence in progress")
    _status_class = "hud-active"
    if _ready and save_discussion.value:
        try:
            ledger.save(chapter=14, design={
                "schema_version": 1, "lab_id": "v1_14", "track_id": track_id,
                "model_id": "v1_14_experiments",
                "evidence": {part: capture.to_dict() for part, capture in get_evidence().items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value, "rationale": rationale.value,
            })
            await ledger.flush()
            _save_status = "SAVED"
        except Exception:
            _save_status = "SAVE FAILED — REPORT STILL AVAILABLE"
            _status_class = "hud-failed"
    notebook_shell([notebook_status(_save_status, "Editing requires saving again; the downloaded discussion is separate from local storage.")], volume=1)
    return


if __name__ == "__main__":
    app.run()
