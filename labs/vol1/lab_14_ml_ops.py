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
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        labs_dir = Path(__file__).resolve().parents[1]
        if str(labs_dir) not in sys.path:
            sys.path.insert(0, str(labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.engine.v1_14_experiments import (
        TRACKS, canary_experiment, experiment_options, get_track_scenario,
        incident_experiment, monitoring_experiment, retraining_experiment,
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
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Deployment track",
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
        "Fewer misses, more investigations": "misses_down_investigations_up",
        "Fewer misses, lower telemetry cost": "misses_down_cost_down",
        "No operational consequence": "no_change",
    }, label="What is the main cost of greater monitoring sensitivity?").form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio({
        "Scheduled always costs less": "scheduled_cheaper",
        "Evidence-triggered may wait longer but launch fewer jobs": "triggered_tradeoff",
        "Retraining changes quality when the job starts": "changes_at_start",
    }, label="How will delayed evidence change retraining?").form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio({
        "Zero traffic is safest and still proves readiness": "zero_certifies",
        "More traffic always lowers exposure": "more_always_safer",
        "Traffic trades faster evidence against candidate exposure": "evidence_exposure",
    }, label="What does canary fraction control?").form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio({
        "Detection alone completes recovery": "detection_is_recovery",
        "Recovery is the sum of response stages": "sequential_stages",
        "Rollback duration cannot affect exposure": "rollback_noncausal",
    }, label="What determines verified recovery time?").form(submit_button_label="Lock Part E prediction")
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
def _(
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
        fraction: canary_experiment(track_id, canary_fraction=fraction, stop_on_decision=False)
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
    e_base = incident_experiment(
        track_id, proxy_threshold=b_threshold.value,
        sampling_interval_hours=b_interval.value, rollback_minutes=30.0,
        affected_fraction=e_affected_fraction,
    )
    e_result = incident_experiment(
        track_id, proxy_threshold=b_threshold.value,
        sampling_interval_hours=b_interval.value, rollback_minutes=e_rollback.value,
        affected_fraction=e_affected_fraction,
    )
    return a_base, a_result, b_base, b_result, c_scheduled, c_triggered, d_results, e_affected_fraction, e_base, e_result


@app.cell
def _(
    a_base, a_prediction, a_result, b_base, b_prediction, b_result,
    c_choice, c_prediction, c_scheduled, c_triggered, capture_evidence,
    d_choice, d_prediction, d_rejected, d_results, e_base, e_prediction,
    e_result, mo, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"baseline": a_base["inputs"], "result": a_result["inputs"]}
    b_upstream = {"baseline": b_base["inputs"], "result": b_result["inputs"]}
    c_upstream = {"choice": c_choice.value, "scheduled": c_scheduled["inputs"], "evidence_triggered": c_triggered["inputs"]}
    d_upstream = {"choice": d_choice.value, "rejected": d_rejected.value}
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
        label="Capture sensitivity comparison", kind="success",
        disabled=b_prediction.value is None or b_base["inputs"] == b_result["inputs"],
        on_click=lambda _value: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs=b_upstream, baseline=b_base, result=b_result,
            upstream_inputs=b_upstream, model_key="v1_14.monitoring",
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
            alternatives=tuple(d_results.values()), decision=d_choice.value,
            model_key="v1_14.canary", chosen_result=d_chosen_result,
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
            model_key="v1_14.incident",
        )),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream, e_capture, e_upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, scenario, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:760px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.track-control{margin:0 0 10px 2px;max-width:330px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}
    .lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}.lab-hud .hud-failed{color:#fca5a5}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.table-wrap{max-width:100%;overflow-x:auto}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 14</span><span>ABOUT 50–55 MIN</span></div><h1>Evidence-Driven Operations</h1><p>Which monitoring, retraining, release, and recovery policy keeps a deployed model useful?</p><div class="pilot-meta"><div><b>Track</b><br>{scenario.label}</div><div><b>Investigation</b><br>One production ML node</div><div><b>Deliverable</b><br>Operations runbook memo</div></div></section>""")
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, css,
        mo.Html('<div class="track-control"><b>Choose a deployment track</b></div>'),
        track, header,
        mo.Html('<p class="pilot-note">All tracks use illustrative outcome-labeled scenarios. The values are teaching fixtures, not measurements of a named production system. On-device tracks (TinyML/Mobile) evaluate single-node inference with companion-host retraining and local partition rollback.</p>'),
    ], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
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
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture after changing the live comparison."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; saved values remain fixed when live controls move.</small></div>')

    def part_a():
        intro = mo.md(f"### A · Does drift prove the model is wrong? (9 min)\nThe {scenario.label} proxy rises before delayed outcomes arrive. Choose a threshold, commit a prediction, and compare proxy alerts with eventual labels.")
        if a_prediction.value is None:
            return mo.vstack([intro, a_threshold, a_prediction])
        figure = go.Figure()
        figure.add_scatter(x=a_result["time_hours"], y=a_result["outcome_quality_pct"], name="Outcome quality", line={"color": COLORS["BlueLine"]})
        figure.add_scatter(x=a_result["time_hours"], y=a_result["proxy_values"], name="Proxy", yaxis="y2", line={"color": COLORS["OrangeLine"]})
        figure.update_layout(height=285, margin=dict(l=20, r=55, t=20, b=35), xaxis_title="Event time (hours)", yaxis_title="Outcome quality (%)", yaxis2=dict(title="Proxy value", overlaying="y", side="right"), legend_orientation="h")
        rows = [
            {"Run": "Balanced baseline", "Alerts": a_base["investigations"], "False investigations": a_base["false_investigations"], "Missed failures": a_base["missed_failures"], "Label confirmation": f'{a_base["first_outcome_confirmation_hours"]:g} h'},
            {"Run": "Selected threshold", "Alerts": a_result["investigations"], "False investigations": a_result["false_investigations"], "Missed failures": a_result["missed_failures"], "Label confirmation": f'{a_result["first_outcome_confirmation_hours"]:g} h'},
        ]
        return mo.vstack([
            intro, a_threshold, a_prediction, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. **Observed replay:** the selected proxy produced **{a_result['false_investigations']} false investigations** and **{a_result['missed_failures']} missed failure observations**. Outcome confirmation remained delayed until **{a_result['first_outcome_confirmation_hours']:g} h**."), kind="info"),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md("MLSysIM compares each sampled proxy event with its supplied eventual outcome. A proxy alert starts an investigation; it does not change or prove the outcome. Label availability equals event time plus the scenario's label delay.")}),
        ])

    def part_b():
        intro = mo.md("### B · How sensitive should monitoring be? (10 min)\nCompare one selected threshold and cadence with the balanced baseline. The operating costs are investigations, missed failures, observation delay, and telemetry expense.")
        if b_prediction.value is None:
            return mo.vstack([intro, mo.hstack([b_threshold, b_interval], widths="equal", wrap=True), b_prediction])
        figure = go.Figure([
            go.Bar(name="False investigations", x=["Balanced", "Selected"], y=[b_base["false_investigations"], b_result["false_investigations"]], marker_color=COLORS["OrangeLine"]),
            go.Bar(name="Missed failures", x=["Balanced", "Selected"], y=[b_base["missed_failures"], b_result["missed_failures"]], marker_color=COLORS["RedLine"]),
        ])
        figure.update_layout(barmode="group", height=270, margin=dict(l=20, r=20, t=20, b=30), yaxis_title="Event count", legend_orientation="h")
        b_base_delay_str = "undetected" if b_base["detection_delay_hours"][0] is None else f'{b_base["detection_delay_hours"][0]:g} h'
        b_res_delay_str = "undetected" if b_result["detection_delay_hours"][0] is None else f'{b_result["detection_delay_hours"][0]:g} h'
        rows = [
            {"Run": "Balanced", "Observations": b_base["observations_collected"], "Investigations": b_base["investigations"], "Telemetry cost": f'${b_base["telemetry_cost_usd"]:,.2f}', "Detection delay": b_base_delay_str},
            {"Run": "Selected", "Observations": b_result["observations_collected"], "Investigations": b_result["investigations"], "Telemetry cost": f'${b_result["telemetry_cost_usd"]:,.2f}', "Detection delay": b_res_delay_str},
        ]
        return mo.vstack([
            intro, mo.hstack([b_threshold, b_interval], widths="equal", wrap=True), b_prediction,
            apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. **Selected result:** **{b_result['false_investigations']} false investigations**, **{b_result['missed_failures']} misses**, and **${b_result['telemetry_cost_usd']:,.2f}** in direct scenario cost. Monitoring changes observation and response; the supplied model outcomes stay fixed."), kind="danger" if b_result["missed_failures"] else "info"),
            b_capture, saved("B"),
            mo.accordion({"Calculation Notes": mo.md("MLSysIM samples the same outcome-labeled trace at the selected cadence. Direct cost equals collected observations plus investigations under the track's illustrative cost assumptions. No readiness score is used.")}),
        ])

    def part_c():
        host_note = (
            f" Retraining for on-device models executes on a companion development host (1 core for {scenario.retraining.training_duration.to('hour').magnitude:g} hours), not on the microcontroller."
            if track_id in ("tinyml", "mobile") else ""
        )
        intro = mo.md(f"### C · When should we retrain? (10 min)\nCompare a calendar schedule with a policy that waits for delayed failure evidence.{host_note} Both policies use the same job duration, resources, and supplied candidate outcome traces.")
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
        lens_note = (
            " On single embedded devices, canary request-splitting represents local shadow execution or session staging, as microcontrollers lack reverse-proxy traffic routers."
            if track_id in ("tinyml", "mobile") else ""
        )
        intro = mo.md(f"### D · What evidence justifies promotion? (11 min)\nCompare the tested canary fractions under one common horizon.{lens_note} The gate requires labeled candidate and baseline outcomes plus minimum coverage in every required cohort. It is an operational evidence rule, not a claim of statistical significance.")
        rows = [
            {"Canary": f"{result['canary_fraction_pct']:g}%", "Exposed": result["exposed_requests"], "Candidate labels": result["candidate_labeled"], "Exposed failures": result["exposed_failures"], "Cohort gate": "PASS" if result["evidence_requirements_met"] else "FAIL", "Decision": result["decision"].upper()}
            for result in d_results.values()
        ]
        if d_prediction.value is None:
            return mo.vstack([intro, d_prediction])
        figure = go.Figure([
            go.Bar(name="Candidate labels", x=[f"{result['canary_fraction_pct']:g}%" for result in d_results.values()], y=[result["candidate_labeled"] for result in d_results.values()], marker_color=COLORS["BlueLine"]),
            go.Bar(name="Exposed failures", x=[f"{result['canary_fraction_pct']:g}%" for result in d_results.values()], y=[result["exposed_failures"] for result in d_results.values()], marker_color=COLORS["RedLine"]),
        ])
        figure.update_layout(barmode="group", height=285, margin=dict(l=20, r=20, t=20, b=30), xaxis_title="Canary traffic", yaxis_title="Observed request count", legend_orientation="h")
        _selected = d_results[0.0] if d_choice.value in (None, "none") else d_results[d_choice.value]
        if d_choice.value == "none":
            _banner = "HOLD RECOMMENDED — candidate outcomes do not justify promotion."
            _callout_kind = "info"
        elif d_choice.value is None:
            _banner = "Select a promotion recommendation to compare against your rejected alternative."
            _callout_kind = "info"
        elif _selected["decision"] == "promote":
            _banner = f"Selected gate decision: PROMOTE after {_selected['candidate_labeled']} candidate labels."
            _callout_kind = "success"
        else:
            _banner = f"Selected gate decision: {_selected['decision'].upper()}."
            _callout_kind = "danger"
        calc_note = (
            "MLSysIM routes the selected traffic fraction, waits for the scenario's delayed labels, counts observed successes, checks required cohort coverage, and applies the stated minimum-evidence margins. More samples alone do not certify significance."
            + (" For single-node embedded systems, canary fractions simulate shadow mode or dual-slot trials without requiring a cloud reverse proxy." if track_id in ("tinyml", "mobile") else "")
        )
        return mo.vstack([
            intro, d_prediction, apply_plotly_theme(figure), table(rows),
            mo.hstack([d_choice, d_rejected], widths="equal", wrap=True),
            mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. **{_banner}** Record a different tested alternative; a no-feasible conclusion is valid only when none of the tested positive fractions supports promotion."), kind=_callout_kind),
            d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md(calc_note)}),
        ])

    def part_e():
        intro = mo.md(f"### E · Does the runbook survive an incident? (8 min)\nReplay the selected monitoring policy and a rollback affecting **{100 * e_affected_fraction:g}%** of requests. Compare the 30-minute rollback baseline with a tested recovery stage.")
        if e_prediction.value is None:
            return mo.vstack([intro, e_rollback, e_prediction])
        figure = go.Figure()
        figure.add_scatter(
            x=[e_result["incident_started_hours"], e_result["detected_hours"], e_result["triage_completed_hours"], e_result["rollback_completed_hours"], e_result["recovery_validated_hours"]],
            y=["Start", "Detect", "Triage", "Rollback", "Validate"], mode="lines+markers",
            marker={"size": 11, "color": COLORS["BlueLine"]}, name="Selected runbook",
        )
        figure.update_layout(height=275, margin=dict(l=70, r=20, t=20, b=35), xaxis_title="Event time (hours)", showlegend=False)
        mttr_base_str = f'{e_base["mean_time_to_recovery_hours"]:.2f} h' if e_base["mean_time_to_recovery_hours"] is not None else "unrecovered"
        mttr_res_str = f'{e_result["mean_time_to_recovery_hours"]:.2f} h' if e_result["mean_time_to_recovery_hours"] is not None else "unrecovered"
        rows = [
            {"Runbook": "30-minute rollback", "MTTR": mttr_base_str, "Exposed requests": e_base["exposed_requests"], "Exposure cost": f'${e_base["exposure_cost_usd"]:,.2f}', "Objective": "PASS" if e_base["met_recovery_objective"] else "FAIL"},
            {"Runbook": "Selected rollback", "MTTR": mttr_res_str, "Exposed requests": e_result["exposed_requests"], "Exposure cost": f'${e_result["exposure_cost_usd"]:,.2f}', "Objective": "PASS" if e_result["met_recovery_objective"] else "FAIL"},
        ]
        callout_msg = (
            f"recovery was verified in **{mttr_res_str}**"
            if e_result["mean_time_to_recovery_hours"] is not None
            else "incident was not recovered"
        )
        calc_note = (
            "MLSysIM orders detection, triage, rollback, and validation in sequence. Exposure ends only when rollback completes. Total incident cost adds harmful-request exposure and response labor once; it contains no overlapping risk score."
            + (" Rapid rollback on embedded or mobile devices relies on local dual-partition (A/B bank) firmware/model switching rather than multi-hour OTA reflashing." if track_id in ("tinyml", "mobile") else "")
        )
        return mo.vstack([
            intro, e_rollback, e_prediction, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. **Selected replay:** {callout_msg}, with **{e_result['exposed_requests']:,} exposed requests** and **${e_result['total_incident_cost_usd']:,.2f}** direct incident cost."), kind="success" if e_result["met_recovery_objective"] else "danger"),
            e_capture, saved("E"),
            mo.accordion({"Calculation Notes": mo.md(calc_note)}),
        ])

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
        "Part A": part_a(), "Part B": part_b(), "Part C": part_c(),
        "Part D": part_d(), "Part E": part_e(), "Synthesis": build_synthesis(),
    })
    tabs
    return (audit,)


@app.cell
def _(
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
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    c_reported = snapshots["C"]["chosen_result"] or snapshots["C"]["result"]
    report = build_lab_report(
        get_lab_metadata("vol1/lab_14_ml_ops.py"), track=track_id, scenario=scenario.label,
        learning_objectives=[
            "Compare drift proxies with delayed outcome labels",
            "Balance monitoring sensitivity, retraining cost, and stale-outcome loss",
            "Require rollout evidence and verify incident recovery",
        ],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "alternatives": snapshots[part]["alternatives"]} for part in "ABCDE"},
        binding_constraints={
            "A": {"false_investigations": snapshots["A"]["result"]["false_investigations"], "missed_failures": snapshots["A"]["result"]["missed_failures"]},
            "B": {"false_investigations": snapshots["B"]["result"]["false_investigations"], "missed_failures": snapshots["B"]["result"]["missed_failures"]},
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
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
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
    _save_status = "EVIDENCE IN PROGRESS"
    _status_class = "hud-active"
    if _ready:
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
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">14 · Evidence-Driven Operations</span><span aria-hidden="true"> · STATUS: </span><span class="{_status_class}">{_save_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
