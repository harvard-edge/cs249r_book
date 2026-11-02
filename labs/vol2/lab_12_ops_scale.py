import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 12: Operating a Shared ML Fleet · MLSysBook")


@app.cell
async def _():
    import sys
    from dataclasses import asdict, is_dataclass
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
    from mlsysim.labs.experiments.v2_12_experiments import (
        evaluate_canary, evaluate_dependency, evaluate_incident,
        evaluate_portfolio, evaluate_release_change, get_operations_track,
        evaluate_coupling, evaluate_monitoring, evaluate_feature_join, evaluate_ownership,
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
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, apply_plotly_theme, asdict,
        audit_evidence, build_lab_report, capture_evidence, evaluate_canary,
        evaluate_dependency, evaluate_incident, evaluate_portfolio,
        evaluate_release_change, evaluate_coupling, evaluate_monitoring, evaluate_feature_join, evaluate_ownership, get_lab_metadata, get_operations_track, go,
        is_dataclass, ledger, mo, report_export_panel,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(ledger, mo, set_evidence):
    from mlsysbook_labs.contexts import preferred_context as _preferred_context
    _track_options = {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"}
    track = mo.ui.dropdown(
        _track_options,
        value=next(label for label, context in _track_options.items() if context == _preferred_context(ledger)), label="Deployment context", on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_operations_track, track):
    track_id = track.value
    profile = get_operations_track(track_id)
    return profile, track_id


@app.cell
def _(mo, track_id):
    _track_key = track_id
    adoption = mo.ui.dropdown(
        {"25%": 0.25, "50%": 0.50, "75%": 0.75, "100%": 1.0},
        value="75%", label="Platform adoption",
    )
    canary_fraction = mo.ui.dropdown(
        {"3%": 0.03, "5%": 0.05, "10%": 0.10}, value="5%", label="Canary traffic",
    )
    canary_hours = mo.ui.slider(4, 48, value=24, step=4, label="Observation duration (hours)")
    response_choice = mo.ui.radio(
        {"Improve detection": "detection", "Improve diagnosis": "diagnosis",
         "Improve mitigation": "mitigation", "Improve recovery": "recovery",
         "Take no action": "none"}, label="Response investment",
    )
    response_rejected = mo.ui.radio(
        {"Detection": "detection", "Diagnosis": "diagnosis",
         "Mitigation": "mitigation", "Recovery": "recovery"},
        label="Rejected alternative",
    )
    coupling = mo.ui.dropdown({"Independent models": "isolated", "One chain": "chain", "Every pair coupled": "dense"}, value="One chain", label="Dependency structure (fixed model count)")
    alert_rate = mo.ui.dropdown({"5% per check": 0.05, "0.3% per check": 0.003}, value="5% per check", label="Calibrated null false-positive rate")
    monitor_incident = mo.ui.dropdown({name: name for name in ("healthy", "stale features", "quality regression", "slow responses", "unavailable")}, value="stale features", label="Service incident")
    monitor_hours = mo.ui.slider(1, 24, value=6, step=1, label="Hours since incident began")
    join_rule = mo.ui.dropdown({"Event time only": "event time only", "Availability time": "availability time"}, value="Event time only", label="Historical feature join")
    useful_utilization = mo.ui.slider(0.0, 1.0, value=0.5, step=0.1, label="Useful capacity utilization")
    return adoption, canary_fraction, canary_hours, response_choice, response_rejected, coupling, alert_rate, monitor_incident, monitor_hours, join_rule, useful_utilization


@app.cell
def _(mo, profile, track_id):
    _track_key = track_id
    release_changed = mo.ui.dropdown(
        {f"Retain {profile.dependency_baseline_version}": False,
         f"Change to {profile.dependency_changed_version}": True},
        value=f"Change to {profile.dependency_changed_version}", label="Candidate upstream version",
    )
    _compat_name = f"{profile.deployment_artifact.replace(' ', '-')}-compatibility"
    release_refresh = mo.ui.multiselect(
        options=["offline-evaluation", "canary-observation", _compat_name], value=[],
        label="Rerun evidence against this exact candidate",
    )
    return release_changed, release_refresh


@app.cell
def _(mo, profile, track_id):
    _track_key = track_id
    _primary_cohort = profile.canary_cohorts[0].name
    a_prediction = mo.ui.radio(
        {"Same new obligations in every graph": "same", "Dense coupling adds more new obligations": "dense_more",
         "A shared platform removes all dependency obligations": "platform_removes"},
        label="At fixed model count, what happens when one model joins the fleet?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"No consumer is blocked": "none", "One consumer is blocked": "one",
         "Two or more consumers are blocked": "multiple"},
        label="How many consumer versions reject the upstream change?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"All cohort requirements pass": "all_pass",
         f"Only the {_primary_cohort} cohort passes": "high_only",
         "At least one cohort lacks observable labels": "labels_pending"},
        label="What evidence will be available when the canary ends?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Healthy HTTP makes promotion safe": "http_enough", "An exhausted freshness budget can block promotion": "vector_gate",
         "Grouping alerts repairs the service budget": "grouping_repairs"},
        label="What happens when features are stale but HTTP responses remain healthy?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"No evidence changes": "none",
         f"Only {profile.deployment_artifact} evidence changes": "package",
         "Offline and canary evidence must be regenerated": "behavior"},
        label="Which evidence becomes stale after the upstream version changes?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Improve detection": "detection", "Improve diagnosis": "diagnosis",
         "Improve mitigation": "mitigation", "Improve recovery": "recovery",
         "Take no action": "none"}, label="Recommended response investment",
    )
    final_rejected = mo.ui.radio(
        {"Detection": "detection", "Diagnosis": "diagnosis",
         "Mitigation": "mitigation", "Recovery": "recovery"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"Portfolio adoption changes": "adoption",
         "A cohort misses its label requirement": "cohort",
         "Incident traffic or stage duration changes": "incident",
         "An artifact version changes": "version"}, label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Illustrative cost assumptions": "cost_assumptions",
         "Unobserved cohort behavior": "cohort_coverage",
         "Incident fractions differ in production": "incident_shape"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Decision rationale",
        placeholder="Cite the saved baseline, chosen result, rejected result, remaining limitation, and trigger.",
     debounce=False)
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    adoption, canary_fraction, canary_hours, evaluate_canary,
    evaluate_dependency, evaluate_incident, evaluate_portfolio,
    evaluate_release_change, evaluate_coupling, evaluate_monitoring, evaluate_feature_join, evaluate_ownership, coupling, alert_rate, monitor_incident, monitor_hours, join_rule, useful_utilization, release_changed, release_refresh, track_id,
):
    coupling_base = evaluate_coupling(track_id, "isolated")
    coupling_result = evaluate_coupling(track_id, coupling.value)
    monitoring_base = evaluate_monitoring(track_id, per_check_false_positive_rate=alert_rate.value, incident="healthy", observation_hours=monitor_hours.value)
    monitoring_result = evaluate_monitoring(track_id, per_check_false_positive_rate=alert_rate.value, incident=monitor_incident.value, observation_hours=monitor_hours.value)
    feature_base = evaluate_feature_join(track_id, "event time only")
    feature_result = evaluate_feature_join(track_id, join_rule.value)
    ownership_base = evaluate_ownership(track_id, 0.0)
    ownership_result = evaluate_ownership(track_id, useful_utilization.value)
    a_base = evaluate_portfolio(track_id, 0.0)
    a_result = evaluate_portfolio(track_id, adoption.value)
    b_base = evaluate_dependency(track_id, False)
    b_result = evaluate_dependency(track_id, True)
    c_base = evaluate_canary(track_id, duration_hours=canary_hours.value, canary_fraction=0.01)
    c_result = evaluate_canary(
        track_id, duration_hours=canary_hours.value, canary_fraction=canary_fraction.value,
    )
    d_base = evaluate_incident(track_id, "baseline")
    d_results = {
        strategy: evaluate_incident(track_id, strategy)
        for strategy in ("detection", "diagnosis", "mitigation", "recovery")
    }
    e_base = evaluate_release_change(track_id, release_changed.value)
    e_result = evaluate_release_change(track_id, release_changed.value, refresh_evidence=tuple(release_refresh.value))
    return a_base, a_result, b_base, b_result, c_base, c_result, d_base, d_results, e_base, e_result, coupling_base, coupling_result, monitoring_base, monitoring_result, feature_base, feature_result, ownership_base, ownership_result


@app.cell
def _(asdict, is_dataclass):
    def serialize(value):
        if is_dataclass(value):
            return {key: serialize(item) for key, item in asdict(value).items()}
        if isinstance(value, dict):
            return {str(key): serialize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [serialize(item) for item in value]
        if hasattr(value, "magnitude") and hasattr(value, "units"):
            return {"magnitude": float(value.magnitude), "unit": f"{value.units:~}"}
        return value

    def run_record(arguments, result):
        return {"inputs": serialize(arguments), "output": serialize(result)}

    return run_record, serialize


@app.cell
def _(
    a_base, a_prediction, a_result, adoption, b_base, b_prediction, b_result,
    c_base, c_prediction, c_result, canary_fraction, canary_hours,
    capture_evidence, d_base, d_prediction, d_results, e_base, e_prediction,
    e_result, mo, response_choice, response_rejected, run_record,
    set_evidence, track_id, coupling, coupling_base, coupling_result, alert_rate, monitor_incident, monitor_hours, monitoring_base, monitoring_result, join_rule, useful_utilization, feature_base, feature_result, ownership_base, ownership_result, release_changed, release_refresh,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    common_upstream = {"track_id": track_id}
    a_capture = mo.ui.button(
        label="Capture platform contrast", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"adoption_fraction": adoption.value, "topology": coupling.value, "useful_utilization": useful_utilization.value},
            baseline=run_record({"track_id": track_id, "adoption_fraction": 0.0, "topology": "isolated", "useful_utilization": 0.0}, {"portfolio": a_base, "coupling": coupling_base, "ownership": ownership_base}),
            result=run_record({"track_id": track_id, "adoption_fraction": adoption.value, "topology": coupling.value, "useful_utilization": useful_utilization.value}, {"portfolio": a_result, "coupling": coupling_result, "ownership": ownership_result}),
            upstream_inputs={**common_upstream, "topology": coupling.value, "useful_utilization": useful_utilization.value},
            decision="adopt" if a_result.pays_back else "defer",
            model_key="evaluate_portfolio + evaluate_coupling + evaluate_ownership",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture dependency trace", kind="success", disabled=b_prediction.value is None,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value, inputs={"changed": True, "join_rule": join_rule.value},
            baseline=run_record({"track_id": track_id, "changed": False, "join_rule": "event time only"}, {"dependency": b_base, "feature": feature_base}),
            result=run_record({"track_id": track_id, "changed": True, "join_rule": join_rule.value}, {"dependency": b_result, "feature": feature_result}),
            upstream_inputs={**common_upstream, "join_rule": join_rule.value}, model_key="evaluate_dependency + evaluate_feature_join",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture canary evidence", kind="success", disabled=c_prediction.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"duration_hours": canary_hours.value, "canary_fraction": canary_fraction.value},
            baseline=run_record(
                {"track_id": track_id, "duration_hours": canary_hours.value, "canary_fraction": 0.01}, c_base,
            ),
            result=run_record(
                {"track_id": track_id, "duration_hours": canary_hours.value,
                 "canary_fraction": canary_fraction.value}, c_result,
            ),
            upstream_inputs=common_upstream, model_key="evaluate_canary",
        )),
    )
    d_invalid = (
        d_prediction.value is None or response_choice.value is None
        or response_rejected.value is None or response_choice.value == response_rejected.value
    )
    selected_strategy = response_rejected.value if response_choice.value == "none" else response_choice.value
    selected_result = d_results.get(selected_strategy, d_base)
    d_capture = mo.ui.button(
        label="Capture response decision", kind="success", disabled=d_invalid,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={"choice": response_choice.value, "tested_strategy": selected_strategy,
                    "rejected": response_rejected.value, "incident": monitor_incident.value, "observation_hours": monitor_hours.value, "per_check_false_positive_rate": alert_rate.value},
            baseline=run_record({"track_id": track_id, "strategy": "baseline", "incident": "healthy", "observation_hours": monitor_hours.value, "per_check_false_positive_rate": alert_rate.value}, {"incident": d_base, "monitoring": monitoring_base}),
            result=run_record({"track_id": track_id, "strategy": selected_strategy, "incident": monitor_incident.value, "observation_hours": monitor_hours.value, "per_check_false_positive_rate": alert_rate.value}, {"incident": selected_result, "monitoring": monitoring_result}),
            upstream_inputs={**common_upstream, "incident": monitor_incident.value, "observation_hours": monitor_hours.value, "per_check_false_positive_rate": alert_rate.value},
            alternatives=[run_record({"track_id": track_id, "strategy": key}, value)
                          for key, value in d_results.items()],
            decision=response_choice.value, model_key="evaluate_incident + evaluate_monitoring",
            chosen_result=(
                run_record({"track_id": track_id, "strategy": "baseline", "incident": monitor_incident.value, "observation_hours": monitor_hours.value, "per_check_false_positive_rate": alert_rate.value}, {"incident": d_base, "monitoring": monitoring_result})
                if response_choice.value == "none" else None
            ),
            result_role=(
                "rejected alternative"
                if response_choice.value == "none" else "tested intervention"
            ),
        )),
    )
    e_capture = mo.ui.button(
        label="Capture release-contract check", kind="success", disabled=e_prediction.value is None,
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"changed": release_changed.value, "refresh_evidence": list(release_refresh.value)},
            baseline=run_record({"track_id": track_id, "changed": release_changed.value, "refresh_evidence": []}, e_base),
            result=run_record({"track_id": track_id, "changed": release_changed.value, "refresh_evidence": list(release_refresh.value)}, e_result),
            upstream_inputs={**common_upstream, "changed": release_changed.value, "refresh_evidence": sorted(release_refresh.value)},
            decision="regenerate" if not e_result.releasable else "release",
            model_key="evaluate_release_change",
        )),
    )
    return a_capture, b_capture, c_capture, d_capture, e_capture


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    from mlsysbook_labs.notebook import (
        notebook_header as _course_header, notebook_shell as _course_shell,
        notebook_learning_map as _course_learning_map,
        notebook_scenario as _course_scenario,
    )
    from mlsysbook_labs.learning_paths import get_learning_path as _course_path
    from mlsysbook_labs import get_lab_metadata as _course_metadata
    css = mo.Html("""
    <style>




    .ops-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0 2px 12px}.saved{border-left:4px solid #16a34a;background:#f0fdf4;padding:9px 12px;border-radius:7px}.table-wrap{max-width:100%;overflow-x:auto}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}

    </style>""")
    header = _course_header(_course_metadata("vol2/lab_12_ops_scale.py"), 'When does shared infrastructure reduce repeated work, and what new obligations does sharing create?')
    _course_shell([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        css,
        header,
        track,
        _course_scenario(_course_path('vol2', 12), track=track.value, volume=2),
        mo.Html('<p class="ops-note">Track values are illustrative fleet scenarios, not measurements of a named product. Calculation Notes state the model boundaries.</p>'),
        _course_learning_map(_course_path('vol2', 12))
    ], gap=0.35, volume=2)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base, a_capture, a_prediction, a_result, adoption,
    apply_plotly_theme, audit_evidence, b_base, b_capture, b_prediction,
    b_result, c_capture, c_prediction, c_result, canary_fraction,
    canary_hours, d_base, d_capture, d_prediction, d_results, e_base,
    e_capture, e_prediction, e_result, final_choice, final_rejected,
    final_risk, final_trigger, get_evidence, go, mo, profile, rationale,
    response_choice, response_rejected, track_id, coupling, coupling_result, alert_rate, monitor_incident, monitor_hours, monitoring_result, join_rule, useful_utilization, feature_result, ownership_result, release_changed, release_refresh,
):
    _saved_prediction_widgets = {'A': a_prediction, 'B': b_prediction, 'C': c_prediction, 'D': d_prediction, 'E': e_prediction}
    from mlsysbook_labs.notebook import notebook_prediction_label as _read_prediction
    from mlsysbook_labs.notebook import notebook_tab_hint as _part_tabs_hint, notebook_shell as _lesson_shell
    from mlsysbook_labs.notebook import notebook_part_reflection as _concept_reflection
    from mlsysbook_labs.learning_paths import get_learning_path as _reflection_path
    _reflection_contract = _reflection_path('vol2', 12)
    _captures = get_evidence()
    _upstream = {part: {"track_id": track_id} for part in "ABCDE"}
    _upstream["A"].update({"topology": coupling.value, "useful_utilization": useful_utilization.value})
    _upstream["B"].update({"join_rule": join_rule.value})
    _upstream["E"].update({"changed": release_changed.value, "refresh_evidence": sorted(release_refresh.value)})
    _upstream["D"].update({"incident": monitor_incident.value, "observation_hours": monitor_hours.value, "per_check_false_positive_rate": alert_rate.value})
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=_upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def money(quantity):
        return f"${quantity.m_as('USD'):,.0f}"

    def money_in_prose(quantity):
        return f"USD {quantity.m_as('USD'):,.0f}"

    def requests(quantity):
        return f"{quantity.m_as('count'):,.0f}"

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style(
            {"max-width": "100%", "overflow-x": "auto"}
        )

    def metric_table(values):
        return table([{"Metric": label, "Value": value} for label, value in values.items()])

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture this part."), kind="danger")
        snapshot = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {_read_prediction(_saved_prediction_widgets[part], value=snapshot["prediction"])}<br><small>Track {snapshot["track"]}; later control changes do not rewrite this result.</small></div>')

    def part_a():
        prompt = mo.md(f"### A · Why does the same model count create different operating work? (10 min)\nThe **{profile.display}** scenario contains **{a_base.model_count} models** over **{a_base.periods} planning periods**. Predict how adding one model changes obligations, then compare topology at fixed model count. Adoption and capacity ownership below are separate economic decisions.")
        if a_prediction.value is None:
            return mo.vstack([prompt, a_prediction])
        figure = go.Figure([
            go.Bar(name="Independent duplication", x=["Independent"], y=[a_result.independent_total.m_as("USD")], marker_color=COLORS["OrangeLine"]),
            go.Bar(name="Shared platform plan", x=[f"{adoption.value:.0%} adopted"], y=[a_result.platform_total.m_as("USD")], marker_color=COLORS["BlueLine"]),
        ])
        figure.update_layout(height=260, margin=dict(l=20, r=20, t=25, b=20), yaxis_title="Portfolio cost (USD)", showlegend=False)
        rows = [{"Adopted models": f"{a_result.adopted_models:.1f}",
                 "Unadopted models": f"{a_result.unadopted_models:.1f}",
                 "Unadopted toil": money(a_result.unadopted_toil_cost),
                 "Savings": money(a_result.savings),
                 "Break-even size": a_result.break_even_model_count if a_result.break_even_model_count is not None else "No break-even"}]
        return mo.vstack([mo.vstack([
            prompt, a_prediction, coupling,
            metric_table({"Models": coupling_result.model_count, "Possible pairs": coupling_result.possible_pairs,
                    "Pairs requiring interface review": coupling_result.coupled_pairs,
                    "New pairs when adding one model": coupling_result.marginal_pairs_for_new_model,
                    "Pairs after addition": coupling_result.pairs_after_addition}),
            mo.md("The graph counts compatibility-review obligations, not observed failures or engineering hours. Switch from a chain to every pair coupled while holding model count fixed. Then compare platform economics below: adoption does not silently remove these dependencies."),
            adoption, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Prediction:** {_read_prediction(a_prediction)}. Compare its graph claim with the review-pair table above. Separately, the independent plan costs **{money_in_prose(a_result.independent_total)}**; the selected platform plan costs **{money_in_prose(a_result.platform_total)}**. Partial adoption preserves duplicated operations and adds **{money_in_prose(a_result.unadopted_toil_cost)}** of residual toil."), kind="info"),
            mo.md("**Capacity ownership is a separate decision.** Annualized capital and operating charges belong to one hypothetical capacity block. Compare the same useful work over one annual horizon; busy time that produces rejected work does not count as useful utilization."),
            useful_utilization,
            metric_table({"Useful hours": f"{ownership_result.useful_hours:,.0f}", "Owned cost": money(ownership_result.owned_cost), "Elastic rental cost": money(ownership_result.rental_cost), "Break-even useful utilization": f"{ownership_result.break_even_useful_utilization:.1%}", "Cheaper": "ownership" if ownership_result.ownership_cheaper else "rental"}),
            mo.md("The engine declares fixed annualized capital, linear useful-hour operation and elastic rental charges. Excludes staffing, financing, idle power and rental commitments. These are policy assumptions, not hardware prices."),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md("MLSysIM sums shared fixed platform cost, adopted per-model migration and operation, and independent operation plus residual toil for unadopted models. Break-even divides fixed platform cost by per-model savings under the selected adoption assumption.")}),
        ]), _concept_reflection(_reflection_contract, "A")])

    def part_b():
        prompt = mo.md(f"### B · Which consumers does one upstream change block? (9 min)\nChange **{profile.dependency_artifact}** from **{profile.dependency_baseline_version}** to **{profile.dependency_changed_version}**. Commit a prediction before revealing consumer contracts.")
        if b_prediction.value is None:
            return mo.vstack([prompt, b_prediction])
        rows = [{"Consumer": check.consumer, "Consumer version": check.consumer_version,
                 "Upstream version": check.upstream_version,
                 "Contract": "COMPATIBLE" if check.compatible else "BLOCKED"}
                for check in b_result.checks]
        return mo.vstack([mo.vstack([
            prompt, b_prediction, table(rows),
            mo.callout(mo.md(f"**Prediction:** {_read_prediction(b_prediction)}. The baseline blocks **{b_base.blocked_consumer_count}** consumers; the changed version blocks **{b_result.blocked_consumer_count}**: {', '.join(b_result.blocked_consumers) or 'none'}. A shared upstream artifact increases reach, while each consumer version still decides compatibility."), kind="danger" if b_result.blocked_consumers else "success"),
            mo.md("**A compatible version still needs a valid timestamp contract.** The trace contains a prior available record and a newer event whose feature arrives after prediction. Compare joining on event time alone with requiring availability before prediction; the selected timestamps and values below come from the same engine trace."),
            join_rule,
            metric_table({"Prediction hour": feature_result.prediction_time_hour, "Selected event hour": feature_result.selected_event_time_hour, "Selected availability hour": feature_result.selected_availability_time_hour, "Training value": feature_result.selected_value, "Serving value at prediction": feature_result.serving_value, "Training/serving match": "yes" if feature_result.matches_serving else "no"}),
            mo.callout(mo.md("The selected feature was available at prediction time." if feature_result.available_at_prediction else "**Leakage:** the selected feature arrived after prediction. An event timestamp before prediction does not make the feature available."), kind="success" if feature_result.available_at_prediction else "danger"),
            b_capture, saved("B"),
            mo.accordion({"Calculation Notes": mo.md("MLSysIM follows directed producer-to-consumer contracts and checks exact accepted upstream versions. An incompatible consumer is not assumed to publish a downstream version silently.")}),
        ]), _concept_reflection(_reflection_contract, "B")])

    def part_c():
        prompt = mo.md("### C · When is the canary informative enough? (10 min)\nChoose a traffic share and duration. The release policy requires a minimum number of observable labels in every cohort.")
        if c_prediction.value is None:
            return mo.vstack([prompt, canary_fraction, canary_hours, c_prediction])
        figure = go.Figure()
        figure.add_bar(name="Observable labels", x=[item.name for item in c_result.cohorts], y=[item.labeled_samples.m_as("count") for item in c_result.cohorts], marker_color=COLORS["BlueLine"])
        figure.add_bar(name="Labels pending", x=[item.name for item in c_result.cohorts], y=[item.pending_labels.m_as("count") for item in c_result.cohorts], marker_color=COLORS["OrangeLine"])
        figure.update_layout(barmode="stack", height=280, margin=dict(l=20, r=20, t=25, b=30), yaxis_title=f"Canary {profile.workload_unit} (count)", legend_orientation="h")
        rows = [{"Cohort": item.name, "Exposed": requests(item.exposed_requests),
                 "Observable labels": requests(item.labeled_samples),
                 "Required": f"{item.required_labeled_samples:,}",
                 "Status": "MET" if item.requirement_met else "NOT MET"}
                for item in c_result.cohorts]
        return mo.vstack([mo.vstack([
            prompt, mo.hstack([canary_fraction, canary_hours], widths="equal", wrap=True),
            c_prediction, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Prediction:** {_read_prediction(c_prediction)}. The selected canary exposes **{requests(c_result.total_exposed_requests)} {profile.workload_unit}** and has **{requests(c_result.total_labeled_samples)} observable labels** when it ends. These are evidence counts, not a claim of statistical significance."), kind="success" if c_result.evidence_requirements_met else "warn"),
            c_capture, saved("C"),
            mo.accordion({"Calculation Notes": mo.md("For each cohort, exposure is traffic rate × canary share × duration. Observable labels use only time before the run ends minus that cohort's label delay. Minimum counts are release-policy assumptions; no power test or confidence claim is made.")}),
        ]), _concept_reflection(_reflection_contract, "C")])

    def part_d():
        prompt = mo.md(f"### D · Which signals deserve a page, and which budget blocks promotion? (11 min)\nAll four interventions evaluate **{profile.incident_context}** starting from the same rate ({profile.incident_request_rate.m_as('count / minute'):,.0f} {profile.workload_unit}/min) and incident stages. Compare affected {profile.workload_unit} rather than a readiness score.")
        if d_prediction.value is None:
            return mo.vstack([prompt, d_prediction])
        labels = ["Baseline", "Detection", "Diagnosis", "Mitigation", "Recovery"]
        results = [d_base, d_results["detection"], d_results["diagnosis"], d_results["mitigation"], d_results["recovery"]]
        figure = go.Figure()
        for index, stage in enumerate(d_base.stages):
            figure.add_bar(name=stage.name, x=labels,
                           y=[result.stages[index].affected_requests.m_as("count") for result in results])
        figure.update_layout(barmode="stack", height=290, margin=dict(l=20, r=20, t=25, b=30), yaxis_title=f"Affected {profile.workload_unit} (count)", legend_orientation="h")
        rows = [{"Strategy": label, f"Affected {profile.workload_unit}": requests(result.total_affected_requests)}
                for label, result in zip(labels, results)]
        selected = d_results.get(response_choice.value, d_base)
        return mo.vstack([mo.vstack([
            prompt, d_prediction,
            mo.hstack([alert_rate, monitor_incident, monitor_hours], wrap=True),
            metric_table({"Independent checks / round": monitoring_result.independent_checks_per_round,
                    "P(any null crossing / round)": f"{monitoring_result.probability_any_false_crossing:.4f}",
                    "Expected raw null crossings / day": f"{monitoring_result.expected_raw_false_crossings_per_day:,.1f}",
                    "Constructed incident raw signals": monitoring_result.raw_incident_signals,
                    "Flat incident pages": monitoring_result.flat_pages,
                    "Grouped incident pages": monitoring_result.hierarchical_pages,
                    "Diagnostic records retained": monitoring_result.diagnostic_records_retained}),
            mo.md("**Routing:** " + ("; ".join(monitoring_result.hierarchical_routes) or "No incident pages") + ". Null crossings are calibrated independent-test expectations, not predicted routed pages. Grouping the supplied shared-cause fixture does not establish detection power; the urgent component page always survives."),
            table([{"SLO dimension": item.dimension, "Bad units": f"{item.observed_bad_fraction:.1%}",
                    "Allowed bad units": f"{item.allowed_bad_fraction:.1%}", "Burn rate": f"{item.burn_rate:.1f}×",
                    "Window budget consumed": f"{item.consumed_budget_fraction:.1%}",
                    "Gate": "EXHAUSTED" if item.exhausted else "within budget"} for item in monitoring_result.service_budgets]),
            mo.callout(mo.md("**Promotion allowed**" if monitoring_result.promotion_allowed else "**Promotion blocked:** " + ", ".join(monitoring_result.blocked_dimensions)), kind="success" if monitoring_result.promotion_allowed else "danger"),
            mo.md("The fixture starts with no prior budget consumption in a 24-hour window. Compare stale features with unavailable responses: HTTP health can stay green while freshness exhausts its budget. Reducing alert noise cannot repair an exhausted service budget. The response investments below use a separate fixed incident-stage trace; changing the monitoring fixture does not fabricate a causal improvement in response time."),
            apply_plotly_theme(figure), table(rows),
            mo.hstack([response_choice, response_rejected], widths="equal", wrap=True),
            mo.callout(mo.md(f"**Prediction:** {_read_prediction(d_prediction)}. The SLO table above tests that claim. Separately, the live response choice yields **{requests(selected.total_affected_requests)} affected {profile.workload_unit}** versus **{requests(d_base.total_affected_requests)}** at baseline. Choosing no action still requires a contrasting tested alternative for the saved record."), kind="info"),
            d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md(f"MLSysIM sums {profile.workload_unit} rate × stage duration × affected fraction across detection, diagnosis, mitigation, and recovery. Each intervention changes only its stated duration or post-mitigation fraction.")}),
        ]), _concept_reflection(_reflection_contract, "D")])

    def part_e():
        compat_evidence = f"{profile.deployment_artifact.replace(' ', '-')}-compatibility"
        prompt = mo.md(f"### E · Which evidence must be regenerated? (7 min)\nChoose whether the release candidate retains or changes **{profile.dependency_artifact}** for the **{profile.deployment_artifact}** artifact (rollback scope: **{profile.rollback_scope}**). Predict which saved evidence still matches the candidate's exact artifact versions.")
        if e_prediction.value is None:
            return mo.vstack([prompt, e_prediction])
        names = ("offline-evaluation", "canary-observation", compat_evidence)
        rows = [{"Evidence": name,
                 "Before selected reruns": "CURRENT" if name not in e_base.stale_evidence else "STALE",
                 "After selected reruns": "STALE" if name in e_result.stale_evidence else "CURRENT"}
                for name in names]
        return mo.vstack([mo.vstack([
            prompt, e_prediction, release_changed, release_refresh, table(rows),
            mo.md("Rerunning a selected record supplies a successful bounded check pinned to this candidate's exact required versions. Unselected evidence retains its original versions. Compare no reruns, offline only, and offline plus canary; refreshing package compatibility alone cannot repair stale upstream-behavior evidence."),
            mo.callout(mo.md(f"**Prediction:** {_read_prediction(e_prediction)}. Release is **{'allowed' if e_result.releasable else 'blocked'}**. Regenerate: **{', '.join(e_result.stale_evidence) or 'none'}**. {compat_evidence} remains current because its recorded model and {profile.deployment_artifact} versions did not change. Reruns are supplied successful checks, not proof of statistical sufficiency."), kind="danger" if not e_result.releasable else "success"),
            e_capture, saved("E"),
            mo.accordion({"Calculation Notes": mo.md("The contract compares candidate artifact versions with the exact artifact versions stored in each evidence record. It uses no governance points. Evidence unaffected by the changed artifact remains current.")}),
        ]), _concept_reflection(_reflection_contract, "E")])

    def synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            rows.append({"Part": part,
                         "Original prediction": _read_prediction(_saved_prediction_widgets[part], value=(capture.to_dict()["prediction"] if capture else "—")),
                         "Evidence": "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING")})
        saved_decision = _captures["D"].to_dict()["decision"] if "D" in _captures else None
        complete = (
            audit.complete
            and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
            and bool(rationale.value.strip()) and final_choice.value != final_rejected.value
            and final_choice.value == saved_decision
        )
        return mo.vstack([
            mo.md("### Synthesis · Defend one fleet operations decision (5 min)\nUse saved evidence to state the chosen response investment, quantify one rejected alternative, name a remaining limitation, and set a reevaluation trigger."),
            table(rows), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
            mo.hstack([final_risk, final_trigger], widths="equal", wrap=True), rationale,
            mo.callout(mo.md("**Ready for the local report.**" if complete else "Complete five current contrasts, match the recommendation to saved Part D, choose a different rejected alternative, and add the evidence-based rationale."), kind="success" if complete else "warn"),
        ])

    _lesson_shell([_part_tabs_hint(), mo.ui.tabs({"Part A": part_a(), "Part B": part_b(), "Part C": part_c(),
                "Part D": part_d(), "Part E": part_e(), "Synthesis": synthesis()})], volume=2)
    return (audit,)


@app.cell
def _():
    from mlsysbook_labs.notebook import notebook_shell as _submit_shell, notebook_submission as _submit_guide
    from mlsysbook_labs.learning_paths import get_learning_path as _submit_path
    _submit_shell([_submit_guide(_submit_path(2, 12))], volume=2)
    return


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    from mlsysbook_labs.notebook import notebook_shell as _workspace_shell
    _captures = get_evidence()
    _saved_decision = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip()) and final_choice.value != final_rejected.value
        and final_choice.value == _saved_decision
    )
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not _ready, _download_readiness(volume=2))
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol2/lab_12_ops_scale.py"), track=track_id, scenario=profile.scenario,
        source_trace=(
            f"mlsysim.engine.v2_12_experiments (evaluate_portfolio, evaluate_dependency, "
            f"evaluate_canary, evaluate_incident, evaluate_coupling, evaluate_monitoring, evaluate_feature_join, evaluate_ownership, evaluate_release_change); "
            f"scenario: {profile.scenario} (illustrative fleet scenario; no empirical benchmark claims)"
        ),
        learning_objectives=[
            "Compare graph coupling at fixed model count and platform adoption economics",
            "Trace versioned dependencies and delayed canary evidence",
            "Route shared incident signals, inspect separate SLO budgets, and defend a response investment",
        ],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"],
                                 "result": snapshots[part].get("chosen_result") or snapshots[part]["result"],
                                 "alternatives": snapshots[part]["alternatives"]}
                          for part in "ABCDE"},
        binding_constraints={"dependency": snapshots["B"]["result"]["output"]["dependency"]["blocked_consumers"],
                             "release": snapshots["E"]["result"]["output"]["stale_evidence"]},
        decisions={"recommendation": final_choice.value,
                   "rejected_alternative": final_rejected.value,
                   "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value,
                        "rejected_alternative": final_rejected.value,
                        "rationale": rationale.value},
        big_takeaways=[
            "Shared fixed cost pays back only when adoption removes enough duplicated operation and toil.",
            "A shared dependency expands change reach and makes consumer-version contracts necessary.",
            "Canary exposure, label availability, incident harm, and release evidence remain separate quantities.",
        ],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"schema_version": 1, "lab_id": "v2_12", "track_id": track_id,
                         "model_id": "v2_12_experiments", "evidence": snapshots,
                         "recommendation": final_choice.value,
                         "rejected_alternative": final_rejected.value,
                         "reevaluation_trigger": final_trigger.value,
                         "residual_risk": final_risk.value, "rationale": rationale.value},
    )
    _workspace_shell([mo.vstack([mo.md("### Download your discussion"), report_export_panel(report)])], volume=2)
    return (report,)


@app.cell
def _(
    audit, final_choice, final_rejected, final_risk, final_trigger, get_evidence, mo, rationale, track_id
):
    from mlsysbook_labs.fleet_evidence import discussion_fingerprint as _discussion_fingerprint
    _captures = get_evidence()
    _saved_decision = _captures['D'].to_dict()['decision'] if 'D' in _captures else None
    _ready = audit.complete and all((widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))) and bool(rationale.value.strip()) and (final_choice.value != final_rejected.value) and (final_choice.value == _saved_decision)
    fleet_discussion_ready = _ready
    fleet_discussion_design = {'schema_version': 1, 'lab_id': 'v2_12', 'track_id': track_id, 'model_id': 'v2_12_experiments', 'evidence': {part: capture.to_dict() for part, capture in _captures.items()}, 'recommendation': final_choice.value, 'rejected_alternative': final_rejected.value, 'reevaluation_trigger': final_trigger.value, 'residual_risk': final_risk.value, 'rationale': rationale.value} if _ready else None
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
        ledger, chapter=12, track=track_id, design=fleet_discussion_design,
        fingerprint=fleet_discussion_fingerprint, ready=fleet_discussion_ready,
        requested=bool(save_discussion.value),
    )
    _discussion_shell([mo.vstack([save_discussion, _notebook_status(_outcome.status, _outcome.detail)])], volume=2)
    return


if __name__ == "__main__":
    app.run()
