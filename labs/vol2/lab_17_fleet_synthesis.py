import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 17: Fleet Synthesis · MLSysBook")


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
    from mlsysim.engine.v2_17_experiments import (
        REQUIRED_MODELS, adverse_condition, apply_patch, audit_ledger_evidence,
        compare_equal_budget_repairs, evaluation_record, evaluate_configuration,
        instructor_repair_plans, instructor_scenario, instructor_stress,
        obligation_stress, record_deployment_decision,
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
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, REQUIRED_MODELS, adverse_condition,
        apply_patch, apply_plotly_theme, audit_evidence, audit_ledger_evidence,
        build_lab_report, capture_evidence, compare_equal_budget_repairs,
        evaluation_record, evaluate_configuration, get_lab_metadata, go,
        instructor_repair_plans, instructor_scenario, instructor_stress, ledger,
        mo, obligation_stress, record_deployment_decision, report_export_panel,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Fleet track", on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(instructor_scenario, track):
    track_id = track.value
    scenario = instructor_scenario(track_id)
    return scenario, track_id


# ZONE B · WIDGETS
@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"None": "none", "One or two": "some", "Three or four": "most"},
        label="How many required student records will replay?",
    ).form(submit_button_label="Lock Part A prediction")
    a_source = mo.ui.radio(
        {"Student evidence only": "ledger_only", "Fill gaps with labeled scenarios": "fallback"},
        label="Evidence source policy",
    )
    b_severity = mo.ui.dropdown(
        {"Moderate scale-up": "moderate", "Severe scale-up": "severe"},
        value="Severe scale-up", label="Scale stress",
    )
    b_prediction = mo.ui.radio(
        {"Compute": "compute", "Communication": "communication", "Coordination": "coordination"},
        label="Which C³ term dominates after scale-up?",
    ).form(submit_button_label="Lock Part B prediction")
    c_serving = mo.ui.dropdown(
        {"Busy": "busy", "Surge": "surge", "Overload": "overload"},
        value="Surge", label="Serving condition",
    )
    c_energy = mo.ui.dropdown(
        {"Steady work": "steady", "Growth": "growth", "Double work": "double"},
        value="Growth", label="Useful-work demand",
    )
    c_prediction = mo.ui.radio(
        {"Serving p99": "serving", "Absolute energy": "energy", "Both": "both", "Neither": "neither"},
        label="Which obligation fails first?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Fabric": "fabric", "Resilience": "resilience", "Efficiency": "efficiency", "None": "none"},
        label="Which equal-budget repair is strongest?",
    ).form(submit_button_label="Lock Part D prediction")
    d_choice = mo.ui.radio(
        {"Fabric": "fabric", "Resilience": "resilience", "Efficiency": "efficiency", "Hold current configuration": "none"},
        label="Repair to carry forward",
    )
    d_rejected = mo.ui.radio(
        {"Fabric": "fabric", "Resilience": "resilience", "Efficiency": "efficiency"},
        label="Rejected alternative",
    )
    e_condition = mo.ui.dropdown(
        {"Fabric contention": "fabric", "Longer recovery": "recovery", "Replica loss": "replica_loss", "Demand growth": "demand"},
        value="Replica loss", label="Held-out condition",
    )
    e_prediction = mo.ui.radio(
        {"Feasible": "feasible", "Infeasible": "infeasible", "Unevaluable": "unevaluable"},
        label="What happens under the held-out condition?",
    ).form(submit_button_label="Lock Part E prediction")
    return (
        a_prediction, a_source, b_prediction, b_severity, c_energy, c_prediction,
        c_serving, d_choice, d_prediction, d_rejected, e_condition, e_prediction,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_decision = mo.ui.radio(
        {"Release": "release", "Restrict rollout": "restrict", "Defer": "defer"},
        label="Deployment judgment",
    )
    final_condition = mo.ui.text_area(
        label="Measurable reevaluation condition",
        placeholder="Name the observed limit and the condition that reopens this decision.",
    )
    final_uncertainty = mo.ui.text_area(
        label="Remaining limitation",
        placeholder="Name evidence or an operating condition this lab did not establish.",
    )
    final_rationale = mo.ui.text_area(
        label="Engineering rationale",
        placeholder="Connect the saved failure, chosen repair, rejected alternative, and held-out replay.",
    )
    return final_condition, final_decision, final_rationale, final_uncertainty


@app.cell
async def _(
    REQUIRED_MODELS, a_source, adverse_condition, apply_patch,
    audit_ledger_evidence, b_severity, c_energy, c_serving,
    compare_equal_budget_repairs, d_choice, e_condition, evaluate_configuration,
    instructor_repair_plans, instructor_stress, ledger, obligation_stress,
    scenario, track_id,
):
    fallbacks = {
        chapter: {"chapter": chapter, "label": "illustrative instructor scenario"}
        for chapter in REQUIRED_MODELS
    }
    a_student = audit_ledger_evidence(ledger._state.history, track_id=track_id)
    a_with_fallback = audit_ledger_evidence(
        ledger._state.history, track_id=track_id, instructor_fallbacks=fallbacks,
    )
    a_carried = a_with_fallback if a_source.value == "fallback" else a_student

    baseline = evaluate_configuration(scenario)
    b_config = apply_patch(scenario, instructor_stress(b_severity.value))
    b_result = evaluate_configuration(b_config)
    c_config = apply_patch(scenario, obligation_stress(c_serving.value, c_energy.value))
    c_result = evaluate_configuration(c_config)

    repairs = instructor_repair_plans(track_id)
    repairs_by_id = {plan.repair_id: plan for plan in repairs}
    d_compare = compare_equal_budget_repairs(
        scenario, stress=instructor_stress("severe"), repairs=repairs,
    )
    d_results = {plan.repair_id: result for plan, result in d_compare.repairs}
    _live_choice = d_choice.value if d_choice.value in repairs_by_id else "fabric"
    _live_plan = repairs_by_id[_live_choice]
    _live_config = apply_patch(d_compare.failing_evaluation.configuration, _live_plan.patch)
    _live_result = d_results[_live_choice]
    carried_config = d_compare.failing_evaluation.configuration if d_choice.value == "none" else _live_config
    carried_result = d_compare.failing_evaluation if d_choice.value == "none" else _live_result
    carried_label = "No repair" if d_choice.value == "none" else _live_plan.label

    e_config = apply_patch(carried_config, adverse_condition(e_condition.value))
    e_result = evaluate_configuration(e_config)
    return (
        a_carried, a_student, a_with_fallback, baseline, b_config, b_result,
        c_config, c_result, carried_config, carried_label, carried_result,
        d_compare, d_results, e_config, e_result, fallbacks, repairs, repairs_by_id,
    )


@app.cell
def _(
    a_prediction, a_source, a_student, a_with_fallback, apply_patch, b_config, b_prediction,
    b_result, b_severity, baseline, c_config, c_energy, c_prediction, c_result,
    c_serving, capture_evidence, carried_config, carried_result, d_choice,
    d_compare, d_prediction, d_rejected, d_results, e_condition, e_config,
    e_prediction, e_result, evaluation_record, fallbacks, ledger, mo, repairs,
    repairs_by_id, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    def audit_record(history, instructor_fallbacks, audit):
        return {
            "inputs": {"history": history, "track_id": track_id, "instructor_fallbacks": instructor_fallbacks},
            "outputs": {
                "ready": audit.ready,
                "items": [
                    {"chapter": item.chapter, "status": item.status, "source": item.source_label, "issue": item.issue}
                    for item in audit.items
                ],
            },
        }

    a_upstream = {"track": track_id, "source_policy": a_source.value}
    b_upstream = {"track": track_id, "severity": b_severity.value}
    c_upstream = {"track": track_id, "serving": c_serving.value, "energy": c_energy.value}
    d_upstream = {"track": track_id, "stress": "severe", "choice": d_choice.value, "rejected": d_rejected.value}
    e_upstream = {**d_upstream, "condition": e_condition.value}

    _a_student_record = audit_record(ledger._state.history, {}, a_student)
    _a_fallback_record = audit_record(ledger._state.history, fallbacks, a_with_fallback)
    a_capture = mo.ui.button(
        label="Capture evidence-source comparison", kind="success",
        disabled=a_prediction.value is None or a_source.value is None,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"source_policy": a_source.value}, baseline=_a_student_record,
            result=_a_fallback_record,
            chosen_result=_a_student_record if a_source.value == "ledger_only" else None,
            result_role="rejected alternative" if a_source.value == "ledger_only" else "tested intervention",
            alternatives=(_a_student_record, _a_fallback_record), decision=a_source.value,
            upstream_inputs=a_upstream, model_key="v2_17_experiments.audit_ledger_evidence",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture C³ scale contrast", kind="success", disabled=b_prediction.value is None,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"severity": b_severity.value},
            baseline=evaluation_record(baseline.configuration, baseline),
            result=evaluation_record(b_config, b_result), decision=b_result.c3_result.dominant_term,
            upstream_inputs=b_upstream, model_key="v2_17_experiments.evaluate_configuration",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture obligation stress", kind="success", disabled=c_prediction.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"serving": c_serving.value, "energy": c_energy.value},
            baseline=evaluation_record(baseline.configuration, baseline),
            result=evaluation_record(c_config, c_result),
            decision=list(c_result.failed_constraints), upstream_inputs=c_upstream,
            model_key="v2_17_experiments.evaluate_configuration",
        )),
    )
    _d_invalid = (
        d_prediction.value is None or d_choice.value is None or d_rejected.value is None
        or (d_choice.value != "none" and d_choice.value == d_rejected.value)
    )
    _tested_id = d_rejected.value if d_choice.value == "none" and d_rejected.value else (
        d_choice.value if d_choice.value in d_results else "fabric"
    )
    _tested_plan = repairs_by_id[_tested_id]
    _tested_config = apply_patch(d_compare.failing_evaluation.configuration, _tested_plan.patch)
    _tested_result = d_results[_tested_id]
    _failing_record = evaluation_record(
        d_compare.failing_evaluation.configuration, d_compare.failing_evaluation,
    )
    d_capture = mo.ui.button(
        label="Capture equal-budget repair decision", kind="success", disabled=_d_invalid,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={"choice": d_choice.value, "rejected": d_rejected.value},
            baseline=_failing_record, result=evaluation_record(_tested_config, _tested_result),
            chosen_result=_failing_record if d_choice.value == "none" else None,
            result_role="rejected alternative" if d_choice.value == "none" else "tested intervention",
            alternatives=tuple(evaluation_record(result.configuration, result) for result in d_results.values()),
            decision=d_choice.value, upstream_inputs=d_upstream,
            model_key="v2_17_experiments.evaluate_configuration",
        )),
    )
    e_capture = mo.ui.button(
        label="Capture held-out replay", kind="success",
        disabled=e_prediction.value is None or d_choice.value is None,
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"carried_repair": d_choice.value, "condition": e_condition.value},
            baseline=evaluation_record(carried_config, carried_result),
            result=evaluation_record(e_config, e_result),
            decision="feasible" if e_result.physical_feasible is True else (
                "infeasible" if e_result.physical_feasible is False else "unevaluable"
            ),
            upstream_inputs=e_upstream, model_key="v2_17_experiments.evaluate_configuration",
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
    .v217-head{background:linear-gradient(135deg,#101827,#1d4f78);color:#fff;padding:22px 24px;border-radius:12px;margin:8px 0 14px}.v217-head h1{font-size:clamp(1.5rem,4vw,2.3rem);margin:4px 0 8px}.v217-meta{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:14px}.v217-meta div{background:#ffffff17;padding:10px;border-radius:8px}.v217-saved{border-left:4px solid #2E8B57;background:#f2faf5;padding:9px 12px}.v217-table{max-width:100%;overflow-x:auto}@media(max-width:760px){.v217-meta{grid-template-columns:1fr}.v217-head{padding:18px;margin-top:30px}}
    </style>""")
    mo.vstack([
        mo.Html(f"<style>{LAB_CSS}</style>"), mo.Html(f"<style>{ACADEMIC_LAB_CSS}</style>"),
        css, track,
        mo.Html(f'''<section class="v217-head"><small>VOLUME II · LAB 17</small><h1>Fleet Synthesis</h1><div>Which repair keeps a complete fleet inside its physical and operating constraints?</div><div class="v217-meta"><div><b>Duration</b><br>50 minutes</div><div><b>Fleet</b><br>{scenario.scenario_label}</div><div><b>Output</b><br>Deployment evidence packet</div></div></section>'''),
        mo.callout(mo.md("Built-in values are **illustrative instructor scenarios**, not production measurements. Compatible student evidence remains separately labeled."), kind="info"),
    ])
    return


# ZONE C · ONE TOP-LEVEL TAB SET
@app.cell
async def _(
    COLORS, a_capture, a_carried, a_prediction, a_source, a_student, a_upstream,
    apply_plotly_theme, audit_evidence, b_capture, b_result, b_prediction,
    b_severity, b_upstream, baseline, c_capture, c_energy, c_prediction,
    c_result, c_serving, c_upstream, carried_label, carried_result, d_capture,
    d_choice, d_compare, d_prediction, d_rejected, d_results, d_upstream,
    e_capture, e_condition, e_prediction, e_result, e_upstream, final_condition,
    final_decision, final_rationale, final_uncertainty, get_evidence, go, mo,
    repairs,
):
    captures = get_evidence()
    upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    evidence_audit = audit_evidence(
        captures, track=baseline.configuration.track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def status(result):
        if result.physical_feasible is True:
            return "FEASIBLE"
        if result.physical_feasible is False:
            return "INFEASIBLE · " + ", ".join(result.failed_constraints)
        return "UNEVALUABLE · " + ", ".join(result.unevaluated_constraints)

    def amount(value, unit, digits=2):
        return f"{value.to(unit).magnitude:.{digits}f} {unit}"

    def saved(part):
        capture = captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in evidence_audit.stale or (part, part) in evidence_audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture after changing dependencies."), kind="danger")
        snapshot = capture.to_dict()
        return mo.Html(f'<div class="v217-saved"><b>Saved snapshot</b> · prediction: {snapshot["prediction"]}<br><small>Later controls do not rewrite this result.</small></div>')

    def check_rows(result):
        rows = []
        for check in result.checks:
            observed = f"{check.observed:~P}" if hasattr(check.observed, "to") else f"{check.observed:.3f}"
            limit = f"{check.limit:~P}" if hasattr(check.limit, "to") else f"{check.limit:.3f}"
            check_status = "PASS" if check.passed is True else "FAIL" if check.passed is False else "UNEVALUABLE"
            rows.append({"Constraint": check.constraint, "Observed": observed, "Limit": limit, "Status": check_status})
        return rows

    def build_part_a():
        if a_prediction.value is None:
            return mo.vstack([mo.md("### A · Which inherited evidence can we trust? (8 min)\nPredict how many required records can be replayed before inspecting the ledger."), a_prediction])
        rows = [{
            "Chapter": carried.chapter,
            "Student record": next(item.status for item in a_student.items if item.chapter == carried.chapter),
            "Carried source": carried.source_label,
            "Status": carried.status,
        } for carried in a_carried.items]
        return mo.vstack([
            mo.md("### A · Which inherited evidence can we trust? (8 min)"), a_prediction, a_source,
            table(rows),
            mo.callout(mo.md(f"**Prediction:** {a_prediction.value}. **Carried packet:** {'ready' if a_carried.ready else 'incomplete'}. Missing evidence does not become failed physics."), kind="success" if a_carried.ready else "danger"),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md("Compatibility requires the stable chapter, track, model, evaluator, unit decoding, and a successful replay. In device tracks (TinyML/Mobile), inherited training and recovery artifacts evaluate the supporting backend cluster, while serving evaluates deployment endpoints. Other legitimate chapter experiments are ignored rather than penalized.")}),
        ])

    def build_part_b():
        if b_prediction.value is None:
            return mo.vstack([mo.md("### B · When does scale stop improving useful work? (10 min)\nChoose a scale stress and predict the dominant C³ term."), b_severity, b_prediction])
        figure = go.Figure()
        for case, result in (("Baseline", baseline.c3_result), ("Scale stress", b_result.c3_result)):
            for label, field, color in (("Compute", "compute_time", COLORS["BlueLine"]), ("Communication", "communication_time", COLORS["OrangeLine"]), ("Coordination", "coordination_time", COLORS["GreenLine"])):
                figure.add_bar(name=label, x=[case], y=[getattr(result, field).to("second").magnitude], marker_color=color, legendgroup=label, showlegend=case == "Baseline")
        figure.update_layout(barmode="stack", height=290, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Fleet step time (s)", legend_orientation="h")
        return mo.vstack([
            mo.md("### B · When does scale stop improving useful work? (10 min)"), b_severity, b_prediction,
            apply_plotly_theme(figure),
            table([{"Run": "Baseline", "Step time": amount(baseline.c3_result.step_time, "second"), "Dominant": baseline.c3_result.dominant_term, "Outcome": status(baseline)}, {"Run": "Scale stress", "Step time": amount(b_result.c3_result.step_time, "second"), "Dominant": b_result.c3_result.dominant_term, "Outcome": status(b_result)}]),
            mo.callout(mo.md(f"**Prediction:** {b_prediction.value}. **Observed dominant term:** {b_result.c3_result.dominant_term}; step time changed from {amount(baseline.c3_result.step_time, 'second')} to {amount(b_result.c3_result.step_time, 'second')}."), kind="info"),
            b_capture, saved("B"),
            mo.accordion({"Calculation Notes": mo.md("The fixed-work model adds visible compute, communication, and coordination time, then subtracts only stated overlap. For TinyML and Mobile tracks, C³ scaling models the supporting backend training cluster; edge microcontrollers/devices do not run collective all-reduce training. Useful work remains fixed.")}),
        ])

    def build_part_c():
        if c_prediction.value is None:
            return mo.vstack([mo.md("### C · Can a fast fleet still violate its obligations? (9 min)\nStress serving execution and absolute useful-work demand while leaving the C³ configuration unchanged."), mo.hstack([c_serving, c_energy], widths="equal", wrap=True), c_prediction])
        return mo.vstack([
            mo.md("### C · Can a fast fleet still violate its obligations? (9 min)"), mo.hstack([c_serving, c_energy], widths="equal", wrap=True), c_prediction,
            table(check_rows(c_result)),
            mo.callout(mo.md(f"**Prediction:** {c_prediction.value}. **Result:** {status(c_result)}. Each constraint keeps its physical unit."), kind="danger" if c_result.physical_feasible is False else "success"),
            c_capture, saved("C"),
            mo.accordion({"Calculation Notes": mo.md("Serving p99 comes from complete simulated request durations in the deployment serving pool. Rejected requests have a separate completion check. Absolute energy and facility capacity govern the backend/aggregation infrastructure (with device constraints bounded separately by local power envelopes).")}),
        ])

    def build_part_d():
        if d_prediction.value is None:
            return mo.vstack([mo.md("### D · Which equal-budget repair survives the whole fleet? (11 min)\nEvery repair starts from the same severe stress and spends the same scenario budget."), d_prediction])
        rows = []
        for plan in repairs:
            result = d_results[plan.repair_id]
            rows.append({"Repair": plan.label, "Budget": f"{plan.cost:~P}", "C³ step": amount(result.c3_result.step_time, "second"), "Energy": amount(result.energy_result.facility_energy, "kilowatt_hour", 3), "Outcome": status(result), "Consequence": plan.consequence})
        _live = d_compare.failing_evaluation if d_choice.value == "none" else d_results.get(d_choice.value, d_compare.failing_evaluation)
        return mo.vstack([
            mo.md("### D · Which equal-budget repair survives the whole fleet? (11 min)"), d_prediction, table(rows),
            mo.hstack([d_choice, d_rejected], widths="equal", wrap=True),
            mo.callout(mo.md(f"**Prediction:** {d_prediction.value}. **Carried choice:** {d_choice.value or 'unset'}; outcome: {status(_live)}. A defensible no-feasible result is valid evidence."), kind="success" if _live.physical_feasible is True else "danger"),
            d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md("Repairs are rerun independently from one frozen failure. Fabric capacity adds energy; a replica adds energy and power; efficiency leaves C³ inputs unchanged.")}),
        ])

    def build_part_e():
        if e_prediction.value is None:
            return mo.vstack([mo.md(f"### E · Does the repair survive a held-out condition? (7 min)\nCarry **{carried_label}** into one additional condition."), e_condition, e_prediction])
        return mo.vstack([
            mo.md("### E · Does the repair survive a held-out condition? (7 min)"), e_condition, e_prediction,
            table([{"Run": "Carried configuration", "C³ step": amount(carried_result.c3_result.step_time, "second"), "Serving p99": amount(carried_result.serving_result.p99_duration, "millisecond"), "Energy": amount(carried_result.energy_result.facility_energy, "kilowatt_hour", 3), "Outcome": status(carried_result)}, {"Run": "Held-out replay", "C³ step": amount(e_result.c3_result.step_time, "second"), "Serving p99": amount(e_result.serving_result.p99_duration, "millisecond"), "Energy": amount(e_result.energy_result.facility_energy, "kilowatt_hour", 3), "Outcome": status(e_result)}]),
            mo.callout(mo.md(f"**Prediction:** {e_prediction.value}. **Held-out result:** {status(e_result)}. Support is limited to the conditions actually replayed."), kind="danger" if e_result.physical_feasible is False else "success"),
            e_capture, saved("E"),
            mo.accordion({"Calculation Notes": mo.md("The held-out replay changes one disclosed causal input set. Evidence labels and learner choices do not alter simulator output.")}),
        ])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = captures.get(part)
            current = capture is not None and part not in evidence_audit.stale and (part, part) not in evidence_audit.identical_pairs
            rows.append({"Part": part, "Prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": "CURRENT" if current else "STALE" if capture else "MISSING"})
        _d_snap = captures["D"].to_dict() if "D" in captures else {}
        _saved_choice = _d_snap.get("decision")
        _saved_rejected = _d_snap.get("inputs", {}).get("rejected")
        _distinct = _saved_rejected is not None and (_saved_choice == "none" or _saved_choice != _saved_rejected)
        _complete = evidence_audit.complete and final_decision.value is not None and _distinct and bool(final_condition.value.strip()) and bool(final_uncertainty.value.strip()) and bool(final_rationale.value.strip())
        return mo.vstack([
            mo.md("### Synthesis · Make the deployment judgment (5 min)\nChoose release, restricted rollout, or defer from saved evidence. State the rejected repair, remaining limitation, and measurable reevaluation condition."),
            table(rows), final_decision, final_condition, final_uncertainty, final_rationale,
            mo.callout(mo.md("**Ready for the local report.**" if _complete else "Capture five current contrasts, retain a distinct rejected repair, and complete the judgment fields."), kind="success" if _complete else "warn"),
        ])

    tabs = mo.ui.tabs({"Part A": build_part_a(), "Part B": build_part_b(), "Part C": build_part_c(), "Part D": build_part_d(), "Part E": build_part_e(), "Synthesis": build_synthesis()})
    tabs
    return (evidence_audit,)


@app.cell
def _(
    build_lab_report, evidence_audit, final_condition, final_decision,
    final_rationale, final_uncertainty, get_evidence, get_lab_metadata,
    mo, report_export_panel, scenario, track_id,
):
    _captures = get_evidence()
    _d_snap = _captures["D"].to_dict() if "D" in _captures else {}
    _choice = _d_snap.get("decision")
    _rejected = _d_snap.get("inputs", {}).get("rejected")
    _distinct = _rejected is not None and (_choice == "none" or _choice != _rejected)
    _ready = evidence_audit.complete and final_decision.value is not None and _distinct and bool(final_condition.value.strip()) and bool(final_uncertainty.value.strip()) and bool(final_rationale.value.strip())
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol2/lab_17_fleet_synthesis.py"), track=track_id,
        scenario=scenario.scenario_label,
        learning_objectives=["Trace a binding C³ constraint across a fleet", "Evaluate recovery, serving, energy, power, and cooling in native units", "Compare equal-budget repairs and defend a bounded deployment judgment"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "chosen_result": snapshots[part].get("chosen_result"), "result_role": snapshots[part].get("result_role"), "alternatives": snapshots[part]["alternatives"]} for part in "ABCDE"},
        binding_constraints={part: snapshots[part]["result"].get("outputs", {}).get("failed_constraints", []) for part in "BCDE"},
        decisions={"deployment_judgment": final_decision.value, "selected_repair": _choice, "rejected_alternative": _rejected, "reevaluation_trigger": final_condition.value},
        final_decision={"decision": final_decision.value, "selected_repair": _choice, "rejected_alternative": _rejected, "rationale": final_rationale.value},
        big_takeaways=["The fleet is the unit of engineering.", "A local repair can move the binding constraint.", "Missing evidence and failed physics require different responses."],
        reflections={"rationale": final_rationale.value, "reevaluation_trigger": final_condition.value},
        residual_risk=final_uncertainty.value,
        result_snapshot={"track": track_id, "captures": snapshots, "deployment_judgment": final_decision.value, "selected_repair": _choice, "rejected_alternative": _rejected, "reevaluation_trigger": final_condition.value, "residual_uncertainty": final_uncertainty.value},
        source_trace={"scenario": "Illustrative instructor scenario unless compatible student evidence is labeled.", "calculations": "Analytical C³, recovery, request-trace, energy, power, and cooling models."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


# ZONE D · COMPLETION-GATED LEDGER HUD
@app.cell
async def _(
    evidence_audit, final_condition, final_decision, final_rationale,
    final_uncertainty, get_evidence, ledger, mo,
    record_deployment_decision, track_id,
):
    _captures = get_evidence()
    _d_snap = _captures["D"].to_dict() if "D" in _captures else {}
    _choice = _d_snap.get("decision")
    _rejected = _d_snap.get("inputs", {}).get("rejected")
    _distinct = _rejected is not None and (_choice == "none" or _choice != _rejected)
    _ready = evidence_audit.complete and final_decision.value is not None and _distinct and bool(final_condition.value.strip()) and bool(final_uncertainty.value.strip()) and bool(final_rationale.value.strip())
    _saved = False
    _save_error = ""
    if _ready:
        try:
            _decision = record_deployment_decision(
                decision=final_decision.value, selected_repair=_choice,
                rejected_alternative=_rejected, condition=final_condition.value,
                residual_uncertainty=final_uncertainty.value,
            )
            ledger.save(chapter=17, design={
                "schema_version": 1, "lab_id": "v2_17", "track_id": track_id,
                "model_id": "v2_17_experiments",
                "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "recommendation": _decision.decision,
                "rejected_alternative": _decision.rejected_alternative,
                "reevaluation_trigger": _decision.condition,
                "residual_risk": _decision.residual_uncertainty,
                "rationale": final_rationale.value,
            })
            await ledger.flush()
            _saved = True
        except Exception as _exc:
            _save_error = str(_exc)
    _state = "SAVED" if _saved else "SAVE FAILED" if _save_error else "EVIDENCE IN PROGRESS"
    _detail = f" · {_save_error}" if _save_error else ""
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">17 · Fleet Synthesis</span><span style="flex:1"></span><span class="hud-label">STATUS</span><span class="hud-active">{_state}{_detail}</span></div>')
    return


if __name__ == "__main__":
    app.run()
