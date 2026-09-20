import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 16: Evidence for Responsible Release · MLSysBook")

# ZONE A · Opening and simulator bootstrap

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
    from mlsysim.core.units import Q_
    from mlsysim.engine.v2_16_experiments import (
        PRIVACY_PROFILES, assess_fairness,
        audit_evidence as simulate_audit_evidence,
        check_privacy_profile, evaluate_capacity, mechanical_gate,
        record_release_decision, track_profile,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata,
        report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import (
        audit_evidence, capture_evidence,
    )

    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, PRIVACY_PROFILES, Q_,
        apply_plotly_theme, assess_fairness, audit_evidence, build_lab_report,
        capture_evidence, check_privacy_profile, evaluate_capacity,
        get_lab_metadata, go, ledger, mechanical_gate, mo,
        record_release_decision, report_export_panel,
        simulate_audit_evidence, track_profile,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="Mobile", label="Deployment track",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(track, track_profile):
    track_id = track.value
    profile = track_profile(track_id)
    return profile, track_id

# ZONE B · Prediction and experiment controls

@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_threshold = mo.ui.slider(0.30, 0.80, value=0.65, step=0.05, label="Decision threshold")
    a_criterion = mo.ui.dropdown(
        {"Demographic parity": "demographic_parity", "Equal opportunity": "equal_opportunity",
         "Predictive parity": "predictive_parity", "Equalized odds": "equalized_odds"},
        value="Equal opportunity", label="Fairness criterion",
    )
    a_allowed_gap = mo.ui.slider(0.05, 0.40, value=0.15, step=0.05, label="Allowed metric gap")
    b_explanation_share = mo.ui.slider(0.01, 0.50, value=0.10, step=0.01, label="Explanation share")
    b_review_share = mo.ui.slider(0.001, 0.10, value=0.02, step=0.001, label="Human-review share")
    c_sampling = mo.ui.slider(0.10, 1.00, value=0.50, step=0.10, label="Eligible cases sampled")
    c_label_delay = mo.ui.slider(1, 80, value=24, step=1, label="Label delay (hours)")
    d_profile = mo.ui.dropdown(
        {"Cohort aggregate counts": "aggregate_counts", "Pseudonymous scored events": "pseudonymous_events",
         "Full case-review trace": "full_review_trace"},
        value="Pseudonymous scored events", label="Evidence data profile",
    )
    d_case_review = mo.ui.dropdown(
        {"Subgroup audit only": False, "Case-level review required": True},
        value="Case-level review required", label="Evidence obligation",
    )
    return (
        a_allowed_gap, a_criterion, a_threshold, b_explanation_share,
        b_review_share, c_label_delay, c_sampling, d_case_review, d_profile,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Chosen criterion passes": "pass", "Chosen criterion fails": "fail",
         "Chosen criterion is undefined": "undefined"},
        label="At the selected threshold, what happens to the chosen criterion?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Both queues stay within capacity": "both_pass", "Explanation queue overloads": "explanation",
         "Human-review queue overloads": "review", "Both queues overload": "both_fail"},
        label="What will the selected coverage require?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Enough evidence for the criterion": "estimable", "Sample misses a subgroup": "missing_group",
         "Class counts make the rate undefined": "undefined"},
        label="What will the monitoring plan reveal?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Aggregate counts": "aggregate_counts", "Pseudonymous events": "pseudonymous_events",
         "Full review trace": "full_review_trace", "No profile": "none"},
        label="Which profile satisfies the evidence obligation?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"All gates pass": "pass", "Fairness blocks": "fairness", "Capacity blocks": "capacity",
         "Audit evidence blocks": "audit", "Privacy obligations block": "privacy"},
        label="What controls the current release evidence?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, profile, track_id):
    _track_key = track_id
    e_choice = mo.ui.radio(
        {"Release": "release", "Restrict deployment": "restrict", "Defer pending evidence": "defer"},
        label="Recorded release choice",
    )
    e_rejected = mo.ui.radio(
        {"Release": "release", "Restrict deployment": "restrict", "Defer pending evidence": "defer"},
        label="Rejected alternative",
    )
    e_owner = mo.ui.dropdown(
        {"No named owner": "", "Fleet responsibility lead": "fleet responsibility lead",
         "Cross-functional owner": "cross-functional safety and product owner"},
        value="Fleet responsibility lead", label="Residual-risk owner",
    )
    _remedies = profile["remedies"]
    e_remedy = mo.ui.dropdown(
        _remedies,
        value=next(iter(_remedies.keys())), label="Remedy",
    )
    e_trigger = mo.ui.dropdown(
        {"Metric exceeds gap": "chosen metric exceeds allowed gap",
         "Review backlog appears": "review backlog becomes nonzero",
         "Audit loses class/group": "audit loses one class or subgroup", "No trigger": ""},
        value="Metric exceeds gap", label="Reevaluation trigger",
    )
    residual_risk = mo.ui.radio(
        {"Rare cohorts remain sparse": "rare cohorts remain sparse",
         "Labels arrive after harm": "labels arrive after harm can occur",
         "Review demand outgrows staffing": "review demand can exceed staffing"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Decision rationale",
        placeholder="Use saved quantities to justify the choice and reject one alternative.",
    )
    return e_choice, e_owner, e_rejected, e_remedy, e_trigger, rationale, residual_risk


@app.cell
def _(
    PRIVACY_PROFILES, Q_, a_allowed_gap, a_criterion, a_threshold,
    assess_fairness, b_explanation_share, b_review_share, c_label_delay,
    c_sampling, check_privacy_profile, d_case_review, d_profile,
    evaluate_capacity, mechanical_gate, profile, simulate_audit_evidence,
    track_id,
):
    a_base_inputs = dict(track_id=track_id, threshold=0.50, criterion=a_criterion.value, allowed_gap=a_allowed_gap.value)
    a_result_inputs = {**a_base_inputs, "threshold": a_threshold.value}
    a_base, a_result = assess_fairness(**a_base_inputs), assess_fairness(**a_result_inputs)

    _capacity_common = dict(
        track_id=track_id, horizon=Q_(8, "hour"),
        explanation_workers=profile["explanation_workers"], reviewers=profile["reviewers"],
        explanation_service_time=profile["explanation_service_time"],
        review_service_time=profile["review_service_time"], target_utilization=0.85,
    )
    b_base_inputs = {**_capacity_common, "explanation_share": 0.01, "review_share": 0.001}
    b_result_inputs = {**_capacity_common, "explanation_share": b_explanation_share.value,
                       "review_share": b_review_share.value}
    b_base, b_result = evaluate_capacity(**b_base_inputs), evaluate_capacity(**b_result_inputs)

    _audit_common = dict(track_id=track_id, threshold=a_threshold.value,
                         criterion=a_criterion.value, allowed_gap=a_allowed_gap.value)
    c_base_inputs = {**_audit_common, "sampling_fraction": 1.0, "label_delay": Q_(1, "hour")}
    c_result_inputs = {**_audit_common, "sampling_fraction": c_sampling.value,
                       "label_delay": Q_(c_label_delay.value, "hour")}
    c_base = simulate_audit_evidence(**c_base_inputs)
    c_result = simulate_audit_evidence(**c_result_inputs)

    _privacy_common = dict(track_id=track_id, require_case_review=d_case_review.value,
                           max_retention=Q_(30, "day"), max_deletion_window=Q_(7, "day"))
    d_base_inputs = {"profile_key": "aggregate_counts", **_privacy_common}
    d_result_inputs = {"profile_key": d_profile.value, **_privacy_common}
    d_base, d_result = check_privacy_profile(**d_base_inputs), check_privacy_profile(**d_result_inputs)
    d_alternatives = {key: check_privacy_profile(profile_key=key, **_privacy_common)
                      for key in PRIVACY_PROFILES}
    e_gate = mechanical_gate(a_result, b_result, c_result, d_result)
    return (
        a_base, a_base_inputs, a_result, a_result_inputs, b_base,
        b_base_inputs, b_result, b_result_inputs, c_base, c_base_inputs,
        c_result, c_result_inputs, d_alternatives, d_base, d_base_inputs,
        d_result, d_result_inputs, e_gate,
    )


@app.cell
def _():
    def quantity_json(value, unit):
        converted = value.to(unit)
        return {"magnitude": float(converted.magnitude), "unit": unit}

    def plain_inputs(inputs):
        return {
            key: ({"magnitude": float(value.magnitude), "unit": str(value.units)}
                  if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }

    def fairness_json(result, inputs):
        return {
            "inputs": plain_inputs(inputs), "criterion_gap": result.criterion_gap,
            "criterion_passes": result.criterion_passes,
            "ground_truth_positive_cases": result.ground_truth_positive_cases,
            "predicted_positive_cases": result.predicted_positive_cases,
            "false_negative_cases": result.false_negative_cases,
            "subgroups": [{
                "subgroup": item.subgroup,
                "counts": {"tp": item.counts.true_positive, "fp": item.counts.false_positive,
                           "tn": item.counts.true_negative, "fn": item.counts.false_negative},
                "selection_rate": item.selection_rate, "true_positive_rate": item.true_positive_rate,
                "false_positive_rate": item.false_positive_rate,
                "positive_predictive_value": item.positive_predictive_value,
            } for item in result.subgroup_metrics],
        }

    def capacity_json(result, inputs):
        return {
            "inputs": plain_inputs(inputs),
            "explanation_arrivals": quantity_json(result.explanation_arrivals, "count/hour"),
            "explanation_capacity": quantity_json(result.explanation_capacity, "count/hour"),
            "explanation_backlog": quantity_json(result.explanation_backlog, "count"),
            "review_arrivals": quantity_json(result.review_arrivals, "count/hour"),
            "review_capacity": quantity_json(result.review_capacity, "count/hour"),
            "review_backlog": quantity_json(result.review_backlog, "count"),
            "explanation_utilization": result.explanation_utilization,
            "review_utilization": result.review_utilization, "feasible": result.feasible,
        }

    def audit_json(result, inputs):
        _observed = result.observed_assessment
        return {
            "inputs": plain_inputs(inputs), "eligible_cases": result.eligible_cases,
            "sampled_cases": result.sampled_cases, "sampled_subgroups": list(result.sampled_subgroups),
            "observed_gap": None if _observed is None else _observed.criterion_gap,
            "population_gap": result.population_assessment.criterion_gap,
            "evidence_complete": result.evidence_complete,
            "limitations": list(result.limitations),
            "label_delay": quantity_json(result.label_delay, "hour"),
        }

    def privacy_json(result, inputs):
        return {
            "inputs": plain_inputs(inputs), "profile": result.profile.key,
            "scope": result.scope,
            "retained_fields": list(result.profile.retained_fields),
            "raw_content_retained": result.profile.raw_content_retained,
            "stable_subject_id_retained": result.profile.stable_subject_id_retained,
            "retention": quantity_json(result.profile.retention, "day"),
            "deletion_window": quantity_json(result.profile.deletion_window, "day"),
            "supports_subgroup_audit": result.supports_subgroup_audit,
            "supports_case_review": result.supports_case_review,
            "passes_requirements": result.passes_requirements,
            "violations": list(result.violations),
        }
    return audit_json, capacity_json, fairness_json, plain_inputs, privacy_json, quantity_json


@app.cell
def _(
    a_base, a_base_inputs, a_prediction, a_result, a_result_inputs,
    audit_json, b_base, b_base_inputs, b_prediction, b_result,
    b_result_inputs, c_base, c_base_inputs, c_prediction, c_result,
    c_result_inputs, capacity_json, capture_evidence, d_alternatives,
    d_base, d_base_inputs, d_prediction, d_result, d_result_inputs,
    e_choice, e_gate, e_owner, e_prediction, e_rejected, e_remedy,
    e_trigger, fairness_json, mechanical_gate, mo, plain_inputs,
    privacy_json, record_release_decision, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"baseline": plain_inputs(a_base_inputs), "result": plain_inputs(a_result_inputs)}
    b_upstream = {"baseline": plain_inputs(b_base_inputs), "result": plain_inputs(b_result_inputs)}
    c_upstream = {"baseline": plain_inputs(c_base_inputs), "result": plain_inputs(c_result_inputs)}
    d_upstream = {"baseline": plain_inputs(d_base_inputs), "result": plain_inputs(d_result_inputs)}
    e_upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream,
                  "choice": e_choice.value, "rejected": e_rejected.value, "owner": e_owner.value,
                  "remedy": e_remedy.value, "trigger": e_trigger.value}

    def _capture(part, prediction, inputs, baseline, result, upstream, model_key, alternatives=(), decision=None):
        return capture_evidence(
            track=track_id, part=part, prediction=prediction, inputs=plain_inputs(inputs),
            baseline=baseline, result=result, alternatives=alternatives,
            decision=decision, upstream_inputs=upstream, model_key=model_key,
        )

    a_capture = mo.ui.button(
        label="Capture threshold contrast", kind="success",
        disabled=a_prediction.value is None or a_base_inputs["threshold"] == a_result_inputs["threshold"],
        on_click=lambda _v: store("A", _capture(
            "A", a_prediction.value, a_result_inputs, fairness_json(a_base, a_base_inputs),
            fairness_json(a_result, a_result_inputs), a_upstream,
            "v2_16_experiments.assess_fairness")),
    )
    b_capture = mo.ui.button(
        label="Capture capacity contrast", kind="success",
        disabled=(b_prediction.value is None or
                  (b_base_inputs["explanation_share"] == b_result_inputs["explanation_share"] and
                   b_base_inputs["review_share"] == b_result_inputs["review_share"])),
        on_click=lambda _v: store("B", _capture(
            "B", b_prediction.value, b_result_inputs, capacity_json(b_base, b_base_inputs),
            capacity_json(b_result, b_result_inputs), b_upstream,
            "v2_16_experiments.evaluate_capacity")),
    )
    c_capture = mo.ui.button(
        label="Capture audit contrast", kind="success",
        disabled=(c_prediction.value is None or
                  (c_base_inputs["sampling_fraction"] == c_result_inputs["sampling_fraction"] and
                   c_base_inputs["label_delay"] == c_result_inputs["label_delay"])),
        on_click=lambda _v: store("C", _capture(
            "C", c_prediction.value, c_result_inputs, audit_json(c_base, c_base_inputs),
            audit_json(c_result, c_result_inputs), c_upstream,
            "v2_16_experiments.audit_evidence")),
    )
    d_capture = mo.ui.button(
        label="Capture privacy-profile contrast", kind="success",
        disabled=d_prediction.value is None or d_base_inputs["profile_key"] == d_result_inputs["profile_key"],
        on_click=lambda _v: store("D", _capture(
            "D", d_prediction.value, d_result_inputs, privacy_json(d_base, d_base_inputs),
            privacy_json(d_result, d_result_inputs), d_upstream,
            "v2_16_experiments.check_privacy_profile",
            tuple(privacy_json(value, {**d_result_inputs, "profile_key": key})
                  for key, value in d_alternatives.items()))),
    )

    _base_gate = mechanical_gate(a_base, b_base, c_base, d_base)
    _base_decision = record_release_decision(
        _base_gate, choice="defer", rejected_alternative="release", owner="", remedy="",
        reevaluation_trigger="",
    )
    _choice = e_choice.value or "defer"
    _rejected = e_rejected.value or ("release" if _choice != "release" else "defer")
    e_decision = record_release_decision(
        e_gate, choice=_choice, rejected_alternative=_rejected, owner=e_owner.value,
        remedy=e_remedy.value, reevaluation_trigger=e_trigger.value,
    )
    _base_gate_inputs = {
        "assessment": fairness_json(a_base, a_base_inputs),
        "capacity": capacity_json(b_base, b_base_inputs),
        "audit": audit_json(c_base, c_base_inputs),
        "privacy": privacy_json(d_base, d_base_inputs),
    }
    _result_gate_inputs = {
        "assessment": fairness_json(a_result, a_result_inputs),
        "capacity": capacity_json(b_result, b_result_inputs),
        "audit": audit_json(c_result, c_result_inputs),
        "privacy": privacy_json(d_result, d_result_inputs),
    }

    def _decision_json(decision, gate_inputs):
        return {
            "inputs": {"gate": gate_inputs, "choice": decision.choice,
                       "rejected_alternative": decision.rejected_alternative,
                       "owner": decision.owner, "remedy": decision.remedy,
                       "reevaluation_trigger": decision.reevaluation_trigger},
            "choice": decision.choice, "rejected_alternative": decision.rejected_alternative,
            "owner": decision.owner, "remedy": decision.remedy,
            "reevaluation_trigger": decision.reevaluation_trigger,
            "mechanical_gate_passes": decision.evidence_gate.passes,
            "mechanical_violations": list(decision.evidence_gate.violations),
            "decision_record_complete": decision.decision_record_complete,
            "missing_fields": list(decision.missing_fields),
        }

    e_capture = mo.ui.button(
        label="Capture release record", kind="success",
        disabled=(e_prediction.value is None or e_choice.value is None or
                  e_rejected.value is None or not e_decision.decision_record_complete),
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value, inputs=e_upstream,
            baseline=_decision_json(_base_decision, _base_gate_inputs),
            result=_decision_json(e_decision, _result_gate_inputs),
            alternatives=(_decision_json(_base_decision, _base_gate_inputs),),
            decision=e_decision.choice, upstream_inputs=e_upstream,
            model_key="v2_16_experiments.record_release_decision")),
    )
    return (
        a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream,
        d_capture, d_upstream, e_capture, e_decision, e_upstream,
    )

# ZONE C · Header and single tabbed investigation surface

@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = mo.Html("""
    <style>
    .rai-head{background:linear-gradient(135deg,#22131b,#7f1d35);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin:30px 0 12px}
    .rai-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}.rai-head h1{font-size:clamp(1.6rem,5vw,2.6rem);line-height:1.08;margin:16px 0 8px}.rai-head p{color:#ffe4e6;max-width:790px}
    .rai-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:9px;margin-top:17px}.rai-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}.rai-note{color:#475569;font-size:.9rem;line-height:1.5}.rai-saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-active{color:#86efac}@media(max-width:520px){.rai-head{border-radius:9px}.rai-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="rai-head"><div class="rai-top"><span>VOLUME II · LAB 16</span><span>ABOUT 50–55 MIN</span></div><h1>Evidence for Responsible Release</h1><p>When do subgroup outcomes, audit evidence, and review capacity justify a fleet release decision?</p><div class="rai-meta"><div><b>Track</b><br>{profile['display']}</div><div><b>Role</b><br>{profile['fleet_role']}</div><div><b>Workload</b><br>{profile['events_per_hour'].m_as('count/hour'):,.0f} ev/h</div><div><b>Capacity</b><br>{profile['explanation_workers']} exp / {profile['reviewers']} rev</div><div><b>Cohorts</b><br>{profile['subgroups'][0]} / {profile['subgroups'][1]}</div><div><b>Output</b><br>Auditable release record</div></div></section>""")
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, track, header, mo.Html(
        '<p class="rai-note">Scores, labels, workloads, and data profiles are illustrative scenario assumptions. Passing a data-handling profile is not a privacy guarantee.</p>')], gap=0.5).style({"padding-top": "32px"})
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_allowed_gap, a_base, a_capture, a_criterion, a_prediction,
    a_result, a_threshold, a_upstream, apply_plotly_theme, audit_evidence,
    b_base, b_capture, b_explanation_share, b_prediction, b_result,
    b_review_share, b_upstream, c_base, c_capture, c_label_delay,
    c_prediction, c_result, c_sampling, c_upstream, d_alternatives,
    d_capture, d_case_review, d_prediction, d_profile, d_result, d_upstream,
    e_capture, e_choice, e_decision, e_gate, e_owner, e_prediction,
    e_rejected, e_remedy, e_trigger, e_upstream, get_evidence, go, mo,
    profile, rationale, residual_risk, track_id,
):
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream,
                 "D": d_upstream, "E": e_upstream}
    evidence_audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=_upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def _table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style(
            {"max-width": "100%", "overflow-x": "auto"})

    def _saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in evidence_audit.stale or (part, part) in evidence_audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture this experiment."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div class="rai-saved"><b>Saved snapshot</b> · prediction: {data["prediction"]}<br><small>The report uses this fixed result.</small></div>')

    def _gap(value):
        return "undefined" if value is None else f"{100 * value:.1f} pp"

    def build_part_a():
        _intro = mo.md("### A · Which fairness claim does the evidence support? (10 min)\nChoose a normative criterion, then replay fixed scores and labels at a new threshold.")
        if a_prediction.value is None:
            return mo.vstack([_intro, a_prediction])
        _fig = go.Figure()
        for item in a_result.subgroup_metrics:
            _fig.add_bar(name=item.subgroup, x=["Selection", "TPR", "FPR", "PPV"], y=[
                100 * item.selection_rate,
                None if item.true_positive_rate is None else 100 * item.true_positive_rate,
                None if item.false_positive_rate is None else 100 * item.false_positive_rate,
                None if item.positive_predictive_value is None else 100 * item.positive_predictive_value])
        _fig.update_layout(barmode="group", height=290, margin=dict(l=20, r=20, t=20, b=20),
                           yaxis_title="Rate (%)", legend_orientation="h")
        _rows = [{"Run": label, "Threshold": f"{result.threshold:.2f}",
                  "Chosen gap": _gap(result.criterion_gap),
                  "Status": "PASS" if result.criterion_passes else "FAIL / INSUFFICIENT",
                  "True labels": result.ground_truth_positive_cases,
                  "False negatives": result.false_negative_cases}
                 for label, result in (("Baseline", a_base), ("Selected", a_result))]
        return mo.vstack([_intro, a_prediction, mo.hstack([a_threshold, a_criterion, a_allowed_gap], wrap=True),
                          apply_plotly_theme(_fig), _table(_rows),
                          mo.callout(mo.md(f"**Prediction:** {a_prediction.value}. The chosen gap is **{_gap(a_result.criterion_gap)}**. The allowed gap classifies evidence; it does not change the **{a_result.ground_truth_positive_cases}** true labels."),
                                     kind="success" if a_result.criterion_passes else "danger"),
                          a_capture, _saved("A"), mo.accordion({"Calculation Notes": mo.md(
                              "MLSysIM thresholds each score, builds subgroup confusion matrices, and derives rates from their denominators. Empty denominators remain undefined. Criterion and tolerance only evaluate those counts.")})])

    def build_part_b():
        _intro = mo.md(f"### B · Can the fleet serve explanations and review? (10 min)\n**Fleet role:** {profile['fleet_role']}. The service tier receives **{profile['events_per_hour'].m_as('count/hour'):,.0f} events/hour**. Predict which queue binds.")
        if b_prediction.value is None:
            return mo.vstack([_intro, b_prediction])
        _fig = go.Figure([
            go.Bar(name="Arrivals", x=["Explanation", "Human review"],
                   y=[b_result.explanation_arrivals.m_as("count/hour"), b_result.review_arrivals.m_as("count/hour")], marker_color=COLORS["OrangeLine"]),
            go.Bar(name="Capacity", x=["Explanation", "Human review"],
                   y=[b_result.explanation_capacity.m_as("count/hour"), b_result.review_capacity.m_as("count/hour")], marker_color=COLORS["BlueLine"])])
        _fig.update_layout(barmode="group", height=280, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Cases/hour")
        _rows = [{"Run": label, "Explanation utilization": f"{100*r.explanation_utilization:.1f}%",
                  "Review utilization": f"{100*r.review_utilization:.1f}%",
                  "Explanation backlog": f"{r.explanation_backlog.m_as('count'):,.0f}",
                  "Review backlog": f"{r.review_backlog.m_as('count'):,.0f}",
                  "Status": "PASS" if r.feasible else "FAIL"}
                 for label, r in (("Baseline", b_base), ("Selected", b_result))]
        return mo.vstack([_intro, b_prediction, mo.hstack([b_explanation_share, b_review_share], widths="equal", wrap=True),
                          apply_plotly_theme(_fig), _table(_rows),
                          mo.callout(mo.md(f"**Prediction:** {b_prediction.value}. Review demand is **{b_result.review_arrivals.m_as('count/hour'):,.0f}/hour** against **{b_result.review_capacity.m_as('count/hour'):,.0f}/hour** capacity. Coverage consumes capacity; it does not repair Part A outcomes."),
                                     kind="success" if b_result.feasible else "danger"),
                          b_capture, _saved("B"), mo.accordion({"Calculation Notes": mo.md(
                              "Arrivals equal fleet rate × selected share into offloaded queues. Capacity equals workers × target utilization ÷ service time. Eight-hour backlog grows only above capacity.")})])

    def build_part_c():
        _intro = mo.md("### C · What can delayed labels and sampling establish? (9 min)\nKeep true cases fixed. Change only which labeled cases the audit observes.")
        if c_prediction.value is None:
            return mo.vstack([_intro, c_prediction])
        _observed = c_result.observed_assessment
        _observed_gap = "undefined" if _observed is None else _gap(_observed.criterion_gap)
        _fig = go.Figure([go.Bar(x=["Fixture", "Eligible", "Sampled"],
                                y=[24, c_result.eligible_cases, c_result.sampled_cases],
                                marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"]])])
        _fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Cases", showlegend=False)
        _rows = [{"Run": label, "Eligible": r.eligible_cases, "Sampled": r.sampled_cases,
                  "Observed gap": "undefined" if r.observed_assessment is None else _gap(r.observed_assessment.criterion_gap),
                  "Population gap": _gap(r.population_assessment.criterion_gap),
                  "Limitations": "; ".join(r.limitations) or "none in fixture"}
                 for label, r in (("Full prompt", c_base), ("Selected", c_result))]
        return mo.vstack([_intro, c_prediction, mo.hstack([c_sampling, c_label_delay], widths="equal", wrap=True),
                          apply_plotly_theme(_fig), _table(_rows),
                          mo.callout(mo.md(f"**Prediction:** {c_prediction.value}. The plan observes **{c_result.sampled_cases}** cases and reports **{_observed_gap}**. The population gap remains **{_gap(c_result.population_assessment.criterion_gap)}**."),
                                     kind="danger" if _observed_gap == "undefined" else "info"),
                          c_capture, _saved("C"), mo.accordion({"Calculation Notes": mo.md(
                              "MLSysIM filters labels by age, applies deterministic cohort sampling, and computes an observed metric separately from the unchanged population assessment.")})])

    def build_part_d():
        _intro = mo.md(f"### D · Which evidence profile meets the data obligation? (9 min)\n**Fleet scope:** {profile['fleet_role']}. Compare retained fields, retention, and deletion windows. No percentage represents privacy.")
        if d_prediction.value is None:
            return mo.vstack([_intro, d_prediction])
        _rows = [{"Profile": r.profile.label,
                  "Scope": r.scope or profile["fleet_role"],
                  "Raw content": "retained" if r.profile.raw_content_retained else "not retained",
                  "Stable ID": "retained" if r.profile.stable_subject_id_retained else "not retained",
                  "Retention": f"{r.profile.retention.m_as('day'):.0f} days",
                  "Deletion": f"{r.profile.deletion_window.m_as('day'):.0f} days",
                  "Case review": "yes" if r.supports_case_review else "no",
                  "Specified checks": "PASS" if r.passes_requirements else "FAIL"}
                 for r in d_alternatives.values()]
        return mo.vstack([_intro, d_prediction, mo.hstack([d_profile, d_case_review], widths="equal", wrap=True),
                          _table(_rows), mo.callout(mo.md(
                              f"**Prediction:** {d_prediction.value}. **{d_result.profile.label}** ({d_result.scope or profile['fleet_role']}) {'meets' if d_result.passes_requirements else 'does not meet'} the specified checks. This is not a privacy guarantee."),
                              kind="success" if d_result.passes_requirements else "danger"),
                          d_capture, _saved("D"), mo.accordion({"Calculation Notes": mo.md(
                              "The backend checks retained cohort labels, required case-level scored events, and explicit retention and deletion durations against the active fleet role.")})])

    def build_part_e():
        _intro = mo.md("### E · What decision follows from the evidence? (8 min)\nInspect four mechanical gates, then record a choice, owner, remedy, and trigger. The policy label cannot change a metric.")
        if e_prediction.value is None:
            return mo.vstack([_intro, e_prediction])
        _estimable = c_result.observed_assessment is not None and c_result.observed_assessment.criterion_gap is not None
        _rows = [{"Gate": "Fairness criterion", "Status": "PASS" if a_result.criterion_passes else "FAIL / INSUFFICIENT"},
                 {"Gate": "Service capacity", "Status": "PASS" if b_result.feasible else "FAIL"},
                 {"Gate": "Audit estimate", "Status": "PASS" if _estimable else "INSUFFICIENT EVIDENCE"},
                 {"Gate": "Privacy obligations", "Status": "PASS" if d_result.passes_requirements else "FAIL"}]
        _message = "all specified gates pass" if e_gate.passes else " · ".join(e_gate.violations)
        return mo.vstack([_intro, e_prediction, _table(_rows),
                          mo.callout(mo.md(f"**Mechanical result:** {_message}. You still own the normative release choice."),
                                     kind="success" if e_gate.passes else "danger"),
                          mo.hstack([e_choice, e_rejected], widths="equal", wrap=True),
                          mo.hstack([e_owner, e_remedy, e_trigger], wrap=True),
                          mo.callout(mo.md("Decision record complete." if e_decision.decision_record_complete else
                                          "Name a distinct alternative, owner, remedy, and trigger."),
                                     kind="info" if e_decision.decision_record_complete else "warn"),
                          e_capture, _saved("E"), mo.accordion({"Calculation Notes": mo.md(
                              "The gate combines observable requirements. record_release_decision stores qualitative governance choices without bonuses or metric changes.")})])

    def build_synthesis():
        _rows = []
        for part in "ABCDE":
            _capture = _captures.get(part)
            _current = _capture and part not in evidence_audit.stale and (part, part) not in evidence_audit.identical_pairs
            _rows.append({"Part": part, "Prediction": _capture.to_dict()["prediction"] if _capture else "—",
                          "Evidence": "CURRENT" if _current else ("STALE" if _capture else "MISSING")})
        _saved_choice = _captures["E"].to_dict()["decision"] if "E" in _captures else None
        _complete = evidence_audit.complete and residual_risk.value is not None and bool(rationale.value.strip()) and _saved_choice == e_choice.value
        return mo.vstack([mo.md("### Synthesis · Defend the release record (5 min)\nState the chosen option, quantify the rejected alternative, name the remaining limitation, and give the reevaluation trigger."),
                          _table(_rows), residual_risk, rationale,
                          mo.callout(mo.md("**Ready for the local report.**" if _complete else
                                          "Capture five current contrasts, keep the Part E choice, choose a limitation, and write the quantified rationale."),
                                     kind="success" if _complete else "warn")])

    tabs = mo.ui.tabs({"Part A": build_part_a(), "Part B": build_part_b(),
                       "Part C": build_part_c(), "Part D": build_part_d(),
                       "Part E": build_part_e(), "Synthesis": build_synthesis()})
    tabs
    return (evidence_audit,)

# ZONE D · Completion-gated report and ledger

@app.cell
def _(
    build_lab_report, e_choice, e_rejected, e_trigger, evidence_audit,
    get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, residual_risk, track_id,
):
    _captures = get_evidence()
    _saved_choice = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = evidence_audit.complete and residual_risk.value is not None and bool(rationale.value.strip()) and _saved_choice == e_choice.value
    mo.stop(not _ready)
    _snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    _result_snapshot = {
        "schema_version": 1, "lab_id": "v2_16", "track_id": track_id,
        "model_id": "v2_16_experiments", "evidence": _snapshots,
        "recommendation": e_choice.value, "rejected_alternative": e_rejected.value,
        "reevaluation_trigger": e_trigger.value, "residual_risk": residual_risk.value,
        "rationale": rationale.value,
    }
    report = build_lab_report(
        get_lab_metadata("vol2/lab_16_responsible_ai.py"), track=track_id,
        scenario=f"{profile['display']} fleet ({profile['fleet_role']}) subgroup evidence and review service",
        learning_objectives=["Compute subgroup metrics from scored labels",
                             "Separate audit evidence from population outcomes",
                             "Size explanation and review capacity",
                             "Record a release choice without a governance score"],
        predictions={part: _snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: _snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": _snapshots[part]["baseline"],
                                 "result": _snapshots[part]["result"],
                                 "alternatives": _snapshots[part]["alternatives"]}
                          for part in "ABCDE"},
        binding_constraints={"release_evidence": _snapshots["E"]["result"]["mechanical_violations"]},
        decisions={"recommendation": e_choice.value, "rejected_alternative": e_rejected.value,
                   "reevaluation_trigger": e_trigger.value},
        final_decision={"recommendation": e_choice.value, "rejected_alternative": e_rejected.value,
                        "rationale": rationale.value},
        big_takeaways=["Fairness criteria can disagree on identical decisions.",
                       "Monitoring changes evidence, not true outcomes.",
                       "Explanation and review consume finite capacity."],
        reflections={"rationale": rationale.value, "reevaluation_trigger": e_trigger.value},
        residual_risk=residual_risk.value, result_snapshot=_result_snapshot,
        source_trace={"scenario_assumption": f"Illustrative fixtures for {profile['fleet_role']}: {profile['events_per_hour'].m_as('count/hour'):,.0f} events/hour ({profile['explanation_workers']} explanation, {profile['reviewers']} review workers); not measurements or a privacy guarantee.",
                      "calculation_boundary": "The simulator computes scored-label metrics, queues, audit evidence, and categorical data checks."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    e_choice, e_rejected, e_trigger, evidence_audit, get_evidence, ledger,
    mo, rationale, residual_risk, track_id,
):
    _captures = get_evidence()
    _saved_choice = _captures["E"].to_dict()["decision"] if "E" in _captures else None
    _ready = evidence_audit.complete and residual_risk.value is not None and bool(rationale.value.strip()) and _saved_choice == e_choice.value
    _save_state = "EVIDENCE IN PROGRESS"
    _save_detail = ""
    if _ready:
        try:
            ledger.save(chapter=16, design={
                "schema_version": 1, "lab_id": "v2_16", "track_id": track_id,
                "model_id": "v2_16_experiments",
                "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "recommendation": e_choice.value, "rejected_alternative": e_rejected.value,
                "reevaluation_trigger": e_trigger.value, "residual_risk": residual_risk.value,
                "rationale": rationale.value,
            })
            await ledger.flush()
            _save_state = "SAVED"
        except Exception as exc:
            _save_state = "SAVE FAILED"
            _save_detail = f" | {type(exc).__name__}"
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span>16 | Evidence for Responsible Release</span><span style="flex:1"></span><span class="hud-label">STATUS</span><span class="hud-active">{_save_state}</span><span>{_save_detail}</span></div>')
    return


if __name__ == "__main__":
    app.run()
