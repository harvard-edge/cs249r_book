import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 15: Responsible Engineering · MLSysBook")


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
    from mlsysim.labs.experiments.v1_15_experiments import (
        LINEAGE_POLICIES, TRACKS, bounded_mean_release, investigate_incident,
        lifecycle_footprint, lifecycle_policy, population_mix, threshold_consequences,
    )
    from mlsysim.labs.experiments.v1_15_experiments import fairness_criteria, proxy_alignment, feedback_replay, FEEDBACK_MAX_ROUNDS
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata, report_export_panel,
    )
    from mlsysim import Scenarios, fmt_percent
    from mlsysbook_labs.learning_paths import get_learning_path
    from mlsysbook_labs.notebook import (
        notebook_bridge, notebook_header, notebook_learning_map, notebook_learning_part, notebook_tab_hint,
        notebook_shell, notebook_status, notebook_table,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence

    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (FEEDBACK_MAX_ROUNDS, proxy_alignment, feedback_replay, fmt_percent, fairness_criteria, Scenarios, get_learning_path, notebook_bridge, notebook_header,
            notebook_learning_map, notebook_learning_part, notebook_tab_hint, notebook_shell, notebook_status, notebook_table,
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, LINEAGE_POLICIES, TRACKS,
        apply_plotly_theme, audit_evidence, bounded_mean_release,
        build_lab_report, capture_evidence, get_lab_metadata, go,
        investigate_incident, ledger, lifecycle_footprint, lifecycle_policy, mo, population_mix,
        report_export_panel, threshold_consequences,
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
def _(TRACKS, track):
    track_id = track.value
    profile = TRACKS[track_id]
    return profile, track_id


@app.cell
def _(FEEDBACK_MAX_ROUNDS, mo, track_id):
    _selected_project = track_id
    fairness_criterion = mo.ui.dropdown({"Demographic parity: selection rate": "demographic_parity", "Equal opportunity: true-positive rate": "equal_opportunity", "Equalized odds: true- and false-positive rates": "equalized_odds"},
                                       value="Equal opportunity: true-positive rate", label="Declared release criterion")
    a_policy = mo.ui.dropdown({"Optimize clicks": "proxy_first", "Use independently audited benefit": "benefit_first"}, value="Use independently audited benefit", label="Recommendation policy")
    a_outcome_audit = mo.ui.dropdown({"Complete independent follow-up": "complete", "No independent outcomes": "none"}, value="Complete independent follow-up", label="Benefit evidence")
    feedback_rounds = mo.ui.slider(1, FEEDBACK_MAX_ROUNDS, value=FEEDBACK_MAX_ROUNDS, label="Supplied rounds to replay")
    feedback_audit = mo.ui.dropdown({"Routine feedback only": "routine", "Independent sample": "independent_sample", "Complete finite-population audit": "complete", "Serving exploration": "exploration"},
                                   value="Independent sample", label="Independent observation policy")
    return a_policy, a_outcome_audit, feedback_rounds, fairness_criterion, feedback_audit


@app.cell
def _(a_policy, a_outcome_audit, proxy_alignment, feedback_replay, feedback_rounds, fairness_criteria, fairness_criterion, feedback_audit, threshold_values, track_id):
    proxy_baseline = proxy_alignment(track_id)
    proxy_selected = proxy_alignment(track_id, policy=a_policy.value, outcome_audit=a_outcome_audit.value)
    fairness_baseline = fairness_criteria(track_id, criterion="demographic_parity")
    fairness_selected = fairness_criteria(track_id, criterion=fairness_criterion.value)
    fairness_candidates = {name: fairness_criteria(track_id, criterion=fairness_criterion.value, threshold=threshold)
                           for name, threshold in threshold_values.items()}
    feedback_baseline = feedback_replay(track_id, rounds=feedback_rounds.value)
    feedback_selected = feedback_replay(track_id, observation_policy=feedback_audit.value, rounds=feedback_rounds.value)
    return proxy_baseline, proxy_selected, fairness_baseline, fairness_selected, fairness_candidates, feedback_baseline, feedback_selected


@app.cell
def _(mo, profile, track_id):
    _track_key = track_id
    a_share = mo.ui.slider(20, 90, value=70, step=5, label="Affected-group share (%)")
    b_choice = mo.ui.radio(
        {
            "Permissive (0.30)": "permissive", "Balanced (0.50)": "balanced",
            "Strict (0.70)": "strict", "Hold: no threshold is defensible": "hold",
        }, label="Threshold recommendation",
    )
    b_rejected = mo.ui.radio(
        {"Permissive (0.30)": "permissive", "Balanced (0.50)": "balanced", "Strict (0.70)": "strict"},
        label="Quantified rejected threshold",
    )
    c_epsilon = mo.ui.slider(0.25, 2.0, value=0.5, step=0.25, label="Epsilon per release")
    c_releases = mo.ui.slider(1, 6, value=4, step=1, label="Independent releases")
    d_demand = mo.ui.dropdown(
        {"Half demand": 0.5, "Reference demand": 1.0, "Double demand": 2.0},
        value="Reference demand", label="Inference demand",
    )
    d_execution = mo.ui.dropdown(
        {"Reference execution": "reference", "Lower-energy candidate · replay supplied": "replayed_candidate",
         "Lower-energy candidate · replay missing": "unreplayed_candidate",
         "Lower-energy candidate · one decision changed": "changed_decision_candidate"},
        value="Lower-energy candidate · replay supplied", label="Execution evidence",
    )
    d_retraining = mo.ui.slider(0, 6, value=2, step=1, label="Retraining runs")
    d_explanations = mo.ui.slider(0, 30, value=int(profile["explanation_share_pct"]), step=1, label="Decisions explained (%)")
    e_policy = mo.ui.dropdown(
        {"Full lineage": "full", "Decision evidence only": "decision_only", "Minimal release record": "minimal"},
        value="Decision evidence only", label="Retained evidence",
    )
    e_delay = mo.ui.slider(0.5, 12.0, value=profile["default_containment_delay_hours"], step=0.5, label="Containment delay (hours)")
    return a_share, b_choice, b_rejected, c_epsilon, c_releases, d_demand, d_execution, d_explanations, d_retraining, e_delay, e_policy


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"More clicks guarantee more benefit": "proxy", "Clicks can rise while independently audited benefit falls": "diverge", "Missing independent labels prove success": "missing"},
        label="Does optimizing clicks establish the intended benefit?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"TPR and FPR both rise": "both_rise", "TPR and FPR both fall": "both_fall", "TPR rises while FPR falls": "opposite", "Neither changes": "neither"},
        label="What happens when the positive-score threshold rises?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Routine feedback identifies all outcomes": "routine", "Independent observation can reveal missing outcomes": "audit", "Any small audit identifies the entire population": "small_audit"},
        label="Can selected feedback keep a population invisible in later rounds?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Initial training": "initial_training", "Retraining": "retraining", "Inference": "inference", "Explanation": "explanation", "Candidate preparation": "preparation"},
        label="Which lifecycle term dominates under this workload?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Fairness outcomes improve": "fairness", "Decision reproduction is lost": "reproduction", "Containment becomes instantaneous": "instant", "Formal epsilon decreases": "epsilon"},
        label="What can disappear when lineage evidence is removed?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Release · permissive threshold": "permissive", "Release · balanced threshold": "balanced", "Release · strict threshold": "strict", "Hold · no defensible release": "hold"},
        label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {"Permissive threshold": "permissive", "Balanced threshold": "balanced", "Strict threshold": "strict"},
        label="Rejected tested alternative",
    )
    final_trigger = mo.ui.radio(
        {"Subgroup TPR or FPR changes": "fairness_shift", "Independent audit coverage changes": "audit_coverage", "Lifecycle demand doubles": "lifecycle_demand", "Required lineage field is missing": "lineage_gap"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Small fixed evaluation fixture": "fixture", "Unmodeled lifecycle boundary": "boundary", "Unspecified consequence severity": "consequences", "Future distribution shift": "shift"},
        label="Residual limitation",
    )
    rationale = mo.ui.text_area(
        label="Concise rationale",
        placeholder="Use saved criterion, error, observation, lifecycle, and reconstructability evidence to defend the choice.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    LINEAGE_POLICIES, a_share, b_choice, b_rejected, bounded_mean_release,
    c_epsilon, c_releases, d_demand, d_execution, d_explanations, d_retraining, e_delay,
    e_policy, investigate_incident, lifecycle_policy, population_mix,
    threshold_consequences, track_id,
):
    threshold_values = {"permissive": 0.3, "balanced": 0.5, "strict": 0.7}
    a_base = population_mix(track_id, threshold=0.5)
    a_result = population_mix(track_id, affected_share_pct=a_share.value, threshold=0.5)
    b_results = {
        name: threshold_consequences(track_id, threshold=value, affected_share_pct=a_share.value)
        for name, value in threshold_values.items()
    }
    if b_choice.value == "hold":
        b_result_key = b_rejected.value or "strict"
        b_base_key = "balanced"
    else:
        b_result_key = b_choice.value or "strict"
        b_base_key = b_rejected.value or "permissive"
    b_base = b_results[b_base_key]
    b_result = b_results[b_result_key]
    carried_policy = b_result_key
    carried_threshold = threshold_values[carried_policy]
    c_base = bounded_mean_release(track_id, epsilon_per_release=2.0, releases=1, seed=15)
    c_result = bounded_mean_release(track_id, epsilon_per_release=c_epsilon.value, releases=c_releases.value, seed=15)
    d_base = lifecycle_policy(track_id)
    d_result = lifecycle_policy(
        track_id, execution=d_execution.value, demand_scale=d_demand.value, retraining_runs=d_retraining.value,
        explanation_share_pct=d_explanations.value,
    )
    e_base = investigate_incident(
        track_id, retained_fields=LINEAGE_POLICIES["full"], containment_delay_hours=0.5,
        threshold=carried_threshold, affected_share_pct=a_share.value,
    )
    e_result = investigate_incident(
        track_id, retained_fields=LINEAGE_POLICIES[e_policy.value], containment_delay_hours=e_delay.value,
        threshold=carried_threshold, affected_share_pct=a_share.value,
    )
    return a_base, a_result, b_base, b_result, b_results, carried_policy, carried_threshold, c_base, c_result, d_base, d_result, e_base, e_result, threshold_values


@app.cell
def _(proxy_baseline, proxy_selected, fairness_baseline, fairness_selected, fairness_candidates, feedback_baseline, feedback_selected,
    a_base, a_prediction, a_result, a_share, b_base, b_choice, b_prediction,
    b_rejected, b_result, b_results, c_base, c_epsilon, c_prediction,
    c_releases, c_result, capture_evidence, d_base, d_demand, d_execution,
    d_explanations, d_prediction, d_result, d_retraining, e_base, e_delay,
    e_policy, e_prediction, e_result, mo, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    _a_base_share = a_base["affected_share_pct"]
    a_upstream = {"baseline": proxy_baseline["inputs"], "result": proxy_selected["inputs"], "criterion": fairness_selected["selected_criterion"]}
    b_upstream = {"choice": b_choice.value, "rejected": b_rejected.value, "criterion": fairness_selected["selected_criterion"]}
    c_upstream = {"baseline": feedback_baseline["inputs"], "result": feedback_selected["inputs"]}
    d_upstream = d_result["inputs"]
    e_upstream = {"threshold": e_result["inputs"]["threshold"], "affected_share_pct": a_share.value, "retention_policy": e_policy.value, "containment_delay_hours": e_delay.value}
    a_capture = mo.ui.button(
        label="Capture proxy-versus-benefit contrast", kind="success", disabled=a_prediction.value is None or proxy_baseline["inputs"] == proxy_selected["inputs"],
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value, inputs=a_upstream,
            baseline=proxy_baseline, result=proxy_selected, upstream_inputs=a_upstream,
            model_key="v1_15_experiments.proxy_alignment",
        )),
    )
    b_invalid = (
        b_prediction.value is None or b_choice.value is None or b_rejected.value is None
        or (b_choice.value != "hold" and b_choice.value == b_rejected.value)
        or (b_choice.value == "hold" and b_rejected.value == "balanced")
    )
    b_capture = mo.ui.button(
        label="Capture threshold decision", kind="success", disabled=b_invalid,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"choice": b_choice.value, "rejected": b_rejected.value}, baseline=fairness_candidates[b_rejected.value or "balanced"],
            result=fairness_candidates[b_choice.value if b_choice.value in fairness_candidates else "balanced"], alternatives=tuple(fairness_candidates.values()), decision=b_choice.value,
            chosen_result=fairness_candidates[b_rejected.value or "balanced"] if b_choice.value == "hold" else fairness_candidates[b_choice.value if b_choice.value in fairness_candidates else "balanced"],
            result_role="rejected alternative" if b_choice.value == "hold" else "chosen candidate",
            upstream_inputs=b_upstream, model_key="v1_15_experiments.fairness_criteria",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture observation contrast", kind="success",
        disabled=c_prediction.value is None or feedback_baseline["inputs"] == feedback_selected["inputs"],
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs=c_upstream, baseline=feedback_baseline, result=feedback_selected, upstream_inputs=c_upstream,
            model_key="v1_15_experiments.feedback_replay",
        )),
    )
    d_capture = mo.ui.button(
        label="Capture lifecycle boundary", kind="success",
        disabled=d_prediction.value is None or d_base["inputs"] == d_result["inputs"],
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value, inputs=d_upstream,
            baseline=d_base, result=d_result, upstream_inputs=d_upstream,
            model_key="v1_15_experiments.lifecycle_policy",
        )),
    )
    e_capture = mo.ui.button(
        label="Capture investigation contrast", kind="success",
        disabled=e_prediction.value is None or (e_policy.value == "full" and e_delay.value == 0.5),
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"retention_policy": e_policy.value, "containment_delay_hours": e_delay.value},
            baseline=e_base, result=e_result, upstream_inputs=e_upstream,
            model_key="v1_15_experiments.investigate_incident",
        )),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream, e_capture, e_upstream


@app.cell
def _(Scenarios, get_lab_metadata, get_learning_path, mo, notebook_bridge,
      notebook_header, notebook_learning_map, notebook_shell, track, track_id):
    from mlsysbook_labs.notebook import notebook_scenario as _scenario
    learning_path = get_learning_path(1, 15)
    journey = Scenarios.DeploymentJourneys.get(track_id)
    episode = journey.episode(15)
    learning_path["narrative"] = {
        "project": journey.mission,
        "scene": episode.scene,
        "carry_forward": episode.handoff,
        "scope": 'The supplied scores, labels, privacy summaries, and incident profiles are controlled reference studies. Their tasks differ from the continuing project; apply the reasoning, not their numerical conclusions.',
    }
    _previous = journey.episode(14).handoff
    notebook_shell([
        notebook_header(get_lab_metadata("vol1/lab_15_responsible_engr.py"),
                        'What evidence and tradeoffs justify a responsible release?',
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
def _(a_policy, a_outcome_audit, proxy_baseline, proxy_selected, feedback_rounds, fmt_percent, notebook_tab_hint, notebook_learning_part, fairness_criterion, fairness_baseline, fairness_selected, fairness_candidates, feedback_audit, feedback_baseline, feedback_selected, learning_path, notebook_shell, notebook_status, notebook_table,
    COLORS, a_base, a_capture, a_prediction, a_result, a_share, a_upstream,
    apply_plotly_theme, audit_evidence, b_capture, b_choice, b_prediction,
    b_rejected, b_results, b_upstream, c_base, c_capture, c_epsilon,
    c_prediction, c_releases, c_result, c_upstream, d_base, d_capture,
    d_demand, d_execution, d_explanations, d_prediction, d_result, d_retraining,
    d_upstream, e_base, e_capture, e_delay, e_policy, e_prediction, e_result,
    e_upstream, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, go, mo, profile, rationale, track_id,
):
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=_upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return notebook_table(rows)

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** A dependency changed, or the saved runs are identical. Recapture this part."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div style="border-left:3px solid #247a4d;padding:8px 12px;background:#f8fafc"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; the report uses this saved result rather than changed live controls.</small></div>')

    def part_a():
        intro = mo.md("The team wants recommendations that people find helpful, but its dashboard rewards clicks. Compare two supplied policies with the same slot budget. Independent follow-up measures the intended benefit. This affected-population recommendation reference does not certify the native project.")
        controls = mo.vstack([a_policy, a_outcome_audit])
        if a_prediction.value is None:
            return mo.vstack([intro, controls, a_prediction])
        outcomes = table([{"Policy": name, "Slots": result["slots"], "Clicks": result["proxy_clicks"], "Independent benefits": result["independent_benefits"] if result["independent_benefits"] is not None else "Unknown", "Follow-ups": result["independent_followups"], "Benefit gate": result["benefit_gate"]} for name, result in (("Starting: optimize clicks", proxy_baseline), ("Selected", proxy_selected))])
        gates = table([{"Criterion": name.replace("_", " "), "Gap (percentage points)": fairness_selected["criterion_gap_pp"][name], "Declared gate": gate} for name, gate in fairness_selected["criterion_gates"].items()])
        return mo.vstack([intro, controls, a_prediction, outcomes,
            mo.callout(mo.md(f"The selected independent-benefit gate is **{proxy_selected['benefit_gate']}**. A click count cannot replace the missing outcome audit. Both policies have the same slot budget."), kind="info"),
            a_capture, saved("A"), mo.md("**Explain:** which policy would the click dashboard reward, and which meets the intended outcome? Remove the independent audit and explain why the answer becomes unknown rather than unchanged."),
            mo.md("**Carry into Part B:** declare the affected-population criterion before comparing thresholds. This is a separate fixed-score study; changing the criterion does not retrain its model."), fairness_criterion,
            mo.accordion({"Inspect the declared criterion": mo.vstack([gates, mo.md(fairness_selected["limitation"])]),
                          "Inspect supplied opportunity records": table(proxy_selected["records"]),
                          "Optional population-mixture reference": mo.vstack([a_share, table([{"Mixture": label, "Aggregate accuracy": result["aggregate"]["accuracy"]} for label, result in (("Baseline", a_base), ("Changed", a_result))])]),
                          "Calculation Notes": mo.md(proxy_selected["limitation"] + f" The independent-benefit obligation is at least {proxy_selected['minimum_benefits']} helpful recommendations. It is a supplied policy, not a universal standard.")})])

    def part_b():
        intro = mo.md("Keep Part A's declared criterion and the audited cases fixed. Compare the supported thresholds, then recommend one or hold. Distinguish a measured error tradeoff from a promised trained-model improvement.")
        if b_prediction.value is None:
            return mo.vstack([intro, b_prediction])
        rows = []
        for name, result in fairness_candidates.items():
            for group, outcomes in result["groups"].items():
                rows.append({"Policy": name.title(), "Population": group, "False positives": outcomes["fp"], "False negatives": outcomes["fn"], "TPR": fmt_percent(outcomes["tpr"], style="symbol"), "FPR": fmt_percent(outcomes["fpr"], style="symbol"), "Accuracy": fmt_percent(outcomes["accuracy"], style="symbol"), "Criterion gate": result["selected_gate"]})
        decision_rows = []
        for role, name in (("Selected", b_choice.value), ("Rejected", b_rejected.value)):
            if name in fairness_candidates:
                result = fairness_candidates[name]
                for group, outcomes in result["groups"].items():
                    decision_rows.append({"Decision role": role, "Threshold": name.title(), "Population": group,
                        "False positives": outcomes["fp"], "False negatives": outcomes["fn"],
                        "Criterion gate": result["selected_gate"]})
        if b_choice.value == "hold":
            decision_feedback = mo.callout(mo.md("**Decision consequence: HOLD.** No threshold is advanced. The rejected candidate's measured errors below remain evidence to discuss; holding does not change those predictions or establish a better model."), kind="warn")
        elif b_choice.value in fairness_candidates:
            decision_feedback = mo.callout(mo.md(f"**Decision consequence:** the selected **{b_choice.value.title()}** threshold has criterion gate **{fairness_candidates[b_choice.value]['selected_gate']}**. Compare its population-specific false positives and false negatives with the rejected threshold below. Passing this supplied criterion does not establish deployment approval."), kind="info")
        else:
            decision_feedback = mo.md("Choose a threshold or hold, and identify an alternative to see the decision contrast.")
        return mo.vstack([intro, b_prediction, table(rows), mo.hstack([b_choice, b_rejected], widths="equal", wrap=True),
            decision_feedback, table(decision_rows) if decision_rows else mo.md("No decision comparison selected."),
            mo.md("**Explain:** which population and error type pays for your choice? If two thresholds produce unchanged predictions, preserve that no-effect result rather than claiming an improvement. Your release or hold recommendation remains limited to this reference."),
            b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md("Each threshold replays identical audited scores and labels. Criterion choice is inherited from Part A. The observed error counts and policy gates do not establish the effects of retraining, reweighting, or untested group-specific thresholds.")})])

    def part_c():
        intro = mo.md("A launch policy favors one population. Its successor updates each population's estimated benefit using only labels the current policy observes. Follow three finite rounds: missing labels stay missing. Independent follow-up buys information; serving exploration spends a ranked slot. Neither is a universal repair.")
        controls = mo.vstack([feedback_audit, feedback_rounds])
        if c_prediction.value is None:
            return mo.vstack([intro, controls, c_prediction])
        rows = []
        for label, result in (("Routine only", feedback_baseline), ("Selected", feedback_selected)):
            for interval in result["history"]:
                for group, observation in interval["groups"].items():
                    rows.append({"Policy": label, "Round": interval["round"], "Population": group, "Estimate before": fmt_percent(observation["estimate_before"], style="symbol"), "Exposed": observation["exposed"], "Observed": observation["observed"], "Missing": observation["unobserved"], "Estimate after": fmt_percent(observation["estimate_after"], style="symbol")})
        return mo.vstack([intro, controls, c_prediction, table(rows),
            mo.callout(mo.md(f"Selected replay: **{feedback_selected['total_followups']} additional follow-ups**, **{feedback_selected['total_displaced_slots']} displaced ranked slots**, and **{feedback_selected['total_benefits']} realized helpful exposures**. The final exposure-count gap is **{feedback_selected['final_exposure_gap']}**. Report the observation cost separately from the outcome."), kind="info"),
            c_capture, saved("C"), mo.md("**Explain:** trace one missing label through the next policy. Compare the first and last supplied round: when can information alter the choice, and when has it not done so yet? Then compare auditing with exploration. Name an application where independent outcomes cannot be observed without making the decision."),
            mo.accordion({"Replay rule and limits": mo.md(f"Each round supplies {feedback_selected['opportunities_per_population_per_round']} new opportunities per population. Historical observed-label seeds: " + "; ".join(f"{group}: {seed['successes']} successes in {seed['count']} labels" for group, seed in feedback_selected["seed_labels"].items()) + ". " + feedback_selected["update_rule"] + " Audits use declared request IDs. Exploration swaps a declared ranked slot for a declared hidden slot. " + feedback_selected["limitation"]),
                          "Inspect request-level replay": table([row for interval in feedback_selected["history"] for row in interval["records"]]),
                          "Optional private-summary reference": mo.vstack([mo.hstack([c_epsilon, c_releases], widths="equal", wrap=True), table([{"Policy": "Baseline", "Noise scale": c_base["laplace_scale"], "Composed epsilon": c_base["basic_composed_epsilon"]}, {"Policy": "Selected", "Noise scale": c_result["laplace_scale"], "Composed epsilon": c_result["basic_composed_epsilon"]}]), mo.md("This separate bounded-mean mechanism does not identify unobserved recommendation outcomes.")])})])

    def part_d():
        setup = mo.md("**Systems architect:** the release must handle the reference request volume, retain the supplied retraining schedule, and explain the required share of decisions over the same operating period. A candidate assumes lower inference energy but requires preparation work. First compare equal service; then try dropping an obligation or withholding replay evidence. A smaller total is not enough to justify a savings claim.")
        obligation = mo.md(f"**Fixed reference obligation:** {profile['daily_inferences']:,} requests per day over {profile['years']:g} years, {d_base['inputs']['retraining_runs']} retraining runs, and {d_base['inputs']['explanation_share_pct']:g}% of decisions explained. Both executions use the same illustrative grid and accounting exclusions.")
        controls = mo.vstack([d_execution, mo.hstack([d_demand, d_retraining, d_explanations], widths="equal", wrap=True)])
        if d_prediction.value is None:
            return mo.vstack([setup, obligation, controls, d_prediction])
        figure = go.Figure()
        colors = (COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["RedLine"], COLORS["Cloud"])
        for term, color in zip(("initial_training", "retraining", "inference", "explanation", "preparation"), colors):
            figure.add_bar(name=term.replace("_", " ").title(), x=["Reference", "Selected"], y=[d_base["terms_kwh"][term], d_result["terms_kwh"][term]], marker_color=color)
        figure.update_layout(barmode="stack", height=300, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Lifecycle energy (kWh)", legend_orientation="h")
        rows = [{
            "Case": label, "Total energy": f"{result['total_energy_kwh']:.2f} kWh",
            "Operational carbon": f"{result['operational_carbon_kg']:.2f} kg CO₂e",
            "Dominant term": result["dominant_term"].replace("_", " "),
        } for label, result in (("Reference", d_base), ("Selected", d_result))]
        checks = [{"Service obligation": name.replace("_", " "), "Matched": "yes" if passed else "no"}
                  for name, passed in d_result["service_checks"].items()]
        replay = ("No supplied replay: quality equivalence is unknown." if d_result["replay_mismatch_count"] is None
                  else f"Finite replay: {d_result['replay_count']} requests, {d_result['replay_mismatch_count']} changed decisions.")
        if d_result["comparison_gate"] == "ineligible":
            verdict = "**Comparison ineligible.** " + ", ".join(name.replace("_", " ") for name in d_result["comparison_failures"]) + ". A lower total cannot establish equal-service savings."
            kind = "danger"
        elif d_result["comparison_gate"] == "insufficient_quality_evidence":
            verdict = "**Insufficient quality evidence.** The accounting can be computed, but the equal-service savings claim remains unsupported."
            kind = "warn"
        elif d_result["supported_energy_saving"]:
            verdict = "**Conditional saving on supplied evidence.** The obligations match, the finite replay matches, and total energy falls after preparation. This does not certify deployment quality."
            kind = "success"
        else:
            verdict = "**Comparable on supplied evidence; no total-energy saving.** Lower inference energy can be outweighed by preparation work."
            kind = "info"
        return mo.vstack([
            setup, obligation, controls, d_prediction, apply_plotly_theme(figure), table(rows), table(checks),
            mo.md(replay), mo.callout(mo.md(verdict), kind=kind),
            mo.md(f"**Your prediction:** {d_prediction.value}. The selected case changes total energy by **{d_result['energy_delta_kwh']:+.2f} kWh** relative to the fixed reference. Explain which work moved and whether this is an eligible comparison."),
            d_capture, saved("D"),
            mo.accordion({"Inspect supplied request replay": table(d_result["replay_rows"]) if d_result["replay_rows"] else mo.md(replay),
                          "Calculation Notes": mo.md(d_result["boundary"] + " " + d_result["limitation"] + f" The candidate inference-energy factor is {d_result['inference_energy_factor']}; its preparation energy is {d_result['preparation_energy_factor']} times initial training energy. Energy sums the displayed terms; operational carbon multiplies energy by the same illustrative grid intensity.")}),
        ])

    def part_e():
        setup = mo.md("**Incident commander:** remove retained evidence and vary response delay. Separate what can be reconstructed from how many decisions occur before containment.")
        if e_prediction.value is None:
            return mo.vstack([setup, mo.hstack([e_policy, e_delay], widths="equal", wrap=True), e_prediction])
        figure = go.Figure([
            go.Bar(name="Reconstructable incident decisions", x=["Full / fast", "Selected"], y=[e_base["fully_reconstructable_decisions"], e_result["fully_reconstructable_decisions"]], marker_color=COLORS["BlueLine"]),
            go.Bar(name="Decisions before containment", x=["Full / fast", "Selected"], y=[e_base["decisions_before_containment"], e_result["decisions_before_containment"]], marker_color=COLORS["OrangeLine"]),
        ])
        figure.update_layout(barmode="group", height=280, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Decisions", legend_orientation="h")
        rows = [{
            "Question": name.replace("_", " ").title(),
            "Full / fast": "ANSWERABLE" if e_base["answerable_questions"][name] else "MISSING EVIDENCE",
            "Selected": "ANSWERABLE" if e_result["answerable_questions"][name] else "MISSING EVIDENCE",
        } for name in e_result["answerable_questions"]]
        kind = "success" if e_result["fully_reconstructable_decisions"] else "danger"
        return mo.vstack([
            setup, mo.hstack([e_policy, e_delay], widths="equal", wrap=True), e_prediction,
            apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {e_prediction.value}. The selected record answers **{e_result['answerable_question_count']} of 3** audit questions and leaves **{e_result['decisions_before_containment']:.0f} decisions** before containment. Removing lineage does not alter the saved fairness outcomes."), kind=kind),
            e_capture, saved("E"),
            mo.accordion({"Calculation Notes": mo.md("Decision reproduction requires model version, threshold, and feature hash; data tracing requires model and training-data versions; release identification requires model and deployment IDs. Exposure equals decision rate × containment delay. Evidence retention changes storage and reconstructability, not predictions or formal epsilon.")}),
        ])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            state = "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING")
            rows.append({"Part": part, "Original prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": state})
        b_decision = _captures["B"].to_dict()["decision"] if "B" in _captures else None
        b_saved_rejected = _captures["B"].to_dict()["inputs"]["rejected"] if "B" in _captures else None
        choice_is_distinct = final_choice.value == "hold" or final_choice.value != final_rejected.value
        complete = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and choice_is_distinct and final_choice.value == b_decision and final_rejected.value == b_saved_rejected
        return mo.vstack([
            mo.md("### Synthesis · Defend a responsible release decision (5 min)\nUse saved evidence to defend the intended benefit and proxy, declared criterion, chosen threshold or hold, selected-label feedback response, rejected alternative, remaining limitation, and reevaluation trigger."),
            table(rows),
            mo.callout(mo.md("Saved snapshots remain fixed while live controls move. Recapture stale evidence before generating the report."), kind="info"),
            mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
            mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale,
            mo.callout(mo.md("**Ready for the local report.**" if complete else "Complete five current contrasts, match the recommendation to the saved Part B decision, compare a different tested threshold when releasing, and add the rationale."), kind="success" if complete else "warn"),
        ])

    tabs = mo.ui.tabs({"Part A": notebook_learning_part(learning_path, "A", part_a()), "Part B": notebook_learning_part(learning_path, "B", part_b()), "Part C": notebook_learning_part(learning_path, "C", part_c()), "Part D": notebook_learning_part(learning_path, "D", part_d()), "Part E": notebook_learning_part(learning_path, "E", part_e()), "Synthesis": build_synthesis()})
    notebook_shell([notebook_tab_hint(), tabs], volume=1)
    return (audit,)


@app.cell(hide_code=True)
def _():
    from mlsysbook_labs.learning_paths import get_learning_path as _submission_path
    from mlsysbook_labs.notebook import notebook_shell as _submission_shell, notebook_submission as _submission
    _submission_shell([_submission(_submission_path(1, 15))])
    return


@app.cell
def _(learning_path, notebook_shell,
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures = get_evidence()
    _b_decision = _captures["B"].to_dict()["decision"] if "B" in _captures else None
    _b_saved_rejected = _captures["B"].to_dict()["inputs"]["rejected"] if "B" in _captures else None
    _choice_is_distinct = final_choice.value == "hold" or final_choice.value != final_rejected.value
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and _choice_is_distinct and final_choice.value == _b_decision and final_rejected.value == _b_saved_rejected
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not _ready, _download_readiness(volume=1))
    _snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_15_responsible_engr.py"), track=track_id,
        scenario="Affected-population recommendation reference; separate lifecycle/incident study: " + profile["scenario"],
        learning_objectives=learning_path["learning_objectives"],
        predictions={part: _snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: _snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": _snapshots[part]["baseline"], "result": _snapshots[part]["result"], "chosen_result": _snapshots[part]["chosen_result"], "result_role": _snapshots[part]["result_role"], "alternatives": _snapshots[part]["alternatives"]} for part in "ABCDE"},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["A proxy can improve while independently audited intended benefit falls.", "Thresholds change measured error consequences under the same declared obligation.", "Routine selected feedback does not establish complete population outcomes; independent observation has a cost."],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value}, residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": _snapshots, "recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": "Illustrative fixed teaching fixtures, not empirical deployment measurements.", "calculations": "All quantitative outcomes are produced by the Chapter 15 simulator experiments."},
    )
    notebook_shell([report_export_panel(report)], volume=1)
    return (report,)




@app.cell
def _(notebook_shell, audit, final_choice, final_rejected, final_risk, final_trigger, get_evidence, ledger, mo, rationale, track_id):
    _captures = get_evidence()
    _b_decision = _captures["B"].to_dict()["decision"] if "B" in _captures else None
    _b_saved_rejected = _captures["B"].to_dict()["inputs"]["rejected"] if "B" in _captures else None
    _choice_is_distinct = final_choice.value == "hold" or final_choice.value != final_rejected.value
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and _choice_is_distinct and final_choice.value == _b_decision and final_rejected.value == _b_saved_rejected
    save_discussion = mo.ui.run_button(label="Save discussion", disabled=not _ready)
    notebook_shell([save_discussion, mo.md("When ready, downloading is separate from saving locally. Editing your evidence or discussion requires saving again.")], volume=1)
    return (save_discussion,)


@app.cell
async def _(notebook_shell, notebook_status, save_discussion, audit, final_choice, final_rejected, final_risk, final_trigger, get_evidence, ledger, mo, rationale, track_id):
    _captures = get_evidence()
    _b_decision = _captures["B"].to_dict()["decision"] if "B" in _captures else None
    _b_saved_rejected = _captures["B"].to_dict()["inputs"]["rejected"] if "B" in _captures else None
    _choice_is_distinct = final_choice.value == "hold" or final_choice.value != final_rejected.value
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and _choice_is_distinct and final_choice.value == _b_decision and final_rejected.value == _b_saved_rejected
    _status = ("Ready to save" if _ready else "Evidence in progress")
    _status_class = "hud-active"
    if _ready and save_discussion.value:
        try:
            ledger.save(chapter=15, design={
                "schema_version": 1, "lab_id": "v1_15", "track_id": track_id,
                "model_id": "v1_15_experiments", "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "recommendation": final_choice.value, "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value, "residual_risk": final_risk.value,
                "rationale": rationale.value,
            })
            await ledger.flush()
        except Exception:
            _status = "LOCAL SAVE FAILED · DOWNLOAD THE REPORT TO KEEP YOUR EVIDENCE"
            _status_class = "hud-failed"
        else:
            _status = "SAVED"
    notebook_shell([notebook_status(_status, "Editing requires saving again; the downloaded discussion is separate from local storage.")], volume=1)
    return


if __name__ == "__main__":
    app.run()
