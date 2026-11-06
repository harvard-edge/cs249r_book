import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 13: The Price of Privacy · MLSysBook")


# ZONE A · OPENING AND BOOTSTRAP
@app.cell
async def _():
    import sys
    from dataclasses import asdict
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
    from mlsysim.labs.experiments.v2_13_experiments import (
        PrivacyEvent, compose_privacy_events, evaluate_retention,
        evaluate_collection_budget, evaluate_gradient_integrity, evaluate_clean_label_poisoning,
        get_collection_assumptions, get_integrity_round,
        evaluate_supplied_quality, evaluate_threat_coverage,
        evaluate_trust_latency, get_deletion_artifacts,
        get_privacy_epsilon_options, get_quality_observation,
        get_threat_control, get_track_scenario, get_trust_mechanism,
        propagate_deletion, release_bounded_mean,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata,
        report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import capture_evidence, audit_evidence

    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, PrivacyEvent, apply_plotly_theme,
        asdict, audit_evidence, build_lab_report, capture_evidence,
        compose_privacy_events, evaluate_retention, evaluate_supplied_quality,
        evaluate_collection_budget, evaluate_gradient_integrity, evaluate_clean_label_poisoning,
        get_collection_assumptions, get_integrity_round,
        evaluate_threat_coverage, evaluate_trust_latency,
        get_deletion_artifacts, get_lab_metadata, get_privacy_epsilon_options,
        get_quality_observation, get_threat_control, get_track_scenario,
        get_trust_mechanism, go, ledger, mo, propagate_deletion,
        release_bounded_mean, report_export_panel,
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
        value=next(label for label, context in _track_options.items() if context == _preferred_context(ledger)), label="Deployment context",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(get_privacy_epsilon_options, get_track_scenario, track):
    track_id = track.value
    scenario = get_track_scenario(track_id)
    epsilon_options = get_privacy_epsilon_options(track_id)
    return epsilon_options, scenario, track_id


# ZONE B · WIDGETS AND EXPERIMENTS
@app.cell
def _(epsilon_options, mo, track_id):
    _track_key = track_id
    mechanism_choices = {
        "Encrypted transport": "transport",
        "Secure aggregation": "secure_aggregation",
        "Confidential runtime": "confidential_runtime",
        "Locked device": "locked_device",
    }
    a_control = mo.ui.dropdown(mechanism_choices, value="Confidential runtime", label="Defense architecture")
    a_poison_budget = mo.ui.slider(0, 2, value=1, step=1, label="Correctly labeled records the attacker may append")
    a_poison_goal = mo.ui.dropdown({"Raise error on one near-boundary target": "targeted", "Raise average broad-cohort squared error": "broad"}, value="Raise error on one near-boundary target", label="Attacker outer objective")
    a_rule = mo.ui.dropdown({"Arithmetic mean": "mean", "Trim one update from each tail": "trimmed_mean"}, value="Trim one update from each tail", label="Backend gradient aggregation")
    a_attack = mo.ui.dropdown({"Opposing update −5": -5.0, "Opposing update −50": -50.0, "Opposing update −500": -500.0}, value="Opposing update −50", label="Compromised worker contribution")
    a_faults = mo.ui.slider(1, 2, value=1, step=1, label="Compromised workers in five-worker round")
    c_days = mo.ui.slider(1, 30, value=30, step=1, label="Collection days")
    c_principals = mo.ui.slider(1, 4, value=1, step=1, label="Independent admitted principals")
    c_lifetime = mo.ui.dropdown({"Daily quota only": "daily", "Daily quota plus lifetime cap": "lifetime"}, value="Daily quota plus lifetime cap", label="Collection control")
    b_epsilon = mo.ui.dropdown(
        {f"ε = {value:g}": value for value in epsilon_options},
        value=f"ε = {epsilon_options[1]:g}", label="Demonstration epsilon",
    )
    c_release_count = mo.ui.slider(2, 4, value=3, step=1, label="Private-data releases")
    c_operation = mo.ui.dropdown(
        {"Queries to fixed released model": "fixed_queries", "New unaccounted fine-tuning access": "new_private_access"},
        value="Queries to fixed released model", label="Follow-on operation",
    )
    c_query_count = mo.ui.slider(1, 10, value=5, step=1, label="Fixed-model queries")
    d_mechanism = mo.ui.dropdown(mechanism_choices, value="Confidential runtime", label="Trust mechanism")
    d_requests = mo.ui.slider(1, 32, value=8, step=1, label="Requests per protected session")
    e_mode = mo.ui.dropdown(
        {"Complete lineage": "complete", "Slow archive": "slow", "Untracked export": "untracked"},
        value="Untracked export", label="Artifact-copy path",
    )
    e_retention = mo.ui.dropdown(
        {"12 hours": 12.0, "24 hours": 24.0, "72 hours": 72.0, "168 hours": 168.0},
        value="24 hours", label="Incident-evidence retention",
    )
    return a_poison_budget, a_poison_goal, a_attack, a_faults, a_rule, c_days, c_lifetime, c_principals, a_control, b_epsilon, c_operation, c_query_count, c_release_count, d_mechanism, d_requests, e_mode, e_retention


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Covered": "covered", "Not covered": "not_covered"},
        label="Does the selected architecture cover the stated adversary?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Noise scale rises; supplied utility may fall": "stronger_cost", "Noise scale falls; supplied utility rises": "reversed", "Guarantee and utility are the same quantity": "same"},
        label="What changes when epsilon is reduced from the baseline?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Every fixed-model query adds epsilon": "queries_add", "Only new DP releases add accounted epsilon": "releases_add", "A new private access automatically has epsilon zero": "access_zero"},
        label="Which repeated operations consume the demonstrated privacy budget?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Protection and deadline both pass": "both", "Protection fails": "protection", "Deadline fails": "deadline", "Both fail": "both_fail"},
        label="What fails for the selected trust mechanism?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Deletion is provable and evidence remains": "both", "Deletion proof fails": "deletion", "Evidence expires": "evidence", "Both constraints fail": "both_fail"},
        label="What is the lifecycle consequence?",
    ).form(submit_button_label="Lock Part E prediction")
    a_poison_prediction = mo.ui.radio({"Correct labels and aggregate accuracy rule out a targeted failure": "audit_sufficient", "Correct labels and unchanged aggregate accuracy can coexist with a target failure": "target_hidden"}, label="What can a clean-label training-data writer hide?").form(submit_button_label="Lock poisoning prediction")
    a_integrity_prediction = mo.ui.radio({"An arbitrarily large opposing contribution can reverse the mean step": "mean_unbounded", "Averaging bounds every contribution": "mean_bounded", "Trimming certifies security for any worker count": "trim_universal"}, label="What can one malicious training worker do?").form(submit_button_label="Lock gradient prediction")
    c_collection_prediction = mo.ui.radio({"Daily quotas cap lifetime collection": "daily_caps", "Daily quotas delay collection; persistent counters bound each principal": "persistent", "A new account inherits the actor's counter automatically": "identity"}, label="What bounds cumulative response collection?").form(submit_button_label="Lock collection prediction")
    return a_poison_prediction, a_integrity_prediction, c_collection_prediction, a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Encrypted transport": "transport", "Secure aggregation": "secure_aggregation", "Confidential runtime": "confidential_runtime", "Locked device": "locked_device", "Defer release pending evidence": "defer"},
        label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {"Encrypted transport": "transport", "Secure aggregation": "secure_aggregation", "Confidential runtime": "confidential_runtime", "Locked device": "locked_device"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"Adversary changes": "adversary", "Deadline tightens": "deadline", "Another private release is proposed": "privacy", "A new artifact copy appears": "lineage"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Unmodeled adversary capability": "threat", "Utility evidence is illustrative": "utility", "Untracked artifact copy": "lineage", "Production randomness and accountant remain external": "accounting"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Decision rationale",
        placeholder="Use saved evidence and quantify why the rejected alternative loses.",
     debounce=False)
    return final_choice, final_rejected, final_risk, final_trigger, rationale


# ZONE C · ONE TABBED INVESTIGATION SURFACE
@app.cell
def _(
    PrivacyEvent, a_control, asdict, b_epsilon, c_operation, c_query_count,
    a_poison_budget, a_poison_goal, a_attack, a_faults, a_rule, c_days, c_lifetime, c_principals,
    evaluate_collection_budget, evaluate_gradient_integrity, evaluate_clean_label_poisoning, get_collection_assumptions, get_integrity_round,
    c_release_count, compose_privacy_events, d_mechanism, d_requests, e_mode,
    e_retention, epsilon_options, evaluate_retention, evaluate_supplied_quality,
    evaluate_threat_coverage, evaluate_trust_latency, get_deletion_artifacts,
    get_quality_observation, get_threat_control, get_trust_mechanism,
    propagate_deletion, release_bounded_mean, scenario, track_id,
):
    def plain(value):
        return asdict(value)

    def control_args(control):
        return {"adversary": scenario.adversary, "control": {"name": control.name, "covered_threats": sorted(control.covered_threats), "protected_boundary": control.protected_boundary}, "required_capabilities": []}

    a_base_control = get_threat_control("transport")
    a_selected_control = get_threat_control(a_control.value)
    a_base_inputs = control_args(a_base_control)
    a_result_inputs = control_args(a_selected_control)
    a_base = {"inputs": a_base_inputs, "output": plain(evaluate_threat_coverage(scenario.adversary, a_base_control))}
    a_result = {"inputs": a_result_inputs, "output": plain(evaluate_threat_coverage(scenario.adversary, a_selected_control))}

    gradient_common = {"honest_updates": list(get_integrity_round(a_faults.value)), "malicious_updates": [a_attack.value] * a_faults.value, "trim_each_tail": 1, "starting_parameter": 1.0, "learning_rate": 0.1}
    for pair, rule in ((a_base, "mean"), (a_result, a_rule.value)):
        pair["inputs"]["gradient_integrity"] = {**gradient_common, "rule": rule}
        pair["output"]["gradient_integrity"] = plain(evaluate_gradient_integrity(**gradient_common, rule=rule))

    for pair, budget in ((a_base, 0), (a_result, a_poison_budget.value)):
        args = {"poison_budget": budget, "attack_objective": a_poison_goal.value}
        pair["inputs"]["clean_label_poisoning"] = args
        pair["output"]["clean_label_poisoning"] = plain(evaluate_clean_label_poisoning(**args))

    baseline_epsilon = epsilon_options[-1]
    release_common = {"values": list(scenario.bounded_values), "lower_bound": scenario.lower_bound, "upper_bound": scenario.upper_bound, "release_id": f"{track_id}-mean-release", "seed": 13, "clip_output": True}
    b_base_release_args = {**release_common, "epsilon": baseline_epsilon}
    b_result_release_args = {**release_common, "epsilon": b_epsilon.value}
    b_base_observation = get_quality_observation(track_id, baseline_epsilon)
    b_result_observation = get_quality_observation(track_id, b_epsilon.value)
    b_base_inputs = {"release_bounded_mean": b_base_release_args, "evaluate_supplied_quality": {"observation": plain(b_base_observation)}}
    b_result_inputs = {"release_bounded_mean": b_result_release_args, "evaluate_supplied_quality": {"observation": plain(b_result_observation)}}
    b_base_release = release_bounded_mean(**b_base_release_args)
    b_result_release = release_bounded_mean(**b_result_release_args)
    b_base_quality = evaluate_supplied_quality(b_base_observation)
    b_result_quality = evaluate_supplied_quality(b_result_observation)
    b_base = {"inputs": b_base_inputs, "output": {"release": plain(b_base_release), "quality": plain(b_base_quality)}}
    b_result = {"inputs": b_result_inputs, "output": {"release": plain(b_result_release), "quality": plain(b_result_quality)}}

    c_base_events = [PrivacyEvent("release-1", "dp_release", epsilon=b_epsilon.value)]
    c_result_events = [PrivacyEvent(f"release-{index}", "dp_release", epsilon=b_epsilon.value) for index in range(1, c_release_count.value + 1)]
    if c_operation.value == "fixed_queries":
        c_result_events.extend(PrivacyEvent(f"query-{index}", "fixed_model_query", source_release_id="release-1") for index in range(1, c_query_count.value + 1))
    else:
        c_result_events.append(PrivacyEvent("new-fine-tune", "new_private_data_access"))
    c_base_inputs = {"events": [plain(event) for event in c_base_events]}
    c_result_inputs = {"events": [plain(event) for event in c_result_events]}
    c_base = {"inputs": c_base_inputs, "output": plain(compose_privacy_events(c_base_events))}
    c_result = {"inputs": c_result_inputs, "output": plain(compose_privacy_events(c_result_events))}

    collection_assumptions = get_collection_assumptions()
    collection_common = {**collection_assumptions, "days": c_days.value, "principal_count": c_principals.value}
    for pair, cap in ((c_base, None), (c_result, collection_assumptions["lifetime_limit_per_principal"] if c_lifetime.value == "lifetime" else None)):
        args = {**collection_common, "lifetime_limit_per_principal": cap}
        pair["inputs"]["collection_budget"] = args
        pair["output"]["collection_budget"] = plain(evaluate_collection_budget(**args))

    def mechanism_args(mechanism):
        return {"mechanism": {"name": mechanism.name, "protected_threats": sorted(mechanism.protected_threats), "fixed_overhead_ms": mechanism.fixed_overhead_ms, "per_request_overhead_ms": mechanism.per_request_overhead_ms}, "adversary": scenario.adversary, "base_latency_ms": scenario.base_latency_ms, "requests_per_session": d_requests.value, "deadline_ms": scenario.deadline_ms}

    d_base_mechanism = get_trust_mechanism("transport")
    d_selected_mechanism = get_trust_mechanism(d_mechanism.value)
    d_base_inputs = mechanism_args(d_base_mechanism)
    d_result_inputs = mechanism_args(d_selected_mechanism)
    d_base = {"inputs": d_base_inputs, "output": plain(evaluate_trust_latency(d_base_mechanism, adversary=scenario.adversary, base_latency_ms=scenario.base_latency_ms, requests_per_session=d_requests.value, deadline_ms=scenario.deadline_ms))}
    d_result = {"inputs": d_result_inputs, "output": plain(evaluate_trust_latency(d_selected_mechanism, adversary=scenario.adversary, base_latency_ms=scenario.base_latency_ms, requests_per_session=d_requests.value, deadline_ms=scenario.deadline_ms))}
    d_alternatives = []
    for mechanism_id in ("transport", "secure_aggregation", "confidential_runtime", "locked_device"):
        mechanism = get_trust_mechanism(mechanism_id)
        arguments = mechanism_args(mechanism)
        evaluated = evaluate_trust_latency(mechanism, adversary=scenario.adversary, base_latency_ms=scenario.base_latency_ms, requests_per_session=d_requests.value, deadline_ms=scenario.deadline_ms)
        d_alternatives.append({"mechanism_id": mechanism_id, "inputs": arguments, "output": plain(evaluated)})

    def lifecycle(mode, retention_hours):
        artifacts = get_deletion_artifacts(track_id, mode)
        deletion_inputs = {"artifacts": [plain(artifact) for artifact in artifacts], "source_artifact_id": artifacts[0].artifact_id, "deletion_sla_hours": scenario.deletion_sla_hours}
        retention_inputs = {"evidence": [plain(event) for event in scenario.incident_evidence], "retention_hours": retention_hours}
        deletion = propagate_deletion(artifacts, source_artifact_id=artifacts[0].artifact_id, deletion_sla_hours=scenario.deletion_sla_hours)
        retention = evaluate_retention(scenario.incident_evidence, retention_hours=retention_hours)
        return {"inputs": {"propagate_deletion": deletion_inputs, "evaluate_retention": retention_inputs}, "output": {"deletion": plain(deletion), "retention": plain(retention)}}

    e_base = lifecycle("complete", 168.0)
    e_result = lifecycle(e_mode.value, e_retention.value)
    return a_base, a_result, b_base, b_result, c_base, c_result, d_alternatives, d_base, d_result, e_base, e_result


# ZONE D · COMPLETION-GATED REPORT AND LEDGER
@app.cell
def _(
    a_base, a_control, a_prediction, a_result, b_base, b_epsilon,
    a_poison_prediction, a_integrity_prediction, c_collection_prediction,
    b_prediction, b_result, c_base, c_operation, c_prediction, c_result,
    capture_evidence, d_alternatives, d_base, d_mechanism, d_prediction, d_requests, d_result,
    e_base, e_mode, e_prediction, e_result, e_retention, mo, set_evidence,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream, b_upstream, d_upstream, e_upstream = {}, {}, {}, {}
    c_upstream = {"epsilon": b_epsilon.value}
    a_capture = mo.ui.button(
        label="Capture boundary comparison", kind="success",
        disabled=a_prediction.value is None or a_integrity_prediction.value is None or a_poison_prediction.value is None or a_result["inputs"]["clean_label_poisoning"]["poison_budget"] == 0 or a_base["inputs"]["gradient_integrity"] == a_result["inputs"]["gradient_integrity"],
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"selected_control": a_control.value, "integrity_prediction": a_integrity_prediction.value, "poisoning_prediction": a_poison_prediction.value}, baseline=a_base,
            result=a_result, upstream_inputs=a_upstream,
            decision=a_control.value, model_key="v2_13_experiments.threat_coverage",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture privacy and utility contrast", kind="success",
        disabled=b_prediction.value is None or b_base["inputs"] == b_result["inputs"],
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"selected_epsilon": b_epsilon.value}, baseline=b_base,
            result=b_result, upstream_inputs=b_upstream,
            alternatives=(b_base, b_result), decision=b_epsilon.value,
            model_key="v2_13_experiments.bounded_mean",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture composition history", kind="success",
        disabled=c_prediction.value is None or c_collection_prediction.value is None or c_base["inputs"]["collection_budget"] == c_result["inputs"]["collection_budget"],
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"operation": c_operation.value, "collection_prediction": c_collection_prediction.value}, baseline=c_base,
            result=c_result, upstream_inputs=c_upstream,
            decision=c_operation.value,
            model_key="v2_13_experiments.privacy_composition",
        )),
    )
    d_capture = mo.ui.button(
        label="Capture trust deadline comparison", kind="success",
        disabled=d_prediction.value is None or d_mechanism.value == "transport",
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={"mechanism": d_mechanism.value, "requests_per_session": d_requests.value},
            baseline=d_base, result=d_result, upstream_inputs=d_upstream,
            alternatives=tuple(d_alternatives), decision=d_mechanism.value,
            model_key="v2_13_experiments.trust_latency",
        )),
    )
    e_capture = mo.ui.button(
        label="Capture lifecycle comparison", kind="success",
        disabled=e_prediction.value is None or e_base["inputs"] == e_result["inputs"],
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"copy_mode": e_mode.value, "retention_hours": e_retention.value},
            baseline=e_base, result=e_result, upstream_inputs=e_upstream,
            decision=e_mode.value, model_key="v2_13_experiments.lifecycle",
        )),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream, e_capture, e_upstream


@app.cell
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




    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}

    </style>""")
    roles_html = f"<div><b>Fleet roles</b><br>{' → '.join(scenario.fleet_roles)}</div>" if getattr(scenario, "fleet_roles", None) else ""
    locus_html = f"<div><b>Attack locus</b><br>{scenario.adversary_locus}</div>" if getattr(scenario, "adversary_locus", None) else ""
    header = _course_header(_course_metadata("vol2/lab_13_security_privacy.py"), 'Which protection can this fleet defend when threat coverage, privacy guarantee, utility, latency, and lifecycle evidence are judged separately?')
    _course_shell([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        css,
        header,
        track,
        _course_scenario(_course_path('vol2', 13), track=track.value, volume=2),
        mo.Html('<p class="pilot-note">All workloads, outcome counts, overheads, and copy paths are illustrative scenarios. Seeded Laplace draws make the exercise replayable; a deterministic public seed is predictable and is not production release randomness. This lab does not implement a DP-SGD accountant.</p>'),
        _course_learning_map(_course_path('vol2', 13))
    ], gap=0.5, volume=2)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base, a_capture, a_control, a_prediction, a_result, a_upstream,
    a_poison_prediction, a_poison_goal, a_poison_budget, a_attack, a_faults, a_rule, a_integrity_prediction, c_collection_prediction, c_days, c_lifetime, c_principals,
    apply_plotly_theme, audit_evidence, b_base, b_capture, b_epsilon,
    b_prediction, b_result, b_upstream, c_base, c_capture, c_operation,
    c_prediction, c_query_count, c_release_count, c_result, c_upstream,
    d_alternatives, d_base, d_capture, d_mechanism, d_prediction, d_requests,
    d_result, d_upstream, e_base, e_capture, e_mode, e_prediction, e_result, e_retention,
    e_upstream, final_choice, final_rejected, final_risk, final_trigger,
    get_collection_assumptions, get_evidence, go, mo, rationale, scenario, track_id,
):
    _saved_prediction_widgets = {'A': a_prediction, 'B': b_prediction, 'C': c_prediction, 'D': d_prediction, 'E': e_prediction}
    from mlsysbook_labs.notebook import notebook_prediction_label as _read_prediction
    from mlsysbook_labs.notebook import notebook_tab_hint as _part_tabs_hint, notebook_shell as _lesson_shell
    from mlsysbook_labs.notebook import notebook_part_reflection as _concept_reflection
    from mlsysbook_labs.learning_paths import get_learning_path as _reflection_path
    _reflection_contract = _reflection_path('vol2', 13)
    _captures = get_evidence()
    upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=upstream,
        contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture after reviewing the changed dependency."), kind="danger")
        data = capture.to_dict()
        extra_predictions = ""
        if part == "A":
            extra_predictions = f"<br>Gradient prediction: {_read_prediction(a_integrity_prediction, value=data['inputs'].get('integrity_prediction'))}<br>Poisoning prediction: {_read_prediction(a_poison_prediction, value=data['inputs'].get('poisoning_prediction'))}"
        elif part == "C":
            extra_predictions = f"<br>Collection prediction: {_read_prediction(c_collection_prediction, value=data['inputs'].get('collection_prediction'))}"
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {_read_prediction(_saved_prediction_widgets[part], value=data["prediction"])}{extra_predictions}<br><small>Track {data["track"]}; later control changes do not rewrite this record.</small></div>')

    def build_part_a():
        locus_text = f" attacking at **{scenario.adversary_locus}**" if getattr(scenario, "adversary_locus", None) else ""
        roles_text = f"\n\n**Fleet topology:** {' → '.join(scenario.fleet_roles)}." if getattr(scenario, "fleet_roles", None) else ""
        intro = mo.md(f"### A · Which boundary does the defense protect? (11 min)\nThe stated adversary is **{scenario.adversary.replace('_', ' ')}**{locus_text}.{roles_text}\n\n*Where mechanisms run:* In distributed and edge fleets (including TinyML), mechanisms run at specific tiers. Locked device execution protects the device hardware/MCU; encrypted transport protects network links; secure aggregation and confidential runtimes protect aggregation gateways or backend servers. Compare encrypted transport with one selected architecture before deciding whether the relevant boundary is covered.")
        integrity_intro = mo.md("#### A second boundary: a worker may write a gradient\nAt the backend training tier, an admitted worker sends an opposing scalar update. Compare averaging with trimming one value from each tail in a five-worker round. Then increase the attack magnitude and add a second compromised worker. Device isolation and transport encryption do not establish that a training update is benign.")
        integrity_controls = mo.hstack([a_rule, a_attack, a_faults], wrap=True)
        poison_intro = mo.md("#### A third boundary: a writer can change what training learns\nA label audit accepts every candidate below: ground truth is 1 for x>0 and 0 otherwise. The attacker can append copies of one correctly labeled point. For each candidate, the defender actually refits a one-feature least-squares classifier; the attacker then picks the fit with the greatest chosen outer loss. Test one near-boundary target separately from a broad held-out cohort.")
        poison_controls = mo.hstack([a_poison_goal, a_poison_budget], wrap=True)
        poison = a_result["output"]["clean_label_poisoning"]
        poison_rows = []
        for label, result in (("No added record", poison["baseline"]), ("Selected clean-label candidate", poison["selected"])):
            poison_rows.append({"Case": label, "Target raw score": f"{result['target_score']:.3g}", "Target squared error": f"{result['target_squared_error']:.3g}", "Target fails": "YES" if result["target_misclassified"] else "NO", "Broad correct": f"{result['broad_correct']}/{result['broad_evaluated']}"})
        poison_panel = mo.vstack([poison_intro, poison_controls, a_poison_prediction, table(poison_rows), mo.callout(mo.md(f"The outer search selected **x={poison['selected']['candidate_feature']}, true label={poison['selected']['candidate_label']}** from **{poison['candidate_features']}**, at dose **{poison['poison_budget']}**. The target is **x={poison['target_feature']}**, true label **{poison['target_true_label']}**. Broad accuracy and a clean label audit do not test this target. These are exact outcomes on constructed data, not measured attack effectiveness on your fleet model."), kind="warn"), mo.accordion({"Poisoning calculation notes": mo.md(f"Clean training features are {poison['training_features']} with labels {poison['training_labels']}; held-out broad features are {poison['broad_features']}. Inner optimization recomputes slope and intercept minimizing training squared error for every candidate. Outer optimization maximizes target squared error or broad mean squared error over the finite candidates. The defender classifies a raw score ≥0.5 as positive; raw least-squares scores are not probabilities. Poisoning write access, candidate feature range, dose, learner and objective are explicit assumptions. This demonstrates the bilevel mechanism, not an optimal continuous attack or a poisoning defense.")})]) if a_poison_prediction.value is not None else mo.vstack([poison_intro, poison_controls, a_poison_prediction])
        gradient_rows = []
        for label, pair in (("Mean baseline", a_base), ("Selected aggregation", a_result)):
            result = pair["output"]["gradient_integrity"]
            gradient_rows.append({"Case": label, "Aggregate": f"{result['aggregate']:.3g}", "Honest mean": f"{result['honest_mean']:.3g}", "Loss before": f"{result['loss_before']:.3g}", "Loss after": f"{result['loss_after']:.3g}", "Descent": "YES" if result["improves_objective"] else "NO"})
        gradient = a_result["output"]["gradient_integrity"]
        integrity_panel = mo.vstack([integrity_intro, integrity_controls, a_integrity_prediction, table(gradient_rows), mo.callout(mo.md("**Fault assumption:** " + ("Within the one-compromised-worker budget for this finite test." if gradient["within_fault_budget"] else "**EXCEEDED.** Two compromised workers defeat the supplied one-tail trim in this round; the selected update increases the objective.")), kind="info" if gradient["within_fault_budget"] else "danger"), mo.callout(mo.md(f"The actual submitted updates are **{gradient['worker_updates']}**. Gathering individual gradients receives **{gradient['gather_per_worker_payloads']:g} payloads per worker**, versus **{gradient['ring_per_worker_payloads']:g}** for ring AllReduce: **{gradient['gather_to_ring_ratio']:g}×** traffic for this gather implementation. Trimming is not a security certificate; try two compromised workers against a one-fault trim rule."), kind="info"), mo.accordion({"Gradient calculation notes": mo.md("One finite scalar round uses L(w)=w²/2 and w′=w−0.1g. Honest updates are constructed noisy gradients near w=1. Mean influence is unbounded as an unconstrained malicious value grows. Trimmed mean sorts and removes one value at each tail; it can fail beyond its declared fault budget and under adaptive or heterogeneous updates. Gather traffic (n−1)M is compared with ring traffic 2(n−1)M/n; this is not a lower bound for all robust protocols. This gradient round does not simulate poisoned training examples or provide a convergence guarantee.")})]) if a_integrity_prediction.value is not None else mo.vstack([integrity_intro, integrity_controls, a_integrity_prediction])
        rows = [
            {"Architecture": a_base["output"]["control_name"], "Execution tier": a_base["output"].get("execution_tier", "Transit"), "Boundary": a_base["output"]["protected_boundary"], "Adversary covered": "YES" if a_base["output"]["covered"] else "NO"},
            {"Architecture": a_result["output"]["control_name"], "Execution tier": a_result["output"].get("execution_tier", "Specified tier"), "Boundary": a_result["output"]["protected_boundary"], "Adversary covered": "YES" if a_result["output"]["covered"] else "NO"},
        ]
        coverage_panel = mo.vstack([table(rows), mo.callout(mo.md(f"**Your prediction:** {_read_prediction(a_prediction)}. **Observed categorical result:** {a_result['output']['control_name']} ({a_result['output'].get('execution_tier', '')}) {'covers' if a_result['output']['covered'] else 'does not cover'} the stated adversary at the **{a_result['output']['protected_boundary']}** boundary."), kind="success" if a_result["output"]["covered"] else "danger")]) if a_prediction.value is not None else mo.Html("")
        return mo.vstack([mo.vstack([intro, a_control, a_prediction, coverage_panel, integrity_panel, poison_panel, a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("Coverage is set membership against an explicit adversary capability. Mechanisms running at gateway or backend tiers (such as secure aggregation or confidential VMs) cannot protect an MCU against an attacker with physical device possession. Latency cannot improve or weaken this categorical result.")})]), _concept_reflection(_reflection_contract, "A")])

    def build_part_b():
        units = getattr(scenario, "metric_units", "") or "units"
        intro = mo.md(f"### B · What utility survives stronger privacy? (10 min)\nCompare two releases of the same bounded mean in **{units}** under fixed-size replace-one adjacency. Quality counts are supplied illustrative outcomes for a matched task; they are not calculated from epsilon, nor do they claim legal compliance.")
        if b_prediction.value is None:
            return mo.vstack([intro, b_epsilon, b_prediction])
        base_release = b_base["output"]["release"]
        result_release = b_result["output"]["release"]
        base_quality = b_base["output"]["quality"]
        result_quality = b_result["output"]["quality"]
        fig = go.Figure([go.Bar(name="Laplace scale", x=[f"ε={base_release['epsilon']:g}", f"ε={result_release['epsilon']:g}"], y=[base_release["laplace_scale"], result_release["laplace_scale"]], marker_color=COLORS["BlueLine"])])
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=25, b=20), yaxis_title=f"Laplace scale ({units})", showlegend=False)
        rows = [
            {"Case": "Baseline", "Task": base_quality["task"], "Epsilon": base_release["epsilon"], "Sensitivity": f"{base_release['sensitivity']:.4g} {units}", "Noise scale": f"{base_release['laplace_scale']:.4g} {units}", "Supplied correct": f"{base_quality['correct']}/{base_quality['evaluated']}"},
            {"Case": "Selected", "Task": result_quality["task"], "Epsilon": result_release["epsilon"], "Sensitivity": f"{result_release['sensitivity']:.4g} {units}", "Noise scale": f"{result_release['laplace_scale']:.4g} {units}", "Supplied correct": f"{result_quality['correct']}/{result_quality['evaluated']}"},
        ]
        return mo.vstack([mo.vstack([intro, b_epsilon, b_prediction, apply_plotly_theme(fig), table(rows), mo.callout(mo.md(f"**Prediction:** {_read_prediction(b_prediction)}. The guarantee calculation yields scale **{base_release['laplace_scale']:.4g} → {result_release['laplace_scale']:.4g} {units}**. The separate supplied evaluation for *{result_quality['task']}* yields **{base_quality['correct']}/{base_quality['evaluated']} → {result_quality['correct']}/{result_quality['evaluated']}** correct."), kind="info"), mo.callout(mo.md("The shown draw uses a deterministic public seed for replay. Predictable randomness is unsuitable for a production privacy-preserving release. No legal compliance is claimed."), kind="warn"), b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md(f"For public bounds [L, U] and n fixed records, replace-one mean sensitivity is (U − L)/n in {units}. The Laplace scale is sensitivity/epsilon. Output clipping is postprocessing and does not change epsilon. Quality counts are illustrative scenario fixtures, not empirical compliance benchmarks.")})]), _concept_reflection(_reflection_contract, "B")])

    def build_part_c():
        intro = mo.md(f"### C · Which repeated operations consume privacy budget? (11 min)\nIn the {scenario.label}, training privacy and retraining mechanisms run at the aggregation gateway or backend service, while client devices perform inference. Start with one explicit DP release (e.g., federated round or telemetry release), then compare multiple new releases with either fixed-model queries (client inference) or a new unaccounted private-data access (backend retraining).")
        operation_control = c_query_count if c_operation.value == "fixed_queries" else mo.md("The new fine-tuning access has no tested accountant in this exercise.")
        collection_intro = mo.md("#### Zero additional DP loss does not mean unlimited collection is safe\nA caller can keep collecting responses from a fixed model. The constructed collection objective is not an empirical extraction threshold. Compare a daily quota with a persistent per-principal cap, then let the same actor obtain more independent identities. Both controls also deny legitimate queries beyond their limits; query counts alone cannot tell which caller is malicious.")
        collection_controls = mo.hstack([c_lifetime, c_days, c_principals], wrap=True)
        collection_rows = []
        collection_fig = go.Figure()
        for label, pair in (("Daily quota only", c_base), ("Selected control", c_result)):
            result = pair["output"]["collection_budget"]
            collection_rows.append({"Case": label, "Collected": result["collected_queries"], "Denied": result["denied_queries"], "Query objective": result["target_queries"], "Reached": "YES" if result["target_collected"] else "NO", "Principals": result["principal_count"]})
            collection_fig.add_scatter(name=label, x=[day["day"] for day in result["history"]], y=[day["cumulative_queries"] for day in result["history"]], mode="lines+markers", line_color=COLORS["BlueLine"] if label == "Daily quota only" else COLORS["OrangeLine"])
        collection_fig.update_layout(height=250, xaxis_title="Collection day", yaxis_title="Cumulative admitted responses", legend_orientation="h", margin=dict(l=20,r=20,t=20,b=20))
        collection_panel = mo.vstack([collection_intro, collection_controls, c_collection_prediction, apply_plotly_theme(collection_fig), table(collection_rows), mo.accordion({"Collection calculation notes": mo.md(f"Each principal attempts {get_collection_assumptions()['queries_per_day_per_principal']} queries per day against a {get_collection_assumptions()['daily_limit_per_principal']}-query daily quota. The optional persistent cap is {get_collection_assumptions()['lifetime_limit_per_principal']} queries per principal; the constructed objective is collecting {get_collection_assumptions()['target_queries']:,} responses. Each daily admission is min(demand, daily limit, remaining lifetime counter). Counters are independent across principals and never reset across days. These are teaching assumptions, not measured extraction requirements. More response detail may expose more information, but surrogate fidelity, membership leakage, account linking, and monetary pricing require separate evidence.")})]) if c_collection_prediction.value is not None else mo.vstack([collection_intro, collection_controls, c_collection_prediction])
        result = c_result["output"]
        fig = go.Figure([go.Bar(x=["One release", "Follow-on history"], y=[c_base["output"]["total_epsilon"], result["total_epsilon"]], marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]])])
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Composed epsilon", showlegend=False)
        rows = [{"Accounted releases": result["accounted_release_count"], "Zero-additional-loss operations": result["zero_additional_training_loss_count"], "Unaccounted private accesses": result["unguaranteed_private_access_count"], "Guarantee complete": "YES" if result["guarantee_complete"] else "NO"}]
        privacy_feedback = mo.vstack([apply_plotly_theme(fig), table(rows), mo.callout(mo.md(f"**Prediction:** {_read_prediction(c_prediction)}. Basic sequential composition gives **ε = {result['total_epsilon']:.3g}**. Fixed released-model queries add zero training privacy loss; an unaccounted new private-data access makes the claimed guarantee incomplete."), kind="success" if result["guarantee_complete"] else "danger")]) if c_prediction.value is not None else mo.Html("")
        return mo.vstack([mo.vstack([intro, mo.hstack([c_operation, c_release_count], widths="equal", wrap=True), operation_control, c_prediction, privacy_feedback, collection_panel, c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md("Basic sequential composition sums epsilon across explicit DP releases. Postprocessing and queries of a fixed released DP model do not revisit private training data. This statement concerns training privacy loss; it does not dismiss model-extraction risk, nor does it constitute a legal privacy guarantee.")})]), _concept_reflection(_reflection_contract, "C")])

    def build_part_d():
        path_text = f" across **{scenario.path_description}**" if getattr(scenario, "path_description", None) else ""
        intro = mo.md(f"### D · Which trust mechanism meets the deadline? (9 min)\nThe fleet request path{path_text} starts at **{scenario.base_latency_ms:g} ms**. Every request, including the first request waiting for session setup, must finish by **{scenario.deadline_ms:g} ms**. Compare cold, steady, and session-average latency. A longer session spreads setup cost across more requests; it does not shorten the first request's wait. Protection scope and runtime deadline are independent checks.")
        if d_prediction.value is None:
            return mo.vstack([intro, d_mechanism, d_requests, d_prediction])
        rows = [{"Mechanism": alternative["output"]["mechanism_name"], "Execution tier": alternative["output"].get("execution_tier", "Fleet tier"), "Protected": "YES" if alternative["output"]["protected"] else "NO", "Cold request": f"{alternative['output']['cold_latency_ms']:.2f} ms", "Steady request": f"{alternative['output']['steady_latency_ms']:.2f} ms", "Session average": f"{alternative['output']['average_latency_ms']:.2f} ms", "Cold deadline": "PASS" if alternative["output"]["meets_deadline"] else "FAIL"} for alternative in d_alternatives]
        fig = go.Figure()
        for name, field, color in (("Cold request", "cold_latency_ms", COLORS["OrangeLine"]), ("Steady request", "steady_latency_ms", COLORS["GreenLine"]), ("Session average", "average_latency_ms", COLORS["BlueLine"])):
            fig.add_bar(name=name, x=["Baseline", "Selected"], y=[d_base["output"][field], d_result["output"][field]], marker_color=color)
        fig.add_hline(y=scenario.deadline_ms, line_dash="dash", annotation_text="Deadline")
        fig.update_layout(height=290, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Latency (ms)", barmode="group", legend_orientation="h")
        return mo.vstack([mo.vstack([intro, mo.hstack([d_mechanism, d_requests], widths="equal", wrap=True), d_prediction, apply_plotly_theme(fig), table(rows), mo.callout(mo.md(f"**Prediction:** {_read_prediction(d_prediction)}. Selected mechanism protection: **{'PASS' if d_result['output']['protected'] else 'FAIL'}** ({d_result['output'].get('execution_tier', '')}). Cold-request deadline: **{'PASS' if d_result['output']['meets_deadline'] else 'FAIL'}** at **{d_result['output']['cold_latency_ms']:.2f} ms**. Steady requests take **{d_result['output']['steady_latency_ms']:.2f} ms**; session average is **{d_result['output']['average_latency_ms']:.2f} ms**. Setup is still pending when the first request arrives."), kind="success" if d_result["output"]["protected"] and d_result["output"]["meets_deadline"] else "danger"), d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md("Cold latency equals base plus per-request overhead plus all fixed session setup. Steady latency excludes setup. Session-average latency adds fixed setup divided by session request count to steady latency. The deadline gate uses cold latency because setup has not completed before the first request arrives. Pre-established sessions would require a separately stated timing contract. Threat coverage remains a separate categorical mapping of the mechanism's execution boundary.")})]), _concept_reflection(_reflection_contract, "D")])

    def build_part_e():
        intro = mo.md("### E · Can the lifecycle policy be defended? (9 min)\nTraverse each descendant artifact copy for deletion, then count which explicit incident events remain inside the selected retention window.")
        if e_prediction.value is None:
            return mo.vstack([intro, e_mode, e_retention, e_prediction])
        deletion = e_result["output"]["deletion"]
        retention = e_result["output"]["retention"]
        status_rows = [{"Artifact": status["artifact_id"], "Tracked": "YES" if status["tracked"] else "NO", "Deleted at": "UNKNOWN" if status["deletion_completed_at_hours"] is None else f"{status['deletion_completed_at_hours']:.1f} h", "Within SLA": "YES" if status["within_sla"] else "NO"} for status in deletion["statuses"]]
        baseline_deletion = e_base["output"]["deletion"]
        baseline_retention = e_base["output"]["retention"]
        summary = [
            {"Case": "Baseline", "Deletion proof": "PASS" if baseline_deletion["complete"] else "FAIL", "Retained incident events": baseline_retention["retained_event_count"], "Expired incident events": baseline_retention["expired_event_count"], "Evidence events deleted with copies": baseline_deletion["incident_evidence_events_lost"]},
            {"Case": "Selected", "Deletion proof": "PASS" if deletion["complete"] else "FAIL", "Retained incident events": retention["retained_event_count"], "Expired incident events": retention["expired_event_count"], "Evidence events deleted with copies": deletion["incident_evidence_events_lost"]},
        ]
        return mo.vstack([mo.vstack([intro, mo.hstack([e_mode, e_retention], widths="equal", wrap=True), e_prediction, table(status_rows), table(summary), mo.callout(mo.md(f"**Prediction:** {_read_prediction(e_prediction)}. Deletion proof **{'passes' if deletion['complete'] else 'fails'}**; **{retention['retained_event_count']}** incident events remain and **{retention['expired_event_count']}** expire under the retention window."), kind="success" if deletion["complete"] else "danger"), e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md("Deletion completion follows the explicit copy graph: a child copy completes after its parent plus its own deletion time. An untracked copy has unknown completion. Baseline lineage failure indicates that multi-tier storage topologies may violate deletion SLAs without copy restructuring. Retention compares event ages independently of deletion proofs. This model demonstrates lifecycle engineering, not legal compliance.")})]), _concept_reflection(_reflection_contract, "E")])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            state = "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING")
            rows.append({"Part": part, "Original prediction": _read_prediction(_saved_prediction_widgets[part], value=(capture.to_dict()["prediction"] if capture else "—")), "Evidence": state})
        saved_d = _captures["D"].to_dict()["decision"] if "D" in _captures else None
        ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and (final_choice.value == "defer" or final_choice.value == saved_d)
        return mo.vstack([mo.md("### Synthesis · Defend one release memo (5 min)\nRecommend the mechanism tested in Part D or defer release pending evidence. Deferring preserves the tested comparison without adopting its mechanism. Quantify a rejected alternative from the saved comparison, state the remaining limitation, and name the event that forces reevaluation."), table(rows), mo.callout(mo.md("Threat coverage, formal guarantee, supplied utility, latency, and lifecycle evidence remain separate claims in the memo."), kind="info"), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale, mo.callout(mo.md("**Ready for the local report.**" if ready else "Capture five current contrasts, recommend the saved Part D mechanism or defer, choose a different rejected mechanism, and add a quantified rationale."), kind="success" if ready else "warn")])

    tabs = mo.ui.tabs({"Part A": build_part_a(), "Part B": build_part_b(), "Part C": build_part_c(), "Part D": build_part_d(), "Part E": build_part_e(), "Synthesis": build_synthesis()})
    _lesson_shell([_part_tabs_hint(), tabs], volume=2)
    return (audit,)


@app.cell
def _():
    from mlsysbook_labs.notebook import notebook_shell as _submit_shell, notebook_submission as _submit_guide
    from mlsysbook_labs.learning_paths import get_learning_path as _submit_path
    _submit_shell([_submit_guide(_submit_path(2, 13))], volume=2)
    return


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, rationale,
    report_export_panel, scenario, track_id,
):
    from mlsysbook_labs.notebook import notebook_shell as _workspace_shell
    _captures = get_evidence()
    _saved_d = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and (final_choice.value == "defer" or final_choice.value == _saved_d)
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not _ready, _download_readiness(volume=2))
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol2/lab_13_security_privacy.py"),
        track=track_id, scenario=scenario.label,
        source_trace=(
            f"mlsysim.engine.v2_13_experiments (evaluate_threat_coverage, release_bounded_mean, "
            f"evaluate_supplied_quality, compose_privacy_events, evaluate_trust_latency, "
            f"propagate_deletion, evaluate_retention, evaluate_collection_budget, evaluate_gradient_integrity, evaluate_clean_label_poisoning); "
            f"scenario: {scenario.label} (illustrative scenario; no empirical benchmark claims)"
        ),
        learning_objectives=["Match a defense boundary to an explicit adversary", "Calculate a bounded-mean Laplace demonstration and basic composition", "Separate privacy guarantee, supplied utility, latency, and deletion evidence", "Test finite gradient integrity, clean-label poisoning, and persistent collection budgets"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "alternatives": snapshots[part]["alternatives"]} for part in "ABCDE"},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["A defense protects only the boundary named by its threat model.", "Basic sequential composition sums explicit releases; fixed-output queries add no training privacy loss.", "Utility observations, system overhead, and deletion evidence do not alter the formal privacy guarantee.", "A clean label audit and aggregate accuracy can miss a targeted training failure; daily quotas do not bound lifetime collection."],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": snapshots, "recommendation": final_choice.value, "rejected": final_rejected.value, "trigger": final_trigger.value, "residual_risk": final_risk.value},
    )
    _workspace_shell([mo.vstack([mo.md("### Download your discussion"), report_export_panel(report)])], volume=2)
    return (report,)


@app.cell
def _(
    audit, final_choice, final_rejected, final_risk, final_trigger, get_evidence, mo, rationale, track_id
):
    from mlsysbook_labs.fleet_evidence import discussion_fingerprint as _discussion_fingerprint
    _captures = get_evidence()
    _saved_d = _captures['D'].to_dict()['decision'] if 'D' in _captures else None
    _ready = audit.complete and all((widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))) and bool(rationale.value.strip()) and (final_choice.value != final_rejected.value) and (final_choice.value == 'defer' or final_choice.value == _saved_d)
    fleet_discussion_ready = _ready
    fleet_discussion_design = {'schema_version': 1, 'lab_id': 'v2_13', 'track_id': track_id, 'model_id': 'v2_13_experiments', 'evidence': {part: capture.to_dict() for part, capture in _captures.items()}, 'recommendation': final_choice.value, 'rejected_alternative': final_rejected.value, 'reevaluation_trigger': final_trigger.value, 'residual_risk': final_risk.value, 'rationale': rationale.value} if _ready else None
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
        ledger, chapter=13, track=track_id, design=fleet_discussion_design,
        fingerprint=fleet_discussion_fingerprint, ready=fleet_discussion_ready,
        requested=bool(save_discussion.value),
    )
    _discussion_shell([mo.vstack([save_discussion, _notebook_status(_outcome.status, _outcome.detail)])], volume=2)
    return


if __name__ == "__main__":
    app.run()
