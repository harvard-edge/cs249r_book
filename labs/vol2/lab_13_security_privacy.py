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
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        labs_dir = Path(__file__).resolve().parents[1]
        if str(labs_dir) not in sys.path:
            sys.path.insert(0, str(labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    from mlsysim.engine.v2_13_experiments import (
        PrivacyEvent, compose_privacy_events, evaluate_retention,
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
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Deployment track",
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
    return a_control, b_epsilon, c_operation, c_query_count, c_release_count, d_mechanism, d_requests, e_mode, e_retention


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
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Encrypted transport": "transport", "Secure aggregation": "secure_aggregation", "Confidential runtime": "confidential_runtime", "Locked device": "locked_device"},
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
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


# ZONE C · ONE TABBED INVESTIGATION SURFACE
@app.cell
def _(
    PrivacyEvent, a_control, asdict, b_epsilon, c_operation, c_query_count,
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
        disabled=a_prediction.value is None or a_control.value == "transport",
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"selected_control": a_control.value}, baseline=a_base,
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
        disabled=c_prediction.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"operation": c_operation.value}, baseline=c_base,
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
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin:30px 0 14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:780px}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    @media(max-width:520px){.pilot-head{border-radius:9px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    roles_html = f"<div><b>Fleet roles</b><br>{' → '.join(scenario.fleet_roles)}</div>" if getattr(scenario, "fleet_roles", None) else ""
    locus_html = f"<div><b>Attack locus</b><br>{scenario.adversary_locus}</div>" if getattr(scenario, "adversary_locus", None) else ""
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME II · LAB 13</span><span>ABOUT 50–55 MIN</span></div><h1>The Price of Privacy</h1><p>Which protection can this fleet defend when threat coverage, privacy guarantee, utility, latency, and lifecycle evidence are judged separately?</p><div class="pilot-meta"><div><b>Track</b><br>{scenario.label}</div><div><b>Adversary</b><br>{scenario.adversary.replace('_', ' ')}</div>{roles_html}{locus_html}<div><b>Deliverable</b><br>Security and privacy release memo</div></div></section>""")
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, css, header, track,
        mo.Html('<p class="pilot-note">All workloads, outcome counts, overheads, and copy paths are illustrative scenarios. Seeded Laplace draws make the exercise replayable; a deterministic public seed is predictable and is not production release randomness. This lab does not implement a DP-SGD accountant.</p>'),
    ], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base, a_capture, a_control, a_prediction, a_result, a_upstream,
    apply_plotly_theme, audit_evidence, b_base, b_capture, b_epsilon,
    b_prediction, b_result, b_upstream, c_base, c_capture, c_operation,
    c_prediction, c_query_count, c_release_count, c_result, c_upstream,
    d_alternatives, d_base, d_capture, d_mechanism, d_prediction, d_requests,
    d_result, d_upstream, e_base, e_capture, e_mode, e_prediction, e_result, e_retention,
    e_upstream, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, go, mo, rationale, scenario, track_id,
):
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
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; later control changes do not rewrite this record.</small></div>')

    def build_part_a():
        locus_text = f" attacking at **{scenario.adversary_locus}**" if getattr(scenario, "adversary_locus", None) else ""
        roles_text = f"\n\n**Fleet topology:** {' → '.join(scenario.fleet_roles)}." if getattr(scenario, "fleet_roles", None) else ""
        intro = mo.md(f"### A · Which boundary does the defense protect? (8 min)\nThe stated adversary is **{scenario.adversary.replace('_', ' ')}**{locus_text}.{roles_text}\n\n*Where mechanisms run:* In distributed and edge fleets (including TinyML), mechanisms run at specific tiers. Locked device execution protects the device hardware/MCU; encrypted transport protects network links; secure aggregation and confidential runtimes protect aggregation gateways or backend servers. Compare encrypted transport with one selected architecture before deciding whether the relevant boundary is covered.")
        if a_prediction.value is None:
            return mo.vstack([intro, a_control, a_prediction])
        rows = [
            {"Architecture": a_base["output"]["control_name"], "Execution tier": a_base["output"].get("execution_tier", "Transit"), "Boundary": a_base["output"]["protected_boundary"], "Adversary covered": "YES" if a_base["output"]["covered"] else "NO"},
            {"Architecture": a_result["output"]["control_name"], "Execution tier": a_result["output"].get("execution_tier", "Specified tier"), "Boundary": a_result["output"]["protected_boundary"], "Adversary covered": "YES" if a_result["output"]["covered"] else "NO"},
        ]
        return mo.vstack([intro, a_control, a_prediction, table(rows), mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. **Observed categorical result:** {a_result['output']['control_name']} ({a_result['output'].get('execution_tier', '')}) {'covers' if a_result['output']['covered'] else 'does not cover'} the stated adversary at the **{a_result['output']['protected_boundary']}** boundary."), kind="success" if a_result["output"]["covered"] else "danger"), a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("Coverage is set membership against an explicit adversary capability. Mechanisms running at gateway or backend tiers (such as secure aggregation or confidential VMs) cannot protect an MCU against an attacker with physical device possession. Latency cannot improve or weaken this categorical result.")})])

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
        return mo.vstack([intro, b_epsilon, b_prediction, apply_plotly_theme(fig), table(rows), mo.callout(mo.md(f"**Prediction:** {b_prediction.value}. The guarantee calculation yields scale **{base_release['laplace_scale']:.4g} → {result_release['laplace_scale']:.4g} {units}**. The separate supplied evaluation for *{result_quality['task']}* yields **{base_quality['correct']}/{base_quality['evaluated']} → {result_quality['correct']}/{result_quality['evaluated']}** correct."), kind="info"), mo.callout(mo.md("The shown draw uses a deterministic public seed for replay. Predictable randomness is unsuitable for a production privacy-preserving release. No legal compliance is claimed."), kind="warn"), b_capture, saved("B"), mo.accordion({"Calculation Notes": mo.md(f"For public bounds [L, U] and n fixed records, replace-one mean sensitivity is (U − L)/n in {units}. The Laplace scale is sensitivity/epsilon. Output clipping is postprocessing and does not change epsilon. Quality counts are illustrative scenario fixtures, not empirical compliance benchmarks.")})])

    def build_part_c():
        intro = mo.md(f"### C · Which repeated operations consume privacy budget? (9 min)\nIn the {scenario.label}, training privacy and retraining mechanisms run at the aggregation gateway or backend service, while client devices perform inference. Start with one explicit DP release (e.g., federated round or telemetry release), then compare multiple new releases with either fixed-model queries (client inference) or a new unaccounted private-data access (backend retraining).")
        operation_control = c_query_count if c_operation.value == "fixed_queries" else mo.md("The new fine-tuning access has no tested accountant in this exercise.")
        if c_prediction.value is None:
            return mo.vstack([intro, c_operation, c_release_count, operation_control, c_prediction])
        result = c_result["output"]
        fig = go.Figure([go.Bar(x=["One release", "Follow-on history"], y=[c_base["output"]["total_epsilon"], result["total_epsilon"]], marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]])])
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Composed epsilon", showlegend=False)
        rows = [{"Accounted releases": result["accounted_release_count"], "Zero-additional-loss operations": result["zero_additional_training_loss_count"], "Unaccounted private accesses": result["unguaranteed_private_access_count"], "Guarantee complete": "YES" if result["guarantee_complete"] else "NO"}]
        return mo.vstack([intro, mo.hstack([c_operation, c_release_count], widths="equal", wrap=True), operation_control, c_prediction, apply_plotly_theme(fig), table(rows), mo.callout(mo.md(f"**Prediction:** {c_prediction.value}. Basic sequential composition gives **ε = {result['total_epsilon']:.3g}**. Fixed released-model queries add zero training privacy loss; an unaccounted new private-data access makes the claimed guarantee incomplete."), kind="success" if result["guarantee_complete"] else "danger"), c_capture, saved("C"), mo.accordion({"Calculation Notes": mo.md("Basic sequential composition sums epsilon across explicit DP releases. Postprocessing and queries of a fixed released DP model do not revisit private training data. This statement concerns training privacy loss; it does not dismiss model-extraction risk, nor does it constitute a legal privacy guarantee.")})])

    def build_part_d():
        path_text = f" across **{scenario.path_description}**" if getattr(scenario, "path_description", None) else ""
        intro = mo.md(f"### D · Which trust mechanism meets the deadline? (9 min)\nThe fleet request path{path_text} starts at **{scenario.base_latency_ms:g} ms** and must finish by **{scenario.deadline_ms:g} ms**. Protection scope (at the mechanism's execution tier) and runtime deadline are independent checks.")
        if d_prediction.value is None:
            return mo.vstack([intro, d_mechanism, d_requests, d_prediction])
        rows = [{"Mechanism": alternative["output"]["mechanism_name"], "Execution tier": alternative["output"].get("execution_tier", "Fleet tier"), "Protected": "YES" if alternative["output"]["protected"] else "NO", "Latency": f"{alternative['output']['total_latency_ms']:.2f} ms", "Deadline": "PASS" if alternative["output"]["meets_deadline"] else "FAIL"} for alternative in d_alternatives]
        fig = go.Figure([go.Bar(x=["Baseline", "Selected"], y=[d_base["output"]["total_latency_ms"], d_result["output"]["total_latency_ms"]], marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]])])
        fig.add_hline(y=scenario.deadline_ms, line_dash="dash", annotation_text="Deadline")
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Request latency (ms)", showlegend=False)
        return mo.vstack([intro, mo.hstack([d_mechanism, d_requests], widths="equal", wrap=True), d_prediction, apply_plotly_theme(fig), table(rows), mo.callout(mo.md(f"**Prediction:** {d_prediction.value}. Selected mechanism protection: **{'PASS' if d_result['output']['protected'] else 'FAIL'}** ({d_result['output'].get('execution_tier', '')}). Deadline: **{'PASS' if d_result['output']['meets_deadline'] else 'FAIL'}** at {d_result['output']['total_latency_ms']:.2f} ms."), kind="success" if d_result["output"]["protected"] and d_result["output"]["meets_deadline"] else "danger"), d_capture, saved("D"), mo.accordion({"Calculation Notes": mo.md("Per-request latency equals base latency plus per-request overhead plus fixed session overhead divided by requests in the session. Threat coverage remains a separate categorical mapping of the mechanism's execution boundary.")})])

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
        return mo.vstack([intro, mo.hstack([e_mode, e_retention], widths="equal", wrap=True), e_prediction, table(status_rows), table(summary), mo.callout(mo.md(f"**Prediction:** {e_prediction.value}. Deletion proof **{'passes' if deletion['complete'] else 'fails'}**; **{retention['retained_event_count']}** incident events remain and **{retention['expired_event_count']}** expire under the retention window."), kind="success" if deletion["complete"] else "danger"), e_capture, saved("E"), mo.accordion({"Calculation Notes": mo.md("Deletion completion follows the explicit copy graph: a child copy completes after its parent plus its own deletion time. An untracked copy has unknown completion. Baseline lineage failure indicates that multi-tier storage topologies may violate deletion SLAs without copy restructuring. Retention compares event ages independently of deletion proofs. This model demonstrates lifecycle engineering, not legal compliance.")})])

    def build_synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            state = "CURRENT" if capture and part not in audit.stale and (part, part) not in audit.identical_pairs else ("STALE" if capture else "MISSING")
            rows.append({"Part": part, "Original prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": state})
        saved_d = _captures["D"].to_dict()["decision"] if "D" in _captures else None
        ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == saved_d
        return mo.vstack([mo.md("### Synthesis · Defend one release memo (5 min)\nChoose one tested mechanism, quantify a rejected alternative from the saved comparison, state the remaining limitation, and name the event that forces reevaluation."), table(rows), mo.callout(mo.md("Threat coverage, formal guarantee, supplied utility, latency, and lifecycle evidence remain separate claims in the memo."), kind="info"), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale, mo.callout(mo.md("**Ready for the local report.**" if ready else "Capture five current contrasts, match the recommendation to saved Part D, choose a different rejected mechanism, and add a quantified rationale."), kind="success" if ready else "warn")])

    tabs = mo.ui.tabs({"Part A": build_part_a(), "Part B": build_part_b(), "Part C": build_part_c(), "Part D": build_part_d(), "Part E": build_part_e(), "Synthesis": build_synthesis()})
    tabs
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, rationale,
    report_export_panel, scenario, track_id,
):
    _captures = get_evidence()
    _saved_d = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved_d
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol2/lab_13_security_privacy.py"),
        track=track_id, scenario=scenario.label,
        source_trace=(
            f"mlsysim.engine.v2_13_experiments (evaluate_threat_coverage, release_bounded_mean, "
            f"evaluate_supplied_quality, compose_privacy_events, evaluate_trust_latency, "
            f"propagate_deletion, evaluate_retention); "
            f"scenario: {scenario.label} (illustrative scenario; no empirical benchmark claims)"
        ),
        learning_objectives=["Match a defense boundary to an explicit adversary", "Calculate a bounded-mean Laplace demonstration and basic composition", "Separate privacy guarantee, supplied utility, latency, and deletion evidence"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "alternatives": snapshots[part]["alternatives"]} for part in "ABCDE"},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["A defense protects only the boundary named by its threat model.", "Basic sequential composition sums explicit releases; fixed-output queries add no training privacy loss.", "Utility observations, system overhead, and deletion evidence do not alter the formal privacy guarantee."],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": snapshots, "recommendation": final_choice.value, "rejected": final_rejected.value, "trigger": final_trigger.value, "residual_risk": final_risk.value},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _saved_d = _captures["D"].to_dict()["decision"] if "D" in _captures else None
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and final_choice.value != final_rejected.value and final_choice.value == _saved_d
    _save_error = None
    _saved = False
    if _ready:
        try:
            ledger.save(
                track=track_id, chapter=13,
                design={
                    "schema_version": 1, "lab_id": "v2_13", "track_id": track_id,
                    "model_id": "v2_13_experiments",
                    "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                    "recommendation": final_choice.value,
                    "rejected_alternative": final_rejected.value,
                    "reevaluation_trigger": final_trigger.value,
                    "residual_risk": final_risk.value, "rationale": rationale.value,
                },
            )
            await ledger.flush()
            _saved = True
        except Exception as error:
            _save_error = f"{type(error).__name__}: {error}"
    _status = "SAVED" if _saved else ("SAVE FAILED" if _save_error else "EVIDENCE IN PROGRESS")
    _hud = mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">13 · The Price of Privacy</span><span class="hud-separator">|</span><span style="flex:1"></span><span class="hud-label">STATUS</span><span class="hud-active">{_status}</span></div>')
    _display = mo.vstack([_hud, mo.callout(mo.md(f"**Ledger save failed.** {_save_error}. The local report remains available above."), kind="danger")]) if _save_error else _hud
    _display
    return


if __name__ == "__main__":
    app.run()
