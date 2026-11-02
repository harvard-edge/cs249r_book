import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 14: Evidence Under Shift · MLSysBook")


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
    from mlsysim.labs.experiments.v2_14_experiments import (
        competence_review_payload, conformal_fixture,
        robustness_review_payload, monitor_experiment_payload,
        selective_experiment_payload, track_context,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata,
        report_export_panel,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence
    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, apply_plotly_theme,
        audit_evidence, build_lab_report, capture_evidence,
        competence_review_payload, conformal_fixture,
        robustness_review_payload, get_lab_metadata, go, ledger, mo,
        monitor_experiment_payload, report_export_panel,
        selective_experiment_payload, track_context,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(ledger, mo, set_evidence):
    from mlsysbook_labs.contexts import preferred_context as _preferred_context
    _track_options = {"TinyML":"tinyml", "Mobile":"mobile", "Edge":"edge", "Cloud":"cloud"}
    track = mo.ui.dropdown(
        _track_options,
        value=next(label for label, context in _track_options.items() if context == _preferred_context(ledger)), label="Deployment context",
        on_change=lambda _value: set_evidence({}),
    )
    return (track,)


@app.cell
def _(track, track_context):
    track_id = track.value
    profile = track_context(track_id)
    return profile, track_id


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_scenario = mo.ui.dropdown(
        {"Covariate shift":"covariate_shift", "Concept drift":"concept_drift",
         "Input corruption":"corruption", "Adversarial input":"attack"},
        value="Concept drift", label="Trace event",
    )
    a_delay = mo.ui.slider(0, 4, value=2, step=1, label="Label delay (periods)")
    b_threshold = mo.ui.slider(0.50, 0.95, value=0.80, step=0.05, label="Acceptance threshold")
    b_capacity = mo.ui.slider(0, 20, value=8, step=1, label="Fallback capacity")
    c_threat = mo.ui.dropdown(
        {"Covariate shift":"covariate_shift", "Concept drift":"concept_drift",
         "Corruption":"corruption", "Attack":"attack"},
        value="Attack", label="Stress evidence",
    )
    c_test = mo.ui.dropdown(
        {"Input filter":"input_filter", "Robust training":"robust_training",
         "Fallback ensemble":"fallback_ensemble"},
        value="Robust training", label="Defense to test",
    )
    c_layers = mo.ui.slider(1, 4, value=4, step=1, label="Ordered poisoning-defense layers")
    c_lineage = mo.ui.dropdown({"Trace descendants for quarantine":True, "Lineage unavailable":False},
                               value="Trace descendants for quarantine", label="Historical artifact response")
    d_routing = mo.ui.dropdown({"Low entropy":"entropy", "Sample agreement":"consistency", "Evidence support":"evidence"},
                               value="Evidence support", label="Generated-answer acceptance")
    d_source = mo.ui.dropdown({"Current release handbook":"current", "Archived release handbook":"archived"},
                              value="Current release handbook", label="Named evidence source")
    c_decision = mo.ui.radio(
        {"Adopt tested defense":"adopt", "Hold baseline":"baseline"},
        label="Decision",
    )
    d_count = mo.ui.dropdown({"3 scores":3, "9 scores":9}, value="3 scores", label="Calibration sample")
    d_alpha = mo.ui.dropdown(
        {"90% target":0.10, "80% target":0.20, "70% target":0.30},
        value="90% target", label="Miscoverage alpha",
    )
    d_exchange = mo.ui.dropdown(
        {"Assume exchangeability for exercise":True, "Known shift; do not assume":False},
        value="Assume exchangeability for exercise", label="Exchangeability premise",
    )
    return a_delay, a_scenario, b_capacity, b_threshold, c_decision, c_test, c_threat, c_layers, c_lineage, d_alpha, d_count, d_exchange, d_routing, d_source


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Feature monitor first":"feature", "Delayed labels first":"labels", "Neither":"neither"},
        label="Which signal first reveals the event?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Under 25%":"under25", "25–49%":"25-49", "50–74%":"50-74", "75% or more":"75plus"},
        label="What fallback share will result?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Benefit with small tax":"small", "Benefit with large tax":"large", "No supported benefit":"none"},
        label="What will the defense evidence show?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Finite threshold":"finite", "Full prediction set":"full"},
        label="What does the corrected rank require?",
    ).form(submit_button_label="Lock Part D prediction")
    c_pipeline_prediction = mo.ui.radio(
        {"Ingress suffices":"ingress", "More layers needed; lineage repairs weights":"repair", "More layers needed; lineage identifies descendants":"trace"},
        label="Can trusted-source screening and clean probes close this poisoning incident?",
    ).form(submit_button_label="Lock pipeline prediction")
    d_generation_prediction = mo.ui.radio(
        {"Agreement proves current truth":"agreement", "Entropy proves current truth":"entropy", "Neither establishes current evidence support":"neither"},
        label="Can the generation signals certify the deployment handbook answers?",
    ).form(submit_button_label="Lock generation prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, c_pipeline_prediction, d_generation_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    choices = {"Keep baseline":"baseline", "Input filter":"input_filter",
               "Robust training":"robust_training", "Fallback ensemble":"fallback_ensemble"}
    final_choice = mo.ui.radio(choices, label="Recommendation")
    final_rejected = mo.ui.radio(choices, label="Rejected alternative")
    final_trigger = mo.ui.radio(
        {"Delayed error threshold":"error", "Fallback overflow":"overflow",
         "Threat outside scope":"scope", "Exchangeability breaks":"exchangeability",
         "Evidence version changes":"evidence", "Trigger probe fails":"trigger"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Delayed labels":"delay", "Rare cohort":"cohort",
         "Adaptive attacker":"attacker", "Fallback capacity":"fallback",
         "Stale handbook":"source", "Historical lineage gap":"lineage"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Concise rationale",
        placeholder="Use saved numbers, reject an alternative, name a poisoning blind spot and lineage response, qualify the handbook evidence, and name the trigger.",
     debounce=False)
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    a_delay, a_scenario, b_capacity, b_threshold, c_test, c_threat, c_layers, c_lineage,
    competence_review_payload, conformal_fixture, d_alpha, d_count,
    d_exchange, d_routing, d_source, robustness_review_payload, monitor_experiment_payload,
    selective_experiment_payload, track_id,
):
    a_base = monitor_experiment_payload(
        track_id, a_scenario.value, feature_threshold=0.20,
        error_threshold=1.0, label_delay_periods=a_delay.value,
    )
    a_result = monitor_experiment_payload(
        track_id, a_scenario.value, feature_threshold=0.20,
        error_threshold=0.25, label_delay_periods=a_delay.value,
    )
    b_base = selective_experiment_payload(track_id, threshold=0.50, fallback_capacity=b_capacity.value)
    b_result = selective_experiment_payload(track_id, threshold=b_threshold.value, fallback_capacity=b_capacity.value)
    b_curve = tuple(
        selective_experiment_payload(track_id, threshold=x, fallback_capacity=b_capacity.value)
        for x in (0.50, 0.60, 0.70, 0.80, 0.90)
    )
    c_base = robustness_review_payload(track_id, "baseline", c_threat.value, layer_count=1, lineage=False)
    c_result = robustness_review_payload(track_id, c_test.value, c_threat.value, layer_count=c_layers.value, lineage=c_lineage.value)
    c_hold = robustness_review_payload(track_id, "baseline", c_threat.value, layer_count=c_layers.value, lineage=c_lineage.value)
    c_alternatives = tuple(
        robustness_review_payload(track_id, x, c_threat.value, layer_count=c_layers.value, lineage=c_lineage.value)
        for x in ("input_filter", "robust_training", "fallback_ensemble")
    )
    base_scores, base_test = conformal_fixture(track_id, 9)
    scores, test_scores = conformal_fixture(track_id, d_count.value)
    d_base = competence_review_payload(base_scores, base_test, alpha=0.20, exchangeability_assumed=True, routing="entropy", evidence_source=d_source.value)
    d_result = competence_review_payload(
        scores, test_scores, alpha=d_alpha.value,
        exchangeability_assumed=d_exchange.value, routing=d_routing.value, evidence_source=d_source.value,
    )
    return a_base, a_result, b_base, b_curve, b_result, c_alternatives, c_base, c_hold, c_result, d_base, d_result


@app.cell
def _(
    a_base, a_delay, a_prediction, a_result, a_scenario, b_base, b_capacity,
    b_prediction, b_result, b_threshold, c_alternatives, c_base, c_hold, c_decision,
    c_prediction, c_pipeline_prediction, c_result, c_test, c_threat, c_layers, c_lineage, capture_evidence, d_alpha,
    d_base, d_count, d_exchange, d_prediction, d_generation_prediction, d_routing, d_source, d_result, mo, set_evidence,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})
    a_upstream = {"scenario_id":a_scenario.value, "label_delay_periods":a_delay.value}
    b_upstream = {"threshold":b_threshold.value, "fallback_capacity":b_capacity.value}
    c_upstream = {"threat_id":c_threat.value, "tested_defense":c_test.value, "decision":c_decision.value, "layer_count":c_layers.value, "lineage":c_lineage.value}
    d_upstream = {"calibration_count":d_count.value, "alpha":d_alpha.value, "exchangeability_assumed":d_exchange.value, "routing":d_routing.value, "evidence_source":d_source.value}
    c_choice = c_test.value if c_decision.value == "adopt" else "baseline"
    c_chosen = c_result if c_decision.value == "adopt" else c_hold
    a_capture = mo.ui.button(
        label="Capture detection contrast", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value, inputs=a_upstream,
            baseline=a_base, result=a_result, upstream_inputs=a_upstream,
            model_key="v2_14_experiments.monitor_experiment_payload")),
    )
    b_capture = mo.ui.button(
        label="Capture risk–coverage contrast", kind="success",
        disabled=b_prediction.value is None or b_threshold.value == 0.50,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value, inputs=b_upstream,
            baseline=b_base, result=b_result, upstream_inputs=b_upstream,
            model_key="v2_14_experiments.selective_experiment_payload")),
    )
    c_capture = mo.ui.button(
        label="Capture defense decision", kind="success",
        disabled=c_prediction.value is None or c_pipeline_prediction.value is None or c_decision.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction={"defense":c_prediction.value,"pipeline":c_pipeline_prediction.value}, inputs=c_upstream,
            baseline=c_base, result=c_result, alternatives=c_alternatives,
            decision=c_choice, chosen_result=c_chosen,
            result_role="chosen intervention" if c_decision.value == "adopt" else "rejected alternative",
            upstream_inputs=c_upstream,
            model_key="v2_14_experiments.robustness_review_payload")),
    )
    d_capture = mo.ui.button(
        label="Capture competence contrasts", kind="success", disabled=d_prediction.value is None or d_generation_prediction.value is None,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction={"conformal":d_prediction.value,"generation":d_generation_prediction.value}, inputs=d_upstream,
            baseline=d_base, result=d_result, upstream_inputs=d_upstream,
            model_key="v2_14_experiments.competence_review_payload")),
    )
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    from mlsysbook_labs.notebook import (
        notebook_header as _course_header, notebook_shell as _course_shell,
        notebook_learning_map as _course_learning_map,
        notebook_scenario as _course_scenario,
    )
    from mlsysbook_labs.learning_paths import get_learning_path as _course_path
    from mlsysbook_labs import get_lab_metadata as _course_metadata
    css = mo.Html("""<style>




    .saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.lab-hud{display:flex;flex-wrap:wrap;gap:10px;background:#101827!important;color:white;padding:14px 18px;border-radius:9px}

    </style>""")
    header = _course_header(_course_metadata("vol2/lab_14_robust_ai.py"), 'When fleet health stays green but predictions become unreliable, which evidence justifies containment?')
    _course_shell([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        css,
        header,
        track,
        _course_scenario(_course_path('vol2', 14), track=track.value, volume=2),
        mo.md("All outcomes are illustrative supplied fixtures. Each Calculation Notes panel states the scope and assumptions."),
        _course_learning_map(_course_path('vol2', 14))
    ], volume=2).style({"padding-top": "32px"})
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base, a_capture, a_delay, a_prediction, a_result, a_scenario,
    a_upstream, apply_plotly_theme, audit_evidence, b_base, b_capacity,
    b_capture, b_curve, b_prediction, b_result, b_threshold, b_upstream,
    c_base, c_capture, c_decision, c_prediction, c_pipeline_prediction, c_result, c_test, c_threat, c_layers, c_lineage,
    c_upstream, d_alpha, d_base, d_capture, d_count, d_exchange, d_prediction,
    d_result, d_upstream, d_routing, d_source, d_generation_prediction, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, go, mo, profile, rationale, track_id,
):
    _saved_prediction_widgets = {'A': a_prediction, 'B': b_prediction, 'C': c_prediction, 'D': d_prediction}
    from mlsysbook_labs.notebook import notebook_prediction_label as _read_prediction
    from mlsysbook_labs.notebook import notebook_tab_hint as _part_tabs_hint, notebook_shell as _lesson_shell
    from mlsysbook_labs.notebook import notebook_part_reflection as _concept_reflection
    from mlsysbook_labs.learning_paths import get_learning_path as _reflection_path
    _reflection_contract = _reflection_path('vol2', 14)
    _captures = get_evidence()
    upstream = {"A":a_upstream, "B":b_upstream, "C":c_upstream, "D":d_upstream}
    audit = audit_evidence(_captures, track=track_id, required_parts=tuple("ABCD"), per_part_upstream_inputs=upstream, contrast_required_parts=tuple("ABCD"))
    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width":"100%", "overflow-x":"auto"})
    def saved(part):
        cap = _captures.get(part)
        if cap is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture."), kind="danger")
        _prediction = cap.to_dict()["prediction"]
        if isinstance(_prediction, dict):
            _prediction = _prediction["defense" if part == "C" else "conformal"]
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · prediction: {_read_prediction(_saved_prediction_widgets[part], value=_prediction)}. All recorded subpredictions are included in the discussion report.</div>')
    def part_a():
        heading = mo.md(f"### A · Can a feature monitor see every silent failure? (10 min)\nChoose a labeled trace for {profile['device_role']} and predict which signal detects the event first.")
        if a_prediction.value is None:
            return mo.vstack([heading, mo.hstack([a_scenario,a_delay],widths="equal",wrap=True), a_prediction])
        summaries = a_result["result"]["summaries"]
        fig = apply_plotly_theme(go.Figure())
        fig.add_scatter(x=[x["period"] for x in summaries],y=[100*x["feature_total_variation"] for x in summaries],name="Feature movement",mode="lines+markers")
        fig.add_scatter(x=[x["period"] for x in summaries],y=[100*x["error_rate"] for x in summaries],name="Labeled error",mode="lines+markers")
        fig.update_layout(height=350,margin=dict(l=50,r=20,t=80,b=65),xaxis_title="Trace period",yaxis_title="Observed rate or distance (%)",legend=dict(orientation="h",x=0,y=1.02,xanchor="left",yanchor="bottom"))
        base,result=a_base["result"],a_result["result"]
        rows=[{"Monitor":"Feature only","First flag":base["first_detection_period"],"Delay":base["detection_delay_periods"],"Misses":base["missed_event_periods"]},{"Monitor":"Feature + labels","First flag":result["first_detection_period"],"Delay":result["detection_delay_periods"],"Misses":result["missed_event_periods"]}]
        interpretation = result.get("interpretation", "Concept drift can leave feature movement at zero.")
        return mo.vstack([mo.vstack([heading,mo.hstack([a_scenario,a_delay],widths="equal",wrap=True),a_prediction,fig,table(rows),mo.callout(mo.md(f"**Prediction:** {_read_prediction(a_prediction)}. Event start: **{result['event_start']}**; combined first flag: **{result['first_detection_period']}**. {interpretation}"),kind="info"),a_capture,saved("A"),mo.accordion({"Calculation Notes":mo.md(f"Total-variation distance uses observed input bins on {profile['device_role']}. Error uses fixture predictions and labels verified at {profile['backend_role']}. Thresholds change flags, false alarms, and delay; they do not change outcomes or authorize retraining.")})]), _concept_reflection(_reflection_contract, "A")])
    def part_b():
        heading=mo.md(f"### B · How much work does abstention move downstream? (10 min)\nPredict fallback demand from {profile['device_role']} to {profile['fallback_destination']} before sweeping the confidence threshold.")
        if b_prediction.value is None:
            return mo.vstack([heading,mo.hstack([b_threshold,b_capacity],widths="equal",wrap=True),b_prediction])
        fig=go.Figure([go.Scatter(x=[100*x["result"]["coverage"] for x in b_curve],y=[100*x["result"]["selective_risk"] for x in b_curve],mode="lines+markers")])
        fig.update_layout(height=260,margin=dict(l=20,r=20,t=20,b=20),xaxis_title=f"Accepted coverage of {profile['unit']} (%)",yaxis_title="Error among accepted (%)",showlegend=False)
        base,result=b_base["result"],b_result["result"]
        base_cohorts = ", ".join(f"{k}: {v}" for k, v in base.get("cohort_fallback_arrivals", {}).items())
        res_cohorts = ", ".join(f"{k}: {v}" for k, v in result.get("cohort_fallback_arrivals", {}).items())
        rows=[
            {"Run":"Baseline","Coverage":f"{100*base['coverage']:.0f}%","Risk":f"{100*base['selective_risk']:.1f}%","Fallback":f"{base['fallback_arrivals']} {profile['unit']}","Overflow":f"{base['fallback_overflow']} {profile['unit']}","Cohort breakdown":base_cohorts},
            {"Run":"Chosen","Coverage":f"{100*result['coverage']:.0f}%","Risk":f"{100*result['selective_risk']:.1f}%","Fallback":f"{result['fallback_arrivals']} {profile['unit']}","Overflow":f"{result['fallback_overflow']} {profile['unit']}","Cohort breakdown":res_cohorts},
        ]
        return mo.vstack([mo.vstack([heading,mo.hstack([b_threshold,b_capacity],widths="equal",wrap=True),b_prediction,apply_plotly_theme(fig),table(rows),mo.callout(mo.md(f"**Prediction:** {_read_prediction(b_prediction)}. The chosen threshold sends **{result['fallback_arrivals']} {profile['unit']}** to **{profile['fallback_destination']}** (capacity **{result['fallback_capacity']} {profile['unit']}**). Cohort breakdown: {res_cohorts}."),kind="danger" if result["fallback_overflow"] else "info"),b_capture,saved("B"),mo.accordion({"Calculation Notes":mo.md(f"Coverage and selective risk use actual confidence scores and labels. Abstaining on {profile['device_role']} routes fallback to {profile['fallback_destination']}. Fallback capacity limits overflow; it cannot change risk or coverage. Cohort breakdowns reveal tail vulnerability in the {profile['scenario_name']}.")})]), _concept_reflection(_reflection_contract, "B")])
    def part_c():
        heading=mo.md(f"### C · Does the defense match the threat and envelope? (14 min)\nCompare supplied clean and stressed outcomes for {profile['device_role']} and record a decision.")
        if c_prediction.value is None or c_pipeline_prediction.value is None:
            return mo.vstack([heading,mo.hstack([c_threat,c_test],widths="equal",wrap=True),c_prediction,
                              mo.md("**Pipeline incident:** an already deployed descendant consumed a batch. Trusted contributors and clean aggregate probes may hide targeted failures. Predict, then replay each ordered layer."),c_pipeline_prediction])
        base,result=c_base["result"],c_result["result"]
        stress="out of scope" if result["stressed_accuracy"] is None else f"{100*result['stressed_accuracy']:.1f}%"
        rows=[
            {"Run":"Baseline","Clean":f"{100*base['clean_accuracy']:.1f}%","Stressed":f"{100*base['stressed_accuracy']:.1f}%","Latency":f"{base['mean_latency']['magnitude']:.1f} {base['mean_latency']['unit']}","Energy":f"{base['mean_energy']['magnitude']:.2f} {base['mean_energy']['unit']}","Fleet boundary":base.get("cost_boundary", "")},
            {"Run":c_test.value,"Clean":f"{100*result['clean_accuracy']:.1f}%","Stressed":stress,"Latency":f"{result['mean_latency']['magnitude']:.1f} {result['mean_latency']['unit']}","Energy":f"{result['mean_energy']['magnitude']:.2f} {result['mean_energy']['unit']}","Fleet boundary":result.get("cost_boundary", "")},
        ]
        pipeline = result["poisoning"]
        pipeline_panel = mo.vstack([
            mo.md("#### Pipeline incident · which layer changes which obligation?\nKeep the inference-defense choice fixed while adding pipeline layers one at a time. Then toggle lineage alone. Training review uses supplied influence findings; it does not train a model."),
            mo.hstack([c_layers,c_lineage],widths="equal",wrap=True),c_pipeline_prediction,
            table([{"Record":r["record"],"Poison in supplied incident":r["poison_truth"],
                    "Allowed source":r["trusted_source"],"Anomaly flag":r["anomaly_flag"],
                    "Training review flag":r["influence_flag"],"Trigger probe fails":r["trigger_probe_fails"],
                    "Disposition":r["disposition"]} for r in pipeline["record_trace"]]),
            table([{"Measure":label,"Ingress only":base["poisoning"][key],"Selected layers":pipeline[key]}
                   for label,key in (("Poison records retained","retained_poison_records"),
                                     ("Legitimate records quarantined","quarantined_legitimate_records"),
                                     ("Promotion blocked","promotion_blocked"),
                                     ("Historical descendant scope known","descendant_scope_known"),
                                     ("Descendant weights repaired","descendant_weights_repaired"))]),
            mo.md("**Identified descendants:** " + (", ".join(pipeline["identified_descendants"]) or "Unknown scope; absence of lineage does not establish absence of affected artifacts.")),
            mo.callout(mo.md(f"The candidate's fixed clean probe passes **{pipeline['clean_probe_correct']}/{pipeline['clean_probe_total']}** cases, while the trigger probe passes **{pipeline['trigger_probe_correct']}/{pipeline['trigger_probe_total']}**. Adding gates changes quarantine or promotion, never those recorded outcomes. Lineage identifies affected descendants; replacement and retesting remain separate work."),kind="warn"),
        ])
        decision=c_test.value if c_decision.value=="adopt" else "baseline"
        boundary_note = f" Boundary: {result['cost_boundary']}." if result.get("cost_boundary") else ""
        return mo.vstack([mo.vstack([heading,mo.hstack([c_threat,c_test],widths="equal",wrap=True),c_prediction,table(rows),c_decision,mo.callout(mo.md(f"**Prediction:** {_read_prediction(c_prediction)}. **Decision:** {decision}. Out-of-scope evidence stays out of scope.{boundary_note}"),kind="success" if result["applicable"] else "danger"),pipeline_panel,c_capture,saved("C"),mo.accordion({"Calculation Notes":mo.md(f"Accuracy comes from fixed clean and stressed outcome fixtures. Latency and energy retain Pint units. No defense gain is extrapolated beyond the named threat scope. Fleet deployment boundaries ({profile['device_role']} vs {profile['gateway_role']}) explicitly document where physical costs are unmodeled.")})]), _concept_reflection(_reflection_contract, "C")])
    def part_d():
        heading=mo.md(f"### D · When does finite calibration force a full set? (14 min)\nApply the corrected conformal rank on {profile['device_role']}, then separate observed coverage from the exchangeability premise.")
        if d_prediction.value is None or d_generation_prediction.value is None:
            return mo.vstack([heading,mo.hstack([d_count,d_alpha],widths="equal",wrap=True),d_exchange,d_prediction,
                              mo.md("**Generated deployment advice:** four recorded handbook answers include confident mistakes. Predict whether agreement or low entropy establishes current support."),d_generation_prediction])
        base,result=d_base["result"],d_result["result"]
        rows=[{"Run":"Reference","n":base["calibration_count"],"Rank":base["corrected_rank"],"Threshold":base["threshold"],"Full set":base["full_set_required"],"Fixture coverage":f"{100*base['empirical_coverage']:.0f}%"},{"Run":"Chosen","n":result["calibration_count"],"Rank":result["corrected_rank"],"Threshold":"∞ / full set" if result["full_set_required"] else result["threshold"],"Full set":result["full_set_required"],"Fixture coverage":f"{100*result['empirical_coverage']:.0f}%"}]
        generation=result["generation"]
        generation_panel=mo.vstack([
            mo.md("#### Generated advice · stability against which reference?\nFirst compare entropy, agreement, and evidence routing at one source. Then switch only the handbook version. Conformal label coverage above does not establish factuality of these generated claims."),
            mo.hstack([d_routing,d_source],widths="equal",wrap=True),d_generation_prediction,
            table([{"Prompt":r["prompt"],"Recorded answer":r["answer"],"Handbook excerpt":r["evidence_excerpt"],
                    "Supported":r["evidence_support"],"Accepted":r["accepted"]} for r in generation["cases"]]),
            table([{"Prompt":r["prompt"],"Entropy signal":r["entropy_signal"],
                    "Sample agreement":f"{100*r['sample_agreement']:.1f}%",
                    "Answer in context":r["context_has_current_answer"],
                    "Current support":r["supported_by_current_reference"]} for r in generation["cases"]]),
            table([{"Measure":label,"Entropy routing":base["generation"][key],"Selected routing":generation[key]}
                   for label,key in (("Accepted answers","accepted"),("Review arrivals","review_arrivals"),
                                     ("Unsupported by selected source","unsupported_accepted"),
                                     ("Wrong against current handbook","current_reference_errors_accepted"),
                                     ("Decode sequences","decode_sequences"),("Evidence checks","evidence_checks"))]),
            mo.callout(mo.md(f"Reference: **{generation['reference']}**. Prompt set: **{generation['prompt_set']}**. Fixed decode protocol: {generation['decode_protocol']}. Retrieval lacks the current answer in **{generation['retrieval_gap_cases']}** case(s); generation overrides available correct context in **{generation['generation_gap_cases']}** case(s). Agreement can repeat a mistake; an archived reference can support obsolete advice. Compare review load and actual sequence/check counts before choosing a route."),kind="info"),
        ])
        return mo.vstack([mo.vstack([heading,mo.hstack([d_count,d_alpha],widths="equal",wrap=True),d_exchange,d_prediction,table(rows),mo.callout(mo.md(f"**Prediction:** {_read_prediction(d_prediction)}. Corrected rank: **{result['corrected_rank']}** with n={result['calibration_count']}. {result['assumption_note']}"),kind="warn" if result["full_set_required"] or not result["exchangeability_assumed"] else "info"),generation_panel,d_capture,saved("D"),mo.accordion({"Calculation Notes":mo.md(f"The rank is ceil((n+1)(1−alpha)). Rank greater than n requires an infinite threshold and full label set. The assumption flag documents exchangeability across the {profile['scenario_name']}; it does not verify or certify it. Generated-answer support is separate supplied adjudication against a named source, prompt set, and fixed decode protocol. Counts of sequences and evidence checks do not model latency, energy, verifier accuracy, or future factuality.")})]), _concept_reflection(_reflection_contract, "D")])
    def synthesis():
        rows=[]
        for part in "ABCD":
            cap=_captures.get(part)
            status="CURRENT" if cap and part not in audit.stale and (part,part) not in audit.identical_pairs else ("STALE" if cap else "MISSING")
            rows.append({"Part":part,"Prediction":cap.to_dict()["prediction"] if cap else "—","Evidence":status})
        saved_decision=_captures["C"].to_dict()["decision"] if "C" in _captures else None
        complete=audit.complete and all(x.value is not None for x in (final_choice,final_rejected,final_trigger,final_risk)) and bool(rationale.value.strip()) and final_choice.value==saved_decision and final_choice.value!=final_rejected.value
        return mo.vstack([mo.md(f"### Synthesis · Defend a bounded fleet decision (5 min)\nChoose one option for the {profile['scenario_name']} ({profile['device_role']} → {profile['gateway_role']}), quantify a rejected alternative, preserve the recorded pipeline and generated-answer routing choices, name the remaining limitation, and state a reevaluation trigger."),table(rows),mo.hstack([final_choice,final_rejected],widths="equal",wrap=True),mo.hstack([final_risk,final_trigger],widths="equal",wrap=True),rationale,mo.callout(mo.md("**Ready for the local report.**" if complete else "Capture four current contrasts, match Part C, reject a different option, and complete the rationale."),kind="success" if complete else "warn")])
    tabs=mo.ui.tabs({"Part A":part_a(),"Part B":part_b(),"Part C":part_c(),"Part D":part_d(),"Synthesis":synthesis()})
    _lesson_shell([_part_tabs_hint(), tabs], volume=2)
    return (audit,)


@app.cell
def _():
    from mlsysbook_labs.notebook import notebook_shell as _submit_shell, notebook_submission as _submit_guide
    from mlsysbook_labs.learning_paths import get_learning_path as _submit_path
    _submit_shell([_submit_guide(_submit_path(2, 14))], volume=2)
    return


@app.cell
def _(audit, build_lab_report, final_choice, final_rejected, final_risk, final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale, report_export_panel, track_id):
    from mlsysbook_labs.notebook import notebook_shell as _workspace_shell
    _captures=get_evidence()
    _decision=_captures["C"].to_dict()["decision"] if "C" in _captures else None
    _ready=audit.complete and all(x.value is not None for x in (final_choice,final_rejected,final_trigger,final_risk)) and bool(rationale.value.strip()) and final_choice.value==_decision and final_choice.value!=final_rejected.value
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not _ready, _download_readiness(volume=2))
    snapshots={part:_captures[part].to_dict() for part in "ABCD"}
    report=build_lab_report(
        get_lab_metadata("vol2/lab_14_robust_ai.py"), track=track_id, scenario=profile["scenario_name"],
        learning_objectives=["Distinguish feature shift from concept drift","Compute detection outcomes from flags and events","Trade selective risk against fallback load","Bound defense and conformal claims", "Trace layered poisoning defenses and lineage", "Separate generation signals from current evidence support"],
        predictions={p:snapshots[p]["prediction"] for p in "ABCD"},
        knob_settings={p:snapshots[p]["inputs"] for p in "ABCD"},
        evidence_summary={p:{"baseline":snapshots[p]["baseline"],"result":snapshots[p]["result"],"result_role":snapshots[p]["result_role"],"chosen_result":snapshots[p]["chosen_result"],"alternatives":snapshots[p]["alternatives"]} for p in "ABCD"},
        binding_constraints={"A":"detection evidence","B":"fallback capacity","C":"threat scope, pipeline layers, lineage","D":"finite rank, exchangeability, named evidence source"},
        decisions={"recommendation":final_choice.value,"rejected_alternative":final_rejected.value,"reevaluation_trigger":final_trigger.value},
        final_decision={"recommendation":final_choice.value,"rejected_alternative":final_rejected.value,"rationale":rationale.value,"pipeline_layers":snapshots["C"]["inputs"]["layer_count"],"lineage":snapshots["C"]["inputs"]["lineage"],"generation_routing":snapshots["D"]["inputs"]["routing"],"evidence_source":snapshots["D"]["inputs"]["evidence_source"]},
        big_takeaways=["Feature monitoring can miss concept drift.","Abstention creates fallback work.","Robustness claims stay within their evidence scope."],
        reflections={"rationale":rationale.value}, residual_risk=final_risk.value,
        result_snapshot={"track":track_id,"captures":snapshots,"recommendation":final_choice.value,"rejected":final_rejected.value,"trigger":final_trigger.value,"residual_risk":final_risk.value},
        source_trace={"scenario":"Illustrative labeled outcome fixtures.","calculations":"MLSysIM v2_14_experiments."},
    )
    _workspace_shell([mo.vstack([mo.md("### Download your discussion"),report_export_panel(report)])], volume=2)
    return (report,)


@app.cell
def _(
    audit, final_choice, final_rejected, final_risk, final_trigger, get_evidence, mo, rationale, track_id
):
    from mlsysbook_labs.fleet_evidence import discussion_fingerprint as _discussion_fingerprint
    _captures = get_evidence()
    _decision = _captures['C'].to_dict()['decision'] if 'C' in _captures else None
    _ready = audit.complete and all((x.value is not None for x in (final_choice, final_rejected, final_trigger, final_risk))) and bool(rationale.value.strip()) and (final_choice.value == _decision) and (final_choice.value != final_rejected.value)
    fleet_discussion_ready = _ready
    fleet_discussion_design = {'schema_version': 1, 'lab_id': 'v2_14', 'track_id': track_id, 'model_id': 'v2_14_experiments', 'evidence': {p: c.to_dict() for p, c in _captures.items()}, 'recommendation': final_choice.value, 'rejected_alternative': final_rejected.value, 'reevaluation_trigger': final_trigger.value, 'residual_risk': final_risk.value, 'rationale': rationale.value} if _ready else None
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
        ledger, chapter=14, track=track_id, design=fleet_discussion_design,
        fingerprint=fleet_discussion_fingerprint, ready=fleet_discussion_ready,
        requested=bool(save_discussion.value),
    )
    _discussion_shell([mo.vstack([save_discussion, _notebook_status(_outcome.status, _outcome.detail)])], volume=2)
    return


if __name__ == "__main__":
    app.run()
