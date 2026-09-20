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
    from mlsysim.engine.v1_15_experiments import (
        LINEAGE_POLICIES, TRACKS, bounded_mean_release, investigate_incident,
        lifecycle_footprint, population_mix, threshold_consequences,
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
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, LINEAGE_POLICIES, TRACKS,
        apply_plotly_theme, audit_evidence, bounded_mean_release,
        build_lab_report, capture_evidence, get_lab_metadata, go,
        investigate_incident, ledger, lifecycle_footprint, mo, population_mix,
        report_export_panel, threshold_consequences,
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
def _(TRACKS, track):
    track_id = track.value
    profile = TRACKS[track_id]
    return profile, track_id


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
        value="Double demand", label="Inference demand",
    )
    d_retraining = mo.ui.slider(0, 6, value=2, step=1, label="Retraining runs")
    d_explanations = mo.ui.slider(0, 30, value=int(profile["explanation_share_pct"]), step=1, label="Decisions explained (%)")
    e_policy = mo.ui.dropdown(
        {"Full lineage": "full", "Decision evidence only": "decision_only", "Minimal release record": "minimal"},
        value="Decision evidence only", label="Retained evidence",
    )
    e_delay = mo.ui.slider(0.5, 12.0, value=profile["default_containment_delay_hours"], step=0.5, label="Containment delay (hours)")
    return a_share, b_choice, b_rejected, c_epsilon, c_releases, d_demand, d_explanations, d_retraining, e_delay, e_policy


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Aggregate accuracy rises": "rises", "Aggregate accuracy falls": "falls", "Only subgroup rates change": "subgroups_change"},
        label="What changes when only the population mixture changes?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"TPR and FPR both rise": "both_rise", "TPR and FPR both fall": "both_fall", "TPR rises while FPR falls": "opposite", "Neither changes": "neither"},
        label="What happens when the positive-score threshold rises?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Less noise and lower per-release spend": "less_noise", "More noise and lower per-release spend": "more_noise", "More noise and higher per-release spend": "more_spend", "Storage alone changes epsilon": "storage_changes_epsilon"},
        label="What does a smaller epsilon per release change?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Initial training": "initial_training", "Retraining": "retraining", "Inference": "inference", "Explanation": "explanation"},
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
        {"Subgroup TPR or FPR changes": "fairness_shift", "Privacy release count changes": "privacy_spend", "Lifecycle demand doubles": "lifecycle_demand", "Required lineage field is missing": "lineage_gap"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Small fixed evaluation fixture": "fixture", "Unmodeled lifecycle boundary": "boundary", "Unspecified consequence severity": "consequences", "Future distribution shift": "shift"},
        label="Residual limitation",
    )
    rationale = mo.ui.text_area(
        label="Concise rationale",
        placeholder="Use saved subgroup errors, privacy utility, lifecycle energy, and reconstructability to defend the choice.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    LINEAGE_POLICIES, a_share, b_choice, b_rejected, bounded_mean_release,
    c_epsilon, c_releases, d_demand, d_explanations, d_retraining, e_delay,
    e_policy, investigate_incident, lifecycle_footprint, population_mix,
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
    d_base = lifecycle_footprint(track_id, demand_scale=1.0, retraining_runs=0, explanation_share_pct=0)
    d_result = lifecycle_footprint(
        track_id, demand_scale=d_demand.value, retraining_runs=d_retraining.value,
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
def _(
    a_base, a_prediction, a_result, a_share, b_base, b_choice, b_prediction,
    b_rejected, b_result, b_results, c_base, c_epsilon, c_prediction,
    c_releases, c_result, capture_evidence, d_base, d_demand,
    d_explanations, d_prediction, d_result, d_retraining, e_base, e_delay,
    e_policy, e_prediction, e_result, mo, set_evidence, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    _a_base_share = a_base["affected_share_pct"]
    a_upstream = {"baseline_affected_share_pct": _a_base_share, "affected_share_pct": a_share.value}
    b_upstream = {"choice": b_choice.value, "rejected": b_rejected.value, "affected_share_pct": a_share.value}
    c_upstream = {"epsilon_per_release": c_epsilon.value, "releases": c_releases.value}
    d_upstream = {"demand_scale": d_demand.value, "retraining_runs": d_retraining.value, "explanation_share_pct": d_explanations.value}
    e_upstream = {"threshold": e_result["inputs"]["threshold"], "affected_share_pct": a_share.value, "retention_policy": e_policy.value, "containment_delay_hours": e_delay.value}
    a_capture = mo.ui.button(
        label="Capture population contrast", kind="success", disabled=a_prediction.value is None or a_share.value == _a_base_share,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value, inputs={"affected_share_pct": a_share.value},
            baseline=a_base, result=a_result, upstream_inputs=a_upstream,
            model_key="v1_15_experiments.population_mix",
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
            inputs={"choice": b_choice.value, "rejected": b_rejected.value}, baseline=b_base,
            result=b_result, alternatives=tuple(b_results.values()), decision=b_choice.value,
            chosen_result=b_base if b_choice.value == "hold" else b_result,
            result_role="rejected alternative" if b_choice.value == "hold" else "chosen candidate",
            upstream_inputs=b_upstream, model_key="v1_15_experiments.threshold_consequences",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture privacy release", kind="success",
        disabled=c_prediction.value is None or (c_epsilon.value == 2.0 and c_releases.value == 1),
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"epsilon_per_release": c_epsilon.value, "releases": c_releases.value},
            baseline=c_base, result=c_result, upstream_inputs=c_upstream,
            model_key="v1_15_experiments.bounded_mean_release",
        )),
    )
    d_capture = mo.ui.button(
        label="Capture lifecycle boundary", kind="success",
        disabled=d_prediction.value is None or (d_demand.value == 1.0 and d_retraining.value == 0 and d_explanations.value == 0),
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value, inputs=d_upstream,
            baseline=d_base, result=d_result, upstream_inputs=d_upstream,
            model_key="v1_15_experiments.lifecycle_footprint",
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
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#111827,#365314);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#ecfccb;max-width:780px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}
    .lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}.lab-hud .hud-failed{color:#fecaca}
    @media(max-width:520px){.pilot-head{border-radius:9px;margin-top:30px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 15</span><span>ABOUT 50 MIN</span></div><h1>Responsibility Is a System Requirement</h1><p>What must we measure, preserve, and trade off before a fast, accurate system is fit to release?</p><div class="pilot-meta"><div><b>Track</b><br>{profile['display']}</div><div><b>Context</b><br>{profile['scenario']}</div><div><b>Deliverable</b><br>Responsible engineering decision memo</div></div></section>""")
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, css, header, track,
        mo.Html('<p class="pilot-note">All workloads, scores, labels, energy terms, and incident rates are explicit illustrative scenario assumptions. Each part’s Calculation Notes states the mechanism and boundary.</p>'),
    ], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base, a_capture, a_prediction, a_result, a_share, a_upstream,
    apply_plotly_theme, audit_evidence, b_capture, b_choice, b_prediction,
    b_rejected, b_results, b_upstream, c_base, c_capture, c_epsilon,
    c_prediction, c_releases, c_result, c_upstream, d_base, d_capture,
    d_demand, d_explanations, d_prediction, d_result, d_retraining,
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
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** A dependency changed, or the saved runs are identical. Recapture this part."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Track {data["track"]}; the report uses this saved result rather than changed live controls.</small></div>')

    def part_a():
        setup = mo.md("### A · Who disappears inside aggregate accuracy? (8 min)\n**Evaluation lead:** keep the fixed decisions and labels, then change only how much of the evaluated population comes from the affected group.")
        if a_prediction.value is None:
            return mo.vstack([setup, a_share, a_prediction])
        figure = go.Figure()
        for label, result in ((f"{a_base['affected_share_pct']:.0f}% affected", a_base), (f"{a_share.value:.0f}% affected", a_result)):
            figure.add_bar(name=label, x=["Reference", "Affected", "Aggregate"], y=[
                100 * result["groups"]["reference"]["accuracy"],
                100 * result["groups"]["affected"]["accuracy"],
                100 * result["aggregate"]["accuracy"],
            ])
        figure.update_layout(barmode="group", height=280, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Accuracy (%)", legend_orientation="h")
        rows = [{
            "Mixture": label, "Reference accuracy": f"{100 * result['groups']['reference']['accuracy']:.1f}%",
            "Affected accuracy": f"{100 * result['groups']['affected']['accuracy']:.1f}%",
            "Aggregate accuracy": f"{100 * result['aggregate']['accuracy']:.1f}%",
        } for label, result in (("Baseline", a_base), ("Changed", a_result))]
        return mo.vstack([
            setup, a_share, a_prediction, apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {a_prediction.value}. The subgroup accuracies remain fixed while aggregate accuracy moves from **{100 * a_base['aggregate']['accuracy']:.1f}%** to **{100 * a_result['aggregate']['accuracy']:.1f}%**."), kind="info"),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md("The simulator computes each subgroup confusion matrix from fixed scores and labels at threshold 0.50. It then weights subgroup rates by population share. Mixture does not alter a subgroup prediction rule.")}),
        ])

    def part_b():
        setup = mo.md("### B · Which threshold expresses the obligation? (9 min)\n**Release owner:** compare the false-positive and false-negative consequences of three thresholds on the same fixed score-and-label fixture.")
        if b_prediction.value is None:
            return mo.vstack([setup, b_prediction])
        figure = go.Figure()
        for group, color in (("reference", COLORS["BlueLine"]), ("affected", COLORS["OrangeLine"])):
            figure.add_scatter(name=f"{group.title()} TPR", x=[0.3, 0.5, 0.7], y=[100 * b_results[key]["groups"][group]["tpr"] for key in ("permissive", "balanced", "strict")], mode="lines+markers", line=dict(color=color))
            figure.add_scatter(name=f"{group.title()} FPR", x=[0.3, 0.5, 0.7], y=[100 * b_results[key]["groups"][group]["fpr"] for key in ("permissive", "balanced", "strict")], mode="lines+markers", line=dict(color=color, dash="dot"))
        figure.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Positive-score threshold", yaxis_title="Rate (%)", legend_orientation="h")
        rows = [{
            "Policy": name.title(), "Expected FP": f"{result['expected_false_positives']:.0f}",
            "FP consequence": result["false_positive_consequence"], "Expected FN": f"{result['expected_false_negatives']:.0f}",
            "FN consequence": result["false_negative_consequence"],
        } for name, result in b_results.items()]
        apply_plotly_theme(figure)
        figure.update_layout(
            legend=dict(orientation="v", x=0, y=-0.35, xanchor="left", yanchor="top"),
            margin=dict(b=170),
            height=440,
        )
        return mo.vstack([
            setup, b_prediction, figure, table(rows),
            mo.hstack([b_choice, b_rejected], widths="equal", wrap=True),
            mo.callout(mo.md(f"**Your prediction:** {b_prediction.value}. Raising the threshold lowers or preserves both TPR and FPR. It transfers expected decisions from false positives toward false negatives; the table leaves their severity as an explicit stakeholder judgment."), kind="info"),
            b_capture, saved("B"),
            mo.accordion({"Calculation Notes": mo.md("A decision is positive when score ≥ threshold. Each threshold replays identical scores and labels, so a higher threshold cannot create a new positive prediction. Counts are expected errors in 10,000 decisions with the selected group mixture.")}),
        ])

    def part_c():
        setup = mo.md("### C · What evidence can we release? (10 min)\n**Privacy engineer:** release a bounded mean while preserving a usable summary and accounting for repeated access to the same private data.")
        if c_prediction.value is None:
            return mo.vstack([setup, mo.hstack([c_epsilon, c_releases], widths="equal", wrap=True), c_prediction])
        figure = go.Figure([
            go.Bar(name="True bounded mean", x=["Reference", "Selected"], y=[c_base["true_bounded_mean"], c_result["true_bounded_mean"]], marker_color=COLORS["BlueLine"]),
            go.Bar(name="Released value", x=["Reference", "Selected"], y=[c_base["released_values"][0], c_result["released_values"][0]], marker_color=COLORS["OrangeLine"]),
        ])
        figure.update_layout(barmode="group", height=275, margin=dict(l=20, r=20, t=20, b=20), yaxis_title=profile["privacy_value_label"], legend_orientation="h")
        rows = [
            {"Case": "Reference", "ε/release": f"{c_base['epsilon_per_release']:.2f}", "Releases": c_base["releases"], "Composed ε": f"{c_base['basic_composed_epsilon']:.2f}", "Noise scale": f"{c_base['laplace_scale']:.2f}", "Absolute error": f"{c_base['mean_absolute_error']:.2f}"},
            {"Case": "Selected", "ε/release": f"{c_result['epsilon_per_release']:.2f}", "Releases": c_result["releases"], "Composed ε": f"{c_result['basic_composed_epsilon']:.2f}", "Noise scale": f"{c_result['laplace_scale']:.2f}", "Absolute error": f"{c_result['mean_absolute_error']:.2f}"},
        ]
        return mo.vstack([
            setup, mo.hstack([c_epsilon, c_releases], widths="equal", wrap=True), c_prediction,
            apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {c_prediction.value}. Raw fixture storage is **{c_result['raw_storage_bytes']} bytes**; {c_result['releases']} released summaries occupy **{c_result['released_summary_bytes']} bytes**. Retention changes storage, while only the specified mechanism and composition determine epsilon."), kind="info"),
            c_capture, saved("C"),
            mo.accordion({"Calculation Notes": mo.md("The bounded-mean Laplace mechanism clips values to public bounds, uses sensitivity (upper − lower) / fixed public n, and adds Laplace noise with scale sensitivity / ε. Basic sequential composition sums ε across independent new releases. Output clipping is postprocessing. The fixed seed supports reproducible simulation only; deployed mechanisms require appropriately generated secret randomness.")}),
        ])

    def part_d():
        setup = mo.md("### D · What does responsibility cost over the lifetime? (10 min)\n**Systems architect:** choose one accounting boundary and compare initial training, retraining, inference, and requested explanations over the same operating period.")
        if d_prediction.value is None:
            return mo.vstack([setup, mo.hstack([d_demand, d_retraining, d_explanations], widths="equal", wrap=True), d_prediction])
        figure = go.Figure()
        colors = (COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["GreenLine"], COLORS["RedLine"])
        for term, color in zip(("initial_training", "retraining", "inference", "explanation"), colors):
            figure.add_bar(name=term.replace("_", " ").title(), x=["Reference", "Selected"], y=[d_base["terms_kwh"][term], d_result["terms_kwh"][term]], marker_color=color)
        figure.update_layout(barmode="stack", height=300, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Lifecycle energy (kWh)", legend_orientation="h")
        rows = [{
            "Case": label, "Total energy": f"{result['total_energy_kwh']:.2f} kWh",
            "Operational carbon": f"{result['operational_carbon_kg']:.2f} kg CO₂e",
            "Dominant term": result["dominant_term"].replace("_", " "),
        } for label, result in (("Reference", d_base), ("Selected", d_result))]
        return mo.vstack([
            setup, mo.hstack([d_demand, d_retraining, d_explanations], widths="equal", wrap=True), d_prediction,
            apply_plotly_theme(figure), table(rows),
            mo.callout(mo.md(f"**Your prediction:** {d_prediction.value}. The selected boundary totals **{d_result['total_energy_kwh']:.2f} kWh** and **{d_result['operational_carbon_kg']:.2f} kg CO₂e**; **{d_result['dominant_term'].replace('_', ' ')}** dominates."), kind="info"),
            d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md(d_result["boundary"] + " Energy equals the sum of the four displayed work terms. Operational carbon equals total energy multiplied by the track’s illustrative grid intensity.")}),
        ])

    def part_e():
        setup = mo.md("### E · Can we investigate and contain failure? (8 min)\n**Incident commander:** remove retained evidence and vary response delay. Separate what can be reconstructed from how many decisions occur before containment.")
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
            mo.md("### Synthesis · Defend a responsible release decision (5 min)\nUse saved evidence to name the chosen threshold or hold, quantify a tested rejected threshold, state a remaining limitation, and define a reevaluation trigger."),
            table(rows),
            mo.callout(mo.md("Saved snapshots remain fixed while live controls move. Recapture stale evidence before generating the report."), kind="info"),
            mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
            mo.hstack([final_trigger, final_risk], widths="equal", wrap=True), rationale,
            mo.callout(mo.md("**Ready for the local report.**" if complete else "Complete five current contrasts, match the recommendation to the saved Part B decision, compare a different tested threshold when releasing, and add the rationale."), kind="success" if complete else "warn"),
        ])

    tabs = mo.ui.tabs({"Part A": part_a(), "Part B": part_b(), "Part C": part_c(), "Part D": part_d(), "Part E": part_e(), "Synthesis": build_synthesis()})
    tabs
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures = get_evidence()
    _b_decision = _captures["B"].to_dict()["decision"] if "B" in _captures else None
    _b_saved_rejected = _captures["B"].to_dict()["inputs"]["rejected"] if "B" in _captures else None
    _choice_is_distinct = final_choice.value == "hold" or final_choice.value != final_rejected.value
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and _choice_is_distinct and final_choice.value == _b_decision and final_rejected.value == _b_saved_rejected
    mo.stop(not _ready)
    _snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_15_responsible_engr.py"), track=track_id, scenario=profile["scenario"],
        learning_objectives=["Expose subgroup outcomes hidden by aggregate accuracy", "Compare threshold, privacy, lifecycle, and accountability consequences", "Defend a release or hold decision with preserved evidence"],
        predictions={part: _snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: _snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {"baseline": _snapshots[part]["baseline"], "result": _snapshots[part]["result"], "chosen_result": _snapshots[part]["chosen_result"], "result_role": _snapshots[part]["result_role"], "alternatives": _snapshots[part]["alternatives"]} for part in "ABCDE"},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["Aggregate accuracy can hide fixed subgroup disparities as population mixture changes.", "Thresholds transfer false-positive and false-negative consequences rather than resolving every obligation.", "Privacy, lifecycle cost, reconstructability, and containment each require a measurable mechanism and boundary."],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value}, residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": _snapshots, "recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value, "residual_risk": final_risk.value},
        source_trace={"scenario": "Illustrative fixed teaching fixtures, not empirical deployment measurements.", "calculations": "All quantitative outcomes are produced by the Chapter 15 simulator experiments."},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(audit, final_choice, final_rejected, final_risk, final_trigger, get_evidence, ledger, mo, rationale, track_id):
    _captures = get_evidence()
    _b_decision = _captures["B"].to_dict()["decision"] if "B" in _captures else None
    _b_saved_rejected = _captures["B"].to_dict()["inputs"]["rejected"] if "B" in _captures else None
    _choice_is_distinct = final_choice.value == "hold" or final_choice.value != final_rejected.value
    _ready = audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk)) and bool(rationale.value.strip()) and _choice_is_distinct and final_choice.value == _b_decision and final_rejected.value == _b_saved_rejected
    _status = "EVIDENCE IN PROGRESS"
    _status_class = "hud-active"
    if _ready:
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
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">15 · Responsibility Is a System Requirement</span><span aria-hidden="true"> · STATUS: </span><span class="{_status_class}">{_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
