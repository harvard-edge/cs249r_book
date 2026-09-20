import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="full",
    app_title="Lab 09: Selection That Pays · MLSysBook",
)


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
    from mlsysim.engine.v1_09_experiments import (
        MODEL_ID, TRACKS, acquisition_option, compare_policies,
        full_pool_baseline, learning_curve, population_shift,
        selection_amortization, serialize_value,
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
        ACADEMIC_LAB_CSS,
        COLORS,
        LAB_CSS,
        MODEL_ID,
        Q_,
        TRACKS,
        acquisition_option,
        apply_plotly_theme,
        audit_evidence,
        build_lab_report,
        capture_evidence,
        compare_policies,
        full_pool_baseline,
        get_lab_metadata,
        go,
        learning_curve,
        ledger,
        mo,
        population_shift,
        report_export_panel,
        selection_amortization,
        serialize_value,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Teaching track",
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
    b_retained = mo.ui.dropdown(
        {"6 of 12": 6, "8 of 12": 8, "10 of 12": 10}, value="8 of 12",
        label="Examples retained",
    )
    b_conclusion = mo.ui.radio(
        {"Uniform": "uniform", "Deduplicate": "deduplicate",
         "Coverage-aware": "coverage", "Hold for more evidence": "none"},
        label="Policy to carry forward",
    )
    b_rejected = mo.ui.radio(
        {"Uniform": "uniform", "Deduplicate": "deduplicate", "Coverage-aware": "coverage"},
        label="Tested alternative",
    )
    c_scoring = mo.ui.slider(
        int(profile["scoring_time_min"].to("microsecond").magnitude),
        int(profile["scoring_time_max"].to("microsecond").magnitude),
        value=int(profile["scoring_time_per_example"].to("microsecond").magnitude),
        step=int(profile["scoring_time_step"].to("microsecond").magnitude),
        label="Scoring time per example (µs)",
    )
    c_runs = mo.ui.slider(1, 100, value=12, step=1, label="Repeated training runs")
    c_decision = mo.ui.radio(
        {"Adopt for this run count": "adopt", "Reuse more times first": "wait",
         "Reject on evidence": "reject"}, label="Amortization decision",
    )
    d_budget = mo.ui.slider(
        int(profile["budget_min"].to("USD").magnitude),
        int(profile["budget_max"].to("USD").magnitude),
        value=int(profile["budget_default"].to("USD").magnitude),
        step=int(profile["budget_step"].to("USD").magnitude),
        label="Available data budget (USD)",
    )
    d_choice = mo.ui.radio(
        {"Expert labels": "labels", "Generated examples": "generated", "Neither": "none"},
        label="Next data purchase",
    )
    d_rejected = mo.ui.radio(
        {"Expert labels": "labels", "Generated examples": "generated"},
        label="Rejected purchase",
    )
    e_rare_share = mo.ui.slider(
        int(profile["population_pct"]["rare"]), 45, value=35, step=1,
        label="Changed rare-cohort share (%)",
    )
    return (
        b_conclusion,
        b_rejected,
        b_retained,
        c_decision,
        c_runs,
        c_scoring,
        d_budget,
        d_choice,
        d_rejected,
        e_rare_share,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Both add similar value": "similar", "Redundant data saturates first": "redundant_first",
         "Informative data saturates first": "informative_first"},
        label="Which pool loses marginal value first?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Uniform has the safest coverage": "uniform", "Deduplication is safest": "deduplicate",
         "Coverage-aware selection is safest": "coverage", "No policy can pass": "none"},
        label="Which equal-size policy will satisfy the evidence checks?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Selection pays after 1–5 runs": "1-5", "Selection pays after 6–20 runs": "6-20",
         "Selection pays after more than 20 runs": "over-20", "It never pays in this sweep": "never"},
        label="When will selection first save time?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Labels preserve the rare cohort better": "labels",
         "Generated examples preserve it better": "generated",
         "The rare-cohort outcome is equal": "equal"},
        label="Which package preserves the rare cohort better?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"The carried policy remains supported": "supported", "Quality fails first": "quality",
         "Representation fails first": "representation", "Both checks fail": "both"},
        label="What happens after the population changes?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Uniform": "uniform", "Deduplicate": "deduplicate",
         "Coverage-aware": "coverage", "Hold for more evidence": "none"},
        label="Final data-selection policy",
    )
    final_rejected = mo.ui.radio(
        {"Uniform": "uniform", "Deduplicate": "deduplicate", "Coverage-aware": "coverage"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"Rare cohort reaches tested share": "population", "Selection reuse falls below break-even": "reuse",
         "Cohort outcome falls below its floor": "quality"}, label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Unobserved population change": "population_shift",
         "Generated-data validation gap": "synthetic_validation",
         "Training I/O omitted from cost": "training_io"}, label="Residual risk",
    )
    rationale = mo.ui.text_area(
        label="Evidence-based rationale",
        placeholder="Use saved counts, outcomes, time, and the rejected alternative.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    Q_,
    acquisition_option,
    b_conclusion,
    b_rejected,
    b_retained,
    c_runs,
    c_scoring,
    compare_policies,
    d_budget,
    e_rare_share,
    full_pool_baseline,
    learning_curve,
    population_shift,
    profile,
    selection_amortization,
    track_id,
):
    a_baseline = learning_curve(track_id, "redundant")
    a_result = learning_curve(track_id, "informative")
    b_comparison = compare_policies(track_id, b_retained.value)
    b_results = {item["policy_id"]: item for item in b_comparison["policies"]}
    b_full = full_pool_baseline(track_id)
    carried_policy = (
        b_conclusion.value
        if b_conclusion.value not in (None, "none")
        else (b_rejected.value or "coverage")
    )
    c_kwargs = {
        "track_id": track_id,
        "policy_id": carried_policy,
        "retained_count": b_retained.value,
        "scoring_time_per_example": Q_(c_scoring.value, "microsecond"),
        "repeated_runs": c_runs.value,
    }
    c_baseline = selection_amortization(**c_kwargs, comparison_path="full_pool")
    c_result = selection_amortization(**c_kwargs, comparison_path="selected_subset")
    budget = Q_(d_budget.value, "USD")
    d_labels = acquisition_option(track_id, "labels", budget)
    d_generated = acquisition_option(track_id, "generated", budget)
    d_none = acquisition_option(track_id, "none", budget)
    baseline_rare = profile["population_pct"]["rare"]
    e_baseline = population_shift(track_id, carried_policy, b_retained.value, baseline_rare)
    e_result = population_shift(track_id, carried_policy, b_retained.value, e_rare_share.value)
    return (
        a_baseline,
        a_result,
        b_full,
        b_results,
        c_baseline,
        c_result,
        carried_policy,
        d_generated,
        d_labels,
        d_none,
        e_baseline,
        e_result,
    )


@app.cell
def _(
    a_baseline,
    a_prediction,
    a_result,
    b_conclusion,
    b_full,
    b_prediction,
    b_rejected,
    b_results,
    b_retained,
    c_baseline,
    c_decision,
    c_prediction,
    c_result,
    capture_evidence,
    carried_policy,
    d_budget,
    d_choice,
    d_generated,
    d_labels,
    d_none,
    d_prediction,
    d_rejected,
    e_baseline,
    e_prediction,
    e_rare_share,
    e_result,
    mo,
    serialize_value,
    set_evidence,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"track_id": track_id, "comparison": "redundant_vs_informative"}
    b_upstream = {"track_id": track_id, "retained_count": b_retained.value}
    c_upstream = {
        **b_upstream,
        "carried_policy": carried_policy,
        "scoring_time_per_example_us": c_result["inputs"]["scoring_time_per_example"].to("microsecond").magnitude,
        "repeated_runs": c_result["repeated_runs"],
    }
    d_upstream = {"track_id": track_id, "budget_usd": d_budget.value}
    e_upstream = {
        **b_upstream, "carried_policy": carried_policy, "rare_share_pct": e_rare_share.value,
    }

    a_capture = mo.ui.button(
        label="Capture learning-curve contrast", kind="success",
        disabled=a_prediction.value is None,
        on_click=lambda _v: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value, inputs=a_upstream,
            baseline=serialize_value(a_baseline), result=serialize_value(a_result),
            upstream_inputs=a_upstream, model_key=a_result["model_key"],
        )),
    )
    b_invalid = (
        b_prediction.value is None or b_conclusion.value is None or b_rejected.value is None
        or (b_conclusion.value != "none" and b_conclusion.value == b_rejected.value)
    )
    b_result_policy = b_rejected.value if b_conclusion.value == "none" else b_conclusion.value
    fallback_policy = "deduplicate" if b_result_policy == "uniform" else "uniform"
    b_baseline_policy = b_rejected.value if b_conclusion.value != "none" else fallback_policy
    b_capture = mo.ui.button(
        label="Capture equal-count policy decision", kind="success", disabled=b_invalid,
        on_click=lambda _v: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={**b_upstream, "decision": b_conclusion.value, "tested_alternative": b_rejected.value},
            baseline=serialize_value(
                b_results[b_baseline_policy] if b_conclusion.value != "none" else b_full
            ),
            result=serialize_value(b_results[b_result_policy]),
            alternatives=serialize_value(tuple(b_results.values())), decision=b_conclusion.value,
            chosen_result=serialize_value(b_full) if b_conclusion.value == "none" else None,
            result_role="rejected alternative" if b_conclusion.value == "none" else "selected candidate",
            upstream_inputs=b_upstream, model_key="evaluate_policy",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture amortization contrast", kind="success",
        disabled=c_prediction.value is None or c_decision.value is None,
        on_click=lambda _v: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={**c_upstream, "decision": c_decision.value},
            baseline=serialize_value(c_baseline), result=serialize_value(c_result),
            decision=c_decision.value, upstream_inputs=c_upstream, model_key=c_result["model_key"],
        )),
    )
    d_invalid = (
        d_prediction.value is None or d_choice.value is None or d_rejected.value is None
        or (d_choice.value != "none" and d_choice.value == d_rejected.value)
    )
    d_options = {"labels": d_labels, "generated": d_generated}
    d_result_id = d_rejected.value if d_choice.value == "none" else d_choice.value
    d_baseline_id = "generated" if d_result_id == "labels" else "labels"
    d_capture = mo.ui.button(
        label="Capture acquisition decision", kind="success", disabled=d_invalid,
        on_click=lambda _v: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={**d_upstream, "decision": d_choice.value, "tested_alternative": d_rejected.value},
            baseline=serialize_value(d_options[d_baseline_id] if d_choice.value != "none" else d_none),
            result=serialize_value(d_options[d_result_id]),
            alternatives=serialize_value(tuple(d_options.values())), decision=d_choice.value,
            chosen_result=serialize_value(d_none) if d_choice.value == "none" else None,
            result_role="rejected alternative" if d_choice.value == "none" else "selected candidate",
            upstream_inputs=d_upstream, model_key="acquisition_option",
        )),
    )
    e_capture = mo.ui.button(
        label="Capture population-change contrast", kind="success",
        disabled=(e_prediction.value is None
                  or e_rare_share.value == e_baseline["baseline_population_pct"]["rare"]),
        on_click=lambda _v: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value, inputs=e_upstream,
            baseline=serialize_value(e_baseline), result=serialize_value(e_result),
            upstream_inputs=e_upstream, model_key=e_result["model_key"],
        )),
    )
    return (
        a_capture,
        a_upstream,
        b_capture,
        b_upstream,
        c_capture,
        c_upstream,
        d_capture,
        d_upstream,
        e_capture,
        e_upstream,
    )


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#70411f);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin:30px 0 14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#ffedd5;max-width:780px}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    @media(max-width:520px){.pilot-head{border-radius:9px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""
    <section class="pilot-head">
      <div class="pilot-top"><span>VOLUME I · LAB 09</span><span>ABOUT 50–55 MIN</span></div>
      <h1>Selection That Pays</h1>
      <p>Which examples should we retain when coverage, selection overhead, and changed populations can reverse the apparent win?</p>
      <div class="pilot-meta">
        <div><b>Track</b><br>{profile['label']}</div>
        <div><b>Workload</b><br>{profile['workload']}</div>
        <div><b>Output</b><br>Data-selection policy memo</div>
      </div>
    </section>""")
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, css, track, header,
        mo.Html('<p class="pilot-note">All pools and quality outcomes are explicit teaching scenarios, not benchmark measurements. Calculation Notes state each comparison boundary.</p>'),
    ], gap=0.5).style({"padding-top": "32px"})
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS,
    a_baseline,
    a_capture,
    a_prediction,
    a_result,
    a_upstream,
    apply_plotly_theme,
    audit_evidence,
    b_capture,
    b_conclusion,
    b_prediction,
    b_rejected,
    b_results,
    b_retained,
    b_upstream,
    c_baseline,
    c_capture,
    c_decision,
    c_prediction,
    c_result,
    c_runs,
    c_scoring,
    c_upstream,
    carried_policy,
    d_budget,
    d_capture,
    d_choice,
    d_generated,
    d_labels,
    d_prediction,
    d_rejected,
    d_upstream,
    e_baseline,
    e_capture,
    e_prediction,
    e_rare_share,
    e_result,
    e_upstream,
    final_choice,
    final_rejected,
    final_risk,
    final_trigger,
    get_evidence,
    go,
    mo,
    profile,
    rationale,
    track_id,
):
    _captures = get_evidence()
    upstream = {
        "A": a_upstream, "B": b_upstream, "C": c_upstream,
        "D": d_upstream, "E": e_upstream,
    }
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=upstream, contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style(
            {"max-width": "100%", "overflow-x": "auto"}
        )

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(
                mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture this part."),
                kind="danger",
            )
        data = capture.to_dict()
        return mo.Html(
            f'<div class="saved"><b>Saved snapshot</b> · prediction: {data["prediction"]}'
            '<br><small>The report keeps this result even if live controls move.</small></div>'
        )

    def part_a():
        intro = mo.md(
            f"### A · When does more data stop being the best investment? (8 min)\n"
            f"Compare supplied learning outcomes for redundant and informative candidate pools of **{profile['workload']}**."
        )
        if a_prediction.value is None:
            return mo.vstack([intro, a_prediction])
        figure = go.Figure()
        for result, label, color in (
            (a_baseline, "Redundant", COLORS["OrangeLine"]),
            (a_result, "Informative", COLORS["BlueLine"]),
        ):
            figure.add_scatter(
                x=[point["retained_count"] for point in result["points"]],
                y=[point["quality_pct"] for point in result["points"]],
                mode="lines+markers", name=label, line={"color": color},
            )
        figure.update_layout(
            height=290, margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Representative examples retained",
            yaxis_title="Supplied quality outcome (%)", legend_orientation="h",
        )
        redundant_last = a_baseline["points"][-1]["marginal_quality_pp"]
        informative_last = a_result["points"][-1]["marginal_quality_pp"]
        return mo.vstack([
            intro, a_prediction, apply_plotly_theme(figure),
            mo.callout(mo.md(
                f"**Your prediction:** {a_prediction.value}. The final two-example increment adds "
                f"**{redundant_last:.1f} points** for the redundant pool and "
                f"**{informative_last:.1f} points** for the informative pool."
            ), kind="info"),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md(
                "Adjacent marginal value is the change between two supplied quality observations. "
                "The simulator does not infer quality from sample count."
            )}),
        ])

    def part_b():
        rows = []
        for policy_id, result in b_results.items():
            counts = result["cohort_counts"]
            rows.append({
                "Policy": policy_id.title(),
                "Retained": result["retained_count"],
                "Size": f"{result['retained_bytes'].to('megabyte').magnitude:.2f} MB",
                "Duplicates": result["duplicate_records"],
                "Common / rare / edge": f"{counts['common']} / {counts['rare']} / {counts['edge_case']}",
                "Quality": f"{result['weighted_quality_pct']:.1f}%",
                "Outcome": "PASS" if result["feasible"] else "FAIL · " + "; ".join(result["failures"]),
            })
        intro = mo.md(
            f"### B · Which examples can we safely remove? (10 min)\n"
            f"Hold retained count constant for **{profile['workload']}**. "
            "Inspect duplicate groups, cohort counts, and supplied outcomes before choosing."
        )
        if b_prediction.value is None:
            return mo.vstack([intro, b_retained, b_prediction])
        chosen = b_results.get(b_conclusion.value) if b_conclusion.value != "none" else None
        message = "No policy selected." if chosen is None else (
            "PASS" if chosen["feasible"] else "FAIL · " + "; ".join(chosen["failures"])
        )
        return mo.vstack([
            intro, b_prediction, b_retained, table(rows),
            mo.hstack([b_conclusion, b_rejected], widths="equal", wrap=True),
            mo.callout(mo.md(
                f"**Your prediction:** {b_prediction.value}. **Live decision:** "
                f"{b_conclusion.value or 'not chosen'}; {message} "
                "A no-feasible decision must name the policy you tested."
            ), kind="danger" if chosen is not None and not chosen["feasible"] else "info"),
            b_capture, saved("B"),
            mo.accordion({"Calculation Notes": mo.md(
                "Every policy retains the same number of explicit records. Duplicate counts come "
                "from repeated duplicate-group IDs; coverage comes from cohort membership; quality remains supplied evidence."
            )}),
        ])

    def part_c():
        intro = mo.md(
            f"### C · Does selection pay for itself? (10 min)\nCarry **{carried_policy}** "
            f"into a complete cost comparison for **{profile['workload']}**: extra read scan, scoring, selection, "
            "and repeated fixed-work training on an off-device development host."
        )
        if c_prediction.value is None:
            return mo.vstack([
                intro, mo.hstack([c_scoring, c_runs], widths="equal", wrap=True), c_prediction,
            ])
        figure = go.Figure([
            go.Bar(
                name="Full pool", x=["Total"],
                y=[c_baseline["reported_total"].to("second").magnitude],
                marker_color=COLORS["OrangeLine"],
            ),
            go.Bar(
                name="Selected subset", x=["Total"],
                y=[c_result["reported_total"].to("second").magnitude],
                marker_color=COLORS["BlueLine"],
            ),
        ])
        figure.update_layout(
            height=270, margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Elapsed time (s)", legend_orientation="h",
        )
        result_word = "saves time" if c_result["selection_pays"] else "does not yet save time"
        support_word = "supported" if c_result["decision_supported"] else "not supported"
        return mo.vstack([
            intro, c_prediction,
            mo.hstack([c_scoring, c_runs], widths="equal", wrap=True),
            apply_plotly_theme(figure),
            table([{
                "Selection scan": f"{c_result['read_time'].to('second').magnitude:.1f} s",
                "Scoring": f"{c_result['scoring_time'].to('second').magnitude:.1f} s",
                "Selection": f"{c_result['selection_time'].to('second').magnitude:.1f} s",
                "First profitable run": c_result["first_profitable_runs"],
            }]),
            c_decision,
            mo.callout(mo.md(
                f"**Your prediction:** {c_prediction.value}. At **{c_runs.value} runs**, "
                f"selection {result_word}; the combined time-and-evidence decision is **{support_word}**."
            ), kind="success" if c_result["decision_supported"] else "danger"),
            c_capture, saved("C"),
            mo.accordion({"Calculation Notes": mo.md(
                "Total selected-path time = one extra read scan + scoring + selection + repeated subset training. "
                "For device tracks, selection scanning, scoring, and training occur on an off-device development host "
                f"({profile['development_peak'].to('teraflop / second').magnitude:.1f} TFLOPS peak), not on the target deployment device. "
                "The full-pool path includes repeated training. Training I/O is excluded, and fixed epochs do not "
                "establish time to common quality."
            )}),
        ])

    def part_d():
        rows = [{
            "Option": result["label"],
            "Created / validated": f"{result['created_examples']} / {result['validated_examples']}",
            "Cost": f"${result['total_cost'].to('USD').magnitude:,.0f}",
            "Turnaround": f"{result['turnaround'].to('hour').magnitude:.0f} h",
            "Rare outcome": f"{result['cohort_outcomes_pct']['rare']:.0f}%",
            "Affordable": "YES" if result["affordable"] else "NO",
        } for result in (d_labels, d_generated)]
        intro = mo.md(
            f"### D · Should the next dollar buy labels or generated examples? (10 min)\n"
            f"Compare explicit creation, validation, turnaround, and supplied cohort outcomes for **{profile['workload']}**."
        )
        if d_prediction.value is None:
            return mo.vstack([intro, d_budget, d_prediction])
        return mo.vstack([
            intro, d_prediction, d_budget, table(rows),
            mo.hstack([d_choice, d_rejected], widths="equal", wrap=True),
            mo.callout(mo.md(
                f"**Your prediction:** {d_prediction.value}. Labels yield "
                f"**{d_labels['cohort_outcomes_pct']['rare']:.0f}%** on the rare cohort; generated examples yield "
                f"**{d_generated['cohort_outcomes_pct']['rare']:.0f}%**. Affordability is shown separately from quality."
            ), kind="info"),
            d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md(
                "Total package cost adds creation and validation. Each outcome belongs only to the displayed "
                "fixed package; the simulator does not extrapolate a universal generated-data quality curve."
            )}),
        ])

    def part_e():
        intro = mo.md(
            f"### E · Does selection survive a changed population? (8 min)\n"
            f"Reweight the fixed **{carried_policy}** cohort outcomes for **{profile['workload']}** before reselection."
        )
        if e_prediction.value is None:
            return mo.vstack([intro, e_rare_share, e_prediction])
        rows = [{
            "Population": "Baseline",
            "Rare share": f"{e_baseline['changed_population_pct']['rare']:.0f}%",
            "Weighted quality": f"{e_baseline['changed_quality_pct']:.1f}%",
            "Underrepresented": ", ".join(e_baseline["underrepresented_cohorts"]) or "none",
            "Outcome": "SUPPORTED" if e_baseline["still_supported"] else "FAIL",
        }, {
            "Population": "Changed",
            "Rare share": f"{e_result['changed_population_pct']['rare']:.0f}%",
            "Weighted quality": f"{e_result['changed_quality_pct']:.1f}%",
            "Underrepresented": ", ".join(e_result["underrepresented_cohorts"]) or "none",
            "Outcome": "SUPPORTED" if e_result["still_supported"] else "FAIL · " + "; ".join(e_result["failures"]),
        }]
        return mo.vstack([
            intro, e_rare_share, e_prediction, table(rows),
            mo.callout(mo.md(
                f"**Your prediction:** {e_prediction.value}. Quality changes by "
                f"**{e_result['quality_change_pp']:.1f} points**. The selected share now trails "
                f"the changed population for **{', '.join(e_result['underrepresented_cohorts']) or 'no growing cohort'}**."
            ), kind="success" if e_result["still_supported"] else "danger"),
            e_capture, saved("E"),
            mo.accordion({"Calculation Notes": mo.md(
                "The changed population reweights fixed cohort outcomes. It does not improve or degrade a cohort rule. "
                "Representation fails only when a cohort grows beyond its selected share."
            )}),
        ])

    def synthesis():
        rows = []
        for part in "ABCDE":
            capture = _captures.get(part)
            rows.append({
                "Part": part,
                "Prediction": capture.to_dict()["prediction"] if capture else "—",
                "Evidence": (
                    "CURRENT"
                    if capture and part not in audit.stale and (part, part) not in audit.identical_pairs
                    else ("STALE" if capture else "MISSING")
                ),
            })
        saved_b = _captures["B"].to_dict() if "B" in _captures else None
        saved_decision = saved_b["decision"] if saved_b else None
        choice_rule = final_choice.value == saved_decision
        rejected_rule = (
            final_rejected.value is not None
            and saved_b is not None
            and final_choice.value == "none"
            and final_rejected.value == saved_b["inputs"]["tested_alternative"]
        )
        if final_choice.value not in (None, "none"):
            rejected_rule = (
                saved_b is not None
                and final_rejected.value == saved_b["baseline"]["inputs"]["policy_id"]
            )
        complete = (
            audit.complete
            and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
            and bool(rationale.value.strip()) and choice_rule and rejected_rule
        )
        return mo.vstack([
            mo.md(
                "### Synthesis · Defend a data-selection policy (5 min)\n"
                "Choose one saved policy, quantify a rejected alternative, state the residual limitation, "
                "and name the trigger that forces reevaluation."
            ),
            table(rows),
            mo.callout(mo.md(
                "Saved snapshots remain fixed while live controls move. Recapture stale evidence before generating the report."
            ), kind="info"),
            mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
            mo.hstack([final_trigger, final_risk], widths="equal", wrap=True),
            rationale,
            mo.callout(mo.md(
                "**Ready for the local report.**" if complete else
                "Complete five current contrasts, match the saved Part B decision, compare a tested alternative, and add the rationale."
            ), kind="success" if complete else "warn"),
        ])

    tabs = mo.ui.tabs({
        "Part A": part_a(), "Part B": part_b(), "Part C": part_c(),
        "Part D": part_d(), "Part E": part_e(), "Synthesis": synthesis(),
    })
    tabs
    return (audit,)


@app.cell
def _(
    audit,
    build_lab_report,
    final_choice,
    final_rejected,
    final_risk,
    final_trigger,
    get_evidence,
    get_lab_metadata,
    mo,
    profile,
    rationale,
    report_export_panel,
    track_id,
):
    _captures = get_evidence()
    _saved_b = _captures["B"].to_dict() if "B" in _captures else None
    _saved_decision = _saved_b["decision"] if _saved_b else None
    _choice_rule = final_choice.value == _saved_decision
    _rejected_rule = (
        final_rejected.value is not None and _saved_b is not None
        and final_choice.value == "none"
        and final_rejected.value == _saved_b["inputs"]["tested_alternative"]
    )
    if final_choice.value not in (None, "none"):
        _rejected_rule = (
            _saved_b is not None
            and final_rejected.value == _saved_b["baseline"]["inputs"]["policy_id"]
        )
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (
            final_choice, final_rejected, final_trigger, final_risk,
        ))
        and bool(rationale.value.strip()) and _choice_rule and _rejected_rule
    )
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_09_data_selection.py"),
        track=track_id,
        scenario=profile["workload"],
        learning_objectives=[
            "Distinguish redundant volume from informative coverage",
            "Compare equal-count removal policies with explicit records",
            "Test selection amortization without treating time as quality",
            "Reevaluate a selected cohort under population change",
        ],
        predictions={part: snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {
            "baseline": snapshots[part]["baseline"],
            "result": snapshots[part]["result"],
            "alternatives": snapshots[part]["alternatives"],
        } for part in "ABCDE"},
        binding_constraints={
            "selection": snapshots["B"]["result"]["failures"],
            "population": snapshots["E"]["result"]["failures"],
        },
        decisions={
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "reevaluation_trigger": final_trigger.value,
        },
        final_decision={
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "rationale": rationale.value,
        },
        big_takeaways=[
            "Equal sample counts can preserve very different cohort coverage.",
            "Selection overhead must be amortized across actual reuse.",
            "A changed population can invalidate a previously supported subset.",
        ],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={
            "track": track_id, "captures": snapshots,
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "reevaluation_trigger": final_trigger.value,
            "residual_risk": final_risk.value,
        },
        source_trace={
            "scenario": "Explicit illustrative candidate pools and supplied outcomes.",
            "calculations": "Deterministic data-selection scenario calculations.",
        },
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return


@app.cell
async def _(
    MODEL_ID,
    audit,
    final_choice,
    final_rejected,
    final_risk,
    final_trigger,
    get_evidence,
    ledger,
    mo,
    rationale,
    track_id,
):
    _captures = get_evidence()
    _saved_b = _captures["B"].to_dict() if "B" in _captures else None
    _saved_decision = _saved_b["decision"] if _saved_b else None
    _choice_rule = final_choice.value == _saved_decision
    _rejected_rule = (
        final_rejected.value is not None and _saved_b is not None
        and final_choice.value == "none"
        and final_rejected.value == _saved_b["inputs"]["tested_alternative"]
    )
    if final_choice.value not in (None, "none"):
        _rejected_rule = (
            _saved_b is not None
            and final_rejected.value == _saved_b["baseline"]["inputs"]["policy_id"]
        )
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (
            final_choice, final_rejected, final_trigger, final_risk,
        ))
        and bool(rationale.value.strip()) and _choice_rule and _rejected_rule
    )
    _saved = False
    _save_error = None
    if _ready:
        try:
            ledger.save(chapter=9, design={
                "schema_version": 1,
                "lab_id": "v1_09",
                "track_id": track_id,
                "model_id": MODEL_ID,
                "evidence": {part: capture.to_dict() for part, capture in get_evidence().items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value,
                "rationale": rationale.value,
            })
            await ledger.flush()
            _saved = True
        except Exception:
            _save_error = True
    if _save_error:
        status = "SAVE FAILED · REPORT STILL AVAILABLE"
    elif _saved:
        status = "SAVED"
    else:
        status = "EVIDENCE IN PROGRESS"
    mo.Html(
        f'<div class="lab-hud"><span class="hud-label">LAB</span>'
        f'<span class="hud-value">09 · Selection That Pays</span>'
        f'<span class="hud-label"> · STATUS: </span><span class="hud-active">{status}</span></div>'
    )
    return


if __name__ == "__main__":
    app.run()
