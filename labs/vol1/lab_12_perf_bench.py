import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 12: Benchmark Claims That Survive · MLSysBook")


@app.cell
async def _():
    import sys
    from dataclasses import replace
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
    from mlsysim.engine.v1_12_experiments import (
        TRACKS, analyze_repeats, audit_protocols, compare_quality_slices,
        compare_scopes, compare_sustained, default_protocol, to_jsonable,
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
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, Q_, TRACKS, analyze_repeats,
        apply_plotly_theme, audit_evidence, audit_protocols, build_lab_report,
        capture_evidence, compare_quality_slices, compare_scopes,
        compare_sustained, default_protocol, get_lab_metadata, go, ledger, mo,
        replace, report_export_panel, to_jsonable,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(mo, set_evidence):
    track = mo.ui.dropdown(
        {"TinyML": "tinyml", "Mobile": "mobile", "Edge": "edge", "Cloud": "cloud"},
        value="TinyML", label="Deployment track", on_change=lambda _value: set_evidence({}),
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
    a_speedup = mo.ui.slider(1.2, 4.0, value=profile.candidate_kernel_speedup, step=0.2, label="Claimed kernel speedup")
    b_warmup = mo.ui.slider(0, 16, value=8, step=2, label="Discarded warmup iterations")
    b_samples = mo.ui.slider(10, 100, value=40, step=10, label="Measured iterations per run")
    b_repeats = mo.ui.slider(2, 8, value=5, step=1, label="Repeated runs")
    c_duration = mo.ui.dropdown(
        {"2 minutes": 120, "5 minutes": 300, "10 minutes": 600},
        value="10 minutes", label="Sustained duration",
    )
    c_stress = mo.ui.slider(1.0, 4.0, value=2.0, step=0.5, label="Variability and thermal stress")
    d_mode = mo.ui.dropdown(
        {"Quality-preserving path": "quality_preserving", "Aggressive path": "aggressive"},
        value="Aggressive path", label="Efficiency candidate",
    )
    d_slice_floor = mo.ui.slider(30, 90, value=60, step=5, label="Worst-slice quality floor (%)")
    d_choice = mo.ui.radio(
        {"Carry quality-preserving path": "quality_preserving", "Carry aggressive path": "aggressive", "Hold for more evidence / no reportable candidate": "none"},
        label="Candidate decision",
    )
    d_rejected = mo.ui.radio(
        {"Quality-preserving path": "quality_preserving", "Aggressive path": "aggressive"},
        label="Tested alternative",
    )
    e_mismatch = mo.ui.dropdown(
        {"Different measurement scope": "scope", "No warmup discard": "warmup_iterations", "Different quality fixture": "quality_fixture_id", "Different power boundary": "power_boundary"},
        value="Different measurement scope", label="Protocol mismatch to inspect",
    )
    e_decision = mo.ui.radio(
        {"Repair, then include": "repair", "Exclude the mismatched result": "exclude"},
        label="Reporting decision",
    )
    return (
        a_speedup, b_repeats, b_samples, b_warmup, c_duration, c_stress,
        d_choice, d_mode, d_rejected, d_slice_floor, e_decision, e_mismatch,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Almost the full kernel gain": "near-kernel", "A smaller but real gain": "diluted", "No whole-path gain": "none"},
        label="How much of the kernel gain reaches the whole request?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Warmup changes the reported center": "warmup", "Only sample count matters": "samples", "Every protocol gives the same result": "same"},
        label="Which protocol choice will matter most?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"Candidate wins latency and energy": "both", "Candidate is faster but costs more energy": "energy", "Candidate misses more deadlines": "deadline"},
        label="What survives a sustained deployment trace?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Both slices preserve quality": "both", "Ordinary passes; difficult fails": "slice-loss", "Overall accuracy alone decides": "aggregate"},
        label="What will the slice evidence show?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"Include as-is": "include", "Qualify but keep the headline": "qualify", "Exclude until repaired": "exclude"},
        label="Can the mismatched comparison support the headline?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Report quality-preserving candidate": "quality_preserving", "Report aggressive candidate": "aggressive", "Hold for more evidence / report no candidate": "none"},
        label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {"Quality-preserving candidate": "quality_preserving", "Aggressive candidate": "aggressive"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"Worst-slice quality crosses its floor": "slice-quality", "Sustained deadline misses increase": "deadline", "Energy per request exceeds the budget": "energy", "Benchmark protocol changes": "protocol"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"The seeded workload may miss deployment cases": "coverage", "Longer operation may reveal more drift": "drift", "Power boundary may omit components": "power-boundary"},
        label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Evidence-based rationale",
        placeholder="Use saved scope, repeat, sustained, slice, and protocol evidence to defend the choice.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    Q_, a_speedup, analyze_repeats, audit_protocols, b_repeats, b_samples,
    b_warmup, c_duration, c_stress, compare_quality_slices, compare_scopes,
    compare_sustained, d_mode, d_slice_floor, default_protocol, e_mismatch,
    replace, track_id,
):
    a_kernel_result = compare_scopes(
        track_id, kernel_speedup=a_speedup.value, reported_scope="kernel"
    )
    a_result = compare_scopes(
        track_id, kernel_speedup=a_speedup.value, reported_scope="whole_path"
    )
    b_baseline = analyze_repeats(track_id, warmup_iterations=0, measured_iterations=10, repeat_count=2)
    b_result = analyze_repeats(
        track_id, warmup_iterations=b_warmup.value,
        measured_iterations=b_samples.value, repeat_count=b_repeats.value,
    )
    c_baseline = compare_sustained(track_id, duration=Q_(30, "second"), stress_scale=c_stress.value)
    c_result = compare_sustained(track_id, duration=Q_(c_duration.value, "second"), stress_scale=c_stress.value)
    d_results = {
        mode: compare_quality_slices(track_id, candidate_mode=mode, worst_slice_floor_pct=d_slice_floor.value)
        for mode in ("quality_preserving", "aggressive")
    }
    d_baseline = compare_quality_slices(
        track_id, candidate_mode="baseline", worst_slice_floor_pct=d_slice_floor.value
    )
    d_result = d_results[d_mode.value]
    e_reference = default_protocol(track_id)
    e_changes = {
        "scope": {"scope": "isolated kernel"},
        "warmup_iterations": {"warmup_iterations": 0},
        "quality_fixture_id": {"quality_fixture_id": f"{track_id}-unmatched-quality"},
        "power_boundary": {"power_boundary": "accelerator only"},
    }
    e_candidate = replace(e_reference, **e_changes[e_mismatch.value])
    e_mismatched = audit_protocols(e_reference, e_candidate)
    e_repaired = audit_protocols(e_reference, e_mismatched["repair"])
    return a_kernel_result, a_result, b_baseline, b_result, c_baseline, c_result, d_baseline, d_result, d_results, e_mismatched, e_repaired


@app.cell
def _(
    a_kernel_result, a_prediction, a_result, a_speedup, b_baseline, b_prediction, b_repeats,
    b_result, b_samples, b_warmup, c_baseline, c_duration, c_prediction,
    c_result, c_stress, capture_evidence, d_baseline, d_choice, d_mode, d_prediction,
    d_rejected, d_results, d_slice_floor, e_decision, e_mismatch,
    e_mismatched, e_prediction, e_repaired, mo, set_evidence, to_jsonable,
    track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = {"kernel_speedup": a_speedup.value}
    b_upstream = {"warmup_iterations": b_warmup.value, "measured_iterations": b_samples.value, "repeat_count": b_repeats.value}
    c_upstream = {"duration_seconds": c_duration.value, "stress_scale": c_stress.value}
    d_upstream = {
        "candidate_mode": d_mode.value, "worst_slice_floor_pct": d_slice_floor.value,
        "choice": d_choice.value, "rejected": d_rejected.value,
    }
    e_upstream = {"mismatch": e_mismatch.value, "decision": e_decision.value}

    a_kernel = {
        "inputs": a_kernel_result["inputs"],
        "sample": to_jsonable(a_kernel_result["reported_summary"]),
        "speedup": to_jsonable(a_kernel_result["reported_speedup"]),
    }
    a_path = {
        "inputs": a_result["inputs"],
        "sample": to_jsonable(a_result["reported_summary"]),
        "speedup": to_jsonable(a_result["reported_speedup"]),
    }
    a_capture = mo.ui.button(
        label="Capture scope contrast", kind="success", disabled=a_prediction.value is None,
        on_click=lambda _value: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs=a_upstream, baseline=a_kernel, result=a_path,
            upstream_inputs=a_upstream, decision="whole-path",
            model_key="v1_12_experiments.compare_scopes",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture measurement protocol", kind="success",
        disabled=b_prediction.value is None or b_result["inputs"] == b_baseline["inputs"],
        on_click=lambda _value: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs=b_upstream, baseline=to_jsonable(b_baseline), result=to_jsonable(b_result),
            upstream_inputs=b_upstream, decision="disclose-repeated-runs",
            model_key="v1_12_experiments.analyze_repeats",
        )),
    )
    c_capture = mo.ui.button(
        label="Capture sustained comparison", kind="success", disabled=c_prediction.value is None,
        on_click=lambda _value: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs=c_upstream, baseline=to_jsonable(c_baseline), result=to_jsonable(c_result),
            upstream_inputs=c_upstream, decision="sustained-evidence",
            model_key="v1_12_experiments.compare_sustained",
        )),
    )
    d_invalid = (
        d_prediction.value is None or d_choice.value is None or d_rejected.value is None
        or (d_choice.value != "none" and d_choice.value == d_rejected.value)
    )
    d_selected_mode = d_rejected.value if d_choice.value == "none" else d_choice.value
    d_capture = mo.ui.button(
        label="Capture slice decision", kind="success", disabled=d_invalid,
        on_click=lambda _value: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={**d_upstream, "evaluated_mode": d_selected_mode},
            baseline=to_jsonable(d_baseline),
            result=to_jsonable(d_results[d_selected_mode]),
            upstream_inputs=d_upstream,
            alternatives=tuple(to_jsonable(d_results[mode]) for mode in d_results),
            decision={"choice": d_choice.value, "rejected": d_rejected.value},
            model_key="v1_12_experiments.compare_quality_slices",
            chosen_result=to_jsonable(d_baseline) if d_choice.value == "none" else to_jsonable(d_results[d_selected_mode]),
            result_role="rejected alternative" if d_choice.value == "none" else "chosen candidate",
        )),
    )
    e_capture = mo.ui.button(
        label="Capture protocol decision", kind="success",
        disabled=e_prediction.value is None or e_decision.value is None,
        on_click=lambda _value: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs=e_upstream, baseline=to_jsonable(e_mismatched), result=to_jsonable(e_repaired),
            upstream_inputs=e_upstream, decision=e_decision.value,
            model_key="v1_12_experiments.audit_protocols",
        )),
    )
    return (
        a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream,
        d_capture, d_upstream, e_capture, e_upstream,
    )


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, mo, profile, track):
    css = mo.Html("""
    <style>
    .bench-head{background:linear-gradient(135deg,#101827,#194f72);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin:30px 0 14px}
    .bench-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .bench-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.bench-head p{color:#dbeafe;max-width:760px}
    .bench-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.bench-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}.table-wrap{max-width:100%;overflow-x:auto}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    @media(max-width:520px){.bench-head{border-radius:9px}.bench-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="bench-head"><div class="bench-top"><span>VOLUME I · LAB 12</span><span>ABOUT 50–55 MIN</span></div><h1>Benchmark Claims That Survive</h1><p>Can a performance gain survive fair, credible, representative measurement while preserving useful behavior?</p><div class="bench-meta"><div><b>Track</b><br>{profile.display_name}</div><div><b>Workload</b><br>{profile.workload_id}</div><div><b>Output</b><br>Auditable benchmark recommendation</div></div></section>""")
    mo.vstack([
        track, css, LAB_CSS, ACADEMIC_LAB_CSS, header,
        mo.md("A fast component is only a hypothesis. Test scope, measurement amount, sustained behavior, quality slices, and protocol parity before deciding what can be reported. These repeatable traces are teaching scenarios, not measurements of a named product."),
    ])
    return


@app.cell
def _(
    COLORS, a_capture, a_prediction, a_result, a_speedup, a_upstream,
    apply_plotly_theme, audit_evidence, b_baseline, b_capture, b_prediction,
    b_repeats, b_result, b_samples, b_upstream, b_warmup, c_baseline,
    c_capture, c_duration, c_prediction, c_result, c_stress, c_upstream,
    d_capture, d_choice, d_mode, d_prediction, d_rejected, d_result,
    d_results, d_slice_floor, d_upstream, e_capture, e_decision, e_mismatch,
    e_mismatched, e_prediction, e_repaired, e_upstream, final_choice,
    final_rejected, final_risk, final_trigger, get_evidence, go, mo, rationale,
    track_id,
):
    def ms(quantity):
        return quantity.m_as("ms")

    def ratio(value):
        return value.magnitude if hasattr(value, "magnitude") else value

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style(
            {"max-width": "100%", "overflow-x": "auto"}
        )

    def saved(part):
        capture = get_evidence().get(part)
        return mo.Html(f'<div class="saved">Saved evidence for Part {part}.</div>') if capture else mo.md("")

    captures = get_evidence()
    audit = audit_evidence(
        captures, track=track_id, required_parts=tuple("ABCDE"),
        contrast_required_parts=tuple("ABCDE"),
        per_part_upstream_inputs={"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream, "E": e_upstream},
    )

    def part_a():
        intro = mo.md("### A · Does local speedup improve the user path? (10 min)\nPredict how much of an isolated kernel gain survives preprocessing, transfer, postprocessing, and integration overhead.")
        if a_prediction.value is None:
            return mo.vstack([intro, a_prediction])
        figure = go.Figure(go.Bar(
            x=["Isolated kernel", "Whole request path"],
            y=[ratio(a_result["kernel_speedup"]), ratio(a_result["whole_path_speedup"])],
            marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]],
            texttemplate="%{y:.2f}×", textposition="outside",
        ))
        figure.update_layout(yaxis_title="Median speedup (×)", showlegend=False)
        apply_plotly_theme(figure)
        return mo.vstack([
            intro, a_prediction, a_speedup, figure,
            mo.callout(mo.md(f"**Prediction:** {a_prediction.value}. The paired trace shows **{ratio(a_result['kernel_speedup']):.2f}×** at the kernel boundary and **{ratio(a_result['whole_path_speedup']):.2f}×** across the request."), kind="info"),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md("Each request uses the same seeded disturbance in both systems. Whole-path time adds preprocessing, kernel, transfer, and postprocessing; the candidate also pays disclosed integration overhead.")}),
        ])

    def part_b():
        intro = mo.md("### B · How much measurement supports the claim? (10 min)\nCompare a short run that includes warmup with repeated runs that disclose warmup, run length, drift, and run-to-run variability.")
        if b_prediction.value is None:
            return mo.vstack([intro, b_prediction])
        first_trace = b_result["traces"][0]
        figure = go.Figure(go.Scatter(
            x=list(range(len(first_trace))), y=[ms(value) for value in first_trace],
            mode="lines+markers", name="Run 1", line={"color": COLORS["BlueLine"]},
        ))
        figure.add_vline(x=b_warmup.value - 0.5, line_dash="dash", annotation_text="measured region")
        figure.update_layout(xaxis_title="Iteration", yaxis_title="Whole-path latency (ms)")
        apply_plotly_theme(figure)
        rows = [
            {"Protocol": "Short, cold-included", "Warmup": 0, "Samples/run": 10, "Runs": 2, "Median": f"{ms(b_baseline['median_of_run_medians']):.2f} ms", "Run CV": f"{b_baseline['run_to_run_cv_pct']:.2f}%"},
            {"Protocol": "Selected", "Warmup": b_warmup.value, "Samples/run": b_samples.value, "Runs": b_repeats.value, "Median": f"{ms(b_result['median_of_run_medians']):.2f} ms", "Run CV": f"{b_result['run_to_run_cv_pct']:.2f}%"},
        ]
        return mo.vstack([
            intro, b_prediction, mo.hstack([b_warmup, b_samples, b_repeats], widths="equal", wrap=True),
            figure, table(rows),
            mo.callout(mo.md(f"**Prediction:** {b_prediction.value}. {b_result['interval_note']}"), kind="info"),
            b_capture, saved("B"),
            mo.accordion({"Calculation Notes": mo.md("Every run has disclosed warmup decay, run offset, iteration noise, and within-run drift. The instrument reports sample medians, p95, p99, and variability across run medians.")}),
        ])

    def part_c():
        intro = mo.md("### C · Does the winner survive deployment? (10 min)\nRun the same paired systems long enough for variability and candidate thermal drift to appear. Compare tail latency, deadline misses, and energy per request.")
        if c_prediction.value is None:
            return mo.vstack([intro, c_prediction])
        figure = go.Figure()
        figure.add_bar(name="Baseline", x=["p99 latency"], y=[ms(c_result["baseline"]["p99"])], marker_color=COLORS["BlueLine"])
        figure.add_bar(name="Candidate", x=["p99 latency"], y=[ms(c_result["candidate"]["p99"])], marker_color=COLORS["OrangeLine"])
        figure.add_hline(y=ms(c_result["deadline"]), line_dash="dash", annotation_text="deadline")
        figure.update_layout(barmode="group", yaxis_title="Latency (ms)")
        apply_plotly_theme(figure)
        rows = [
            {"Trace": "30-second check", "Candidate p99": f"{ms(c_baseline['candidate']['p99']):.2f} ms", "Candidate misses": f"{c_baseline['candidate_deadline_miss_pct']:.1f}%", "Candidate energy": f"{c_baseline['candidate_energy_per_request'].m_as('J'):.4f} J/request"},
            {"Trace": f"{c_duration.value}-second sustained", "Candidate p99": f"{ms(c_result['candidate']['p99']):.2f} ms", "Candidate misses": f"{c_result['candidate_deadline_miss_pct']:.1f}%", "Candidate energy": f"{c_result['candidate_energy_per_request'].m_as('J'):.4f} J/request"},
        ]
        consequence = "The candidate uses more median energy per request." if c_result["candidate_uses_more_energy"] else "The candidate uses less median energy per request in this scenario."
        return mo.vstack([
            intro, c_prediction, mo.hstack([c_duration, c_stress], widths="equal", wrap=True),
            figure, table(rows),
            mo.callout(mo.md(f"**Prediction:** {c_prediction.value}. Sustained candidate deadline misses are **{c_result['candidate_deadline_miss_pct']:.1f}%** versus **{c_result['baseline_deadline_miss_pct']:.1f}%** for baseline. {consequence}"), kind="warn" if c_result["candidate_deadline_misses"] > c_result["baseline_deadline_misses"] else "info"),
            c_capture, saved("C"),
            mo.accordion({"Calculation Notes": mo.md("The candidate accelerates only the kernel, adds integration overhead, and draws its disclosed request power. Energy per request is power × complete request time. Tail metrics come from complete simulated request durations.")}),
        ])

    def part_d():
        intro = mo.md("### D · Did efficiency preserve useful behavior? (10 min)\nUse the same fixed labels and paired candidate scores for ordinary and difficult slices. Aggregate and worst-slice floors answer different questions.")
        if d_prediction.value is None:
            return mo.vstack([intro, d_prediction])
        figure = go.Figure()
        figure.add_bar(name="Baseline", x=["Ordinary", "Difficult"], y=[d_result["slices"][name]["baseline_accuracy_pct"] for name in ("ordinary", "difficult")], marker_color=COLORS["BlueLine"])
        figure.add_bar(name="Candidate", x=["Ordinary", "Difficult"], y=[d_result["slices"][name]["candidate_accuracy_pct"] for name in ("ordinary", "difficult")], marker_color=COLORS["OrangeLine"])
        figure.add_hline(y=d_slice_floor.value, line_dash="dash", annotation_text="worst-slice floor")
        figure.update_layout(barmode="group", yaxis_title="Observed accuracy (%)", yaxis_range=[0, 105])
        apply_plotly_theme(figure)
        rows = [{
            "Candidate": mode.replace("_", " ").title(),
            "Overall": f"{result['candidate_overall_accuracy_pct']:.1f}%",
            "Worst slice": f"{result['candidate_worst_slice_accuracy_pct']:.1f}%",
            f"Aggregate floor (≥{d_result['quality_floor_pct']:.0f}%)": "PASS" if result["candidate_aggregate_eligible"] else "FAIL",
            "Slice floor": "PASS" if result["candidate_worst_slice_eligible"] else "FAIL",
        } for mode, result in d_results.items()]
        decision_invalid = d_choice.value is not None and d_choice.value != "none" and d_choice.value == d_rejected.value
        return mo.vstack([
            intro, d_prediction, mo.hstack([d_mode, d_slice_floor], widths="equal", wrap=True),
            figure, table(rows), mo.hstack([d_choice, d_rejected], widths="equal", wrap=True),
            mo.callout(mo.md(f"**Prediction:** {d_prediction.value}. The selected candidate's observed overall result is **{d_result['candidate_overall_accuracy_pct']:.1f}%**; its difficult slice is **{d_result['slices']['difficult']['candidate_accuracy_pct']:.1f}%**."), kind="danger" if not d_result["candidate_worst_slice_eligible"] else "success"),
            mo.callout(mo.md("Choose a different tested alternative." if decision_invalid else "Your decision will carry into synthesis."), kind="warn" if decision_invalid else "info"),
            d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md("Accuracy comes from fixed binary labels and scores. The aggregate floor changes eligibility only; it does not change predictions. The worst-slice floor is evaluated separately.")}),
        ])

    def part_e():
        intro = mo.md("### E · Which comparison can we honestly report? (7 min)\nIntroduce one protocol mismatch, inspect the exact field, and decide whether to exclude the result or repair the protocol before inclusion.")
        if e_prediction.value is None:
            return mo.vstack([intro, e_prediction])
        rows = [{"Field": item["field"], "Reference": item["reference"], "Candidate": item["candidate"], "Effect": "Exclude from headline comparison"} for item in e_mismatched["mismatches"]]
        status_rows = [
            {"Protocol": "Mismatched", "Comparable": "YES" if e_mismatched["comparable"] else "NO", "Headline": "INCLUDE" if e_mismatched["include_in_headline_comparison"] else "EXCLUDE"},
            {"Protocol": "Repaired", "Comparable": "YES" if e_repaired["comparable"] else "NO", "Headline": "INCLUDE" if e_repaired["include_in_headline_comparison"] else "EXCLUDE"},
        ]
        return mo.vstack([
            intro, e_prediction, e_mismatch, table(rows), table(status_rows), e_decision,
            mo.callout(mo.md(f"**Prediction:** {e_prediction.value}. The mismatch changes inclusion directly; there is no validity score. Repaired status: **{e_repaired['claim_status']}**."), kind="info"),
            e_capture, saved("E"),
            mo.accordion({"Calculation Notes": mo.md("The audit compares workload, scope, warmup, measured iterations, repeats, trace identity, quality fixture and floor, and power boundary field by field.")}),
        ])

    def synthesis():
        rows = []
        for part in "ABCDE":
            capture = captures.get(part)
            state = "MISSING"
            if capture:
                state = "STALE" if part in audit.stale or (part, part) in audit.identical_pairs else "CURRENT"
            rows.append({"Part": part, "Locked prediction": capture.to_dict()["prediction"] if capture else "—", "Evidence": state})
        d_saved = captures["D"].to_dict()["decision"] if "D" in captures else {}
        ready = (
            audit.complete and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
            and bool(rationale.value.strip()) and final_choice.value == d_saved.get("choice")
            and final_rejected.value == d_saved.get("rejected")
            and (final_choice.value == "none" or final_choice.value != final_rejected.value)
        )
        return mo.vstack([
            mo.md("### Synthesis · Defend one reportable claim (5 min)\nChoose a candidate or conclude that none is reportable. Quantify the tested alternative, name the remaining limitation, and set a trigger that would reopen the decision."),
            table(rows), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
            mo.hstack([final_risk, final_trigger], widths="equal", wrap=True), rationale,
            mo.callout(mo.md("**Ready for the local report.**" if ready else "Capture five current contrasts and match synthesis to the saved Part D candidate decision."), kind="success" if ready else "warn"),
        ])

    mo.ui.tabs({"Part A": part_a(), "Part B": part_b(), "Part C": part_c(), "Part D": part_d(), "Part E": part_e(), "Synthesis": synthesis()})
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures = get_evidence()
    _d_saved = _captures["D"].to_dict()["decision"] if "D" in _captures else {}
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip())
        and final_choice.value == _d_saved.get("choice")
        and final_rejected.value == _d_saved.get("rejected")
        and (final_choice.value == "none" or final_choice.value != final_rejected.value)
    )
    mo.stop(not _ready)
    _snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_12_perf_bench.py"),
        track=track_id, scenario=profile.workload_id,
        learning_objectives=[
            "Distinguish component speedup from whole-path benefit",
            "Design repeated measurements that disclose warmup and drift",
            "Require sustained, slice, and protocol evidence before reporting a gain",
        ],
        predictions={part: snapshot["prediction"] for part, snapshot in _snapshots.items()},
        knob_settings={part: snapshot["inputs"] for part, snapshot in _snapshots.items()},
        evidence_summary={part: {
            "baseline": snapshot["baseline"], "result": snapshot["result"],
            "result_role": snapshot["result_role"],
            "chosen_result": snapshot["chosen_result"],
            "alternatives": snapshot["alternatives"],
        } for part, snapshot in _snapshots.items()},
        binding_constraints={
            "quality": (_snapshots["D"]["chosen_result"] or _snapshots["D"]["result"]).get("candidate_worst_slice_eligible"),
            "protocol": _snapshots["E"]["baseline"].get("mismatches"),
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
            "A kernel gain is bounded by unchanged whole-path work.",
            "Warmup and drift require disclosed repeated-run evidence.",
            "A claim must preserve difficult-slice behavior and protocol parity.",
        ],
        reflections={"rationale": rationale.value, "reevaluation_trigger": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={
            "track": track_id, "captures": _snapshots,
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "reevaluation_trigger": final_trigger.value,
            "residual_risk": final_risk.value,
        },
        source_trace={"scenario": "Disclosed illustrative seeded fixture", "calculations": "MLSysIM"},
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _d_saved = _captures["D"].to_dict()["decision"] if "D" in _captures else {}
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip())
        and final_choice.value == _d_saved.get("choice")
        and final_rejected.value == _d_saved.get("rejected")
        and (final_choice.value == "none" or final_choice.value != final_rejected.value)
    )
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
        try:
            ledger.save(chapter=12, design={
                "schema_version": 1,
                "lab_id": "v1_12",
                "track_id": track_id,
                "model_id": "v1_12_experiments",
                "evidence": {part: capture.to_dict() for part, capture in _captures.items()},
                "recommendation": final_choice.value,
                "rejected_alternative": final_rejected.value,
                "reevaluation_trigger": final_trigger.value,
                "residual_risk": final_risk.value,
                "rationale": rationale.value,
            })
            await ledger.flush()
        except Exception:
            _status = "LOCAL SAVE FAILED · DOWNLOAD THE REPORT TO KEEP YOUR EVIDENCE"
        else:
            _status = "SAVED"
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span class="hud-value">12 · Benchmark Claims That Survive · STATUS: </span><span class="hud-active">{_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
