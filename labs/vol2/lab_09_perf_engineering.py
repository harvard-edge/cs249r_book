import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 09: Recover Useful Progress · MLSysBook")


@app.cell
async def _():
    import sys
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
    from mlsysim.labs.experiments.v2_09_experiments import (
        MODEL_ID, compare_configs, evaluate, get_scenario, result_to_dict,
        PROFILING_TARGETS, PROFILING_CADENCES, sample_fleet_profile,
        goodput_study, compare_goodput_study, goodput_comparison_issues,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata,
        get_lab_track_variant, get_track_profile, report_export_panel, track_context,
    )
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence
    from mlsysbook_labs.notebook import (
        notebook_shell, notebook_part, notebook_workspace, notebook_table,
        notebook_cards, notebook_progress, notebook_status, notebook_bridge, notebook_part_reflection, notebook_tab_hint,
    )
    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, MODEL_ID, apply_plotly_theme,
        PROFILING_TARGETS, PROFILING_CADENCES, sample_fleet_profile,
        audit_evidence, build_lab_report, capture_evidence, compare_configs,
        evaluate, get_lab_metadata, get_lab_track_variant, get_scenario,
        get_track_profile, go, ledger, mo, report_export_panel, result_to_dict,
        track_context, goodput_study, compare_goodput_study,
        goodput_comparison_issues, notebook_shell, notebook_part,
        notebook_workspace, notebook_table, notebook_cards, notebook_progress,
        notebook_status, notebook_bridge, notebook_part_reflection, notebook_tab_hint,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    get_saved_discussion, set_saved_discussion = mo.state(None)
    return get_evidence, set_evidence, get_saved_discussion, set_saved_discussion


@app.cell
def _(ledger, mo, set_evidence, set_saved_discussion):
    from mlsysbook_labs.contexts import preferred_context as _preferred_context
    _track_options = {"TinyML": "tinyml", "MobileML": "mobile", "EdgeML": "edge", "CloudML": "cloud"}
    def reset_track(_value):
        set_evidence({})
        set_saved_discussion(None)
    track = mo.ui.dropdown(
        _track_options,
        value=next(label for label, context in _track_options.items() if context == _preferred_context(ledger)), label="Deployment context", on_change=reset_track,
    )
    return (track,)


@app.cell
def _(get_lab_metadata, get_lab_track_variant, get_scenario, get_track_profile, track):
    track_id = track.value
    scenario = get_scenario(track_id)
    track_profile = get_track_profile(track_id)
    _metadata = get_lab_metadata("vol2/lab_09_perf_engineering.py")
    track_variant = get_lab_track_variant(_metadata.lab_id, track_id)
    return scenario, track_id, track_profile, track_variant


@app.cell
def _():
    from mlsysbook_labs.learning_paths import get_learning_path
    learning_path = get_learning_path("vol2", 9)
    return (learning_path,)


@app.cell
def _(PROFILING_TARGETS, PROFILING_CADENCES, mo, track_id):
    _track = track_id
    a_wait = mo.ui.slider(0, 48, step=4, value=16, label="Incomplete allocation / wait (baseline-run equivalents)")
    a_startup = mo.ui.slider(0, 24, step=4, value=8, label="Startup (baseline-run equivalents)")
    a_discarded = mo.ui.slider(0, 16, step=2, value=4, label="Discarded work (baseline-run equivalents)")
    b_window = mo.ui.dropdown({"Same offered window": 128, "Shorter offered window": 64}, value="Shorter offered window", label="Candidate measurement window")
    b_profile_target = mo.ui.dropdown(PROFILING_TARGETS, value="Fleet average", label="Profile scope")
    b_profile_cadence = mo.ui.dropdown(PROFILING_CADENCES, value="Every eighth tick", label="Sampling cadence")
    b_profile_prediction = mo.ui.radio(["Both anomalies", "Only one anomaly", "Neither anomaly"], label="Predict which anomalies this scope and cadence will detect")
    c_rewrite = mo.ui.radio({"Fusion": "fusion", "Tiling": "tiling", "Fusion and tiling": "combined"}, value="Fusion", label="Program intervention")
    d_window = mo.ui.dropdown({"Short window": 32, "Medium window": 128, "Long window": 131072}, value="Short window", label="Fixed offered window (baseline-run equivalents)")
    d_startup = mo.ui.slider(0, 48, step=4, value=24, label="Device release/startup delay (baseline-run equivalents; excludes host AOT)")
    e_wait = mo.ui.slider(0, 48, step=4, value=8, label="Joint plan allocation wait")
    e_startup = mo.ui.slider(0, 24, step=4, value=8, label="Joint plan startup")
    e_discarded = mo.ui.slider(0, 16, step=2, value=4, label="Joint plan discarded work")
    e_compiled = mo.ui.checkbox(value=True, label="Use compiled program (Cloud JIT charged here; device AOT on build host)")
    e_fusion = mo.ui.checkbox(value=True, label="Fuse the program path")
    return (a_wait, a_startup, a_discarded, b_window, b_profile_target, b_profile_cadence, b_profile_prediction, c_rewrite, d_window, d_startup,
            e_wait, e_startup, e_discarded, e_compiled, e_fusion)


@app.cell
def _(mo, track_id):
    _track = track_id
    prediction_a = mo.ui.radio(['Scheduling loss changes', 'Runtime loss changes', 'Both boundaries change'], label='Which boundary will your first change affect?').form(submit_button_label="Record prediction")
    explanation_a = mo.ui.text_area(label="Part A interpretation", placeholder="Name the changed input, cite the observed consequence, and explain what reverses the claim.", debounce=False)
    prediction_b = mo.ui.radio(['Directly comparable', 'Different measurement contract'], label='Can these two reports support a direct optimization claim?').form(submit_button_label="Record prediction")
    explanation_b = mo.ui.text_area(label="Part B interpretation", placeholder="Name the changed input, cite the observed consequence, and explain what reverses the claim.", debounce=False)
    prediction_c = mo.ui.radio(['Compute', 'Movement', 'Dispatch / setup'], label='Which program term should the team inspect first?').form(submit_button_label="Record prediction")
    explanation_c = mo.ui.text_area(label="Part C interpretation", placeholder="Name the changed input, cite the observed consequence, and explain what reverses the claim.", debounce=False)
    prediction_d = mo.ui.radio(['Compiled path wins', 'Uncompiled path wins', 'Depends on offered window'], label='Will the faster steady path improve useful output in this window?').form(submit_button_label="Record prediction")
    explanation_d = mo.ui.text_area(label="Part D interpretation", placeholder="Name the changed input, cite the observed consequence, and explain what reverses the claim.", debounce=False)
    prediction_e = mo.ui.radio(['Program gain survives', 'Other boundaries erase the gain', 'Need the complete accounting'], label='Will your joint plan outperform the supplied baseline?').form(submit_button_label="Record prediction")
    explanation_e = mo.ui.text_area(label="Part E interpretation", placeholder="Name the changed input, cite the observed consequence, and explain what reverses the claim.", debounce=False)
    final_choice = mo.ui.radio(["Adopt saved joint plan", "Retain baseline", "Hold for more evidence"], label="Team decision")
    final_rejected = mo.ui.radio(["Adopt saved joint plan", "Retain baseline", "Hold for more evidence"], label="Rejected alternative")
    final_trigger = mo.ui.text_area(label="Revisit condition", placeholder="Name a workload, window, quality, or hardware-ideal change that requires a new comparison.", debounce=False)
    final_risk = mo.ui.text_area(label="Remaining evidence gap", placeholder="Distinguish this controlled accounting fixture from real telemetry and arrival/deadline evidence.", debounce=False)
    rationale = mo.ui.text_area(label="Integrated discussion", placeholder="Use your saved A–E comparisons to defend a decision, its mechanism, and a rejected alternative.", debounce=False)
    return (prediction_a, explanation_a, prediction_b, explanation_b, prediction_c, explanation_c, prediction_d, explanation_d, prediction_e, explanation_e, final_choice, final_rejected, final_trigger, final_risk, rationale)


@app.cell
def _(prediction_a, explanation_a, prediction_b, explanation_b, prediction_c, explanation_c, prediction_d, explanation_d, prediction_e, explanation_e):
    predictions = {"A": prediction_a, "B": prediction_b, "C": prediction_c, "D": prediction_d, "E": prediction_e}
    explanations = {"A": explanation_a, "B": explanation_b, "C": explanation_c, "D": explanation_d, "E": explanation_e}
    prediction_values = {part: widget.value for part, widget in predictions.items()}
    explanation_values = {part: widget.value for part, widget in explanations.items()}
    return predictions, explanations, prediction_values, explanation_values


@app.cell
def _(
    a_wait, a_startup, a_discarded, b_window, b_profile_target, b_profile_cadence, b_profile_prediction, sample_fleet_profile, c_rewrite, d_window, d_startup,
    e_wait, e_startup, e_discarded, e_compiled, e_fusion,
    goodput_study, compare_goodput_study, goodput_comparison_issues,
    evaluate, compare_configs, result_to_dict, track_id,
):
    a_comparison = compare_goodput_study(track_id, allocation_wait_runs=a_wait.value,
        startup_runs=a_startup.value, discarded_runs=a_discarded.value)
    _b_baseline = goodput_study(track_id)
    _b_result = goodput_study(track_id, window_runs=b_window.value, fusion=True)
    b_comparison = {"baseline": _b_baseline, "result": _b_result,
        "comparison_issues": goodput_comparison_issues(_b_baseline, _b_result)}
    b_profile = sample_fleet_profile(target=b_profile_target.value, cadence=b_profile_cadence.value)
    b_comparison["baseline"]["profiling"] = sample_fleet_profile(target="fleet_average", cadence=8)
    b_comparison["baseline"]["inputs"]["profiling"] = b_comparison["baseline"]["profiling"]["inputs"]
    b_comparison["result"]["profiling"] = b_profile
    b_comparison["result"]["inputs"]["profiling"] = {**b_profile["inputs"], "prediction": b_profile_prediction.value}
    _flags = {"fusion": {"fusion": True}, "tiling": {"tiling": True},
              "combined": {"fusion": True, "tiling": True}}
    c_comparison = compare_configs(track_id, baseline_options={}, result_options=_flags[c_rewrite.value])
    d_baseline = goodput_study(track_id, window_runs=d_window.value)
    d_result = goodput_study(track_id, window_runs=d_window.value, compiled=True,
        startup_runs=8 if track_id == "cloud" else d_startup.value)
    d_comparison = {"baseline": d_baseline, "result": d_result,
        "comparison_issues": goodput_comparison_issues(d_baseline, d_result)}
    e_comparison = compare_goodput_study(track_id, allocation_wait_runs=e_wait.value,
        startup_runs=e_startup.value, discarded_runs=e_discarded.value,
        compiled=e_compiled.value, fusion=e_fusion.value)
    comparisons = {
        "A": a_comparison, "B": b_comparison,
        "C": {"baseline": result_to_dict(c_comparison["baseline"]),
              "result": result_to_dict(c_comparison["result"])},
        "D": d_comparison, "E": e_comparison,
    }
    return comparisons, c_comparison, b_profile


@app.cell
def _(MODEL_ID, capture_evidence, comparisons, explanations, explanation_values, mo, predictions, prediction_values, set_evidence, track_id):
    def retain(part):
        comparison = comparisons[part]
        snapshot = capture_evidence(track=track_id, part=part,
            prediction=prediction_values[part],
            inputs=comparison["result"]["inputs"], baseline=comparison["baseline"],
            result=comparison["result"], model_key=MODEL_ID,
            decision={"interpretation": explanation_values[part],
                      "comparison_issues": comparison.get("comparison_issues", [])})
        set_evidence(lambda current: {**current, part: snapshot})
    capture_buttons = {part: mo.ui.button(label=f"Save Part {part} evidence", kind="success",
        disabled=prediction_values[part] is None or not explanation_values[part].strip()
            or (part == "B" and comparisons["B"]["result"]["inputs"]["profiling"]["prediction"] is None)
            or comparisons[part]["baseline"]["inputs"] == comparisons[part]["result"]["inputs"],
        on_click=lambda _v, part=part: retain(part)) for part in "ABCDE"}
    return (capture_buttons,)


@app.cell
def _(
    ACADEMIC_LAB_CSS, LAB_CSS, mo, scenario, track, track_context,
    track_profile, track_variant,
):
    from mlsysbook_labs.notebook import (
        notebook_header as _course_header, notebook_shell as _course_shell,
        notebook_learning_map as _course_learning_map,
        notebook_scenario as _course_scenario,
    )
    from mlsysbook_labs.learning_paths import get_learning_path as _course_path
    from mlsysbook_labs import get_lab_metadata as _course_metadata
    header = _course_header(_course_metadata("vol2/lab_09_perf_engineering.py"), 'Where does offered fleet capacity stop becoming useful progress?')
    selector = _course_shell([mo.md("**Choose the fleet context**"), track], volume=2).style(
        {
            "max-width": "100%", "padding": "10px 12px",
            "border": "1px solid #dbe4ee", "border-radius": "9px",
            "background": "#fff", "margin-bottom": "8px",
        }
    )
    context = mo.md(f"**Selected workload:** {scenario.workload}\n\n**Fleet boundary:** {scenario.fleet_semantics}")
    _course_shell([
        header,
        selector,
        _course_scenario(_course_path('vol2', 9), track=track.value, volume=2),
        context,
        mo.Html('<p class="pilot-note">Begin with the supplied project case; no earlier results are needed. These comparisons use illustrative calculations rather than measurements from a running fleet. Quality and workload identity stay fixed.</p>'),
        _course_learning_map(_course_path('vol2', 9))
    ], gap=0.5, volume=2)
    return


@app.cell
def _(
    COLORS, apply_plotly_theme, audit_evidence, a_wait, a_startup, a_discarded,
    b_window, b_profile_target, b_profile_cadence, b_profile_prediction, b_profile, c_rewrite, d_window, d_startup, e_wait, e_startup, e_discarded, e_compiled,
    e_fusion, comparisons, c_comparison, capture_buttons, explanations,
    get_evidence, go, mo, notebook_shell, notebook_part, notebook_workspace,
    notebook_table, notebook_cards, notebook_progress, notebook_bridge,
    predictions, prediction_values, explanation_values, learning_path, notebook_part_reflection, notebook_tab_hint, track_id, final_choice, final_rejected, final_trigger, final_risk, rationale,
):
    _captures = get_evidence()
    audit = audit_evidence(_captures, track=track_id, required_parts=tuple("ABCDE"), contrast_required_parts=tuple("ABCDE"))

    def saved(part):
        if part not in _captures:
            return mo.md("**Saved evidence:** none yet. Live changes update immediately.")
        record = _captures[part].to_dict()
        changed = record["inputs"] != comparisons[part]["result"]["inputs"]
        note_changed = record["decision"]["interpretation"] != explanation_values[part]
        state = "Saved comparison differs from the live trial." if changed else "Saved comparison matches the live trial."
        note = " Interpretation edited; save again to attach it." if note_changed else ""
        return mo.md(f"**Immutable snapshot retained.** {state}{note} Original prediction: {record['prediction']}.")

    def factors(comparison):
        def display_factor(role, key):
            run = comparison[role]
            if key == "program" and not run["program_identifiable"]:
                return "Unavailable: no retained execution"
            if key == "runtime" and not run["runtime_identifiable"]:
                return "Unavailable: no allocation"
            return f"{run['factors'][key]:.4f}"
        return notebook_table([
            {"Factor": key.title(), "Baseline": display_factor("baseline", key),
             "Candidate": display_factor("result", key)}
            for key in ("scheduling", "runtime", "program", "total")
        ], caption="Dimensionless fractions. Each factor uses the capacity-time passed by the preceding boundary.")

    def waterfall(comparison):
        figure = go.Figure()
        for term, color in (("scheduling", COLORS["BlueLine"]), ("runtime", COLORS["OrangeLine"]),
                            ("program", COLORS["RedLine"]), ("useful", COLORS["GreenLine"])):
            figure.add_bar(name=term.title(), y=["Baseline", "Candidate"],
                x=[comparison[key]["losses"][term] for key in ("baseline", "result")],
                orientation="h", marker_color=color)
        figure.update_layout(barmode="stack", height=260, xaxis_title="Share of offered capacity-time",
            xaxis_range=[0, 1], margin=dict(l=20, r=20, t=10, b=45), legend_orientation="h")
        apply_plotly_theme(figure)
        figure.update_layout(autosize=True, margin=dict(l=72, r=12, t=8, b=90),
            legend=dict(orientation="h", y=-0.5, x=0), xaxis_title_standoff=8)
        return mo.ui.plotly(figure, config={"responsive": True, "displayModeBar": False})

    def feedback(comparison):
        issues = comparison.get("comparison_issues", [])
        status = "Contract mismatch: " + "; ".join(str(x) for x in issues) if issues else "The supplied measurement contracts match."
        observations = comparison["result"].get("observations", [])
        details = "\n\n".join(f"**{item['event']}**: {item['duration_s_per_unit']:.6g} s per execution unit · {item['boundary']} boundary" for item in observations)
        failure = comparison["result"].get("failure")
        if failure:
            details += f"\n\n**Boundary failure:** {failure}. No retained work means program efficiency is unavailable, not a measured zero."
        return mo.callout(mo.md(status + "\n\n" + details), kind="warn" if issues or failure else "info")

    def investigation(part, title, scene, controls, instrument, next_step):
        intro = notebook_part(f"{part} · {title}", scene)
        if prediction_values[part] is None:
            return notebook_shell([intro, predictions[part], mo.md(f"**Next:** {next_step}")], volume=2)
        return notebook_shell([
            intro, predictions[part],
            notebook_workspace(controls, instrument), notebook_part_reflection(learning_path, part), explanations[part],
            capture_buttons[part],
            mo.md("Change a condition before saving a contrast." if comparisons[part]["baseline"]["inputs"] == comparisons[part]["result"]["inputs"] else "Changed settings with an unchanged outcome still form valid evidence."),
            saved(part), mo.md(f"**Next:** {next_step}"),
        ], volume=2)

    _a = comparisons["A"]
    part_a = investigation("A", "Account for the missing capacity",
        "The fleet team sees occupied devices but fewer retained results. Offered capacity-time is the full window; complete allocations pass the scheduling boundary; retained execution passes runtime; the fixed hardware ideal measures useful progress. Only atomic completed-and-retained executions earn useful credit; partial executions do not. Change one loss at a time before combining changes.",
        [a_wait, a_startup, a_discarded], [waterfall(_a), factors(_a), feedback(_a),
            notebook_table([{"Boundary": key, "Baseline capacity-time (execution-unit s)": f"{_a['baseline']['boundaries_s'][key]:.6g}",
                "Candidate capacity-time (execution-unit s)": f"{_a['result']['boundaries_s'][key]:.6g}"} for key in ("offered", "allocated", "productive", "ideal")]),
            mo.accordion({"Explain the accounting": mo.md("Each loss is charged once. Wait before complete allocation belongs to scheduling; startup and discarded work consume allocated time without retained execution. Program loss is the gap between retained execution and the fixed ideal. Use the waterfall to distinguish a conditional factor from its share of the original offered capacity.")})],
        "Check whether another team's attractive report used the same measurement contract.")

    _b = comparisons["B"]
    part_b = investigation("B", "Decide which reports can be compared",
        "A teammate proposes a fused path and supplies a favorable efficiency report. First inspect its window, workload artifact, population, precision, and hardware ideal. Repair the window mismatch before attributing a change to the program intervention.",
        [b_window, b_profile_target, b_profile_cadence, b_profile_prediction], [factors(_b), feedback(_b),
            mo.md("**Zoom before collecting everything.** A separate synthetic backend-job reference (shared by all tracks, not the endpoint path above) has eight ranks and 24 ticks. It contains an input stall on rank 3 at ticks [5,7) and a slow-rank burst on rank 6 at [17,20). Fleet averages dilute local anomalies; sampling starts at tick zero. Predict, then try every-tick versus sparse collection and a healthy versus suspect rank."),
            (notebook_table([
                {"Event": row["event"], "Rank / interval": f"{row['rank']} · {row['event_interval']}",
                 "Observation": (f"Detected at {row['overlapping_sample_ticks']}" if row["detected"] else f"Missed: {row['reason']}")}
                for row in b_profile["observations"]
            ], caption="Fixture truth versus observed samples. The flag threshold is 0.25 excess; an absence of flags is not evidence of health.") if b_profile_prediction.value is not None else mo.md("Choose a sampling prediction to reveal observations.")),
            (mo.callout(mo.md(f"**Detected:** {', '.join(b_profile['detected_events']) or 'none'}. **Missed:** {b_profile['missed_events']}. Collection takes {b_profile['sample_count']} records and {b_profile['collection_cost_s']:.4f} collector-seconds over a {b_profile['window_s']:.0f}-second window. This is a declared collection-cost fixture, not a measured workload slowdown. More detailed samples buy attribution at collection cost; wrong-rank selection and sparse cadence can still omit the event."), kind="info") if b_profile_prediction.value is not None else mo.md("")),
            notebook_table([{"Field": {"artifact": "Model", "workload": "Task", "population": "Group", "precision": "Format", "hardware_ideal": "Ideal", "window_s": "Window (s)", "fleet_units": "Units", "warmup_policy": "Setup", "sampling_policy": "Sample", "quality_evidence": "Quality", "measurement_unit": "Unit", "retention_policy": "Credit"}.get(key, key),
                "Supplied record": (str(_b['baseline']['contract'][key]) if _b['baseline']['contract'][key] == _b['result']['contract'][key]
                    else f"Baseline: {_b['baseline']['contract'][key]:.6g}; candidate: {_b['result']['contract'][key]:.6g}" if isinstance(_b['baseline']['contract'][key], float)
                    else f"Baseline: {_b['baseline']['contract'][key]}; candidate: {_b['result']['contract'][key]}")}
                for key in _b['baseline']['contract'] if key != "reference_fingerprint"], caption="Matching fields appear once. Differences name both records."),
            mo.md("A mismatched report is useful evidence about comparability, but is not an optimization win. Save either the diagnosed mismatch or the repaired comparison and explain the claim it permits.")],
        "Use a matched program comparison to locate the path term that could explain a gain.")

    _base, _result = c_comparison["baseline"], c_comparison["result"]
    _profile = go.Figure()
    for _label, _field, _color in (("Compute ceiling", "compute_time", COLORS["OrangeLine"]),
                                  ("Movement ceiling", "movement_time", COLORS["BlueLine"]),
                                  ("Dispatch", "launch_time", COLORS["GreenLine"])):
        _profile.add_bar(name=_label, x=["Baseline", "Candidate"],
            y=[getattr(run, _field).to("ms").magnitude for run in (_base, _result)], marker_color=_color)
    _profile.update_layout(barmode="group", height=285, yaxis_title="Diagnostic term (ms)",
        margin=dict(l=20,r=20,t=10,b=50), legend_orientation="h")
    apply_plotly_theme(_profile)
    _profile.update_layout(autosize=True, margin=dict(l=58, r=12, t=8, b=100), legend=dict(orientation="h", y=-0.45, x=0))
    part_c = investigation("C", "Trace a program loss to its mechanism",
        "Scheduler and job records localize the loss, but cannot expose its kernel mechanism. Inspect a bounded phase-path fixture at fixed batch and precision; select a rewrite that changes traffic, reuse, or dispatch. These terms are diagnostic ceilings and are not additive wall-clock phases.",
        [c_rewrite], [mo.ui.plotly(_profile, config={"responsive": True, "displayModeBar": False}), notebook_table([
            {"Run": label, "Bottleneck": run.bottleneck, "Whole-path latency (ms)": f"{run.latency.to('ms').magnitude:.6g}",
             "Memory feasible": run.memory_feasible} for label, run in (("Baseline", _base), ("Candidate", _result))]),
            mo.callout(mo.md(f"The candidate path is **{_result.bottleneck} bound**. The evaluator reports **{c_comparison['speedup']:.3f}×** whole-path speedup. Inspect the changed term and the workspace consequence before recommending it."), kind="info"),
            mo.accordion({"Optional depth: local optimization limits": mo.md("Fusion can reduce intermediates and launches; tiling trades workspace for reuse. Their joint gain must be recomputed on one path. This local analytical fixture does not simulate rank traces, interconnect contention, compiler placement, or measured fleet-wide MFU. A local gain must be reconciled with the complete accounting in Part E.")})],
        "Identify where setup belongs and compare runtime/program consequences inside the fixed endpoint window.")

    _d = comparisons["D"]
    part_d = investigation("D", "Test a runtime/program crossover",
        "Compare the compiled and uncompiled path inside the same offered window. For CloudML, one-time JIT setup consumes held runtime. Device tracks compile on a separate build host: AOT setup is excluded from endpoint capacity-time; a separately selected release/startup delay supplies the runtime cost. These are distinct physical boundaries, not the same compilation story.",
        [d_window, *([] if track_id == "cloud" else [d_startup])], [waterfall(_d), factors(_d), feedback(_d),
            notebook_table([{"Setup boundary": "Endpoint JIT required setup", "Candidate seconds": f"{_d['result']['compile_setup_s']:.6g}"},
                {"Setup boundary": "Endpoint JIT within observed window", "Candidate seconds": f"{_d['result']['compile_setup_charged_s']:.6g}"},
                {"Setup boundary": "Separate build-host AOT, excluded", "Candidate seconds": f"{_d['result']['build_host_setup_s']:.6g}"}]),
            mo.md("Try a short and a long window. Explain a program-factor improvement alongside any runtime-factor reduction; an unchanged or worse total is a legitimate result.")],
        "Combine allocation, runtime, and program decisions without multiplying independently measured speedups.")

    _e = comparisons["E"]
    part_e = investigation("E", "Judge the complete intervention",
        "The team must choose an investment, not the prettiest local metric. Hold the offered window and named ideal fixed, choose a joint plan, and inspect all three boundaries. Saving a failed plan is valid evidence for retaining the baseline or holding for better records.",
        [e_wait, e_startup, e_discarded, e_compiled, e_fusion],
        [waterfall(_e), factors(_e), feedback(_e)],
        "Defend a team decision from the saved joint plan and identify what serving-arrival evidence remains open.")

    _rows = [{"Part": p, "Evidence": "Saved" if p in _captures else "Missing",
        "Saved versus live": ("Different live trial; decision uses saved record" if _captures[p].to_dict()["inputs"] != comparisons[p]["result"]["inputs"] else "Matches live trial") if p in _captures else "No record"} for p in "ABCDE"]
    decision_ready = (audit.complete and final_choice.value is not None and final_rejected.value is not None
        and final_choice.value != final_rejected.value and bool(rationale.value.strip())
        and bool(final_risk.value.strip()) and bool(final_trigger.value.strip()))
    _synthesis = notebook_shell([
        notebook_part("Team decision", "Use your saved A–E comparisons. Moving a live control does not change those copies. Cite the saved joint plan, explain its competing consequences, and name a changed condition that would reverse the recommendation."),
        notebook_table(_rows), final_choice, final_rejected, rationale, final_risk, final_trigger,
        mo.callout(mo.md("Ready to download and explicitly save." if decision_ready else "Save all five investigations, choose distinct recommendation and rejected alternative, and complete the discussion, evidence gap, and revisit condition."), kind="success" if decision_ready else "warn"),
        notebook_bridge("Allocation and recovery records motivate the supplied loss categories; no earlier decisions were imported.",
            "Carry forward the saved accounting contract, joint plan, and rejected alternative.",
            "Serving must test arrivals, queueing, admission, and deadlines. This lab does not establish those results."),
    ], volume=2)
    notebook_shell([notebook_progress({p: "Saved" if p in _captures else "Not saved" for p in "ABCDE"}),
        notebook_tab_hint(), mo.ui.tabs({"A · Account": part_a, "B · Contract": part_b, "C · Profile": part_c,
                    "D · Compile": part_d, "E · Compose": part_e, "Synthesis · Team decision": _synthesis})], volume=2)
    return audit, decision_ready


@app.cell
def _():
    from mlsysbook_labs.notebook import notebook_shell as _submit_shell, notebook_submission as _submit_guide
    from mlsysbook_labs.learning_paths import get_learning_path as _submit_path
    _submit_shell([_submit_guide(_submit_path(2, 9))], volume=2)
    return


@app.cell
def _(
    decision_ready, audit, build_lab_report, final_choice, final_rejected,
    final_risk, final_trigger, get_evidence, get_lab_metadata, mo, rationale,
    report_export_panel, scenario, track_id, notebook_shell, set_saved_discussion, learning_path,
):
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not decision_ready, _download_readiness(volume=2))
    _snapshots = {part: get_evidence()[part].to_dict() for part in "ABCDE"}
    report = build_lab_report(
        get_lab_metadata("vol2/lab_09_perf_engineering.py"), track=track_id,
        scenario=scenario.workload,
        learning_objectives=learning_path["learning_objectives"],
        predictions={part: record["prediction"] for part, record in _snapshots.items()},
        knob_settings={part: record["inputs"] for part, record in _snapshots.items()},
        evidence_summary=_snapshots,
        binding_constraints={"measurement_contract": _snapshots["E"]["result"]["contract"], "audit": audit.to_dict()},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value},
        final_decision={"recommendation": final_choice.value, "rationale": rationale.value},
        big_takeaways=["Every lost interval belongs to one nested boundary.",
            "A useful optimization claim requires the same accounting contract.",
            "Whole-window progress must include setup and competing factor effects."],
        reflections={"part_interpretations": {part: record["decision"]["interpretation"] for part, record in _snapshots.items()},
                     "rationale": rationale.value, "revisit": final_trigger.value},
        residual_risk=final_risk.value,
        result_snapshot={"track": track_id, "captures": _snapshots,
            "recommendation": final_choice.value, "rejected": final_rejected.value,
            "rationale": rationale.value, "revisit": final_trigger.value, "gap": final_risk.value},
        source_trace={"hardware": scenario.hardware_name, "scenario": scenario.provenance,
            "calculations": "MLSysIM v2_09_experiments nested goodput study and bounded phase path"},
    )
    _save = mo.ui.button(label="Save discussion locally", kind="success",
        on_click=lambda _v: set_saved_discussion(report.snapshot["result_snapshot"]))
    notebook_shell([mo.md("### Download your discussion"), report_export_panel(report), _save,
        mo.md("Download contains exact saved inputs, original predictions, interpretations, contracts, and results. Editing a draft does not update the explicitly saved discussion.")], volume=2)
    return report, _save


@app.cell
async def _(MODEL_ID, get_saved_discussion, ledger, mo, notebook_shell, notebook_status):
    from mlsysbook_labs.fleet_evidence import discussion_fingerprint, save_completed_discussion
    _discussion = get_saved_discussion()
    _message = "Not saved locally. Download and local saving are separate actions."
    if _discussion is not None:
        _outcome = await save_completed_discussion(ledger, chapter=9, track=_discussion["track"],
            design={"schema_version": 2, "lab_id": "v2_09", "model_id": MODEL_ID, **_discussion},
            fingerprint=discussion_fingerprint({"schema_version": 2, "lab_id": "v2_09", "model_id": MODEL_ID, **_discussion}),
            ready=True, requested=True)
        _message = f"{_outcome.status}: {_outcome.detail}"
    notebook_shell([notebook_status("Local storage", _message)], volume=2)
    return


@app.cell
def _(get_saved_discussion, report, notebook_shell, notebook_status):
    _saved = get_saved_discussion()
    _current = report.snapshot["result_snapshot"]
    _label = "Not saved" if _saved is None else "Saved" if _saved == _current else "Changed since save"
    notebook_shell([notebook_status("Discussion state", _label)], volume=2)
    return


if __name__ == "__main__":
    app.run()
