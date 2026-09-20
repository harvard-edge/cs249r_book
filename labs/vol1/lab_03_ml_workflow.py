import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 03: Carry the Constraint · MLSysBook")


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
    from mlsysim.engine.v1_03_experiments import (
        TRACKS, compare_iteration_plans, compare_validation_stages,
        evaluate_release_checks, feedback_timeline, hold_workflow_decision, simulate_iterations,
        trace_requirement, track_summary,
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
        ACADEMIC_LAB_CSS, COLORS, LAB_CSS, TRACKS, apply_plotly_theme,
        audit_evidence, build_lab_report, capture_evidence,
        compare_iteration_plans, compare_validation_stages,
        evaluate_release_checks, feedback_timeline, get_lab_metadata, go,
        hold_workflow_decision, ledger, mo, report_export_panel, simulate_iterations,
        trace_requirement, track_summary,
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
def _(track, track_summary):
    track_id = track.value
    profile = track_summary(track_id)
    return profile, track_id


@app.cell
def _(mo, profile, track_id):
    _track_key = track_id
    _requirement_options = {
        value["label"]: key for key, value in profile["requirements"].items()
    }
    requirement_choice = mo.ui.dropdown(
        _requirement_options,
        value=profile["requirements"][profile["default_requirement"]]["label"],
        label="Requirement to carry",
    )
    late_stage = mo.ui.dropdown(
        {"Validation": "validation", "Deployment": "deployment", "Monitoring": "monitoring"},
        value="Deployment", label="Later discovery point",
    )
    escalation = mo.ui.slider(
        1.0, 2.0, value=1.0, step=0.25, label="Rework sensitivity multiplier",
    )
    timing_decision = mo.ui.radio(
        {"Pay for the data-stage check": "early", "Defer to the later stage": "late"},
        label="Validation timing decision",
    )
    development_budget = mo.ui.slider(
        4, 14, value=int(profile["development_budget_days"]), step=1,
        label="Development budget (days)",
    )
    plan_choice = mo.ui.radio(
        {"Rapid offline loop": "rapid_offline", "Target checks in the loop": "target_in_loop",
         "Hold: neither plan is defensible": "none"},
        label="Iteration-plan decision",
    )
    plan_rejected = mo.ui.radio(
        {"Rapid offline loop": "rapid_offline", "Target checks in the loop": "target_in_loop"},
        label="Tested alternative",
    )
    release_choice = mo.ui.radio(
        {"Offline checks": "offline", "Target checks": "target",
         "Combined checks": "combined", "Hold the release": "hold"},
        label="Release decision",
    )
    release_rejected = mo.ui.radio(
        {"Offline checks": "offline", "Target checks": "target", "Combined checks": "combined"},
        label="Compared check plan",
    )
    signal_interval = mo.ui.slider(
        2, 10, value=4, step=1, label="Signal-check interval (hours)",
    )
    outcome_delay = mo.ui.dropdown(
        {"12 hours": 12, "24 hours": 24, "36 hours": 36, "48 hours": 48},
        value=f"{int(profile['default_outcome_delay_hours'])} hours", label="Outcome-label delay",
    )
    feedback_revisit = mo.ui.radio(
        {"Validation after an alert": "validation", "Data after labeled outcomes": "data",
         "Modeling immediately after an alert": "modeling"},
        label="Stage to revisit",
    )
    return (
        development_budget, escalation, feedback_revisit, late_stage,
        outcome_delay, plan_choice, plan_rejected, release_choice,
        release_rejected, requirement_choice, signal_interval, timing_decision,
    )


@app.cell
def _(mo, track_id):
    _track_key = track_id
    a_prediction = mo.ui.radio(
        {"Mostly data artifacts": "data", "Mostly modeling artifacts": "modeling",
         "Mostly validation or deployment artifacts": "late"},
        label="Where will the two requirements differ most?",
    ).form(submit_button_label="Lock Part A prediction")
    b_prediction = mo.ui.radio(
        {"Earlier discovery costs less overall": "early_less",
         "Both discovery points cost the same": "same",
         "Later discovery costs less overall": "late_less"},
        label="Which check timing has the lower response cost?",
    ).form(submit_button_label="Lock Part B prediction")
    c_prediction = mo.ui.radio(
        {"More iterations guarantee a releasable candidate": "count_wins",
         "Fewer target-aware iterations can produce the releasable candidate": "target_wins",
         "Neither plan completes a failed iteration": "no_failures"},
        label="What will the fixed time budget reveal?",
    ).form(submit_button_label="Lock Part C prediction")
    d_prediction = mo.ui.radio(
        {"Offline checks catch every seeded defect": "offline_all",
         "Target checks catch every seeded defect": "target_all",
         "Each single check plan leaves a different blind spot": "both_partial"},
        label="Which evidence is sufficient for release?",
    ).form(submit_button_label="Lock Part D prediction")
    e_prediction = mo.ui.radio(
        {"An alert proves task harm": "alert_proves_harm",
         "An alert starts investigation; outcomes establish harm": "signal_then_outcome",
         "Monitoring repairs the changed behavior": "monitor_repairs"},
        label="What can the first production signal establish?",
    ).form(submit_button_label="Lock Part E prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction, e_prediction


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Rapid offline loop": "rapid_offline", "Target checks in the loop": "target_in_loop",
         "Hold: no tested plan is defensible": "none"}, label="Recommendation",
    )
    final_rejected = mo.ui.radio(
        {"Rapid offline loop": "rapid_offline", "Target checks in the loop": "target_in_loop"},
        label="Quantified rejected alternative",
    )
    final_trigger = mo.ui.radio(
        {"A release check exposes a new defect": "new_defect",
         "Labeled outcomes cross the task floor": "outcome_floor",
         "The development-time budget changes": "budget_change"},
        label="Reevaluation trigger",
    )
    final_risk = mo.ui.radio(
        {"Unseeded target failure": "unseeded_failure", "Delayed outcome labels": "label_delay",
         "Rework-cost assumptions": "rework_assumptions"}, label="Remaining limitation",
    )
    rationale = mo.ui.text_area(
        label="Decision rationale",
        placeholder="Use saved iteration counts, target outcomes, the rejected plan, release evidence, and a reevaluation trigger.",
    )
    return final_choice, final_rejected, final_risk, final_trigger, rationale


@app.cell
def _(
    TRACKS, compare_iteration_plans, compare_validation_stages,
    development_budget, escalation, evaluate_release_checks,
    feedback_timeline, late_stage, outcome_delay, release_choice,
    release_rejected, requirement_choice, hold_workflow_decision, simulate_iterations,
    signal_interval, trace_requirement, track_id,
):
    _requirement_ids = tuple(TRACKS[track_id]["requirements"])
    _selected_requirement = requirement_choice.value
    _other_requirement = next(key for key in _requirement_ids if key != _selected_requirement)
    a_base = trace_requirement(track_id, _selected_requirement)
    a_result = trace_requirement(track_id, _other_requirement)
    b_comparison = compare_validation_stages(
        track_id, "data", late_stage.value, requirement_id=_selected_requirement,
        escalation_factor=escalation.value,
    )
    c_comparison = compare_iteration_plans(track_id, development_budget.value)
    c_rapid = simulate_iterations(track_id, "rapid_offline", development_budget.value)
    c_target = simulate_iterations(track_id, "target_in_loop", development_budget.value)
    c_hold = hold_workflow_decision(track_id, "iteration")
    d_results = {
        plan: evaluate_release_checks(track_id, plan)
        for plan in ("offline", "target", "combined")
    }
    d_hold = hold_workflow_decision(track_id, "release")
    _selected_release = release_choice.value if release_choice.value in d_results else None
    _rejected_release = release_rejected.value or "offline"
    if _selected_release is not None:
        d_result = d_results[_selected_release]
        d_base = d_results[_rejected_release] if _rejected_release != _selected_release else d_results["target" if _selected_release != "target" else "offline"]
    else:
        d_base = d_hold
        d_result = d_results[_rejected_release]
    _default_outcome_delay = TRACKS[track_id]["feedback"]["outcome_delay_hours"]
    e_base = feedback_timeline(track_id, 11, _default_outcome_delay)
    e_result = feedback_timeline(track_id, signal_interval.value, outcome_delay.value)
    return (
        a_base, a_result, b_comparison, c_comparison, c_hold, c_rapid, c_target,
        d_base, d_hold, d_result, d_results, e_base, e_result,
    )


@app.cell
def _(
    a_base, a_prediction, a_result, b_comparison, b_prediction,
    c_hold, c_prediction, c_rapid, c_target, capture_evidence, d_base,
    d_hold, d_prediction, d_result, d_results, development_budget, e_base,
    e_prediction, e_result, escalation, feedback_revisit, late_stage,
    mo, outcome_delay, plan_choice, plan_rejected, release_choice,
    release_rejected, requirement_choice, set_evidence, signal_interval,
    timing_decision, track_id,
):
    def store(part, capture):
        set_evidence(lambda current: {**current, part: capture})

    a_upstream = None
    b_upstream = {"carried_requirement": requirement_choice.value}
    c_upstream = None
    d_upstream = {"carried_iteration_plan": plan_choice.value}
    e_upstream = {"release_decision": release_choice.value}
    a_capture = mo.ui.button(
        label="Capture requirement contrast", kind="success",
        disabled=a_prediction.value is None,
        on_click=lambda _value: store("A", capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"carried_requirement": requirement_choice.value},
            baseline=a_base, result=a_result, alternatives=(a_base, a_result),
            decision=requirement_choice.value, upstream_inputs=a_upstream,
            model_key="v1_03_experiments.trace_requirement",
            chosen_result=a_base, result_role="compared requirement",
        )),
    )
    b_capture = mo.ui.button(
        label="Capture validation-timing contrast", kind="success",
        disabled=b_prediction.value is None or timing_decision.value is None,
        on_click=lambda _value: store("B", capture_evidence(
            track=track_id, part="B", prediction=b_prediction.value,
            inputs={"later_stage": late_stage.value,
                    "escalation_factor": escalation.value,
                    "decision": timing_decision.value},
            baseline=b_comparison["earlier"], result=b_comparison["later"],
            alternatives=(b_comparison["earlier"], b_comparison["later"]),
            decision=timing_decision.value, upstream_inputs=b_upstream,
            model_key="v1_03_experiments.validation_timing",
            chosen_result=(b_comparison["earlier"] if timing_decision.value == "early"
                           else b_comparison["later"]),
            result_role="later discovery condition",
        )),
    )
    _plan_invalid = (
        c_prediction.value is None or plan_choice.value is None
        or plan_rejected.value is None
        or (plan_choice.value != "none" and plan_choice.value == plan_rejected.value)
    )
    if plan_choice.value == "rapid_offline":
        _c_baseline, _c_result = c_target, c_rapid
        _c_chosen, _c_role = c_rapid, "selected candidate"
    elif plan_choice.value == "target_in_loop":
        _c_baseline, _c_result = c_rapid, c_target
        _c_chosen, _c_role = c_target, "selected candidate"
    else:
        _c_rejected = plan_rejected.value or "rapid_offline"
        _c_result = c_rapid if _c_rejected == "rapid_offline" else c_target
        _c_baseline = c_hold
        _c_chosen, _c_role = c_hold, "rejected alternative"
    c_capture = mo.ui.button(
        label="Capture fixed-budget comparison", kind="success", disabled=_plan_invalid,
        on_click=lambda _value: store("C", capture_evidence(
            track=track_id, part="C", prediction=c_prediction.value,
            inputs={"development_budget_days": development_budget.value,
                    "choice": plan_choice.value, "rejected": plan_rejected.value},
            baseline=_c_baseline, result=_c_result, alternatives=(c_rapid, c_target, c_hold),
            decision=plan_choice.value, upstream_inputs=c_upstream,
            model_key="v1_03_experiments.simulate_iterations",
            chosen_result=_c_chosen, result_role=_c_role,
        )),
    )
    _release_invalid = (
        d_prediction.value is None or release_choice.value is None
        or release_rejected.value is None
        or (release_choice.value != "hold" and release_choice.value == release_rejected.value)
    )
    if release_choice.value in d_results:
        _d_chosen, _d_role = d_results[release_choice.value], "selected check plan"
    else:
        _d_chosen, _d_role = d_hold, "rejected alternative"
    d_capture = mo.ui.button(
        label="Capture release-evidence comparison", kind="success",
        disabled=_release_invalid,
        on_click=lambda _value: store("D", capture_evidence(
            track=track_id, part="D", prediction=d_prediction.value,
            inputs={"choice": release_choice.value, "rejected": release_rejected.value},
            baseline=d_base, result=d_result, alternatives=(*tuple(d_results.values()), d_hold),
            decision=release_choice.value, upstream_inputs=d_upstream,
            model_key="v1_03_experiments.evaluate_release_checks",
            chosen_result=_d_chosen, result_role=_d_role,
        )),
    )
    e_capture = mo.ui.button(
        label="Capture production-feedback contrast", kind="success",
        disabled=e_prediction.value is None or feedback_revisit.value is None,
        on_click=lambda _value: store("E", capture_evidence(
            track=track_id, part="E", prediction=e_prediction.value,
            inputs={"signal_check_interval_hours": signal_interval.value,
                    "outcome_delay_hours": outcome_delay.value,
                    "revisit_stage": feedback_revisit.value},
            baseline=e_base, result=e_result, alternatives=(e_base, e_result),
            decision=feedback_revisit.value, upstream_inputs=e_upstream,
            model_key="v1_03_experiments.feedback_timeline",
            chosen_result=e_result, result_role="selected observation policy",
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
    .pilot-head{background:linear-gradient(135deg,#101827,#28496b);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin:30px 0 14px}
    .pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:780px}
    .pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-value{color:#fff}.lab-hud .hud-active{color:#86efac}
    .table-wrap{max-width:100%;overflow-x:auto}@media(max-width:520px){.pilot-head{border-radius:9px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>VOLUME I · LAB 03</span><span>ABOUT 50–55 MIN</span></div>
    <h1>Carry the Constraint</h1><p>When should evidence interrupt the ML workflow, and what must production send backward?</p>
    <div class="pilot-meta"><div><b>Track</b><br>{profile['display']}</div><div><b>Context</b><br>{profile['scenario']}</div><div><b>Deliverable</b><br>Workflow recommendation with a release rule</div></div></section>""")
    mo.vstack([
        LAB_CSS, ACADEMIC_LAB_CSS, css, header, track,
        mo.Html('<p class="pilot-note">All workflow costs, defect fixtures, and task outcomes are illustrative scenario assumptions. Open Calculation Notes in each part to inspect their scope.</p>'),
    ], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(
    COLORS, a_base, a_capture, a_prediction, a_result, a_upstream,
    apply_plotly_theme, audit_evidence, b_capture, b_comparison,
    b_prediction, b_upstream, c_capture, c_comparison, c_prediction,
    c_rapid, c_target, c_upstream, d_capture, d_prediction, d_result,
    d_results, d_upstream, development_budget, e_base, e_capture,
    e_prediction, e_result, e_upstream, escalation, feedback_revisit,
    final_choice, final_rejected, final_risk, final_trigger, get_evidence,
    go, late_stage, mo, outcome_delay, plan_choice, plan_rejected,
    profile, rationale, release_choice, release_rejected, requirement_choice,
    signal_interval, timing_decision, track_id,
):
    _captures = get_evidence()
    _upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream,
                 "D": d_upstream, "E": e_upstream}
    audit = audit_evidence(
        _captures, track=track_id, required_parts=tuple("ABCDE"),
        per_part_upstream_inputs=_upstream,
        contrast_required_parts=tuple("ABCDE"),
    )

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style(
            {"max-width": "100%", "overflow-x": "auto"}
        )

    def saved(part):
        _capture = _captures.get(part)
        if _capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md(
                "**STALE OR NON-CONTRASTING EVIDENCE.** A carried decision changed, or the saved runs used identical inputs. Recapture this part."
            ), kind="danger")
        _snapshot = _capture.to_dict()
        return mo.Html(
            f'<div class="saved"><b>Saved snapshot</b> · original prediction: {_snapshot["prediction"]}'
            f'<br><small>Track {_snapshot["track"]}; later live-control changes do not rewrite this record.</small></div>'
        )

    def part_a():
        _intro = mo.md(
            f"### A · Which requirement changes the development plan? (8 min)\n"
            f"Development context: **{profile['scenario']}**. Compare **{a_base['requirement_label']}** ({a_base['requirement_limit']}) with "
            f"**{a_result['requirement_label']}** ({a_result['requirement_limit']}) before choosing which one the team must carry through every stage."
        )
        if a_prediction.value is None:
            return mo.vstack([_intro, a_prediction])
        _stage_names = tuple(a_base["affected_stage_counts"])
        _fig = go.Figure()
        _fig.add_bar(name=a_base["requirement_label"], x=_stage_names,
                     y=tuple(a_base["affected_stage_counts"].values()), marker_color=COLORS["BlueLine"])
        _fig.add_bar(name=a_result["requirement_label"], x=_stage_names,
                     y=tuple(a_result["affected_stage_counts"].values()), marker_color=COLORS["OrangeLine"])
        _fig.update_layout(height=270, margin=dict(l=25, r=20, t=20, b=55),
                           barmode="group", yaxis_title="Affected lifecycle artifacts",
                           legend_orientation="h")
        _rows = [{
            "Requirement": result["requirement_label"],
            "Limit": result["requirement_limit"],
            "Seeded defect": result["seeded_defect"],
            "Affected artifacts": ", ".join(result["affected_artifacts"]),
        } for result in (a_base, a_result)]
        return mo.vstack([
            _intro, a_prediction, apply_plotly_theme(_fig), table(_rows),
            requirement_choice,
            mo.callout(mo.md(
                f"**Your prediction:** {a_prediction.value}. Carrying **{a_base['requirement_label']}** changes only artifacts reachable from that requirement in the lifecycle graph."
            ), kind="info"),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md(
                "The instrument traverses a small explicit dependency graph. An artifact appears only when it depends directly or indirectly on the selected requirement; unrelated roots are preserved as a control."
            )}),
        ])

    def part_b():
        _intro = mo.md(
            f"### B · When should we pay to check it? (10 min)\n"
            f"Development context: **{profile['scenario']}**. The carried requirement is **{a_base['requirement_label']}** ({a_base['requirement_limit']}). Predict before comparing a data-stage check with a later check."
        )
        if b_prediction.value is None:
            return mo.vstack([_intro, late_stage, escalation, b_prediction])
        _early, _late = b_comparison["earlier"], b_comparison["later"]
        _fig = go.Figure()
        _fig.add_bar(name="Inspection", x=[_early["discovery_stage"], _late["discovery_stage"]],
                     y=[_early["inspection_days"], _late["inspection_days"]], marker_color=COLORS["BlueLine"])
        _fig.add_bar(name="Rework", x=[_early["discovery_stage"], _late["discovery_stage"]],
                     y=[_early["rework_days"], _late["rework_days"]], marker_color=COLORS["RedLine"])
        _fig.update_layout(barmode="stack", height=285, margin=dict(l=25, r=20, t=20, b=35),
                           yaxis_title="Person-days", legend_orientation="h")
        _rows = [{
            "Discovery": result["discovery_stage"],
            "Inspection days": result["inspection_days"],
            "Artifacts reopened": result["incurred_artifact_count"],
            "Direct rework days": result["base_rework_days"],
            "Scenario rework days": result["rework_days"],
            "Total response days": result["total_response_days"],
        } for result in (_early, _late)]
        return mo.vstack([
            _intro, b_prediction, mo.hstack([late_stage, escalation], widths="equal", wrap=True),
            apply_plotly_theme(_fig), table(_rows),
            mo.callout(mo.md(
                f"**Avoidable response time:** {b_comparison['avoidable_days']:.2f} person-days. This is a sum of work on artifacts already created, with the visible sensitivity multiplier applied."
            ), kind="danger" if b_comparison["avoidable_days"] > 0 else "info"),
            timing_decision, b_capture, saved("B"),
            mo.accordion({"Calculation Notes": mo.md(
                "Base rework is the sum of the displayed affected-artifact costs incurred by the discovery stage. The multiplier is a learner-selected sensitivity assumption; stage number does not generate an exponential cost law."
            )}),
        ])

    def part_c():
        _intro = mo.md(
            f"### C · Does faster iteration produce a better result? (10 min)\n"
            f"Development context: **{profile['scenario']}** carrying **{a_base['requirement_label']}**. Both plans receive the same {development_budget.value}-day budget and follow supplied candidate outcomes for this matched task."
        )
        if c_prediction.value is None:
            return mo.vstack([_intro, development_budget, c_prediction])
        _fig = go.Figure([go.Bar(
            x=["Rapid offline", "Target in loop"],
            y=[c_rapid["completed_iterations"], c_target["completed_iterations"]],
            marker_color=[COLORS["OrangeLine"], COLORS["GreenLine"]],
        )])
        _fig.update_layout(height=260, margin=dict(l=25, r=20, t=20, b=35),
                           yaxis_title="Completed iterations", showlegend=False)
        _rows = [{
            "Plan": "Rapid offline" if result["plan_id"] == "rapid_offline" else "Target in loop",
            "Budget days": result["development_budget_days"],
            "Completed": result["completed_iterations"],
            "Failed target iterations": result["failed_iterations"],
            "Best target task success": "not observed" if result["best_target_task_success_pct"] is None else f"{result['best_target_task_success_pct']:.1f}%",
            "Releasable candidate": "YES" if result["has_releasable_candidate"] else "NO",
        } for result in (c_rapid, c_target)]
        _choice_status = "No tested plan selected" if plan_choice.value == "none" else f"Selected {plan_choice.value}"
        return mo.vstack([
            _intro, c_prediction, development_budget, apply_plotly_theme(_fig), table(_rows),
            mo.callout(mo.md(
                f"**Result:** the rapid plan completes {c_comparison['iteration_count_delta']} more iterations. Candidate release still depends on observed target task success, including unsuccessful attempts."
            ), kind="info"),
            mo.hstack([plan_choice, plan_rejected], widths="equal", wrap=True),
            mo.callout(mo.md(f"**Decision status:** {_choice_status}."),
                       kind="warn" if plan_choice.value == "none" else "success"),
            c_capture, saved("C"),
            mo.accordion({"Calculation Notes": mo.md(
                "The simulator walks the supplied candidate trajectory until the next iteration would exceed the time budget. Task-success percentages are illustrative observations for this one scenario, not a universal quality equation."
            )}),
        ])

    def part_d():
        _intro = mo.md(
            f"### D · What evidence makes release defensible? (10 min)\n"
            f"Development context: **{profile['scenario']}** carrying **{a_base['requirement_label']}**. The checks receive the same seeded defects: **{', '.join(d_results['combined']['seeded_defects'])}**. Predict each check's blind spot before seeing the matrix."
        )
        if d_prediction.value is None:
            return mo.vstack([_intro, d_prediction])
        _fig = go.Figure()
        _fig.add_bar(
            name="Detected", x=["Offline", "Target", "Combined"],
            y=[d_results[key]["detected_defect_count"] for key in ("offline", "target", "combined")],
            marker_color=COLORS["GreenLine"],
        )
        _fig.add_bar(
            name="Escaped", x=["Offline", "Target", "Combined"],
            y=[d_results[key]["escaped_defect_count"] for key in ("offline", "target", "combined")],
            marker_color=COLORS["RedLine"],
        )
        _fig.update_layout(barmode="stack", height=280, margin=dict(l=25, r=20, t=20, b=35),
                           yaxis_title="Seeded defects", legend_orientation="h")
        _rows = [{
            "Check plan": key.title(),
            "Inspection days": result["inspection_days"],
            "Detected": ", ".join(result["detected_defects"]),
            "Escaped": ", ".join(result["escaped_defects"]) or "none",
            "Decision": result["decision"].upper(),
        } for key, result in d_results.items()]
        return mo.vstack([
            _intro, d_prediction, apply_plotly_theme(_fig), table(_rows),
            mo.hstack([release_choice, release_rejected], widths="equal", wrap=True),
            mo.callout(mo.md(
                f"**Compared live result:** {d_result['check_plan']} detects {d_result['detected_defect_count']} of {d_result['seeded_defect_count']} seeded defects and leaves {d_result['escaped_defect_count']} visible blind spots."
            ), kind="success" if d_result["release_defensible"] else "danger"),
            d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md(
                "Each check has an explicit inspection time and detection set. Combined evidence is the union of offline and target detections. Passing covers only the named seeded defects; it cannot prove that no other defect exists."
            )}),
        ])

    def part_e():
        _intro = mo.md(
            f"### E · What should production send backward? (8 min)\n"
            f"Development context: **{profile['scenario']}** ({e_result['requests_after_change']} post-shift requests carrying **{a_base['requirement_label']}**). A world change, an input alert, and delayed labeled outcomes are different events. Predict what the first signal establishes."
        )
        if e_prediction.value is None:
            return mo.vstack([_intro, signal_interval, outcome_delay, e_prediction])
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=[event["hour"] for event in e_result["timeline"]],
            y=[event["kind"] for event in e_result["timeline"]],
            mode="markers+lines", marker=dict(size=12, color=COLORS["BlueLine"]),
            line=dict(color=COLORS["BlueLine"]),
        ))
        _fig.update_layout(height=285, margin=dict(l=25, r=20, t=20, b=35),
                           xaxis_title="Hours since release", yaxis_title="Event", showlegend=False)
        _rows = [{
            "Policy": "11-hour baseline" if result is e_base else "Selected cadence",
            "Signal interval": f"{result['signal_check_interval_hours']:.0f} h",
            "Signal delay": f"{result['signal_detection_delay_hours']:.1f} h",
            "Checks through evidence": result["inspection_count_through_evidence"],
            "Outcome-label delay": f"{result['outcome_delay_hours']:.0f} h",
            "Adverse outcomes": result["adverse_outcomes_after_change"],
        } for result in (e_base, e_result)]
        return mo.vstack([
            _intro, e_prediction,
            mo.hstack([signal_interval, outcome_delay], widths="equal", wrap=True),
            apply_plotly_theme(_fig), table(_rows),
            mo.callout(mo.md(
                "A shorter interval changes observation delay and inspection count. The fixed request and adverse-outcome counts do not change. An alert revisits validation; labeled outcomes can send the team back to data."
            ), kind="info"),
            feedback_revisit, e_capture, saved("E"),
            mo.accordion({"Calculation Notes": mo.md(
                "Signal checks occur on a fixed cadence after release. The first check at or after the population change supplies the alert time. Outcome evidence appears after the selected label delay. Monitoring observes both timelines and does not alter the underlying outcomes."
            )}),
        ])

    def build_synthesis():
        _rows = []
        for _part in "ABCDE":
            _capture = _captures.get(_part)
            _current = (
                _capture is not None and _part not in audit.stale
                and (_part, _part) not in audit.identical_pairs
            )
            _rows.append({
                "Part": _part,
                "Original prediction": _capture.to_dict()["prediction"] if _capture else "—",
                "Evidence": "CURRENT" if _current else ("STALE" if _capture else "MISSING"),
            })
        _saved_choice = _captures["C"].to_dict()["decision"] if "C" in _captures else None
        _complete = (
            audit.complete
            and all(widget.value is not None for widget in
                    (final_choice, final_rejected, final_trigger, final_risk))
            and bool(rationale.value.strip())
            and final_choice.value != final_rejected.value
            and final_choice.value == _saved_choice
        )
        return mo.vstack([
            mo.md(
                f"### Synthesis · Defend the workflow (5 min)\n"
                f"Development context: **{profile['scenario']}** ({profile['display']} track). "
                "Use saved evidence to state the chosen iteration plan, a quantified rejected plan, the release evidence required, one remaining limitation, and the event that triggers reevaluation."
            ),
            table(_rows),
            mo.callout(mo.md(
                "Saved snapshots remain fixed while live controls move. Recapture evidence marked stale before generating the local report."
            ), kind="info"),
            mo.hstack([final_choice, final_rejected], widths="equal", wrap=True),
            mo.hstack([final_trigger, final_risk], widths="equal", wrap=True),
            rationale,
            mo.callout(mo.md(
                "**Ready for the local report.**" if _complete else
                "Complete five current contrasts, match the recommendation to saved Part C, compare a different tested plan, and add the rationale."
            ), kind="success" if _complete else "warn"),
        ])

    tabs = mo.ui.tabs({
        "Part A": part_a(), "Part B": part_b(), "Part C": part_c(),
        "Part D": part_d(), "Part E": part_e(), "Synthesis": build_synthesis(),
    })
    tabs
    return (audit,)


@app.cell
def _(
    audit, build_lab_report, final_choice, final_rejected, final_risk,
    final_trigger, get_evidence, get_lab_metadata, mo, profile, rationale,
    report_export_panel, track_id,
):
    _captures = get_evidence()
    _saved_choice = _captures["C"].to_dict()["decision"] if "C" in _captures else None
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in
                (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip())
        and final_choice.value != final_rejected.value
        and final_choice.value == _saved_choice
    )
    mo.stop(not _ready)
    _snapshots = {part: _captures[part].to_dict() for part in "ABCDE"}
    _d_chosen = _snapshots["D"].get("chosen_result")
    _d_effective = _d_chosen if _d_chosen is not None else _snapshots["D"]["result"]
    _carried_req = _snapshots["A"]["chosen_result"]["requirement_label"]
    _carried_limit = _snapshots["A"]["chosen_result"]["requirement_limit"]
    report = build_lab_report(
        get_lab_metadata("vol1/lab_03_ml_workflow.py"),
        track=track_id, scenario=profile["scenario"],
        learning_objectives=[
            "Trace a requirement through lifecycle artifacts",
            "Compare validation timing, iteration cadence, and release evidence",
            "Separate production alerts from delayed outcome evidence",
        ],
        predictions={part: _snapshots[part]["prediction"] for part in "ABCDE"},
        knob_settings={part: _snapshots[part]["inputs"] for part in "ABCDE"},
        evidence_summary={part: {
                "baseline": _snapshots[part]["baseline"],
                "result": _snapshots[part]["result"],
                "alternatives": _snapshots[part]["alternatives"],
                "chosen_result": _snapshots[part].get("chosen_result"),
                "result_role": _snapshots[part].get("result_role"),
            } for part in "ABCDE"},
        binding_constraints={
            "development_context": profile["scenario"],
            "carried_requirement": f"{_carried_req} ({_carried_limit})",
            "rework": _snapshots["B"]["result"]["incurred_artifacts"],
            "release_blind_spots": _d_effective.get("escaped_defects", ("release held",)),
        },
        decisions={
            "carried_requirement": _carried_req,
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "release_decision": _snapshots["D"]["decision"],
            "reevaluation_trigger": final_trigger.value,
        },
        final_decision={
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "rationale": rationale.value,
        },
        big_takeaways=[
            "Requirements propagate through explicit lifecycle dependencies.",
            "More iterations do not guarantee a better target outcome.",
            "Alerts start investigation; labeled outcomes establish task harm.",
        ],
        reflections={"rationale": rationale.value, "remaining_limitation": final_risk.value},
        residual_risk=final_risk.value,
        result_snapshot={
            "track": track_id,
            "development_context": profile["scenario"],
            "carried_requirement": _carried_req,
            "captures": _snapshots,
            "recommendation": final_choice.value,
            "rejected_alternative": final_rejected.value,
            "reevaluation_trigger": final_trigger.value,
            "residual_risk": final_risk.value,
        },
        source_trace={
            "scenario": "Illustrative Chapter 3 workflow fixtures.",
            "calculations": "MLSysIM workflow experiment results.",
        },
    )
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(
    audit, final_choice, final_rejected, final_risk, final_trigger,
    get_evidence, ledger, mo, rationale, track_id,
):
    _captures = get_evidence()
    _saved_choice = _captures["C"].to_dict()["decision"] if "C" in _captures else None
    _ready = (
        audit.complete
        and all(widget.value is not None for widget in
                (final_choice, final_rejected, final_trigger, final_risk))
        and bool(rationale.value.strip())
        and final_choice.value != final_rejected.value
        and final_choice.value == _saved_choice
    )
    _status = "EVIDENCE IN PROGRESS"
    if _ready:
        try:
            ledger.save(chapter=3, design={
                "schema_version": 1,
                "lab_id": "v1_03",
                "track_id": track_id,
                "model_id": "v1_03_experiments",
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
    mo.Html(
        f'<div class="lab-hud"><span>LAB 03 · Carry the Constraint · STATUS: {_status}</span></div>'
    )
    return


if __name__ == "__main__":
    app.run()
