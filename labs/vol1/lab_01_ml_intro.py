import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 01: Evidence Before Optimization · MLSysBook")


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
    from mlsysim.engine.constraint_explorer import (
        INTRO_SCENARIO_VERSION, INTRO_TASK_OPTIONS, evaluate,
        introductory_profile, speedup, task_factors,
    )
    from mlsysim.core.units import Q_
    from mlsysim.fmt import fmt, fmt_count, fmt_energy, fmt_memory, fmt_percent, fmt_time
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, apply_plotly_theme
    from mlsysbook_labs.notebook import notebook_bridge, notebook_header, notebook_shell, notebook_status, notebook_table
    from mlsysbook_labs import build_lab_report, get_lab_metadata, report_export_panel
    from mlsysbook_labs.contexts import CONTEXT_OPTIONS, get_context, preferred_context
    from mlsysbook_labs.experiment_evidence import capture_evidence, audit_evidence
    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (
        COLORS, CONTEXT_OPTIONS, INTRO_SCENARIO_VERSION, INTRO_TASK_OPTIONS,
        Q_, apply_plotly_theme, audit_evidence, build_lab_report,
        capture_evidence, evaluate, fmt, fmt_count, fmt_energy, fmt_memory,
        fmt_percent, fmt_time, get_context, get_lab_metadata, go,
        introductory_profile, ledger, mo, notebook_bridge, notebook_header, notebook_shell, notebook_status, notebook_table,
        preferred_context, report_export_panel,
        speedup, task_factors,
    )



@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    get_pinned, set_pinned = mo.state({})
    return get_evidence, get_pinned, set_evidence, set_pinned


@app.cell
def _(CONTEXT_OPTIONS, get_context, ledger, mo, preferred_context, set_evidence, set_pinned):
    def reset_context(_value):
        set_evidence({})
        set_pinned({})
    track = mo.ui.dropdown(CONTEXT_OPTIONS, value=get_context(preferred_context(ledger)).label,
                           label="Deployment context", on_change=reset_context)
    return (track,)


@app.cell
def _(get_context, introductory_profile, track):
    track_id = track.value
    context = get_context(track_id)
    profile = introductory_profile(track_id)
    project = context.scenario
    episode = project.episode(1)
    return context, episode, profile, project, track_id


@app.cell
def _(INTRO_TASK_OPTIONS, mo, profile, track_id):
    _context_reset = track_id
    a_shift = mo.ui.slider(steps=list(profile["shift_options"]), value=profile["shift_options"][-2],
                           label="Increase difficult-cohort share (percentage points)")
    a_labels = mo.ui.checkbox(label="Reveal the supplied labeled-outcome study")
    a_interpretation = mo.ui.radio(["Investigate population coverage", "Investigate execution", "Evidence remains insufficient"],
                                    label="What should we investigate first?")
    b_term = mo.ui.dropdown({"Data movement": "movement", "Arithmetic": "compute", "Fixed overhead": "overhead"},
                            value="Arithmetic", label="Improve one execution term")
    b_scale = mo.ui.slider(steps=list(profile["capability_options"]), value=profile["capability_options"][1],
                           label="Capability multiplier (×; only selected term changes)")
    c_candidate = mo.ui.dropdown({name.title(): name for name in profile["candidates"]},
                                 value="Balanced", label="Candidate to investigate")
    c_rejected = mo.ui.dropdown({name.title(): name for name in profile["candidates"]},
                                value="Large", label="Compared alternative")
    c_decision = mo.ui.radio(["Advance this candidate for further validation", "Hold: requirements or evidence unresolved"],
                             label="Deployment recommendation")
    d_card = mo.ui.dropdown(INTRO_TASK_OPTIONS, value=next(iter(INTRO_TASK_OPTIONS)), label="Host study intervention")
    d_choice = mo.ui.dropdown({"Keep baseline / no action": "baseline", **INTRO_TASK_OPTIONS},
                              value="Keep baseline / no action", label="Which tested host option would you retain?")
    final_trigger = mo.ui.radio(["Population or outcome evidence changes", "Execution requirements or workload changes", "Host study quality evidence changes"],
                               label="Revisit this recommendation when…")
    rationale = mo.ui.text_area(label="Your discussion: recommendation and evidence",
                               placeholder="Use your saved contrasts: benefit, cost, rejected alternative, and remaining uncertainty.")
    return (a_interpretation, a_labels, a_shift, b_scale, b_term, c_candidate,
            c_decision, c_rejected, d_card, d_choice, final_trigger, rationale)


@app.cell
def _(mo, track_id):
    _context_reset = track_id
    a_prediction = mo.ui.radio(["Quality only", "Execution only", "Both", "Neither"],
                               label="With fixed weights and execution, what might fail under the stated population change?").form(submit_button_label="Lock A prediction")
    b_prediction = mo.ui.radio(["Movement", "Arithmetic", "Overhead", "Need more evidence"],
                               label="Which term should give the largest end-to-end benefit at the same capability multiplier?").form(submit_button_label="Lock B prediction")
    c_prediction = mo.ui.radio(["Highest quality is deployable", "Higher quality can violate a resource requirement", "Quality evidence alone is insufficient"],
                               label="Does higher quality establish deployment readiness?").form(submit_button_label="Lock C prediction")
    d_prediction = mo.ui.radio(["Fewer samples", "Fewer operations per sample", "Fewer realized seconds per operation", "Cannot infer quality from any factor alone"],
                               label="Which efficiency change would you investigate first?").form(submit_button_label="Lock D prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction


@app.cell
def _(INTRO_SCENARIO_VERSION, a_shift, b_scale, b_term, c_candidate, c_rejected, d_card, evaluate,
      get_evidence, get_pinned, task_factors, track_id):
    baseline = evaluate(track_id, scenario_version=INTRO_SCENARIO_VERSION)
    a_result = evaluate(track_id, scenario_version=INTRO_SCENARIO_VERSION, shift_pct=a_shift.value)
    b_result = evaluate(track_id, scenario_version=INTRO_SCENARIO_VERSION, **{f"{b_term.value}_scale": b_scale.value})
    captures = get_evidence()
    pins = get_pinned()
    saved_population = captures["A"].to_dict()["result"]["shift_pct"] if "A" in captures else None
    c_results = {name: evaluate(track_id, scenario_version=INTRO_SCENARIO_VERSION, candidate=name, shift_pct=saved_population or 0)
                 for name in ("compact", "balanced", "large")}
    c_result = c_results[c_candidate.value]
    c_other = c_results[c_rejected.value]
    d_base = task_factors(track_id)
    d_result = task_factors(track_id, d_card.value)
    return (a_result, b_result, baseline, c_other, c_result, c_results,
            captures, d_base, d_result, pins, saved_population)


@app.cell
def _(INTRO_SCENARIO_VERSION, a_interpretation, a_labels, a_prediction,
      a_result, a_shift, b_prediction, b_result, b_scale, b_term, baseline,
      c_decision, c_other, c_prediction, c_result, capture_evidence, captures,
      d_base, d_choice, d_prediction, d_result, mo, pins, saved_population,
      set_evidence, set_pinned, task_factors, track_id):
    def store(part, prediction, inputs, base, result, *, alternatives=(), decision=None,
              chosen=None, model_key="constraint_explorer.evaluate", upstream=None):
        snapshot = capture_evidence(track=track_id, part=part, prediction=prediction,
            inputs={"scenario_version": INTRO_SCENARIO_VERSION, **inputs},
            baseline=base, result=result, alternatives=alternatives, decision=decision,
            chosen_result=chosen, upstream_inputs=upstream, model_key=model_key)
        set_evidence(lambda current: {**current, part: snapshot})
    a_upstream = {"version": INTRO_SCENARIO_VERSION}
    b_upstream = {"version": INTRO_SCENARIO_VERSION}
    c_upstream = {"version": INTRO_SCENARIO_VERSION, "saved_population": saved_population}
    d_upstream = {"version": INTRO_SCENARIO_VERSION}
    a_capture = mo.ui.button(label="Save population contrast", kind="success",
        disabled=a_prediction.value is None or not a_labels.value or a_shift.value == 0 or a_interpretation.value is None,
        on_click=lambda _v: store("A", a_prediction.value, {"shift_pct": a_shift.value}, baseline, a_result,
                                  decision=a_interpretation.value, upstream=a_upstream))
    b_pin = mo.ui.button(label="Pin this execution alternative",
        disabled=b_prediction.value is None or b_scale.value == 1,
        on_click=lambda _v: set_pinned(lambda current: {**current, "B": b_result}))
    b_pair = pins.get("B")
    b_different = b_pair is not None and b_pair["inputs"].get(f"{b_term.value}_scale", 1) == 1 and b_scale.value != 1
    b_capture = mo.ui.button(label="Save two execution alternatives", kind="success",
        disabled=b_prediction.value is None or not b_different,
        on_click=lambda _v: store("B", b_prediction.value, {"term": b_term.value, "scale": b_scale.value},
                                  baseline, b_result, alternatives=(b_pair, b_result), upstream=b_upstream))
    c_capture = mo.ui.button(label="Save candidate decision", kind="success",
        disabled=c_prediction.value is None or saved_population is None or c_decision.value is None
                 or c_result["candidate"] == c_other["candidate"]
                 or (c_decision.value.startswith("Advance") and not c_result["feasible"]),
        on_click=lambda _v: store("C", c_prediction.value, {"candidate": c_result["candidate"], "rejected": c_other["candidate"]},
                                  c_other, c_result, alternatives=(c_other, c_result),
                                  decision=c_decision.value, chosen=c_result, upstream=c_upstream))
    d_pin = mo.ui.button(label="Pin this host study",
        disabled=d_prediction.value is None,
        on_click=lambda _v: set_pinned(lambda current: {**current, "D": d_result}))
    d_pair = pins.get("D")
    tested = {"baseline", d_result["card_id"]} | ({d_pair["card_id"]} if d_pair else set())
    d_selected = task_factors(track_id, d_choice.value)
    d_capture = mo.ui.button(label="Save host-study decision", kind="success",
        disabled=d_prediction.value is None or d_pair is None or d_pair["card_id"] == d_result["card_id"]
                 or d_choice.value not in tested or not d_selected["feasible"],
        on_click=lambda _v: store("D", d_prediction.value, {"tested": [d_pair["card_id"], d_result["card_id"]], "choice": d_choice.value},
                                  d_base, d_result, alternatives=(d_pair, d_result), decision=d_choice.value,
                                  chosen=d_selected, model_key="constraint_explorer.task_factors.v1", upstream=d_upstream))
    return (a_capture, a_upstream, b_capture, b_pair, b_pin, b_upstream,
            c_capture, c_upstream, d_capture, d_pair, d_pin, d_selected, d_upstream)


@app.cell
def _(context, episode, get_lab_metadata, mo, notebook_bridge, notebook_header, notebook_shell, project, track):
    from mlsysbook_labs.notebook import notebook_scenario as _scenario
    from mlsysbook_labs.learning_paths import get_learning_path
    from mlsysbook_labs.notebook import notebook_learning_map
    _learning = get_learning_path(1, 1)
    _learning["next_lab"]["question"] = project.episode(2).question
    notebook_shell([
        notebook_header(get_lab_metadata("vol1/lab_01_ml_intro.py"),
            episode.question,
            label="Lab 01 · Introduction", topics=("Outcome evidence", "Execution costs", "Deployment requirements", "Efficiency levers")),
        mo.hstack([track, mo.md(f"**Project:** {project.title}")], justify="start", gap=1, wrap=True),
        _scenario(_learning, track=track.value, scene=episode.scene),
        mo.md("**Starting case:** supplied mechanism studies. Earlier lab completion is optional."),
        notebook_learning_map(_learning),
        mo.accordion({"Project brief and reading": mo.md(f"**This episode:** {episode.task}\n\n{project.mission}\n\n{project.study_boundary}\n\n**Read first:** the introduction’s sections on silent degradation, the iron law, and three dimensions of efficiency. [Open the chapter](https://mlsysbook.ai/vol1/introduction/introduction.html).\n\nThese are supplied illustrative studies, not device benchmarks. Any prediction unlocks exploration. Capture contrasting runs, explain your decision in Synthesis, and download the report. A justified hold is a valid conclusion. Your project continues even in a fresh session; prior chapter completion is not required. Changing context resets this lab’s captures and predictions rather than carrying another project’s evidence forward.")}),
    ])
    return



@app.cell
def _(COLORS, Q_, a_capture, a_interpretation, a_labels, a_prediction,
      a_result, a_shift, a_upstream, apply_plotly_theme, audit_evidence,
      b_capture, b_pair, b_pin, b_prediction, b_result, b_scale, b_term,
      b_upstream, baseline, c_candidate, c_capture, c_decision, c_other,
      c_prediction, c_rejected, c_result, c_results, c_upstream, captures,
      d_base, d_capture, d_card, d_choice, d_pair, d_pin, d_prediction,
      d_result, d_selected, d_upstream, final_trigger, fmt, fmt_count,
      fmt_energy, fmt_memory, fmt_percent, fmt_time, go, mo, profile, project, episode,
      rationale, saved_population, speedup, track_id, notebook_shell, notebook_bridge, notebook_table):
    upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream}
    audit = audit_evidence(captures, track=track_id, required_parts=tuple("ABCD"),
        per_part_upstream_inputs=upstream, contrast_required_parts=tuple("ABCD"))
    def table(rows):
        return notebook_table(rows)
    def time(value, unit="millisecond"):
        return fmt_time(Q_(value, unit), unit=unit, precision=None)
    def percent(value):
        return "Unknown" if value is None else fmt_percent(value / 100, style="symbol")
    def memory(value):
        return fmt_memory(Q_(value, "megabyte"), unit="megabyte", precision=None)
    def energy(value):
        return fmt_energy(Q_(value, "millijoule"), unit="millijoule", precision=None)
    def verdict(result):
        return "Requirements pass" if result["feasible"] else "FAIL · " + ", ".join(result["violations"])
    def saved(part):
        if part not in captures:
            return mo.md("*No saved contrast yet.*")
        if part in audit.stale:
            return mo.callout("Saved evidence is stale: an upstream saved population changed. Revisit and recapture.", kind="warn")
        return mo.callout(f"Saved contrast · original prediction: {captures[part].to_dict()['prediction']}", kind="success")
    def latency_chart(rows):
        fig = go.Figure()
        for label, result in rows:
            for field, color in (("movement_ms", "BlueLine"), ("compute_ms", "OrangeLine"), ("overhead_ms", "GreenLine")):
                fig.add_bar(x=[label], y=[result[field]], name=field.removesuffix("_ms").title(),
                    marker_color=COLORS[color], legendgroup=field, showlegend=label == rows[0][0])
        fig.add_hline(y=baseline["latency_limit_ms"], line_dash="dash", annotation_text="Deadline")
        apply_plotly_theme(fig)
        fig.update_layout(barmode="stack", yaxis_title="Sequential execution time (ms)",
                          height=350, margin=dict(l=40, r=20, t=25, b=100),
                          legend=dict(orientation="h", yanchor="top", y=-0.24,
                                      xanchor="center", x=0.5))
        return fig
    def investigate(prompts):
        return mo.accordion({"Investigate before you decide": mo.md("\n\n".join(f"- {prompt}" for prompt in prompts))})
    def part_a():
        opening = mo.md(f"### A · Healthy runtime, changed behavior (9 min)\nKeep the **balanced model, weights, and execution** fixed. Compare {profile['cohorts'][0]} with {profile['cohorts'][1]}. The slider changes their mix, not either cohort's supplied quality rate.")
        if a_prediction.value is None:
            return mo.vstack([opening, a_shift, a_prediction])
        mix = go.Figure([go.Bar(x=["Baseline", "Changed"], y=[baseline["difficult_share_pct"], a_result["difficult_share_pct"]], marker_color=COLORS["OrangeLine"])])
        mix.update_layout(yaxis_title="Difficult-cohort share (%)", height=230, margin=dict(l=25, r=20, t=15, b=25))
        items = [opening, a_shift, a_prediction, apply_plotly_theme(mix), a_labels,
            investigate(("Before revealing outcomes, what can the population change establish—and what is still unknown?", "Find a changed mixture with healthy runtime. Reveal outcomes: does the task still meet its requirement?", f"For {project.title.lower()}, what project-specific labeled outcomes would you collect before claiming the same failure mechanism?"))]
        if not a_labels.value:
            items.append(mo.callout("The population changed. Predictive quality remains unknown until outcome evidence is available. Execution remains unchanged.", kind="info"))
            return mo.vstack(items)
        quality = go.Figure([go.Bar(x=["Baseline", "Changed"], y=[baseline["quality_pct"], a_result["quality_pct"]], marker_color=[COLORS["BlueLine"], COLORS["OrangeLine"]])])
        quality.add_hline(y=baseline["quality_floor_pct"], line_dash="dash", annotation_text="Quality floor")
        quality.update_layout(yaxis_title="Supplied weighted quality (%)", height=250, margin=dict(l=25, r=20, t=30, b=25))
        items.extend([apply_plotly_theme(quality), table([{"Run": label, "Quality": percent(r["quality_pct"]), "Latency": time(r["latency_ms"]), "Runtime": "PASS" if r["runtime_feasible"] else "FAIL"} for label, r in (("Baseline", baseline), ("Changed", a_result))]),
            mo.callout(f"Your prediction: {a_prediction.value}. Supplied cohort rates give {percent(a_result['quality_pct'])} quality; runtime is unchanged. A population warning alone would not establish this loss.", kind="danger" if not a_result["feasible"] else "info"), a_interpretation, a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md(f"Weighted quality uses population shares and supplied cohort rates: ordinary **{percent(baseline['easy_quality_pct'])}**, difficult **{percent(baseline['difficult_quality_pct'])}**. This is an illustrative outcome study, not a universal drift-to-accuracy law. Move back toward the original mixture to recover; save the population you want to investigate in Part C.")})])
        return mo.vstack(items)
    def part_b():
        opening = mo.md("### B · Improve the right part of time (11 min)\nDouble or quadruple one capability. What remains untouched? Pin one alternative, then test a different one. These counterfactuals show possible benefit, not the price or power of an actual upgrade.")
        pre = table([{"Movement": time(baseline["movement_ms"]), "Arithmetic": time(baseline["compute_ms"]), "Overhead": time(baseline["overhead_ms"])}])
        if b_prediction.value is None:
            return mo.vstack([opening, pre, b_prediction])
        rows = [("Baseline", baseline)] + ([("Pinned alternative", b_pair)] if b_pair else []) + [("Live alternative", b_result)]
        return mo.vstack([opening, b_prediction, mo.hstack([b_term, b_scale], wrap=True), latency_chart(rows),
            mo.callout(f"Your prediction: {b_prediction.value}. Live end-to-end speedup: {fmt(speedup(baseline, b_result), precision=None)}×. Only the selected term changed; the untouched terms cap the benefit.", kind="info"),
            b_pin, b_capture, saved("B"), investigate(("Pin an arithmetic improvement, then give movement the same multiplier. Which yields more end-to-end benefit?", "Increase the winning capability again. Which untouched cost now limits the improvement?", "Would this evidence justify buying an upgrade? Name the missing cost or power evidence before recommending one.")), mo.accordion({"Calculation Notes": mo.md("Sequential time = bytes moved / bandwidth + operations / effective arithmetic rate + fixed overhead. A capability multiplier divides only its term. This experiment does not model queues, overlap, upgrade cost, or hardware energy changes. Pinning preserves settings while the live comparison changes.")})])
    def part_c():
        opening = mo.md("### C · Better quality, deployable system? (10 min)\nAll requirements stay fixed. A higher-quality candidate still has to fit memory, latency, and energy limits. Compare two candidates, then choose advance or hold.")
        if saved_population is None:
            return mo.vstack([opening, mo.callout("Save the labeled population contrast in Part A first. This part uses that snapshot, not the current slider.", kind="warn")])
        if c_prediction.value is None:
            return mo.vstack([opening, mo.md(f"Saved difficult-cohort increase: **{fmt(saved_population, precision=0)} percentage points**."), c_prediction])
        fig = go.Figure()
        for name, result in c_results.items():
            fig.add_scatter(x=[result["latency_ms"]], y=[result["quality_pct"]], mode="markers+text", text=[name.title()], textposition="top center", name=name.title(), marker=dict(size=14, symbol="circle" if result["feasible"] else "x"))
        fig.add_vline(x=baseline["latency_limit_ms"], line_dash="dash", annotation_text="Deadline")
        fig.add_hline(y=baseline["quality_floor_pct"], line_dash="dash", annotation_text="Quality floor")
        fig.update_layout(xaxis_title="Modeled request latency (ms)", yaxis_title="Supplied quality (%)", height=310, margin=dict(l=30, r=30, t=30, b=30), showlegend=False)
        rows = [{"Candidate": name.title(), "Quality / floor": f"{percent(r['quality_pct'])} / {percent(r['quality_floor_pct'])}", "Memory / limit": f"{memory(r['memory_mb'])} / {memory(r['memory_limit_mb'])}", "Latency / limit": f"{time(r['latency_ms'])} / {time(r['latency_limit_ms'])}", "Energy / limit": f"{energy(r['energy_mj'])} / {energy(r['energy_limit_mj'])}", "Outcome": verdict(r)} for name, r in c_results.items()]
        return mo.vstack([opening, c_prediction, mo.md(f"Using saved population increase: **{fmt(saved_population, precision=0)} points**. Recapturing A updates this experiment and makes its old decision stale."),
            apply_plotly_theme(fig), table(rows), mo.hstack([c_candidate, c_rejected], wrap=True),
            mo.callout(f"Your prediction: {c_prediction.value}. Selected candidate: {c_result['candidate']} · {verdict(c_result)}.", kind="success" if c_result["feasible"] else "danger"), c_decision, c_capture, saved("C"),
            investigate(("Compare a higher-quality candidate with a smaller alternative under exactly the same saved population and requirements.", "Find the binding requirement for each rejected candidate. Would relaxing a different requirement actually help?", f"What additional validation would {project.stakeholder} need before a modeled pass could become a deployment decision?")),
            mo.accordion({"Calculation Notes": mo.md("Every requirement must pass. Quality rates are supplied for this task and population. Candidate size changes illustrative work, traffic, and memory together; it does not derive accuracy. Energy is a bounded phase-cost assumption, not measured device battery life. If no candidate passes, hold and state which evidence or requirement must change.")})])
    def part_d():
        opening = mo.md("### D · Three routes to efficiency (11 min)\nSwitch to a **separate illustrative single-host training study**. Every deployment context uses this same study. It does not train on the endpoint or improve Part C's candidate automatically. Which task factor changes, and does supplied quality evidence justify keeping the change?")
        if d_prediction.value is None:
            return mo.vstack([opening, d_prediction])
        rows = [("Baseline", d_base)] + ([("Pinned study", d_pair)] if d_pair else []) + [("Live study", d_result)]
        fig = go.Figure([go.Bar(x=[label for label, _r in rows], y=[r["duration_seconds"] for _label, r in rows], marker_color=COLORS["BlueLine"])])
        fig.update_layout(yaxis_title="Illustrative task duration (s)", height=240, margin=dict(l=25, r=20, t=15, b=25))
        factor_rows = [{"Study": label, "Changed factor": r["changed_factor"], "Samples/task": fmt_count(r["samples_per_task"]), "Operations/sample": fmt_count(r["operations_per_sample"]), "Realized time/operation": fmt_time(Q_(r["seconds_per_operation"], "second"), unit="nanosecond", precision=None, per="operation"), "Duration": time(r["duration_seconds"], "second"), "Quality / floor": f"{percent(r['quality_pct'])} / {percent(r['quality_floor_pct'])}", "Outcome": verdict(r)} for label, r in rows]
        return mo.vstack([opening, d_prediction, d_card, apply_plotly_theme(fig), table(factor_rows),
            mo.callout(f"Your prediction: {d_prediction.value}. Live study changes {d_result['changed_factor']}. {verdict(d_result)}. Equal task durations do not imply equal quality evidence.", kind="success" if d_result["feasible"] else "danger"),
            d_pin, mo.md("Pin one card and inspect another before saving. Keeping the original host study is a valid no-action decision."), d_choice,
            mo.callout(f"Selected host option: {d_selected['label']} · {verdict(d_selected)}", kind="info"), d_capture, saved("D"),
            investigate(("Pin one intervention, then find another with a different changed task factor. Compare duration and supplied quality together.", "Inspect the option with missing quality evidence. Can its duration establish that it is the better choice?", "Choose an intervention or keep baseline. Explain what host-study evidence transfers to your deployment project and what does not.")),
            mo.accordion({"Calculation Notes": mo.md("Task time = samples/task × operations/sample × realized seconds/operation. The last factor includes execution costs, so movement is not added again. Quality is supplied independently; missing evaluation stays unknown. These cards do not simulate training or selection overhead, predict accuracy, or establish an optimization recipe.")})])
    def synthesis():
        rows = [{"Part": p, "Saved prediction": captures[p].to_dict()["prediction"] if p in captures else "—", "Evidence": "STALE" if p in audit.stale else ("CURRENT" if p in captures else "MISSING")} for p in "ABCD"]
        c_saved = captures["C"].to_dict() if "C" in captures else None
        d_saved = captures["D"].to_dict() if "D" in captures else None
        decisions = mo.md(f"**Deployment:** {c_saved['decision'] if c_saved else 'not captured'}. **Host study:** {d_saved['decision'] if d_saved else 'not captured'}. These are separate decisions; the host cards do not repair deployment evidence.")
        return mo.vstack([mo.md("### Synthesis · Defend the next step (6 min)\nUse the saved results to connect **requirement → evidence → benefit and cost → rejected alternative → next investigation**. A well-supported hold completes the lab."),
            table(rows), decisions,
            notebook_bridge("Current outcome, execution and candidate comparisons." if audit.complete else "Finish the missing or stale comparisons.",
                "Target validation and the transfer from host study to deployment.",
                project.episode(2).question, labels=("Carry forward", "Still open", "Next")),
            mo.accordion({"Project handoff": mo.md(episode.handoff)}),
            mo.md(f"In your discussion, distinguish what the supplied studies establish from what remains untested for **{project.title.lower()}**. Include one tempting intervention you rejected and a next measurement that could change your recommendation."), final_trigger, rationale,
            mo.callout("Four current contrasts are ready. Add your recommendation and revisit trigger to export and save." if audit.complete else "Finish missing or stale contrasts. Changing live controls does not rewrite saved evidence.", kind="success" if audit.complete else "warn"),
            mo.md("**Takeaways:** runtime health is not quality evidence; a component speedup leaves untouched time; better quality must fit fixed requirements; efficiency factors do not predict accuracy.\n\n**Next:** the system-paradigms lab tests where inference should run. TinyTorch builds model machinery; StaffML rehearses explaining your decisions.")])
    tabs = mo.ui.tabs({"Part A": part_a(), "Part B": part_b(), "Part C": part_c(), "Part D": part_d(), "Synthesis": synthesis()})
    ready = audit.complete and final_trigger.value is not None and bool(rationale.value.strip())
    from mlsysbook_labs.notebook import notebook_tab_hint as _tab_hint
    notebook_shell([_tab_hint(), tabs])
    return audit, ready


@app.cell(hide_code=True)
def _():
    from mlsysbook_labs.learning_paths import get_learning_path as _submission_path
    from mlsysbook_labs.notebook import notebook_shell as _submission_shell, notebook_submission as _submission
    _submission_shell([_submission(_submission_path(1, 1))])
    return


@app.cell
def _(INTRO_SCENARIO_VERSION, build_lab_report, captures, context, final_trigger,
      get_lab_metadata, mo, notebook_shell, project, rationale, ready, report_export_panel, save_discussion, track_id):
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not ready, _download_readiness(volume=1))
    snapshots = {part: captures[part].to_dict() for part in "ABCD"}
    final_decision = {"deployment": snapshots["C"]["decision"],
                      "candidate": snapshots["C"]["chosen_result"]["candidate"],
                      "host_study": snapshots["D"]["decision"],
                      "rationale": rationale.value, "revisit_trigger": final_trigger.value,
                      "deployment_scenario_id": project.scenario_id, "deployment_scenario_version": project.version}
    from mlsysbook_labs.learning_paths import get_learning_path as _get_learning_path
    report = build_lab_report(get_lab_metadata("vol1/lab_01_ml_intro.py"), track=track_id, scenario=project.title,
        predictions={p: s["prediction"] for p, s in snapshots.items()},
        knob_settings={p: s["inputs"] for p, s in snapshots.items()}, evidence_summary=snapshots,
        final_decision=final_decision, decisions=final_decision,
        learning_objectives=_get_learning_path(1, 1)["learning_objectives"],
        big_takeaways=["Population signals require outcome evidence.", "Untouched execution costs remain.", "Quality and resource requirements must pass together.", "Task-factor changes do not predict accuracy."],
        reflections={"remaining_risk": rationale.value}, residual_risk=rationale.value,
        source_trace={
            "engine": "mlsysim.engine.constraint_explorer",
            "deployment_journey": {"scenario_id": project.scenario_id, "version": project.version},
            "deployment_profile": f"constraint_explorer.TRACKS[{track_id}] · illustrative envelope",
            "host_study": "Scenarios.ControlledTaskFactors · independent supplied task factors and quality observations",
            "scenario_version": INTRO_SCENARIO_VERSION,
            "captured_models": {part: snapshot["model_key"] for part, snapshot in snapshots.items()},
            "scenario_assumptions": {part: snapshot["result"]["provenance"] for part, snapshot in snapshots.items()},
        },
        result_snapshot={"scenario_version": INTRO_SCENARIO_VERSION, "captures": snapshots, **final_decision})
    notebook_shell([report_export_panel(report), save_discussion,
        mo.md("When ready, downloading is separate from saving locally. Save discussion keeps your completed work locally; save again after making changes.")])
    return final_decision, report




@app.cell
def _(captures, final_trigger, mo, rationale, ready, track_id):
    # Recreate the deliberate action when evidence or discussion inputs change.
    _reset = (track_id, captures, final_trigger.value, rationale.value)
    save_discussion = mo.ui.run_button(label="Save discussion", disabled=not ready)
    return (save_discussion,)


@app.cell
async def _(INTRO_SCENARIO_VERSION, captures, final_trigger, ledger, notebook_shell, notebook_status, project, rationale, ready, save_discussion, track_id):
    _status = "Ready to save" if ready else "Evidence in progress"
    if ready and save_discussion.value:
        try:
            await ledger.load_async()
            await ledger.asave(track=track_id, chapter=1, design={"schema_version": 1, "lab_id": "v1_01", "track_id": track_id,
                "model_id": "constraint_explorer", "scenario_version": INTRO_SCENARIO_VERSION,
                "deployment_scenario_id": project.scenario_id, "deployment_scenario_version": project.version,
                "recommendation": captures["C"].to_dict()["decision"], "rejected_alternative": captures["C"].to_dict()["baseline"]["candidate"],
                "reevaluation_trigger": final_trigger.value, "residual_risk": rationale.value,
                "rationale": rationale.value, "evidence": {p: captures[p].to_dict() for p in "ABCD"}})
            _status = "Saved"
        except Exception:
            _status = "Storage unavailable · keep the download and retry saving"
    notebook_shell([notebook_status(_status, "When your discussion is ready, downloading is separate from saving locally.")])
    return


if __name__ == "__main__":
    app.run()
