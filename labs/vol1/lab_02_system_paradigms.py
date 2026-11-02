import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 02: Where Should Inference Run? · MLSysBook")


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
    from mlsysim.labs.experiments.v1_02_experiments import (
        STUDY_VERSION, locality_study, resource_study, operation_study,
        placement_study, study_profile,
    )
    from mlsysim.core.units import Q_
    from mlsysim.engine.scenarios import Scenarios
    from mlsysim.fmt import fmt, fmt_bandwidth, fmt_time, fmt_memory, fmt_energy, fmt_percent
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, apply_plotly_theme
    from mlsysbook_labs.notebook import (notebook_shell, notebook_header, notebook_workspace,
        notebook_status, notebook_progress, notebook_cards, notebook_table, notebook_part, notebook_bridge)
    from mlsysbook_labs import build_lab_report, get_lab_metadata, report_export_panel
    from mlsysbook_labs.contexts import CONTEXT_OPTIONS, get_context, preferred_context
    from mlsysbook_labs.experiment_evidence import capture_evidence, audit_evidence
    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (COLORS, CONTEXT_OPTIONS, Q_, Scenarios, STUDY_VERSION, apply_plotly_theme,
            notebook_shell, notebook_header, notebook_workspace, notebook_status,
            notebook_progress, notebook_cards, notebook_table, notebook_part, notebook_bridge,
            audit_evidence, build_lab_report, capture_evidence, fmt, fmt_bandwidth, fmt_energy,
            fmt_memory, fmt_percent, fmt_time, get_context, get_lab_metadata, go,
            ledger, locality_study, mo, operation_study, placement_study,
            preferred_context, report_export_panel, resource_study, study_profile)


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    get_pinned, set_pinned = mo.state({})
    get_interpretations, set_interpretations = mo.state({})
    get_trials, set_trials = mo.state({})
    return (get_evidence, get_pinned, set_evidence, set_pinned,
            get_interpretations, set_interpretations, get_trials, set_trials)


@app.cell
def _(CONTEXT_OPTIONS, get_context, ledger, mo, preferred_context, set_evidence, set_pinned, set_interpretations, set_trials):
    def reset_context(_value):
        set_evidence({})
        set_pinned({})
        set_interpretations({})
        set_trials({})
    track = mo.ui.dropdown(CONTEXT_OPTIONS, value=get_context(preferred_context(ledger)).label,
                           label="Deployment context", on_change=reset_context)
    return (track,)


@app.cell
def _(Scenarios, get_context, study_profile, track):
    track_id = track.value
    context = get_context(track_id)
    profile = study_profile(track_id)
    story = Scenarios.DeploymentJourneys.get(track_id)
    episode = story.episode(2)
    return context, episode, profile, story, track_id


@app.cell
def _(Q_, fmt_bandwidth, fmt_memory, mo, profile, track_id):
    _reset = track_id
    def payload_label(value):
        return str(fmt_memory(Q_(value, "megabyte"), unit="MB", precision=None))
    def rate_label(value):
        return str(fmt_bandwidth(Q_(value, "megabit/second"), unit="megabit/second", precision=None))
    a_payload = mo.ui.dropdown({payload_label(value): value for value in profile["payload_options_mb"]},
        value=payload_label(profile["payload_options_mb"][-1]), label="Request payload")
    a_rate = mo.ui.dropdown({rate_label(value): value for value in profile["connection_options_mbps"]},
        value=rate_label(profile["connection_mbps"]), label="Symmetric connection rate")
    b_case = mo.ui.dropdown({row["display"]: key for key, row in profile["resource_cases"].items()},
        value="Larger working set", label="Workload case")
    _interventions = {row["display"]: key for key, row in profile["interventions"].items()
                     if (key != "bandwidth" if track_id == "tinyml" else key != "flash")}
    b_intervention = mo.ui.dropdown(_interventions, value="More arithmetic capability", label="Change one resource")
    c_duty = mo.ui.slider(steps=sorted(set(profile["duty_options"] + [profile["baseline_duty_cycle"], profile["minimum_observation_fraction"]])),
                          value=profile["baseline_duty_cycle"], label="Active observation fraction")
    c_battery = mo.ui.dropdown({f"{value:g}× mission energy allowance": value for value in profile["battery_multipliers"]},
        value="1× mission energy allowance", label="Battery reserve" if track_id in ("tinyml", "mobile") else "Node energy allowance")
    d_connection = mo.ui.checkbox(value=True, label="Remote connection available")
    d_privacy = mo.ui.checkbox(value=True, label="Raw input may leave the endpoint")
    d_choice = mo.ui.dropdown({"Hold: evidence or requirements unresolved": "hold", "Local": "local",
        "Raw remote": "raw", "Event filter + remote": "event_filter", "Feature summary + remote": "feature_summary"},
        value="Hold: evidence or requirements unresolved", label="Retain for further validation", full_width=True)
    d_rejected = mo.ui.dropdown({"Local": "local", "Raw remote": "raw", "Event filter + remote": "event_filter", "Feature summary + remote": "feature_summary"},
        value="Raw remote", label="Rejected alternative", full_width=True)
    investigation_notes = mo.ui.dictionary({
        part: mo.ui.text_area(label=f"{part} · Explain your saved contrast",
            placeholder=prompt, full_width=True, rows=3)
        for part, prompt in {
            "A": "Name what you held fixed and changed. Which term explains the result? What link or payload change could reverse your conclusion, and what would still be unmeasured?",
            "B": "Name the workload and resource changed in your saved pair. What was repaired, and what remained binding? Would the same purchase help the ordinary workload?",
            "C": "Use your saved pair to explain what activity or reserve changed. Which service, energy, or power condition became binding? What requirement would reverse your policy?",
            "D": "Name a candidate whose status changed and the gate responsible. Which alternative did you reject and why? What evidence or requirement change would reverse your decision?",
        }.items()})
    rationale = mo.ui.text_area(label="Mechanism and decision",
        placeholder="Explain the binding constraint, chosen or held placement, rejected alternative, and evidence from your saved runs.", full_width=True, rows=5)
    uncertainty = mo.ui.text_area(label="Uncertainty and revisit condition",
        placeholder="Name the target-specific measurement that must replace this generic study before your project proceeds. What remains unpriced, and what change would make you rerun the decision?", full_width=True, rows=4)
    return (a_payload, a_rate, b_case, b_intervention, c_battery, c_duty,
            d_choice, d_connection, d_privacy, d_rejected, investigation_notes, rationale, uncertainty)


@app.cell
def _(mo, track_id):
    _reset = track_id
    a_prediction = mo.ui.radio(["Payload reduction", "Connection upgrade", "Neither is sufficient", "Need more evidence"],
        label="Which change would you try first to meet the remote deadline?").form(bordered=False, submit_button_label="Lock A prediction")
    b_prediction = mo.ui.radio(["Arithmetic capability", "Working-memory capacity", "Storage capacity or movement bandwidth", "Need more evidence"],
        label="Which resource would you change first for this workload?").form(bordered=False, submit_button_label="Lock B prediction")
    c_prediction = mo.ui.radio(["Reduce observation", "Increase energy allowance", "Keep the current policy", "Need more evidence"],
        label="Which operating change would you try first for this mission?").form(bordered=False, submit_button_label="Lock C prediction")
    d_prediction = mo.ui.radio(["Local", "Raw remote", "Filtered remote", "Hold for more evidence"],
        label="Raw-data permission is withdrawn. Which placement would you investigate first?").form(bordered=False, submit_button_label="Lock D prediction")
    return a_prediction, b_prediction, c_prediction, d_prediction


@app.cell
def _(STUDY_VERSION, a_payload, a_rate, b_case, b_intervention, c_battery, c_duty,
      d_connection, d_privacy, get_evidence, get_pinned, locality_study,
      operation_study, placement_study, profile, resource_study, track_id):
    captures = get_evidence()
    pins = get_pinned()
    a_result = locality_study(track_id, payload_mb=a_payload.value, connection_mbps=a_rate.value)
    a_ideal = locality_study(track_id, payload_mb=a_payload.value, connection_mbps=a_rate.value, ideal_compute=True)
    b_result = resource_study(track_id, workload_case=b_case.value, intervention=b_intervention.value)
    b_base = resource_study(track_id, workload_case=b_case.value)
    c_result = operation_study(track_id, duty_cycle=c_duty.value, battery_multiplier=c_battery.value)
    c_off = operation_study(track_id, duty_cycle=0.0, battery_multiplier=c_battery.value)
    c_more = operation_study(track_id, duty_cycle=c_duty.value, battery_multiplier=profile["battery_multipliers"][-1])
    saved_inputs = {part: captures[part].to_dict()["result"]["inputs"] for part in "ABC" if part in captures}
    d_ready = len(saved_inputs) == len("ABC")
    d_arguments = dict(track_id=track_id, payload_mb=saved_inputs.get("A", {}).get("payload_mb", profile["payload_mb"]),
        connection_mbps=saved_inputs.get("A", {}).get("connection_mbps", profile["connection_mbps"]),
        workload_case=saved_inputs.get("B", {}).get("workload_case", "ordinary"),
        intervention=saved_inputs.get("B", {}).get("intervention", "baseline"),
        duty_cycle=saved_inputs.get("C", {}).get("duty_cycle", profile["baseline_duty_cycle"]),
        battery_multiplier=saved_inputs.get("C", {}).get("battery_multiplier", profile["battery_multipliers"][0]))
    d_base = placement_study(**d_arguments)
    d_result = placement_study(**d_arguments, connectivity_available=d_connection.value, raw_data_allowed=d_privacy.value)
    return (a_ideal, a_result, b_base, b_result, c_more, c_off, c_result, captures,
            d_base, d_ready, d_result, pins, saved_inputs)


@app.cell
def _(STUDY_VERSION, a_ideal, a_prediction, a_result, b_base, b_prediction,
      b_result, c_prediction, c_result, capture_evidence, captures, d_base,
      d_choice, d_prediction, d_ready, d_rejected, d_result, mo, pins,
      saved_inputs, set_evidence, set_pinned, set_trials, track_id):
    def store(part, prediction, baseline, result, *, alternatives=(), decision=None, chosen=None, upstream=None):
        snapshot = capture_evidence(track=track_id, part=part, prediction=prediction,
            inputs=result["inputs"], baseline=baseline, result=result,
            alternatives=alternatives, decision=decision, chosen_result=chosen,
            upstream_inputs=upstream, model_key="v1_02_experiments")
        set_evidence(lambda current: {**current, part: snapshot})
        set_trials(lambda current: {**current, part: [*current.get(part, []), snapshot.to_dict()]})
    version_inputs = {"scenario_version": STUDY_VERSION}
    d_upstream = {**version_inputs, "saved_conditions": saved_inputs}
    a_pin = mo.ui.button(label="Pin this network run", disabled=a_prediction.value is None,
        on_click=lambda _v: set_pinned(lambda current: {**current, "A": a_result}))
    a_pair = pins.get("A")
    a_capture = mo.ui.button(label="Save locality contrast", kind="success",
        disabled=a_prediction.value is None or a_pair is None or a_pair["inputs"] == a_result["inputs"],
        on_click=lambda _v: store("A", a_prediction.value, a_pair, a_result, alternatives=(a_ideal,), upstream=version_inputs))
    b_pin = mo.ui.button(label="Pin this resource attempt", disabled=b_prediction.value is None,
        on_click=lambda _v: set_pinned(lambda current: {**current, "B": b_result}))
    b_pair = pins.get("B")
    b_capture = mo.ui.button(label="Save resource contrast", kind="success",
        disabled=b_prediction.value is None or b_pair is None or b_pair["inputs"] == b_result["inputs"]
                 or b_pair["inputs"]["workload_case"] != b_result["inputs"]["workload_case"],
        on_click=lambda _v: store("B", b_prediction.value, b_pair, b_result, alternatives=(b_base,), upstream=version_inputs))
    c_pin = mo.ui.button(label="Pin this operating policy", disabled=c_prediction.value is None,
        on_click=lambda _v: set_pinned(lambda current: {**current, "C": c_result}))
    c_pair = pins.get("C")
    c_capture = mo.ui.button(label="Save operation contrast", kind="success",
        disabled=c_prediction.value is None or c_pair is None or c_pair["inputs"] == c_result["inputs"],
        on_click=lambda _v: store("C", c_prediction.value, c_pair, c_result, upstream=version_inputs))
    d_selected = d_result["candidates"].get(d_choice.value)
    d_capture = mo.ui.button(label="Save stressed placement decision", kind="success",
        disabled=not d_ready or d_prediction.value is None or d_base["inputs"] == d_result["inputs"]
                 or d_choice.value == d_rejected.value or (d_selected is not None and not d_selected["feasible"]),
        on_click=lambda _v: store("D", d_prediction.value, d_base, d_result,
            alternatives=tuple(d_result["candidates"].values()), decision={"chosen": d_choice.value, "rejected": d_rejected.value},
            chosen=d_selected, upstream=d_upstream))
    return (a_capture, a_pair, a_pin, b_capture, b_pair, b_pin, c_capture,
            c_pair, c_pin, d_capture, d_selected, d_upstream, version_inputs)


@app.cell
def _(get_interpretations, get_trials):
    # State reads stay outside the callback-defining cell: Marimo suppresses
    # self-triggered reruns, so a setter and getter in one cell leave stale UI.
    interpretations = get_interpretations()
    trial_history = get_trials()
    return interpretations, trial_history


@app.cell
def _(captures, interpretations, investigation_notes, mo, set_interpretations):
    def attach(part):
        record = {"capture": captures[part].to_dict(), "response": investigation_notes.value[part]}
        set_interpretations(lambda current: {**current, part: record})
    attach_notes = {part: mo.ui.button(label=f"Attach {part} interpretation",
        disabled=part not in captures or not investigation_notes.value[part].strip()
            or (part in interpretations and interpretations[part]["capture"] == captures[part].to_dict()
                and interpretations[part]["response"] == investigation_notes.value[part]),
        on_click=lambda _v, part=part: attach(part)) for part in "ABCD"}
    notes_current = {part: part in captures and part in interpretations
        and interpretations[part]["capture"] == captures[part].to_dict()
        and interpretations[part]["response"] == investigation_notes.value[part]
        for part in "ABCD"}
    return attach_notes, notes_current


@app.cell
def _(context, episode, get_lab_metadata, mo, notebook_header, notebook_shell, notebook_bridge, profile, story, track):
    from mlsysbook_labs.notebook import notebook_scenario as _scenario
    from mlsysbook_labs.learning_paths import get_learning_path
    from mlsysbook_labs.notebook import notebook_learning_map
    _learning = get_learning_path(1, 2)
    _learning["next_lab"]["question"] = story.episode(3).question
    notebook_shell([
        notebook_header(get_lab_metadata("vol1/lab_02_system_paradigms.py"),
            question=episode.question, label="LAB 02",
            topics=("Locality", "Resources", "Operation", "Placement")),
        mo.hstack([track, mo.md(f"**{story.title}**")], wrap=True, justify="start"),
        _scenario(_learning, track=track.value, scene=episode.scene),
        notebook_learning_map(_learning),
        mo.accordion({"Reading and assumptions": mo.md(
            f"**Mission:** {story.mission}\n\n{story.study_boundary}\n\n"
            "**Read first:** [System Paradigms](https://mlsysbook.ai/vol1/system_paradigms/system_paradigms.html), especially the locality invariant, device energy budgets, and deployment decision framework.\n\n"
            "Investigate A–D: make a prediction, try a purposeful contrast, explain the result, then test a reversal. Save another contrast when it changes your thinking; the download keeps saved run records and your latest attached explanations. Attach a concise interpretation to each saved pair, then download your discussion for submission. These are illustrative controlled studies, not device measurements. Wrong predictions unlock the same experiments. Keeping a baseline or holding the decision is valid.")})])
    return



@app.cell
def _(audit_evidence, captures, d_upstream, track_id, version_inputs):
    audit = audit_evidence(captures, track=track_id, required_parts=tuple("ABCD"),
        per_part_upstream_inputs={**{part: version_inputs for part in "ABC"}, "D": d_upstream},
        contrast_required_parts=tuple("ABCD"))
    return (audit,)


@app.cell
def _(audit, captures, episode, story):
    # Summarize frozen evidence only. Live controls cannot revise the team record.
    compatible = {part: record.to_dict() for part, record in captures.items()
                  if part not in audit.stale and part not in audit.missing_contrasts
                  and not any(part in pair for pair in audit.identical_pairs)}
    _placement = compatible.get("D") if audit.complete else None
    labels = {"local":"Local", "raw":"Raw remote", "event_filter":"Event filter + remote",
              "feature_summary":"Feature summary + remote"}
    if _placement:
        chosen = _placement["decision"]["chosen"]
        rejected = _placement["decision"]["rejected"]
        rejected_row = _placement["result"]["candidates"][rejected]
        failures = rejected_row["violations"]
        decisive = ", ".join(value.replace("_", " ") for value in failures)
        decision_brief = {
            "state":"hold" if chosen == "hold" else "conditional",
            "decision":"Hold for further evidence" if chosen == "hold" else labels[chosen] + " retained for target validation",
            "decisive_obligation":("Rejected alternative fails: " + decisive) if decisive else "No supplied gate excludes the rejected alternative; justify the choice using the saved comparison.",
            "rejected_alternative":labels[rejected],
            "scope":"Saved controlled-study screens only; target readiness and lifecycle cost remain unresolved.",
        }
    else:
        decision_brief = {
            "state":"needs_update" if "D" in audit.stale else "in_progress",
            "decision":"Earlier placement needs updating" if "D" in audit.stale else "Placement not yet recorded",
            "decisive_obligation":"Re-save placement using the current saved A/B/C conditions." if "D" in audit.stale else "Save a meaningful comparison in A/B/C, then challenge a placement in D.",
            "rejected_alternative":"Not current" if "D" in audit.stale else "Not yet recorded",
            "scope":"No current saved placement claim.",
        }
    decision_brief.update({
        "project":story.title,
        "target_evidence_needed":"Measure task-matched target quality, complete-path timing, whole-device energy and sustained operation; complete the cost and data-handling review.",
        "next_chapter_question":story.episode(3).question,
    })
    return (decision_brief,)


@app.cell
def _(COLORS, Q_, a_capture, a_ideal, a_pair, a_payload, a_pin, a_prediction,
      a_rate, a_result, apply_plotly_theme, audit, b_base, b_capture,
      b_case, b_intervention, b_pair, b_pin, b_prediction, b_result, c_battery,
      c_capture, c_duty, c_more, c_off, c_pair, c_pin, c_prediction, c_result,
      captures, d_base, d_capture, d_choice, d_connection, d_prediction,
      d_privacy, d_ready, d_rejected, d_result, d_selected, d_upstream, fmt,
      fmt_energy, fmt_memory, fmt_percent, fmt_time, go, mo, rationale,
      track_id, uncertainty, version_inputs, operation_study, profile, notebook_shell,
      notebook_workspace, notebook_status, notebook_progress, notebook_cards, notebook_table, notebook_part,
      attach_notes, investigation_notes, notes_current, trial_history, decision_brief, notebook_bridge):
    def table(rows):
        return notebook_table([{key: str(value) for key, value in row.items()} for row in rows])
    def time(value):
        return "Unavailable" if value is None else str(fmt_time(Q_(value, "millisecond"), unit="millisecond", precision=None))
    def memory(value):
        return str(fmt_memory(Q_(value, "megabyte"), unit="MB", precision=None))
    def energy(value):
        return str(fmt_energy(Q_(value, "watt_hour"), unit="Wh", precision=None))
    def power(value):
        return str(fmt(Q_(value, "watt"), unit="W", precision=None))
    def percent(value):
        return str(fmt_percent(value, style="symbol"))
    def status(result):
        return "PASS supplied screens" if result["feasible"] else "FAIL: " + ", ".join(item.replace("_", " ") for item in result["violations"])
    def saved(part):
        capture = captures.get(part)
        if capture is None:
            return notebook_status("Not saved", "Pin a run, change one control, then save the contrast.")
        if part in audit.stale:
            return notebook_status("Saved · needs update", "Earlier saved conditions changed. Save this decision again.")
        return notebook_status("Saved", "Live controls do not rewrite this record. Original prediction: " + str(capture.to_dict()["prediction"]))
    def interpretation(part):
        record = captures.get(part)
        conditions = mo.md("Save a contrast first; then explain that exact pair.")
        if record is not None:
            snapshot = record.to_dict()
            before, after = snapshot["baseline"]["inputs"], snapshot["result"]["inputs"]
            labels = {"payload_mb":"Request payload", "connection_mbps":"Connection rate",
                "ideal_compute":"Ideal remote processor", "workload_case":"Workload",
                "intervention":"Resource intervention", "duty_cycle":"Active observation",
                "battery_multiplier":"Energy allowance", "connectivity_available":"Remote connection",
                "raw_data_allowed":"Raw-data permission"}
            def display(key, value):
                if key == "payload_mb": return memory(value)
                if key == "connection_mbps": return str(fmt(Q_(value,"megabit/second"),unit="megabit/second",precision=None))
                if key == "duty_cycle": return percent(value)
                if key == "battery_multiplier": return f"{value:g}× supplied allowance"
                if key == "workload_case": return profile["resource_cases"][value]["display"]
                if key == "intervention": return profile["interventions"][value]["display"]
                if isinstance(value,bool): return "Yes" if value else "No"
                return str(value)
            changed = [key for key in labels if key in after and before.get(key) != after[key]]
            held = [key for key in labels if key in after and before.get(key) == after[key]]
            conditions = mo.vstack([mo.md("**Your saved comparison**"), table([
                {"Changed":labels[key],"Pinned":display(key,before[key]),"Saved":display(key,after[key])} for key in changed]),
                mo.accordion({"Held fixed":table([{"Condition":labels[key],"Value":display(key,after[key])} for key in held])}),
                mo.callout(mo.md("More than one condition changed. This pair cannot isolate their individual effects; keep it as exploratory evidence or save a controlled pair."),kind="warn") if len(changed)>1 else mo.md("One supplied condition changed; compare the mechanism with your prediction.")])
        return mo.vstack([conditions, investigation_notes[part], attach_notes[part],
            notebook_status("Interpretation attached" if notes_current[part] else "Next · attach your interpretation",
                "Attach a concise explanation of this saved pair. Editing or recapturing requires attaching it again.")])
    def route(steps, *, placement_review=False):
        return mo.vstack([
            mo.md("**Choose your review.** Your saved A/B/C conditions form the starting case. Change one requirement, then explicitly choose a placement or hold and a rejected alternative." if placement_review else "**Choose your first trial.** Pin a starting run, change one condition, and save a comparison that tests your prediction."),
            mo.accordion({"Help me design a comparison": mo.accordion({
                label: mo.md(step) for label, step in zip(("Look at", "Compare", "Explain", "Challenge"), steps)
            })})])
    def locality():
        intro = notebook_part("A · Can the remote path meet the mission deadline?", "Your team wants remote inference. Find which change is worth testing before requesting a server upgrade.")
        if a_prediction.value is None:
            return mo.vstack([intro, mo.md(f"**Starting conditions:** {memory(a_result['inputs']['payload_mb'])} request; deadline {time(a_result['deadline_ms'])}. Your prediction is tentative: the instruments unlock whichever option you choose."), a_prediction])
        rows = [("Pinned", a_pair)] if a_pair else []
        rows += [("Live", a_result), ("Ideal remote arithmetic", a_ideal)]
        figure = go.Figure()
        for field, label, color in (("request_upload_ms", "Upload", COLORS["BlueLine"]),
            ("propagation_rtt_ms", "Propagation", COLORS["OrangeLine"]),
            ("remote_execution_ms", "Arithmetic", COLORS["GreenLine"]),
            ("response_download_ms", "Response", COLORS["OrangeLine"])):
            figure.add_bar(name=label, x=[label for label, _ in rows], y=[row[field] for _, row in rows], marker_color=color)
        figure.add_hline(y=a_result["deadline_ms"], line_dash="dash", annotation_text="Deadline")
        figure.update_layout(barmode="stack", yaxis_title="Optimistic remote path (ms)", height=300, legend_orientation="h", margin=dict(l=20,r=20,t=30,b=20))
        return mo.vstack([intro, notebook_workspace([route(["Which segment occupies the remote path, and which segments remain in the ideal-processor bar?", "Pin one payload, hold the link fixed, and change only payload. Or hold payload fixed and change only connection rate.", "Use the segment that changed to explain the bound. Which untouched term limits your remedy, and why does a lower-bound pass leave real readiness unresolved?", "Can another link or payload setting reverse your conclusion? Identify the remaining measurement needed before approving the path."]), a_payload, a_rate, mo.hstack([a_pin,a_capture], wrap=True, justify="start")], [apply_plotly_theme(figure)]),
            mo.accordion({"Run details":table([{"Run": label, "Bound": time(row["lower_bound_ms"]), "Conclusion": row["readiness"]} for label,row in rows])}),
            mo.callout(mo.md(f"**Prediction:** {a_prediction.value}. **Observed:** {a_result['readiness']}; the ideal-processor bound is {time(a_ideal['lower_bound_ms'])}."),kind="info"),
            mo.md("A lower-bound pass leaves complete-path readiness unresolved."),
            saved("A"), interpretation("A"), mo.accordion({"Calculation Notes": mo.md("Bound = payload/link + fiber round trip + remote execution + response/link. Symmetric link; no protocol, queueing, sensing or tail-latency evidence. An ideal processor removes only the execution term.")})])
    def resources():
        intro = notebook_part("B · Which resource should the team change?", "A teammate proposes a faster processor. Investigate that purchase against a different repair for the same workload.")
        if b_prediction.value is None:
            return mo.vstack([intro, mo.md("**Starting workload:** " + profile["resource_cases"][b_result["inputs"]["workload_case"]]["display"] + ". Predict an intervention before inspecting the resource pools. You can revise your view after experimenting."),b_prediction])
        rows = [("Original",b_base)] + ([("Pinned",b_pair)] if b_pair else []) + [("Live",b_result)]
        figure = go.Figure()
        for key,label,color in (("compute_ms","Arithmetic",COLORS["GreenLine"]),("movement_ms","Memory movement",COLORS["BlueLine"]),("overhead_ms","Overhead",COLORS["OrangeLine"])):
            figure.add_bar(name=label,x=[label for label,_ in rows],y=[row[key] if all(pool["required_mb"] <= pool["capacity_mb"] for pool in row["pools"]) else None for _,row in rows],marker_color=color)
        figure.update_layout(barmode="stack",height=280,yaxis_title="Hypothetical component time (ms)",legend_orientation="h",margin=dict(l=20,r=20,t=20,b=20))
        pools = go.Figure()
        for field, label, color in (("required_mb", "Required", COLORS["BlueLine"]), ("capacity_mb", "Available", COLORS["GreenLine"])):
            pools.add_bar(name=label, x=[row["pool"] for row in b_result["pools"]],
                y=[row[field] for row in b_result["pools"]], marker_color=color)
        pools.update_layout(barmode="group", height=250, yaxis_title="Live memory pools (MB)",
            legend_orientation="h", margin=dict(l=20,r=20,t=20,b=20))
        return mo.vstack([intro, notebook_workspace([route(["Which required memory pool exceeds available capacity? If every pool fits, which timing component dominates?", "Keep one workload fixed. Pin an arithmetic intervention, then compare it with a capacity intervention or another resource available in this context.", "Explain which resource the intervention changes and which requirement it leaves untouched. Why can a shorter arithmetic estimate fail to produce an executable path?", "Switch to the ordinary workload: does the same purchase still address its binding resource? Return to one workload before saving a contrast."]), b_case,b_intervention,
            mo.hstack([b_pin,b_capture],wrap=True,justify="start")],
            [apply_plotly_theme(pools), apply_plotly_theme(figure)]),
            mo.accordion({"Run details":table([{"Run":label,"Pool":pool["pool"],"Required":memory(pool["required_mb"]),"Available":memory(pool["capacity_mb"]),"Outcome":status(row)} for label,row in rows for pool in row["pools"]])}),mo.callout(mo.md(f"**Prediction:** {b_prediction.value}. **Observed:** {status(b_result)}; binding resource: {b_result['binding_resource']}.") ,kind="info"),mo.md("Timing bars are unavailable until every memory pool fits. A short analytical component estimate cannot make an overflowed path executable."),
            saved("B"),interpretation("B"),mo.accordion({"Calculation Notes":mo.md("The fixed-format task has separately supplied working-state demand, traffic and arithmetic. A controlled hypothetical changes one resource only. TinyML has distinct SRAM and flash pools; no bandwidth upgrade is supplied for it. No accuracy gain, price or real speedup is inferred from these interventions.")})])
    def operation():
        intro = notebook_part("C · Which operating policy can the mission tolerate?", "The operations team asks for longer endurance without losing the required observation service.")
        if c_prediction.value is None:
            return mo.vstack([intro, mo.md(f"**Mission requirements:** observe at least {percent(c_result['minimum_observation_fraction'])}; energy allowance {energy(c_result['mission_energy_budget_wh'])}; sustained-power ceiling {power(c_result['average_power_budget_w'])}."),c_prediction])
        rows = ([("Pinned",c_pair)] if c_pair else []) + [("Live",c_result),("Larger allowance, same activity",c_more),("Always off",c_off)]
        curve = [operation_study(track_id, duty_cycle=value, battery_multiplier=c_battery.value)
            for value in sorted(set(profile["duty_options"] + [profile["minimum_observation_fraction"], profile["baseline_duty_cycle"]]))]
        # Every point and threshold is a returned study result; the notebook draws only.
        from plotly.subplots import make_subplots
        figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.16,
            subplot_titles=("Mission energy", "Sustained power"))
        for axis, field, budget, color, unit in (
            (1,"energy_wh","mission_energy_budget_wh",COLORS["BlueLine"],"Energy (Wh)"),
            (2,"average_power_w","average_power_budget_w",COLORS["OrangeLine"],"Power (W)")):
            figure.add_scatter(x=[row["duty_cycle"] for row in curve], y=[row[field] for row in curve],
                mode="lines", name="Activity sweep", line_color=color, showlegend=axis==1,row=axis,col=1)
            figure.add_hline(y=c_result[budget], line_dash="dash", annotation_text="Budget", row=axis,col=1)
            figure.add_vline(x=c_result["minimum_observation_fraction"], line_dash="dot", row=axis,col=1)
            for label, record, symbol in (("Live",c_result,"circle"),("Pinned",c_pair,"diamond")):
                if record is not None:
                    figure.add_scatter(x=[record["duty_cycle"]],y=[record[field]],mode="markers",name=label,
                        marker=dict(size=11,symbol=symbol,color=COLORS["BlueLine"] if label == "Live" else COLORS["Grey"]),showlegend=axis==1,row=axis,col=1)
            figure.update_yaxes(title_text=unit,row=axis,col=1)
        figure.update_xaxes(tickformat=".0%", title_text="Active observation",row=2,col=1)
        figure.update_layout(height=420,legend_orientation="h",margin=dict(l=35,r=20,t=45,b=35))
        cards = notebook_cards([
            ("Observation", percent(c_result["duty_cycle"]), "pass" if c_result["duty_cycle"] >= c_result["minimum_observation_fraction"] else "fail"),
            ("Energy", energy(c_result["energy_wh"]), "pass" if c_result["energy_wh"] <= c_result["mission_energy_budget_wh"] else "fail"),
            ("Sustained power (W)", power(c_result["average_power_w"]), "pass" if c_result["average_power_w"] <= c_result["average_power_budget_w"] else "fail")])
        return mo.vstack([intro, notebook_workspace([route(["Which of the observation, energy and sustained-power screens currently fail? Where is the live point relative to each boundary?", "Pin the current policy, then change only activity. Alternatively hold activity fixed and change only the energy allowance.", "Explain why the changed control affects some screens and leaves others unchanged. Distinguish supplying energy from meeting observation and power requirements; identify unmodeled whole-device costs.", "Can you satisfy all supplied screens together? If not, identify a requirement change that could reverse the hold."]), c_duty,c_battery,
            mo.hstack([c_pin,c_capture],wrap=True,justify="start")],[cards,apply_plotly_theme(figure)]),
            mo.callout(mo.md(f"**Prediction:** {c_prediction.value}. **Observed:** {status(c_result)}."),kind="info"),
            mo.md("Dotted lines mark minimum observation; dashed lines use the **live** budgets. A pinned marker keeps its original allowance, shown in Compare policies."),
            saved("C"),interpretation("C"),mo.accordion({"Compare policies": table([{"Run":label,"Observation":percent(row["duty_cycle"]),"Energy":energy(row["energy_wh"]),"Allowance":energy(row["mission_energy_budget_wh"]),"Power":power(row["average_power_w"]),"Outcome":status(row)} for label,row in rows]),
                "Calculation Notes":mo.md(c_result["assumption"] + " Average power = active power × active fraction + idle power × inactive fraction; mission energy = average power × horizon. The mission horizon and minimum observation are supplied requirements.")})])
    def placement():
        intro = notebook_part("D · What should the team retain for validation?", "The deployment review changes a requirement. Use your saved A/B/C conditions to challenge the original placement.")
        if not d_ready:
            return mo.vstack([intro,mo.callout(mo.md("Save A, B and C first. This part uses those records, not their moving live controls."),kind="warn")])
        if d_prediction.value is None:
            return mo.vstack([intro,mo.md("**Starting review:** saved A/B/C conditions, remote connection available, raw-data permission granted. Predict what you would investigate before changing one requirement."),d_prediction])
        rows=[]
        for condition,result in (("Original conditions",d_base),("Stressed conditions",d_result)):
            for key,row in result["candidates"].items():
                rows.append({"Conditions":condition,"Placement":key.replace("_"," "),"Path":time(row["latency_ms"]) if row["gates"]["resource"] else "Unavailable until memory fits",
                    "Failed gates":", ".join(item.replace("_", " ") for item in row["violations"]) or "None in supplied study",
                    "Cost":row["cost_status"]})
        screen_labels = {"task":"Task retention", "privacy":"Data permission", "connectivity":"Connection",
            "deadline":"Deadline", "resource":"Memory", "service":"Observation service",
            "sustained_power":"Sustained power", "mission_energy":"Mission energy", "upload":"Upload allowance"}
        candidate_labels = {"local":"Local", "raw":"Raw remote", "event_filter":"Event filter", "feature_summary":"Features"}
        gates=[{"Screen":label, **{candidate_labels[key]:"PASS" if row["gates"][gate] else "FAIL"
            for key,row in d_result["candidates"].items()}} for gate,label in screen_labels.items()]
        return mo.vstack([intro, notebook_workspace([route(["Which candidate gates differ, and which obligations remain unresolved even for a candidate that passes?", "Change only raw-data permission and compare with original conditions. Or restore permission and remove only connectivity.", "Explain why the changed requirement excludes particular paths. Separate failed supplied gates from unknown target readiness and incomplete cost; justify a retained candidate or hold.", "What requirement or new measurement would reverse your choice? Check whether the rejected alternative actually fails a gate or remains a viable competitor."], placement_review=True), d_connection,d_privacy,d_choice,d_rejected,d_capture], [table(gates)]),
            mo.accordion({"Compare original and stressed conditions":table(rows),"Inspect partial cost components":table([
                {"Placement":key,"Component":component,"USD":str(fmt(Q_(value,"USD"),unit="USD",precision=None))}
                for key,row in d_result["candidates"].items() if row["feasible"] for component,value in row["cost_components_usd"].items()])}),
            mo.accordion({"Cost boundary":mo.md(d_result["cost_boundary"])}),
            mo.callout(mo.md(f"**Prediction:** {d_prediction.value}. **Observed:** {'At least one candidate passes supplied screens' if d_result['feasible'] else 'No candidate passes supplied screens'}; lifecycle cost and target readiness remain unresolved."),kind="info"),
            mo.callout(mo.md("Hold is always available. Retaining a candidate requires passing every supplied screen; it still needs target measurements and a complete cost comparison. Whole-device and remote energy remain unknown: sensing, radio and conversion losses are unmodeled. Filtered paths must keep feature definitions, preprocessing and model versions synchronized. Derived records still need a privacy review."),kind="info"),
            saved("D"),interpretation("D"),mo.accordion({"Calculation Notes":mo.md(d_result["timing_assumption"] + " " + d_result["service_assumption"] + " Task retention is supplied matched-task evidence, never inferred from fewer bytes. The simpler baseline is assumed already rejected for this supplied task; no model-selection experiment is being performed here.")})])
    def team_decision():
        return mo.vstack([mo.md("### Team decision so far"),
            notebook_status(decision_brief["decision"], decision_brief["scope"]),
            mo.md("**Rejected alternative screen:** " + decision_brief["decisive_obligation"] + "\n\n"
                "**Rejected alternative:** " + decision_brief["rejected_alternative"]),
            notebook_bridge("Your saved comparisons and placement or hold decision." if decision_brief["state"] in ("hold", "conditional") else "A current placement decision still needs recording.",
                "Target readiness, whole-device costs and data-handling obligations.",
                decision_brief["next_chapter_question"], labels=("Carry forward", "Still open", "Next")),
            mo.accordion({"Target measurements still needed": mo.md(decision_brief["target_evidence_needed"])})])
    def synthesis():
        return mo.vstack([team_decision(),notebook_part("Discussion · Defend a placement or hold", "Recommend a placement or hold. Explain the binding limit, rejected alternative, and condition that would reverse your decision."),
            table([{"Part":part,"Evidence":"Missing" if part not in captures else "Stale" if part in audit.stale else "Saved", "Interpretation":"Attached" if notes_current[part] else "Pending", "Saved contrasts":str(len(trial_history.get(part, [])))} for part in "ABCD"]),
            rationale,uncertainty,mo.callout(mo.md("Your download includes original and stressed conditions separately, every saved contrast, your latest attached part interpretations, and this discussion. Name the target-specific evidence needed before transferring the generic results to your project. Submit it as your instructor requests. LLM assistance does not replace reasoning tied to your actual runs."),kind="info")])
    tabs = mo.ui.tabs({"Part A · Locality":locality(),"Part B · Resources":resources(),
        "Part C · Operation":operation(),"Part D · Placement":placement(),"Synthesis":synthesis()})
    complete = audit.complete and all(notes_current.values()) and bool(rationale.value.strip()) and bool(uncertainty.value.strip())
    from mlsysbook_labs.notebook import notebook_tab_hint as _tab_hint
    notebook_shell([_tab_hint(), notebook_progress({part: "Saved · needs update" if part in audit.stale else "Saved" if part in captures else "Not saved" for part in "ABCD"}), tabs])
    return (complete,)


@app.cell(hide_code=True)
def _():
    from mlsysbook_labs.learning_paths import get_learning_path as _submission_path
    from mlsysbook_labs.notebook import notebook_shell as _submission_shell, notebook_submission as _submission
    _submission_shell([_submission(_submission_path(1, 2))])
    return


@app.cell
def _(build_lab_report, captures, complete, get_lab_metadata, mo, profile,
      rationale, report_export_panel, track_id, uncertainty, notebook_shell, save_discussion, story, episode, interpretations, trial_history, decision_brief):
    from mlsysbook_labs.notebook import notebook_download_readiness as _download_readiness
    mo.stop(not complete, _download_readiness(volume=1))
    snapshots = {part: captures[part].to_dict() for part in "ABCD"}
    decision = snapshots["D"]["decision"]
    from mlsysbook_labs.learning_paths import get_learning_path as _get_learning_path
    report = build_lab_report(get_lab_metadata("vol1/lab_02_system_paradigms.py"),
        track=track_id, scenario=f"{story.title}: {episode.task}",
        learning_objectives=_get_learning_path(1, 2)["learning_objectives"],
        predictions={part:row["prediction"] for part,row in snapshots.items()},
        knob_settings={part:row["inputs"] for part,row in snapshots.items()},
        evidence_summary=snapshots,
        binding_constraints={part:row["result"]["violations"] for part,row in snapshots.items()},
        decisions=decision, final_decision={**decision,"rationale":rationale.value},
        big_takeaways=["A lower-bound pass is not validated readiness.","Capacity, bandwidth and arithmetic are different resources.","A larger battery does not raise a sustained-power ceiling.","Excluded candidates cannot win on cost; incomplete costs cannot establish a lifecycle winner."],
        reflections={**{f"Part {part} interpretation":row["response"] for part,row in interpretations.items()},
            "mechanism_and_decision":rationale.value,"uncertainty_and_revisit":uncertainty.value,"next_lab_handoff":episode.handoff,"team_decision_so_far":decision_brief},
        residual_risk=uncertainty.value,
        result_snapshot={"track":track_id,"captures":snapshots,"decision":decision,
            "interpretations":interpretations,"investigation_history":trial_history,"team_decision_brief":decision_brief,
            "journey":{"scenario_id":story.scenario_id,"version":story.version,"chapter":2,"handoff":episode.handoff}},
        source_trace={"scenario_version":profile["version"],"model":"v1_02_experiments","fixture":"Scenarios.PlacementStudy; illustrative, not target measurements"})
    notebook_shell([report_export_panel(report), save_discussion,
        mo.md("Download when ready. **Save discussion** stores a local copy of the completed discussion. Reopening starts a fresh investigation; later labs must check compatibility before using stored evidence.")])
    return (report,)




@app.cell
def _(captures, complete, interpretations, investigation_notes, mo, rationale, track_id, uncertainty):
    # Recreate the deliberate action whenever any discussion input changes.
    _reset = (track_id, rationale.value, uncertainty.value, captures, interpretations, investigation_notes.value)
    save_discussion = mo.ui.run_button(label="Save discussion", disabled=not complete)
    return (save_discussion,)


@app.cell
async def _(captures, complete, interpretations, ledger, notebook_shell, notebook_status, rationale, save_discussion, story, episode, track_id, trial_history, uncertainty, decision_brief):
    _saved = False
    _error = None
    if complete and save_discussion.value:
        try:
            await ledger.load_async()
            await ledger.asave(track=track_id, chapter=2, design={
                "schema_version":1,"lab_id":"v1_02","model_id":"v1_02_experiments","track_id":track_id,
                "evidence":{part:row.to_dict() for part,row in captures.items()},
                "recommendation":captures["D"].to_dict()["decision"]["chosen"],
                "rejected_alternative":captures["D"].to_dict()["decision"]["rejected"],
                "reevaluation_trigger":uncertainty.value,"residual_risk":uncertainty.value,
                "rationale":rationale.value,"uncertainty_and_revisit":uncertainty.value,
                "interpretations":interpretations,"investigation_history":trial_history,"team_decision_brief":decision_brief,
                "journey":{"scenario_id":story.scenario_id,"version":story.version,"chapter":2,"handoff":episode.handoff}})
            _saved = True
        except Exception as exc:
            _error = str(exc)
    _status = "Saved" if _saved else "Storage unavailable" if _error else "Ready to save" if complete else "Evidence in progress"
    notebook_shell([notebook_status(_status, "Download remains available." if complete else "Save and interpret a contrast in each part, then write your discussion.")])
    return


if __name__ == "__main__":
    app.run()
