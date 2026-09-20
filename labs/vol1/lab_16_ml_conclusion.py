import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Lab 16: The Architect's Audit · MLSysBook")


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
    from mlsysim.engine.v1_16_experiments import (
        DeploymentEnvelope, audit_ledger, compare_prepared_intervention,
        evaluate_envelope_fraction, evaluate_joint_design,
        prepared_instructor_scenario, transfer_design,
    )
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import ACADEMIC_LAB_CSS, build_lab_report, get_lab_metadata, report_export_panel
    from mlsysbook_labs.experiment_evidence import capture_evidence, audit_evidence
    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        _loaded = await ledger.load_async()
    return (ACADEMIC_LAB_CSS, COLORS, DeploymentEnvelope, LAB_CSS,
            apply_plotly_theme, asdict, audit_evidence, audit_ledger,
            build_lab_report, capture_evidence, compare_prepared_intervention,
            evaluate_envelope_fraction, evaluate_joint_design, get_lab_metadata,
            go, ledger, mo, prepared_instructor_scenario, report_export_panel,
            transfer_design)


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
def _(track):
    track_id = track.value
    track_labels = {"tinyml": "TinyML", "mobile": "Mobile", "edge": "Edge", "cloud": "Cloud"}
    return track_id, track_labels


@app.cell
def _(audit_ledger, evaluate_joint_design, ledger, prepared_instructor_scenario, track_id):
    student_history = dict(ledger._state.history)
    ledger_audit = audit_ledger(student_history, track_id)
    instructor = prepared_instructor_scenario(track_id)
    baseline_result = evaluate_joint_design(instructor["snapshot"], instructor["envelope"])
    return baseline_result, instructor, ledger_audit, student_history


@app.cell
def _(mo, track_id, track_labels):
    part_a_prediction = mo.ui.radio(
        {"Complete and replayed": "complete", "Some evidence is missing": "missing", "Some evidence is incompatible or unreplayed": "incompatible"},
        label="What will the actual ledger audit show?",
    ).form(submit_button_label="Lock Part A prediction")
    intervention = mo.ui.dropdown(
        {"Quantized candidate": "quantize", "Fused-runtime candidate": "fuse_runtime", "Larger-model candidate": "larger_model"},
        value="Quantized candidate", label="Prepared alternative",
    )
    part_b_prediction = mo.ui.radio(
        {"Memory": "memory", "Latency": "latency", "Energy": "energy", "Quality": "quality"},
        label="Which margin changes most?",
    ).form(submit_button_label="Lock Part B prediction")
    part_b_decision = mo.ui.radio(
        {"Choose prepared candidate": "candidate", "Keep baseline": "baseline", "Hold release": "hold"},
        label="Decision after seeing both outcomes",
    )
    boundary_axis = mo.ui.dropdown(
        {"Memory limit": "memory", "Latency limit": "latency", "Energy limit": "energy"},
        value="Latency limit", label="Limit to tighten",
    )
    boundary_fraction = mo.ui.slider(0.55, 1.05, value=0.80, step=0.05, label="Fraction of current limit")
    part_c_prediction = mo.ui.radio(
        {"Still passes": "pass", "Crosses the selected limit": "fail"},
        label="Does the saved design remain inside?",
    ).form(submit_button_label="Lock Part C prediction")
    target_options = {label: key for key, label in track_labels.items() if key != track_id}
    target_track = mo.ui.dropdown(target_options, value=next(iter(target_options)), label="Second deployment envelope")
    part_d_prediction = mo.ui.radio(
        {"Everything transfers": "all", "Physics transfers; quality is unavailable": "physical", "A physical requirement fails": "failure"},
        label="What survives transfer?",
    ).form(submit_button_label="Lock Part D prediction")
    return boundary_axis, boundary_fraction, intervention, part_a_prediction, part_b_decision, part_b_prediction, part_c_prediction, part_d_prediction, target_track


@app.cell
def _(mo, track_id):
    _track_key = track_id
    final_choice = mo.ui.radio(
        {"Keep baseline": "baseline", "Quantized candidate": "quantize", "Fused-runtime candidate": "fuse_runtime", "Larger-model candidate": "larger_model", "Hold release": "hold", "No feasible choice": "none"}, label="Recommendation")
    final_rejected = mo.ui.radio(
        {"Baseline": "baseline", "Quantized candidate": "quantize", "Fused-runtime candidate": "fuse_runtime", "Larger-model candidate": "larger_model"}, label="Tested rejected alternative")
    final_trigger = mo.ui.radio(
        {"Memory reaches limit": "memory", "Latency reaches limit": "latency", "Energy reaches limit": "energy", "Matched quality evidence fails": "quality"}, label="Reevaluation trigger")
    residual_risk = mo.ui.text(label="Remaining limitation", placeholder="Name evidence this lab still does not establish.")
    rationale = mo.ui.text_area(label="Release rationale", placeholder="Connect evidence, choice, rejected alternative, consequence, and trigger.")
    return final_choice, final_rejected, final_trigger, rationale, residual_risk


@app.cell
def _(asdict, audit_ledger, boundary_axis, boundary_fraction,
      compare_prepared_intervention, evaluate_envelope_fraction, instructor,
      intervention, prepared_instructor_scenario, student_history,
      target_track, track_id, transfer_design):
    intervention_value = intervention.value if intervention.value in {"quantize", "fuse_runtime", "larger_model"} else "quantize"
    boundary_axis_value = boundary_axis.value if boundary_axis.value in {"memory", "latency", "energy"} else "latency"
    boundary_fraction_value = boundary_fraction.value if isinstance(boundary_fraction.value, (int, float)) else 0.80
    target_track_value = target_track.value if target_track.value in {"tinyml", "mobile", "edge", "cloud"} and target_track.value != track_id else next(key for key in ("tinyml", "mobile", "edge", "cloud") if key != track_id)
    intervention_comparison = compare_prepared_intervention(track_id, intervention_value)
    envelope = instructor["envelope"]
    boundary_comparison = evaluate_envelope_fraction(
        instructor["snapshot"], envelope,
        requirement=boundary_axis_value, fraction=boundary_fraction_value)
    target_scenario = prepared_instructor_scenario(target_track_value)
    transfer_result = transfer_design(instructor["snapshot"], target_scenario["envelope"])
    def audit_arm(value, inputs):
        return {"inputs": inputs, "selected_track_id": value.selected_track_id,
                "counts": value.counts, "release_evidence_complete": value.release_evidence_complete,
                "records": [asdict(record) for record in value.records]}
    audit_inputs = {"history": student_history, "track_id": track_id, "required_chapters": list(range(1, 16))}
    empty_inputs = {"history": student_history, "track_id": track_id, "required_chapters": list(range(1, 9))}
    audit_baseline = audit_arm(audit_ledger(student_history, track_id, required_chapters=tuple(range(1, 9))), empty_inputs)
    audit_result = audit_arm(audit_ledger(student_history, track_id), audit_inputs)
    def physical_arm(result):
        return {**result, "inputs": {"snapshot": result["snapshot"], "envelope": result["envelope"]}}
    return audit_baseline, audit_inputs, audit_result, boundary_axis_value, boundary_comparison, boundary_fraction_value, intervention_comparison, intervention_value, physical_arm, target_scenario, target_track_value, transfer_result


@app.cell
def _(asdict, audit_baseline, audit_inputs, audit_result, baseline_result,
      boundary_axis_value, boundary_comparison, boundary_fraction_value, capture_evidence,
      intervention_comparison, intervention_value, mo, part_a_prediction,
      part_b_decision, part_b_prediction, part_c_prediction, part_d_prediction, physical_arm,
      set_evidence, target_scenario, target_track_value, track_id, transfer_result):
    def store(part, captured):
        set_evidence(lambda current: {**current, part: captured})
    a_upstream = {"track_id": track_id, "history": audit_inputs["history"]}
    b_upstream = {"track_id": track_id, "intervention": intervention_value, "decision": part_b_decision.value}
    c_upstream = {"track_id": track_id, "axis": boundary_axis_value, "fraction": boundary_fraction_value}
    d_upstream = {"source_track": track_id, "target_track": target_track_value}
    a_capture = mo.ui.button(label="Capture ledger contrast", kind="success", disabled=part_a_prediction.value is None,
        on_click=lambda _v: store("A", capture_evidence(track=track_id, part="A", prediction=part_a_prediction.value,
            inputs=audit_inputs, baseline=audit_baseline, result=audit_result, upstream_inputs=a_upstream,
            model_key="v1_16.audit_ledger.v1")))
    _b_base = physical_arm(intervention_comparison["baseline"])
    _b_candidate = physical_arm(intervention_comparison["candidate"])
    _b_decision = intervention_value if part_b_decision.value == "candidate" else part_b_decision.value
    _b_chosen = _b_candidate if part_b_decision.value == "candidate" else (_b_base if part_b_decision.value in {"baseline", "hold"} else None)
    _b_role = "chosen candidate" if part_b_decision.value == "candidate" else "rejected alternative"
    b_capture = mo.ui.button(label="Capture alternative contrast", kind="success", disabled=part_b_prediction.value is None or part_b_decision.value is None,
        on_click=lambda _v: store("B", capture_evidence(track=track_id, part="B", prediction=part_b_prediction.value,
            inputs={"intervention": intervention_value}, baseline=_b_base,
            result=_b_candidate, alternatives=(intervention_comparison,),
            decision=_b_decision, chosen_result=_b_chosen, result_role=_b_role,
            upstream_inputs=b_upstream, model_key="v1_16.evaluate_joint_design.v1")))
    c_capture = mo.ui.button(label="Capture boundary contrast", kind="success", disabled=part_c_prediction.value is None,
        on_click=lambda _v: store("C", capture_evidence(track=track_id, part="C", prediction=part_c_prediction.value,
            inputs={"axis": boundary_axis_value, "fraction": boundary_fraction_value},
            baseline=physical_arm(boundary_comparison["baseline"]), result=physical_arm(boundary_comparison["changed"]),
            decision=part_c_prediction.value, upstream_inputs=c_upstream, model_key="v1_16.evaluate_joint_design.v1")))
    d_capture = mo.ui.button(label="Capture transfer contrast", kind="success", disabled=part_d_prediction.value is None,
        on_click=lambda _v: store("D", capture_evidence(track=track_id, part="D", prediction=part_d_prediction.value,
            inputs={"source_track": track_id, "target_track": target_track_value}, baseline=physical_arm(baseline_result),
            result=physical_arm(transfer_result), alternatives=({"target_envelope": asdict(target_scenario["envelope"])},),
            decision=part_d_prediction.value, upstream_inputs=d_upstream, model_key="v1_16.evaluate_joint_design.v1")))
    return a_capture, a_upstream, b_capture, b_upstream, c_capture, c_upstream, d_capture, d_upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, instructor, mo, track, track_id, track_labels):
    css = mo.Html("""<style>
    .cap-head{background:linear-gradient(135deg,#111827,#164e63);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:12px}
    .cap-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}
    .cap-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:14px 0 8px}.cap-head p{color:#cffafe;max-width:760px}
    .cap-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:16px}.cap-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}
    .cap-note{color:#475569;font-size:.9rem;line-height:1.5}.saved{border-left:4px solid #16a34a;background:#f0fdf4;padding:9px 12px;border-radius:7px}
    .lab-hud{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#111827!important;color:#fff;padding:14px 18px;border-radius:9px}.lab-hud .hud-label{color:#a7b9cf}.lab-hud .hud-active{color:#86efac}
    @media(max-width:520px){.cap-head{border-radius:9px;margin-top:30px}.cap-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""{css.text}<section class="cap-head"><div class="cap-top"><span>VOLUME I · LAB 16</span><span>ABOUT 45–55 MIN</span></div><h1>The Architect's Audit</h1><p>Does the accumulated design still fit when its evidence and constraints meet?</p><div class="cap-meta"><div><b>Track</b><br>{track_labels[track_id]}</div><div><b>Investigation</b><br>Cross-layer release audit</div><div><b>Output</b><br>Defensible release memo</div></div></section>""")
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, header, track,
               mo.Html(f'<p class="cap-note"><b>Prepared scenario:</b> {instructor["scenario_assumption"]} It remains separate from your actual ledger.</p>')], gap=0.5)
    return


@app.cell
def _(mo):
    mo.sidebar([mo.md("## Lab navigation"), mo.outline(label="Sections")])
    return


@app.cell
def _(a_capture, a_upstream, audit_evidence,
      b_capture, b_upstream, boundary_axis, boundary_comparison,
      boundary_fraction, c_capture, c_upstream, d_capture, d_upstream,
      final_choice, final_rejected, final_trigger, get_evidence,
      intervention, intervention_comparison, intervention_value, ledger_audit, mo,
      part_a_prediction, part_b_decision, part_b_prediction, part_c_prediction,
      part_d_prediction, rationale, residual_risk, target_track, target_track_value,
      track_id, track_labels, transfer_result):
    _captures = get_evidence()
    upstream = {"A": a_upstream, "B": b_upstream, "C": c_upstream, "D": d_upstream}
    evidence_audit = audit_evidence(_captures, track=track_id, required_parts=tuple("ABCD"),
        per_part_upstream_inputs=upstream, contrast_required_parts=tuple("ABCD"))
    def table(rows):
        return mo.ui.table(rows, pagination=False).style({"max-width": "100%", "overflow-x": "auto"})
    def saved(part):
        captured = _captures.get(part)
        if captured is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in evidence_audit.stale or (part, part) in evidence_audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Recapture this comparison."), kind="danger")
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {captured.to_dict()["prediction"]}<br><small>The report keeps this result when live controls change.</small></div>')
    def part_a():
        if part_a_prediction.value is None:
            return mo.vstack([mo.md("### A · Does the accumulated design fit together? (10 min)\nPredict before opening the actual ledger."), part_a_prediction])
        rows = [{"Chapter": record.chapter, "Status": record.status.value, "Reason": record.reason or "replayed evidence"} for record in ledger_audit.records]
        return mo.vstack([mo.md("### A · Does the accumulated design fit together? (10 min)"), part_a_prediction, table(rows),
            mo.callout(mo.md(f"Found **{ledger_audit.counts['demonstrated']} replayed records whose chosen design passes**, **{ledger_audit.counts['failed']} replayed records with observed requirement failures**, **{ledger_audit.counts['missing']} missing**, **{ledger_audit.counts['incomplete']} incomplete**, **{ledger_audit.counts['incompatible']} incompatible**, and **{ledger_audit.counts['schema_valid_unreplayed']} schema-valid but unreplayed** records. Missing evidence is not a failed requirement."), kind="info"),
            a_capture, saved("A"), mo.accordion({"Calculation Notes": mo.md("The audit validates the stable ledger schema and directly replays supported saved inputs. Unknown models remain unreplayed; prose never supplies a pass result.")})])
    def part_b():
        if part_b_prediction.value is None:
            return mo.vstack([mo.md("### B · Where does one alternative move the constraint? (12 min)\nChoose a fully declared scenario alternative, then predict its consequence."), intervention, part_b_prediction])
        base, candidate = intervention_comparison["baseline"], intervention_comparison["candidate"]
        rows = [{"Design": "Baseline", "Memory (MB)": base["memory_mb"], "Latency (ms)": base["latency_ms"], "Energy (mJ)": base["energy_mj"], "Quality (%)": base["quality_pct"], "Failed": ", ".join(base["failed_requirements"]) or "none"},
                {"Design": intervention_value, "Memory (MB)": candidate["memory_mb"], "Latency (ms)": candidate["latency_ms"], "Energy (mJ)": candidate["energy_mj"], "Quality (%)": candidate["quality_pct"], "Failed": ", ".join(candidate["failed_requirements"]) or "none"}]
        return mo.vstack([mo.md("### B · Where does one alternative move the constraint? (12 min)"), intervention, part_b_prediction, table(rows), part_b_decision,
            mo.callout(mo.md("Each alternative is a full declared fixture. Its matched-task quality observation is supplied separately from physical calculations."), kind="info"), b_capture, saved("B"),
            mo.accordion({"Calculation Notes": mo.md("Memory sums weights, activations, and workspace. Latency sums movement, compute, and fixed overhead. Phase energy is power multiplied by phase duration.")})])
    def part_c():
        if part_c_prediction.value is None:
            return mo.vstack([mo.md("### C · Where does the recommendation stop working? (10 min)\nTighten one explicit limit while holding the design fixed."), boundary_axis, boundary_fraction, part_c_prediction])
        changed = boundary_comparison["changed"]
        rows = [{"Envelope": "Original", "Outcome": "PASS" if boundary_comparison["baseline"]["feasible"] else "FAIL", "Failed": ", ".join(boundary_comparison["baseline"]["failed_requirements"]) or "none"},
                {"Envelope": "Tightened", "Outcome": "PASS" if changed["feasible"] else "FAIL", "Failed": ", ".join(changed["failed_requirements"]) or "none"}]
        return mo.vstack([mo.md("### C · Where does the recommendation stop working? (10 min)"), boundary_axis, boundary_fraction, part_c_prediction, table(rows),
            mo.callout(mo.md(f"The tightened envelope is **{'feasible' if changed['feasible'] else 'not feasible'}**. Acceptance changed; physical outcomes did not."), kind="success" if changed["feasible"] else "danger"), c_capture, saved("C"),
            mo.accordion({"Calculation Notes": mo.md("This changes one named limit in physical units. It does not use a universal workload or capability scale.")})])
    def part_d():
        if part_d_prediction.value is None:
            return mo.vstack([mo.md("### D · Can the reasoning transfer? (10 min)\nApply the same physical design to another explicit envelope."), target_track, part_d_prediction])
        rows = [{"Target": track_labels[target_track_value], "Memory (MB)": transfer_result["memory_mb"], "Latency (ms)": transfer_result["latency_ms"], "Energy (mJ)": transfer_result["energy_mj"], "Quality": "unavailable" if transfer_result["quality_pct"] is None else transfer_result["quality_pct"], "Failed": ", ".join(transfer_result["failed_requirements"]) or "none"}]
        return mo.vstack([mo.md("### D · Can the reasoning transfer? (10 min)"), target_track, part_d_prediction, table(rows),
            mo.callout(mo.md("Physical quantities can be reevaluated. Quality remains unavailable because the target task and population differ."), kind="info"), d_capture, saved("D"),
            mo.accordion({"Calculation Notes": mo.md("The saved design is compared with the second envelope. No target quality value is invented.")})])
    def synthesis():
        rows = [{"Part": part, "Prediction": _captures[part].to_dict()["prediction"] if part in _captures else "—", "Evidence": "current" if part in _captures and part not in evidence_audit.stale and (part, part) not in evidence_audit.identical_pairs else "missing or stale"} for part in "ABCD"]
        tested_decision = _captures["B"].to_dict()["decision"] if "B" in _captures else None
        complete = evidence_audit.complete and all(w.value is not None for w in (final_choice, final_rejected, final_trigger)) and final_choice.value == tested_decision and final_choice.value != final_rejected.value and bool(residual_risk.value.strip()) and bool(rationale.value.strip())
        return mo.vstack([mo.md("### Synthesis · Defend the operating envelope (8 min)\nChoose one option, quantify a tested rejected alternative, name the remaining limitation, and set a trigger."), table(rows), mo.hstack([final_choice, final_rejected], widths="equal", wrap=True), final_trigger, residual_risk, rationale,
            mo.callout(mo.md("**Ready for the local report.**" if complete else "Capture four current contrasts, use the saved Part B verdict, choose the other tested design, and complete the limitation, trigger, and rationale."), kind="success" if complete else "warn")])
    mo.ui.tabs({"Part A": part_a(), "Part B": part_b(), "Part C": part_c(), "Part D": part_d(), "Synthesis": synthesis()})
    return (evidence_audit,)


@app.cell
def _(build_lab_report, evidence_audit, final_choice, final_rejected,
      final_trigger, get_evidence, get_lab_metadata, instructor, mo, rationale,
      report_export_panel, residual_risk, track_id, track_labels):
    _captures = get_evidence()
    _tested_decision = _captures["B"].to_dict()["decision"] if "B" in _captures else None
    _ready = evidence_audit.complete and all(w.value is not None for w in (final_choice, final_rejected, final_trigger)) and final_choice.value == _tested_decision and final_choice.value != final_rejected.value and bool(residual_risk.value.strip()) and bool(rationale.value.strip())
    mo.stop(not _ready)
    snapshots = {part: _captures[part].to_dict() for part in "ABCD"}
    report = build_lab_report(
        get_lab_metadata("vol1/lab_16_ml_conclusion.py"), track=track_id,
        scenario=f"{track_labels[track_id]} cross-layer release audit",
        learning_objectives=["Validate accumulated evidence without filling gaps", "Compose constraints across layers", "Find a boundary and test transfer"],
        predictions={part: snapshots[part]["prediction"] for part in "ABCD"},
        knob_settings={part: snapshots[part]["inputs"] for part in "ABCD"},
        evidence_summary={part: {"baseline": snapshots[part]["baseline"], "result": snapshots[part]["result"], "alternatives": snapshots[part]["alternatives"]} for part in "ABCD"},
        binding_constraints={part: snapshots[part]["result"].get("failed_requirements", []) for part in "BCD"},
        decisions={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value},
        final_decision={"recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "rationale": rationale.value},
        big_takeaways=["Missing evidence and failed constraints demand different responses.", "A local change moves costs across the system.", "Quality transfers only across matched tasks and populations."],
        reflections={"rationale": rationale.value, "remaining_limitation": residual_risk.value}, residual_risk=residual_risk.value,
        result_snapshot={"track": track_id, "captures": snapshots, "recommendation": final_choice.value, "rejected": final_rejected.value, "trigger": final_trigger.value, "residual_risk": residual_risk.value},
        source_trace={"scenario": instructor["scenario_assumption"], "quality": instructor["quality_assumption"]})
    mo.vstack([mo.md("## Local evidence report"), report_export_panel(report)])
    return (report,)


@app.cell
async def _(evidence_audit, final_choice, final_rejected, final_trigger,
      get_evidence, ledger, mo, rationale, residual_risk, track_id):
    _captures = get_evidence()
    _tested_decision = _captures["B"].to_dict()["decision"] if "B" in _captures else None
    _ready = evidence_audit.complete and all(w.value is not None for w in (final_choice, final_rejected, final_trigger)) and final_choice.value == _tested_decision and final_choice.value != final_rejected.value and bool(residual_risk.value.strip()) and bool(rationale.value.strip())
    _saved = False
    _save_error = None
    if _ready:
        try:
            ledger.save(chapter=16, design={"schema_version": 1, "lab_id": "v1_16", "track_id": track_id, "model_id": "v1_16_experiments", "evidence": {part: capture.to_dict() for part, capture in get_evidence().items()}, "recommendation": final_choice.value, "rejected_alternative": final_rejected.value, "reevaluation_trigger": final_trigger.value, "residual_risk": residual_risk.value, "rationale": rationale.value})
            await ledger.flush()
            _saved = True
        except Exception as exc:
            _save_error = str(exc)
    _status = "SAVED" if _saved else (f"SAVE FAILED · {_save_error}" if _save_error else "EVIDENCE IN PROGRESS")
    mo.Html(f'<div class="lab-hud"><span class="hud-label">LAB</span><span>16 · The Architect\'s Audit</span><span style="flex:1"></span><span class="hud-label">STATUS</span><span class="hud-active">{_status}</span></div>')
    return


if __name__ == "__main__":
    app.run()
