import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="Lab 00: Welcome to Your Deployment Project · MLSysBook")


@app.cell
async def _():
    import json
    import sys
    from pathlib import Path
    import marimo as mo

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.3.0-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS
    from mlsysbook_labs import get_lab_metadata
    from mlsysbook_labs.notebook import notebook_bridge, notebook_header, notebook_shell, notebook_status
    from mlsysbook_labs.contexts import CONTEXT_OPTIONS, get_context, preferred_context, restore_orientation
    from mlsysbook_labs.experiment_evidence import capture_evidence

    ledger = DesignLedger(volume="vol1")
    if ledger.is_wasm:
        await ledger.load_async()
    restored_orientation = restore_orientation(ledger)
    return (
        COLORS, CONTEXT_OPTIONS, DesignLedger,
        capture_evidence, get_context, json, ledger, mo,
        get_lab_metadata, notebook_bridge, notebook_header, notebook_shell, notebook_status,
        preferred_context, restored_orientation,
    )


@app.cell
def _(mo, restored_orientation):
    _previous = restored_orientation.get("comparison")
    get_capture, set_capture = mo.state(_previous if isinstance(_previous, dict) else None)
    get_save_status, set_save_status = mo.state({})
    return get_capture, get_save_status, set_capture, set_save_status


@app.cell
def _(CONTEXT_OPTIONS, get_context, ledger, mo, preferred_context, set_capture, set_save_status):
    def _change_context(_value):
        set_capture(None)
        set_save_status({})

    track = mo.ui.dropdown(
        CONTEXT_OPTIONS, value=get_context(preferred_context(ledger)).label,
        label="My deployment context", on_change=_change_context,
    )
    return (track,)


@app.cell
def _(get_context, track):
    track_id = track.value
    context = get_context(track_id)
    project = context.scenario
    orientation_episode = project.episode(0)
    return context, orientation_episode, project, track_id


@app.cell
def _(CONTEXT_OPTIONS, mo, track_id):
    _track_key = track_id
    compare_context = mo.ui.dropdown(
        CONTEXT_OPTIONS, value="MobileML" if track_id == "tinyml" else "TinyML",
        label="Compare with",
    )
    orientation_predictions = {
        "Changing my prediction changes the case": "prediction",
        "Changing the case control changes the live card": "control",
        "Saving a comparison changes its results": "save",
    }
    prediction = mo.ui.radio(
        orientation_predictions,
        label="Which action will change the live case card?",
    ).form(submit_button_label="Lock my prediction")
    case = mo.ui.dropdown(
        {"Baseline briefing": "baseline", "Alternative briefing": "alternative"},
        value="Baseline briefing", label="Live case",
    )
    return case, compare_context, orientation_predictions, prediction


@app.cell
def _(case, orientation_episode, project, track_id):
    # This is an interface rehearsal, not a physical or quality model.
    baseline_card = {
        "inputs": {"case": "baseline", "context": track_id},
        "briefing": project.mission,
        "next_action": "Keep a reference for comparison.",
    }
    live_card = baseline_card if case.value == "baseline" else {
        "inputs": {"case": "alternative", "context": track_id},
        "briefing": orientation_episode.task,
        "next_action": "Compare the new setting with the original before deciding.",
    }
    return baseline_card, live_card


@app.cell
def _(baseline_card, capture_evidence, case, live_card, mo, prediction, set_capture, set_save_status, track_id):
    def _capture(_value):
        snapshot = capture_evidence(
            track=track_id, part="orientation", prediction=prediction.value,
            inputs=live_card["inputs"], baseline=baseline_card, result=live_card,
            decision="Compare before deciding", model_key="orientation.interface.v1",
            result_role="interface rehearsal",
        )
        set_capture(snapshot.to_dict())
        set_save_status({})

    capture_button = mo.ui.button(
        label="Capture this comparison", kind="success", on_click=_capture,
        disabled=prediction.value is None or case.value == "baseline",
        tooltip="Lock a prediction and select the alternative briefing first.",
    )
    return (capture_button,)


@app.cell
def _(get_capture, mo, track_id):
    _captured = get_capture()
    save_button = mo.ui.run_button(
        label="Save my progress", kind="success",
        disabled=not _captured or _captured.get("track") != track_id,
    )
    return (save_button,)


@app.cell
async def _(DesignLedger, get_capture, ledger, mo, project, save_button, set_save_status, track_id):
    mo.stop(not save_button.value)
    _comparison = get_capture()
    mo.stop(not _comparison or _comparison.get("track") != track_id)
    _payload = {
        "schema_version": 1, "orientation_version": 1,
        "lab_id": "v1_00_architects_portal", "preferred_track": track_id,
        "scenario_id": project.scenario_id, "scenario_version": project.version,
        "comparison": _comparison,
    }
    set_save_status({"status": "pending"})
    try:
        # Preserve more recent chapter records before writing the preference.
        await ledger.load_async()
        await ledger.asave(track=track_id, chapter=0, design=_payload)
        _fresh = DesignLedger(volume="vol1")
        await _fresh.load_async()
        if _fresh.last_load_error or _fresh.get_design(0) != _payload:
            raise RuntimeError("Your saved progress could not be restored.")
        set_save_status({"status": "saved", "payload": _payload})
    except Exception as _error:
        set_save_status({"status": "unavailable", "message": str(_error), "payload": _payload})
    return


@app.cell(hide_code=True)
def _(context, get_lab_metadata, mo, notebook_bridge, notebook_header, notebook_shell, project, track):
    from mlsysbook_labs.notebook import notebook_scenario as _scenario
    from mlsysbook_labs.learning_paths import get_learning_path
    from mlsysbook_labs.notebook import notebook_learning_map
    _learning = get_learning_path(1, 0)
    _learning["next_lab"]["question"] = project.episode(1).question
    ORIENTATION_CSS = """<style>
    .orientation-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:12px;margin:12px 0}
    .orientation-card{border:1px solid #cbd5e1;border-radius:9px;padding:16px;background:#f8fafc;color:#172033;overflow-wrap:anywhere}
    .orientation-card h3{margin:0 0 8px;font-size:1.1rem}.orientation-card p{margin:6px 0}
    .orientation-saved{border-left:4px solid #15803d;background:#f0fdf4;padding:12px;border-radius:7px;color:#14532d}
    </style>"""
    notebook_shell([
        notebook_header(get_lab_metadata("vol1/lab_00_introduction.py"),
            "Choose a deployment project, then rehearse how you will investigate it.",
            label="Lab 00 · Orientation", topics=("Choose a context", "Try the controls", "Save and resume")),
        mo.hstack([track, mo.md(f"**Your project:** {project.title}")], justify="start", gap=1, wrap=True),
        _scenario(_learning, track=track.value, scene=project.episode(0).scene + "\n\n" + _learning["narrative"]["scene"]),
        notebook_learning_map(_learning),
        mo.accordion({"Before you begin": mo.md("No prior reading or local installation is needed for the browser route. This orientation teaches the notebook workflow. Lab 01 begins the technical investigations.")}),
    ])
    return (ORIENTATION_CSS,)



@app.cell(hide_code=True)
def _(
    ORIENTATION_CSS, baseline_card, capture_button, case, compare_context, context, get_capture,
    get_context, get_save_status, json, ledger, live_card, mo, prediction,
    restored_orientation, save_button, track_id, orientation_predictions, notebook_shell,
    project, orientation_episode, notebook_bridge,
):
    _captured = get_capture()
    _status = get_save_status()

    def _card(title, body):
        return f'<section class="orientation-card"><h3>{title}</h3>{body}</section>'

    def _contexts():
        other = get_context(compare_context.value)
        question = "Which choice meets the task while respecting its resource limits?"
        return mo.vstack([
            mo.md("## A · Choose your deployment project\nRevisit one project across the series. Each chapter adds a different engineering question; its evidence informs what you investigate next."),
            compare_context,
            mo.Html(ORIENTATION_CSS + '<div class="orientation-grid">' + _card(context.label, f'<p><b>Project:</b> {project.title}</p><p>{project.mission}</p><p><b>Watch:</b> {context.pressure}</p>') + _card(other.label, f'<p><b>Project:</b> {other.scenario.title}</p><p>{other.scenario.mission}</p><p><b>Watch:</b> {other.pressure}</p>') + '</div>'),
            mo.callout(mo.md(f"**Shared question:** {question}"), kind="info"),
            mo.md("Compare another project, then choose the one you want to follow. Contexts are different parts of the deployment hierarchy, not difficulty levels. You can change projects later: the current lab starts fresh so one project’s evidence is never silently reused for another. TinyML model development happens on a separate host; the sensing device runs inference."),
            mo.accordion({
                "Your mission and chapter journey": mo.md(f"**Stakeholder:** {project.stakeholder}\n\n**Deployment boundary:** {project.boundary}\n\n**Success means:**\n\n- " + "\n- ".join(project.success) + "\n\n**Still unknown:**\n\n- " + "\n- ".join(project.unknowns) + f"\n\n**Orientation:** {orientation_episode.task}\n\n**Introduction:** {project.episode(1).task}\n\n**System paradigms:** {project.episode(2).task}\n\nLater chapters revisit the project through data, models, training, resource use, and deployment. Each lab names the chapter concept it tests; this is a series of design investigations, not a promise that an unvalidated prototype is ready to ship.\n\n{project.study_boundary}"),
                "Where the labs fit": mo.md("**Read** the book for the mental models. **Build** their machinery in TinyTorch. **Experiment** with choices and tradeoffs here. **Explain** your reasoning in StaffML interview practice."),
                "The interfaces you will use": mo.md("| Interface | What you do | What to notice |\n|:--|:--|:--|\n| Prediction form | Select an expectation, then lock it. | Any prediction unlocks the experiment. |\n| Controls | Change a supported choice with a selector, slider, or toggle. | Which condition changes and which stays fixed. |\n| Instruments | Inspect a chart, budget, diagram, trace, or table. | The benefit, competing cost, boundary, or missing evidence. |\n| Pin and capture | Retain a tested setting or freeze a comparison. | Saved evidence stays separate from live controls. |\n| Discussion and download | Explain your decision using captured runs and export it. | Your interpretation is separate from the recorded results. |\n| Save status | Check whether your progress was saved and can be restored. | A download remains available when storage is unavailable. |\n\nThe instruments vary by chapter. This orientation rehearses the controls and saving workflow; Lab 01 begins the technical investigations."),
            }),
        ])

    def _rehearsal():
        opening = mo.md(f"## B · Rehearse your project workflow\nUse **{project.title.lower()}** to try two briefing views. The control changes which project question you inspect. This is an interface rehearsal; it does not predict model quality or simulate a device.")
        if prediction.value is None:
            return mo.vstack([opening, prediction, mo.md("Any prediction unlocks the rehearsal. Being wrong is useful: it gives you something to compare with what happens.")])
        _saved_note = mo.callout(mo.md("No comparison captured yet. Select the alternative briefing, then capture it."), kind="warn")
        if _captured:
            _saved_note = mo.Html(ORIENTATION_CSS + f'<div class="orientation-saved"><b>Captured comparison</b><br>Prediction: {next(label for label, key in orientation_predictions.items() if key == _captured["prediction"])}<br>Saved case: {_captured["result"]["inputs"]["case"]}. Live case: {case.value}.<br>Moving the control does not rewrite the saved comparison.</div>')
        return mo.vstack([
            opening, prediction, case,
            mo.Html(ORIENTATION_CSS + '<div class="orientation-grid">' + _card('Baseline briefing', f'<p>{baseline_card["briefing"]}</p><p>{baseline_card["next_action"]}</p>') + _card('Live case · ' + case.value, f'<p>{live_card["briefing"]}</p><p>{live_card["next_action"]}</p>') + '</div>'),
            mo.callout(mo.md("**Try it:** choose the alternative, capture the comparison, then return the live control to baseline. Check which briefing changed and which comparison stayed fixed."), kind="info"),
            capture_button, _saved_note,
            mo.md("Your prediction records an expectation. The case control changes what you inspect. Capture freezes a comparison. None of those actions makes evidence stronger by itself."),
        ])

    def _save():
        _message = mo.callout(mo.md("Your comparison is ready. Save it to check persistence, or keep a download." if _captured else "Capture a comparison in Part B to enable saving."), kind="info" if _captured else "warn")
        if _status.get("status") == "pending":
            _message = mo.callout(mo.md("Saving your preference and checking it can be read back…"), kind="info")
        elif _status.get("status") == "saved":
            _message = mo.callout(mo.md("**Saved and read back.** Reload this page to restore the comparison, then open the introduction lab. It will start with your preferred context."), kind="success")
        elif _status.get("status") == "unavailable":
            _message = mo.callout(mo.md("**Storage is unavailable.** Your comparison is still available in this session. Download your work below, or retry saving. You can start the next lab without saved progress."), kind="warn")
        _items = [
            mo.md("## C · Save your progress\nSaving keeps your chosen context and comparison for your next visit. The notebook checks that your saved progress can be restored. Progress is local to this browser and site, or to the local account when running natively."),
            save_button, _message,
        ]
        if restored_orientation:
            _items.append(mo.callout(mo.md("**Restored from an earlier visit.** The preferred context and captured comparison were restored from your Volume I progress."), kind="success"))
        if ledger.last_load_error:
            _items.append(mo.callout(mo.md("Your earlier progress could not be restored. You can continue with a fresh session or download this session's work."), kind="warn"))
        if _captured:
            payload = {
                "schema_version": 1, "orientation_version": 1,
                "lab_id": "v1_00_architects_portal", "preferred_track": track_id,
                "comparison": _captured,
                "scenario_id": project.scenario_id, "scenario_version": project.version,
            }
            _items.append(mo.download(json.dumps(payload, indent=2).encode(), filename="mlsysbook-orientation.json", label="Download your work"))
        _items.append(mo.accordion({"What persists?": mo.md("The saved context and captured comparison persist when storage is available. Unsaved control movements do not. Clearing browser data or using another browser can remove access to your saved progress. The download is a readable backup; automatic file import is not part of this orientation.")}))
        return mo.vstack(_items)

    def _next():
        return mo.vstack([
            mo.md("## D · Start the first investigation\nYou are ready when you can choose a context, distinguish a prediction from a control, capture a comparison, and save it or keep a download."),
            notebook_bridge("Your selected project; save your progress or download your work.",
                "Deployment evidence begins in the introduction.",
                project.episode(1).question, labels=("Carry forward", "Still open", "Next")),
            mo.accordion({"Project handoff": mo.md(orientation_episode.handoff)}),
            mo.md("[Open Lab 01](../../lab_01_ml_intro/) · [Read the introduction](https://mlsysbook.ai/vol1/introduction/introduction.html)"),
            mo.md("**Local run:** open `labs/vol1/lab_01_ml_intro.py` with Marimo. A generated browser lab uses the launch link above."),
            mo.accordion({"Working without saved progress": mo.md("Each lab can start with the same project brief and fresh study defaults. Choose your project there and begin. The narrative continues even without saved progress. Prior results count as evidence only when the task and assumptions match; a project preference is not a measurement.")}),
        ])

    from mlsysbook_labs.notebook import notebook_tab_hint as _tab_hint
    notebook_shell([_tab_hint(), mo.ui.tabs({
        "A · Compare contexts": _contexts(), "B · Try the workflow": _rehearsal(),
        "C · Save and resume": _save(), "D · Next step": _next(),
    })])
    return


@app.cell(hide_code=True)
def _(context, get_capture, get_save_status, notebook_shell, notebook_status, restored_orientation):
    _status = get_save_status().get("status")
    _label = "PROGRESS SAVED" if _status == "saved" else ("PROGRESS RESTORED" if restored_orientation and not _status else ("COMPARISON CAPTURED" if get_capture() else "TRY THE WORKFLOW"))
    notebook_shell([notebook_status(_label, context.label)])
    return


@app.cell(hide_code=True)
def _():
    from mlsysbook_labs.learning_paths import get_learning_path as _submission_path
    from mlsysbook_labs.notebook import notebook_shell as _submission_shell, notebook_submission as _submission
    _submission_shell([_submission(_submission_path(1, 0), orientation=True)])
    return


if __name__ == "__main__":
    app.run()
