import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="Lab 00: Choose a fleet project · MLSysBook")


@app.cell
async def _():
    import sys
    import json
    from pathlib import Path
    import marimo as mo
    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.experiments.v2_01_experiments import TRACK_PROFILES
    from mlsysbook_labs.contexts import CONTEXT_OPTIONS, preferred_context, restore_orientation
    from mlsysbook_labs.onboarding import briefing_comparison, capture_briefing, setup_record, save_setup, volume_route
    from mlsysbook_labs.notebook import notebook_shell, notebook_header, notebook_bridge, notebook_learning_map, notebook_tab_hint, notebook_scenario, notebook_submission
    from mlsysbook_labs.learning_paths import get_learning_path
    from mlsysbook_labs import get_lab_metadata
    ledger = DesignLedger(volume="vol2")
    if ledger.is_wasm:
        await ledger.load_async()
    restored = restore_orientation(ledger)
    get_capture, set_capture = mo.state(restored.get("comparison"))
    get_status, set_status = mo.state({})
    return CONTEXT_OPTIONS, get_lab_metadata, TRACK_PROFILES, briefing_comparison, capture_briefing, get_capture, get_learning_path, get_status, json, ledger, mo, notebook_bridge, notebook_header, notebook_learning_map, notebook_shell, notebook_tab_hint, notebook_scenario, notebook_submission, preferred_context, restored, save_setup, set_capture, set_status, setup_record, volume_route


@app.cell
def _(CONTEXT_OPTIONS, ledger, mo, preferred_context, set_capture, set_status):
    def _change(_value):
        set_capture(None)
        set_status({})
    track = mo.ui.dropdown(CONTEXT_OPTIONS,
        value=next(label for label, key in CONTEXT_OPTIONS.items() if key == preferred_context(ledger)),
        label="My fleet project", full_width=True, on_change=_change)
    return (track,)


@app.cell
def _(TRACK_PROFILES, track):
    track_id = track.value
    profile = TRACK_PROFILES[track_id]
    return profile, track_id


@app.cell
def _(mo, track_id):
    _identity = track_id
    prediction = mo.ui.radio({
        "Changing my prediction changes the briefing": "prediction",
        "Changing the view changes the live briefing": "control",
        "Saving changes the captured briefing": "save",
    }, label="Which action changes the live briefing?").form(submit_button_label="Lock my prediction")
    view = mo.ui.dropdown({"Project setting": "baseline", "Model preparation": "alternative"},
        value="Project setting", label="Live briefing view", full_width=True)
    return prediction, view


@app.cell
def _(briefing_comparison, profile, track_id, view):
    baseline, live = briefing_comparison(track_id, profile, view.value)
    return baseline, live


@app.cell
def _(baseline, capture_briefing, live, mo, prediction, set_capture, set_status, track_id, view):
    def _capture(_value):
        set_capture(capture_briefing(track_id, prediction.value, baseline, live))
        set_status({})
    capture = mo.ui.button(label="Capture this comparison", kind="success", on_click=_capture,
        disabled=prediction.value is None or view.value != "alternative")
    return (capture,)


@app.cell
def _(get_capture, mo, track_id):
    _saved = get_capture()
    save = mo.ui.run_button(label="Save my progress", kind="success",
        disabled=not _saved or _saved.get("track") != track_id)
    return (save,)


@app.cell
async def _(get_capture, ledger, mo, save, save_setup, set_status, setup_record, track_id):
    mo.stop(not save.value)
    _payload = setup_record(2, track_id, get_capture())
    try:
        await save_setup(ledger, _payload)
        set_status({"status": "saved"})
    except Exception as _error:
        set_status({"status": "unavailable", "message": str(_error)})
    return


@app.cell
def _(get_lab_metadata, get_learning_path, mo, notebook_bridge, notebook_header, notebook_learning_map, notebook_scenario, notebook_shell, profile, track):
    _contract = get_learning_path(2, 0)
    _metadata = get_lab_metadata("vol2/lab_00_orientation.py")
    notebook_shell([
        notebook_header(_metadata, "Choose a continuing project and try how the labs work before the technical investigations.", label="Lab 00 · Orientation", topics=("Choose a track", "Try the workflow", "Save and resume")),
        track,
        notebook_scenario(_contract, track=track.value, volume=2),
        mo.md(f"**Model preparation for this project:** {profile.scaling_system}. Tiny and mobile devices do not become distributed training workers just because their backend scales."),
        notebook_learning_map(_contract),
    ], volume=2)
    return


@app.cell
def _(baseline, capture, get_capture, get_learning_path, get_status, json, live, mo, notebook_shell, notebook_tab_hint, prediction, restored, save, setup_record, track_id, view, volume_route, notebook_submission):
    _captured = get_capture()
    _status = get_status()
    _a = mo.vstack([
        mo.md("## A · Choose your continuing project\nChoose the deployment context that interests you. The project continues from infrastructure and communication through training, serving, and a final recommendation. Each lab follows the same structure: a situation and clear learning goals, Parts A–D or A–E that each explore a chapter concept, and a discussion that connects your findings. The tracks change the setting and constraints, not the difficulty. Switching tracks begins a new rehearsal so your comparisons keep their original context."),
        mo.md("**The learning goal:** explain what improves, what it costs, which condition stayed fixed, and what evidence could reverse your decision. The book provides the models, TinyTorch builds the machinery, these labs explore tradeoffs, and StaffML practices explaining your reasoning."),
        mo.md("**Before you begin:** the browser route needs no local installation. This orientation lets you practice the interface before Lab 01 begins the technical investigations."),
    ])
    _b_items = [mo.md("## B · Try the investigation cycle\nPredict first. Then change one view and inspect the immediate feedback. This rehearsal changes text, not a physical model."), prediction]
    if prediction.value is not None:
        _b_items += [view,
            mo.md(f"**Baseline · project setting:** {baseline['system']}\n\n**Live · {view.value}:** {live['system']}\n\n**Question now:** {live['question']}"),
            mo.callout(mo.md("Your prediction is locked. The view control updates the briefing immediately. Capture keeps a copy of the comparison for your discussion. Select Model preparation, capture, then switch back to Project setting to see how your saved comparison stays the same."), kind="info"), capture]
        if _captured:
            _same = _captured["result"]["inputs"] == live["inputs"]
            _b_items += [mo.callout(mo.md(f"**Captured briefing:** {_captured['result']['system']}.\n\n**Live state:** {'matches the capture' if _same else 'changed since capture; your saved comparison stays the same'}."), kind="success" if _same else "warn")]
    _c_items = [mo.md("## C · Save, reload, resume\nKeep the comparison you want to discuss, then save your progress to return to this project later. Download your work for a copy you can read and share. Saving checks that your project and comparison can be loaded again. This page does not import downloaded files."), save]
    if _status.get("status") == "saved":
        _c_items += [mo.callout(mo.md("**Saved and read back.** Reload to restore this track and comparison. Lab 01 will also use your preferred track."), kind="success")]
    elif _status.get("status") == "unavailable":
        _c_items += [mo.callout(mo.md("Storage is unavailable. Keep the download and retry; you can begin Lab 01 without saved progress."), kind="warn")]
    elif restored:
        _c_items += [mo.callout(mo.md("**Welcome back.** Your saved track and captured comparison were loaded from Volume II."), kind="success")]
    else:
        _c_items += [mo.md("Capture a comparison in Part B to enable saving and download.")]
    if _captured:
        _c_items += [mo.download(json.dumps(setup_record(2, track_id, _captured), indent=2).encode(), filename="mlsysbook-vol2-setup.json", label="Download your work")]
    _c_items += [mo.md("Progress is local to this browser and site, or to the local account in a native run. Clearing browser data can remove access. Unsaved control movements do not persist. A saved preference is not deployment evidence.")]
    _route = [_entry for _entry in volume_route(2) if _entry["chapter"] > 0]
    _d = mo.vstack([
        mo.md("## D · See the whole route\nEach lab introduces its own concepts. This route previews the questions; it does not require their answers yet. Carry the project identity forward, then use compatible captured results in each discussion. A supported hold is a valid outcome."),
        mo.md("\n".join(f"- **Lab {_entry['chapter']:02d} · {_entry['title']}** — {_entry.get('goal', _entry['starting_point'])}" for _entry in _route)),
        mo.md("**Ready to proceed:** choose a track, lock a prediction, change the briefing, capture it, and save or download your work. In later labs, submit a discussion that cites captured comparisons, explains a rejected alternative, identifies uncertainty, and gives a reversal condition."),
        mo.md("[Open Lab 01 · Introduction](../../lab_01_introduction/) · [Read the Volume II introduction](https://mlsysbook.ai/vol2/introduction/introduction.html)"),
        mo.accordion({"Run locally": mo.md("Open `labs/vol2/lab_00_orientation.py` with Marimo to rehearse setup, then `labs/vol2/lab_01_introduction.py` to begin investigations. Each lab also supports a standalone start with supplied defaults; no earlier volume is required.")}),
    ])
    notebook_shell([notebook_tab_hint(), mo.ui.tabs({"A · Choose a project": _a, "B · Try the workflow": mo.vstack(_b_items), "C · Save and resume": mo.vstack(_c_items), "D · Full route and next": _d}), notebook_submission(get_learning_path(2, 0), orientation=True)], volume=2)
    return


if __name__ == "__main__":
    app.run()

