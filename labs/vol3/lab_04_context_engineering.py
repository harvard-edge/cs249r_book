import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="Volume III · Context Engineering")


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
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)
    from mlsysbook_labs import agentic_companion as companion
    from mlsysbook_labs import get_lab_metadata
    from mlsysbook_labs.learning_paths import get_learning_path
    from mlsysim.labs.experiments.v3_studies import choices, study, projects, definition
    contract = get_learning_path("vol3", 4)
    metadata = get_lab_metadata("vol3/lab_04_context_engineering.py")
    get_saved, set_saved = mo.state({})
    get_discussion, set_discussion = mo.state(None)
    return choices, companion, contract, definition, get_discussion, set_discussion, get_saved, metadata, mo, projects, set_saved, study


@app.cell
def _(companion, mo, projects):
    project_catalog = projects()
    track = companion.project_selector(mo, project_catalog)
    return project_catalog, track


@app.cell
def _(companion, contract, metadata, mo, project_catalog, track):
    companion.opening(mo, contract, metadata, project_catalog, track)
    return


@app.cell
def _(choices, companion, contract, definition, mo, track):
    predictions, trials, interpretations = companion.investigation_controls(mo, 4, contract, track.value, choices, definition)
    return interpretations, predictions, trials


@app.cell
def _(companion, contract, predictions, study, track, trials):
    live = companion.evaluate_trials(4, contract, track.value, predictions.value, trials.value, study)
    return (live,)


@app.cell
def _(companion, contract, interpretations, live, mo, predictions, set_saved, track):
    captures = companion.evidence_actions(mo, 4, contract, track.value, live, predictions.value, interpretations.value, set_saved)
    return (captures,)


@app.cell
def _(get_saved):
    saved = get_saved()
    return (saved,)


@app.cell
def _(captures, companion, contract, definition, interpretations, live, mo, predictions, saved, track, trials):
    companion.part_workspace(mo, contract, track.value, predictions, trials, interpretations, live, captures, saved, definition)
    return


@app.cell
def _(mo, track):
    _identity = track.value
    decision = mo.ui.text_area(label="What do you recommend, why, and what would change your mind?", full_width=True, debounce=False)
    return (decision,)


@app.cell
def _(companion, contract, decision, interpretations, live, mo, predictions, saved, set_discussion, track):
    record = companion.discussion_record(contract, track.value, live, interpretations.value, predictions.value, saved, decision.value)
    save_discussion = companion.discussion_action(mo, record, set_discussion)
    return record, save_discussion


@app.cell
def _(get_discussion):
    saved_discussion = get_discussion()
    return (saved_discussion,)


@app.cell
def _(companion, contract, decision, mo, record, save_discussion, saved_discussion):
    companion.final_workspace(mo, contract, decision, record, save_discussion, saved_discussion)
    return


if __name__ == "__main__":
    app.run()
