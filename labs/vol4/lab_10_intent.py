import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full", app_title="Physical AI · MLSysBook")


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
    from mlsysbook_labs.learning_paths import get_learning_path
    from mlsysbook_labs.notebook import notebook_shell
    from mlsysbook_labs.schemas import LabMetadata
    from mlsysbook_labs.physical_companion import (
        physical_controls, physical_discussion_control, physical_navigation, physical_opening,
        physical_investigations, physical_discussion, physical_actions, physical_discussion_action,
    )
    contract = get_learning_path(4, 10)
    metadata = LabMetadata(lab_id="v4_10", title=contract["title"], volume="Volume IV",
                           chapter="Lab 10", book_anchor="sec-intent")
    return (mo, contract, metadata, notebook_shell,
            physical_controls, physical_discussion_control, physical_navigation, physical_opening,
            physical_investigations, physical_discussion, physical_actions, physical_discussion_action)


@app.cell
def _(mo):
    get_pinned, set_pinned = mo.state({}, allow_self_loops=True)
    get_saved, set_saved = mo.state({}, allow_self_loops=True)
    get_discussion, set_discussion = mo.state("", allow_self_loops=True)
    return get_pinned, set_pinned, get_saved, set_saved, get_discussion, set_discussion


@app.cell
def _(contract, metadata, part_navigation, physical_opening):
    physical_opening(contract, metadata, part_navigation.value.split(" · ", 1)[0])
    return


@app.cell
def _(contract, physical_controls, physical_discussion_control, physical_navigation):
    investigations = physical_controls(contract["chapter"])
    discussion = physical_discussion_control()
    part_navigation = physical_navigation(contract)
    return investigations, discussion, part_navigation


@app.cell
def _(contract, investigations, physical_actions, get_pinned, set_pinned, get_saved, set_saved, physical_discussion_action, discussion, set_discussion):
    actions = physical_actions(contract, investigations, get_pinned, set_pinned, get_saved, set_saved)
    save_button, discussion_download, evidence_download = physical_discussion_action(contract, investigations, discussion, get_saved, get_pinned, set_discussion)
    return actions, save_button, discussion_download, evidence_download


@app.cell
def _(part_navigation, notebook_shell):
    notebook_shell([part_navigation], volume=4)
    return


@app.cell
def _(contract, investigations, physical_investigations, get_pinned, set_pinned, get_saved, set_saved, part_navigation, actions):
    physical_investigations(contract, investigations, investigations.value,
                            get_pinned, set_pinned, get_saved, set_saved, part_navigation.value.split(" · ", 1)[0], get_pinned(), get_saved(), actions)
    return


@app.cell
def _(contract, investigations, get_saved, get_pinned, discussion, physical_discussion, get_discussion, set_discussion, save_button, discussion_download, evidence_download):
    physical_discussion(contract, investigations.value, get_saved(), get_pinned(), discussion,
                        discussion.value, get_discussion, set_discussion, save_button, discussion_download, evidence_download)
    return


if __name__ == "__main__":
    app.run()
