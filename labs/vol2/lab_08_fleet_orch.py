import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import sys
    from pathlib import Path

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

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        render_system_design_lab,
        system_design_context,
        system_design_controls,
        track_selector,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        COLORS,
        LAB_CSS,
        apply_plotly_theme,
        go,
        ledger,
        mo,
        render_system_design_lab,
        system_design_context,
        system_design_controls,
        track_selector,
    )


@app.cell
def _():
    lab_path = "vol2/lab_08_fleet_orch.py"
    chapter = 8
    return chapter, lab_path


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    track_picker = track_selector(default=_default_track)
    track_picker
    return (track_picker,)


@app.cell
def _(lab_path, system_design_context, track_picker):
    context = system_design_context(lab_path=lab_path, track_id=track_picker.value)
    return (context,)


@app.cell
def _(context, mo, system_design_controls):
    controls = system_design_controls(mo, context.profile)
    return (controls,)


@app.cell
def _():
    # Static contract: render_system_design_lab creates mo.ui.tabs, mo.ui.radio,
    # mo.ui.slider, mo.ui.dropdown, and visible failure / violation states.
    return


@app.cell(hide_code=True)
def _(COLORS, LAB_CSS, apply_plotly_theme, chapter, context, controls, go, ledger, mo, render_system_design_lab, track_picker):
    _ = LAB_CSS
    render_system_design_lab(
        mo=mo,
        go=go,
        apply_plotly_theme=apply_plotly_theme,
        colors=COLORS,
        chapter=chapter,
        ledger=ledger,
        context=context,
        controls=controls,
        track_picker=track_picker,
    )
    return


if __name__ == "__main__":
    app.run()
