"""Smoke tests for all runnable scripts in mlsysim/examples/."""

from pathlib import Path
import runpy
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_script_runs(script_path: Path, capsys):
    """Every script in examples/*.py should execute cleanly under __main__."""
    runpy.run_path(str(script_path), run_name="__main__")
    captured = capsys.readouterr()
    assert captured.out.strip(), f"{script_path.name} produced no stdout"
