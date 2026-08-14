"""
Regression tests for milestone-05's final grading logic.

Milestone 05 (milestones/05_2017_transformer/01_vaswani_attention.py) decides
pass/fail via `print_final_results`, which combines three challenge results
with `passed1 and passed2 and passed3`. This has no direct test coverage
today, so these tests lock in the existing (audited, correct) behavior.
"""

import importlib.util
import sys
from pathlib import Path

TINYTORCH_ROOT = Path(__file__).resolve().parents[2]
MILESTONE_05_SCRIPT = TINYTORCH_ROOT / "milestones" / "05_2017_transformer" / "01_vaswani_attention.py"

sys.path.insert(0, str(TINYTORCH_ROOT))
sys.path.insert(0, str(TINYTORCH_ROOT / "milestones"))


def _import_milestone_05():
    spec = importlib.util.spec_from_file_location(MILESTONE_05_SCRIPT.stem, MILESTONE_05_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_print_final_results_all_challenges_passing(capsys):
    module = _import_milestone_05()

    results = {
        "reversal": (True, 96.0),
        "copying": (True, 97.5),
        "mixed": (True, 91.0),
    }

    returncode = module.print_final_results(results)

    captured = capsys.readouterr()
    assert returncode == 0
    assert "MILESTONE 05 COMPLETE" in captured.out
    assert "CHALLENGES FAILED" not in captured.out


def test_print_final_results_only_one_challenge_passing(capsys):
    module = _import_milestone_05()

    results = {
        "reversal": (True, 96.0),
        "copying": (False, 40.0),
        "mixed": (False, 30.0),
    }

    returncode = module.print_final_results(results)

    captured = capsys.readouterr()
    assert returncode == 1
    assert "CHALLENGES FAILED" in captured.out
    assert "Copying" in captured.out
    assert "Mixed Tasks" in captured.out
    assert "Reversal" not in captured.out.split("CHALLENGES FAILED")[1].split("\n")[0]


def test_print_final_results_all_challenges_failing(capsys):
    module = _import_milestone_05()

    results = {
        "reversal": (False, 10.0),
        "copying": (False, 20.0),
        "mixed": (False, 5.0),
    }

    returncode = module.print_final_results(results)

    captured = capsys.readouterr()
    assert returncode == 1
    assert "CHALLENGES FAILED" in captured.out
    assert "Reversal" in captured.out.split("CHALLENGES FAILED")[1].split("\n")[0]
    assert "Copying" in captured.out.split("CHALLENGES FAILED")[1].split("\n")[0]
    assert "Mixed Tasks" in captured.out.split("CHALLENGES FAILED")[1].split("\n")[0]
