"""
Release regression tests for student-facing CLI and API correctness.
"""

import importlib.util
import io
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from rich.console import Console

from tito.commands.export_utils import find_source_file_for_export
from tito.commands.milestone import (
    MILESTONE_SCRIPTS,
    MilestoneCommand,
    MilestoneSystem,
    _module_progress_to_int,
    _required_modules_for,
    _validate_required_exports,
)
from tito.commands.module.workflow import ModuleWorkflowCommand
from tito.commands.package.reset import ResetCommand
from tito.core.config import CLIConfig


TINYTORCH_ROOT = Path(__file__).resolve().parents[2]


def _import_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mlperf_full_requirements_match_all_default_parts():
    milestone = MILESTONE_SCRIPTS["06"]
    part_union = sorted({
        module
        for script in milestone["scripts"]
        for module in script["required_modules"]
    })

    assert _required_modules_for(milestone) == part_union
    assert {11, 12}.issubset(set(milestone["required_modules"]))


def test_export_validator_rejects_silent_none_exports(monkeypatch):
    import tinytorch

    monkeypatch.setattr(tinytorch, "Tensor", None)

    failures = _validate_required_exports([1])

    assert "tinytorch.Tensor: exported as None" in failures


def test_export_validator_accepts_current_core_exports():
    assert _validate_required_exports([1, 2, 3]) == []


def test_export_source_mappings_match_current_default_exp_targets():
    assert (
        find_source_file_for_export(Path("tinytorch/perf/benchmarking.py"))
        == "src/19_benchmarking/19_benchmarking.py"
    )
    assert (
        find_source_file_for_export(Path("tinytorch/olympics.py"))
        == "src/20_capstone/20_capstone.py"
    )


def test_module_workflow_reports_default_exp_export_paths(monkeypatch):
    monkeypatch.chdir(TINYTORCH_ROOT)
    command = ModuleWorkflowCommand(CLIConfig.from_project_root(TINYTORCH_ROOT))

    assert command._get_export_path_for_module("19_benchmarking") == "tinytorch/perf/benchmarking.py"
    assert command._get_export_path_for_module("20_capstone") == "tinytorch/olympics.py"


def test_module_next_steps_use_start_subcommand():
    command = ModuleWorkflowCommand(CLIConfig.from_project_root(TINYTORCH_ROOT))
    output = io.StringIO()
    command.console = Console(file=output, width=120)

    command.show_next_steps("01")

    text = output.getvalue()
    assert "tito module start 02" in text


def test_root_public_api_exports_completed_module_symbols():
    import tinytorch

    expected_symbols = [
        "BatchNorm2d",
        "TinyGPT",
        "Profiler",
        "quick_profile",
        "Quantizer",
        "quantize_int8",
        "dequantize_int8",
        "Benchmark",
        "MLPerf",
    ]

    for symbol in expected_symbols:
        assert symbol in tinytorch.__all__
        assert getattr(tinytorch, symbol) is not None


def test_scalar_left_tensor_ops_preserve_autograd():
    from tinytorch import Tensor

    x = Tensor([2.0, 4.0], requires_grad=True)

    np.testing.assert_allclose((2 + x).data, [4.0, 6.0])
    np.testing.assert_allclose((10 - x).data, [8.0, 6.0])
    np.testing.assert_allclose((3 * x).data, [6.0, 12.0])
    np.testing.assert_allclose((12 / x).data, [6.0, 3.0])

    loss = (10 - x).sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, [-1.0, -1.0])

    x.zero_grad()
    loss = (12 / x).sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, [-3.0, -0.75])


def test_mlperf_optimization_loads_packaged_tinydigits():
    script = TINYTORCH_ROOT / "milestones" / "06_2018_mlperf" / "01_optimization_olympics.py"
    module = _import_script(script)

    train_images, train_labels, test_images, test_labels = module.load_tinydigits_arrays(TINYTORCH_ROOT)

    assert train_images.shape[1:] == (8, 8)
    assert test_images.shape[1:] == (8, 8)
    assert train_labels.shape[0] == train_images.shape[0]
    assert test_labels.shape[0] == test_images.shape[0]
    assert set(np.unique(train_labels)).issubset(set(range(10)))


def test_generation_speedup_import_error_lists_actual_requirements():
    text = (
        TINYTORCH_ROOT
        / "milestones"
        / "06_2018_mlperf"
        / "02_generation_speedup.py"
    ).read_text(encoding="utf-8")

    assert "modules 01-08, 11, 12, 14, and 18" in text
    assert "modules 11-17" not in text


def test_milestone_list_uses_actual_history_start_year():
    env = os.environ.copy()
    env["TITO_ALLOW_SYSTEM"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "tito.main", "milestone", "list", "--simple"],
        cwd=TINYTORCH_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "1958 to 2018" in result.stdout
    assert "1957 to 2018" not in result.stdout


def test_package_reset_success_messages_render_real_newlines(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    command = ResetCommand(CLIConfig.from_project_root(TINYTORCH_ROOT))
    output = io.StringIO()
    command.console = Console(file=output, width=120)

    assert command._reset_progress(Namespace(force=True, backup=False)) == 0
    assert command._reset_milestones(Namespace(force=True, backup=False)) == 0

    text = output.getvalue()
    assert "\\n" not in text
    assert "You can re-complete modules with:" in text
    assert "tito module complete XX" in text
    assert "You can re-run milestones with:" in text
    assert "tito milestone run XX" in text


def test_generated_warning_points_to_current_export_command():
    text = (TINYTORCH_ROOT / "tito" / "commands" / "export_utils.py").read_text(encoding="utf-8")

    assert "tito module complete XX" in text
    assert "tito module complete <module_name>" not in text


def _write_milestone_progress_files(tmp_path, completed_modules, unlocked_milestones=None):
    tito_dir = tmp_path / ".tito"
    tito_dir.mkdir(exist_ok=True)
    (tito_dir / "progress.json").write_text(json.dumps({"completed_modules": completed_modules}))
    (tito_dir / "milestones.json").write_text(json.dumps({
        "completed_milestones": [],
        "completion_dates": {},
        "unlocked_milestones": unlocked_milestones or [],
        "unlock_dates": {},
        "total_unlocked": len(unlocked_milestones or []),
        "achievements": [],
    }))


def test_can_unlock_true_when_required_and_trigger_complete_and_not_unlocked(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_milestone_progress_files(tmp_path, completed_modules=[1, 2, 3])

    system = MilestoneSystem(CLIConfig.from_project_root(TINYTORCH_ROOT))
    status = system.get_milestone_status()

    assert status["milestones"]["01"]["can_unlock"] is True


def test_can_unlock_false_when_already_unlocked(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_milestone_progress_files(tmp_path, completed_modules=[1, 2, 3], unlocked_milestones=["01"])

    system = MilestoneSystem(CLIConfig.from_project_root(TINYTORCH_ROOT))
    status = system.get_milestone_status()

    assert status["milestones"]["01"]["can_unlock"] is False


def test_can_unlock_false_when_nothing_completed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_milestone_progress_files(tmp_path, completed_modules=[])

    system = MilestoneSystem(CLIConfig.from_project_root(TINYTORCH_ROOT))
    status = system.get_milestone_status()

    assert status["milestones"]["01"]["can_unlock"] is False


def test_module_progress_to_int_accepts_int():
    assert _module_progress_to_int(6) == 6


def test_module_progress_to_int_parses_numeric_prefix_string():
    assert _module_progress_to_int("06_autograd") == 6


def test_module_progress_to_int_rejects_unparseable_string():
    assert _module_progress_to_int("abc") is None


def test_module_progress_to_int_rejects_wrong_type():
    assert _module_progress_to_int(None) is None
    assert _module_progress_to_int(3.5) is None


def test_export_validator_reports_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tinytorch.core.tensor", None)

    failures = _validate_required_exports([1])

    assert any(
        failure.startswith("tinytorch.core.tensor.Tensor: import failed")
        for failure in failures
    )


def test_run_command_stops_after_first_failure_when_noninteractive(monkeypatch):
    monkeypatch.chdir(TINYTORCH_ROOT)
    command = MilestoneCommand(CLIConfig.from_project_root(TINYTORCH_ROOT))
    command.console = Console(file=io.StringIO(), width=120)

    args = Namespace(milestone_id="06", part=None, skip_checks=True)

    run_calls = []

    def fake_run(cmd, **kwargs):
        run_calls.append(cmd)
        result = MagicMock()
        result.returncode = 1
        return result

    with patch("tito.commands.milestone.subprocess.run", side_effect=fake_run), \
         patch("sys.stdin") as mock_stdin, \
         patch("sys.stdout") as mock_stdout, \
         patch("builtins.input") as mock_input:
        mock_stdin.isatty.return_value = False
        mock_stdout.isatty.return_value = False

        returncode = command._handle_run_command(args)

    assert returncode == 1
    assert len(run_calls) == 1
    mock_input.assert_not_called()
