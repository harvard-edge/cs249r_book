"""
Release regression tests for student-facing CLI and API correctness.
"""

import importlib.util
import io
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from rich.console import Console

from tito.commands.export_utils import find_source_file_for_export
from tito.commands.milestone import (
    MILESTONE_SCRIPTS,
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
        text=True, encoding='utf-8', errors='replace',
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
