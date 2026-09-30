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
    assert {11, 12, 13}.issubset(set(milestone["required_modules"]))
    assert {11, 12, 13, 18}.issubset(set(milestone["scripts"][1]["required_modules"]))


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
    # Run the milestone with Module 13 unavailable and check the real behavior:
    # a clean exit 1 naming the modules to export, not a raw traceback. A
    # text search of the source passed even when the guard had been removed.
    script = (
        TINYTORCH_ROOT
        / "milestones"
        / "06_2018_mlperf"
        / "02_generation_speedup.py"
    )
    runner = (
        "import runpy, sys\n"
        "sys.modules['tinytorch.core.transformers'] = None\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", runner],
        cwd=TINYTORCH_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert "modules 01-08, 11-13, and 18" in result.stdout
    assert "modules 11-17" not in result.stdout


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
    assert "1958 to 2024" in result.stdout
    assert "1957 to 2024" not in result.stdout


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


def test_milestone_05_requirements_cover_all_used_modules():
    """Verify Milestone 05 required_modules includes Trainer (08) and Tokenizer (10)."""
    milestone = MILESTONE_SCRIPTS["05"]
    required = set(milestone["required_modules"])

    assert 8 in required, "Module 08 (Trainer) must be in Milestone 05 requirements"
    assert 10 in required, "Module 10 (Tokenization) must be in Milestone 05 requirements"
    assert {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13} == required


def test_workflow_detects_all_milestones_for_module():
    """Verify Module 13 reports both Milestone 05 and Milestone 06."""
    command = ModuleWorkflowCommand(CLIConfig.from_project_root(TINYTORCH_ROOT))
    milestones = command._get_milestones_for_module(13)
    milestone_ids = [m[0] for m in milestones]

    assert "05" in milestone_ids
    assert "06" in milestone_ids


def test_open_jupyter_logs_to_file_not_pipe(monkeypatch, tmp_path):
    """Regression test: verify _open_jupyter redirects stdout/stderr to a log file instead of PIPE.

    Unconsumed subprocess.PIPE buffers fill up rapidly and deadlock the Jupyter
    Tornado event loop, causing kernel connection freezes on Windows/WSL/Linux.
    """
    command = ModuleWorkflowCommand(CLIConfig.from_project_root(tmp_path))
    module_dir = tmp_path / "modules" / "01_tensor"
    module_dir.mkdir(parents=True)
    (module_dir / "01_tensor.ipynb").touch()

    popen_calls = []

    class DummyProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        popen_calls.append(kwargs)
        return DummyProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("time.sleep", lambda _: None)

    ret = command._open_jupyter("01_tensor")
    assert ret == 0
    assert len(popen_calls) == 1
    call_kwargs = popen_calls[0]

    # Must NOT be subprocess.PIPE
    assert call_kwargs.get("stdout") != subprocess.PIPE
    assert call_kwargs.get("stderr") != subprocess.PIPE
    assert call_kwargs.get("stderr") == subprocess.STDOUT

    # stdout must be a file object pointing to .tito/jupyter.log
    stdout_file = call_kwargs.get("stdout")
    assert hasattr(stdout_file, "write") or hasattr(stdout_file, "fileno")
    assert Path(stdout_file.name) == command._jupyter_log_file()



def _run_kernels_milestone(patch_lines):
    """Run Milestone 07 with extension behavior patched in-process."""
    script = TINYTORCH_ROOT / "milestones" / "07_2024_kernels" / "01_custom_kernels.py"
    runner = "\n".join(
        ["import runpy", "import numpy as np", "import tinytorch.extensions.simd_ops as s"]
        + patch_lines
        + [f"runpy.run_path({str(script)!r}, run_name='__main__')"]
    )
    return subprocess.run(
        [sys.executable, "-c", runner],
        cwd=TINYTORCH_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_kernels_milestone_reports_numpy_fallback_honestly():
    # Every extension silently falls back to NumPy. The milestone once printed
    # [PASS] for NumPy vs NumPy and credited it as a native kernel. Milestone 07
    # now grades the student's Module 17 code (test_gates_kernels.py); a native
    # kernel that wasn't built must be reported as such, never as verified.
    result = _run_kernels_milestone(
        ["s.simd_build_info = lambda: {'built': False, 'error': 'no compiler (test)'}"]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    native_lines = [line for line in result.stdout.splitlines() if "not built" in line]
    assert native_lines, result.stdout
    assert not any("verified" in line.lower() for line in native_lines)


def test_kernels_milestone_flags_wrong_bundled_kernel():
    # A bundled native kernel that computes the wrong answer must be flagged,
    # not reported as agreeing with NumPy. It doesn't fail the student, whose
    # Module 17 code is what the milestone grades.
    result = _run_kernels_milestone(
        [
            "s.simd_build_info = lambda: {'built': True, 'openmp': False}",
            "s.simd_matmul = lambda a, b: np.zeros((a.shape[0], b.shape[1]), np.float32)",
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "disagrees with NumPy" in result.stdout


def _load_tinycopilot():
    path = TINYTORCH_ROOT / "milestones" / "05_2017_transformer" / "03_tinycopilot.py"
    spec = importlib.util.spec_from_file_location("tinycopilot_milestone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tinycopilot_scored_prompts_are_absent_from_training_corpus():
    # The syntax gate once scored prompts that appeared 6-7 times each in the
    # training text, so it measured recall. Scored prompts must stay held out.
    module = _load_tinycopilot()
    corpus = (TINYTORCH_ROOT / "datasets" / "tinypy" / "tinypy_sample.txt").read_text(encoding="utf-8")
    leaked = [p for p in module.HELDOUT_PROMPTS if p in corpus]
    assert module.HELDOUT_PROMPTS, "TinyCopilot needs held-out prompts to score"
    assert not leaked, f"Scored prompts found in training corpus: {leaked}"


def test_tinycopilot_ast_gate_rejects_garbage():
    # The gate once trimmed lines and appended `pass` until anything parsed,
    # so these all counted as valid Python.
    check = _load_tinycopilot().check_ast_validity
    garbage = [
        ("def add(a, b):", "\n)))) !!! garbage $$$"),
        ("class Linear:", "\n\x00@@@"),
        ("def f(x):", "\n    for i in in in ====="),
        ("def f(x):", "\n    return x\n    y = = 1\n    z = = 2"),
        ("def f(x):", "\n    y = = 1"),
        ("def f(x):", "\n    return x\n    y = =\n\n"),
        ("def f(x):", ""),
    ]
    for prompt, completion in garbage:
        is_valid, _ = check(prompt + completion, prompt)
        assert not is_valid, f"AST gate accepted invalid completion: {completion!r}"

    # Honest completions still pass, including one cut off mid-line by the budget
    assert check("def f(x):\n    return x\n\n", "def f(x):")[0]
    assert check("def f(x):\n    y = x + 1\n    return y\n    y = =", "def f(x):")[0]


def test_chat_overfitting_diagnosis_follows_measured_test_loss():
    # Labels were once assigned by epoch number: the midpoint epoch was always
    # "Sweet Spot" even when held-out loss rose every epoch.
    path = TINYTORCH_ROOT / "milestones" / "05_2017_transformer" / "04_tinygpt_chat.py"
    spec = importlib.util.spec_from_file_location("tinygpt_chat_milestone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rising = [(1, 5.4, 5.99, 0.5), (2, 3.8, 6.35, 2.5), (3, 2.0, 7.69, 5.7), (4, 0.7, 9.10, 8.4)]
    labels = [diag for diag, _ in module.diagnose_epochs(rising)]
    assert labels[0].startswith("Sweet Spot")
    assert all(label.startswith("Overfitting") for label in labels[1:])

    u_shaped = [(1, 5.0, 6.0, 1.0), (2, 3.0, 4.0, 1.0), (3, 1.0, 5.0, 4.0)]
    labels = [diag for diag, _ in module.diagnose_epochs(u_shaped)]
    assert labels[0].startswith("Learning")
    assert labels[1].startswith("Sweet Spot")
    assert labels[2].startswith("Overfitting")
