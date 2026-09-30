"""
Honest-reporting regressions for milestones and benchmark tooling.

Each test pins a claim that was once hardcoded or a gate that once passed
unconditionally (audit of 2026-09-28). Printed claims must be computed from
measured values, and pass/fail must follow a real check.
"""

import importlib.util
import inspect
import io
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

TINYTORCH_ROOT = Path(__file__).resolve().parents[2]
MILESTONES = TINYTORCH_ROOT / "milestones"


def _load(relpath: str, name: str):
    path = MILESTONES / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quiet_console():
    return Console(file=io.StringIO(), width=200, force_terminal=False)


# ---------------------------------------------------------------------------
# Milestone 06: Optimization Olympics
# ---------------------------------------------------------------------------

def _olympics():
    return _load("06_2018_mlperf/01_optimization_olympics.py", "olympics_honest")


def test_olympics_pareto_label_comes_from_computed_frontier():
    # INT8 and Full Stack were labeled "Pareto (Peak/Optimal)" by name,
    # whatever pareto_frontier() returned.
    module = _olympics()
    for name in ("INT8 Quantized", "Full Stack (Quant+Cache)"):
        assert "Dominated" in module.pareto_status(name, {"Baseline FP32"})
        assert "Pareto" in module.pareto_status(name, {name})
    source = (MILESTONES / "06_2018_mlperf" / "01_optimization_olympics.py").read_text()
    assert "Pareto (Peak)" not in source and "Pareto (Optimal)" not in source


def _synthesis(module, full_stack_on_frontier):
    # Values measured on 2026-09-28: 3.95x MLP codes, 3.73x CNN, pruned 95.05%.
    return "\n".join(module.synthesis_lines(
        mlp={"base_bytes": 9640, "quant_bytes": 2442, "base_acc": 86.5,
             "quant_acc": 86.0, "quant_on_frontier": True},
        cnn={"base_bytes": 2664, "quant_bytes": 714, "agree_quant": 99.99,
             "agree_pruned": 95.05, "frontier": ["Baseline FP32", "INT8 Quantized"]},
        gpt={"cache_max_abs_diff": 1.3e-6, "full_stack_on_frontier": full_stack_on_frontier,
             "frontier": ["Baseline FP32 (Recompute)"], "n_candidates": 4},
    ))


def test_olympics_synthesis_formats_measured_values():
    module = _olympics()
    text = _synthesis(module, full_stack_on_frontier=True)
    assert "3.95×" in text and "3.73×" in text
    assert "4×" not in text
    assert "100%" not in text and "100% signal fidelity" not in text
    assert "86.5% → 86.0%" in text and "-0.5 points" in text
    assert "95.05%" in text
    # CNN and GPT are untrained: quality is agreement, never accuracy.
    assert "not accuracy" in text
    assert "Pareto optimum" not in text


def test_olympics_synthesis_never_claims_frontier_it_did_not_compute():
    text = _synthesis(_olympics(), full_stack_on_frontier=False)
    assert "Quant+Cache is dominated" in text
    assert "on the measured latency/memory/agreement frontier" not in text


def test_olympics_cache_exactness_is_measured():
    module = _olympics()
    base = {"base_bytes": 4, "quant_bytes": 1, "base_acc": 1.0, "quant_acc": 1.0,
            "quant_on_frontier": True}
    cnn = {"base_bytes": 4, "quant_bytes": 1, "agree_quant": 1.0, "agree_pruned": 1.0,
           "frontier": []}
    gpt = {"cache_max_abs_diff": module.max_abs_diff(np.zeros(3), np.array([0, 0, 0.5])),
           "full_stack_on_frontier": False, "frontier": [], "n_candidates": 4}
    text = "\n".join(module.synthesis_lines(base, cnn, gpt))
    assert "does NOT match" in text


# ---------------------------------------------------------------------------
# tito benchmark
# ---------------------------------------------------------------------------

def _benchmark_command():
    from tito.commands.benchmark import BenchmarkCommand
    from tito.core.config import CLIConfig
    command = BenchmarkCommand(CLIConfig.from_project_root(TINYTORCH_ROOT))
    command.console = _quiet_console()
    return command


def test_benchmark_geometric_mean_is_a_geometric_mean():
    from tito.commands import benchmark
    # The old code summed times while calling it a geometric mean.
    assert benchmark.geometric_mean([1.0, 4.0, 16.0]) == pytest.approx(4.0)
    summary = benchmark.summarize_times({"a": 1.0, "b": 4.0, "c": 16.0})
    assert summary["geometric_mean_ms"] == pytest.approx(4.0)
    assert "score" not in summary


def test_benchmark_capstone_without_module_20_fails_and_saves_nothing(monkeypatch, tmp_path):
    # It used to sleep(1) and save {"basic_score": 75}.
    import builtins
    real_import = builtins.__import__

    def no_olympics(name, *args, **kwargs):
        if name == "tinytorch.olympics":
            raise ImportError("Module 20 not exported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_olympics)
    monkeypatch.chdir(tmp_path)
    command = _benchmark_command()
    assert command.run(Namespace(benchmark_command="capstone", track="all", skip_submit=False)) == 1
    assert not (tmp_path / ".tito").exists()
    assert "basic_score" not in command.console.file.getvalue()


def test_benchmark_capstone_never_prints_placeholder_scores(monkeypatch, tmp_path):
    import types
    monkeypatch.setitem(sys.modules, "tinytorch.olympics",
                        types.SimpleNamespace(generate_submission=lambda *a, **k: None))
    monkeypatch.chdir(tmp_path)
    command = _benchmark_command()
    assert command.run(Namespace(benchmark_command="capstone", track="all", skip_submit=False)) == 1
    output = command.console.file.getvalue()
    assert "90/100" not in output and "87.5" not in output
    assert not (tmp_path / ".tito").exists()


def test_benchmark_baseline_is_local_only_and_reports_real_cpu_count(monkeypatch, tmp_path):
    import rich.prompt

    def no_prompt(*args, **kwargs):
        raise AssertionError("baseline must not prompt to submit; there is no upload")

    monkeypatch.setattr(rich.prompt.Confirm, "ask", no_prompt)
    monkeypatch.chdir(tmp_path)
    command = _benchmark_command()
    assert command.run(Namespace(benchmark_command="baseline", skip_submit=False)) == 0
    output = command.console.file.getvalue()
    assert "/100" not in output
    assert "stored locally only" in output
    assert "not your TinyTorch" in output
    saved = list((tmp_path / ".tito" / "benchmarks").glob("baseline_*.json"))
    assert len(saved) == 1
    result = json.loads(saved[0].read_text())
    assert result["system_info"]["cpu_count"] == str(os.cpu_count())
    assert result["metrics"]["geometric_mean_ms"] > 0


# ---------------------------------------------------------------------------
# Milestones 03 and 04: accuracy gates
# ---------------------------------------------------------------------------

def test_mlp_milestone_fails_below_accuracy_target(monkeypatch):
    # It printed "Success!" and exited 0 at any accuracy.
    module = _load("03_1986_mlp/01_rumelhart_tinydigits.py", "mlp_honest")
    monkeypatch.setenv("TINYTORCH_NON_INTERACTIVE", "1")
    monkeypatch.setattr(module, "console", _quiet_console())
    monkeypatch.setattr(module, "evaluate_accuracy",
                        lambda model, images, labels: (10.0, np.zeros(len(labels.data), dtype=int)))
    assert module.train_mlp() == 1
    output = module.console.file.getvalue()
    assert "FAILED" in output and "Success" not in output


def test_mlp_milestone_passes_with_real_training(monkeypatch):
    # The gate must not fail a correct implementation (measured ~82%).
    module = _load("03_1986_mlp/01_rumelhart_tinydigits.py", "mlp_honest_real")
    monkeypatch.setenv("TINYTORCH_NON_INTERACTIVE", "1")
    monkeypatch.setattr(module, "console", _quiet_console())
    assert module.train_mlp() == 0


def test_cnn_milestone_fails_below_accuracy_target(monkeypatch):
    module = _load("04_1998_cnn/01_lecun_tinydigits.py", "cnn_honest")
    monkeypatch.setenv("TINYTORCH_NON_INTERACTIVE", "1")
    monkeypatch.setattr(module, "console", _quiet_console())
    monkeypatch.setattr(module, "train_epoch", lambda *args, **kwargs: 2.3)
    monkeypatch.setattr(module, "evaluate_accuracy", lambda model, images, labels: (10.0, 2.3))
    assert module.train_cnn() == 1
    output = module.console.file.getvalue()
    assert "FAILED" in output and "Success" not in output


# ---------------------------------------------------------------------------
# Milestone 05: TinyGPT and TinyCopilot
# ---------------------------------------------------------------------------

def test_tinygpt_success_panel_reports_measured_loss_not_coherence():
    module = _load("05_2017_transformer/01_tinygpt_shakespeare.py", "tinygpt_honest")
    text = "\n".join(module.convergence_report(2.6376, 0.6435, 59))
    assert "0.6435" in text and "1.90" in text and "59" in text
    assert "coherent autoregressive" not in text.lower()
    source = (MILESTONES / "05_2017_transformer" / "01_tinygpt_shakespeare.py").read_text()
    assert "Model Status : Syntactically coherent" not in source


def test_tinycopilot_quick_flag_changes_training_budget_not_corpus():
    # --quick claimed a "sample dataset", but the bundled sample always loaded.
    module = _load("05_2017_transformer/03_tinycopilot.py", "tinycopilot_honest")
    assert "sample_only" not in inspect.signature(module.load_tinypy_corpus).parameters
    corpus = module.load_tinypy_corpus()
    assert module.QUICK_TOKEN_LIMIT < module.FULL_TOKEN_LIMIT <= len(corpus)


# ---------------------------------------------------------------------------
# Milestone 04 Part 2: CIFAR-10 without the dataset
# ---------------------------------------------------------------------------

def test_cifar_quick_test_without_dataset_exits_with_documented_code(monkeypatch, capsys):
    # A non-interactive --quick-test died with a traceback (exit 1).
    module = _load("04_1998_cnn/02_lecun_cifar10.py", "cifar_honest")

    class NoDownload:
        def get_cifar10(self):
            raise RuntimeError("CIFAR-10 download canceled by user")

    monkeypatch.setattr(module, "DatasetManager", NoDownload)
    monkeypatch.setattr(module, "visualize_cifar_cnn", lambda: None)
    monkeypatch.setattr(sys, "argv", ["02_lecun_cifar10.py", "--quick-test"])
    assert module.main() == module.EXIT_DATASET_UNAVAILABLE == 2
    output = capsys.readouterr().out
    assert "--test-only" in output and "TINYTORCH_AUTO_DOWNLOAD=1" in output
