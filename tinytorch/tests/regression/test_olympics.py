"""Tests for TinyTorch Olympics and Capstone Benchmarking.

Verifies:
1. SimpleMLP architecture, forward pass, and parameter counting
2. BenchmarkReport profiling: latency, throughput, accuracy, and storage
3. OlympicEvent qualification rules across all events
4. Submission schema generation and automated validation
5. tito olympics CLI commands
"""

import os
import sys
import json
import pytest
import subprocess
import numpy as np
from pathlib import Path

from tinytorch.core.tensor import Tensor
from tinytorch.olympics import (
    SimpleMLP,
    BenchmarkReport,
    OlympicEvent,
    qualifies_event,
    generate_submission,
    save_submission,
    validate_submission_schema,
)


class TestTinyTorchOlympics:
    """Validate Olympic benchmarking and event evaluation."""

    def test_simple_mlp_forward_and_params(self):
        mlp = SimpleMLP(input_size=8, hidden_size=16, output_size=4)
        x = Tensor(np.ones((2, 8), dtype=np.float32))
        out = mlp.forward(x)
        assert out.shape == (2, 4)
        # Parameters: fc1 weights (8*16) + bias (16) + fc2 weights (16*4) + bias (4)
        # = 128 + 16 + 64 + 4 = 212
        assert mlp.count_parameters() == 212

    def test_benchmark_report_metrics(self):
        mlp = SimpleMLP(input_size=4, hidden_size=8, output_size=2)
        X_test = Tensor(np.random.default_rng(42).standard_normal((10, 4)).astype(np.float32))
        y_test = np.random.default_rng(42).integers(0, 2, size=10)

        report = BenchmarkReport(model_name="test_mlp")
        metrics = report.benchmark_model(mlp, X_test, y_test, num_runs=5)

        assert "parameter_count" in metrics
        assert "model_size_mb" in metrics
        assert "accuracy" in metrics
        assert "latency_ms_mean" in metrics
        assert "latency_ms_median" in metrics
        assert "throughput_samples_per_sec" in metrics

        assert metrics["parameter_count"] > 0
        assert metrics["model_size_mb"] > 0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["latency_ms_median"] > 0
        assert metrics["throughput_samples_per_sec"] > 0

    def test_olympic_event_qualification(self):
        good_metrics = {
            "accuracy": 0.92,
            "latency_ms_median": 1.5,
            "model_size_mb": 0.05,
        }
        # Meets 85% accuracy floor
        assert qualifies_event(good_metrics, OlympicEvent.LATENCY_SPRINT)
        assert qualifies_event(good_metrics, OlympicEvent.MEMORY_CHALLENGE)
        assert qualifies_event(good_metrics, OlympicEvent.ACCURACY_CONTEST)
        assert qualifies_event(good_metrics, OlympicEvent.EXTREME_PUSH)
        assert qualifies_event(good_metrics, OlympicEvent.ALL_AROUND)

        low_acc_metrics = {
            "accuracy": 0.82,
            "latency_ms_median": 0.5,
            "model_size_mb": 0.01,
        }
        # Fails 85% floor for sprint & memory, passes 80% for extreme_push, passes all_around
        assert not qualifies_event(low_acc_metrics, OlympicEvent.LATENCY_SPRINT)
        assert not qualifies_event(low_acc_metrics, OlympicEvent.MEMORY_CHALLENGE)
        assert qualifies_event(low_acc_metrics, OlympicEvent.EXTREME_PUSH)
        assert qualifies_event(low_acc_metrics, OlympicEvent.ALL_AROUND)

        # Invalid metrics
        with pytest.raises(ValueError):
            qualifies_event({"accuracy": -0.1, "latency_ms_median": 1.0, "model_size_mb": 1.0}, OlympicEvent.ALL_AROUND)
        with pytest.raises(ValueError):
            qualifies_event({"accuracy": 0.9, "latency_ms_median": -1.0, "model_size_mb": 1.0}, OlympicEvent.ALL_AROUND)

    def test_generate_and_validate_submission(self, tmp_path):
        mlp = SimpleMLP(input_size=4, hidden_size=8, output_size=2)
        X_test = Tensor(np.random.default_rng(1).standard_normal((10, 4)).astype(np.float32))
        y_test = np.random.default_rng(1).integers(0, 2, size=10)

        report = BenchmarkReport(model_name="baseline_mlp")
        report.benchmark_model(mlp, X_test, y_test, num_runs=5)

        submission = generate_submission(
            baseline_report=report,
            student_name="Ada Lovelace",
        )
        assert validate_submission_schema(submission)

        # Test save and reload
        filepath = str(tmp_path / "submission.json")
        save_submission(submission, filepath)
        with open(filepath, encoding='utf-8') as f:
            loaded = json.load(f)
        assert validate_submission_schema(loaded)
        assert loaded["student_name"] == "Ada Lovelace"

    def test_olympics_cli_commands(self):
        repo_root = Path(__file__).parent.parent.parent
        tito_bin = repo_root / "bin" / "tito"
        env = dict(os.environ)
        env["TITO_ALLOW_SYSTEM"] = "1"

        # Test logo subcommand
        res_logo = subprocess.run(
            [sys.executable, str(tito_bin), "olympics", "logo"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert res_logo.returncode == 0
        assert "TINYTORCH OLYMPICS" in res_logo.stdout

        # Test status subcommand
        res_status = subprocess.run(
            [sys.executable, str(tito_bin), "olympics", "status"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert res_status.returncode == 0
