"""
Milestone Full Run Tests
========================

These tests run each milestone script fully to verify the complete
educational experience works end-to-end.

This is Option C: Full run (~10-15 minutes total)
- Runs each milestone with actual training
- Verifies outputs are correct (accuracy thresholds, etc.)
- Suitable for release validation, not regular CI

Usage:
    pytest tests/milestones/test_milestones_run.py -v
    pytest tests/milestones/test_milestones_run.py -v -k "milestone_01"
"""

import subprocess
import sys
import os
import re
import pytest
from pathlib import Path


# Get the tinytorch root directory
TINYTORCH_ROOT = Path(__file__).parent.parent.parent


def reported_accuracy(output: str, label: str) -> float:
    """Read one named final-results row, never a target or progress percentage."""
    # 2026-09-11: max(any percentage) let missing/poor final accuracy pass.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
    matches = re.findall(re.escape(label) + r"[ \t│|:]+(\d+(?:\.\d+)?)%", plain)
    assert len(matches) == 1, f"Expected one final {label!r} metric, found {matches}"
    value = float(matches[0])
    assert 0 <= value <= 100, f"Invalid accuracy: {value}%"
    return value


def run_milestone(milestone_id: str, part: int = None, timeout: int = 300) -> tuple[int, str, str]:
    """
    Run a milestone via tito CLI and capture output.

    Args:
        milestone_id: Milestone ID (01-06)
        part: Optional part number for multi-part milestones
        timeout: Timeout in seconds (default 5 minutes)

    Returns:
        (return_code, stdout, stderr)
    """
    # Use the bin/tito script directly
    tito_script = TINYTORCH_ROOT / "bin" / "tito"

    # Invoke bin/tito with the interpreter running the tests, not the shebang's
    # `env python3`. Otherwise the milestone runs under whatever python3 happens
    # to be first on PATH, which may not be the venv pytest is running in.
    cmd = [
        sys.executable,
        str(tito_script),
        "milestone", "run", milestone_id,
        "--skip-checks"  # Skip prerequisite checks since we're testing
    ]

    if part is not None:
        cmd.extend(["--part", str(part)])

    env = os.environ.copy()
    env["TITO_ALLOW_SYSTEM"] = "1"  # Allow running without venv
    env["PYTHONPATH"] = str(TINYTORCH_ROOT)

    # Auto-answer prompts by providing 'n' to stdin (decline syncing achievements, etc.)
    result = subprocess.run(
        cmd,
        cwd=TINYTORCH_ROOT,
        capture_output=True,
        text=True, encoding='utf-8', errors='replace',
        timeout=timeout,
        env=env,
        input="n\nn\nn\n"  # Answer 'n' to any prompts
    )

    return result.returncode, result.stdout, result.stderr


class TestMilestoneRuns:
    """Test that all milestones run successfully and produce correct output."""

    @pytest.mark.slow
    def test_milestone_01_perceptron(self):
        """Milestone 01: Perceptron (1958) - Forward pass with random weights."""
        returncode, stdout, stderr = run_milestone("01", timeout=60)

        # Should complete and show milestone achievement
        assert returncode == 0, f"Milestone 01 failed:\nstdout: {stdout}\nstderr: {stderr}"
        assert "MILESTONE ACHIEVED!" in stdout or "Milestone 01" in stdout
        assert "Model Parameters" in stdout or "Decision Line" in stdout
        assert "random" in stdout.lower()

    @pytest.mark.slow
    def test_milestone_02_xor_crisis(self):
        """Milestone 02: XOR Crisis (1969) - Demonstrates XOR problem."""
        returncode, stdout, stderr = run_milestone("02", timeout=60)

        assert returncode == 0, f"Milestone 02 failed:\nstdout: {stdout}\nstderr: {stderr}"
        assert "MILESTONE ACHIEVED!" in stdout or "Milestone 02" in stdout
        assert "CONFIRMED: XOR is UNSOLVABLE!" in stdout or "75%" in stdout

    @pytest.mark.slow
    def test_milestone_03_mlp_revival(self):
        """Milestone 03: MLP Revival (1986) - Solves XOR and trains on digits."""
        returncode, stdout, stderr = run_milestone("03", timeout=180)

        assert returncode == 0, f"Milestone 03 failed:\nstdout: {stdout}\nstderr: {stderr}"
        assert "MILESTONE ACHIEVED!" in stdout or "Milestone 03" in stdout
        assert "XOR Solved" in stdout or "100%" in stdout
        assert "TinyDigits" in stdout or "Test accuracy:" in stdout

    @pytest.mark.slow
    def test_milestone_04_cnn_tinydigits(self):
        """Milestone 04: CNN Revolution (1998) - TinyDigits (default, no download)."""
        returncode, stdout, stderr = run_milestone("04", timeout=360)

        assert returncode == 0, f"Milestone 04 failed:\nstdout: {stdout}\nstderr: {stderr}"

        # Should use TinyDigits (not CIFAR)
        assert "TinyDigits" in stdout or "tinydigits" in stdout.lower() or "8x8" in stdout

        accuracy = reported_accuracy(stdout, "Test Accuracy")
        assert accuracy >= 70, f"CNN final test accuracy too low: {accuracy}%"

    @pytest.mark.slow
    def test_milestone_05_transformer(self):
        """Milestone 05: Transformer Era (2017) - Sequence reversal with attention."""
        returncode, stdout, stderr = run_milestone("05", timeout=180)

        assert returncode == 0, f"Milestone 05 failed:\nstdout: {stdout}\nstderr: {stderr}"

        # Should mention attention/transformer
        assert "attention" in stdout.lower() or "transformer" in stdout.lower()

        accuracy = reported_accuracy(stdout, "1. Reversal")
        assert accuracy >= 95, f"Transformer final reversal accuracy too low: {accuracy}%"

    @pytest.mark.slow
    def test_milestone_06_mlperf(self):
        """Milestone 06: MLPerf Benchmarks (2018) - Optimization techniques."""
        returncode, stdout, stderr = run_milestone("06", timeout=180)

        assert returncode == 0, f"Milestone 06 failed:\nstdout: {stdout}\nstderr: {stderr}"
        assert "MILESTONE ACHIEVED!" in stdout or "Milestone 06" in stdout

        # Should mention optimization techniques
        assert any(term in stdout.lower() for term in [
            "quantiz", "compress", "cache", "kv", "speedup", "accelerat"
        ])

        # Should show compression ratio (4x for INT8)
        assert "4" in stdout and ("compress" in stdout.lower() or "×" in stdout or "x" in stdout.lower())

    @pytest.mark.slow
    def test_milestone_07_tinygpt(self):
        """Milestone 07: Generative LLM (2020) - TinyGPT on Shakespeare."""
        returncode, stdout, stderr = run_milestone("07", timeout=180)

        assert returncode == 0, f"Milestone 07 failed:\nstdout: {stdout}\nstderr: {stderr}"
        assert "MILESTONE ACHIEVED!" in stdout or "Milestone 07" in stdout
        assert "TinyGPT" in stdout or "Shakespeare" in stdout
        assert "Generated Output:" in stdout or "Sample:" in stdout
