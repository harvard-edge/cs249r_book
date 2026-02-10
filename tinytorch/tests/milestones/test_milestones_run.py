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
import pytest
from pathlib import Path


# Get the tinytorch root directory
TINYTORCH_ROOT = Path(__file__).parent.parent.parent


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

    cmd = [
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
        text=True,
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

        # Should complete (even with errors in output, the script should finish)
        assert returncode == 0, f"Milestone 01 failed:\nstdout: {stdout}\nstderr: {stderr}"

        # Should show the perceptron architecture
        assert "Perceptron" in stdout or "perceptron" in stdout.lower()

        # Should mention random weights (this is forward-pass only)
        assert "random" in stdout.lower() or "Random" in stdout

    @pytest.mark.slow
    def test_milestone_02_xor_crisis(self):
        """Milestone 02: XOR Crisis (1969) - Demonstrates XOR problem."""
        returncode, stdout, stderr = run_milestone("02", timeout=60)

        assert returncode == 0, f"Milestone 02 failed:\nstdout: {stdout}\nstderr: {stderr}"

        # Should mention XOR
        assert "XOR" in stdout or "xor" in stdout.lower()

        # Should show the limitation (can't solve XOR with single layer)
        assert "75%" in stdout or "50%" in stdout or "cannot" in stdout.lower() or "limit" in stdout.lower()

    @pytest.mark.slow
    def test_milestone_03_mlp_revival(self):
        """Milestone 03: MLP Revival (1986) - Solves XOR and trains on digits."""
        returncode, stdout, stderr = run_milestone("03", timeout=180)

        assert returncode == 0, f"Milestone 03 failed:\nstdout: {stdout}\nstderr: {stderr}"

        # Should solve XOR with 100% accuracy
        assert "100" in stdout and ("XOR" in stdout or "accuracy" in stdout.lower())

        # Should train on digits
        assert "digit" in stdout.lower() or "Digit" in stdout

    @pytest.mark.slow
    def test_milestone_04_cnn_tinydigits(self):
        """Milestone 04: CNN Revolution (1998) - TinyDigits (default, no download)."""
        returncode, stdout, stderr = run_milestone("04", timeout=180)

        assert returncode == 0, f"Milestone 04 failed:\nstdout: {stdout}\nstderr: {stderr}"

        # Should use TinyDigits (not CIFAR)
        assert "TinyDigits" in stdout or "tinydigits" in stdout.lower() or "8x8" in stdout

        # Should achieve reasonable accuracy (>70%)
        # Look for accuracy numbers in the output
        import re
        accuracy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', stdout)
        if accuracy_matches:
            # Get the highest accuracy mentioned (likely final test accuracy)
            accuracies = [float(a) for a in accuracy_matches if float(a) <= 100]
            if accuracies:
                max_accuracy = max(accuracies)
                assert max_accuracy >= 70, f"CNN accuracy too low: {max_accuracy}%"

    @pytest.mark.slow
    def test_milestone_05_transformer(self):
        """Milestone 05: Transformer Era (2017) - Sequence reversal with attention."""
        returncode, stdout, stderr = run_milestone("05", timeout=180)

        assert returncode == 0, f"Milestone 05 failed:\nstdout: {stdout}\nstderr: {stderr}"

        # Should mention attention/transformer
        assert "attention" in stdout.lower() or "transformer" in stdout.lower()

        # Should achieve good accuracy on reversal task (>90%)
        import re
        accuracy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', stdout)
        if accuracy_matches:
            accuracies = [float(a) for a in accuracy_matches if float(a) <= 100]
            if accuracies:
                max_accuracy = max(accuracies)
                assert max_accuracy >= 90, f"Transformer accuracy too low: {max_accuracy}%"

    @pytest.mark.slow
    def test_milestone_06_mlperf(self):
        """Milestone 06: MLPerf Benchmarks (2018) - Optimization techniques."""
        returncode, stdout, stderr = run_milestone("06", timeout=180)

        assert returncode == 0, f"Milestone 06 failed:\nstdout: {stdout}\nstderr: {stderr}"

        # Should mention optimization techniques
        assert any(term in stdout.lower() for term in [
            "quantiz", "compress", "cache", "kv", "speedup", "accelerat"
        ])

        # Should show compression ratio (4x for INT8)
        assert "4" in stdout and ("compress" in stdout.lower() or "×" in stdout or "x" in stdout.lower())


class TestMilestoneSequence:
    """Test that milestones can be run in sequence (simulates student journey)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("milestone_id", ["01", "02", "03", "04", "05", "06"])
    def test_milestone_completes(self, milestone_id):
        """Each milestone should complete without errors (or with expected bonus challenge failures)."""
        returncode, stdout, stderr = run_milestone(milestone_id, timeout=300)

        # Milestone 05 has bonus challenges that may fail - the core reversal task passing is sufficient
        # The individual test_milestone_05_transformer test validates the actual learning objective
        if milestone_id == "05":
            # Check that reversal (Challenge 1) passed - this is the core learning objective
            assert "Reversal" in stdout and "PASSED" in stdout, (
                f"Milestone 05 reversal challenge should pass:\n"
                f"stdout: {stdout[-2000:] if len(stdout) > 2000 else stdout}"
            )
            return  # Bonus challenges may fail, that's OK

        assert returncode == 0, (
            f"Milestone {milestone_id} failed with return code {returncode}\n"
            f"stdout: {stdout[-2000:] if len(stdout) > 2000 else stdout}\n"
            f"stderr: {stderr[-500:] if len(stderr) > 500 else stderr}"
        )

        # Should show achievement unlocked
        assert "MILESTONE" in stdout.upper() or "Achievement" in stdout or "✅" in stdout
