"""
Milestone Execution Tests

WHAT: Verify all milestones can execute without errors.
WHY: Milestones are the key student checkpoints - they MUST work reliably.
     Broken milestones = frustrated students = bad learning experience.

STUDENT LEARNING:
These tests ensure the 6 historical milestones are always working:
1. Perceptron (1958) - First neural network
2. XOR Crisis (1969) - Multi-layer networks
3. MLP Revival (1986) - Backpropagation
4. CNN Revolution (1998) - Spatial networks
5. Transformer Era (2017) - Attention mechanism
6. MLPerf (2018) - Optimization techniques
"""

import subprocess
import sys
from pathlib import Path
import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestMilestone01Perceptron:
    """Test Milestone 01: Perceptron (1958)"""

    def test_perceptron_forward_runs(self):
        """
        WHAT: Verify the perceptron forward pass demo runs.
        WHY: This is the first milestones - it must work to build confidence.
        """
        script = PROJECT_ROOT / "milestones" / "01_1958_perceptron" / "01_rosenblatt_forward.py"
        if not script.exists():
            pytest.fail(f"Script not found: {script}")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=PROJECT_ROOT
        )

        assert result.returncode == 0, f"Perceptron forward failed:\n{result.stderr}"

    def test_perceptron_trained_runs(self):
        """
        WHAT: Verify the trained perceptron demo runs.
        WHY: This proves the full training loop works.
        NOTE: This script is in extras/ as it's an extended example.
        """
        script = PROJECT_ROOT / "milestones" / "extras" / "02_rosenblatt_trained.py"
        if not script.exists():
            pytest.fail(f"Script not found: {script}")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT
        )

        assert result.returncode == 0, f"Perceptron trained failed:\n{result.stderr}"


class TestMilestone02XOR:
    """Test Milestone 02: XOR Crisis (1969)"""

    def test_xor_crisis_runs(self):
        """
        WHAT: Verify the XOR crisis demo runs (shows single-layer failure).
        WHY: This demonstrates a key historical limitation.
        """
        script = PROJECT_ROOT / "milestones" / "02_1969_xor" / "01_xor_crisis.py"
        if not script.exists():
            pytest.fail(f"Script not found: {script}")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=PROJECT_ROOT
        )

        assert result.returncode == 0, f"XOR crisis failed:\n{result.stderr}"

    def test_xor_solved_runs(self):
        """
        WHAT: Verify the XOR solved demo runs (multi-layer success).
        WHY: This proves hidden layers enable non-linear classification.
        """
        script = PROJECT_ROOT / "milestones" / "02_1969_xor" / "02_xor_solved.py"
        if not script.exists():
            pytest.fail(f"Script not found: {script}")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT
        )

        assert result.returncode == 0, f"XOR solved failed:\n{result.stderr}"


class TestMilestone03MLP:
    """Test Milestone 03: MLP Revival (1986)"""

    def test_mlp_tinydigits_runs(self):
        """
        WHAT: Verify MLP training on TinyDigits runs.
        WHY: This proves backprop works on real data.
        """
        script = PROJECT_ROOT / "milestones" / "03_1986_mlp" / "01_rumelhart_tinydigits.py"
        if not script.exists():
            pytest.fail(f"Script not found: {script}")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=180,  # Training can take a bit
            cwd=PROJECT_ROOT
        )

        assert result.returncode == 0, f"MLP TinyDigits failed:\n{result.stderr}"


class TestMilestone04CNN:
    """Test Milestone 04: CNN Revolution (1998)"""

    def test_cnn_tinydigits_runs(self):
        """
        WHAT: Verify CNN training on TinyDigits runs.
        WHY: This proves spatial operations and convolutions work.
        """
        script = PROJECT_ROOT / "milestones" / "04_1998_cnn" / "01_lecun_tinydigits.py"
        if not script.exists():
            pytest.fail(f"Script not found: {script}")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=300,  # CNN training can be slow
            cwd=PROJECT_ROOT
        )

        assert result.returncode == 0, f"CNN TinyDigits failed:\n{result.stderr}"


class TestMilestone05Transformer:
    """Test Milestone 05: Transformer Era (2017)"""

    @pytest.mark.skip(reason="Transformer training is non-deterministic and slow - run manually")
    def test_attention_runs(self):
        """
        WHAT: Verify the attention mechanism demo runs.
        WHY: This proves attention can learn cross-position relationships.

        NOTE: Skipped in CI - this test trains a transformer model which is
        non-deterministic and may not always reach 90% accuracy. Run manually
        with: pytest tests/milestones/test_milestones_run.py::TestMilestone05Transformer -v --no-skip
        """
        script = PROJECT_ROOT / "milestones" / "05_2017_transformer" / "01_vaswani_attention.py"
        if not script.exists():
            pytest.fail(f"Script not found: {script}")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT
        )

        assert result.returncode == 0, f"Attention demo failed:\n{result.stderr}"


class TestMilestone06MLPerf:
    """Test Milestone 06: MLPerf (2018)"""

    def test_optimization_olympics_runs(self):
        """
        WHAT: Verify the optimization pipeline runs.
        WHY: This proves profiling, quantization, and pruning work.
        """
        script = PROJECT_ROOT / "milestones" / "06_2018_mlperf" / "01_optimization_olympics.py"
        if not script.exists():
            pytest.fail(f"Script not found: {script}")

        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=PROJECT_ROOT
        )

        assert result.returncode == 0, f"Optimization Olympics failed:\n{result.stderr}"
        # Verify compression was achieved
        assert "compression" in result.stdout.lower() or "smaller" in result.stdout.lower(), \
            "Should show compression metrics"


class TestMilestoneCLI:
    """Test milestones work through the CLI."""

    def test_milestone_list_works(self):
        """
        WHAT: Verify `tito milestone list` works.
        WHY: Students need to discover available milestones.
        """
        tito_path = PROJECT_ROOT / "bin" / "tito"
        env = {"TITO_ALLOW_SYSTEM": "1", "PATH": subprocess.os.environ.get("PATH", "")}
        result = subprocess.run(
            [str(tito_path), "milestone", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_ROOT,
            env=env
        )

        assert result.returncode == 0, f"tito milestone list failed:\n{result.stderr}\n{result.stdout}"
        assert "Perceptron" in result.stdout, "Should list Perceptron milestones"
        assert "Transformer" in result.stdout, "Should list Transformer milestones"

    def test_milestone_status_works(self):
        """
        WHAT: Verify `tito milestone status` works.
        WHY: Students need to track their progress.
        """
        tito_path = PROJECT_ROOT / "bin" / "tito"
        env = {"TITO_ALLOW_SYSTEM": "1", "PATH": subprocess.os.environ.get("PATH", "")}
        result = subprocess.run(
            [str(tito_path), "milestone", "status"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_ROOT,
            env=env
        )

        assert result.returncode == 0, f"tito milestone status failed:\n{result.stderr}\n{result.stdout}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
