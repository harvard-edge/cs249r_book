"""
Milestone Full Run Tests
========================

These tests run each milestone script fully to verify the complete
educational experience works end-to-end.

This is Option C: Full run (~10-15 minutes total)
- Runs each milestone with actual training
- Verifies outputs are correct (accuracy thresholds, etc.)
- Suitable for release validation, not regular CI

Isolation (2026-09-29): tito keeps student state in ``.tito/`` under the
directory it runs from, and these runs used to write the checkout's real
``.tito/milestones.json``. Each run now happens in a temporary project whose
entries are symlinks to this checkout, except for a private ``.tito/``.

Assertions check the return code and lines only the milestone SCRIPT prints
on a real pass. They never accept tito's own banner text ("Milestone 0N",
descriptions such as "75% max"), which prints whether or not the script
succeeded.

Usage:
    pytest tests/milestones/test_milestones_run.py -v
    pytest tests/milestones/test_milestones_run.py -v -k "milestone_01"
"""

import json
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


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    """A temp project mirroring this checkout, with its own empty .tito/."""
    root = tmp_path_factory.mktemp("tito_project")
    for entry in TINYTORCH_ROOT.iterdir():
        if entry.name == ".tito":
            continue
        (root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (root / ".tito").mkdir()
    return root


def run_milestone(project: Path, milestone_id: str, part: int = None,
                  timeout: int = 300) -> tuple[int, str, str]:
    """
    Run a milestone via the tito CLI inside the temp project and capture output.

    Uses --skip-checks (no prerequisite checks), which runs the scripts as a
    demo and records nothing; the test then confirms the temp ledger holds
    no completion.

    Returns:
        (return_code, stdout, stderr)
    """
    # Invoke tito with the interpreter running the tests, not bin/tito: that
    # wrapper chdirs into the checkout, which would put .tito/ back in the repo.
    cmd = [
        sys.executable, "-c",
        "import sys; from tito.main import main; sys.exit(main())",
        "milestone", "run", milestone_id,
        "--skip-checks", "--non-interactive",
    ]

    if part is not None:
        cmd.extend(["--part", str(part)])

    env = os.environ.copy()
    env["TITO_ALLOW_SYSTEM"] = "1"  # Allow running without venv
    env["PYTHONPATH"] = str(project)
    env["TITO_NO_SYNC"] = "1"

    result = subprocess.run(
        cmd,
        cwd=project,
        capture_output=True,
        text=True, encoding='utf-8', errors='replace',
        timeout=timeout,
        env=env,
        input="n\nn\nn\n"  # Answer 'n' to any prompts
    )

    ledger = project / ".tito" / "milestones.json"
    if ledger.exists():
        data = json.loads(ledger.read_text(encoding="utf-8"))
        assert not data.get("completed_milestones"), "--skip-checks must not record completion"
        assert not data.get("part_results"), "--skip-checks must not record part results"

    return result.returncode, result.stdout, result.stderr


def _plain(text: str) -> str:
    """Strip ANSI and undo Rich panel wrapping so phrases can be matched."""
    plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"[│\s]+", " ", plain)


class TestMilestoneRuns:
    """Each milestone script runs and prints its own pass-only output."""

    @pytest.mark.slow
    def test_milestone_01_perceptron(self, project):
        """Milestone 01: Perceptron (1958) - Forward pass with random weights."""
        returncode, stdout, stderr = run_milestone(project, "01", timeout=60)

        assert returncode == 0, f"Milestone 01 failed:\nstdout: {stdout}\nstderr: {stderr}"
        flowed = _plain(stdout)
        # Panel titles and the closing experiment note are script output only.
        assert "Model Parameters" in flowed
        assert "Predictions with Decision Boundary" in flowed
        assert "Run this script multiple times!" in flowed

    @pytest.mark.slow
    def test_milestone_02_xor_crisis(self, project):
        """Milestone 02: XOR Crisis (1969) - Demonstrates XOR problem."""
        returncode, stdout, stderr = run_milestone(project, "02", timeout=60)

        assert returncode == 0, f"Milestone 02 failed:\nstdout: {stdout}\nstderr: {stderr}"
        # 2026-09-29: this once accepted "75%", which tito's description prints.
        flowed = _plain(stdout)
        assert "CONFIRMED: XOR is UNSOLVABLE!" in flowed
        assert "No matter how you draw a single line, at least one point is wrong." in flowed

    @pytest.mark.slow
    def test_milestone_03_mlp_revival(self, project):
        """Milestone 03 Part 1: hidden layers solve XOR."""
        # 2026-09-28: this once accepted OR-ed banner strings that always print.
        returncode, stdout, stderr = run_milestone(project, "03", part=1, timeout=180)

        assert returncode == 0, f"Milestone 03 failed:\nstdout: {stdout}\nstderr: {stderr}"
        accuracy = reported_accuracy(stdout, "Final accuracy")
        assert accuracy == 100.0, f"XOR not solved: final accuracy {accuracy}%"

    @pytest.mark.slow
    def test_milestone_03_tinydigits(self, project):
        """Milestone 03 Part 2: MLP on TinyDigits must meet its accuracy gate."""
        returncode, stdout, stderr = run_milestone(project, "03", part=2, timeout=180)

        assert returncode == 0, f"Milestone 03 Part 2 failed:\nstdout: {stdout}\nstderr: {stderr}"
        # The script exits 1 below 75%; measured 82.0-82.5% on 2026-09-28.
        accuracy = reported_accuracy(stdout, "Test Accuracy")
        assert accuracy >= 75, f"MLP final test accuracy too low: {accuracy}%"

    @pytest.mark.slow
    def test_milestone_04_cnn_tinydigits(self, project):
        """Milestone 04: CNN Revolution (1998) - TinyDigits (the required part)."""
        returncode, stdout, stderr = run_milestone(project, "04", timeout=360)

        assert returncode == 0, f"Milestone 04 failed:\nstdout: {stdout}\nstderr: {stderr}"
        # The script exits 1 below 75%; measured 85.0-86.0% on 2026-09-28.
        accuracy = reported_accuracy(stdout, "Test Accuracy")
        assert accuracy >= 75, f"CNN final test accuracy too low: {accuracy}%"

    @pytest.mark.slow
    def test_milestone_05_transformer(self, project):
        """Milestone 05 Part 1: TinyGPT on Shakespeare."""
        returncode, stdout, stderr = run_milestone(project, "05", part=1, timeout=180)

        assert returncode == 0, f"Milestone 05 failed:\nstdout: {stdout}\nstderr: {stderr}"
        flowed = _plain(stdout)
        # "TinyGPT"/"Shakespeare" are in tito's banner; these are script-only.
        assert "Generated Output:" in flowed
        assert "Success!" in flowed

    @pytest.mark.slow
    def test_milestone_05_tinycopilot(self, project):
        """Milestone 05 Part 3: TinyCopilot on TinyPy Python code."""
        returncode, stdout, stderr = run_milestone(project, "05", part=3, timeout=180)

        assert returncode == 0, f"Milestone 05 Part 3 failed:\nstdout: {stdout}\nstderr: {stderr}"
        flowed = _plain(stdout)
        assert "Syntactic Validity on Unseen Function Names" in flowed
        assert "Success!" in flowed

    @pytest.mark.slow
    def test_milestone_05_conversational_chat(self, project):
        """Milestone 05 Part 4: Conversational Q&A & Overfitting Detective."""
        returncode, stdout, stderr = run_milestone(project, "05", part=4, timeout=180)

        assert returncode == 0, f"Milestone 05 Part 4 failed:\nstdout: {stdout}\nstderr: {stderr}"
        flowed = _plain(stdout)
        assert "The Overfitting Detective: Train vs. Test Generalization Gap" in flowed
        assert "Success!" in flowed

    @pytest.mark.slow
    def test_milestone_06_mlperf(self, project):
        """Milestone 06 Part 1: compression on the MLPerf Pareto frontier."""
        returncode, stdout, stderr = run_milestone(project, "06", part=1, timeout=180)

        assert returncode == 0, f"Milestone 06 failed:\nstdout: {stdout}\nstderr: {stderr}"

        # 2026-09-28: `"4" in stdout` passed on any output containing a digit 4.
        # Parse the measured INT8 ratio; FP32 -> INT8 codes approach 4x from
        # below because scale/zero-point metadata is counted (3.95x measured).
        plain = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
        # 2026-09-29: sizes now come from the stored INT8 code arrays plus scale metadata.
        ratios = re.findall(r"INT8 artifact counted from YOUR code arrays \(with scale metadata\):\s+[\d,]+ bytes\s+\((\d+\.\d+)× smaller\)", plain)
        assert len(ratios) == 1, f"Expected one measured INT8 ratio line, found {ratios}"
        ratio = float(ratios[0])
        assert 3.5 <= ratio <= 4.0, f"Implausible INT8 compression ratio: {ratio}×"
        # The synthesis must quote the same measured ratio, not a rounded "4×".
        assert f"INT8 codes are {ratio:.2f}× smaller" in _plain(stdout)

    @pytest.mark.slow
    def test_milestone_07_kernels(self, project):
        """Milestone 07: Custom Kernels (2024) - YOUR Module 17 kernels vs bundled native ones."""
        returncode, stdout, stderr = run_milestone(project, "07", timeout=120)

        assert returncode == 0, f"Milestone 07 failed:\nstdout: {stdout}\nstderr: {stderr}"
        # "simd"/"kernel" appear in tito's description; these lines do not.
        assert "[SUCCESS] Milestone 07 complete: YOUR acceleration kernels are correct." in stdout
        assert "[PASS]" in stdout
