from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_generate_review_packets(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "tools/generate_review_packets.py",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "wrote 9 packet(s)" in result.stdout
    assert (tmp_path / "README.md").is_file()

    nanogpt = (tmp_path / "nanogpt-train.md").read_text()
    assert "## Quality Contract" in nanogpt
    assert "Reference protocol" in nanogpt
    assert "No public-release warning" in nanogpt

    prefill = (tmp_path / "nanogpt-inference__prefill.md").read_text()
    assert "## Checkpoint Lineage" in prefill
    assert "Source quality" in prefill
    assert "cross_entropy_loss lower 2.3 basis=reference_runs" in prefill

    slm = (tmp_path / "smollm2-chat-inference__baseline.md").read_text()
    assert "## Functional Contract" in slm
    assert "Model license" in slm
    assert "Apache-2.0" in slm

    check = subprocess.run(
        [
            sys.executable,
            "tools/generate_review_packets.py",
            "--output-dir",
            str(tmp_path),
            "--check",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    assert "review packets are current (9 packet(s))" in check.stdout
