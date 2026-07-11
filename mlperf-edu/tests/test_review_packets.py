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
    assert "wrote 8 packet(s)" in result.stdout
    assert (tmp_path / "README.md").is_file()

    nanogpt = (tmp_path / "nanogpt-train.md").read_text()
    assert "## Quality Contract" in nanogpt
    assert "Reference protocol" in nanogpt
    assert "## Measurement and Evidence Contract" in nanogpt
    assert "## Taxonomy Evidence" in nanogpt
    assert "value=unmeasured; evidence=none; sha256=none" in nanogpt
    assert "pending-clean-public-candidate-reference-summary" in nanogpt
    assert "baseline is not backed by a committed reference summary" in nanogpt

    prefill = (tmp_path / "nanogpt-inference__prefill.md").read_text()
    assert "## Checkpoint Lineage" in prefill
    assert "Source quality" in prefill
    assert "cross_entropy_loss lower 2.3 basis=reference_runs" in prefill
    assert "primary_metric=prefill_tokens_per_sec" in prefill
    assert (
        "shared checkpoint source nanogpt-train is not backed by a committed" in prefill
    )
    assert "reference summary" in prefill

    slm = (tmp_path / "smollm2-chat-inference__baseline.md").read_text()
    assert "## Functional Contract" in slm
    assert "Model license" in slm
    assert "Apache-2.0" in slm
    assert "mlperf-edu-slm-quality/0.1" in slm
    assert "primary_metric=output_tokens_per_sec" in slm
    assert "calibration values are informational" in slm

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
    assert "review packets are current (8 packet(s))" in check.stdout
