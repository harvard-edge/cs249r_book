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
    assert "committed-reference-summary" in nanogpt
    assert "historical-protocol-superseded" in nanogpt
    assert "replacement blocker" in nanogpt
    assert "reference_results/nanogpt-train/" in nanogpt
    assert "external-publication blocker" in nanogpt
    assert "baseline is not backed by a committed reference summary" not in nanogpt

    prefill = (tmp_path / "nanogpt-inference__prefill.md").read_text()
    assert "## Checkpoint Lineage" in prefill
    assert "Source quality" in prefill
    assert "cross_entropy_loss lower 2.3 basis=reference_runs" in prefill
    assert "primary_metric=prefill_tokens_per_sec" in prefill
    assert "reference_results/nanogpt-prefill/" in prefill
    assert "nanogpt-prefill_max_20260711T084140.159367Z" in prefill
    assert "bc3e8f01c279d1d2bbf0f8a24b15e85270584a5d35e16883a9751b4b5a04b68b" in prefill
    assert "metric_values_by_seed=" in prefill
    assert "not an MLCommons-verified result" in prefill
    assert "raw reference package for shared checkpoint source nanogpt-train" in prefill
    assert (
        "shared checkpoint source nanogpt-train has only protocol-superseded" in prefill
    )
    assert "no published package URL is recorded" in prefill
    train_command = "mlperf run --workload nanogpt-train --profile max"
    inference_command = (
        "mlperf run --workload nanogpt-inference --variant prefill --profile max"
    )
    assert train_command in prefill
    assert inference_command in prefill
    assert prefill.index(train_command) < prefill.index(inference_command)
    assert "mlperf fetch --workload nanogpt-train --profile max" in prefill
    assert "--dry-run" not in prefill
    assert "mlperf verify" in prefill
    assert "mlperf grade" in prefill

    slm = (tmp_path / "smollm2-chat-inference__baseline.md").read_text()
    assert "## Functional Contract" in slm
    assert "Model license" in slm
    assert "Apache-2.0" in slm
    assert "mlperf-edu-slm-quality/0.2" in slm
    assert "token-weighted-continuation-nll" in slm
    assert "primary_metric=output_tokens_per_sec" in slm
    assert "reference_results/slm-decode/" in slm
    assert "c13f7b7afb626cd4f3cdcb9620693a95ce8d46881d1e8c6f18ba0234442f1185" in slm
    assert "metric_values_by_seed=" in slm
    assert "not an MLCommons-verified result" in slm
    assert "external-publication blocker" in slm
    assert "historical-protocol-superseded" in slm
    assert "replacement blocker" in slm
    assert "calibration values are informational" not in slm

    for packet in tmp_path.glob("*.md"):
        for line in packet.read_text().splitlines():
            assert not line.endswith("|  |"), (packet.name, line)

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
