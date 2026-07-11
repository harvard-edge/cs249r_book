from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mlperf.registry import load_registry
from tools import check_taxonomy


AXES = ("working_set", "arithmetic_intensity", "dispatch")


def test_registry_withholds_all_taxonomy_claims_without_committed_evidence():
    workloads = load_registry()

    assert len(workloads) == 30
    for workload in workloads.values():
        regime = workload.raw.get("regime") or {}
        for axis in AXES:
            block = regime.get(axis) or {}
            assert block.get("value") == "unmeasured", (workload.id, axis)
            assert block.get("note"), (workload.id, axis)
            assert "evidence_sidecar" not in block, (workload.id, axis)
            assert "evidence_sha256" not in block, (workload.id, axis)
            assert "evidence_sha256_short" not in block, (workload.id, axis)


def test_measured_axis_requires_committed_evidence():
    errors = check_taxonomy.check_axis_evidence(
        "language/example",
        "arithmetic_intensity",
        {"value": "compute_bound", "flops_per_byte": 120.0},
    )

    assert any("requires a committed evidence_sidecar" in error for error in errors)
    assert any(
        "evidence fields require a committed evidence_sidecar" in error
        for error in errors
    )


def test_committed_evidence_requires_exact_digest_and_matching_claim(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(check_taxonomy, "REPO_ROOT", tmp_path)
    evidence_dir = tmp_path / "roofline"
    evidence_dir.mkdir()
    caps_path = evidence_dir / "machine_caps.json"
    caps_path.write_text('{"peak_flops": 1, "peak_bandwidth": 1}', encoding="utf-8")
    caps_digest = hashlib.sha256(caps_path.read_bytes()).hexdigest()
    evidence_path = evidence_dir / "example.json"
    payload = {
        "schema": "mlperf-edu-roofline/1.0",
        "workload": "example",
        "platform": {
            "peak_source": "measured",
            "peak_evidence_file": "roofline/machine_caps.json",
            "peak_evidence_sha256": caps_digest,
            "hardware_fingerprint": "test-platform-fingerprint",
        },
        "measurement": {
            "n_iter": 3,
            "warmup_iterations": 1,
            "synchronized": True,
            "operation_count_method": "audited analytical model",
            "byte_count_method": "audited analytical model",
            "wall_time_s": 1.0,
            "analytic_flops_total": 100.0,
            "analytic_bytes_total": 10.0,
            "achieved_FLOPS": 100.0,
            "achieved_BW_GBps": 1.0,
            "intensity_FLOPS_per_byte": 10.0,
            "dispatch_utilization": 0.5,
        },
        "regime_inference": {
            "axis_arithmetic_intensity": "compute_bound",
            "axis_dispatch": "dispatch_bound",
        },
    }
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    block = {
        "value": "compute_bound",
        "flops_per_byte": 120.0,
        "evidence_sidecar": "roofline/example.json",
        "evidence_sha256": digest,
    }

    assert (
        check_taxonomy.check_axis_evidence(
            "language/example", "arithmetic_intensity", block
        )
        == []
    )

    mismatch = {**block, "evidence_sha256": "0" * 64}
    errors = check_taxonomy.check_axis_evidence(
        "language/example", "arithmetic_intensity", mismatch
    )
    assert any("evidence_sha256 mismatch" in error for error in errors)


def test_declared_quality_asset_exists_and_matches_digest():
    workload = load_registry()["slm-decode"]

    assert check_taxonomy.check_workload_evidence("slm/slm-decode", workload.raw) == []


def test_committed_baseline_requires_content_addressed_package():
    errors = check_taxonomy.check_workload_evidence(
        "vision/example",
        {
            "verified_baseline": {
                "evidence_status": "committed-reference-summary",
                "review_eligible": True,
                "reference_package_availability": "local-handoff",
                "external_publication_status": "pending",
            }
        },
    )

    assert any("evidence_file path is missing" in error for error in errors)

    errors = check_taxonomy.check_workload_evidence(
        "vision/example",
        {
            "verified_baseline": {
                "evidence_status": "pending-clean-public-candidate-reference-summary",
                "review_eligible": True,
            }
        },
    )
    assert any("may be true only" in error for error in errors)


def test_reference_summary_indexes_every_raw_artifact_with_full_hashes():
    body = {
        "public": {"status": "score-bearing"},
        "quality_target": {
            "reference_runs": 1,
            "reference_protocol": {"seeds": [0]},
        },
    }
    payload = {
        "schema": "mlperf-edu-reference-evidence/0.2",
        "workload": "example",
        "status": "valid",
        "eligible_for_public_baseline": True,
        "evidence_tier": "public-candidate",
        "public_status": "score-bearing",
        "invalid_reasons": [],
        "acceptance": {"passed": True},
        "source": {
            "git_dirty": False,
            "git_sha": "a" * 40,
            "tool_sha256": "sha256:" + "b" * 64,
        },
        "seeds_requested": [0],
        "basis": {"reference_protocol": {"seeds": [0]}},
        "runs": [
            {
                "requested_seed": 0,
                "execution_ok": True,
                "evidence_valid": True,
                "seed_match": True,
                "manifest_verified": True,
                "quality_target_met": True,
                "timed_out": False,
                "invalid_reasons": [],
                "grade": {"passed": True},
                "report_path": "seed_0/report.json",
                "manifest_path": "seed_0/run.provd.json",
                "artifacts": [
                    {
                        "role": "report",
                        "path": "seed_0/report.json",
                        "sha256": "sha256:" + "c" * 64,
                        "n_bytes": 100,
                    },
                    {
                        "role": "provenance",
                        "path": "seed_0/run.provd.json",
                        "sha256": "sha256:" + "d" * 64,
                        "n_bytes": 200,
                    },
                ],
            }
        ],
    }

    assert check_taxonomy.check_reference_summary("vision/example", body, payload) == []

    payload["runs"][0]["artifacts"][0]["sha256"] = "sha256:short"
    errors = check_taxonomy.check_reference_summary("vision/example", body, payload)
    assert any("does not contain a full SHA-256 digest" in error for error in errors)


def test_taxonomy_cli_passes_current_registry():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/check_taxonomy.py"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Inspected 30 workloads." in result.stdout
    assert "PASS: 30 workloads consistent with taxonomy invariants." in result.stdout
