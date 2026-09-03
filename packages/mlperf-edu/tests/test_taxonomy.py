from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mlperf.registry import load_registry
from tools import check_taxonomy


AXES = ("working_set", "arithmetic_intensity", "dispatch")


def test_committed_causal_lineage_rechecks_every_bound_digest():
    checkpoint = "sha256:" + "a" * 64
    report = "sha256:" + "b" * 64
    provenance = "sha256:" + "c" * 64
    package = "sha256:" + "d" * 64
    training_id = check_taxonomy.CAUSAL_TRAINING_CASE_ID
    training_evidence = "sha256:" + "e" * 64
    binding = {
        "source_training_case_id": training_id,
        "source_training_evidence_id": "training-evidence",
        "source_training_evidence_sha256": training_evidence,
        "source_training_execution_index": 3,
        "source_training_checkpoint_sha256": checkpoint,
        "source_training_report_sha256": report,
        "source_training_provenance_sha256": provenance,
        "source_training_package_sha256": package,
    }
    indexed = {
        training_id: {
            "evidence_id": "training-evidence",
            "evidence_sha256": training_evidence,
        }
    }
    payloads = {
        training_id: {
            "aggregate": {"quality": {"median": 0.75}},
            "runs": [
                {
                    "execution_index": 3,
                    "quality_value": 0.75,
                    "artifacts": [
                        {"role": "checkpoint", "sha256": checkpoint},
                        {"role": "report", "sha256": report},
                        {"role": "provenance", "sha256": provenance},
                    ],
                }
            ],
        }
    }
    for identifier in check_taxonomy.CAUSAL_INFERENCE_CASE_IDS:
        indexed[identifier] = {"source_training": copy.deepcopy(binding)}
        payloads[identifier] = {
            "nanogpt_training_lineage": {"package_sha256": package},
            "runs": [
                {
                    "artifacts": [
                        {"role": "checkpoint", "sha256": checkpoint},
                        {"role": "source_training_report", "sha256": report},
                        {
                            "role": "source_training_provenance",
                            "sha256": provenance,
                        },
                    ]
                }
            ],
        }

    assert check_taxonomy.check_case_source_training_lineage(indexed, payloads) == []

    tampered = copy.deepcopy(indexed)
    tampered[check_taxonomy.CAUSAL_INFERENCE_CASE_IDS[-1]]["source_training"][
        "source_training_package_sha256"
    ] = "sha256:" + "f" * 64
    errors = check_taxonomy.check_case_source_training_lineage(tampered, payloads)
    assert any("do not share one source_training" in error for error in errors)
    assert any("staged package digest" in error for error in errors)


def test_registry_withholds_all_taxonomy_claims_without_committed_evidence():
    workloads = load_registry()

    assert len(workloads) == 14
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


def test_reference_summary_indexes_every_raw_artifact_with_full_hashes():
    body = {
        "public": {"status": "score-bearing"},
        "quality_target": {
            "metric": "accuracy",
            "value": 0.7,
            "direction": "higher",
            "reference_runs": 1,
            "reference_protocol": {"profile": "max", "seeds": [0]},
        },
    }
    payload = {
        "schema": "mlperf-edu-reference-evidence/0.3",
        "workload": "example",
        "status": "valid",
        "eligible_for_public_baseline": True,
        "evidence_tier": "public-candidate",
        "public_status": "score-bearing",
        "profile": "max",
        "evidence_id": "example_max_attempt",
        "quality_metric": "accuracy",
        "quality_direction": "higher",
        "quality_target": 0.7,
        "reference_metric_role": "quality",
        "primary_metric": {"name": "accuracy", "role": "quality"},
        "functional_gate": None,
        "repeatability": None,
        "invalid_reasons": [],
        "acceptance": {
            "passed": True,
            "statistic": "median",
            "operator": ">=",
            "target": 0.7,
            "value": 0.75,
        },
        "source": {
            "git_dirty": False,
            "git_sha": "a" * 40,
            "git_status_sha256": check_taxonomy.EMPTY_SHA256,
            "git_patch_sha256": check_taxonomy.EMPTY_SHA256,
            "tool_sha256": check_taxonomy.SWEEP_TOOL_SHA256,
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
                "quality_value": 0.75,
                "wall_seconds": 1.25,
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
        "aggregate": {
            "primary_metric": {
                "count": 1,
                "median": 0.75,
                "mean": 0.75,
                "min": 0.75,
                "max": 0.75,
                "stdev": 0.0,
            },
            "quality": {
                "count": 1,
                "median": 0.75,
                "mean": 0.75,
                "min": 0.75,
                "max": 0.75,
                "stdev": 0.0,
            },
            "wall_seconds": {
                "count": 1,
                "median": 1.25,
                "mean": 1.25,
                "min": 1.25,
                "max": 1.25,
                "stdev": 0.0,
            },
        },
    }

    assert check_taxonomy.check_reference_summary("vision/example", body, payload) == []

    payload["runs"][0]["artifacts"][0]["sha256"] = "sha256:short"
    errors = check_taxonomy.check_reference_summary("vision/example", body, payload)
    assert any("does not contain a full SHA-256 digest" in error for error in errors)


def test_schema_04_score_summary_separates_timed_primary_and_quality_gate():
    body = {
        "public": {"status": "score-bearing"},
        "measurement_protocol": {"primary_metric": "train_and_eval_seconds"},
        "quality_target": {
            "metric": "accuracy",
            "value": 0.7,
            "direction": "higher",
            "tolerance": 0.0,
            "reference_runs": 1,
            "reference_protocol": {"profile": "max", "seeds": [0]},
        },
    }
    run = {
        "requested_seed": 0,
        "execution_ok": True,
        "evidence_valid": True,
        "seed_match": True,
        "manifest_verified": True,
        "quality_target_met": True,
        "primary_metric_declared": "train_and_eval_seconds",
        "primary_metric_key": "train_and_eval_seconds",
        "primary_metric_value": 12.5,
        "quality_metric_declared": "accuracy",
        "quality_metric_key": "accuracy",
        "quality_value": 0.75,
        "functional_metric_declared": None,
        "functional_metric_key": None,
        "functional_metric_value": None,
        "wall_seconds": 13.0,
        "timed_out": False,
        "invalid_reasons": [],
        "grade": {
            "passed": True,
            "target_met": True,
            "metric": "accuracy",
            "value": 0.75,
            "target": 0.7,
        },
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
    payload = {
        "schema": "mlperf-edu-reference-evidence/0.4",
        "workload": "example",
        "status": "valid",
        "eligible_for_public_baseline": True,
        "evidence_tier": "public-candidate",
        "public_status": "score-bearing",
        "profile": "max",
        "evidence_id": "example_max_attempt",
        "reference_metric_role": "performance",
        "primary_metric": {
            "name": "train_and_eval_seconds",
            "role": "performance",
        },
        "quality_metric": "accuracy",
        "quality_target": 0.7,
        "quality_direction": "higher",
        "quality_gate": {
            "metric": "accuracy",
            "target": 0.7,
            "direction": "higher",
            "tolerance": 0.0,
            "all_runs_must_pass": True,
        },
        "functional_gate": None,
        "repeatability": None,
        "invalid_reasons": [],
        "acceptance": {
            "passed": True,
            "statistic": "median",
            "operator": ">=",
            "target": 0.7,
            "value": 0.75,
            "all_runs_passed": True,
            "passed_runs": 1,
            "run_count": 1,
            "tolerance": 0.0,
        },
        "source": {
            "git_dirty": False,
            "git_sha": "a" * 40,
            "git_status_sha256": check_taxonomy.EMPTY_SHA256,
            "git_patch_sha256": check_taxonomy.EMPTY_SHA256,
            "tool_sha256": check_taxonomy.SWEEP_TOOL_SHA256,
        },
        "seeds_requested": [0],
        "basis": {
            "reference_protocol": {"seeds": [0]},
            "quality_target": {
                "metric": "accuracy",
                "target": 0.7,
                "direction": "higher",
                "tolerance": 0.0,
                "all_runs_must_pass": True,
            },
        },
        "runs": [run],
        "aggregate": {
            "primary_metric": {
                "count": 1,
                "median": 12.5,
                "mean": 12.5,
                "min": 12.5,
                "max": 12.5,
                "stdev": 0.0,
            },
            "quality": {
                "count": 1,
                "median": 0.75,
                "mean": 0.75,
                "min": 0.75,
                "max": 0.75,
                "stdev": 0.0,
            },
            "wall_seconds": {
                "count": 1,
                "median": 13.0,
                "mean": 13.0,
                "min": 13.0,
                "max": 13.0,
                "stdev": 0.0,
            },
        },
    }

    assert check_taxonomy.check_reference_summary("vision/example", body, payload) == []

    payload["runs"][0]["primary_metric_value"] = 0.75
    errors = check_taxonomy.check_reference_summary("vision/example", body, payload)
    assert any("aggregate.primary_metric" in error for error in errors)

    payload["runs"][0]["primary_metric_value"] = 12.5
    payload["runs"][0]["quality_value"] = 0.1
    errors = check_taxonomy.check_reference_summary("vision/example", body, payload)
    assert any(
        "does not satisfy the registry quality target" in error for error in errors
    )


def committed_summary(workload_id: str) -> tuple[dict, dict]:
    workload = load_registry()[workload_id]
    body = json.loads(json.dumps(workload.raw))
    evidence_file = body["verified_baseline"]["evidence_file"]
    path = Path(__file__).resolve().parents[1] / evidence_file
    return body, json.loads(path.read_text())


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
    assert "Inspected 14 workloads." in result.stdout
    assert "PASS: 14 workloads consistent with taxonomy invariants." in result.stdout
