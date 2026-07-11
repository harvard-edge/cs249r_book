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


def test_superseded_baseline_requires_exact_lifecycle_and_keeps_internal_checks():
    workload = load_registry()["resnet18-train"]
    body = json.loads(json.dumps(workload.raw))
    body["verified_baseline"]["protocol_compatibility"] = "superseded"
    body["verified_baseline"]["review_eligible"] = True
    body["verified_baseline"]["replacement_required"] = False

    errors = check_taxonomy.check_workload_evidence("vision/resnet18-train", body)
    assert any(
        "historical evidence lifecycle must be exactly" in error for error in errors
    )

    evidence_file = workload.raw["verified_baseline"]["evidence_file"]
    payload = json.loads(
        (Path(__file__).resolve().parents[1] / evidence_file).read_text()
    )
    assert (
        check_taxonomy.check_historical_reference_summary(
            "vision/resnet18-train", payload
        )
        == []
    )
    payload["aggregate"]["quality"]["mean"] += 0.01
    errors = check_taxonomy.check_historical_reference_summary(
        "vision/resnet18-train", payload
    )
    assert any("recomputed value" in error for error in errors)


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


def test_committed_baseline_display_fields_cannot_drift_from_summary():
    body, payload = committed_summary("resnet18-train")

    assert (
        check_taxonomy.check_reference_summary("vision/resnet18-train", body, payload)
        == []
    )

    body["verified_baseline"]["median"] += 0.01
    body["verified_baseline"]["metric_values_by_seed"][0] -= 0.01
    body["verified_baseline"]["source_git_sha"] = "0" * 40
    errors = check_taxonomy.check_reference_summary(
        "vision/resnet18-train", body, payload
    )
    assert any("verified_baseline.median" in error for error in errors)
    assert any("verified_baseline.metric_values_by_seed" in error for error in errors)
    assert any("verified_baseline.source_git_sha" in error for error in errors)


def test_committed_summary_aggregates_are_recomputed_from_seed_values():
    body, payload = committed_summary("anomaly-ae-train")
    payload["aggregate"]["primary_metric"]["mean"] += 0.01

    errors = check_taxonomy.check_reference_summary(
        "tiny/anomaly-ae-train", body, payload
    )

    assert any("recomputed value" in error for error in errors)
    assert any("verified_baseline.mean" in error for error in errors)


def test_schema_04_performance_summary_uses_primary_metric_not_quality_metric():
    body, payload = committed_summary("slm-decode")

    assert payload["quality_metric"] is None
    assert check_taxonomy.check_reference_summary("slm/slm-decode", body, payload) == []


def test_reference_summary_acceptance_cannot_drift_or_hide_a_failed_seed():
    body, payload = committed_summary("resnet18-train")
    payload["acceptance"]["value"] += 0.01

    errors = check_taxonomy.check_reference_summary(
        "vision/resnet18-train", body, payload
    )
    assert any("acceptance.value" in error for error in errors)

    body, payload = committed_summary("resnet18-train")
    payload["runs"][0]["quality_value"] = 0.1
    errors = check_taxonomy.check_reference_summary(
        "vision/resnet18-train", body, payload
    )
    assert any(
        "does not satisfy the registry quality target" in error for error in errors
    )


def test_reference_summary_paths_and_ids_are_portable():
    body, payload = committed_summary("resnet18-train")
    payload["evidence_id"] = "bad\nidentifier"
    payload["runs"][0]["artifacts"][0]["path"] = "..\\outside.json"

    errors = check_taxonomy.check_reference_summary(
        "vision/resnet18-train", body, payload
    )
    assert any("evidence_id is not portable" in error for error in errors)
    assert any("path is missing, absolute, or escapes" in error for error in errors)


def test_nanogpt_lineage_requires_stable_indexed_training_artifacts():
    body, payload = committed_summary("nanogpt-prefill")
    payload["runs"][0]["artifacts"] = [
        artifact
        for artifact in payload["runs"][0]["artifacts"]
        if artifact["role"] != "source_training_report"
    ]

    errors = check_taxonomy.check_reference_summary(
        "language/nanogpt-prefill", body, payload
    )

    assert any("exactly one source_training_report" in error for error in errors)


def test_nanogpt_lineage_roles_are_checked_before_baseline_promotion():
    body, payload = committed_summary("nanogpt-prefill")
    body.pop("verified_baseline")
    payload["runs"][0]["artifacts"] = [
        artifact
        for artifact in payload["runs"][0]["artifacts"]
        if artifact["role"] != "source_training_provenance"
    ]

    errors = check_taxonomy.check_reference_summary(
        "language/nanogpt-prefill", body, payload
    )
    assert any("exactly one source_training_provenance" in error for error in errors)


def test_slm_summary_requires_content_addressed_model_metadata():
    body, payload = committed_summary("slm-decode")
    payload["runs"][0]["artifacts"] = [
        artifact
        for artifact in payload["runs"][0]["artifacts"]
        if artifact["role"] != "model_metadata"
    ]

    errors = check_taxonomy.check_reference_summary("slm/slm-decode", body, payload)

    assert any("lacks model_metadata" in error for error in errors)


def test_slm_model_metadata_is_checked_before_baseline_promotion():
    body, payload = committed_summary("slm-decode")
    body.pop("verified_baseline")
    payload["runs"][0]["artifacts"] = [
        artifact
        for artifact in payload["runs"][0]["artifacts"]
        if artifact["role"] != "model_metadata"
    ]

    errors = check_taxonomy.check_reference_summary("slm/slm-decode", body, payload)
    assert any("lacks model_metadata" in error for error in errors)


def test_checkpoint_dependents_bind_to_committed_training_summary():
    workloads = {
        workload_id: json.loads(json.dumps(workload.raw))
        for workload_id, workload in load_registry().items()
    }
    workloads["nanogpt-prefill"]["verified_baseline"][
        "source_training_evidence_sha256"
    ] = "0" * 64

    errors = check_taxonomy.check_shared_checkpoint_evidence(workloads)

    assert any("source_training_evidence_sha256" in error for error in errors)


def test_checkpoint_dependents_bind_the_selected_seed_and_checkpoint_digest():
    workloads = {
        workload_id: json.loads(json.dumps(workload.raw))
        for workload_id, workload in load_registry().items()
    }
    workloads["nanogpt-prefill"]["verified_baseline"]["source_training_seed"] = 1
    workloads["nanogpt-decode"]["verified_baseline"][
        "source_training_checkpoint_sha256"
    ] = "0" * 64

    errors = check_taxonomy.check_shared_checkpoint_evidence(workloads)
    assert any(
        "does not select the committed median-quality" in error for error in errors
    )
    assert any("does not match the selected training run" in error for error in errors)


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
