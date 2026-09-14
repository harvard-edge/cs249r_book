from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

from mlperf.registry import load_registry, public_contract_issues
from tools import check_taxonomy
from tools import import_reference_evidence as evidence
from tools import sync_verified_baselines as sync


def test_check_keeps_draft_results_out_of_verified_registry_baselines(
    monkeypatch, capsys, tmp_path: Path
):
    monkeypatch.setattr(sync, "INDEX_PATH", tmp_path / "missing-index.json")
    monkeypatch.setattr(sys, "argv", ["sync_verified_baselines.py", "--check"])

    assert sync.main() == 0
    assert "no draft result is exposed" in capsys.readouterr().out


def test_sync_refuses_to_promote_draft_results(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setattr(sync, "INDEX_PATH", tmp_path / "missing-index.json")
    monkeypatch.setattr(sys, "argv", ["sync_verified_baselines.py"])

    assert sync.main() == 1
    assert "draft results must not be synchronized" in capsys.readouterr().err


def _payload(case: evidence.EvidenceCase) -> dict:
    primary_values = [10.00, 10.05, 10.10, 10.15, 10.20]
    quality_values = [float(case.gate["target"])] * 5
    runs = [
        {
            "primary_metric_value": primary,
            "quality_value": quality,
            "data_mode": case.data_mode,
            "backend": "pytorch-mps",
            "chip": "Apple test",
            "evidence_valid": True,
            "promotion_contract": {"status": "passed"},
        }
        for primary, quality in zip(primary_values, quality_values)
    ]
    primary = evidence.aggregate(primary_values)
    quality = evidence.aggregate(quality_values)
    wall = evidence.aggregate([value + 1.0 for value in primary_values])
    return {
        "schema": evidence.SUMMARY_SCHEMA,
        "evidence_id": f"{case.workload.id}_max_20260712T120000.000000Z",
        "workload": case.workload.id,
        "status": "valid",
        "evidence_tier": "promotion-candidate",
        "eligible_for_promotion": True,
        "eligible_for_public_baseline": False,
        "invalid_reasons": [],
        "source": {
            "git_sha": "a" * 40,
            "git_dirty": False,
            "git_status_sha256": evidence.EMPTY_SHA256,
            "git_patch_sha256": evidence.EMPTY_SHA256,
        },
        "profile": "max",
        "mode": case.mode,
        "phase": case.phase,
        "result_role": case.result_role,
        "device_requested": "mps",
        "comparison_fingerprint_sha256": "b" * 64,
        "canonical_seed": case.canonical_seed,
        "primary_metric": {
            "name": case.measurement_protocol["primary_metric"],
        },
        "primary_metric_repeatability": evidence._repeatability(primary_values, case),
        "quality_metric": case.gate["metric"],
        "quality_gate": {
            "metric": case.gate["metric"],
            "target": case.gate["target"],
            "direction": case.gate["direction"],
        },
        "functional_gate": (
            {
                "metric": case.gate["metric"],
                "target": case.gate["target"],
            }
            if case.result_role == "performance-bearing"
            else None
        ),
        "acceptance": {"value": 5},
        "aggregate": {
            "primary_metric": primary,
            "quality": quality if case.result_role == "score-bearing" else None,
            "wall_seconds": wall,
        },
        "basis": {
            "reference_protocol": {
                "profile": "max",
                "seeds": [case.canonical_seed] * 5,
            }
        },
        "runs": runs,
    }


def _entry(case: evidence.EvidenceCase) -> dict:
    entry = {
        "case_id": case.case_id,
        "workload": case.workload.id,
        "evidence_id": f"{case.workload.id}_max_20260712T120000.000000Z",
        "path": f"reference_results/{case.case_id}/summary.json",
        "evidence_sha256": "sha256:" + "c" * 64,
    }
    if case.result_role == "performance-bearing":
        entry["source_training"] = {
            "source_training_case_id": "causal-language-modeling__max__training",
            "source_training_checkpoint_sha256": "sha256:" + "d" * 64,
        }
    return entry


def test_score_baseline_preserves_timing_and_quality_distributions():
    case = evidence.expected_cases()["image-classification__max__inference"]
    baseline = sync.build_baseline(_entry(case), _payload(case))

    assert baseline["case_id"] == case.case_id
    assert baseline["result_role"] == "score-bearing"
    assert baseline["accepted_runs"] == 5
    assert baseline["primary_metric"] == "inference_and_evaluation_seconds"
    assert len(baseline["metric_values"]) == 5
    assert baseline["quality_metric"] == "top1_accuracy"
    assert baseline["top1_accuracy"] == case.gate["target"]
    assert len(baseline["quality_values"]) == 5


def test_performance_baseline_records_functional_gate_without_quality_score():
    case = evidence.expected_cases()[
        "causal-language-modeling__max__inference__prefill"
    ]
    payload = _payload(case)
    baseline = sync.build_baseline(_entry(case), payload)

    assert baseline["result_role"] == "performance-bearing"
    assert baseline["functional_passes"] == 5
    assert baseline["functional_gate"] == payload["functional_gate"]
    assert baseline["source_training"] == _entry(case)["source_training"]
    assert "quality_metric" not in baseline
    assert "quality_values" not in baseline


def test_synchronized_contract_promotes_canonical_case_and_keeps_case_map():
    registry = load_registry()
    workload = registry["image-classification"]
    case = evidence.expected_cases()["image-classification__max__inference"]
    entry = _entry(case)
    payload = _payload(case)

    result = sync.synchronized_contract(
        {"id": workload.id, **workload.raw},
        {case.case_id: (entry, payload)},
    )

    assert result["public"]["status"] == "score-bearing"
    assert set(result["verified_baselines"]) == {case.case_id}
    assert result["verified_baseline"] == result["verified_baselines"][case.case_id]
    assert result["quality_target"]["reference_runs"] == 5
    assert result["quality_target"]["variance_summary"]["runs"] == 5
    assert (
        result["quality_target"]["reference_protocol"]["fresh_process_per_run"] is True
    )

    promoted = replace(
        workload,
        public_status=result["public"]["status"],
        public_rationale=result["public"]["rationale"],
        quality_reference_runs=5,
        quality_variance_summary=result["quality_target"]["variance_summary"],
        quality_reference_protocol=result["quality_target"]["reference_protocol"],
        raw=result,
    )
    assert public_contract_issues(promoted) == []
    assert (
        check_taxonomy.check_promoted_case_summary(
            "vision/image-classification",
            result,
            result["verified_baseline"],
            payload,
        )
        == []
    )
