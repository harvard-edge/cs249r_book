import json
import math
from pathlib import Path

import pytest

from tools import import_provisional_reference_results as provisional
from tools import import_reference_evidence as promotion


ROOT = Path(__file__).resolve().parents[1]


def _run(index: int, *, primary: float, quality: float) -> dict:
    return {
        "execution_index": index,
        "primary_metric_value": primary,
        "quality_value": quality,
        "quality_target_met": True,
        "wall_seconds": primary + 1.0,
        "device_executed": "cpu",
        "backend": "pytorch-cpu",
        "chip": "test-chip",
        "data_mode": "real",
        "comparison_fingerprint_sha256": f"fingerprint-{index}",
        "artifacts": [
            {
                "role": "report",
                "sha256": "sha256:" + "a" * 64,
                "n_bytes": 100,
            },
            {
                "role": "provenance",
                "sha256": "sha256:" + "b" * 64,
                "n_bytes": 200,
            },
        ],
    }


def _payload(case: promotion.EvidenceCase, *, runs: int) -> dict:
    rows = [
        _run(index, primary=10.0 + index, quality=0.9) for index in range(1, runs + 1)
    ]
    return {
        "schema": promotion.SUMMARY_SCHEMA,
        "evidence_id": "test-evidence",
        "eligible_for_promotion": runs == 5,
        "device_requested": "cpu",
        "canonical_seed": case.canonical_seed,
        "primary_metric": {"name": "train_and_eval_seconds"},
        "quality_metric": "test_accuracy",
        "quality_gate": {"metric": "test_accuracy", "target": 0.8},
        "functional_gate": None,
        "primary_metric_repeatability": {
            "metric": "train_and_eval_seconds",
            "coefficient_of_variation": 0.01 if runs == 5 else 0.0,
            "limit": 0.05,
            "passed": True,
        },
        "source": {"git_sha": "a" * 40},
        "runs": rows,
    }


def test_summary_record_keeps_verified_and_provisional_semantics_separate():
    case = promotion.expected_cases()["graph-node-classification__max__training"]
    verified = provisional.build_summary_record(
        case,
        _payload(case, runs=5),
        b"verified",
        evidence_class="five-run-verified",
    )
    draft = provisional.build_summary_record(
        case,
        _payload(case, runs=1),
        b"draft",
        evidence_class="single-run-provisional",
    )

    assert verified["review_eligible"] is True
    assert verified["eligible_for_promotion"] is True
    assert verified["repeatability"]["observed"] is True
    assert verified["measurement"]["run_count"] == 5
    assert draft["review_eligible"] is False
    assert draft["eligible_for_promotion"] is False
    assert draft["repeatability"] == {
        "observed": False,
        "metric": "train_and_eval_seconds",
        "coefficient_of_variation": None,
        "limit": 0.05,
        "passed": None,
        "note": "One execution cannot establish timing repeatability.",
    }


def test_index_requires_complete_registry_derived_case_closure():
    cases = promotion.expected_cases()
    records = {
        identifier: {
            "case_id": identifier,
            "workload": case.workload.id,
            "mode": case.mode,
            "phase": case.phase,
            "result_role": case.result_role,
            "evidence_class": "single-run-provisional",
            "eligible_for_promotion": False,
        }
        for identifier, case in cases.items()
    }
    index = provisional.build_index(
        records,
        source_git_sha="a" * 40,
        source_lock={"schema": "lock", "file_count": 1, "contract_count": 12},
        source_lock_bytes=b"lock",
    )
    assert index["workload_count"] == 9
    assert index["case_count"] == 12
    assert index["five_run_verified_case_count"] == 0
    assert index["provisional_case_count"] == 12

    records.pop(next(iter(records)))
    with pytest.raises(ValueError, match="closure mismatch"):
        provisional.build_index(
            records,
            source_git_sha="a" * 40,
            source_lock={"schema": "lock", "file_count": 1, "contract_count": 12},
            source_lock_bytes=b"lock",
        )


def test_committed_provisional_snapshot_is_closed_content_addressed_and_scrubbed():
    index_path = ROOT / "provisional_results" / "index.json"
    index = json.loads(index_path.read_text())
    assert index["schema"] == provisional.INDEX_SCHEMA
    assert index["workload_count"] == 9
    assert index["case_count"] == 12
    assert index["five_run_verified_case_count"] == 6
    assert index["provisional_case_count"] == 6
    assert index["publication_status"] == "draft-provisional-not-mlcommons-verified"

    classes = {}
    for entry in index["cases"]:
        relative = Path(entry["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        record_path = ROOT / relative
        data = record_path.read_bytes()
        assert promotion.sha256_bytes(data) == entry["sha256"]
        assert b"/Users/" not in data
        assert b"/private/" not in data
        record = json.loads(data)
        assert record["schema"] == provisional.RECORD_SCHEMA
        assert record["case_id"] == entry["case_id"]
        assert record["evidence_class"] == entry["evidence_class"]
        classes[entry["case_id"]] = record

    causal = classes["causal-language-modeling__max__training"]
    assert causal["evidence_class"] == "two-run-provisional"
    assert causal["review_eligible"] is False
    assert causal["repeatability"]["passed"] is False
    assert math.isclose(
        causal["repeatability"]["coefficient_of_variation"],
        0.05193783416636238,
    )
    assert causal["lineage_package"]["verification_checks"] == 52

    mirror = ROOT / "src" / "mlperf_edu" / "provisional_results"
    for path in (ROOT / "provisional_results").glob("*.json"):
        assert path.read_bytes() == (mirror / path.name).read_bytes()
