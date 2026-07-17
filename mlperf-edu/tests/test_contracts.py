from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from mlperf.contracts import evaluate_promotion_contract
from mlperf.registry import Workload, load_registry


def _artifact_paths(tmp_path: Path, stem: str) -> dict[str, str]:
    report_path = tmp_path / f"{stem}_report.json"
    provenance_path = tmp_path / f"{stem}.provd.json"
    report_path.write_text("{}\n")
    provenance_path.write_text("{}\n")
    return {"report": str(report_path), "provenance": str(provenance_path)}


def _quality_value(quality: dict[str, object]) -> float:
    target = float(quality["target"])
    direction = quality["direction"]
    if direction == "higher":
        return target + 0.01
    if direction == "lower":
        return max(0.0001, target - 0.001)
    return target


def _canonical_report(workload: Workload, tmp_path: Path) -> dict[str, object]:
    contract = workload.raw["canonical_max_contract"]
    quality = copy.deepcopy(contract["quality"])
    quality_value = _quality_value(quality)
    model: object = contract["model_id"]
    if contract.get("model_revision"):
        model = {
            "id": contract["model_id"],
            "revision": contract["model_revision"],
        }
    dataset: object = contract["dataset"]
    if contract.get("dataset_sha256"):
        dataset = {
            "name": contract["dataset"],
            "sha256": contract["dataset_sha256"],
        }
    primary_metric = workload.raw["measurement_protocol"]["primary_metric"]
    quality_metric = str(quality["metric_key"])
    metrics = {primary_metric: 1.0, quality_metric: quality_value}
    report_quality = {
        **quality,
        "quality_required": True,
        "target_met": True,
        "override": False,
    }
    return {
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "mode": contract.get("mode"),
        "phase": None,
        "scenario": workload.scenario,
        "status": "passed",
        "model": model,
        "dataset": dataset,
        "data_mode": contract["data_mode"],
        "seed": (contract.get("config") or {}).get("random_seed", 42),
        "config": copy.deepcopy(contract["config"]),
        "measurement_protocol": copy.deepcopy(workload.raw["measurement_protocol"]),
        "metrics": metrics,
        "quality": report_quality,
        "artifacts": _artifact_paths(tmp_path, workload.id),
    }


@pytest.mark.parametrize(
    "workload_id",
    sorted(
        workload_id
        for workload_id, workload in load_registry().items()
        if workload.raw.get("promotion_scope", True)
    ),
)
def test_every_promotion_scope_max_contract_is_promotion_eligible(
    workload_id: str, tmp_path: Path
) -> None:
    workload = load_registry()[workload_id]
    result = evaluate_promotion_contract(
        workload, _canonical_report(workload, tmp_path)
    )

    assert result["status"] == "passed", result["issues"]
    assert result["promotion_eligible"] is True
    assert result["result_role"] == "score-bearing"


@pytest.mark.parametrize(
    "workload_id",
    sorted(
        workload_id
        for workload_id, workload in load_registry().items()
        if not workload.raw.get("promotion_scope", True)
    ),
)
def test_functional_setup_max_contract_is_not_promotion_eligible(
    workload_id: str, tmp_path: Path
) -> None:
    workload = load_registry()[workload_id]
    result = evaluate_promotion_contract(
        workload, _canonical_report(workload, tmp_path)
    )

    assert result["status"] == "failed"
    assert result["promotion_eligible"] is False
    assert any(
        "not eligible for score-bearing review" in issue for issue in result["issues"]
    )


def test_graph_contract_applies_declared_accuracy_tolerance(tmp_path: Path) -> None:
    workload = load_registry()["graph-node-classification"]
    report = _canonical_report(workload, tmp_path)
    quality = workload.raw["canonical_max_contract"]["quality"]
    report["metrics"]["test_accuracy"] = (
        float(quality["target"]) - float(quality["tolerance"]) + 0.0001
    )

    result = evaluate_promotion_contract(workload, report)

    assert result["status"] == "passed", result["issues"]
    assert result["promotion_eligible"] is True


def test_graph_contract_rejects_accuracy_outside_tolerance(tmp_path: Path) -> None:
    workload = load_registry()["graph-node-classification"]
    report = _canonical_report(workload, tmp_path)
    quality = workload.raw["canonical_max_contract"]["quality"]
    report["metrics"]["test_accuracy"] = (
        float(quality["target"]) - float(quality["tolerance"]) - 0.0001
    )

    result = evaluate_promotion_contract(workload, report)

    assert result["status"] == "failed"
    assert result["promotion_eligible"] is False
    assert any("with tolerance" in issue for issue in result["issues"])


def _checkpoint_lineage(tmp_path: Path) -> dict[str, object]:
    checkpoint = tmp_path / "checkpoint.pt"
    source_report = tmp_path / "training_report.json"
    source_manifest = tmp_path / "training.provd.json"
    checkpoint.write_bytes(b"checkpoint")
    source_report.write_text("{}\n")
    source_manifest.write_text("{}\n")

    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "source_report_path": str(source_report),
        "source_report_sha256": digest(source_report),
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": digest(source_manifest),
        "source_manifest_verified": True,
        "source_quality_target_met": True,
    }


@pytest.mark.parametrize("phase", ["full", "prefill", "decode"])
def test_causal_inference_phase_contracts_are_promotion_eligible(
    phase: str, tmp_path: Path
) -> None:
    workload = load_registry()["causal-language-modeling"]
    phase_contract = workload.raw["mode_contracts"]["inference"]["phases"][phase]
    quality = copy.deepcopy(phase_contract["quality"])
    primary_metric = phase_contract["measurement_protocol"]["primary_metric"]
    metrics: dict[str, object] = {
        primary_metric: 10.0,
        str(quality["metric_key"]): _quality_value(quality),
    }
    metrics.update(
        {
            key: [0.01] * count
            for key, count in phase_contract["timing_sample_counts"].items()
        }
    )
    report = {
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "mode": "inference",
        "phase": phase,
        "scenario": phase_contract["scenario"],
        "status": "passed",
        "model": "nanogpt-shakespeare-char",
        "dataset": "prompt-suite-local",
        "data_mode": "checkpoint-backed",
        "seed": 1337,
        "config": copy.deepcopy(phase_contract["config"]),
        "measurement_protocol": copy.deepcopy(phase_contract["measurement_protocol"]),
        "metrics": metrics,
        "quality": {
            **quality,
            "quality_required": True,
            "target_met": True,
            "override": False,
        },
        "checkpoint_provenance": _checkpoint_lineage(tmp_path),
        "artifacts": _artifact_paths(tmp_path, f"causal-{phase}"),
    }

    result = evaluate_promotion_contract(workload, report)

    assert result["status"] == "passed", result["issues"]
    assert result["result_role"] == "performance-bearing"
