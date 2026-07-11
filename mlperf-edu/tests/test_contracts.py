from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from mlperf.contracts import evaluate_report_contract
from mlperf.registry import Workload, load_registry
from mlperf.runners.slm import (
    aggregate_slm_quality_cases,
    load_slm_quality_suite,
    slm_quality_gate_results,
    slm_quality_suite_path,
)


def _artifacts(tmp_path: Path) -> dict[str, str]:
    report_path = tmp_path / "report.json"
    provenance_path = tmp_path / "report.provd.json"
    report_path.write_text("{}\n")
    provenance_path.write_text("{}\n")
    return {"report": str(report_path), "provenance": str(provenance_path)}


def _canonical_identity(workload: Workload) -> dict[str, object]:
    contract = workload.raw["canonical_max_contract"]
    dataset: object = contract["dataset"]
    if contract.get("dataset_sha256"):
        dataset = {
            "name": contract["dataset"],
            "sha256": contract["dataset_sha256"],
        }
    return {
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "canonical_workload": workload.canonical_workload or workload.id,
        "variant": workload.variant,
        "scenario": workload.scenario,
        "profile": "max",
        "status": "passed",
        "model": contract["model_id"],
        "dataset": dataset,
        "data_mode": contract["data_mode"],
        "seed": 0,
        "config": copy.deepcopy(contract["config"]),
        "measurement_protocol": copy.deepcopy(workload.raw["measurement_protocol"]),
    }


def _resnet_report(workload: Workload, tmp_path: Path) -> dict[str, object]:
    report = _canonical_identity(workload)
    report.update(
        {
            "metrics": {"top1_accuracy": 0.9, "train_and_eval_seconds": 10.0},
            "quality": {
                **copy.deepcopy(workload.raw["canonical_max_contract"]["quality"]),
                "metric_key": "top1_accuracy",
                "quality_required": True,
                "target_met": True,
                "override": False,
            },
            "artifacts": _artifacts(tmp_path),
        }
    )
    return report


def _dlrm_report(workload: Workload, tmp_path: Path) -> dict[str, object]:
    report = _canonical_identity(workload)
    report.update(
        {
            "metrics": {
                "best_roc_auc": 0.77,
                "roc_auc": 0.77,
                "train_and_eval_seconds": 10.0,
            },
            "quality": {
                **copy.deepcopy(workload.raw["canonical_max_contract"]["quality"]),
                "quality_required": True,
                "target_met": True,
                "override": False,
            },
            "artifacts": _artifacts(tmp_path),
        }
    )
    return report


def _anomaly_report(workload: Workload, tmp_path: Path) -> dict[str, object]:
    report = _canonical_identity(workload)
    report.update(
        {
            "metrics": {
                "anomaly_auroc": 0.94,
                "anomaly_worst_class_auroc": 0.92,
                "anomaly_min_control_margin": 0.25,
                "train_and_eval_seconds": 10.0,
            },
            "quality": {
                **copy.deepcopy(workload.raw["canonical_max_contract"]["quality"]),
                "quality_required": True,
                "target_met": True,
                "override": False,
                "gates": {
                    "passed": True,
                    "macro_auroc": {
                        "value": 0.94,
                        "target": 0.93,
                        "direction": "higher",
                        "met": True,
                    },
                    "worst_class_auroc": {
                        "value": 0.92,
                        "target": 0.90,
                        "direction": "higher",
                        "met": True,
                    },
                    "control_margin": {
                        "value": 0.25,
                        "target": 0.20,
                        "direction": "higher",
                        "met": True,
                    },
                },
            },
            "artifacts": _artifacts(tmp_path),
        }
    )
    return report


def _nanogpt_prefill_report(workload: Workload, tmp_path: Path) -> dict[str, object]:
    report = _canonical_identity(workload)
    checkpoint = tmp_path / "checkpoint.pt"
    source_report = tmp_path / "source-report.json"
    source_manifest = tmp_path / "source.provd.json"
    for path, payload in (
        (checkpoint, b"checkpoint"),
        (source_report, b"{}\n"),
        (source_manifest, b"{}\n"),
    ):
        path.write_bytes(payload)

    report.update(
        {
            "metrics": {
                "prefill_tokens_per_sec": 100.0,
                "prefill_latency_samples_s": [0.01] * 20,
            },
            "quality": {
                **copy.deepcopy(workload.raw["canonical_max_contract"]["quality"]),
                "metric_key": "prefill_tokens_per_sec",
                "quality_required": True,
                "target_met": True,
            },
            "measurement_protocol": copy.deepcopy(workload.raw["measurement_protocol"]),
            "checkpoint_provenance": {
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": "sha256:"
                + hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "source_report_path": str(source_report),
                "source_report_sha256": "sha256:"
                + hashlib.sha256(source_report.read_bytes()).hexdigest(),
                "source_manifest_path": str(source_manifest),
                "source_manifest_sha256": "sha256:"
                + hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                "source_manifest_verified": True,
                "source_quality_target_met": True,
            },
            "artifacts": _artifacts(tmp_path),
        }
    )
    return report


def _slm_report(workload: Workload, tmp_path: Path) -> dict[str, object]:
    report = _canonical_identity(workload)
    contract = workload.raw["canonical_max_contract"]
    evaluation_contract = contract["quality_evaluation"]
    fixture = load_slm_quality_suite(slm_quality_suite_path().read_bytes())
    case_results = [
        {
            "id": case["id"],
            "category": case["category"],
            "mean_nll": 1.5 + (index // 4) * 0.05,
            "continuation_tokens": fixture["expected_continuation_tokens"][case["id"]],
        }
        for index, case in enumerate(fixture["cases"])
    ]
    quality_result = {
        "suite": evaluation_contract["suite"],
        "fixture_version": evaluation_contract["fixture_version"],
        "suite_sha256": evaluation_contract["suite_sha256"],
        "cases": evaluation_contract["cases"],
        "categories": evaluation_contract["categories"],
        "aggregation": evaluation_contract["aggregation"],
        "category_guard": evaluation_contract["category_guard"],
        "case_results": case_results,
        **aggregate_slm_quality_cases(case_results),
    }
    overall_target = evaluation_contract["gates"]["overall_perplexity"]["target"]
    category_target = evaluation_contract["gates"]["worst_category_perplexity"][
        "target"
    ]
    quality_gates = slm_quality_gate_results(
        quality_result,
        max_perplexity=overall_target,
        max_worst_category_perplexity=category_target,
    )
    report.update(
        {
            "model": {
                "id": contract["model_id"],
                "revision": contract["model_revision"],
                "n_params": contract["model_n_params"],
            },
            "metrics": {
                "generated_tokens": 16,
                "output_tokens_per_sec": 100.0,
                "quality_perplexity": quality_result["perplexity"],
                "quality_worst_category_perplexity": quality_result[
                    "worst_category_perplexity"
                ],
                "quality_total_continuation_tokens": quality_result[
                    "total_continuation_tokens"
                ],
                "prefill_latency_samples_s": [0.01] * 20,
                "request_ttft_samples_s": [0.02] * 20,
                "itl_samples_s": [0.01] * 300,
                "request_end_to_end_samples_s": [0.2] * 20,
            },
            "quality": {
                **copy.deepcopy(contract["quality"]),
                "metric_key": "generated_tokens",
                "quality_required": True,
                "target_met": True,
                "override": False,
            },
            "measurement_protocol": copy.deepcopy(workload.raw["measurement_protocol"]),
            "model_asset": {"revision": contract["model_revision"]},
            "quality_evaluation": {
                "suite": evaluation_contract["suite"],
                "fixture_version": evaluation_contract["fixture_version"],
                "cases": evaluation_contract["cases"],
                "categories": evaluation_contract["categories"],
                "aggregation": evaluation_contract["aggregation"],
                "category_guard": evaluation_contract["category_guard"],
                "gates": quality_gates,
                "max_quantized_nll_delta": evaluation_contract[
                    "max_quantized_nll_delta"
                ],
                "status": "passed",
                "result": quality_result,
            },
            "artifacts": _artifacts(tmp_path),
        }
    )
    return report


@pytest.mark.parametrize(
    ("mutation", "issue_fragment"),
    [
        (lambda report: report["config"].update(epochs=1), "config.epochs"),
        (lambda report: report["quality"].update(target=0.1), "quality.target"),
        (
            lambda report: report["quality"].update(direction="lower"),
            "quality.direction",
        ),
        (
            lambda report: report["metrics"].update(top1_accuracy=0.1),
            "does not satisfy canonical target",
        ),
        (
            lambda report: report["metrics"].update(train_and_eval_seconds=0.0),
            "primary metric 'train_and_eval_seconds' must be positive",
        ),
        (
            lambda report: report["measurement_protocol"].update(
                timing_scope="training without validation"
            ),
            "measurement_protocol.timing_scope",
        ),
        (lambda report: report.update(model="resnet50"), "model identity"),
        (lambda report: report.update(dataset="cifar10"), "dataset identity"),
        (
            lambda report: report["dataset"].update(sha256="sha256:" + "0" * 64),
            "dataset SHA-256",
        ),
        (lambda report: report.update(workload="other"), "workload identity"),
    ],
)
def test_resnet_canonical_max_mutations_fail_closed(tmp_path, mutation, issue_fragment):
    workload = load_registry()["resnet18-train"]
    report = _resnet_report(workload, tmp_path)
    assert evaluate_report_contract(workload, report)["status"] == "passed"

    mutation(report)
    result = evaluate_report_contract(workload, report)
    assert result["status"] == "failed"
    assert any(issue_fragment in issue for issue in result["issues"])


def test_nanogpt_protocol_and_timing_mutations_fail_closed(tmp_path):
    workload = load_registry()["nanogpt-prefill"]
    report = _nanogpt_prefill_report(workload, tmp_path)
    assert evaluate_report_contract(workload, report)["status"] == "passed"

    report["config"]["context_len"] = 16
    report["measurement_protocol"]["measured_runs"] = 3
    report["metrics"]["prefill_latency_samples_s"] = [0.01] * 3
    result = evaluate_report_contract(workload, report)
    assert result["status"] == "failed"
    assert any("config.context_len" in issue for issue in result["issues"])
    assert any(
        "measurement_protocol.measured_runs" in issue for issue in result["issues"]
    )
    assert any("canonical protocol requires 20" in issue for issue in result["issues"])


@pytest.mark.parametrize(
    ("mutation", "issue_fragment"),
    [
        (
            lambda report: report["config"]["split"].update(
                validation="random-20-percent"
            ),
            "config.split",
        ),
        (
            lambda report: report["config"].update(
                feature_recipe="rating-aggregate-leakage"
            ),
            "config.feature_recipe",
        ),
        (
            lambda report: report["quality"].update(metric_key="best_accuracy"),
            "quality.metric_key",
        ),
        (
            lambda report: report["quality"].update(override=True),
            "quality target override",
        ),
    ],
)
def test_dlrm_fixed_split_and_auc_contract_fail_closed(
    tmp_path, mutation, issue_fragment
):
    workload = load_registry()["micro-dlrm-train"]
    report = _dlrm_report(workload, tmp_path)
    assert evaluate_report_contract(workload, report)["status"] == "passed"

    mutation(report)
    result = evaluate_report_contract(workload, report)
    assert result["status"] == "failed"
    assert any(issue_fragment in issue for issue in result["issues"])


@pytest.mark.parametrize(
    ("mutation", "issue_fragment"),
    [
        (
            lambda report: report["quality"]["gates"]["worst_class_auroc"].update(
                value=0.5
            ),
            "does not satisfy canonical target",
        ),
        (
            lambda report: report["quality"]["gates"]["control_margin"].update(
                target=0.0
            ),
            "control_margin.target",
        ),
        (
            lambda report: report["quality"]["gates"].update(passed=False),
            "conjunction did not pass",
        ),
    ],
)
def test_anomaly_auxiliary_quality_gates_fail_closed(
    tmp_path, mutation, issue_fragment
):
    workload = load_registry()["anomaly-ae-train"]
    report = _anomaly_report(workload, tmp_path)
    assert evaluate_report_contract(workload, report)["status"] == "passed"

    mutation(report)
    result = evaluate_report_contract(workload, report)
    assert result["status"] == "failed"
    assert any(issue_fragment in issue for issue in result["issues"])


@pytest.mark.parametrize(
    ("mutation", "issue_fragment"),
    [
        (
            lambda report: report["model"].update(id="Qwen/Qwen3-0.6B"),
            "model identity",
        ),
        (
            lambda report: report["model"].update(revision="main"),
            "model revision",
        ),
        (
            lambda report: report["model"].update(n_params=135_000_000),
            "model parameter count",
        ),
        (
            lambda report: report["config"].update(requested_decode_tokens=4),
            "config.requested_decode_tokens",
        ),
        (lambda report: report["quality"].update(target=1), "quality.target"),
        (
            lambda report: report["quality_evaluation"]["gates"][
                "overall_perplexity"
            ].update(target=1000.0),
            "quality_evaluation.gates",
        ),
        (
            lambda report: report["quality_evaluation"]["result"].update(
                suite_sha256="sha256:" + "0" * 64
            ),
            "quality_evaluation.suite_sha256",
        ),
        (
            lambda report: report["quality_evaluation"]["result"].update(
                fixture_version="1.0.0"
            ),
            "quality_evaluation.fixture_version",
        ),
        (
            lambda report: report["quality_evaluation"]["result"].update(
                aggregation="case-mean-nll"
            ),
            "quality_evaluation.aggregation",
        ),
        (
            lambda report: report["quality_evaluation"]["result"]["case_results"].pop(),
            "case count",
        ),
        (
            lambda report: report["quality_evaluation"]["result"]["case_results"][
                0
            ].update(id="invented-easier-case"),
            "id does not match the packaged fixture",
        ),
        (
            lambda report: report["quality_evaluation"]["result"]["case_results"][
                0
            ].update(continuation_tokens=1),
            "token count does not match the pinned tokenizer contract",
        ),
        (
            lambda report: report["quality_evaluation"]["result"].update(
                worst_category_perplexity=1.0
            ),
            "worst_category_perplexity does not match",
        ),
        (
            lambda report: report["quality_evaluation"]["gates"][
                "worst_category_perplexity"
            ].update(met=False),
            "did not pass",
        ),
        (
            lambda report: report["measurement_protocol"].update(measured_runs=3),
            "measurement_protocol.measured_runs",
        ),
    ],
)
def test_slm_canonical_max_mutations_fail_closed(tmp_path, mutation, issue_fragment):
    workload = load_registry()["slm-decode"]
    report = _slm_report(workload, tmp_path)
    assert evaluate_report_contract(workload, report)["status"] == "passed"

    mutation(report)
    result = evaluate_report_contract(workload, report)
    assert result["status"] == "failed"
    assert any(issue_fragment in issue for issue in result["issues"])
