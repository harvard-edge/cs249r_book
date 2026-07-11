from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from mlperf.registry import load_registry
from mlperf.runners import extended


WORKLOAD_IDS = (
    "micro-bert-train",
    "micro-diffusion-train",
    "micro-gnn-train",
    "micro-lstm-train",
    "micro-rl-train",
    "nano-lora-finetune",
    "nano-moe-train",
    "nanogpt-decode-fp16-b16",
    "nanogpt-decode-fp32-b16",
    "nanogpt-decode-spec",
)


def call_runner(spec: str, workload, output_dir: Path):
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    return function(workload, output_dir)


@pytest.fixture(scope="module")
def extended_reports(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("extended-max-contracts")
    workloads = load_registry()
    reports = {}
    min_reports = {}
    manifests = {}
    for workload_id in WORKLOAD_IDS:
        workload = workloads[workload_id]
        runner = workload.raw["runner"]
        min_reports[workload_id] = call_runner(
            runner["min"], workload, output_root / workload_id / "min"
        )
        reports[workload_id] = call_runner(
            runner["max"], workload, output_root / workload_id / "max"
        )
        manifest_path = Path(reports[workload_id]["artifacts"]["provenance"])
        manifests[workload_id] = json.loads(manifest_path.read_text())
    return workloads, min_reports, reports, manifests


def report_param_count(report: dict) -> int:
    model = report.get("model") or {}
    return int(model["n_params"])


def min_param_count(report: dict) -> int:
    metrics = report.get("metrics") or {}
    return int(metrics.get("n_params", metrics.get("n_params_total")))


def test_extended_max_registry_and_reports_have_exact_parity(extended_reports):
    workloads, _min_reports, reports, manifests = extended_reports

    for workload_id in WORKLOAD_IDS:
        workload = workloads[workload_id]
        report = reports[workload_id]
        assert report["status"] == "passed", workload_id
        assert report["profile"] == "max", workload_id
        assert report["data_mode"] == "synthetic-micro-shard", workload_id
        assert report["config"] == workload.raw["max_execution"]["config"]
        assert report_param_count(report) == int(workload.raw["params"])
        assert report["metrics"]["max_micro_shard"] is True
        assert report["functional_check"]["passed"] is True
        assert report["functional_check"]["checks"]
        assert all(report["functional_check"]["checks"].values())
        assert report["quality"]["quality_required"] is False
        assert report["quality"]["target_met"] is None
        assert manifests[workload_id]["scenario"] == workload.scenario


def test_extended_max_is_materially_larger_than_min(extended_reports):
    _workloads, min_reports, reports, _manifests = extended_reports

    for workload_id in WORKLOAD_IDS:
        minimum = min_reports[workload_id]
        maximum = reports[workload_id]
        max_params = report_param_count(maximum)
        min_params = min_param_count(minimum)
        config = maximum["config"]
        metrics = maximum["metrics"]
        larger_model = max_params > min_params
        multiple_training_steps = int(metrics.get("train_steps", 0)) >= 3
        multiple_episodes = int(metrics.get("episodes", 0)) >= 4
        repeated_decode = int(metrics.get("measured_requests", 0)) >= 2
        assert (
            larger_model
            or multiple_training_steps
            or multiple_episodes
            or repeated_decode
        ), (
            workload_id,
            min_params,
            max_params,
            config,
        )

    for workload_id in (
        "nanogpt-decode-fp16-b16",
        "nanogpt-decode-fp32-b16",
    ):
        report = reports[workload_id]
        assert report["config"]["batch_size"] == 16
        assert report["config"]["decode_steps"] == 8
        assert report["metrics"]["total_output_tokens"] == 384

    speculative = reports["nanogpt-decode-spec"]
    assert speculative["config"]["gamma"] == 4
    assert speculative["config"]["decode_tokens"] == 8
    assert speculative["metrics"]["total_tokens_emitted"] == 16


def test_extended_max_dtype_and_lora_contracts_are_exact(extended_reports):
    _workloads, _min_reports, reports, _manifests = extended_reports

    fp16 = reports["nanogpt-decode-fp16-b16"]
    fp32 = reports["nanogpt-decode-fp32-b16"]
    assert fp16["model"]["dtype"] == "fp16"
    assert fp16["backend"] == "pytorch-cpu-fp16"
    assert fp32["model"]["dtype"] == "fp32"
    assert fp32["backend"] == "pytorch-cpu-fp32"

    lora = reports["nano-lora-finetune"]
    assert lora["config"]["rank"] == 8
    assert lora["config"]["alpha"] == 16
    assert lora["metrics"]["n_lora_adapters"] == 2
    assert lora["metrics"]["n_lora_trainable_params"] == 4096
    assert lora["metrics"]["base_grad_norm"] == 0.0
    assert lora["metrics"]["frozen_parameter_delta_l2"] == 0.0
    assert lora["metrics"]["lora_grad_norm"] > 0.0


def test_write_max_report_fails_closed_on_functional_check(tmp_path):
    workload = load_registry()["micro-bert-train"]
    report = extended.write_max_report(
        workload,
        tmp_path,
        root=Path(__file__).resolve().parents[1],
        seed=42,
        config={"train_steps": 4},
        model_metadata={"architecture": "test", "n_params": 1, "dtype": "float32"},
        metrics={"parameter_delta_l2": 0.0},
        functional_checks={"completed_steps": True, "parameters_updated": False},
        dataset_name="synthetic-test",
        quality_note="test fixture",
    )

    assert report["status"] == "quality_failed"
    assert report["functional_check"]["passed"] is False
    assert report["quality"]["quality_required"] is False

    with pytest.raises(TypeError, match="functional checks must be boolean"):
        extended.write_max_report(
            workload,
            tmp_path / "invalid",
            root=Path(__file__).resolve().parents[1],
            seed=42,
            config={},
            model_metadata={"architecture": "test", "n_params": 1},
            metrics={},
            functional_checks={"not_boolean": 1},
            dataset_name="synthetic-test",
            quality_note="test fixture",
        )
