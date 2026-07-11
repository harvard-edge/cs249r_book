from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlperf.reference.cloud.micro_dlrm import MicroDLRMWhiteBox
from mlperf.registry import load_registry
from mlperf.runners import agent as agent_runners
from mlperf.runners.dlrm import run_distributed_max
from mlperf.runners.tiny import run_wake_vision_max, run_wake_vision_min


AGENT_MAX_RUNNERS = {
    "nano-rag-agent": agent_runners.run_rag_max,
    "nano-codegen-agent": agent_runners.run_codegen_max,
    "nano-react-agent": agent_runners.run_react_max,
    "nano-toolcall-agent": agent_runners.run_toolcall_max,
}

AGENT_MAX_ENVIRONMENT = (
    "MLPERF_EDU_RAG_MAX_PASSAGES",
    "MLPERF_EDU_CODEGEN_MAX_RETRIES",
    "MLPERF_EDU_REACT_MAX_STEPS",
    "MLPERF_EDU_TOOLCALL_MAX_QUERIES",
)


def _manifest(output_dir: Path, workload_id: str, profile: str = "max") -> dict:
    path = output_dir / f"{workload_id}_{profile}.provd.json"
    return json.loads(path.read_text())


@pytest.mark.parametrize("workload_id", AGENT_MAX_RUNNERS)
def test_agent_default_max_reports_match_native_contract(
    workload_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for variable in AGENT_MAX_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)

    workload = load_registry()[workload_id]
    execution = workload.raw["max_execution"]
    report = AGENT_MAX_RUNNERS[workload_id](workload, tmp_path)

    expected_config = {
        **execution["model_config"],
        **execution["run_config"],
    }
    declared_params = int(str(workload.raw["params"]).replace(",", ""))
    assert workload.scenario == "single_stream"
    assert report["scenario"] == workload.scenario
    assert _manifest(tmp_path, workload_id)["scenario"] == workload.scenario
    assert report["data_mode"] == execution["data_mode"]
    assert report["dataset"] == workload.dataset
    assert report["config"] == expected_config
    assert report["config"]["batch_size"] == 1
    assert execution["declared_dataset_used"] is True
    assert workload.dataset == "synthetic-deterministic-agent-prompts"
    assert report["metrics"]["n_params"] == declared_params


def test_toolcall_max_fails_closed_when_generated_calls_are_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from mlperf.reference.cloud.nano_toolcall_agent import NanoToolCallAgent

    def invalid_calls(self, input_ids, n_queries=10):
        del self, input_ids
        return {
            "queries": [
                {"query_idx": index, "call_valid": False} for index in range(n_queries)
            ],
            "total_generation_ms": 1.0,
            "total_classification_ms": 1.0,
            "total_dispatch_ms": 1.0,
            "total_ms": 3.0,
            "valid_calls": 0,
            "total_queries": n_queries,
            "queries_per_second": n_queries / 0.003,
        }

    monkeypatch.setattr(NanoToolCallAgent, "forward_with_timing", invalid_calls)
    workload = load_registry()["nano-toolcall-agent"]
    report = agent_runners.run_toolcall_max(workload, tmp_path)

    assert report["status"] == "functional_failed"
    assert report["metrics"]["functional_check_passed"] is False
    assert report["quality"]["functional_check"]["passed"] is False
    assert report["quality"]["functional_check"]["every_call_valid"] is False
    assert workload.raw["max_execution"]["functional_check_enforced"] is True
    assert (
        report["quality"]["functional_check"]["condition"]
        == workload.raw["functional_check"]["condition"]
    )


def test_distributed_max_reports_exact_payload_and_training_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from mlperf.reference.distributed import ddp_runner

    monkeypatch.setattr(
        ddp_runner,
        "run_ddp",
        lambda **_kwargs: {
            "final_loss": 0.5,
            "backward_with_allreduce_time_per_step_ms": 1.25,
            "n_params": 22_977,
            "gradient_payload_bytes_fp32": 91_908,
        },
    )
    monkeypatch.setattr(
        ddp_runner,
        "run_gradacc_baseline",
        lambda *_args, **_kwargs: {"final_loss": 0.5},
    )
    monkeypatch.delenv("MLPERF_EDU_DDP_MAX_WORLD_SIZE", raising=False)
    monkeypatch.delenv("MLPERF_EDU_DDP_WORLD_SIZE", raising=False)
    monkeypatch.delenv("MLPERF_EDU_DDP_MAX_STEPS", raising=False)
    monkeypatch.delenv("MLPERF_EDU_DDP_MAX_MICRO_BATCH", raising=False)
    monkeypatch.delenv("MLPERF_EDU_DDP_MAX_REL_LOSS_TARGET", raising=False)

    workload = load_registry()["micro-dlrm-distributed"]
    report = run_distributed_max(workload, tmp_path)
    model = MicroDLRMWhiteBox()
    n_params = sum(parameter.numel() for parameter in model.parameters())
    payload_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    assert n_params == 22_977
    assert payload_bytes == 91_908
    assert int(str(workload.raw["params"]).replace(",", "")) == n_params
    assert report["scenario"] == workload.scenario == "training"
    assert _manifest(tmp_path, workload.id)["scenario"] == "training"
    assert report["config"] == workload.raw["max_execution"]["run_config"]
    assert report["metrics"]["n_params"] == n_params
    assert report["metrics"]["gradient_payload_bytes_fp32"] == payload_bytes
    assert report["metrics"]["backward_with_allreduce_time_per_step_ms"] == 1.25
    assert "allreduce_time_per_step_ms" not in report["metrics"]


def test_wake_vision_default_max_is_honest_training_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for variable in (
        "MLPERF_EDU_DEVICE",
        "MLPERF_EDU_WAKE_MAX_BATCH_SIZE",
        "MLPERF_EDU_WAKE_MAX_BATCHES",
        "MLPERF_EDU_WAKE_MAX_VAL_BATCHES",
        "MLPERF_EDU_WAKE_MAX_LR",
    ):
        monkeypatch.delenv(variable, raising=False)

    workload = load_registry()["wake-vision-vww"]
    report = run_wake_vision_max(workload, tmp_path)

    assert report["scenario"] == workload.scenario == "training"
    assert _manifest(tmp_path, workload.id)["scenario"] == "training"
    assert report["data_mode"] == workload.raw["max_execution"]["data_mode"]
    assert report["dataset"] == workload.dataset
    assert report["config"] == workload.raw["max_execution"]["run_config"]
    assert report["metrics"]["n_params"] == 8_514
    assert int(str(workload.raw["params"]).replace(",", "")) == 8_514


def test_wake_vision_min_report_and_manifest_use_training_scenario(tmp_path: Path):
    workload = load_registry()["wake-vision-vww"]
    report = run_wake_vision_min(workload, tmp_path)

    assert report["scenario"] == workload.scenario == "training"
    assert _manifest(tmp_path, workload.id, profile="min")["scenario"] == "training"
