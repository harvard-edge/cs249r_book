from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import mlperf.runners.slm as slm
from mlperf.registry import load_registry
from mlperf.runners.slm import (
    DEFAULT_MAX_PERPLEXITY,
    DEFAULT_MAX_WORST_CATEGORY_PERPLEXITY,
    SLM_QUALITY_AGGREGATION,
    SLM_QUALITY_CATEGORY_GUARD,
    SLM_QUALITY_FIXTURE_VERSION,
    SLM_QUALITY_SCHEMA,
    aggregate_slm_quality_cases,
    load_slm_quality_suite,
    slm_quality_gate_results,
    slm_quality_suite_path,
)


FIXTURE_SHA256 = "3d6d06b99dd92f1cf86fcde10f77b4db060397003bf654cc52c3148087ede556"


def _fixture() -> dict[str, object]:
    return json.loads(slm_quality_suite_path().read_bytes())


def test_slm_v2_fixture_is_balanced_attributed_and_content_addressed() -> None:
    raw = slm_quality_suite_path().read_bytes()
    fixture = load_slm_quality_suite(raw)

    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    assert fixture["schema"] == SLM_QUALITY_SCHEMA
    assert fixture["fixture_version"] == SLM_QUALITY_FIXTURE_VERSION
    assert fixture["aggregation"] == {
        "primary": SLM_QUALITY_AGGREGATION,
        "category": SLM_QUALITY_AGGREGATION,
        "guard": SLM_QUALITY_CATEGORY_GUARD,
    }
    cases = fixture["cases"]
    assert len(cases) == 28
    categories = fixture["category_definitions"]
    assert len(categories) == 7
    assert {
        category: sum(case["category"] == category for case in cases)
        for category in categories
    } == {category: 4 for category in categories}
    assert all(case["source"] == "MLPerf EDU project-authored" for case in cases)
    expected_tokens = fixture["expected_continuation_tokens"]
    assert set(expected_tokens) == {case["id"] for case in cases}
    assert sum(expected_tokens.values()) == 75


def test_slm_registry_contract_binds_fixture_aggregation_and_gates() -> None:
    workload = load_registry()["slm-decode"].raw
    quality = workload["quality_evaluation"]
    canonical = workload["canonical_max_contract"]["quality_evaluation"]

    assert workload["canonical_max_contract"]["model_n_params"] == 134_515_008
    assert quality["suite"] == SLM_QUALITY_SCHEMA
    assert quality["fixture_version"] == SLM_QUALITY_FIXTURE_VERSION
    assert quality["asset_sha256"] == FIXTURE_SHA256
    assert quality["cases"] == 28
    assert quality["categories"] == 7
    assert quality["aggregation"] == SLM_QUALITY_AGGREGATION
    assert quality["category_guard"] == SLM_QUALITY_CATEGORY_GUARD
    assert quality["maximum"] == DEFAULT_MAX_PERPLEXITY
    assert quality["worst_category_maximum"] == DEFAULT_MAX_WORST_CATEGORY_PERPLEXITY
    assert canonical == {
        "suite": SLM_QUALITY_SCHEMA,
        "fixture_version": SLM_QUALITY_FIXTURE_VERSION,
        "suite_sha256": f"sha256:{FIXTURE_SHA256}",
        "cases": 28,
        "categories": 7,
        "aggregation": SLM_QUALITY_AGGREGATION,
        "category_guard": SLM_QUALITY_CATEGORY_GUARD,
        "gates": {
            "overall_perplexity": {
                "metric_key": "perplexity",
                "target": DEFAULT_MAX_PERPLEXITY,
                "direction": "lower",
            },
            "worst_category_perplexity": {
                "metric_key": "worst_category_perplexity",
                "target": DEFAULT_MAX_WORST_CATEGORY_PERPLEXITY,
                "direction": "lower",
            },
        },
        "max_quantized_nll_delta": 0.1,
    }


def test_slm_aggregation_weights_tokens_instead_of_cases() -> None:
    aggregate = aggregate_slm_quality_cases(
        [
            {
                "id": "long-easy",
                "category": "easy",
                "mean_nll": 1.0,
                "continuation_tokens": 9,
            },
            {
                "id": "short-hard",
                "category": "hard",
                "mean_nll": 5.0,
                "continuation_tokens": 1,
            },
        ]
    )

    assert aggregate["mean_nll"] == pytest.approx(1.4)
    assert aggregate["mean_nll"] != pytest.approx(3.0)
    assert aggregate["total_continuation_tokens"] == 10
    assert aggregate["worst_category"] == "hard"
    assert aggregate["worst_category_perplexity"] == pytest.approx(math.exp(5.0))


def test_slm_worst_category_gate_prevents_easy_cases_from_masking_weakness() -> None:
    aggregate = aggregate_slm_quality_cases(
        [
            {
                "id": "easy",
                "category": "easy",
                "mean_nll": math.log(2.0),
                "continuation_tokens": 95,
            },
            {
                "id": "weak",
                "category": "weak",
                "mean_nll": math.log(30.0),
                "continuation_tokens": 5,
            },
        ]
    )
    gates = slm_quality_gate_results(
        aggregate,
        max_perplexity=7.0,
        max_worst_category_perplexity=24.0,
    )

    assert aggregate["perplexity"] < 7.0
    assert gates["overall_perplexity"]["met"] is True
    assert gates["worst_category_perplexity"]["met"] is False
    assert gates["passed"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda fixture: fixture.update(schema="mlperf-edu-slm-quality/0.1"), "schema"),
        (lambda fixture: fixture.pop("attribution"), "attribution"),
        (
            lambda fixture: fixture["aggregation"].update(primary="case-mean-nll"),
            "token-weighted",
        ),
        (lambda fixture: fixture.update(cases=fixture["cases"][:19]), "20 cases"),
        (
            lambda fixture: fixture["cases"][1].update(id=fixture["cases"][0]["id"]),
            "duplicates case id",
        ),
        (
            lambda fixture: fixture["cases"][0].update(category="undeclared"),
            "undeclared category",
        ),
        (
            lambda fixture: fixture["cases"][0].update(continuation="no-space"),
            "begin with a space",
        ),
        (
            lambda fixture: fixture["expected_continuation_tokens"].pop(
                fixture["cases"][0]["id"]
            ),
            "cover every case exactly",
        ),
        (
            lambda fixture: fixture["expected_continuation_tokens"].update(
                {fixture["cases"][0]["id"]: 0}
            ),
            "positive integers",
        ),
    ],
)
def test_slm_fixture_mutations_fail_closed(mutation, message) -> None:
    fixture = copy.deepcopy(_fixture())
    mutation(fixture)

    with pytest.raises(ValueError, match=message):
        load_slm_quality_suite(json.dumps(fixture).encode())


def test_slm_aggregation_rejects_nonfinite_or_zero_token_cases() -> None:
    with pytest.raises(ValueError, match="invalid NLL"):
        aggregate_slm_quality_cases(
            [
                {
                    "id": "bad",
                    "category": "bad",
                    "mean_nll": float("nan"),
                    "continuation_tokens": 1,
                }
            ]
        )
    with pytest.raises(ValueError, match="invalid token count"):
        aggregate_slm_quality_cases(
            [
                {
                    "id": "bad",
                    "category": "bad",
                    "mean_nll": 1.0,
                    "continuation_tokens": 0,
                }
            ]
        )


def _quality_result(
    *, mean_nll: float, perplexity: float, worst_category_perplexity: float
) -> dict[str, object]:
    return {
        "suite": SLM_QUALITY_SCHEMA,
        "fixture_version": SLM_QUALITY_FIXTURE_VERSION,
        "aggregation": SLM_QUALITY_AGGREGATION,
        "category_guard": SLM_QUALITY_CATEGORY_GUARD,
        "cases": 28,
        "categories": 7,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "worst_category": "benchmarking",
        "worst_category_perplexity": worst_category_perplexity,
        "total_continuation_tokens": 75,
    }


def _quality_results_for_gate(
    *, quantized: bool, failed_gate: str
) -> list[dict[str, object]]:
    passing = _quality_result(
        mean_nll=1.0,
        perplexity=math.exp(1.0),
        worst_category_perplexity=4.0,
    )
    if failed_gate == "overall_perplexity":
        task = _quality_result(
            mean_nll=math.log(8.0),
            perplexity=8.0,
            worst_category_perplexity=8.0,
        )
    elif failed_gate == "worst_category_perplexity":
        task = _quality_result(
            mean_nll=math.log(6.0),
            perplexity=6.0,
            worst_category_perplexity=30.0,
        )
    elif failed_gate == "quantized_nll_delta":
        assert quantized
        task = _quality_result(
            mean_nll=1.30,
            perplexity=math.exp(1.30),
            worst_category_perplexity=4.0,
        )
    else:  # pragma: no cover - the parameter matrix below owns this vocabulary
        raise AssertionError(f"unknown failed gate: {failed_gate}")
    return [passing, task] if quantized else [task]


def _install_mock_real_slm(
    monkeypatch: pytest.MonkeyPatch,
    quality_results: list[dict[str, object]],
    *,
    asset_files: list[dict[str, object]] | None = None,
    build_kwargs_sink: list[dict[str, object]] | None = None,
) -> None:
    model = torch.nn.Linear(1, 1)
    remaining_quality_results = iter(quality_results)

    monkeypatch.setenv("MLPERF_EDU_DEVICE", "cpu")
    monkeypatch.setenv("MLPERF_EDU_SLM_WARMUP_RUNS", "1")
    monkeypatch.setenv("MLPERF_EDU_SLM_MEASURED_RUNS", "3")
    for name in (
        "MLPERF_EDU_SLM_TINY",
        "MLPERF_EDU_SLM_TARGET_TOKENS",
        "MLPERF_EDU_SLM_MAX_PERPLEXITY",
        "MLPERF_EDU_SLM_MAX_WORST_CATEGORY_PERPLEXITY",
        "MLPERF_EDU_SLM_MAX_NLL_DELTA",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(
        slm,
        "load_hf_model",
        lambda device: {
            "model": model,
            "tokenizer": object(),
            "model_id": slm.DEFAULT_MODEL_ID,
            "model_alias": "smollm2-135m",
            "model_type": "mock-causal-lm",
            "revision": slm.DEFAULT_MODEL_REVISION,
            "config": {"model_type": "mock-causal-lm"},
            "asset_files": asset_files or [],
        },
    )
    monkeypatch.setattr(
        slm,
        "evaluate_slm_quality",
        lambda model, tokenizer, device: next(remaining_quality_results),
    )
    monkeypatch.setattr(
        slm,
        "apply_quantization",
        lambda model, quantization: (model, torch.device("cpu")),
    )

    def fake_encode_prompts(
        prompts: list[str],
        *,
        tokenizer: object,
        device: torch.device,
        max_context_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        del tokenizer, max_context_tokens
        input_ids = torch.ones((len(prompts), 4), dtype=torch.long, device=device)
        return input_ids, torch.ones_like(input_ids), prompts

    def fake_request(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        decode_tokens: int,
    ) -> dict[str, object]:
        del model, attention_mask
        batch_size = int(input_ids.shape[0])
        return {
            "prefill_latency_s": 0.005,
            "request_ttft_s": 0.006,
            "itl_samples_s": [0.001] * (decode_tokens - 1),
            "request_end_to_end_latency_s": 0.021,
            "prefill": SimpleNamespace(
                logits=torch.zeros((batch_size, input_ids.shape[-1], 8))
            ),
            "generated": torch.ones(
                (batch_size, decode_tokens), dtype=torch.long, device=input_ids.device
            ),
        }

    class FakeManifest:
        def __init__(self, scenario: str) -> None:
            self.scenario = scenario

        def to_dict(self) -> dict[str, str]:
            return {"schema": "mlperf-provd/0.1", "scenario": self.scenario}

    monkeypatch.setattr(slm, "encode_prompts", fake_encode_prompts)
    monkeypatch.setattr(slm, "_run_slm_request", fake_request)
    monkeypatch.setattr(slm, "decode_output", lambda output_ids, tokenizer: "mock")
    monkeypatch.setattr(
        slm, "suppress_stderr_lines_containing", lambda *needles: nullcontext()
    )
    monkeypatch.setattr(slm, "detect_hardware", lambda: {"machine_class": "test"})

    def fake_build_provd(**kwargs):
        if build_kwargs_sink is not None:
            build_kwargs_sink.append(kwargs)
        return FakeManifest(str(kwargs["scenario"]))

    monkeypatch.setattr(
        slm,
        "build_provd",
        fake_build_provd,
    )


@pytest.mark.parametrize(
    ("workload_id", "runner_name", "failed_gate", "expected_scenario"),
    [
        ("slm-decode", "run_decode_max", "overall_perplexity", "single_stream"),
        (
            "slm-decode",
            "run_decode_max",
            "worst_category_perplexity",
            "single_stream",
        ),
        (
            "slm-quantized-decode",
            "run_quantized_decode_max",
            "overall_perplexity",
            "single_stream",
        ),
        (
            "slm-quantized-decode",
            "run_quantized_decode_max",
            "worst_category_perplexity",
            "single_stream",
        ),
        (
            "slm-quantized-decode",
            "run_quantized_decode_max",
            "quantized_nll_delta",
            "single_stream",
        ),
        (
            "slm-batched-decode",
            "run_batched_decode_max",
            "overall_perplexity",
            "offline",
        ),
        (
            "slm-batched-decode",
            "run_batched_decode_max",
            "worst_category_perplexity",
            "offline",
        ),
        (
            "slm-long-context-decode",
            "run_long_context_decode_max",
            "overall_perplexity",
            "single_stream",
        ),
        (
            "slm-long-context-decode",
            "run_long_context_decode_max",
            "worst_category_perplexity",
            "single_stream",
        ),
    ],
)
def test_real_model_slm_max_fails_closed_and_binds_declared_scenario(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    workload_id: str,
    runner_name: str,
    failed_gate: str,
    expected_scenario: str,
) -> None:
    workload = load_registry()[workload_id]
    quality_results = _quality_results_for_gate(
        quantized=workload_id == "slm-quantized-decode",
        failed_gate=failed_gate,
    )
    _install_mock_real_slm(monkeypatch, quality_results)

    report = getattr(slm, runner_name)(workload, tmp_path)
    manifest = json.loads((tmp_path / f"{workload_id}_max.provd.json").read_text())

    assert report["config"]["tiny_local"] is False
    assert report["status"] == "quality_failed"
    assert report["quality"]["target_met"] is False
    assert report["quality_evaluation"]["status"] == "failed"
    assert report["quality_evaluation"]["gates"][failed_gate]["met"] is False
    assert report["quality_evaluation"]["contract"]["effective_limits"] == (
        slm.slm_quality_gate_limits(workload)
    )
    assert report["quality_evaluation"]["contract"]["environment_overrides"] == []
    assert workload.scenario == expected_scenario
    assert report["scenario"] == workload.scenario
    assert manifest["scenario"] == workload.scenario


def test_slm_registry_scenarios_match_native_execution_modes() -> None:
    workloads = load_registry()

    assert {
        workload_id: workloads[workload_id].scenario
        for workload_id in (
            "slm-decode",
            "slm-quantized-decode",
            "slm-batched-decode",
            "slm-long-context-decode",
        )
    } == {
        "slm-decode": "single_stream",
        "slm-quantized-decode": "single_stream",
        "slm-batched-decode": "offline",
        "slm-long-context-decode": "single_stream",
    }


def test_hf_snapshot_asset_records_require_config_tokenizer_and_weights(tmp_path):
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n")
    (snapshot / "tokenizer.json").write_text("{}\n")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "README.md").write_text("not execution-critical\n")

    records = slm.hf_model_asset_records(snapshot)

    assert {(record["logical_path"], record["role"]) for record in records} == {
        ("config.json", "config"),
        ("tokenizer.json", "tokenizer"),
        ("model.safetensors", "weights"),
    }
    assert slm.resolved_snapshot_revision(snapshot, fallback="main") == "abc123"


def test_real_model_slm_manifest_binds_resolved_model_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    asset_root = tmp_path / "hf"
    asset_root.mkdir()
    config = asset_root / "config.json"
    tokenizer = asset_root / "tokenizer.json"
    weights = asset_root / "model.safetensors"
    config.write_text("{}\n")
    tokenizer.write_text("{}\n")
    weights.write_bytes(b"weights")
    asset_files = [
        {"path": config, "logical_path": "config.json", "role": "config"},
        {"path": tokenizer, "logical_path": "tokenizer.json", "role": "tokenizer"},
        {"path": weights, "logical_path": "model.safetensors", "role": "weights"},
    ]
    build_kwargs: list[dict[str, object]] = []
    _install_mock_real_slm(
        monkeypatch,
        [
            _quality_result(
                mean_nll=1.0,
                perplexity=math.exp(1.0),
                worst_category_perplexity=4.0,
            )
        ],
        asset_files=asset_files,
        build_kwargs_sink=build_kwargs,
    )

    report = slm.run_decode_max(load_registry()["slm-decode"], tmp_path)

    assert report["status"] == "passed"
    assert report["model"]["n_params"] == 2
    assert build_kwargs
    manifest_kwargs = build_kwargs[0]
    assert manifest_kwargs["weights_files"] == asset_files
    assert manifest_kwargs["weights_name"] == slm.DEFAULT_MODEL_ID
    assert manifest_kwargs["weights_revision"] == slm.DEFAULT_MODEL_REVISION
    assert manifest_kwargs["weights_n_params"] == 2
    assert manifest_kwargs["weights_dtype"] == "float32"


def test_quantized_variant_uses_calibrated_limits_without_changing_baseline() -> None:
    workloads = load_registry()
    quantized_workload = workloads["slm-quantized-decode"]
    baseline_limits = slm.slm_quality_gate_limits(workloads["slm-decode"])
    quantized_limits = slm.slm_quality_gate_limits(quantized_workload)

    assert baseline_limits == {
        "max_perplexity": 7.0,
        "max_worst_category_perplexity": 24.0,
        "max_quantized_nll_delta": 0.1,
    }
    assert quantized_limits == {
        "max_perplexity": 7.0,
        "max_worst_category_perplexity": 25.0,
        "max_quantized_nll_delta": 0.25,
    }

    calibration = quantized_workload.raw["calibration_observation"]
    quantized_mean_nll = math.log(calibration["quantized_perplexity"])
    reference_mean_nll = quantized_mean_nll - calibration["quantized_nll_delta"]
    reference = _quality_result(
        mean_nll=reference_mean_nll,
        perplexity=calibration["reference_perplexity"],
        worst_category_perplexity=calibration["reference_worst_category_perplexity"],
    )
    quantized = _quality_result(
        mean_nll=quantized_mean_nll,
        perplexity=calibration["quantized_perplexity"],
        worst_category_perplexity=calibration["quantized_worst_category_perplexity"],
    )
    gates = slm_quality_gate_results(
        quantized,
        max_perplexity=quantized_limits["max_perplexity"],
        max_worst_category_perplexity=quantized_limits["max_worst_category_perplexity"],
        reference_result=reference,
        max_quantized_nll_delta=quantized_limits["max_quantized_nll_delta"],
    )

    assert gates["passed"] is True
    assert gates["overall_perplexity"]["met"] is True
    assert gates["worst_category_perplexity"]["met"] is True
    assert gates["quantized_nll_delta"]["met"] is True
