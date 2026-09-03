from __future__ import annotations

import json

import pytest
import torch

from mlperf.reference.cloud.nanogpt_decode import NanoGPTDecode
from mlperf.reference.cloud.nanogpt_prefill import (
    FIXED_PROMPT_SEED,
    NanoGPTPrefill,
    fixed_token_prompt,
)
from mlperf.registry import load_registry
from mlperf.runners.nanogpt import _aggregate_decode_results, run_decode_min


class _FakeNanoGPT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = {
            "vocab_size": 16,
            "n_embd": 8,
            "n_head": 2,
            "n_layer": 1,
            "max_seq_len": 32,
        }
        self.calls: list[dict[str, object]] = []
        self.returned_caches: list[object] = []

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        use_kv_cache: bool,
        past_key_values: object | None = None,
    ) -> tuple[torch.Tensor, object]:
        assert use_kv_cache is True
        token_id = len(self.calls) + 1
        logits = torch.zeros((*tokens.shape, self.config["vocab_size"]))
        logits[:, -1, token_id] = 1.0
        cache_length = int(tokens.shape[-1])
        if past_key_values is not None:
            cache_length += int(past_key_values[0][0].shape[-2])
        key = torch.zeros(
            tokens.shape[0],
            self.config["n_head"],
            cache_length,
            self.config["n_embd"] // self.config["n_head"],
        )
        cache = ((key, key.clone()),)
        self.calls.append(
            {
                "input_shape": tuple(tokens.shape),
                "past_key_values": past_key_values,
            }
        )
        self.returned_caches.append(cache)
        return logits, cache


def test_nanogpt_request_ttft_ends_at_first_token_then_itl_uses_cache() -> None:
    model = _FakeNanoGPT()

    result = NanoGPTDecode(model, prefill_ctx=4, decode_steps=4, batch_size=1).run(
        emit_sidecar=False
    )

    assert [call["input_shape"] for call in model.calls] == [
        (1, 4),
        (1, 1),
        (1, 1),
        (1, 1),
    ]
    assert model.calls[0]["past_key_values"] is None
    for index in range(1, len(model.calls)):
        assert model.calls[index]["past_key_values"] is model.returned_caches[index - 1]
    assert result["generated_token_ids"] == [[1, 2, 3, 4]]
    assert len(result["itl_samples_s"]) == 3
    assert result["request_ttft_s"] >= result["prefill_latency_s"]
    assert result["first_decode_latency_s"] == result["itl_samples_s"][0]
    assert result["ttft_s"] == result["request_ttft_s"]
    assert result["request_end_to_end_latency_s"] >= result["request_ttft_s"]


def test_nanogpt_inference_prompt_is_fixed_across_run_seeds() -> None:
    torch.manual_seed(0)
    first, first_digest = fixed_token_prompt(
        batch_size=1,
        context_len=1792,
        vocab_size=128,
        device=torch.device("cpu"),
    )
    torch.manual_seed(999)
    second, second_digest = fixed_token_prompt(
        batch_size=1,
        context_len=1792,
        vocab_size=128,
        device=torch.device("cpu"),
    )

    assert FIXED_PROMPT_SEED == 314159
    assert torch.equal(first, second)
    assert first_digest == second_digest
    assert first_digest == (
        "1d64ab92c0b6a2f941af0f26f61f41cd64fb55426b3154e69977eaebd46adfa1"
    )


def test_nanogpt_prefill_materializes_complete_kv_cache() -> None:
    model = _FakeNanoGPT()

    result = NanoGPTPrefill(model, context_len=4, batch_size=1).run(
        n_warmup=1,
        n_iter=3,
        emit_sidecar=False,
    )

    assert len(model.calls) == 4
    assert all(call["input_shape"] == (1, 4) for call in model.calls)
    assert all(call["past_key_values"] is None for call in model.calls)
    assert result["kv_cache_materialized"] is True
    assert result["kv_cache_bytes"] > 0
    assert len(result["prefill_latency_samples_s"]) == 3


def test_nanogpt_aggregate_rejects_incomplete_or_decode_only_ttft() -> None:
    valid = {
        "prefill_ctx": 4,
        "decode_steps": 4,
        "batch_size": 1,
        "prompt_seed": FIXED_PROMPT_SEED,
        "prompt_sha256": "fixed-prompt",
        "prefill_latency_s": 4.0,
        "first_decode_latency_s": 2.0,
        "request_ttft_s": 4.0,
        "request_end_to_end_latency_s": 11.0,
        "itl_samples_s": [2.0, 2.0, 2.0],
        "kv_cache_bytes": 128,
        "achieved_bw_gbps": 0.25,
    }
    aggregate = _aggregate_decode_results([valid, valid])
    assert aggregate["request_ttft_s"] == 4.0
    assert aggregate["first_decode_latency_s"] == 2.0
    assert aggregate["output_tokens_per_sec"] == 0.5

    prefill_excluding_ttft = {**valid, "request_ttft_s": 3.0}
    with pytest.raises(ValueError, match="include prompt prefill"):
        _aggregate_decode_results([prefill_excluding_ttft])

    shifted_first_decode = {**valid, "first_decode_latency_s": 1.0}
    with pytest.raises(ValueError, match="first subsequent-token ITL"):
        _aggregate_decode_results([shifted_first_decode])

    changed_prompt = {**valid, "prompt_sha256": "different-prompt"}
    with pytest.raises(ValueError, match="one fixed canonical prompt"):
        _aggregate_decode_results([valid, changed_prompt])

    missing_interval = {**valid, "itl_samples_s": [2.0, 2.0]}
    with pytest.raises(ValueError, match="one ITL sample"):
        _aggregate_decode_results([missing_interval])

    with pytest.raises(ValueError, match="at least two"):
        NanoGPTDecode(model=_FakeNanoGPT(), prefill_ctx=4, decode_steps=1)


def test_nanogpt_min_report_and_manifest_use_single_stream(tmp_path) -> None:
    workload = load_registry()["causal-language-modeling"]

    report = run_decode_min(workload, tmp_path)
    manifest = json.loads(
        (tmp_path / "causal-language-modeling_inference_decode_min.provd.json").read_text()
    )

    phase_contract = workload.raw["mode_contracts"]["inference"]["phases"]["decode"]
    assert phase_contract["scenario"] == "single_stream"
    assert report["scenario"] == "single_stream"
    assert report["measurement_mode"] == "sequential_microbenchmark"
    assert manifest["scenario"] == "single_stream"
    assert report["metrics"]["request_ttft_s"] >= report["metrics"]["prefill_latency_s"]
    assert (
        report["metrics"]["first_decode_latency_s"]
        == report["metrics"]["itl_samples_s"][0]
    )
