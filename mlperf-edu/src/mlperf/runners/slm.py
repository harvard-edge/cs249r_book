from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from importlib import resources
from pathlib import Path
from typing import Any

import torch

from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import configured_seed


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SLM_QUALITY_SCHEMA = "mlperf-edu-slm-quality/0.2"
SLM_QUALITY_FIXTURE_VERSION = "2.0.0"
SLM_QUALITY_AGGREGATION = "token-weighted-continuation-nll"
SLM_QUALITY_CATEGORY_GUARD = "maximum-category-perplexity"
SLM_QUALITY_MIN_CASES = 20
DEFAULT_MAX_PERPLEXITY = 7.0
DEFAULT_MAX_WORST_CATEGORY_PERPLEXITY = 24.0
MODEL_ALIASES = {
    "smollm2-135m": DEFAULT_MODEL_ID,
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
}
DEFAULT_PROMPT = "Explain why reproducible ML benchmarking matters in one sentence."


def run_decode_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic SLM decode smoke using a tiny Transformers model."""
    return run_decode(
        workload, output_dir, profile="min", tiny_local=True, quantization=None
    )


def run_decode_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run off-the-shelf SLM decode with a Hugging Face model by default."""
    tiny_local = os.environ.get("MLPERF_EDU_SLM_TINY", "0") == "1"
    return run_decode(
        workload, output_dir, profile="max", tiny_local=tiny_local, quantization=None
    )


def run_quantized_decode_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic quantized SLM decode smoke."""
    return run_decode(
        workload,
        output_dir,
        profile="min",
        tiny_local=True,
        quantization="dynamic-int8",
    )


def run_quantized_decode_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run off-the-shelf SLM decode with CPU dynamic int8 quantization."""
    tiny_local = os.environ.get("MLPERF_EDU_SLM_TINY", "0") == "1"
    return run_decode(
        workload,
        output_dir,
        profile="max",
        tiny_local=tiny_local,
        quantization="dynamic-int8",
    )


def run_batched_decode_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run deterministic batched SLM decode with a tiny local model."""
    batch_size = int(os.environ.get("MLPERF_EDU_SLM_BATCH_SIZE", "4"))
    return run_decode(
        workload,
        output_dir,
        profile="min",
        tiny_local=True,
        quantization=None,
        batch_size=batch_size,
    )


def run_batched_decode_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run off-the-shelf batched SLM decode for throughput studies."""
    tiny_local = os.environ.get("MLPERF_EDU_SLM_TINY", "0") == "1"
    batch_size = int(os.environ.get("MLPERF_EDU_SLM_BATCH_SIZE", "4"))
    return run_decode(
        workload,
        output_dir,
        profile="max",
        tiny_local=tiny_local,
        quantization=None,
        batch_size=batch_size,
    )


def run_long_context_decode_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run deterministic long-context SLM decode with a tiny local model."""
    context_tokens = int(os.environ.get("MLPERF_EDU_SLM_LONG_CONTEXT_TOKENS", "96"))
    return run_decode(
        workload,
        output_dir,
        profile="min",
        tiny_local=True,
        quantization=None,
        context_tokens=context_tokens,
        prompt_mode="long-context",
    )


def run_long_context_decode_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run long-context SLM decode for prefill/KV-cache scaling studies."""
    tiny_local = os.environ.get("MLPERF_EDU_SLM_TINY", "0") == "1"
    default_context = "96" if tiny_local else "512"
    context_tokens = int(
        os.environ.get("MLPERF_EDU_SLM_LONG_CONTEXT_TOKENS", default_context)
    )
    return run_decode(
        workload,
        output_dir,
        profile="max",
        tiny_local=tiny_local,
        quantization=None,
        context_tokens=context_tokens,
        prompt_mode="long-context",
    )


def run_decode(
    workload: Workload,
    output_dir: Path,
    *,
    profile: str,
    tiny_local: bool,
    quantization: str | None,
    batch_size: int = 1,
    context_tokens: int | None = None,
    prompt_mode: str = "single",
) -> dict[str, Any]:
    seed = configured_seed()
    torch.manual_seed(seed)
    scenario = str(workload.scenario or "").strip()
    if not scenario:
        raise ValueError(f"SLM workload {workload.id!r} must declare a scenario")
    device = select_device()
    max_context_tokens = context_tokens or int(
        os.environ.get("MLPERF_EDU_SLM_CONTEXT_TOKENS", "64")
    )
    prompt = os.environ.get("MLPERF_EDU_SLM_PROMPT", DEFAULT_PROMPT)
    if prompt_mode == "long-context":
        prompt = long_context_prompt(prompt, target_tokens=max_context_tokens)
    default_decode_tokens = "16" if profile == "max" else "8"
    decode_tokens = int(
        os.environ.get("MLPERF_EDU_SLM_DECODE_TOKENS", default_decode_tokens)
    )
    if decode_tokens < 2:
        raise ValueError(
            "SLM decode measurement requires at least two output tokens so that "
            "inter-token latency is defined"
        )
    target_tokens = int(
        os.environ.get(
            "MLPERF_EDU_SLM_TARGET_TOKENS", workload.quality_value or decode_tokens
        )
    )

    if tiny_local:
        model_bundle = build_tiny_model(device)
    else:
        model_bundle = load_hf_model(device)

    model = model_bundle["model"]
    tokenizer = model_bundle.get("tokenizer")
    source_model_dtype = model_parameter_dtype(model)
    reference_quality = None
    if quantization and tokenizer is not None:
        model.eval()
        reference_quality = evaluate_slm_quality(model, tokenizer, device)
    if quantization:
        model, device = apply_quantization(model, quantization)
        model_bundle["model"] = model
    model.eval()
    task_quality = (
        evaluate_slm_quality(model, tokenizer, device)
        if tokenizer is not None
        else None
    )

    prompts = prompt_batch(prompt, batch_size)
    input_ids, attention_mask, rendered_prompts = encode_prompts(
        prompts,
        tokenizer=tokenizer,
        device=device,
        max_context_tokens=max_context_tokens,
    )

    warning_filter = (
        suppress_stderr_lines_containing("qnnpack incorrectly ignores reduce_range")
        if quantization == "dynamic-int8"
        else nullcontext()
    )
    warmup_runs = int(os.environ.get("MLPERF_EDU_SLM_WARMUP_RUNS", "3"))
    measured_runs = int(os.environ.get("MLPERF_EDU_SLM_MEASURED_RUNS", "20"))
    if warmup_runs < 1 or measured_runs < 3:
        raise ValueError(
            "SLM max measurement requires >=1 warmup and >=3 measured runs"
        )
    with warning_filter:
        for _ in range(warmup_runs):
            _run_slm_request(
                model,
                input_ids,
                attention_mask,
                decode_tokens=decode_tokens,
            )
        measurements = [
            _run_slm_request(
                model,
                input_ids,
                attention_mask,
                decode_tokens=decode_tokens,
            )
            for _ in range(measured_runs)
        ]

    prefill_latencies = [float(item["prefill_latency_s"]) for item in measurements]
    ttft_latencies = [float(item["request_ttft_s"]) for item in measurements]
    request_latencies = [
        float(item["request_end_to_end_latency_s"]) for item in measurements
    ]
    generation_latencies = [
        sum(float(value) for value in item["itl_samples_s"]) for item in measurements
    ]
    itl_samples = [
        float(value) for item in measurements for value in item["itl_samples_s"]
    ]
    if not itl_samples or any(value <= 0 for value in itl_samples):
        raise ValueError(
            "SLM decode measurement produced invalid inter-token latency samples"
        )
    prefill_latency = statistics.median(prefill_latencies)
    request_ttft = statistics.median(ttft_latencies)
    request_latency = statistics.median(request_latencies)
    generation_latency = statistics.median(generation_latencies)
    median_itl = statistics.median(itl_samples)
    prefill = measurements[-1]["prefill"]
    generated = measurements[-1]["generated"]

    measured_batch_size = int(input_ids.shape[0])
    context_tokens = int(input_ids.shape[-1])
    generated_tokens = int(generated.shape[-1])
    total_context_tokens = context_tokens * measured_batch_size
    total_generated_tokens = generated_tokens * measured_batch_size
    decode_interval_tokens = max(0, generated_tokens - 1) * measured_batch_size
    output_ids = generated[0]
    output_text = decode_output(output_ids, tokenizer)
    declared_quality_limits = slm_quality_gate_limits(workload)
    quality_limit_environment = {
        "max_perplexity": "MLPERF_EDU_SLM_MAX_PERPLEXITY",
        "max_worst_category_perplexity": (
            "MLPERF_EDU_SLM_MAX_WORST_CATEGORY_PERPLEXITY"
        ),
        "max_quantized_nll_delta": "MLPERF_EDU_SLM_MAX_NLL_DELTA",
    }
    effective_quality_limits = {
        name: validate_slm_quality_limit(
            os.environ.get(environment_name, declared_quality_limits[name]),
            label=environment_name
            if environment_name in os.environ
            else f"quality_evaluation.{name}",
        )
        for name, environment_name in quality_limit_environment.items()
    }
    absolute_perplexity_limit = effective_quality_limits["max_perplexity"]
    worst_category_perplexity_limit = effective_quality_limits[
        "max_worst_category_perplexity"
    ]
    quantized_nll_delta_limit = effective_quality_limits["max_quantized_nll_delta"]
    task_quality_gates = slm_quality_gate_results(
        task_quality,
        max_perplexity=absolute_perplexity_limit,
        max_worst_category_perplexity=worst_category_perplexity_limit,
        reference_result=reference_quality,
        max_quantized_nll_delta=quantized_nll_delta_limit,
    )
    task_quality_met = task_quality_gates["passed"] is True
    target_met = generated_tokens >= target_tokens and (tiny_local or task_quality_met)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_{profile}_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_{profile}.provd.json").resolve()
    metadata_path = (output_dir / f"{workload.id}_{profile}_model.json").resolve()

    n_params = sum(p.numel() for p in model.parameters())
    model_state_bytes = state_dict_nbytes(model)
    model_metadata = {
        "schema": "mlperf-edu-model-metadata/0.1",
        "model_id": model_bundle["model_id"],
        "model_alias": model_bundle.get("model_alias"),
        "model_type": model_bundle["model_type"],
        "revision": model_bundle.get("revision"),
        "profile": profile,
        "tiny_local": tiny_local,
        "quantization": quantization,
        "batch_size": measured_batch_size,
        "prompt_mode": prompt_mode,
        "configured_context_tokens": max_context_tokens,
        "n_params": int(n_params),
        "model_state_bytes": int(model_state_bytes),
        "device": str(device),
        "prompt": rendered_prompts[0],
        "prompts": rendered_prompts,
        "generated_token_ids": [int(x) for x in output_ids.detach().cpu().tolist()],
        "generated_text": output_text[:1000],
        "config": model_bundle.get("config", {}),
        "task_quality": task_quality,
        "reference_task_quality": reference_quality,
        "model_asset_count": len(model_bundle.get("asset_files") or []),
        "model_asset_roles": sorted(
            {str(item.get("role")) for item in model_bundle.get("asset_files") or []}
        ),
    }
    metadata_path.write_text(
        json.dumps(model_metadata, indent=2, sort_keys=True) + "\n"
    )

    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "scenario": scenario,
        "profile": profile,
        "status": "passed" if target_met else "quality_failed",
        "backend": f"transformers-{device.type}"
        + (f"-{quantization}" if quantization else ""),
        "data_mode": data_mode(
            tiny_local=tiny_local,
            batch_size=measured_batch_size,
            prompt_mode=prompt_mode,
        ),
        "seed": seed,
        "config": {
            "requested_decode_tokens": decode_tokens,
            "target_generated_tokens": target_tokens,
            "max_context_tokens": max_context_tokens,
            "prompt_mode": prompt_mode,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "batch_size": measured_batch_size,
            "quantization": quantization,
            "tiny_local": tiny_local,
        },
        "model": {
            "id": model_bundle["model_id"],
            "alias": model_bundle.get("model_alias"),
            "type": model_bundle["model_type"],
            "revision": model_bundle.get("revision"),
            "n_params": int(n_params),
        },
        "metrics": {
            "batch_size": measured_batch_size,
            "requested_decode_tokens": decode_tokens,
            "target_generated_tokens": target_tokens,
            "configured_context_tokens": max_context_tokens,
            "context_tokens": context_tokens,
            "total_context_tokens": total_context_tokens,
            "generated_tokens": generated_tokens,
            "total_generated_tokens": total_generated_tokens,
            "prefill_latency_s": float(prefill_latency),
            "prefill_latency_p90_s": percentile(prefill_latencies, 0.90),
            "prefill_latency_p99_s": percentile(prefill_latencies, 0.99),
            "generation_latency_s": float(generation_latency),
            "generation_latency_p90_s": percentile(generation_latencies, 0.90),
            "generation_latency_p99_s": percentile(generation_latencies, 0.99),
            "prefill_latency_samples_s": prefill_latencies,
            "request_ttft_samples_s": ttft_latencies,
            "itl_samples_s": itl_samples,
            "request_end_to_end_samples_s": request_latencies,
            "request_end_to_end_latency_s": float(request_latency),
            "request_end_to_end_latency_p90_s": percentile(request_latencies, 0.90),
            "request_end_to_end_latency_p99_s": percentile(request_latencies, 0.99),
            "time_to_first_token_s": float(request_ttft),
            "time_to_first_token_p90_s": percentile(ttft_latencies, 0.90),
            "time_to_first_token_p99_s": percentile(ttft_latencies, 0.99),
            "inter_token_latency_s": float(median_itl),
            "inter_token_latency_p90_s": percentile(itl_samples, 0.90),
            "inter_token_latency_p99_s": percentile(itl_samples, 0.99),
            "decode_interval_tokens": decode_interval_tokens,
            "prefill_tokens_per_sec": float(total_context_tokens / prefill_latency)
            if prefill_latency
            else 0.0,
            "requests_per_sec": float(measured_batch_size / request_latency)
            if request_latency
            else 0.0,
            "output_tokens_per_sec": float(measured_batch_size / median_itl),
            "n_params": int(n_params),
            "model_state_bytes": int(model_state_bytes),
            "prompt_chars": len(rendered_prompts[0]),
            "logits_shape": list(prefill.logits.shape),
            "quality_mean_nll": task_quality.get("mean_nll") if task_quality else None,
            "quality_perplexity": task_quality.get("perplexity")
            if task_quality
            else None,
            "quality_worst_category_perplexity": task_quality.get(
                "worst_category_perplexity"
            )
            if task_quality
            else None,
            "quality_total_continuation_tokens": task_quality.get(
                "total_continuation_tokens"
            )
            if task_quality
            else None,
            "quality_nll_delta": (
                task_quality["mean_nll"] - reference_quality["mean_nll"]
                if task_quality and reference_quality
                else 0.0
                if task_quality
                else None
            ),
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "generated_tokens",
            "target": target_tokens,
            "direction": "higher",
            "quality_required": True,
            "target_met": target_met,
            "override": "MLPERF_EDU_SLM_TARGET_TOKENS" in os.environ,
            "note": "The serving gate requires the requested output length and bounded continuation perplexity, computed token-weighted overall and in the weakest category; quantized runs must also preserve NLL parity.",
        },
        "quality_evaluation": {
            "status": "passed" if (tiny_local or task_quality_met) else "failed",
            "suite": SLM_QUALITY_SCHEMA,
            "fixture_version": SLM_QUALITY_FIXTURE_VERSION,
            "cases": task_quality.get("cases") if task_quality else None,
            "categories": task_quality.get("categories") if task_quality else None,
            "aggregation": SLM_QUALITY_AGGREGATION,
            "category_guard": SLM_QUALITY_CATEGORY_GUARD,
            "gates": task_quality_gates,
            "max_quantized_nll_delta": quantized_nll_delta_limit,
            "contract": {
                "source": (
                    "workload.quality_evaluation"
                    if workload.raw.get("quality_evaluation") is not None
                    else "runner-defaults"
                ),
                "declared_limits": declared_quality_limits,
                "effective_limits": effective_quality_limits,
                "environment_overrides": sorted(
                    environment_name
                    for environment_name in quality_limit_environment.values()
                    if environment_name in os.environ
                ),
            },
            "result": task_quality,
            "reference_result": reference_quality,
        },
        "measurement_protocol": {
            **(workload.raw.get("measurement_protocol") or {}),
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "raw_sample_metrics": [
                "prefill_latency_samples_s",
                "request_ttft_samples_s",
                "itl_samples_s",
                "request_end_to_end_samples_s",
            ],
            "timing_scope": (
                "one cache-reusing greedy request path per fixed prompt batch; "
                "TTFT spans prompt prefill through the first output token, ITL "
                "samples time only subsequent cached-token steps, and end-to-end "
                "latency spans the complete request"
            ),
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
            "model_metadata": str(metadata_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    manifest_args: dict[str, Any] = {
        "workload": workload.id,
        "scenario": scenario,
        "division": "open",
        "hardware_fingerprint": detect_hardware(),
        "report": report,
        "report_path": report_path,
        "dataset_name": model_bundle["model_id"],
        "dataset_files": [metadata_path, slm_quality_suite_path()],
        "rng_seed": seed,
        "torch_state_bytes": torch.get_rng_state().numpy().tobytes(),
        "repo_root": find_project_root(),
    }
    asset_files = model_bundle.get("asset_files")
    if asset_files:
        manifest_args.update(
            {
                "weights_files": asset_files,
                "weights_name": model_bundle["model_id"],
                "weights_revision": model_bundle.get("revision"),
                "weights_n_params": int(n_params),
                "weights_dtype": source_model_dtype,
            }
        )
    manifest = build_provd(**manifest_args)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def _run_slm_request(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    decode_tokens: int,
) -> dict[str, Any]:
    """Run one fixed-length greedy request while reusing the prompt KV cache."""
    if decode_tokens < 2:
        raise ValueError(
            "SLM request requires at least two output tokens to measure inter-token latency"
        )
    synchronize_device(input_ids.device)
    request_start = time.perf_counter()
    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        )
    synchronize_device(input_ids.device)
    prefill_latency = time.perf_counter() - request_start

    past_key_values = getattr(prefill, "past_key_values", None)
    if past_key_values is None:
        raise ValueError("SLM model did not return a KV cache for greedy decode")
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    synchronize_device(input_ids.device)
    request_ttft = time.perf_counter() - request_start
    generated_tokens = [next_token]
    decode_attention_mask = attention_mask
    itl_samples: list[float] = []

    with torch.inference_mode():
        for _ in range(decode_tokens - 1):
            decode_attention_mask = torch.cat(
                (
                    decode_attention_mask,
                    torch.ones(
                        (decode_attention_mask.shape[0], 1),
                        dtype=decode_attention_mask.dtype,
                        device=decode_attention_mask.device,
                    ),
                ),
                dim=-1,
            )
            synchronize_device(input_ids.device)
            token_start = time.perf_counter()
            decode_output = model(
                input_ids=next_token,
                attention_mask=decode_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = getattr(decode_output, "past_key_values", None)
            if past_key_values is None:
                raise ValueError("SLM model stopped returning a KV cache during decode")
            next_token = decode_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            synchronize_device(input_ids.device)
            itl_samples.append(time.perf_counter() - token_start)
            generated_tokens.append(next_token)

    request_end_to_end_latency = time.perf_counter() - request_start
    return {
        "prefill_latency_s": prefill_latency,
        "request_ttft_s": request_ttft,
        "itl_samples_s": itl_samples,
        "request_end_to_end_latency_s": request_end_to_end_latency,
        "prefill": prefill,
        "generated": torch.cat(generated_tokens, dim=-1),
    }


def evaluate_slm_quality(
    model: torch.nn.Module, tokenizer: Any, device: torch.device
) -> dict[str, Any]:
    """Measure token-weighted continuation NLL through a cache-reusing path."""
    suite_path = slm_quality_suite_path()
    raw = suite_path.read_bytes()
    suite = load_slm_quality_suite(raw)
    case_results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case in suite["cases"]:
            prompt = str(case["prompt"])
            continuation = str(case["continuation"])
            case_nll, continuation_tokens = _continuation_nll_with_cache(
                model,
                tokenizer,
                device,
                prompt=prompt,
                continuation=continuation,
            )
            expected_tokens = suite["expected_continuation_tokens"][case["id"]]
            if continuation_tokens != expected_tokens:
                raise ValueError(
                    f"SLM quality case {case['id']!r} tokenized to "
                    f"{continuation_tokens} continuation tokens; the pinned fixture "
                    f"requires {expected_tokens}"
                )
            case_results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "mean_nll": case_nll,
                    "continuation_tokens": continuation_tokens,
                }
            )
    aggregate = aggregate_slm_quality_cases(case_results)
    return {
        "suite": suite["schema"],
        "fixture_version": suite["fixture_version"],
        "aggregation": suite["aggregation"]["primary"],
        "category_guard": suite["aggregation"]["guard"],
        "cases": len(case_results),
        "categories": len(suite["category_definitions"]),
        **aggregate,
        "case_results": case_results,
        "suite_sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
    }


def load_slm_quality_suite(raw: bytes) -> dict[str, Any]:
    """Parse and fail closed on the versioned, attributed SLM fixture."""
    try:
        suite = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SLM quality fixture is not valid UTF-8 JSON") from exc
    if not isinstance(suite, dict):
        raise ValueError("SLM quality fixture must be a JSON object")
    if suite.get("schema") != SLM_QUALITY_SCHEMA:
        raise ValueError(f"SLM quality fixture schema must be {SLM_QUALITY_SCHEMA!r}")
    if suite.get("fixture_version") != SLM_QUALITY_FIXTURE_VERSION:
        raise ValueError(
            f"SLM quality fixture version must be {SLM_QUALITY_FIXTURE_VERSION!r}"
        )

    attribution = suite.get("attribution")
    if not isinstance(attribution, dict) or any(
        not str(attribution.get(field) or "").strip()
        for field in ("creator", "license", "method")
    ):
        raise ValueError(
            "SLM quality fixture attribution must declare creator, license, and method"
        )

    aggregation = suite.get("aggregation")
    expected_aggregation = {
        "primary": SLM_QUALITY_AGGREGATION,
        "category": SLM_QUALITY_AGGREGATION,
        "guard": SLM_QUALITY_CATEGORY_GUARD,
    }
    if aggregation != expected_aggregation:
        raise ValueError(
            "SLM quality fixture must use token-weighted overall/category NLL "
            "and the maximum-category-perplexity guard"
        )

    category_definitions = suite.get("category_definitions")
    if not isinstance(category_definitions, dict) or len(category_definitions) < 4:
        raise ValueError("SLM quality fixture must define at least four categories")
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(description, str)
        or not description.strip()
        for name, description in category_definitions.items()
    ):
        raise ValueError("SLM quality fixture category definitions must be nonempty")

    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) < SLM_QUALITY_MIN_CASES:
        raise ValueError(
            f"SLM quality fixture must contain at least {SLM_QUALITY_MIN_CASES} cases"
        )
    case_ids: set[str] = set()
    category_counts = {str(category): 0 for category in category_definitions}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"SLM quality fixture case {index} must be an object")
        for field in ("id", "category", "source", "prompt", "continuation"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(
                    f"SLM quality fixture case {index} must declare nonempty {field}"
                )
        case_id = str(case["id"])
        if case_id in case_ids:
            raise ValueError(f"SLM quality fixture duplicates case id {case_id!r}")
        case_ids.add(case_id)
        category = str(case["category"])
        if category not in category_counts:
            raise ValueError(
                f"SLM quality fixture case {case_id!r} uses undeclared category {category!r}"
            )
        if not str(case["continuation"]).startswith(" "):
            raise ValueError(
                f"SLM quality fixture case {case_id!r} continuation must begin with a space"
            )
        category_counts[category] += 1
    expected_tokens = suite.get("expected_continuation_tokens")
    if not isinstance(expected_tokens, dict) or set(expected_tokens) != case_ids:
        raise ValueError(
            "SLM quality fixture expected_continuation_tokens must cover every case exactly"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in expected_tokens.values()
    ):
        raise ValueError(
            "SLM quality fixture expected continuation-token counts must be positive integers"
        )
    sparse_categories = sorted(
        category for category, count in category_counts.items() if count < 3
    )
    if sparse_categories:
        raise ValueError(
            "SLM quality fixture categories must each contain at least three cases: "
            f"{sparse_categories}"
        )
    return suite


def aggregate_slm_quality_cases(
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate continuation losses by token globally and within categories."""
    if not case_results:
        raise ValueError("SLM quality aggregation requires at least one case")
    category_rows: dict[str, list[dict[str, Any]]] = {}
    for index, result in enumerate(case_results):
        category = result.get("category")
        nll = result.get("mean_nll")
        token_count = result.get("continuation_tokens")
        if not isinstance(category, str) or not category:
            raise ValueError(f"SLM quality result {index} has no category")
        if (
            isinstance(nll, bool)
            or not isinstance(nll, (int, float))
            or not math.isfinite(float(nll))
            or float(nll) < 0
        ):
            raise ValueError(f"SLM quality result {index} has invalid NLL")
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 1
        ):
            raise ValueError(f"SLM quality result {index} has invalid token count")
        category_rows.setdefault(category, []).append(result)

    total_tokens = sum(int(result["continuation_tokens"]) for result in case_results)
    total_nll = math.fsum(
        float(result["mean_nll"]) * int(result["continuation_tokens"])
        for result in case_results
    )
    mean_nll = total_nll / total_tokens
    category_results: dict[str, dict[str, Any]] = {}
    for category in sorted(category_rows):
        rows = category_rows[category]
        category_tokens = sum(int(row["continuation_tokens"]) for row in rows)
        category_nll = (
            math.fsum(
                float(row["mean_nll"]) * int(row["continuation_tokens"]) for row in rows
            )
            / category_tokens
        )
        category_results[category] = {
            "cases": len(rows),
            "continuation_tokens": category_tokens,
            "mean_nll": category_nll,
            "perplexity": math.exp(min(category_nll, 50.0)),
        }
    worst_category = max(
        category_results,
        key=lambda category: (
            float(category_results[category]["mean_nll"]),
            category,
        ),
    )
    return {
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 50.0)),
        "total_continuation_tokens": total_tokens,
        "category_results": category_results,
        "worst_category": worst_category,
        "worst_category_nll": category_results[worst_category]["mean_nll"],
        "worst_category_perplexity": category_results[worst_category]["perplexity"],
    }


def slm_quality_gate_results(
    result: dict[str, Any] | None,
    *,
    max_perplexity: float,
    max_worst_category_perplexity: float,
    reference_result: dict[str, Any] | None = None,
    max_quantized_nll_delta: float = 0.1,
) -> dict[str, Any]:
    """Evaluate the conjunctive overall, category, and optional parity gates."""
    gates: dict[str, Any] = {}
    for name, metric_key, target in (
        ("overall_perplexity", "perplexity", max_perplexity),
        (
            "worst_category_perplexity",
            "worst_category_perplexity",
            max_worst_category_perplexity,
        ),
    ):
        value = result.get(metric_key) if isinstance(result, dict) else None
        met = bool(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) <= float(target)
        )
        gates[name] = {
            "metric_key": metric_key,
            "value": value,
            "target": float(target),
            "direction": "lower",
            "met": met,
        }
    if reference_result is not None:
        result_nll = result.get("mean_nll") if isinstance(result, dict) else None
        reference_nll = reference_result.get("mean_nll")
        delta = (
            float(result_nll) - float(reference_nll)
            if isinstance(result_nll, (int, float))
            and not isinstance(result_nll, bool)
            and isinstance(reference_nll, (int, float))
            and not isinstance(reference_nll, bool)
            else None
        )
        gates["quantized_nll_delta"] = {
            "metric_key": "mean_nll_delta",
            "value": delta,
            "target": float(max_quantized_nll_delta),
            "direction": "lower",
            "met": bool(
                delta is not None
                and math.isfinite(delta)
                and delta <= float(max_quantized_nll_delta)
            ),
        }
    gates["passed"] = all(
        isinstance(gate, dict) and gate.get("met") is True
        for name, gate in gates.items()
        if name != "passed"
    )
    return gates


def validate_slm_quality_limit(value: Any, *, label: str) -> float:
    """Return one finite, nonnegative SLM quality-gate limit."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return numeric


def slm_quality_gate_limits(workload: Workload) -> dict[str, float]:
    """Resolve quality ceilings from the selected workload variant contract."""
    evaluation = workload.raw.get("quality_evaluation")
    if evaluation is None:
        evaluation = {}
    if not isinstance(evaluation, dict):
        raise ValueError(
            f"SLM workload {workload.id!r} quality_evaluation must be a mapping"
        )
    expected_bindings = {
        "suite": SLM_QUALITY_SCHEMA,
        "fixture_version": SLM_QUALITY_FIXTURE_VERSION,
        "aggregation": SLM_QUALITY_AGGREGATION,
        "category_guard": SLM_QUALITY_CATEGORY_GUARD,
    }
    for field, expected in expected_bindings.items():
        if field in evaluation and evaluation[field] != expected:
            raise ValueError(
                f"SLM workload {workload.id!r} quality_evaluation.{field} "
                f"must be {expected!r}"
            )
    return {
        "max_perplexity": validate_slm_quality_limit(
            evaluation.get("maximum", DEFAULT_MAX_PERPLEXITY),
            label=f"{workload.id}.quality_evaluation.maximum",
        ),
        "max_worst_category_perplexity": validate_slm_quality_limit(
            evaluation.get(
                "worst_category_maximum", DEFAULT_MAX_WORST_CATEGORY_PERPLEXITY
            ),
            label=f"{workload.id}.quality_evaluation.worst_category_maximum",
        ),
        "max_quantized_nll_delta": validate_slm_quality_limit(
            evaluation.get("max_quantized_nll_delta", 0.1),
            label=f"{workload.id}.quality_evaluation.max_quantized_nll_delta",
        ),
    }


def _continuation_nll_with_cache(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    *,
    prompt: str,
    continuation: str,
) -> tuple[float, int]:
    """Score exact continuation tokens after one prompt prefill."""
    prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    continuation_inputs = tokenizer(
        continuation, return_tensors="pt", add_special_tokens=False
    )
    prompt_ids = prompt_inputs["input_ids"].to(device)
    prompt_attention_mask = prompt_inputs.get(
        "attention_mask", torch.ones_like(prompt_ids)
    ).to(device)
    continuation_ids = continuation_inputs["input_ids"].to(device)
    if prompt_ids.shape[0] != 1 or continuation_ids.shape[0] != 1:
        raise ValueError("SLM quality cases must tokenize to a single request")
    if prompt_ids.shape[-1] < 1:
        raise ValueError("SLM quality prompt produced no tokens")
    if continuation_ids.shape[-1] < 1:
        raise ValueError("SLM quality continuation produced no tokens")

    output = model(
        input_ids=prompt_ids,
        attention_mask=prompt_attention_mask,
        use_cache=True,
    )
    past_key_values = getattr(output, "past_key_values", None)
    if past_key_values is None:
        raise ValueError("SLM model did not return a KV cache for quality evaluation")
    logits = output.logits[:, -1, :]
    attention_mask = prompt_attention_mask
    token_losses: list[float] = []

    for index in range(int(continuation_ids.shape[-1])):
        target = continuation_ids[:, index]
        token_loss = torch.nn.functional.cross_entropy(logits.float(), target)
        token_losses.append(float(token_loss.item()))
        if index == continuation_ids.shape[-1] - 1:
            continue
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=-1,
        )
        output = model(
            input_ids=target[:, None],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = getattr(output, "past_key_values", None)
        if past_key_values is None:
            raise ValueError(
                "SLM model stopped returning a KV cache during quality evaluation"
            )
        logits = output.logits[:, -1, :]

    return statistics.fmean(token_losses), len(token_losses)


def slm_quality_suite_path() -> Path:
    return Path(
        str(resources.files("mlperf_edu").joinpath("slm_quality_prompts.json"))
    ).resolve()


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


def build_tiny_model(device: torch.device) -> dict[str, Any]:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=128,
        n_positions=128,
        n_ctx=128,
        n_embd=32,
        n_layer=1,
        n_head=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config).to(device)
    return {
        "model": model,
        "tokenizer": None,
        "model_id": "transformers:gpt2-tiny-random-local",
        "model_alias": "tiny-local",
        "model_type": "gpt2-random-config",
        "revision": "local-deterministic-config-v1",
        "pad_token_id": 0,
        "eos_token_id": 2,
        "config": config.to_dict(),
    }


def load_hf_model(device: torch.device) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    alias_or_id = os.environ.get("MLPERF_EDU_SLM_MODEL_ID", DEFAULT_MODEL_ID)
    model_id = resolve_model_id(alias_or_id)
    revision = os.environ.get("MLPERF_EDU_SLM_REVISION", DEFAULT_MODEL_REVISION)
    local_only = os.environ.get("MLPERF_EDU_SLM_LOCAL_ONLY", "0") == "1"
    with suppress_stderr_lines_containing(
        "Warning: You are sending unauthenticated requests to the HF Hub",
        "Loading weights:",
    ):
        snapshot_path = snapshot_model(
            model_id, revision=revision, local_only=local_only
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_path, local_files_only=True
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model.to(device)
    config = getattr(model, "config", None)
    resolved_revision = resolved_snapshot_revision(snapshot_path, fallback=revision)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_id": model_id,
        "model_alias": alias_or_id if alias_or_id != model_id else None,
        "model_type": getattr(config, "model_type", type(model).__name__),
        "revision": resolved_revision,
        "requested_revision": revision,
        "snapshot_path": str(snapshot_path),
        "asset_files": hf_model_asset_records(snapshot_path),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "config": config.to_dict() if config is not None else {},
    }


def apply_quantization(
    model: torch.nn.Module, quantization: str
) -> tuple[torch.nn.Module, torch.device]:
    if quantization != "dynamic-int8":
        raise ValueError(f"unsupported SLM quantization mode: {quantization}")
    cpu = torch.device("cpu")
    select_quantized_engine()
    model = model.to(cpu).float()
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized, cpu


@contextmanager
def suppress_stderr_lines_containing(*needles: str):
    """Suppress known noisy stderr lines while preserving any other stderr output."""
    sys.stderr.flush()
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), fd)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, fd)
            os.close(saved)
            tmp.seek(0)
            captured = tmp.read().decode("utf-8", errors="replace")
            for line in captured.splitlines():
                if not any(needle in line for needle in needles):
                    print(line, file=sys.stderr)


def select_quantized_engine() -> None:
    supported = set(torch.backends.quantized.supported_engines)
    current = torch.backends.quantized.engine
    if current and current != "none":
        return
    for engine in ("qnnpack", "fbgemm", "x86"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return
    raise RuntimeError(
        f"No supported quantized backend is available; supported={sorted(supported)}"
    )


def state_dict_nbytes(model: torch.nn.Module) -> int:
    total = 0
    for value in model.state_dict().values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return int(total)


def model_parameter_dtype(model: torch.nn.Module) -> str | None:
    dtypes = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
        }
    )
    if not dtypes:
        return None
    if len(dtypes) == 1:
        return dtypes[0]
    return ",".join(dtypes)


def encode_prompt(
    prompt: str,
    *,
    tokenizer: Any | None,
    device: torch.device,
    max_context_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    input_ids, attention_mask, rendered_prompts = encode_prompts(
        [prompt],
        tokenizer=tokenizer,
        device=device,
        max_context_tokens=max_context_tokens,
    )
    return input_ids, attention_mask, rendered_prompts[0]


def encode_prompts(
    prompts: list[str],
    *,
    tokenizer: Any | None,
    device: torch.device,
    max_context_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if tokenizer is None:
        encoded = []
        for prompt in prompts:
            token_ids = [max(3, min(127, (ord(ch) % 125) + 3)) for ch in prompt]
            encoded.append(token_ids[:max_context_tokens] or [1])
        max_len = max(len(token_ids) for token_ids in encoded)
        input_ids = torch.zeros(
            (len(encoded), max_len), dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for idx, token_ids in enumerate(encoded):
            input_ids[idx, -len(token_ids) :] = torch.tensor(
                token_ids, dtype=torch.long, device=device
            )
            attention_mask[idx, -len(token_ids) :] = 1
        return input_ids, attention_mask, prompts

    rendered_prompts = []
    for prompt in prompts:
        rendered_prompt = prompt
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered_prompt = prompt
        rendered_prompts.append(rendered_prompt)
    inputs = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_tokens,
        padding=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return input_ids, attention_mask, rendered_prompts


def prompt_batch(prompt: str, batch_size: int) -> list[str]:
    if batch_size < 1:
        raise ValueError("SLM batch size must be >= 1")
    if batch_size == 1:
        return [prompt]
    return [f"{prompt} [request {idx + 1}/{batch_size}]" for idx in range(batch_size)]


def long_context_prompt(prompt: str, *, target_tokens: int) -> str:
    if target_tokens < 1:
        raise ValueError("SLM context tokens must be >= 1")
    fragment = (
        f"{prompt} Context scaling note: benchmark reports prefill latency, "
        "KV-cache pressure, and decode throughput under a longer deterministic prompt. "
    )
    repeats = max(1, (target_tokens * 8) // max(1, len(fragment)) + 2)
    return (fragment * repeats).strip()


def data_mode(*, tiny_local: bool, batch_size: int, prompt_mode: str) -> str:
    mode = "synthetic-tokenized" if tiny_local else "local-prompt"
    if prompt_mode == "long-context":
        mode += "-long-context"
    if batch_size > 1:
        mode += "-batch"
    return mode


def decode_output(output_ids: torch.Tensor, tokenizer: Any | None) -> str:
    if tokenizer is None:
        return " ".join(str(int(x)) for x in output_ids.detach().cpu().tolist())
    return tokenizer.decode(
        output_ids.detach().cpu().tolist(), skip_special_tokens=True
    )


def select_device() -> torch.device:
    override = os.environ.get("MLPERF_EDU_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_id(alias_or_id: str) -> str:
    return MODEL_ALIASES.get(alias_or_id, alias_or_id)


def snapshot_model(
    model_id_or_alias: str | None = None,
    *,
    revision: str | None = None,
    local_only: bool = False,
) -> Path:
    from huggingface_hub import snapshot_download

    model_id = resolve_model_id(
        model_id_or_alias or os.environ.get("MLPERF_EDU_SLM_MODEL_ID", DEFAULT_MODEL_ID)
    )
    resolved_revision = revision or os.environ.get(
        "MLPERF_EDU_SLM_REVISION", DEFAULT_MODEL_REVISION
    )
    path = snapshot_download(
        repo_id=model_id, revision=resolved_revision, local_files_only=local_only
    )
    return Path(path)


def resolved_snapshot_revision(snapshot_path: Path, *, fallback: str) -> str:
    path = Path(snapshot_path)
    if path.name and path.parent.name == "snapshots":
        return path.name
    return fallback


def hf_model_asset_records(snapshot_path: Path) -> list[dict[str, Any]]:
    """Return content-critical HF snapshot files with archive-stable names."""
    root = Path(snapshot_path)
    if not root.is_dir():
        raise ValueError(f"HF model snapshot is missing or not a directory: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        logical_path = path.relative_to(root).as_posix()
        if _skip_hf_snapshot_asset(logical_path):
            continue
        records.append(
            {
                "path": path,
                "logical_path": logical_path,
                "role": hf_model_asset_role(logical_path),
            }
        )
    roles = {str(record["role"]) for record in records}
    missing = sorted({"config", "tokenizer", "weights"} - roles)
    if missing:
        raise ValueError(
            f"HF model snapshot {root} is missing required asset roles: {missing}"
        )
    return records


def _skip_hf_snapshot_asset(logical_path: str) -> bool:
    name = Path(logical_path).name
    return name in {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "LICENSE.txt",
        "LICENSE.md",
    }


def hf_model_asset_role(logical_path: str) -> str:
    name = Path(logical_path).name
    if name in {"config.json", "generation_config.json"}:
        return "config"
    if name in {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "added_tokens.json",
    }:
        return "tokenizer"
    if name.endswith(".safetensors.index.json") or name.endswith(".bin.index.json"):
        return "weights_index"
    if name.endswith((".safetensors", ".bin", ".pt", ".pth", ".gguf")):
        return "weights"
    if name.endswith(".json"):
        return "model_metadata"
    return "model_asset"
