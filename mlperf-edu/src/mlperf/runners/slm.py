from __future__ import annotations

import json
import os
import sys
import time
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch

from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_ALIASES = {
    "smollm2-135m": DEFAULT_MODEL_ID,
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
}
DEFAULT_PROMPT = "Explain why reproducible ML benchmarking matters in one sentence."


def run_decode_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic SLM decode smoke using a tiny Transformers model."""
    return run_decode(workload, output_dir, profile="min", tiny_local=True, quantization=None)


def run_decode_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run off-the-shelf SLM decode with a Hugging Face model by default."""
    tiny_local = os.environ.get("MLPERF_EDU_SLM_TINY", "0") == "1"
    return run_decode(workload, output_dir, profile="max", tiny_local=tiny_local, quantization=None)


def run_quantized_decode_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic quantized SLM decode smoke."""
    return run_decode(workload, output_dir, profile="min", tiny_local=True, quantization="dynamic-int8")


def run_quantized_decode_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run off-the-shelf SLM decode with CPU dynamic int8 quantization."""
    tiny_local = os.environ.get("MLPERF_EDU_SLM_TINY", "0") == "1"
    return run_decode(workload, output_dir, profile="max", tiny_local=tiny_local, quantization="dynamic-int8")


def run_batched_decode_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run deterministic batched SLM decode with a tiny local model."""
    batch_size = int(os.environ.get("MLPERF_EDU_SLM_BATCH_SIZE", "4"))
    return run_decode(workload, output_dir, profile="min", tiny_local=True, quantization=None, batch_size=batch_size)


def run_batched_decode_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run off-the-shelf batched SLM decode for throughput studies."""
    tiny_local = os.environ.get("MLPERF_EDU_SLM_TINY", "0") == "1"
    batch_size = int(os.environ.get("MLPERF_EDU_SLM_BATCH_SIZE", "4"))
    return run_decode(workload, output_dir, profile="max", tiny_local=tiny_local, quantization=None, batch_size=batch_size)


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
    context_tokens = int(os.environ.get("MLPERF_EDU_SLM_LONG_CONTEXT_TOKENS", default_context))
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
    seed = int(os.environ.get("MLPERF_EDU_SLM_SEED", "42"))
    torch.manual_seed(seed)
    device = select_device()
    max_context_tokens = context_tokens or int(os.environ.get("MLPERF_EDU_SLM_CONTEXT_TOKENS", "64"))
    prompt = os.environ.get("MLPERF_EDU_SLM_PROMPT", DEFAULT_PROMPT)
    if prompt_mode == "long-context":
        prompt = long_context_prompt(prompt, target_tokens=max_context_tokens)
    default_decode_tokens = "16" if profile == "max" else "8"
    decode_tokens = int(os.environ.get("MLPERF_EDU_SLM_DECODE_TOKENS", default_decode_tokens))
    target_tokens = int(os.environ.get("MLPERF_EDU_SLM_TARGET_TOKENS", workload.quality_value or decode_tokens))

    if tiny_local:
        model_bundle = build_tiny_model(device)
    else:
        model_bundle = load_hf_model(device)

    model = model_bundle["model"]
    tokenizer = model_bundle.get("tokenizer")
    if quantization:
        model, device = apply_quantization(model, quantization)
        model_bundle["model"] = model
    model.eval()

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
    with warning_filter:
        start_prefill = time.perf_counter()
        with torch.inference_mode():
            prefill = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        prefill_latency = time.perf_counter() - start_prefill

        start_generate = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=decode_tokens,
                do_sample=False,
                pad_token_id=model_bundle["pad_token_id"],
                eos_token_id=model_bundle["eos_token_id"],
                use_cache=True,
            )
        generation_latency = time.perf_counter() - start_generate

    measured_batch_size = int(input_ids.shape[0])
    context_tokens = int(input_ids.shape[-1])
    total_tokens = int(generated.shape[-1])
    generated_tokens = max(0, total_tokens - context_tokens)
    total_context_tokens = context_tokens * measured_batch_size
    total_generated_tokens = generated_tokens * measured_batch_size
    per_token_decode_latency = float(generation_latency / generated_tokens) if generated_tokens else 0.0
    output_ids = generated[0, context_tokens:]
    output_text = decode_output(output_ids, tokenizer)
    target_met = generated_tokens >= target_tokens

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
    }
    metadata_path.write_text(json.dumps(model_metadata, indent=2, sort_keys=True) + "\n")

    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": profile,
        "status": "passed" if target_met else "quality_failed",
        "backend": f"transformers-{device.type}" + (f"-{quantization}" if quantization else ""),
        "data_mode": data_mode(tiny_local=tiny_local, batch_size=measured_batch_size, prompt_mode=prompt_mode),
        "seed": seed,
        "config": {
            "requested_decode_tokens": decode_tokens,
            "target_generated_tokens": target_tokens,
            "max_context_tokens": max_context_tokens,
            "prompt_mode": prompt_mode,
            "batch_size": measured_batch_size,
            "quantization": quantization,
            "tiny_local": tiny_local,
        },
        "model": {
            "id": model_bundle["model_id"],
            "alias": model_bundle.get("model_alias"),
            "type": model_bundle["model_type"],
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
            "generation_latency_s": float(generation_latency),
            "time_to_first_token_s": float(prefill_latency + per_token_decode_latency),
            "inter_token_latency_s": per_token_decode_latency,
            "prefill_tokens_per_sec": float(total_context_tokens / prefill_latency) if prefill_latency else 0.0,
            "requests_per_sec": float(measured_batch_size / generation_latency) if generation_latency else 0.0,
            "output_tokens_per_sec": float(total_generated_tokens / generation_latency) if generation_latency else 0.0,
            "n_params": int(n_params),
            "model_state_bytes": int(model_state_bytes),
            "prompt_chars": len(rendered_prompts[0]),
            "logits_shape": list(prefill.logits.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": target_tokens,
            "direction": "higher",
            "quality_required": True,
            "target_met": target_met,
            "override": "MLPERF_EDU_SLM_TARGET_TOKENS" in os.environ,
            "note": "Functional serving check: each request must produce at least the target token count. Research scoring should compare latency, throughput, memory, batching, context length, and optional task prompts.",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
            "model_metadata": str(metadata_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    manifest = build_provd(
        workload=workload.id,
        scenario="single_stream",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name=model_bundle["model_id"],
        dataset_files=[metadata_path],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=find_project_root(),
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


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
        "pad_token_id": 0,
        "eos_token_id": 2,
        "config": config.to_dict(),
    }


def load_hf_model(device: torch.device) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    alias_or_id = os.environ.get("MLPERF_EDU_SLM_MODEL_ID", DEFAULT_MODEL_ID)
    model_id = resolve_model_id(alias_or_id)
    local_only = os.environ.get("MLPERF_EDU_SLM_LOCAL_ONLY", "0") == "1"
    with suppress_stderr_lines_containing(
        "Warning: You are sending unauthenticated requests to the HF Hub",
        "Loading weights:",
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    config = getattr(model, "config", None)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_id": model_id,
        "model_alias": alias_or_id if alias_or_id != model_id else None,
        "model_type": getattr(config, "model_type", type(model).__name__),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "config": config.to_dict() if config is not None else {},
    }


def apply_quantization(model: torch.nn.Module, quantization: str) -> tuple[torch.nn.Module, torch.device]:
    if quantization != "dynamic-int8":
        raise ValueError(f"unsupported SLM quantization mode: {quantization}")
    cpu = torch.device("cpu")
    select_quantized_engine()
    model = model.to(cpu).float()
    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
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
    raise RuntimeError(f"No supported quantized backend is available; supported={sorted(supported)}")


def state_dict_nbytes(model: torch.nn.Module) -> int:
    total = 0
    for value in model.state_dict().values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return int(total)


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
        input_ids = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for idx, token_ids in enumerate(encoded):
            input_ids[idx, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device=device)
            attention_mask[idx, : len(token_ids)] = 1
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
    return tokenizer.decode(output_ids.detach().cpu().tolist(), skip_special_tokens=True)


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


def snapshot_model(model_id_or_alias: str | None = None, *, local_only: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    model_id = resolve_model_id(model_id_or_alias or os.environ.get("MLPERF_EDU_SLM_MODEL_ID", DEFAULT_MODEL_ID))
    path = snapshot_download(repo_id=model_id, local_files_only=local_only)
    return Path(path)
