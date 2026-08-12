from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from mlperf.assets import ensure_tinyshakespeare, sha256_file
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd, verify_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    TrainingProgress,
    configured_seed,
    select_torch_device,
    synchronize_device,
    training_measurement_protocol,
)


def run_causal_language_modeling_min(
    workload: Workload,
    output_dir: Path,
    *,
    mode: str = "training",
    phase: str | None = None,
) -> dict[str, Any]:
    return _dispatch_causal_language_modeling(
        workload, output_dir, profile="min", mode=mode, phase=phase
    )


def run_causal_language_modeling_max(
    workload: Workload,
    output_dir: Path,
    *,
    mode: str = "training",
    phase: str | None = None,
) -> dict[str, Any]:
    return _dispatch_causal_language_modeling(
        workload, output_dir, profile="max", mode=mode, phase=phase
    )


def _dispatch_causal_language_modeling(
    workload: Workload,
    output_dir: Path,
    *,
    profile: str,
    mode: str,
    phase: str | None,
) -> dict[str, Any]:
    if mode == "training":
        report = (
            run_min(workload, output_dir)
            if profile == "min"
            else run_max(workload, output_dir)
        )
        report["mode"] = "training"
        report["phase"] = None
        return report
    if mode != "inference":
        raise ValueError(f"unsupported causal-language-modeling mode: {mode}")
    resolved_phase = phase or "full"
    if resolved_phase == "prefill":
        report = (
            run_prefill_min(workload, output_dir)
            if profile == "min"
            else run_prefill_max(workload, output_dir)
        )
    elif resolved_phase in {"full", "decode"}:
        report = (
            run_decode_min(workload, output_dir, phase=resolved_phase)
            if profile == "min"
            else run_decode_max(workload, output_dir, phase=resolved_phase)
        )
    else:
        raise ValueError(
            f"unsupported causal-language-modeling inference phase: {resolved_phase}"
        )
    report["mode"] = "inference"
    report["phase"] = resolved_phase
    phase_contract = (
        ((workload.raw.get("mode_contracts") or {}).get("inference") or {}).get(
            "phases"
        )
        or {}
    ).get(resolved_phase) or {}
    if phase_contract.get("measurement_protocol"):
        report["measurement_protocol"] = {
            **phase_contract["measurement_protocol"],
            **{
                key: value
                for key, value in (report.get("measurement_protocol") or {}).items()
                if key in {"warmup_runs", "measured_runs"}
            },
        }
    return report


def _artifact_stem(workload: Workload, mode: str, phase: str | None = None) -> str:
    suffix = f"_{phase}" if phase else ""
    return f"{workload.id}_{mode}{suffix}"


def run_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step NanoGPT training smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed(default=1337)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    model = NanoGPTWhiteBox(
        vocab_size=128,
        n_embd=32,
        n_head=4,
        n_layer=1,
        max_seq_len=32,
    ).to(device)
    model.train()

    batch_size = 2
    seq_len = 16
    inputs = torch.randint(
        0, 96, (batch_size, seq_len), dtype=torch.long, device=device
    )
    targets = torch.roll(inputs, shifts=-1, dims=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    logits, loss = model(inputs, targets=targets)
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    n_params = sum(p.numel() for p in model.parameters())
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(workload, "training")
    report_path = (output_dir / f"{stem}_min_report.json").resolve()
    manifest_path = (output_dir / f"{stem}_min.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "min",
        "mode": "training",
        "phase": None,
        "status": "passed",
        "backend": "pytorch-cpu",
        "data_mode": "synthetic-deterministic",
        "seed": seed,
        "metrics": {
            "loss": float(loss.item()),
            "duration_seconds": float(duration),
            "tokens": int(batch_size * seq_len),
            "tokens_per_second": float((batch_size * seq_len) / duration),
            "n_params": int(n_params),
            "logits_shape": list(logits.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "min profile validates execution only; max profile owns quality checks.",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
        },
    }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "training",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name="synthetic-deterministic",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    return report


def run_prefill_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic NanoGPT prefill inference smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.nanogpt_prefill import NanoGPTPrefill
    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    context_len = _env_int("MLPERF_EDU_PREFILL_MIN_CONTEXT", 32)
    batch_size = _env_int("MLPERF_EDU_PREFILL_MIN_BATCH", 1)
    n_warmup = _env_int("MLPERF_EDU_PREFILL_MIN_WARMUP", 1)
    n_iter = _env_int("MLPERF_EDU_PREFILL_MIN_ITER", 3)

    model = NanoGPTWhiteBox(
        vocab_size=128,
        n_embd=32,
        n_head=4,
        n_layer=1,
        max_seq_len=max(64, context_len),
    ).to(device)
    result = NanoGPTPrefill(model, context_len=context_len, batch_size=batch_size).run(
        n_warmup=n_warmup,
        n_iter=n_iter,
        emit_sidecar=False,
    )
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(workload, "inference", "prefill")
    report_path = (output_dir / f"{stem}_min_report.json").resolve()
    manifest_path = (output_dir / f"{stem}_min.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "min",
        "mode": "inference",
        "phase": "prefill",
        "status": "passed",
        "backend": "pytorch-cpu",
        "data_mode": "synthetic-deterministic",
        "seed": seed,
        "metrics": {
            **result,
            "n_params": int(n_params),
            "n_iter": n_iter,
            "n_warmup": n_warmup,
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "min profile validates deterministic prefill execution only.",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario="offline",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name="synthetic-deterministic-prompt",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_prefill_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run checkpoint-backed NanoGPT prefill inference."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.nanogpt_prefill import NanoGPTPrefill

    seed = configured_seed()
    torch.manual_seed(seed)
    device = select_torch_device()
    context_len = _env_int("MLPERF_EDU_PREFILL_MAX_CONTEXT", 256)
    batch_size = _env_int("MLPERF_EDU_PREFILL_MAX_BATCH", 1)
    n_warmup = _env_int("MLPERF_EDU_PREFILL_MAX_WARMUP", 3)
    n_iter = _env_int("MLPERF_EDU_PREFILL_MAX_ITER", 20)
    model, checkpoint_path, checkpoint_lineage = _load_max_nanogpt_model(
        output_dir, device, context_len
    )

    result = NanoGPTPrefill(model, context_len=context_len, batch_size=batch_size).run(
        n_warmup=n_warmup,
        n_iter=n_iter,
        emit_sidecar=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    target_met = bool(result.get("prefill_tokens_per_sec", 0) > 0)
    phase_contract = workload.raw["mode_contracts"]["inference"]["phases"]["prefill"]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(workload, "inference", "prefill")
    report_path = (output_dir / f"{stem}_max_report.json").resolve()
    manifest_path = (output_dir / f"{stem}_max.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "mode": "inference",
        "phase": "prefill",
        "status": "passed" if target_met else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "model": "nanogpt-shakespeare-char",
        "data_mode": "checkpoint-backed",
        "dataset": "prompt-suite-local",
        "scenario": phase_contract["scenario"],
        "seed": seed,
        "config": dict(phase_contract["config"]),
        "prompt": {
            "prompt_seed": result["prompt_seed"],
            "prompt_sha256": result["prompt_sha256"],
            "kv_cache_materialized": result["kv_cache_materialized"],
        },
        "metrics": {
            **result,
            "n_params": int(n_params),
        },
        "quality": {
            "metric": "prefill_tokens_per_sec",
            "metric_key": "prefill_tokens_per_sec",
            "target": 0.0,
            "direction": "higher",
            "quality_required": True,
            "target_met": target_met,
            "note": "The functional gate requires a quality-approved checkpoint and positive measured prefill throughput.",
        },
        "measurement_protocol": dict(phase_contract["measurement_protocol"]),
        "checkpoint_provenance": checkpoint_lineage,
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
            "checkpoint": str(checkpoint_path),
            "source_training_report": checkpoint_lineage["source_report_path"],
            "source_training_provenance": checkpoint_lineage["source_manifest_path"],
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario="offline",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        weights_path=checkpoint_path,
        weights_n_params=n_params,
        weights_dtype="float32",
        dataset_name="trained-nanogpt-checkpoint",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_decode_min(
    workload: Workload, output_dir: Path, *, phase: str = "decode"
) -> dict[str, Any]:
    """Run a deterministic NanoGPT KV-cache decode smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.nanogpt_decode import NanoGPTDecode
    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    prefill_ctx = _env_int("MLPERF_EDU_DECODE_MIN_PREFILL_CTX", 16)
    decode_steps = _env_int("MLPERF_EDU_DECODE_MIN_STEPS", 4)
    batch_size = _env_int("MLPERF_EDU_DECODE_MIN_BATCH", 1)

    model = NanoGPTWhiteBox(
        vocab_size=128,
        n_embd=32,
        n_head=4,
        n_layer=1,
        max_seq_len=max(64, prefill_ctx + decode_steps),
    ).to(device)
    result = NanoGPTDecode(
        model,
        prefill_ctx=prefill_ctx,
        decode_steps=decode_steps,
        batch_size=batch_size,
    ).run(emit_sidecar=False)
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(workload, "inference", phase)
    report_path = (output_dir / f"{stem}_min_report.json").resolve()
    manifest_path = (output_dir / f"{stem}_min.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "min",
        "mode": "inference",
        "phase": phase,
        "status": "passed",
        "backend": "pytorch-cpu",
        "data_mode": "synthetic-deterministic",
        "scenario": "single_stream",
        "measurement_mode": "sequential_microbenchmark",
        "seed": seed,
        "metrics": {
            **result,
            "n_params": int(n_params),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "min profile validates deterministic KV-cache decode execution only.",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
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
        dataset_name="synthetic-deterministic-prompt",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_decode_max(
    workload: Workload, output_dir: Path, *, phase: str = "decode"
) -> dict[str, Any]:
    """Run checkpoint-backed NanoGPT KV-cache decode inference."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.nanogpt_decode import NanoGPTDecode

    seed = configured_seed()
    torch.manual_seed(seed)
    device = select_torch_device()
    prefill_ctx = _env_int("MLPERF_EDU_DECODE_MAX_PREFILL_CTX", 192)
    decode_steps = _env_int("MLPERF_EDU_DECODE_MAX_STEPS", 64)
    batch_size = _env_int("MLPERF_EDU_DECODE_MAX_BATCH", 1)
    warmup_runs = _env_int("MLPERF_EDU_DECODE_MAX_WARMUPS", 3)
    repetitions = _env_int("MLPERF_EDU_DECODE_MAX_REPETITIONS", 20)
    model, checkpoint_path, checkpoint_lineage = _load_max_nanogpt_model(
        output_dir, device, prefill_ctx + decode_steps
    )

    decode = NanoGPTDecode(
        model,
        prefill_ctx=prefill_ctx,
        decode_steps=decode_steps,
        batch_size=batch_size,
    )
    for _ in range(warmup_runs):
        decode.run(emit_sidecar=False)
    results = [decode.run(emit_sidecar=False) for _ in range(repetitions)]
    result = _aggregate_decode_results(results)
    n_params = sum(p.numel() for p in model.parameters())
    target_met = bool(
        result.get("decode_steps") == decode_steps
        and result.get("output_tokens_per_sec", 0) > 0
    )
    phase_contract = workload.raw["mode_contracts"]["inference"]["phases"][phase]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(workload, "inference", phase)
    report_path = (output_dir / f"{stem}_max_report.json").resolve()
    manifest_path = (output_dir / f"{stem}_max.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "mode": "inference",
        "phase": phase,
        "status": "passed" if target_met else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "model": "nanogpt-shakespeare-char",
        "data_mode": "checkpoint-backed",
        "dataset": "prompt-suite-local",
        "scenario": phase_contract["scenario"],
        "measurement_mode": "sequential_microbenchmark",
        "seed": seed,
        "config": dict(phase_contract["config"]),
        "prompt": {
            "prompt_seed": result["prompt_seed"],
            "prompt_sha256": result["prompt_sha256"],
        },
        "metrics": {
            **result,
            "n_params": int(n_params),
        },
        "quality": {
            "metric": "decode_steps",
            "metric_key": "decode_steps",
            "target": decode_steps,
            "direction": "equal",
            "quality_required": True,
            "target_met": target_met,
            "note": "The functional gate requires a quality-approved checkpoint, the configured decode length, and positive throughput.",
        },
        "measurement_protocol": dict(phase_contract["measurement_protocol"]),
        "checkpoint_provenance": checkpoint_lineage,
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
            "checkpoint": str(checkpoint_path),
            "source_training_report": checkpoint_lineage["source_report_path"],
            "source_training_provenance": checkpoint_lineage["source_manifest_path"],
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
        weights_path=checkpoint_path,
        weights_n_params=n_params,
        weights_dtype="float32",
        dataset_name="trained-nanogpt-checkpoint",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


class _TextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.tokens[idx : idx + self.seq_len],
            self.tokens[idx + 1 : idx + self.seq_len + 1],
        )


class _NonOverlappingTextDataset(Dataset):
    """Deterministic disjoint contexts for representative quality evaluation."""

    def __init__(self, tokens: torch.Tensor, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        self.total_target_tokens = max(0, len(tokens) - 1)

    def __len__(self) -> int:
        return self.total_target_tokens // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        return (
            self.tokens[start : start + self.seq_len],
            self.tokens[start + 1 : start + self.seq_len + 1],
        )


def run_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run the TinyShakespeare NanoGPT max-profile quality target."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed(default=1337)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    asset = ensure_tinyshakespeare(download=True)
    batch_size = _env_int("MLPERF_EDU_MAX_BATCH_SIZE", 64)
    seq_len = _env_int("MLPERF_EDU_MAX_SEQ_LEN", 256)
    max_iters = _env_int("MLPERF_EDU_MAX_ITERS", 5000)
    eval_interval = _env_int("MLPERF_EDU_MAX_EVAL_INTERVAL", 250)
    eval_iters = _env_int("MLPERF_EDU_MAX_EVAL_ITERS", 200)
    lr = _env_float("MLPERF_EDU_MAX_LR", 1e-3)
    min_lr = _env_float("MLPERF_EDU_MAX_MIN_LR", 1e-4)
    warmup_iters = _env_int("MLPERF_EDU_MAX_WARMUP_ITERS", 100)
    beta2 = _env_float("MLPERF_EDU_MAX_BETA2", 0.99)
    model_size = os.environ.get("MLPERF_EDU_MAX_MODEL_SIZE", "base")

    device = select_torch_device()
    model_kwargs = _max_model_kwargs(model_size, seq_len)
    model = NanoGPTWhiteBox(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_tokens, val_tokens = _tinyshakespeare_token_tensors(asset.root)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, beta2),
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    eval_iterations: list[int] = []
    iteration_times: list[float] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    progress = TrainingProgress(workload.id, max_iters + 1, unit="iter")
    synchronize_device(device)
    start = time.perf_counter()
    for iteration in range(max_iters + 1):
        iteration_start = time.perf_counter()
        learning_rate = _canonical_nanogpt_lr(
            iteration,
            learning_rate=lr,
            min_lr=min_lr,
            warmup_iters=warmup_iters,
            decay_iters=max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if iteration % eval_interval == 0:
            train_eval, val_eval = _estimate_canonical_losses(
                model,
                train_tokens,
                val_tokens,
                batch_size=batch_size,
                seq_len=seq_len,
                eval_iters=eval_iters,
                device=device,
            )
            train_losses.append(train_eval)
            val_losses.append(val_eval)
            eval_iterations.append(iteration)
            if val_eval < best_val_loss:
                best_val_loss = val_eval
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
        inputs, targets = _random_nanogpt_batch(
            train_tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets=targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        iteration_times.append(time.perf_counter() - iteration_start)
        progress.update(
            iteration + 1,
            train_loss=train_losses[-1] if train_losses else float("nan"),
            val_loss=val_losses[-1] if val_losses else float("nan"),
            best_val=best_val_loss,
        )

    synchronize_device(device)
    duration = time.perf_counter() - start
    progress.close(f"best validation loss {best_val_loss:.4f}")
    target = _env_float(
        "MLPERF_EDU_MAX_QUALITY_TARGET", float(workload.quality_value or 1.4697)
    )
    target_met = best_val_loss <= target

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(workload, "training")
    report_path = (output_dir / f"{stem}_max_report.json").resolve()
    manifest_path = (output_dir / f"{stem}_max.provd.json").resolve()
    checkpoint_path = (output_dir / f"{stem}_max_checkpoint.pt").resolve()
    torch.save(best_state or model.state_dict(), checkpoint_path)

    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "mode": "training",
        "phase": None,
        "status": "passed" if target_met else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "model": "nanogpt-shakespeare-char",
        "data_mode": "real",
        "dataset": {
            "name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
        },
        "seed": seed,
        "measurement_protocol": training_measurement_protocol(workload),
        "config": {
            "random_seed": seed,
            "vocab_size": model_kwargs["vocab_size"],
            "n_layer": model_kwargs["n_layer"],
            "n_head": model_kwargs["n_head"],
            "n_embd": model_kwargs["n_embd"],
            "dropout": model_kwargs.get("dropout", 0.0),
            "bias": model_kwargs.get("bias", True),
            "batch_size": batch_size,
            "block_size": seq_len,
            "max_iters": max_iters,
            "eval_interval": eval_interval,
            "eval_iters": eval_iters,
            "learning_rate": lr,
            "min_lr": min_lr,
            "warmup_iters": warmup_iters,
            "beta2": beta2,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
        },
        "metrics": {
            "final_train_eval_loss": float(train_losses[-1]),
            "final_val_eval_loss": float(val_losses[-1]),
            "best_val_loss": float(best_val_loss),
            "cross_entropy_loss": float(best_val_loss),
            "duration_seconds": float(duration),
            "train_and_eval_seconds": float(duration),
            "tokens": int((max_iters + 1) * batch_size * seq_len),
            "tokens_per_second": float(
                ((max_iters + 1) * batch_size * seq_len) / duration
            )
            if duration > 0
            else 0.0,
            "n_params": int(n_params),
            "iteration_times": iteration_times,
            "eval_iterations": eval_iterations,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "cross_entropy_loss",
            "target": target,
            "direction": "lower",
            "target_met": target_met,
            "quality_required": True,
            "override": "MLPERF_EDU_MAX_QUALITY_TARGET" in os.environ,
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
            "checkpoint": str(checkpoint_path),
        },
    }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "training",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        weights_path=checkpoint_path,
        weights_n_params=n_params,
        weights_dtype="float32",
        dataset_name=asset.name,
        dataset_files=list(asset.files),
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def _max_model_kwargs(model_size: str, seq_len: int) -> dict[str, Any]:
    if model_size == "tiny":
        return {
            "vocab_size": 128,
            "n_embd": 32,
            "n_head": 4,
            "n_layer": 1,
            "max_seq_len": max(32, seq_len),
        }
    if model_size != "base":
        raise ValueError("MLPERF_EDU_MAX_MODEL_SIZE must be 'base' or 'tiny'")
    return {
        "vocab_size": 65,
        "n_embd": 384,
        "n_head": 6,
        "n_layer": 6,
        "max_seq_len": max(256, seq_len),
        "dropout": 0.2,
        "bias": True,
    }


def _load_max_nanogpt_model(
    output_dir: Path,
    device: torch.device,
    required_seq_len: int,
) -> tuple[torch.nn.Module, Path, dict[str, Any]]:
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    checkpoint = (
        Path(
            os.environ.get(
                "MLPERF_EDU_NANOGPT_CHECKPOINT",
                output_dir / "causal-language-modeling_training_max_checkpoint.pt",
            )
        )
        .expanduser()
        .resolve()
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"NanoGPT max checkpoint not found at {checkpoint}. "
            "Run `mlperf run --workload causal-language-modeling --mode training --profile max --output-dir <same-dir>` first, "
            "or set MLPERF_EDU_NANOGPT_CHECKPOINT."
        )

    source_report_path = (
        Path(
            os.environ.get(
                "MLPERF_EDU_NANOGPT_TRAIN_REPORT",
                checkpoint.parent / "causal-language-modeling_training_max_report.json",
            )
        )
        .expanduser()
        .resolve()
    )
    source_manifest_path = (
        Path(
            os.environ.get(
                "MLPERF_EDU_NANOGPT_TRAIN_MANIFEST",
                checkpoint.parent / "causal-language-modeling_training_max.provd.json",
            )
        )
        .expanduser()
        .resolve()
    )
    missing = [
        str(path)
        for path in (source_report_path, source_manifest_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "NanoGPT inference requires the quality-approved training report and "
            f"provenance manifest alongside the checkpoint; missing {missing}"
        )

    verification = verify_provd(source_manifest_path, repo_root=root)
    if not verification.all_ok:
        failed = [name for name, ok, _detail in verification.checks if not ok]
        raise ValueError(
            f"NanoGPT source training provenance failed verification: {failed}"
        )
    source_report = json.loads(source_report_path.read_text())
    source_quality = source_report.get("quality") or {}
    if (
        source_report.get("workload") != "causal-language-modeling"
        or source_report.get("mode") != "training"
        or source_report.get("profile") != "max"
        or source_report.get("status") != "passed"
        or source_report.get("data_mode") != "real"
        or source_quality.get("quality_required") is not True
        or source_quality.get("target_met") is not True
    ):
        raise ValueError(
            "NanoGPT source checkpoint does not have a passing real-data max training report"
        )

    checkpoint_sha256 = f"sha256:{sha256_file(checkpoint)}"
    source_manifest = json.loads(source_manifest_path.read_text())
    bound_checkpoint_sha256 = (
        (source_manifest.get("leaves") or {}).get("weights") or {}
    ).get("sha256")
    if bound_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError(
            "NanoGPT checkpoint SHA-256 does not match the verified source training manifest"
        )

    model_size = os.environ.get("MLPERF_EDU_MAX_MODEL_SIZE", "base")
    model = NanoGPTWhiteBox(**_max_model_kwargs(model_size, required_seq_len)).to(
        device
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    metrics = source_report.get("metrics") or {}
    lineage = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "source_workload": "causal-language-modeling",
        "source_report_path": str(source_report_path),
        "source_report_sha256": f"sha256:{sha256_file(source_report_path)}",
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": f"sha256:{sha256_file(source_manifest_path)}",
        "source_manifest_verified": True,
        "source_seed": source_report.get("seed"),
        "source_quality_metric": source_quality.get("metric"),
        "source_quality_value": metrics.get(
            source_quality.get("metric_key") or "cross_entropy_loss"
        ),
        "source_quality_target": source_quality.get("target"),
        "source_quality_target_met": True,
    }
    return model, checkpoint, lineage


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _aggregate_decode_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("decode measurement requires at least one repetition")
    first = results[0]
    itl_samples = [
        float(value) for result in results for value in result.get("itl_samples_s", [])
    ]
    ttft_samples = [float(result["request_ttft_s"]) for result in results]
    first_decode_samples = [
        float(result["first_decode_latency_s"]) for result in results
    ]
    prefill_samples = [float(result["prefill_latency_s"]) for result in results]
    request_end_to_end_samples = [
        float(result["request_end_to_end_latency_s"]) for result in results
    ]
    prompt_identities = {
        (result.get("prompt_seed"), result.get("prompt_sha256")) for result in results
    }
    if len(prompt_identities) != 1:
        raise ValueError("decode repetitions must use one fixed canonical prompt")
    expected_itl_samples = int(first["decode_steps"]) - 1
    if any(
        len(result.get("itl_samples_s", [])) != expected_itl_samples
        for result in results
    ):
        raise ValueError(
            "decode measurement did not retain one ITL sample per subsequent token"
        )
    if not itl_samples or any(value <= 0 for value in itl_samples):
        raise ValueError(
            "decode measurement produced invalid inter-token latency samples"
        )
    if any(
        result["first_decode_latency_s"] != result["itl_samples_s"][0]
        for result in results
    ):
        raise ValueError(
            "first-decode latency must be the first subsequent-token ITL sample"
        )
    if any(ttft < prefill for ttft, prefill in zip(ttft_samples, prefill_samples)):
        raise ValueError(
            "request TTFT must include prompt prefill and first-token selection"
        )
    if any(
        float(result["request_end_to_end_latency_s"])
        < float(result["request_ttft_s"])
        + sum(float(value) for value in result["itl_samples_s"])
        for result in results
    ):
        raise ValueError(
            "request end-to-end latency must span TTFT and every subsequent-token interval"
        )
    median_itl = statistics.median(itl_samples)
    return {
        "phase": "decode",
        "prefill_ctx": int(first["prefill_ctx"]),
        "decode_steps": int(first["decode_steps"]),
        "batch_size": int(first["batch_size"]),
        "prompt_seed": int(first["prompt_seed"]),
        "prompt_sha256": str(first["prompt_sha256"]),
        "prefill_warm_s": statistics.median(prefill_samples),
        "prefill_latency_s": statistics.median(prefill_samples),
        "prefill_warm_p90_s": _percentile(prefill_samples, 0.90),
        "prefill_warm_p99_s": _percentile(prefill_samples, 0.99),
        "prefill_latency_samples_s": prefill_samples,
        "first_decode_latency_s": statistics.median(first_decode_samples),
        "first_decode_latency_p90_s": _percentile(first_decode_samples, 0.90),
        "first_decode_latency_p99_s": _percentile(first_decode_samples, 0.99),
        "first_decode_latency_samples_s": first_decode_samples,
        "request_ttft_s": statistics.median(ttft_samples),
        "ttft_s": statistics.median(ttft_samples),
        "ttft_p90_s": _percentile(ttft_samples, 0.90),
        "ttft_p99_s": _percentile(ttft_samples, 0.99),
        "itl_median_s": median_itl,
        "itl_p90_s": _percentile(itl_samples, 0.90),
        "itl_p99_s": _percentile(itl_samples, 0.99),
        "request_ttft_samples_s": ttft_samples,
        "itl_samples_s": itl_samples,
        "request_end_to_end_latency_s": statistics.median(request_end_to_end_samples),
        "request_end_to_end_latency_p90_s": _percentile(
            request_end_to_end_samples, 0.90
        ),
        "request_end_to_end_latency_p99_s": _percentile(
            request_end_to_end_samples, 0.99
        ),
        "request_end_to_end_samples_s": request_end_to_end_samples,
        "kv_cache_bytes": int(first["kv_cache_bytes"]),
        "achieved_bw_gbps": statistics.median(
            float(result["achieved_bw_gbps"]) for result in results
        ),
        "output_tokens_per_sec": int(first["batch_size"]) / median_itl,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


def _read_tokens(path: Path) -> torch.Tensor:
    text = path.read_text(encoding="utf-8")
    return torch.tensor(list(text.encode("ascii", errors="replace")), dtype=torch.long)


def _tinyshakespeare_token_tensors(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    full_text = (root / "tinyshakespeare.txt").read_text(encoding="utf-8")
    characters = sorted(set(full_text))
    if len(characters) != 65:
        raise ValueError(
            f"canonical Tiny Shakespeare tokenizer expected 65 symbols, found {len(characters)}"
        )
    stoi = {character: index for index, character in enumerate(characters)}

    def encode(path: Path) -> torch.Tensor:
        text = path.read_text(encoding="utf-8")
        return torch.tensor([stoi[character] for character in text], dtype=torch.long)

    train = encode(root / "tinyshakespeare_train.txt")
    validation = encode(root / "tinyshakespeare_val.txt")
    if len(train) != 1_003_854 or len(validation) != 111_540:
        raise ValueError(
            "canonical nanoGPT split expected 1,003,854 train and 111,540 validation tokens"
        )
    return train, validation


def _random_nanogpt_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(len(tokens) - seq_len, (batch_size,))
    inputs = torch.stack([tokens[index : index + seq_len] for index in indices])
    targets = torch.stack(
        [tokens[index + 1 : index + 1 + seq_len] for index in indices]
    )
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def _estimate_canonical_losses(
    model: torch.nn.Module,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    eval_iters: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    aggregates: list[float] = []
    for tokens in (train_tokens, validation_tokens):
        losses: list[float] = []
        for _ in range(eval_iters):
            inputs, targets = _random_nanogpt_batch(
                tokens,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
            )
            _, loss = model(inputs, targets=targets)
            losses.append(float(loss.item()))
        aggregates.append(statistics.fmean(losses))
    model.train()
    return aggregates[0], aggregates[1]


def _canonical_nanogpt_lr(
    iteration: int,
    *,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    decay_iters: int,
) -> float:
    if iteration < warmup_iters:
        return learning_rate * (iteration + 1) / (warmup_iters + 1)
    if iteration > decay_iters:
        return min_lr
    decay_ratio = (iteration - warmup_iters) / (decay_iters - warmup_iters)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coefficient * (learning_rate - min_lr)


def _tinyshakespeare_loaders(
    *,
    train_path: Path,
    val_path: Path,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = _TextDataset(_read_tokens(train_path), seq_len=seq_len)
    val_ds = _NonOverlappingTextDataset(_read_tokens(val_path), seq_len=seq_len)
    if len(train_ds) < batch_size or len(val_ds) < batch_size:
        raise ValueError(
            "TinyShakespeare split is too small for the configured max run; "
            "increase dataset size or lower batch/sequence length."
        )
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=generator,
        ),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    )


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_batches: int,
) -> tuple[float, int]:
    model.train()
    losses: list[float] = []
    tokens_seen = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets=targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        tokens_seen += int(inputs.numel())
    return (sum(losses) / len(losses), tokens_seen) if losses else (float("inf"), 0)


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None,
) -> tuple[float, int, int]:
    model.eval()
    weighted_loss = 0.0
    evaluated_tokens = 0
    evaluated_batches = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        _, loss = model(inputs.to(device), targets=targets.to(device))
        batch_tokens = int(targets.numel())
        weighted_loss += float(loss.item()) * batch_tokens
        evaluated_tokens += batch_tokens
        evaluated_batches += 1
    if not evaluated_tokens:
        return float("inf"), 0, 0
    return weighted_loss / evaluated_tokens, evaluated_tokens, evaluated_batches
