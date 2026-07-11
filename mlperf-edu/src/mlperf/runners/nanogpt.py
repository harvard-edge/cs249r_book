from __future__ import annotations

import json
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
from mlperf.runners.common import configured_seed


def run_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step NanoGPT training smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed()
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
    report_path = (output_dir / "nanogpt-train_min_report.json").resolve()
    manifest_path = (output_dir / "nanogpt-train_min.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "min",
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
        scenario="train",
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
    report_path = (output_dir / "nanogpt-prefill_min_report.json").resolve()
    manifest_path = (output_dir / "nanogpt-prefill_min.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "min",
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
    device = _select_device()
    context_len = _env_int("MLPERF_EDU_PREFILL_MAX_CONTEXT", 1792)
    batch_size = _env_int("MLPERF_EDU_PREFILL_MAX_BATCH", 1)
    n_warmup = _env_int("MLPERF_EDU_PREFILL_MAX_WARMUP", 3)
    n_iter = _env_int("MLPERF_EDU_PREFILL_MAX_ITER", 10)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "nanogpt-prefill_max_report.json").resolve()
    manifest_path = (output_dir / "nanogpt-prefill_max.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "data_mode": "checkpoint-backed",
        "seed": seed,
        "config": {
            "context_len": context_len,
            "batch_size": batch_size,
            "n_warmup": n_warmup,
            "n_iter": n_iter,
            "model_size": os.environ.get("MLPERF_EDU_MAX_MODEL_SIZE", "base"),
        },
        "metrics": {
            **result,
            "n_params": int(n_params),
        },
        "quality": {
            "metric": "prefill_tokens_per_sec",
            "metric_key": "prefill_tokens_per_sec",
            "target": "> 0",
            "quality_required": True,
            "target_met": target_met,
            "note": "The functional gate requires a quality-approved checkpoint and positive measured prefill throughput.",
        },
        "measurement_protocol": {
            "warmup_runs": n_warmup,
            "measured_runs": n_iter,
            "latency_statistics": ["median", "p90", "p99"],
            "timing_scope": "synchronized checkpoint-backed forward passes over one fixed-shape prompt batch",
        },
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


def run_decode_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
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
    report_path = (output_dir / "nanogpt-decode_min_report.json").resolve()
    manifest_path = (output_dir / "nanogpt-decode_min.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "min",
        "status": "passed",
        "backend": "pytorch-cpu",
        "data_mode": "synthetic-deterministic",
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
        scenario="server",
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


def run_decode_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run checkpoint-backed NanoGPT KV-cache decode inference."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.nanogpt_decode import NanoGPTDecode

    seed = configured_seed()
    torch.manual_seed(seed)
    device = _select_device()
    prefill_ctx = _env_int("MLPERF_EDU_DECODE_MAX_PREFILL_CTX", 1792)
    decode_steps = _env_int("MLPERF_EDU_DECODE_MAX_STEPS", 64)
    batch_size = _env_int("MLPERF_EDU_DECODE_MAX_BATCH", 1)
    repetitions = _env_int("MLPERF_EDU_DECODE_MAX_REPETITIONS", 5)
    model, checkpoint_path, checkpoint_lineage = _load_max_nanogpt_model(
        output_dir, device, prefill_ctx + decode_steps
    )

    decode = NanoGPTDecode(
        model,
        prefill_ctx=prefill_ctx,
        decode_steps=decode_steps,
        batch_size=batch_size,
    )
    decode.run(emit_sidecar=False)
    results = [decode.run(emit_sidecar=False) for _ in range(repetitions)]
    result = _aggregate_decode_results(results)
    n_params = sum(p.numel() for p in model.parameters())
    target_met = bool(
        result.get("decode_steps") == decode_steps
        and result.get("output_tokens_per_sec", 0) > 0
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "nanogpt-decode_max_report.json").resolve()
    manifest_path = (output_dir / "nanogpt-decode_max.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "data_mode": "checkpoint-backed",
        "seed": seed,
        "config": {
            "prefill_ctx": prefill_ctx,
            "decode_steps": decode_steps,
            "batch_size": batch_size,
            "repetitions": repetitions,
            "model_size": os.environ.get("MLPERF_EDU_MAX_MODEL_SIZE", "base"),
        },
        "metrics": {
            **result,
            "n_params": int(n_params),
        },
        "quality": {
            "metric": "decode_steps",
            "metric_key": "decode_steps",
            "target": decode_steps,
            "quality_required": True,
            "target_met": target_met,
            "note": "The functional gate requires a quality-approved checkpoint, the configured decode length, and positive throughput.",
        },
        "measurement_protocol": {
            "warmup_runs": 1,
            "measured_runs": repetitions,
            "latency_statistics": ["median", "p90", "p99"],
            "timing_scope": "synchronized checkpoint-backed requests with per-token ITL samples",
        },
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
        scenario="server",
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


def run_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run the TinyShakespeare NanoGPT max-profile quality target."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    asset = ensure_tinyshakespeare(download=True)
    batch_size = _env_int("MLPERF_EDU_MAX_BATCH_SIZE", 16)
    seq_len = _env_int("MLPERF_EDU_MAX_SEQ_LEN", 64)
    epochs = _env_int("MLPERF_EDU_MAX_EPOCHS", 25)
    batches_per_epoch = _env_int("MLPERF_EDU_MAX_BATCHES_PER_EPOCH", 100)
    val_batches = _env_int("MLPERF_EDU_MAX_VAL_BATCHES", 20)
    lr = _env_float("MLPERF_EDU_MAX_LR", 3e-4)
    model_size = os.environ.get("MLPERF_EDU_MAX_MODEL_SIZE", "base")

    device = _select_device()
    model_kwargs = _max_model_kwargs(model_size, seq_len)
    model = NanoGPTWhiteBox(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_loader, val_loader = _tinyshakespeare_loaders(
        train_path=asset.root / "tinyshakespeare_train.txt",
        val_path=asset.root / "tinyshakespeare_val.txt",
        batch_size=batch_size,
        seq_len=seq_len,
        seed=seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=lr * 0.1,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    epoch_times: list[float] = []
    start = time.perf_counter()
    tokens_seen = 0

    for _epoch in range(epochs):
        t0 = time.perf_counter()
        train_loss, train_tokens = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_batches=batches_per_epoch,
        )
        val_loss = _validate(model, val_loader, device, max_batches=val_batches)
        scheduler.step()
        tokens_seen += train_tokens
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(time.perf_counter() - t0)

    duration = time.perf_counter() - start
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    target = _env_float(
        "MLPERF_EDU_MAX_QUALITY_TARGET", float(workload.quality_value or 2.3)
    )
    target_met = final_val_loss <= target

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "nanogpt-train_max_report.json").resolve()
    manifest_path = (output_dir / "nanogpt-train_max.provd.json").resolve()
    checkpoint_path = (output_dir / "nanogpt-train_max_checkpoint.pt").resolve()
    torch.save(model.state_dict(), checkpoint_path)

    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "data_mode": "real",
        "dataset": {
            "name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
        },
        "seed": seed,
        "config": {
            "model_size": model_size,
            "model_kwargs": model_kwargs,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "epochs": epochs,
            "batches_per_epoch": batches_per_epoch,
            "val_batches": val_batches,
            "lr": lr,
        },
        "metrics": {
            "final_train_loss": float(final_train_loss),
            "final_val_loss": float(final_val_loss),
            "cross_entropy_loss": float(final_val_loss),
            "duration_seconds": float(duration),
            "tokens": int(tokens_seen),
            "tokens_per_second": float(tokens_seen / duration) if duration > 0 else 0.0,
            "n_params": int(n_params),
            "epoch_times": epoch_times,
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
        scenario="train",
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


def _max_model_kwargs(model_size: str, seq_len: int) -> dict[str, int]:
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
        "vocab_size": 128,
        "n_embd": 384,
        "n_head": 6,
        "n_layer": 6,
        "max_seq_len": max(2048, seq_len),
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
                output_dir / "nanogpt-train_max_checkpoint.pt",
            )
        )
        .expanduser()
        .resolve()
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"NanoGPT max checkpoint not found at {checkpoint}. "
            "Run `mlperf run --workload nanogpt-train --profile max --output-dir <same-dir>` first, "
            "or set MLPERF_EDU_NANOGPT_CHECKPOINT."
        )

    source_report_path = (
        Path(
            os.environ.get(
                "MLPERF_EDU_NANOGPT_TRAIN_REPORT",
                checkpoint.parent / "nanogpt-train_max_report.json",
            )
        )
        .expanduser()
        .resolve()
    )
    source_manifest_path = (
        Path(
            os.environ.get(
                "MLPERF_EDU_NANOGPT_TRAIN_MANIFEST",
                checkpoint.parent / "nanogpt-train_max.provd.json",
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
        source_report.get("workload") != "nanogpt-train"
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
        "source_workload": "nanogpt-train",
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


def _select_device() -> torch.device:
    requested = os.environ.get("MLPERF_EDU_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    ttft_samples = [float(result["ttft_s"]) for result in results]
    prefill_samples = [float(result["prefill_warm_s"]) for result in results]
    if not itl_samples:
        raise ValueError("decode measurement produced no inter-token latency samples")
    median_itl = statistics.median(itl_samples)
    return {
        "phase": "decode",
        "prefill_ctx": int(first["prefill_ctx"]),
        "decode_steps": int(first["decode_steps"]),
        "batch_size": int(first["batch_size"]),
        "prefill_warm_s": statistics.median(prefill_samples),
        "prefill_warm_p90_s": _percentile(prefill_samples, 0.90),
        "prefill_warm_p99_s": _percentile(prefill_samples, 0.99),
        "ttft_s": statistics.median(ttft_samples),
        "ttft_p90_s": _percentile(ttft_samples, 0.90),
        "ttft_p99_s": _percentile(ttft_samples, 0.99),
        "itl_median_s": median_itl,
        "itl_p90_s": _percentile(itl_samples, 0.90),
        "itl_p99_s": _percentile(itl_samples, 0.99),
        "request_ttft_samples_s": ttft_samples,
        "itl_samples_s": itl_samples,
        "kv_cache_bytes": int(first["kv_cache_bytes"]),
        "achieved_bw_gbps": statistics.median(
            float(result["achieved_bw_gbps"]) for result in results
        ),
        "output_tokens_per_sec": 1.0 / median_itl,
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


def _tinyshakespeare_loaders(
    *,
    train_path: Path,
    val_path: Path,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = _TextDataset(_read_tokens(train_path), seq_len=seq_len)
    val_ds = _TextDataset(_read_tokens(val_path), seq_len=seq_len)
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
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True),
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
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        _, loss = model(inputs.to(device), targets=targets.to(device))
        losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else float("inf")
