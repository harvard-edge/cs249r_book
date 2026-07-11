from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mlperf.assets import ensure_movielens_100k
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.reference.dataset_factory import _dlrm_collate_fn as _base_dlrm_collate_fn
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    configured_seed,
    synchronize_device,
    training_measurement_protocol,
)


def run_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step Micro-DLRM training smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.micro_dlrm import MicroDLRMWhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 8
    table_sizes = [943, 1682, 21]

    model = MicroDLRMWhiteBox().to(device)
    model.train()
    dense = torch.randn(batch_size, 16, device=device)
    sparse_indices = [
        torch.randint(0, n, (batch_size,), dtype=torch.long, device=device)
        for n in table_sizes
    ]
    sparse_offsets = [
        torch.arange(batch_size, dtype=torch.long, device=device) for _ in table_sizes
    ]
    labels = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    outputs = model(dense, sparse_indices, sparse_offsets)
    loss = F.binary_cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    preds = (outputs.detach() >= 0.5).float()
    accuracy = float((preds == labels).float().mean().item())
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "micro-dlrm-train_min_report.json").resolve()
    manifest_path = (output_dir / "micro-dlrm-train_min.provd.json").resolve()
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
            "accuracy": accuracy,
            "duration_seconds": float(duration),
            "samples": batch_size,
            "samples_per_second": float(batch_size / duration),
            "n_params": int(n_params),
            "output_shape": list(outputs.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "min profile validates DLRM execution only; max profile owns MovieLens quality checks.",
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
        dataset_name="synthetic-deterministic-dlrm",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_dram_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic Micro-DLRM-DRAM training smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.micro_dlrm_dram import MicroDLRMDRAM

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 8
    table_sizes = [943, 1682, 21]

    model = MicroDLRMDRAM(m_spa=32, virtual_table_size=16_384, sparse_grad=True).to(
        device
    )
    model.train()
    dense = torch.randn(batch_size, 16, device=device)
    sparse_indices = [
        torch.randint(0, n, (batch_size,), dtype=torch.long, device=device)
        for n in table_sizes
    ]
    sparse_offsets = [
        torch.arange(batch_size, dtype=torch.long, device=device) for _ in table_sizes
    ]
    labels = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32, device=device)

    optimizer = torch.optim.SparseAdam([model.virtual_emb.weight], lr=1e-3)
    dense_optimizer = torch.optim.AdamW(
        [p for name, p in model.named_parameters() if name != "virtual_emb.weight"],
        lr=1e-3,
    )
    start = time.perf_counter()
    outputs = model(dense, sparse_indices, sparse_offsets)
    loss = F.binary_cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    dense_optimizer.step()
    duration = time.perf_counter() - start

    preds = (outputs.detach() >= 0.5).float()
    accuracy = float((preds == labels).float().mean().item())
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "micro-dlrm-dram-train_min_report.json").resolve()
    manifest_path = (output_dir / "micro-dlrm-dram-train_min.provd.json").resolve()
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
            "accuracy": accuracy,
            "duration_seconds": float(duration),
            "samples": batch_size,
            "samples_per_second": float(batch_size / duration),
            "n_params": int(n_params),
            "working_set_bytes": int(model.working_set_bytes()),
            "output_shape": list(outputs.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "min profile validates hashed virtual embedding execution; max profile should scale the table for DRAM bandwidth studies.",
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
        dataset_name="synthetic-deterministic-dlrm-dram",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_distributed_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a tiny two-rank DLRM DDP smoke on localhost Gloo."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.distributed.ddp_runner import run_ddp, run_gradacc_baseline

    seed = configured_seed()
    torch.manual_seed(seed)
    world_size = _env_int("MLPERF_EDU_DDP_WORLD_SIZE", 2)
    n_steps = _env_int("MLPERF_EDU_DDP_STEPS", 2)
    micro_batch = _env_int("MLPERF_EDU_DDP_MICRO_BATCH", 4)

    start = time.perf_counter()
    ddp = run_ddp(n_steps=n_steps, micro_batch=micro_batch, world_size=world_size)
    baseline = run_gradacc_baseline(
        n_steps=n_steps, micro_batch=micro_batch, world_size=world_size
    )
    duration = time.perf_counter() - start

    error = ddp.get("error")
    loss_delta = None
    relative_loss_delta = None
    if not error:
        loss_delta = abs(float(ddp["final_loss"]) - float(baseline["final_loss"]))
        denom = max(abs(float(baseline["final_loss"])), 1e-12)
        relative_loss_delta = loss_delta / denom
    target = float(os.environ.get("MLPERF_EDU_DDP_REL_LOSS_TARGET", "1.0"))
    target_met = (
        (error is None)
        and (relative_loss_delta is not None)
        and relative_loss_delta <= target
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "micro-dlrm-distributed_min_report.json").resolve()
    manifest_path = (output_dir / "micro-dlrm-distributed_min.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "min",
        "scenario": workload.scenario,
        "status": "passed" if target_met else "quality_failed",
        "backend": "torch-distributed-gloo-cpu",
        "data_mode": "synthetic-deterministic",
        "dataset": workload.dataset,
        "seed": seed,
        "config": {
            "world_size": world_size,
            "n_steps": n_steps,
            "micro_batch": micro_batch,
            "backend": "gloo",
            "transport": "localhost-loopback",
        },
        "metrics": {
            "duration_seconds": float(duration),
            "ddp_final_loss": float(ddp.get("final_loss", 0.0)) if not error else None,
            "gradacc_final_loss": float(baseline["final_loss"]),
            "loss_delta": float(loss_delta) if loss_delta is not None else None,
            "relative_loss_delta": float(relative_loss_delta)
            if relative_loss_delta is not None
            else None,
            "backward_with_allreduce_time_per_step_ms": float(
                ddp.get("backward_with_allreduce_time_per_step_ms", 0.0)
            )
            if not error
            else None,
            "n_params": int(ddp.get("n_params", 0)) if not error else None,
            "gradient_payload_bytes_fp32": int(
                ddp.get("gradient_payload_bytes_fp32", 0)
            )
            if not error
            else None,
        },
        "quality": {
            "metric": "relative_loss_delta",
            "target": target,
            "direction": "lower",
            "target_met": target_met,
            "quality_required": True,
            "note": "min profile validates that localhost Gloo DDP runs and stays close to a gradient-accumulation baseline.",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
        },
    }
    if error:
        report["error"] = error
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario,
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name=workload.dataset or "synthetic-deterministic-ddp",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_dram_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run the cache-stress DLRM variant on MovieLens-100K."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.micro_dlrm_dram import MicroDLRMDRAM
    from mlperf.reference.dataset_factory import load_movielens_fixed_split

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device(os.environ.get("MLPERF_EDU_DEVICE", "cpu"))
    asset = ensure_movielens_100k(download=True)

    batch_size = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_BATCH_SIZE", 256)
    epochs = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_EPOCHS", 10)
    batches_per_epoch = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_BATCHES_PER_EPOCH", 25)
    val_batches = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_VAL_BATCHES", 50)
    lr = _env_float("MLPERF_EDU_DLRM_DRAM_MAX_LR", 1e-2)
    m_spa = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_M_SPA", 256)
    virtual_table_size = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_VIRTUAL_TABLE_SIZE", 65_536)

    train_ds, val_ds = load_movielens_fixed_split(str(asset.root))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_dlrm_dram_collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_dlrm_dram_collate_fn,
    )

    model = MicroDLRMDRAM(
        m_spa=m_spa, virtual_table_size=virtual_table_size, sparse_grad=True
    ).to(device)
    sparse_optimizer = torch.optim.SparseAdam([model.virtual_emb.weight], lr=lr)
    dense_optimizer = torch.optim.AdamW(
        [p for name, p in model.named_parameters() if name != "virtual_emb.weight"],
        lr=lr,
        weight_decay=1e-4,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    val_aurocs: list[float] = []
    epoch_times: list[float] = []
    samples_seen = 0
    start = time.perf_counter()
    for _epoch in range(epochs):
        t0 = time.perf_counter()
        train_loss, train_samples = _train_dram_epoch(
            model,
            train_loader,
            sparse_optimizer,
            dense_optimizer,
            device,
            max_batches=batches_per_epoch,
        )
        val_loss, val_acc, val_auc = _validate(
            model, val_loader, device, max_batches=val_batches
        )
        samples_seen += train_samples
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_aurocs.append(val_auc)
        epoch_times.append(time.perf_counter() - t0)
    duration = time.perf_counter() - start

    final_accuracy = val_accuracies[-1]
    target = _env_float(
        "MLPERF_EDU_DLRM_DRAM_MAX_ACCURACY_TARGET",
        float(workload.quality_value or 0.65),
    )
    target_met = final_accuracy >= target
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "micro-dlrm-dram-train_max_report.json").resolve()
    manifest_path = (output_dir / "micro-dlrm-dram-train_max.provd.json").resolve()
    checkpoint_path = (output_dir / "micro-dlrm-dram-train_max_checkpoint.pt").resolve()
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
            "batch_size": batch_size,
            "epochs": epochs,
            "batches_per_epoch": batches_per_epoch,
            "val_batches": val_batches,
            "lr": lr,
            "m_spa": m_spa,
            "virtual_table_size": virtual_table_size,
            "split": {
                "train": "u1.base",
                "validation": "u1.test",
                "training_seed_affects_split": False,
            },
            "feature_recipe": (
                "demographics-item-genres-v2-no-rating-aggregates-plus-user-item-cross"
            ),
            "memory_regime_claim": "unmeasured; requires hardware-local profiling",
        },
        "metrics": {
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
            "final_accuracy": float(final_accuracy),
            "final_roc_auc": float(val_aurocs[-1]),
            "duration_seconds": float(duration),
            "samples": int(samples_seen),
            "samples_per_second": float(samples_seen / duration)
            if duration > 0
            else 0.0,
            "n_params": int(n_params),
            "working_set_bytes": int(model.working_set_bytes()),
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_aurocs": val_aurocs,
            "val_accuracies": val_accuracies,
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": target,
            "direction": "higher",
            "target_met": target_met,
            "quality_required": True,
            "override": "MLPERF_EDU_DLRM_DRAM_MAX_ACCURACY_TARGET" in os.environ,
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


def run_distributed_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a standard localhost DDP-vs-gradient-accumulation equivalence check."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.distributed.ddp_runner import run_ddp, run_gradacc_baseline

    seed = configured_seed()
    torch.manual_seed(seed)
    world_size = _env_int(
        "MLPERF_EDU_DDP_MAX_WORLD_SIZE", _env_int("MLPERF_EDU_DDP_WORLD_SIZE", 2)
    )
    n_steps = _env_int("MLPERF_EDU_DDP_MAX_STEPS", 4)
    micro_batch = _env_int("MLPERF_EDU_DDP_MAX_MICRO_BATCH", 8)

    start = time.perf_counter()
    ddp = run_ddp(n_steps=n_steps, micro_batch=micro_batch, world_size=world_size)
    baseline = run_gradacc_baseline(
        n_steps=n_steps, micro_batch=micro_batch, world_size=world_size
    )
    duration = time.perf_counter() - start

    error = ddp.get("error")
    loss_delta = None
    relative_loss_delta = None
    if not error:
        loss_delta = abs(float(ddp["final_loss"]) - float(baseline["final_loss"]))
        denom = max(abs(float(baseline["final_loss"])), 1e-12)
        relative_loss_delta = loss_delta / denom
    target = _env_float("MLPERF_EDU_DDP_MAX_REL_LOSS_TARGET", 0.05)
    target_met = (
        (error is None)
        and (relative_loss_delta is not None)
        and relative_loss_delta <= target
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "micro-dlrm-distributed_max_report.json").resolve()
    manifest_path = (output_dir / "micro-dlrm-distributed_max.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "scenario": workload.scenario,
        "status": "passed" if target_met else "quality_failed",
        "backend": "torch-distributed-gloo-cpu",
        "data_mode": "synthetic-deterministic",
        "dataset": workload.dataset,
        "seed": seed,
        "config": {
            "world_size": world_size,
            "n_steps": n_steps,
            "micro_batch": micro_batch,
            "backend": "gloo",
            "transport": "localhost-loopback",
        },
        "metrics": {
            "duration_seconds": float(duration),
            "ddp_final_loss": float(ddp.get("final_loss", 0.0)) if not error else None,
            "gradacc_final_loss": float(baseline["final_loss"]),
            "loss_delta": float(loss_delta) if loss_delta is not None else None,
            "relative_loss_delta": float(relative_loss_delta)
            if relative_loss_delta is not None
            else None,
            "backward_with_allreduce_time_per_step_ms": float(
                ddp.get("backward_with_allreduce_time_per_step_ms", 0.0)
            )
            if not error
            else None,
            "n_params": int(ddp.get("n_params", 0)) if not error else None,
            "gradient_payload_bytes_fp32": int(
                ddp.get("gradient_payload_bytes_fp32", 0)
            )
            if not error
            else None,
        },
        "quality": {
            "metric": "relative_loss_delta",
            "target": target,
            "direction": "lower",
            "target_met": target_met,
            "quality_required": True,
            "override": "MLPERF_EDU_DDP_MAX_REL_LOSS_TARGET" in os.environ,
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
        },
    }
    if error:
        report["error"] = error
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario,
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name=workload.dataset or "synthetic-deterministic-ddp",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run Micro-DLRM on real MovieLens-100K with an accuracy target."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.cloud.micro_dlrm import MicroDLRMWhiteBox
    from mlperf.reference.dataset_factory import (
        _dlrm_collate_fn,
        load_movielens_fixed_split,
    )

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device(os.environ.get("MLPERF_EDU_DEVICE", "cpu"))
    asset = ensure_movielens_100k(download=True)

    batch_size = _env_int("MLPERF_EDU_DLRM_MAX_BATCH_SIZE", 256)
    epochs = _env_int("MLPERF_EDU_DLRM_MAX_EPOCHS", 21)
    batches_per_epoch = _env_int("MLPERF_EDU_DLRM_MAX_BATCHES_PER_EPOCH", 50)
    evaluation_batches = _env_int("MLPERF_EDU_DLRM_MAX_EVALUATION_BATCHES", 100)
    lr = _env_float("MLPERF_EDU_DLRM_MAX_LR", 1e-2)

    train_ds, evaluation_ds = load_movielens_fixed_split(str(asset.root))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_dlrm_collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    evaluation_loader = DataLoader(
        evaluation_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_dlrm_collate_fn,
    )

    model = MicroDLRMWhiteBox().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses: list[float] = []
    epoch_times: list[float] = []
    samples_seen = 0
    synchronize_device(device)
    start = time.perf_counter()
    for _epoch in range(epochs):
        t0 = time.perf_counter()
        train_loss, train_samples = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_batches=batches_per_epoch,
        )
        samples_seen += train_samples
        train_losses.append(train_loss)
        epoch_times.append(time.perf_counter() - t0)
    evaluation_loss, evaluation_accuracy, evaluation_auroc = _validate(
        model, evaluation_loader, device, max_batches=evaluation_batches
    )
    synchronize_device(device)
    duration = time.perf_counter() - start

    target = _env_float(
        "MLPERF_EDU_DLRM_MAX_AUROC_TARGET", float(workload.quality_value or 0.7)
    )
    target_met = evaluation_auroc >= target
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "micro-dlrm-train_max_report.json").resolve()
    manifest_path = (output_dir / "micro-dlrm-train_max.provd.json").resolve()
    checkpoint_path = (output_dir / "micro-dlrm-train_max_checkpoint.pt").resolve()
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
        "measurement_protocol": training_measurement_protocol(workload),
        "config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "batches_per_epoch": batches_per_epoch,
            "evaluation_batches": evaluation_batches,
            "lr": lr,
            "quality_epoch_selection": "fixed_final_epoch",
            "split": {
                "train": "u1.base",
                "evaluation": "u1.test",
                "training_seed_affects_split": False,
            },
            "feature_recipe": "demographics-item-genres-v2-no-rating-aggregates",
        },
        "metrics": {
            "final_train_loss": float(train_losses[-1]),
            "evaluation_loss": float(evaluation_loss),
            "evaluation_accuracy": float(evaluation_accuracy),
            "evaluation_roc_auc": float(evaluation_auroc),
            "roc_auc": float(evaluation_auroc),
            "duration_seconds": float(duration),
            "train_and_eval_seconds": float(duration),
            "samples": int(samples_seen),
            "samples_per_second": float(samples_seen / duration)
            if duration > 0
            else 0.0,
            "n_params": int(n_params),
            "epoch_times": epoch_times,
            "train_losses": train_losses,
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": target,
            "direction": "higher",
            "target_met": target_met,
            "quality_required": True,
            "override": "MLPERF_EDU_DLRM_MAX_AUROC_TARGET" in os.environ,
            "metric_key": "roc_auc",
            "note": "Quality is ROC AUC on the untouched u1.test split after the fixed final training epoch; evaluation labels never select a checkpoint.",
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


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _dlrm_dram_collate_fn(batch):
    """Add a stable user-item cross so MovieLens addresses the stress table broadly."""
    dense, sparse_indices, sparse_offsets, labels = _base_dlrm_collate_fn(batch)
    user_ids, item_ids = sparse_indices[:2]
    user_item_cross = user_ids * 1682 + item_ids
    return (
        dense,
        [user_ids, item_ids, user_item_cross],
        sparse_offsets,
        labels,
    )


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _move_sparse(batch, device: torch.device):
    dense, sparse_indices, sparse_offsets, labels = batch
    return (
        dense.to(device),
        [item.to(device) for item in sparse_indices],
        [item.to(device) for item in sparse_offsets],
        labels.to(device),
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
    samples = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        dense, sparse_indices, sparse_offsets, labels = _move_sparse(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(dense, sparse_indices, sparse_offsets)
        loss = F.binary_cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        samples += int(labels.numel())
    return (sum(losses) / len(losses), samples) if losses else (float("inf"), 0)


def _train_dram_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    sparse_optimizer: torch.optim.Optimizer,
    dense_optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_batches: int,
) -> tuple[float, int]:
    model.train()
    losses: list[float] = []
    samples = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        dense, sparse_indices, sparse_offsets, labels = _move_sparse(batch, device)
        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)
        outputs = model(dense, sparse_indices, sparse_offsets)
        loss = F.binary_cross_entropy(outputs, labels)
        loss.backward()
        sparse_optimizer.step()
        dense_optimizer.step()
        losses.append(float(loss.item()))
        samples += int(labels.numel())
    return (sum(losses) / len(losses), samples) if losses else (float("inf"), 0)


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    scores: list[float] = []
    binary_labels: list[int] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        dense, sparse_indices, sparse_offsets, labels = _move_sparse(batch, device)
        outputs = model(dense, sparse_indices, sparse_offsets)
        loss = F.binary_cross_entropy(outputs, labels)
        preds = (outputs >= 0.5).float()
        losses.append(float(loss.item()))
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
        scores.extend(float(value) for value in outputs.detach().cpu().view(-1))
        binary_labels.extend(int(value) for value in labels.detach().cpu().view(-1))
    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy, _binary_auroc(scores, binary_labels)


def _binary_auroc(scores: list[float], labels: list[int]) -> float:
    """Compute binary ROC AUC using average ranks for tied scores."""
    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum_positive += average_rank * sum(
            label == 1 for _, label in pairs[index:end]
        )
        index = end
    positives = sum(label == 1 for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    return float(
        (rank_sum_positive - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )
