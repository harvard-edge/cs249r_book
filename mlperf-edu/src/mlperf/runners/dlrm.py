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
from mlperf.registry import Workload, find_project_root


def run_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step Micro-DLRM training smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from reference.cloud.micro_dlrm import MicroDLRMWhiteBox

    seed = 42
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
        torch.arange(batch_size, dtype=torch.long, device=device)
        for _ in table_sizes
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
        scenario="train",
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
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def run_dram_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic Micro-DLRM-DRAM training smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from reference.cloud.micro_dlrm_dram import MicroDLRMDRAM

    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 8
    table_sizes = [943, 1682, 21]

    model = MicroDLRMDRAM(m_spa=32, virtual_table_size=16_384, sparse_grad=True).to(device)
    model.train()
    dense = torch.randn(batch_size, 16, device=device)
    sparse_indices = [
        torch.randint(0, n, (batch_size,), dtype=torch.long, device=device)
        for n in table_sizes
    ]
    sparse_offsets = [
        torch.arange(batch_size, dtype=torch.long, device=device)
        for _ in table_sizes
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
        scenario="train",
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
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def run_distributed_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a tiny two-rank DLRM DDP smoke on localhost Gloo."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from reference.distributed.ddp_runner import run_ddp, run_gradacc_baseline

    seed = 42
    torch.manual_seed(seed)
    world_size = _env_int("MLPERF_EDU_DDP_WORLD_SIZE", 2)
    n_steps = _env_int("MLPERF_EDU_DDP_STEPS", 2)
    micro_batch = _env_int("MLPERF_EDU_DDP_MICRO_BATCH", 4)

    start = time.perf_counter()
    ddp = run_ddp(n_steps=n_steps, micro_batch=micro_batch, world_size=world_size)
    baseline = run_gradacc_baseline(n_steps=n_steps, micro_batch=micro_batch, world_size=world_size)
    duration = time.perf_counter() - start

    error = ddp.get("error")
    loss_delta = None
    relative_loss_delta = None
    if not error:
        loss_delta = abs(float(ddp["final_loss"]) - float(baseline["final_loss"]))
        denom = max(abs(float(baseline["final_loss"])), 1e-12)
        relative_loss_delta = loss_delta / denom
    target = float(os.environ.get("MLPERF_EDU_DDP_REL_LOSS_TARGET", "1.0"))
    target_met = (error is None) and (relative_loss_delta is not None) and relative_loss_delta <= target

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "micro-dlrm-distributed_min_report.json").resolve()
    manifest_path = (output_dir / "micro-dlrm-distributed_min.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "min",
        "status": "passed" if target_met else "quality_failed",
        "backend": "torch-distributed-gloo-cpu",
        "data_mode": "synthetic-deterministic",
        "seed": seed,
        "config": {
            "world_size": world_size,
            "n_steps": n_steps,
            "micro_batch": micro_batch,
        },
        "metrics": {
            "duration_seconds": float(duration),
            "ddp_final_loss": float(ddp.get("final_loss", 0.0)) if not error else None,
            "gradacc_final_loss": float(baseline["final_loss"]),
            "loss_delta": float(loss_delta) if loss_delta is not None else None,
            "relative_loss_delta": float(relative_loss_delta) if relative_loss_delta is not None else None,
            "allreduce_time_per_step_ms": float(ddp.get("allreduce_time_per_step_ms", 0.0)) if not error else None,
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
        scenario="train",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name="synthetic-deterministic-ddp",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def run_dram_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run the DRAM-bound DLRM variant on MovieLens-100K."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from reference.cloud.micro_dlrm_dram import MicroDLRMDRAM
    from reference.dataset_factory import MovieLensRecommendationDataset, _dlrm_collate_fn

    seed = _env_int("MLPERF_EDU_MAX_SEED", 42)
    torch.manual_seed(seed)
    device = torch.device(os.environ.get("MLPERF_EDU_DEVICE", "cpu"))
    asset = ensure_movielens_100k(download=True)

    batch_size = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_BATCH_SIZE", 256)
    epochs = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_EPOCHS", 10)
    batches_per_epoch = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_BATCHES_PER_EPOCH", 25)
    val_batches = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_VAL_BATCHES", 50)
    lr = _env_float("MLPERF_EDU_DLRM_DRAM_MAX_LR", 1e-2)
    m_spa = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_M_SPA", 64)
    virtual_table_size = _env_int("MLPERF_EDU_DLRM_DRAM_MAX_VIRTUAL_TABLE_SIZE", 65_536)

    dataset = MovieLensRecommendationDataset(data_dir=str(asset.root))
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_dlrm_collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=_dlrm_collate_fn,
    )

    model = MicroDLRMDRAM(m_spa=m_spa, virtual_table_size=virtual_table_size, sparse_grad=True).to(device)
    sparse_optimizer = torch.optim.SparseAdam([model.virtual_emb.weight], lr=lr)
    dense_optimizer = torch.optim.AdamW(
        [p for name, p in model.named_parameters() if name != "virtual_emb.weight"],
        lr=lr,
        weight_decay=1e-4,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
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
        val_loss, val_acc = _validate(model, val_loader, device, max_batches=val_batches)
        samples_seen += train_samples
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        epoch_times.append(time.perf_counter() - t0)
    duration = time.perf_counter() - start

    final_accuracy = val_accuracies[-1]
    target = _env_float("MLPERF_EDU_DLRM_DRAM_MAX_ACCURACY_TARGET", float(workload.quality_value or 0.65))
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
        },
        "metrics": {
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
            "final_accuracy": float(final_accuracy),
            "duration_seconds": float(duration),
            "samples": int(samples_seen),
            "samples_per_second": float(samples_seen / duration) if duration > 0 else 0.0,
            "n_params": int(n_params),
            "working_set_bytes": int(model.working_set_bytes()),
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "val_losses": val_losses,
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
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def run_distributed_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a standard localhost DDP-vs-gradient-accumulation equivalence check."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from reference.distributed.ddp_runner import run_ddp, run_gradacc_baseline

    seed = 42
    torch.manual_seed(seed)
    world_size = _env_int("MLPERF_EDU_DDP_MAX_WORLD_SIZE", _env_int("MLPERF_EDU_DDP_WORLD_SIZE", 2))
    n_steps = _env_int("MLPERF_EDU_DDP_MAX_STEPS", 4)
    micro_batch = _env_int("MLPERF_EDU_DDP_MAX_MICRO_BATCH", 8)

    start = time.perf_counter()
    ddp = run_ddp(n_steps=n_steps, micro_batch=micro_batch, world_size=world_size)
    baseline = run_gradacc_baseline(n_steps=n_steps, micro_batch=micro_batch, world_size=world_size)
    duration = time.perf_counter() - start

    error = ddp.get("error")
    loss_delta = None
    relative_loss_delta = None
    if not error:
        loss_delta = abs(float(ddp["final_loss"]) - float(baseline["final_loss"]))
        denom = max(abs(float(baseline["final_loss"])), 1e-12)
        relative_loss_delta = loss_delta / denom
    target = _env_float("MLPERF_EDU_DDP_MAX_REL_LOSS_TARGET", 0.05)
    target_met = (error is None) and (relative_loss_delta is not None) and relative_loss_delta <= target

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "micro-dlrm-distributed_max_report.json").resolve()
    manifest_path = (output_dir / "micro-dlrm-distributed_max.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": "torch-distributed-gloo-cpu",
        "data_mode": "synthetic-deterministic",
        "seed": seed,
        "config": {
            "world_size": world_size,
            "n_steps": n_steps,
            "micro_batch": micro_batch,
        },
        "metrics": {
            "duration_seconds": float(duration),
            "ddp_final_loss": float(ddp.get("final_loss", 0.0)) if not error else None,
            "gradacc_final_loss": float(baseline["final_loss"]),
            "loss_delta": float(loss_delta) if loss_delta is not None else None,
            "relative_loss_delta": float(relative_loss_delta) if relative_loss_delta is not None else None,
            "allreduce_time_per_step_ms": float(ddp.get("allreduce_time_per_step_ms", 0.0)) if not error else None,
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
        scenario="train",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name="synthetic-deterministic-ddp",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def run_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run Micro-DLRM on real MovieLens-100K with an accuracy target."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from reference.cloud.micro_dlrm import MicroDLRMWhiteBox
    from reference.dataset_factory import MovieLensRecommendationDataset, _dlrm_collate_fn

    seed = _env_int("MLPERF_EDU_MAX_SEED", 42)
    torch.manual_seed(seed)
    device = torch.device(os.environ.get("MLPERF_EDU_DEVICE", "cpu"))
    asset = ensure_movielens_100k(download=True)

    batch_size = _env_int("MLPERF_EDU_DLRM_MAX_BATCH_SIZE", 256)
    epochs = _env_int("MLPERF_EDU_DLRM_MAX_EPOCHS", 21)
    batches_per_epoch = _env_int("MLPERF_EDU_DLRM_MAX_BATCHES_PER_EPOCH", 50)
    val_batches = _env_int("MLPERF_EDU_DLRM_MAX_VAL_BATCHES", 100)
    lr = _env_float("MLPERF_EDU_DLRM_MAX_LR", 1e-2)

    dataset = MovieLensRecommendationDataset(data_dir=str(asset.root))
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_dlrm_collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=_dlrm_collate_fn,
    )

    model = MicroDLRMWhiteBox().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    epoch_times: list[float] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    samples_seen = 0
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
        val_loss, val_acc = _validate(model, val_loader, device, max_batches=val_batches)
        samples_seen += train_samples
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        epoch_times.append(time.perf_counter() - t0)
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = len(val_accuracies)
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    duration = time.perf_counter() - start

    final_accuracy = val_accuracies[-1]
    target = _env_float("MLPERF_EDU_DLRM_MAX_ACCURACY_TARGET", float(workload.quality_value or 0.7))
    target_met = best_accuracy >= target
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "micro-dlrm-train_max_report.json").resolve()
    manifest_path = (output_dir / "micro-dlrm-train_max.provd.json").resolve()
    checkpoint_path = (output_dir / "micro-dlrm-train_max_checkpoint.pt").resolve()
    torch.save(best_state or model.state_dict(), checkpoint_path)

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
            "quality_epoch_selection": "best_validation_accuracy",
        },
        "metrics": {
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
            "final_accuracy": float(final_accuracy),
            "last_epoch_accuracy": float(final_accuracy),
            "best_train_loss": float(best_train_loss),
            "best_val_loss": float(best_val_loss),
            "best_accuracy": float(best_accuracy),
            "best_epoch": int(best_epoch),
            "duration_seconds": float(duration),
            "samples": int(samples_seen),
            "samples_per_second": float(samples_seen / duration) if duration > 0 else 0.0,
            "n_params": int(n_params),
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": target,
            "direction": "higher",
            "target_met": target_met,
            "quality_required": True,
            "override": "MLPERF_EDU_DLRM_MAX_ACCURACY_TARGET" in os.environ,
            "metric_key": "best_accuracy",
            "note": "Quality is evaluated at the best validation checkpoint reached during the run; final_* metrics record the last epoch.",
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
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return report


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


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
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
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
    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy
