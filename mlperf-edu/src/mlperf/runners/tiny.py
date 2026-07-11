from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mlperf.assets import DatasetAsset, ensure_mnist, sha256_file
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import configured_seed


def run_anomaly_ae_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step anomaly autoencoder training smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.anomaly_detection_ae import AnomalyDetectionAE

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 4
    input_dim = 784

    model = AnomalyDetectionAE(input_dim=input_dim, bottleneck_dim=8).to(device)
    model.train()
    inputs = torch.rand(batch_size, input_dim, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    reconstruction, loss = model(inputs)
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    scores = model.anomaly_score(inputs).detach().cpu()
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "anomaly-ae-train_min_report.json").resolve()
    manifest_path = (output_dir / "anomaly-ae-train_min.provd.json").resolve()
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
            "reconstruction_mse": float(loss.item()),
            "duration_seconds": float(duration),
            "samples": batch_size,
            "samples_per_second": float(batch_size / duration),
            "n_params": int(n_params),
            "input_shape": list(inputs.shape),
            "reconstruction_shape": list(reconstruction.shape),
            "anomaly_score_mean": float(scores.mean().item()),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "min profile validates autoencoder execution only; max profile owns MNIST reconstruction checks.",
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
        dataset_name="synthetic-deterministic-vectors",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_dscnn_kws_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step DS-CNN keyword-spotting smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.dscnn_kws import DSCNN

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 4

    model = DSCNN(num_classes=12).to(device)
    model.train()
    inputs = torch.randn(batch_size, 1, 40, 101, device=device)
    labels = torch.randint(0, 12, (batch_size,), dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    logits, loss = model(inputs, targets=labels)
    assert loss is not None
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    accuracy = float((logits.detach().argmax(dim=1) == labels).float().mean().item())
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "dscnn-kws-train_min_report.json").resolve()
    manifest_path = (output_dir / "dscnn-kws-train_min.provd.json").resolve()
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
            "model_size_bytes_fp32": int(n_params * 4),
            "model_size_bytes_int8": int(n_params),
            "input_shape": list(inputs.shape),
            "logits_shape": list(logits.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "min profile validates DS-CNN execution only; max profile will own Speech Commands quality checks.",
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
        dataset_name="synthetic-deterministic-mel-spectrograms",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_wake_vision_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step visual wake-word smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.wake_vision_vww import MicroNet

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 4

    model = MicroNet(num_classes=2).to(device)
    model.train()
    inputs = torch.randn(batch_size, 1, 96, 96, device=device)
    labels = torch.randint(0, 2, (batch_size,), dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    logits = model(inputs)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    accuracy = float((logits.detach().argmax(dim=1) == labels).float().mean().item())
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "wake-vision-vww_min_report.json").resolve()
    manifest_path = (output_dir / "wake-vision-vww_min.provd.json").resolve()
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
            "model_size_bytes_fp32": int(n_params * 4),
            "model_size_bytes_int8": int(n_params),
            "input_shape": list(inputs.shape),
            "logits_shape": list(logits.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "min profile validates MicroNet execution only; max profile will own Wake Vision quality checks.",
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
        dataset_name="synthetic-deterministic-vww-images",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_dscnn_kws_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a synthetic micro-shard DS-CNN keyword-spotting training profile."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.dscnn_kws import DSCNN

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device(os.environ.get("MLPERF_EDU_DEVICE", "cpu"))
    batch_size = int(os.environ.get("MLPERF_EDU_DSCNN_MAX_BATCH_SIZE", "8"))
    train_batches = int(os.environ.get("MLPERF_EDU_DSCNN_MAX_BATCHES", "5"))
    val_batches = int(os.environ.get("MLPERF_EDU_DSCNN_MAX_VAL_BATCHES", "2"))
    lr = float(os.environ.get("MLPERF_EDU_DSCNN_MAX_LR", "0.001"))

    train_loader, val_loader = _synthetic_classifier_loaders(
        train_shape=(train_batches * batch_size, 1, 40, 101),
        val_shape=(val_batches * batch_size, 1, 40, 101),
        n_classes=12,
        batch_size=batch_size,
        seed=seed,
    )
    model = DSCNN(num_classes=12).to(device)
    return _run_tiny_classifier_max(
        workload,
        output_dir,
        root=root,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        seed=seed,
        lr=lr,
        max_train_batches=train_batches,
        max_val_batches=val_batches,
        dataset_name="synthetic-deterministic-speech-commands-microshard",
        report_stem="dscnn-kws-train",
        quality_note="max profile validates repeatable DS-CNN training and checkpointing on a synthetic micro-shard; Speech Commands quality is a future real-data check.",
    )


def run_wake_vision_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a synthetic micro-shard visual wake-word training profile."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.wake_vision_vww import MicroNet

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device(os.environ.get("MLPERF_EDU_DEVICE", "cpu"))
    batch_size = int(os.environ.get("MLPERF_EDU_WAKE_MAX_BATCH_SIZE", "8"))
    train_batches = int(os.environ.get("MLPERF_EDU_WAKE_MAX_BATCHES", "5"))
    val_batches = int(os.environ.get("MLPERF_EDU_WAKE_MAX_VAL_BATCHES", "2"))
    lr = float(os.environ.get("MLPERF_EDU_WAKE_MAX_LR", "0.001"))

    train_loader, val_loader = _synthetic_classifier_loaders(
        train_shape=(train_batches * batch_size, 1, 96, 96),
        val_shape=(val_batches * batch_size, 1, 96, 96),
        n_classes=2,
        batch_size=batch_size,
        seed=seed,
    )
    model = MicroNet(num_classes=2).to(device)
    return _run_tiny_classifier_max(
        workload,
        output_dir,
        root=root,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        seed=seed,
        lr=lr,
        max_train_batches=train_batches,
        max_val_batches=val_batches,
        dataset_name="synthetic-deterministic-wake-vision-microshard",
        report_stem="wake-vision-vww",
        quality_note="max profile validates repeatable MicroNet training and checkpointing on a synthetic micro-shard; Wake Vision quality is a future real-data check.",
    )


def run_anomaly_ae_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run anomaly autoencoder on MNIST or an explicit local tensor shard."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.anomaly_detection_ae import (
        AnomalyDetectionAE,
        get_mnist_anomaly_dataloaders,
    )

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device(os.environ.get("MLPERF_EDU_DEVICE", "cpu"))
    batch_size = int(os.environ.get("MLPERF_EDU_ANOMALY_MAX_BATCH_SIZE", 64))
    epochs = int(os.environ.get("MLPERF_EDU_ANOMALY_MAX_EPOCHS", 20))
    batches_per_epoch = int(
        os.environ.get("MLPERF_EDU_ANOMALY_MAX_BATCHES_PER_EPOCH", 50)
    )
    requested_val_batches = int(os.environ.get("MLPERF_EDU_ANOMALY_MAX_VAL_BATCHES", 0))
    lr = float(os.environ.get("MLPERF_EDU_ANOMALY_MAX_LR", 1e-3))

    shard_path = os.environ.get("MLPERF_EDU_ANOMALY_MAX_TENSOR_PATH")
    if shard_path:
        asset, train_loader, val_loader = _load_tensor_shard(
            Path(shard_path), batch_size
        )
    else:
        asset = ensure_mnist(download=True)
        train_loader, val_loader = get_mnist_anomaly_dataloaders(
            batch_size=batch_size,
            data_dir=str(asset.root),
            num_workers=0,
        )

    val_batches = requested_val_batches or len(val_loader)

    model = AnomalyDetectionAE(input_dim=784, bottleneck_dim=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_aurocs: list[float] = []
    val_auprcs: list[float] = []
    epoch_times: list[float] = []
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
        val_loss, val_auroc, val_auprc, evaluated_samples = _evaluate_anomaly(
            model, val_loader, device, max_batches=val_batches
        )
        samples_seen += train_samples
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aurocs.append(val_auroc)
        val_auprcs.append(val_auprc)
        epoch_times.append(time.perf_counter() - t0)
    duration = time.perf_counter() - start

    target = float(
        os.environ.get(
            "MLPERF_EDU_ANOMALY_MAX_AUROC_TARGET", workload.quality_value or 0.85
        )
    )
    final_reconstruction_mse = train_losses[-1]
    final_auroc = val_aurocs[-1]
    labels_available = final_auroc == final_auroc
    target_met = labels_available and final_auroc >= target
    quality_required = not shard_path or labels_available
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "anomaly-ae-train_max_report.json").resolve()
    manifest_path = (output_dir / "anomaly-ae-train_max.provd.json").resolve()
    checkpoint_path = (output_dir / "anomaly-ae-train_max_checkpoint.pt").resolve()
    torch.save(model.state_dict(), checkpoint_path)

    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed"
        if (target_met or not quality_required)
        else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "data_mode": "real" if not shard_path else "local-tensor-shard",
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
        },
        "metrics": {
            "final_reconstruction_mse": float(final_reconstruction_mse),
            "final_val_reconstruction_mse": float(val_losses[-1]),
            "anomaly_auroc": float(final_auroc) if labels_available else None,
            "anomaly_auprc": float(val_auprcs[-1]) if labels_available else None,
            "evaluation_samples": int(evaluated_samples),
            "duration_seconds": float(duration),
            "samples": int(samples_seen),
            "samples_per_second": float(samples_seen / duration)
            if duration > 0
            else 0.0,
            "n_params": int(n_params),
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_aurocs": [
                float(value) if value == value else None for value in val_aurocs
            ],
            "val_auprcs": [
                float(value) if value == value else None for value in val_auprcs
            ],
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "anomaly_auroc",
            "target": target,
            "direction": "higher",
            "target_met": target_met if quality_required else None,
            "quality_required": quality_required,
            "override": "MLPERF_EDU_ANOMALY_MAX_AUROC_TARGET" in os.environ,
            "note": (
                "AUROC is computed from per-sample reconstruction error on the full "
                "MNIST normal/anomaly evaluation split."
                if quality_required
                else "The local tensor shard has no anomaly labels, so it validates execution only."
            ),
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


def _synthetic_classifier_loaders(
    *,
    train_shape: tuple[int, ...],
    val_shape: tuple[int, ...],
    n_classes: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    train_x = torch.randn(*train_shape, generator=generator)
    val_x = torch.randn(*val_shape, generator=generator)
    train_y = torch.arange(train_shape[0], dtype=torch.long) % n_classes
    val_y = torch.arange(val_shape[0], dtype=torch.long) % n_classes
    return (
        DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        ),
        DataLoader(
            TensorDataset(val_x, val_y),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        ),
    )


def _run_tiny_classifier_max(
    workload: Workload,
    output_dir: Path,
    *,
    root: Path,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    seed: int,
    lr: float,
    max_train_batches: int,
    max_val_batches: int,
    dataset_name: str,
    report_stem: str,
    quality_note: str,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    start = time.perf_counter()
    train_loss, train_accuracy, samples_seen = _train_classifier_epoch(
        model,
        train_loader,
        optimizer,
        device,
        max_batches=max_train_batches,
    )
    val_loss, val_accuracy = _validate_classifier(
        model, val_loader, device, max_batches=max_val_batches
    )
    duration = time.perf_counter() - start
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{report_stem}_max_report.json").resolve()
    manifest_path = (output_dir / f"{report_stem}_max.provd.json").resolve()
    checkpoint_path = (output_dir / f"{report_stem}_max_checkpoint.pt").resolve()
    torch.save(model.state_dict(), checkpoint_path)

    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed",
        "backend": f"pytorch-{device.type}",
        "data_mode": "synthetic-microshard",
        "seed": seed,
        "config": {
            "batch_size": train_loader.batch_size,
            "train_batches": max_train_batches,
            "val_batches": max_val_batches,
            "lr": lr,
        },
        "metrics": {
            "final_train_loss": float(train_loss),
            "final_val_loss": float(val_loss),
            "final_accuracy": float(val_accuracy),
            "train_accuracy": float(train_accuracy),
            "duration_seconds": float(duration),
            "samples": int(samples_seen),
            "samples_per_second": float(samples_seen / duration)
            if duration > 0
            else 0.0,
            "n_params": int(n_params),
            "model_size_bytes_fp32": int(n_params * 4),
            "model_size_bytes_int8": int(n_params),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": quality_note,
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
        dataset_name=dataset_name,
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def _train_classifier_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_batches: int,
) -> tuple[float, float, int]:
    model.train()
    losses: list[float] = []
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        if isinstance(output, tuple):
            logits, loss = output
            if loss is None:
                loss = F.cross_entropy(logits, labels)
        else:
            logits = output
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy, total


@torch.no_grad()
def _validate_classifier(
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
    for batch_idx, (inputs, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        losses.append(float(F.cross_entropy(logits, labels).item()))
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


def _load_tensor_shard(
    path: Path, batch_size: int
) -> tuple[DatasetAsset, DataLoader, DataLoader]:
    data = torch.load(path, map_location="cpu")
    train_x = data["train"].float()
    val_x = data["val"].float()
    train_loader = DataLoader(
        TensorDataset(train_x), batch_size=batch_size, shuffle=True, drop_last=True
    )
    if "val_labels" in data:
        val_ds = TensorDataset(val_x, data["val_labels"].long())
    else:
        val_ds = TensorDataset(val_x)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    asset = DatasetAsset(
        name="mnist-local-tensor-shard",
        root=path.parent.resolve(),
        files=(path.resolve(),),
        sha256=f"sha256:{sha256_file(path)}",
        n_bytes=path.stat().st_size,
        source="local-tensor-shard",
    )
    return asset, train_loader, val_loader


def _batch_inputs(batch, device: torch.device) -> torch.Tensor:
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    return x.to(device).view(x.size(0), -1)


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
        x = _batch_inputs(batch, device)
        reconstruction, _ = model(x)
        loss = F.mse_loss(reconstruction, x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        samples += int(x.size(0))
    return (sum(losses) / len(losses), samples) if losses else (float("inf"), 0)


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
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = _batch_inputs(batch, device)
        reconstruction, _ = model(x)
        losses.append(float(F.mse_loss(reconstruction, x).item()))
    return sum(losses) / len(losses) if losses else float("inf")


@torch.no_grad()
def _evaluate_anomaly(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
) -> tuple[float, float, float, int]:
    """Return reconstruction MSE, AUROC, AUPRC, and evaluated sample count."""
    model.eval()
    losses: list[float] = []
    scores: list[float] = []
    labels: list[int] = []
    evaluated = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = _batch_inputs(batch, device)
        reconstruction, _ = model(x)
        per_sample = ((reconstruction - x) ** 2).mean(dim=1)
        losses.append(float(per_sample.mean().item()))
        evaluated += int(x.size(0))
        if isinstance(batch, (tuple, list)) and len(batch) > 1:
            scores.extend(float(value) for value in per_sample.detach().cpu().tolist())
            labels.extend(int(value) for value in batch[1].detach().cpu().tolist())

    mse = sum(losses) / len(losses) if losses else float("inf")
    if not labels or len(set(labels)) < 2:
        return mse, float("nan"), float("nan"), evaluated
    return (
        mse,
        _binary_auroc(scores, labels),
        _binary_average_precision(scores, labels),
        evaluated,
    )


def _binary_auroc(scores: list[float], labels: list[int]) -> float:
    """Compute binary AUROC using average ranks for tied scores."""
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
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


def _binary_average_precision(scores: list[float], labels: list[int]) -> float:
    """Compute average precision from scores sorted from most to least anomalous."""
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(label == 1 for label in labels)
    if positives == 0:
        return float("nan")
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return float(precision_sum / positives)
