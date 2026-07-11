from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mlperf.assets import (
    DatasetAsset,
    ensure_fashion_mnist,
    load_fashion_mnist_dataset,
    sha256_file,
)
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    configured_seed,
    synchronize_device,
    training_measurement_protocol,
)


def run_resnet18_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step ResNet-18 training smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.edge.resnet_train import ResNet18WhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 2

    model = ResNet18WhiteBox(num_classes=100).to(device)
    model.train()
    images = torch.randn(batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, 100, (batch_size,), dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    accuracy = float((logits.detach().argmax(dim=1) == labels).float().mean().item())
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "resnet18-train_min_report.json").resolve()
    manifest_path = (output_dir / "resnet18-train_min.provd.json").resolve()
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
            "logits_shape": list(logits.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "min profile validates ResNet execution only; max profile owns Fashion-MNIST quality checks.",
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
        dataset_name="synthetic-deterministic-images",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_mobilenetv2_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic one-step MobileNetV2 training smoke."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.mobile.mobilenet_core import MobileNetV2Local

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 2

    model = MobileNetV2Local(num_classes=100).to(device)
    model.train()
    images = torch.randn(batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, 100, (batch_size,), dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

    accuracy = float((logits.detach().argmax(dim=1) == labels).float().mean().item())
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "mobilenetv2-train_min_report.json").resolve()
    manifest_path = (output_dir / "mobilenetv2-train_min.provd.json").resolve()
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
            "logits_shape": list(logits.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "min profile validates MobileNetV2 execution only; max profile will own Fashion-MNIST quality checks.",
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
        dataset_name="synthetic-deterministic-images",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_mobilenet_composed_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    return run_mobilenet_composed(workload, output_dir, profile="min", repetitions=1)


def run_mobilenet_composed_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    repetitions = int(os.environ.get("MLPERF_EDU_MOBILENET_COMP_REPETITIONS", "5"))
    return run_mobilenet_composed(
        workload, output_dir, profile="max", repetitions=repetitions
    )


def run_mobilenet_composed(
    workload: Workload, output_dir: Path, *, profile: str, repetitions: int
) -> dict[str, Any]:
    """Run fp16 MobileNetV2 with composed pruning and fake-int8 accounting."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.mobile.mobilenet_compress import (
        effective_param_bytes,
        fake_quantize_int8,
        prune_2of4,
    )
    from mlperf.reference.mobile.mobilenet_core import MobileNetV2Local

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = int(os.environ.get("MLPERF_EDU_MOBILENET_COMP_BATCH_SIZE", "2"))
    if batch_size < 1 or repetitions < 1:
        raise ValueError(
            "composed MobileNet requires positive batch and repetition counts"
        )

    model = MobileNetV2Local(num_classes=100).to(device=device, dtype=torch.float16)
    model.eval()
    baseline_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    sparsity = prune_2of4(model)
    quant = fake_quantize_int8(model)
    effective_bytes = effective_param_bytes(model)

    images = torch.randn(batch_size, 3, 32, 32, device=device, dtype=torch.float16)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = None
        for _ in range(repetitions):
            logits = model(images)
    duration = time.perf_counter() - start
    assert logits is not None

    n_params = sum(p.numel() for p in model.parameters())
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_{profile}_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_{profile}.provd.json").resolve()
    functional_target = 1.0
    functional_met = bool(
        repetitions >= 1
        and duration > 0
        and torch.isfinite(logits).all().item()
        and list(logits.shape) == [batch_size, 100]
        and baseline_param_bytes > effective_bytes > 0
        and baseline_param_bytes / effective_bytes > functional_target
        and all(parameter.dtype == torch.float16 for parameter in model.parameters())
    )
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": profile,
        "status": "passed" if functional_met else "quality_failed",
        "backend": "pytorch-cpu-fp16",
        "data_mode": "synthetic-deterministic",
        "seed": seed,
        "compression": {
            "structured_sparsity": "2:4",
            "quantization": "fake-int8",
            "execution_precision": "fp16",
            "runtime_note": "The dense runtime is fp16. Structured sparsity and int8 are algorithmic storage accounting only; this path does not use fused sparse/int8 kernels.",
        },
        "metrics": {
            "duration_seconds": float(duration),
            "samples": int(batch_size * repetitions),
            "samples_per_second": float((batch_size * repetitions) / duration)
            if duration > 0
            else 0.0,
            "n_params": int(n_params),
            "baseline_param_bytes": int(baseline_param_bytes),
            "effective_param_bytes": int(effective_bytes),
            "effective_compression_ratio": float(baseline_param_bytes / effective_bytes)
            if effective_bytes
            else 0.0,
            "sparsity_actual": float(sparsity["sparsity_actual"]),
            "n_quantized_params": int(quant["n_quantized_params"]),
            "logits_shape": list(logits.shape),
            "repetitions": repetitions,
            "execution_dtype": "float16",
            "functional_check_met": functional_met,
        },
        "quality": {
            "metric": "effective_compression_ratio",
            "metric_key": "effective_compression_ratio",
            "target": functional_target,
            "direction": "higher",
            "target_met": functional_met,
            "quality_required": True,
            "note": "The systems-only functional gate requires finite fp16 logits and a computed packed-storage ratio above one. It does not claim task accuracy or kernel speedup.",
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "offline",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name="synthetic-deterministic-images",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_resnet18_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run ResNet-18 on Fashion-MNIST or an explicit local tensor shard."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.edge.resnet_train import ResNet18WhiteBox

    seed = configured_seed()
    torch.manual_seed(seed)
    device = _select_device()
    batch_size = int(os.environ.get("MLPERF_EDU_RESNET_MAX_BATCH_SIZE", 64))
    epochs = int(os.environ.get("MLPERF_EDU_RESNET_MAX_EPOCHS", 5))
    batches_per_epoch = int(
        os.environ.get("MLPERF_EDU_RESNET_MAX_BATCHES_PER_EPOCH", 100)
    )
    requested_val_batches = int(os.environ.get("MLPERF_EDU_RESNET_MAX_VAL_BATCHES", 0))
    lr = float(os.environ.get("MLPERF_EDU_RESNET_MAX_LR", 1e-3))

    shard_path = os.environ.get("MLPERF_EDU_RESNET_MAX_TENSOR_PATH")
    if shard_path:
        asset, train_loader, val_loader = _load_tensor_shard(
            Path(shard_path), batch_size
        )
        num_classes = 100
    else:
        asset, train_loader, val_loader = _fashion_mnist_loaders(
            batch_size=batch_size, seed=seed
        )
        num_classes = 10

    val_batches = requested_val_batches or len(val_loader)

    model = ResNet18WhiteBox(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
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
        val_loss, val_acc, evaluated_samples = _validate(
            model, val_loader, device, max_batches=val_batches
        )
        samples_seen += train_samples
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        epoch_times.append(time.perf_counter() - t0)
    synchronize_device(device)
    duration = time.perf_counter() - start
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(
            "ResNet-18 train-and-evaluation duration must be finite and positive"
        )

    final_accuracy = val_accuracies[-1]
    target = float(
        os.environ.get(
            "MLPERF_EDU_RESNET_MAX_ACCURACY_TARGET", workload.quality_value or 0.36
        )
    )
    target_met = final_accuracy >= target
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "resnet18-train_max_report.json").resolve()
    manifest_path = (output_dir / "resnet18-train_max.provd.json").resolve()
    checkpoint_path = (output_dir / "resnet18-train_max_checkpoint.pt").resolve()
    torch.save(model.state_dict(), checkpoint_path)

    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
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
        "measurement_protocol": training_measurement_protocol(workload),
        "config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "batches_per_epoch": batches_per_epoch,
            "val_batches": val_batches,
            "lr": lr,
        },
        "metrics": {
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
            "final_accuracy": float(final_accuracy),
            "top1_accuracy": float(final_accuracy),
            "evaluation_samples": int(evaluated_samples),
            "duration_seconds": float(duration),
            "train_and_eval_seconds": float(duration),
            "samples": int(samples_seen),
            "samples_per_second": float(samples_seen / duration)
            if duration > 0
            else 0.0,
            "n_params": int(n_params),
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "top1_accuracy",
            "target": target,
            "direction": "higher",
            "target_met": target_met,
            "quality_required": True,
            "override": "MLPERF_EDU_RESNET_MAX_ACCURACY_TARGET" in os.environ,
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


def run_mobilenetv2_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run MobileNetV2 on Fashion-MNIST or an explicit local tensor shard."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.mobile.mobilenet_core import MobileNetV2Local

    seed = configured_seed()
    torch.manual_seed(seed)
    device = _select_device()
    batch_size = int(os.environ.get("MLPERF_EDU_MOBILENET_MAX_BATCH_SIZE", 64))
    epochs = int(os.environ.get("MLPERF_EDU_MOBILENET_MAX_EPOCHS", 8))
    batches_per_epoch = int(
        os.environ.get("MLPERF_EDU_MOBILENET_MAX_BATCHES_PER_EPOCH", 100)
    )
    requested_val_batches = int(
        os.environ.get("MLPERF_EDU_MOBILENET_MAX_VAL_BATCHES", 0)
    )
    lr = float(os.environ.get("MLPERF_EDU_MOBILENET_MAX_LR", 1e-4))

    shard_path = os.environ.get("MLPERF_EDU_MOBILENET_MAX_TENSOR_PATH")
    if shard_path:
        asset, train_loader, val_loader = _load_tensor_shard(
            Path(shard_path), batch_size
        )
        num_classes = 100
    else:
        asset, train_loader, val_loader = _fashion_mnist_loaders(
            batch_size=batch_size, seed=seed
        )
        num_classes = 10

    val_batches = requested_val_batches or len(val_loader)

    model = MobileNetV2Local(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
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
        val_loss, val_acc, evaluated_samples = _validate(
            model, val_loader, device, max_batches=val_batches
        )
        samples_seen += train_samples
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        epoch_times.append(time.perf_counter() - t0)
    synchronize_device(device)
    duration = time.perf_counter() - start
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(
            "MobileNetV2 train-and-evaluation duration must be finite and positive"
        )

    final_accuracy = val_accuracies[-1]
    target = float(
        os.environ.get(
            "MLPERF_EDU_MOBILENET_MAX_ACCURACY_TARGET", workload.quality_value or 0.4
        )
    )
    target_met = final_accuracy >= target
    n_params = sum(p.numel() for p in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / "mobilenetv2-train_max_report.json").resolve()
    manifest_path = (output_dir / "mobilenetv2-train_max.provd.json").resolve()
    checkpoint_path = (output_dir / "mobilenetv2-train_max_checkpoint.pt").resolve()
    torch.save(model.state_dict(), checkpoint_path)

    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
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
        "measurement_protocol": training_measurement_protocol(workload),
        "config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "batches_per_epoch": batches_per_epoch,
            "val_batches": val_batches,
            "lr": lr,
        },
        "metrics": {
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
            "final_accuracy": float(final_accuracy),
            "top1_accuracy": float(final_accuracy),
            "evaluation_samples": int(evaluated_samples),
            "duration_seconds": float(duration),
            "train_and_eval_seconds": float(duration),
            "samples": int(samples_seen),
            "samples_per_second": float(samples_seen / duration)
            if duration > 0
            else 0.0,
            "n_params": int(n_params),
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "top1_accuracy",
            "target": target,
            "direction": "higher",
            "target_met": target_met,
            "quality_required": True,
            "override": "MLPERF_EDU_MOBILENET_MAX_ACCURACY_TARGET" in os.environ,
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


def _select_device() -> torch.device:
    requested = os.environ.get("MLPERF_EDU_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fashion_mnist_loaders(
    batch_size: int, *, seed: int
) -> tuple[DatasetAsset, DataLoader, DataLoader]:
    asset = ensure_fashion_mnist(download=True)
    import torchvision.transforms as transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
        ]
    )
    train_ds = load_fashion_mnist_dataset(
        root=asset.root, train=True, download=False, transform=train_transform
    )
    val_ds = load_fashion_mnist_dataset(
        root=asset.root, train=False, download=False, transform=val_transform
    )
    return (
        asset,
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        ),
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0
        ),
    )


def _load_tensor_shard(
    path: Path, batch_size: int
) -> tuple[DatasetAsset, DataLoader, DataLoader]:
    data = torch.load(path, map_location="cpu")
    train_ds = TensorDataset(data["train_images"].float(), data["train_labels"].long())
    val_ds = TensorDataset(data["val_images"].float(), data["val_labels"].long())
    asset = DatasetAsset(
        name="cifar100-local-tensor-shard",
        root=path.parent.resolve(),
        files=(path.resolve(),),
        sha256=f"sha256:{sha256_file(path)}",
        n_bytes=path.stat().st_size,
        source="local-tensor-shard",
    )
    return (
        asset,
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
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
    samples = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
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
) -> tuple[float, float, int]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        losses.append(float(loss.item()))
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy, total
