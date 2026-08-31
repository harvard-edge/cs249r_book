#!/usr/bin/env python3
"""Lab 1. Measure ResNet-18 training-loop optimizations.

This lab is an educational experiment, not a score-bearing MLPerf EDU run.
Use the product CLI for a canonical image-classification artifact:

    mlperf run --workload image-classification --profile min

The ``--smoke`` path is deterministic, CPU-only, and network-free. It runs a
real ResNet-18 forward pass, backward pass, optimizer update, and validation
pass so CI can check the complete lab entry point quickly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "batch_size": 8,
        "num_workers": 0,
        "learning_rate": 0.05,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "augmentation": False,
        "schedule": False,
    },
    "optimized": {
        "batch_size": 64,
        "num_workers": min(4, os.cpu_count() or 1),
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "augmentation": True,
        "schedule": True,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ResNet-18 training optimization lab."
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="baseline")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a deterministic, CPU-only, network-free functional smoke.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument(
        "--augmentation", dest="augmentation", action="store_true"
    )
    augmentation.add_argument(
        "--no-augmentation", dest="augmentation", action="store_false"
    )
    parser.set_defaults(augmentation=None)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--schedule", dest="schedule", action="store_true")
    schedule.add_argument("--no-schedule", dest="schedule", action="store_false")
    parser.set_defaults(schedule=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=100,
        help="Maximum optimizer steps per epoch. Use 0 for the whole loader.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=50,
        help="Maximum validation batches. Use 0 for the whole loader.",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=None,
        help="Optional classroom target. It is not an official benchmark threshold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional JSON result path."
    )
    return parser


def positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(PRESETS[args.preset])
    for name in (
        "batch_size",
        "num_workers",
        "learning_rate",
        "momentum",
        "weight_decay",
        "augmentation",
        "schedule",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = value

    positive("batch size", config["batch_size"])
    positive("learning rate", config["learning_rate"])
    positive("epochs", args.epochs)
    if config["num_workers"] < 0:
        raise ValueError("num workers cannot be negative")
    if args.max_train_batches < 0 or args.max_validation_batches < 0:
        raise ValueError("batch limits cannot be negative")
    if args.target_accuracy is not None and not 0.0 <= args.target_accuracy <= 1.0:
        raise ValueError("target accuracy must be in [0, 1]")

    if args.smoke:
        config.update(batch_size=2, num_workers=0, augmentation=False)
    return config


def choose_device(requested: str, *, smoke: bool) -> torch.device:
    if smoke or requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError(f"unsupported device: {requested}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synthetic_loaders(batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
    """Return a tiny deterministic shard without accessing Fashion-MNIST."""
    generator = torch.Generator().manual_seed(seed)
    sample_count = max(4, batch_size * 2)
    images = torch.randn(sample_count, 3, 32, 32, generator=generator)
    labels = torch.arange(sample_count, dtype=torch.long) % 10
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, loader


def fashion_mnist_transforms(*, augmentation: bool) -> tuple[Any, Any]:
    """Build the registered Fashion-MNIST transforms for this lab."""
    import torchvision.transforms as transforms

    common: list[Any] = [
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
    ]
    train_operations = list(common)
    if augmentation:
        train_operations.extend(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )
    normalization = transforms.Normalize(
        (0.2860, 0.2860, 0.2860),
        (0.3530, 0.3530, 0.3530),
    )
    train_operations.extend([transforms.ToTensor(), normalization])
    validation_operations = [*common, transforms.ToTensor(), normalization]
    return transforms.Compose(train_operations), transforms.Compose(
        validation_operations
    )


def load_data(
    *, config: dict[str, Any], smoke: bool, seed: int
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    if smoke:
        train_loader, validation_loader = synthetic_loaders(config["batch_size"], seed)
        return (
            train_loader,
            validation_loader,
            {
                "name": "synthetic-deterministic",
                "source": "generated by the Lab 1 smoke path",
                "sha256": None,
            },
        )

    from mlperf.assets import ensure_fashion_mnist, load_fashion_mnist_dataset

    asset = ensure_fashion_mnist(download=True)
    train_transform, validation_transform = fashion_mnist_transforms(
        augmentation=config["augmentation"]
    )
    train_dataset = load_fashion_mnist_dataset(
        root=asset.root,
        train=True,
        download=False,
        transform=train_transform,
    )
    validation_dataset = load_fashion_mnist_dataset(
        root=asset.root,
        train=False,
        download=False,
        transform=validation_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        train_loader,
        validation_loader,
        {
            "name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
        },
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int,
) -> dict[str, float | int]:
    model.train()
    start = time.perf_counter()
    total_loss = 0.0
    correct = 0
    samples = 0
    steps = 0
    for step, (features, labels) in enumerate(loader, start=1):
        if max_batches and step > max_batches:
            break
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        samples += labels.numel()
        steps += 1
    duration = time.perf_counter() - start
    if not steps or not samples:
        raise RuntimeError("training loader produced no batches")
    return {
        "loss": total_loss / samples,
        "accuracy": correct / samples,
        "samples": samples,
        "optimizer_steps": steps,
        "duration_seconds": duration,
        "samples_per_second": samples / duration,
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    correct = 0
    samples = 0
    with torch.inference_mode():
        for step, (features, labels) in enumerate(loader, start=1):
            if max_batches and step > max_batches:
                break
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("validation produced a non-finite loss")
            total_loss += float(loss.item()) * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            samples += labels.numel()
    if not samples:
        raise RuntimeError("validation loader produced no batches")
    return {
        "loss": total_loss / samples,
        "accuracy": correct / samples,
        "samples": samples,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        args.epochs = 1
        args.max_train_batches = 1
        args.max_validation_batches = 1
    config = resolve_config(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device, smoke=args.smoke)

    from torchvision.models import resnet18

    train_loader, validation_loader, dataset = load_data(
        config=config, smoke=args.smoke, seed=args.seed
    )
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if config["schedule"]
        else None
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.max_train_batches,
        )
        metrics["epoch"] = epoch
        history.append(metrics)
        if scheduler is not None:
            scheduler.step()

    validation = evaluate(
        model,
        validation_loader,
        device,
        args.max_validation_batches,
    )
    finite = all(
        math.isfinite(float(value))
        for value in (history[-1]["loss"], validation["loss"], validation["accuracy"])
    )
    if not finite:
        raise RuntimeError("functional check failed because a metric is not finite")

    target_passed = (
        None
        if args.target_accuracy is None
        else validation["accuracy"] >= args.target_accuracy
    )
    result: dict[str, Any] = {
        "schema": "mlperf-edu-lab-result/0.1",
        "lab": "lab1-training-optimization",
        "status": "quality-failed" if target_passed is False else "passed",
        "result_scope": "functional-smoke" if args.smoke else "classroom-experiment",
        "canonical_result": False,
        "seed": args.seed,
        "device": str(device),
        "model": {
            "name": "resnet18-whitebox",
            "num_classes": 10,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "data_mode": "synthetic" if args.smoke else "real",
        "dataset": dataset,
        "preset": args.preset,
        "config": config,
        "epochs": args.epochs,
        "history": history,
        "validation": validation,
        "functional_check": {"finite_loss_and_accuracy": True, "passed": True},
        "classroom_target": {
            "accuracy": args.target_accuracy,
            "passed": target_passed,
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (RuntimeError, ValueError) as exc:
        print(f"LAB 1 FAIL: {exc}", file=sys.stderr)
        return 1

    train = result["history"][-1]
    validation = result["validation"]
    print("MLPerf EDU Lab 1")
    print(f"  scope: {result['result_scope']}")
    print(f"  preset: {result['preset']} on {result['device']}")
    print(f"  train loss: {train['loss']:.4f}")
    print(f"  train throughput: {train['samples_per_second']:.2f} samples/s")
    print(f"  validation accuracy: {validation['accuracy']:.2%}")
    if result["classroom_target"]["accuracy"] is not None:
        state = "passed" if result["classroom_target"]["passed"] else "not reached"
        print(f"  optional classroom target: {state}")
        if result["classroom_target"]["passed"] is False:
            print("LAB 1 QUALITY TARGET NOT REACHED", file=sys.stderr)
            return 2
    print("LAB 1 SMOKE PASS" if args.smoke else "LAB 1 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
