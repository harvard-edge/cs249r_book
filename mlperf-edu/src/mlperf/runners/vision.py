from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mlperf.assets import (
    MLPERF_TINY_COMMIT,
    ensure_cifar10,
    ensure_mlperf_tiny_image,
    load_cifar10_dataset,
    mlperf_tiny_image_paths,
    sha256_file,
)
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    configured_seed,
    select_torch_device,
    synchronize_device,
)


def run_image_classification_min(
    workload: Workload, output_dir: Path
) -> dict[str, Any]:
    """Run a deterministic smoke test of the MLPerf Tiny fused graph."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.mlperf_tiny_resnet import MLPerfTinyFusedResNet8

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 2
    model = MLPerfTinyFusedResNet8().to(device).eval()
    images = torch.randn(batch_size, 3, 32, 32, device=device)

    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(images)
    duration = time.perf_counter() - start
    n_params = sum(parameter.numel() for parameter in model.parameters())

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_min_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_min.provd.json").resolve()
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
            "note": (
                "The min profile validates the fused ResNet8 graph only. It does "
                "not support a quality claim."
            ),
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


def run_image_classification_max(
    workload: Workload, output_dir: Path
) -> dict[str, Any]:
    """Evaluate the official MLPerf Tiny float ResNet8 on its accuracy set."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.mlperf_tiny_resnet import (
        MLPERF_TINY_FLOAT_MODEL_SHA256,
        load_mlperf_tiny_float_resnet,
    )

    seed = configured_seed()
    torch.manual_seed(seed)
    device = select_torch_device()
    batch_size = int(
        os.environ.get("MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE", "32")
    )
    repetitions = int(
        os.environ.get("MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_REPETITIONS", "50")
    )
    if batch_size <= 0 or repetitions <= 0:
        raise ValueError(
            "image-classification batch size and repetitions must be positive"
        )

    dataset_asset = ensure_cifar10(download=True)
    evaluation_asset = ensure_mlperf_tiny_image(download=True)
    evaluation_paths = mlperf_tiny_image_paths()
    canonical_contract = workload.raw.get("canonical_max_contract") or {}
    evaluator_contract = canonical_contract.get("evaluator") or {}
    if not isinstance(evaluator_contract, dict) or not evaluator_contract:
        raise ValueError("image-classification evaluator contract is missing")
    evaluator_digest = f"sha256:{sha256_file(evaluation_paths['evaluator_source'])}"
    if (
        evaluator_contract.get("revision") != MLPERF_TINY_COMMIT
        or evaluator_contract.get("source_sha256") != evaluator_digest
    ):
        raise ValueError(
            "image-classification evaluator asset differs from the canonical contract"
        )
    indices = np.load(evaluation_paths["performance_indices"], allow_pickle=False)
    if indices.shape != (200,) or len(np.unique(indices)) != 200:
        raise ValueError(
            "MLPerf Tiny image accuracy set must contain 200 unique indices"
        )
    test_dataset = load_cifar10_dataset(
        root=dataset_asset.root,
        train=False,
        download=False,
        transform=_cifar10_raw_float_tensor,
    )
    if int(indices.min()) < 0 or int(indices.max()) >= len(test_dataset):
        raise ValueError("MLPerf Tiny image accuracy indices are outside CIFAR-10")
    accuracy_dataset = Subset(test_dataset, indices.astype(np.int64).tolist())
    loader = DataLoader(
        accuracy_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    model = load_mlperf_tiny_float_resnet(evaluation_paths["float_model"]).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())

    with torch.inference_mode():
        warmed_batch_sizes: set[int] = set()
        for images, _ in loader:
            current_batch_size = int(images.shape[0])
            if current_batch_size in warmed_batch_sizes:
                continue
            model(images.to(device))
            warmed_batch_sizes.add(current_batch_size)
    synchronize_device(device)
    start = time.perf_counter()
    correct = 0
    evaluated_samples = 0
    with torch.inference_mode():
        for repetition in range(repetitions):
            for images, labels in loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                if repetition == 0:
                    correct += int((predictions == labels).sum().item())
                    evaluated_samples += int(labels.numel())
    synchronize_device(device)
    duration = time.perf_counter() - start
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("MLPerf Tiny ResNet8 inference duration must be positive")
    if evaluated_samples != 200:
        raise RuntimeError(
            f"MLPerf Tiny image evaluation expected 200 samples, got {evaluated_samples}"
        )

    top1_accuracy = correct / evaluated_samples
    total_samples = evaluated_samples * repetitions
    target = float(workload.quality_value or 0.85)
    target_met = top1_accuracy >= target
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_max_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_max.provd.json").resolve()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "model": "mlperf-tiny-pretrained-resnet8-float",
        "data_mode": "real",
        "dataset": {
            "name": dataset_asset.name,
            "source": dataset_asset.source,
            "root": str(dataset_asset.root),
            "sha256": dataset_asset.sha256,
            "n_bytes": dataset_asset.n_bytes,
            "split": "MLPerf-Tiny-200-sample-accuracy-set",
            "performance_indices_sha256": (
                f"sha256:{sha256_file(evaluation_paths['performance_indices'])}"
            ),
        },
        "model_source": {
            "repository": "https://github.com/mlcommons/tiny",
            "commit": "1afd2c9820f795965a6134facd0b4dfae41ef23f",
            "path": "benchmark/training/image_classification/trained_models/pretrainedResnet.tflite",
            "sha256": f"sha256:{MLPERF_TINY_FLOAT_MODEL_SHA256}",
            "format": "official-float32-tflite-weights-loaded-into-exact-pytorch-graph",
        },
        "evaluation_bundle": {
            "name": evaluation_asset.name,
            "source": evaluation_asset.source,
            "root": str(evaluation_asset.root),
            "sha256": evaluation_asset.sha256,
            "n_bytes": evaluation_asset.n_bytes,
        },
        "evaluator": dict(evaluator_contract),
        "seed": seed,
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "batch_size": batch_size,
            "repetitions": repetitions,
            "evaluation_samples": 200,
            "input_dtype": "float32",
            "input_range": "0..255",
            "input_layout": "NCHW",
            "scenario": "offline",
        },
        "metrics": {
            "top1_accuracy": float(top1_accuracy),
            "correct": correct,
            "evaluation_samples": int(evaluated_samples),
            "duration_seconds": float(duration),
            "inference_and_evaluation_seconds": float(duration),
            "samples": int(total_samples),
            "samples_per_second": float(total_samples / duration),
            "n_params": int(n_params),
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "top1_accuracy",
            "target": target,
            "direction": "higher",
            "target_met": target_met,
            "quality_required": True,
            "override": False,
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
            "weights": str(evaluation_paths["float_model"]),
            "performance_indices": str(evaluation_paths["performance_indices"]),
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
        weights_path=evaluation_paths["float_model"],
        weights_n_params=n_params,
        weights_dtype="float32",
        dataset_name=dataset_asset.name,
        dataset_files=[*dataset_asset.files, evaluation_paths["performance_indices"]],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def _cifar10_raw_float_tensor(image: Any) -> torch.Tensor:
    """Convert a CIFAR image to the raw 0..255 float input used upstream."""
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1)
