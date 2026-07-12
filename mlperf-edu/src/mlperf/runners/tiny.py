from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlperf.assets import ensure_mlperf_tiny_kws, mlperf_tiny_kws_paths
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import configured_seed, synchronize_device


def run_keyword_spotting_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic smoke test of the MLPerf Tiny DS-CNN graph."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.mlperf_tiny_kws import MLPerfTinyKWS

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 4
    model = MLPerfTinyKWS().to(device).eval()
    inputs = torch.randn(batch_size, 1, 49, 10, device=device)

    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(inputs)
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
            "model_size_bytes_fp32": int(n_params * 4),
            "model_size_bytes_int8": int(n_params),
            "input_shape": list(inputs.shape),
            "logits_shape": list(logits.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": (
                "The min profile validates the pinned DS-CNN graph only. The max "
                "profile owns the official 1,000-example accuracy contract."
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


def run_keyword_spotting_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Evaluate the official MLPerf Tiny DS-CNN through its PyTorch adapter."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.mlperf_tiny_kws import load_mlperf_tiny_kws

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device(os.environ.get("MLPERF_EDU_DEVICE", "cpu"))
    batch_size = int(os.environ.get("MLPERF_EDU_KEYWORD_SPOTTING_MAX_BATCH_SIZE", "64"))
    repetitions = int(
        os.environ.get("MLPERF_EDU_KEYWORD_SPOTTING_MAX_REPETITIONS", "200")
    )
    if batch_size < 1 or repetitions < 1:
        raise ValueError(
            "keyword spotting requires positive batch size and repetitions"
        )

    asset = ensure_mlperf_tiny_kws(download=True)
    paths = mlperf_tiny_kws_paths()
    model, adapter = load_mlperf_tiny_kws(paths["float_model"], paths["int8_model"])
    model = model.to(device).eval()
    inputs, labels = _load_mlperf_tiny_kws_accuracy_set(
        asset.root,
        scale=float(adapter["input_scale"]),
        zero_point=int(adapter["input_zero_point"]),
    )
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.inference_mode():
        for warmup_size in sorted({batch_size, len(inputs) % batch_size} - {0}):
            model(inputs[:warmup_size])
    synchronize_device(device)
    outputs: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for repetition in range(repetitions):
            for start_index in range(0, len(inputs), batch_size):
                logits = model(inputs[start_index : start_index + batch_size])
                if repetition == 0:
                    outputs.append(logits.detach())
    synchronize_device(device)
    duration = time.perf_counter() - start
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(
            "keyword-spotting inference duration must be finite and positive"
        )

    predictions = torch.cat(outputs).argmax(dim=1)
    accuracy = float((predictions == labels).float().mean().item())
    target = float(workload.quality_value or 0.90)
    target_met = accuracy >= target
    n_params = sum(parameter.numel() for parameter in model.parameters())
    total_samples = len(inputs) * repetitions

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
        "model": "mlperf-tiny-ds-cnn",
        "data_mode": "real-preprocessed-mlperf-tiny-accuracy-set",
        "dataset": {
            "name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
        },
        "model_source": adapter,
        "seed": seed,
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "batch_size": batch_size,
            "repetitions": repetitions,
            "samples": len(inputs),
            "input_shape": [1, 49, 10],
            "source_input_dtype": "int8",
            "execution_dtype": "float32",
            "adapter": "fused-tflite-weights-to-pytorch-v1",
        },
        "metrics": {
            "top1_accuracy": accuracy,
            "evaluation_samples": len(inputs),
            "duration_seconds": duration,
            "inference_seconds": duration,
            "samples": total_samples,
            "samples_per_second": total_samples / duration,
            "n_params": n_params,
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
            "weights": str(paths["float_model"]),
            "source_float_model": str(paths["float_model"]),
            "source_int8_model": str(paths["int8_model"]),
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
        weights_path=paths["float_model"],
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


def _load_mlperf_tiny_kws_accuracy_set(
    dataset_root: Path, *, scale: float, zero_point: int
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[np.ndarray] = []
    labels: list[int] = []
    with (dataset_root / "y_labels.csv").open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 3:
                raise ValueError(f"invalid MLPerf Tiny KWS label row: {row}")
            path = dataset_root / row[0]
            values = np.fromfile(path, dtype=np.int8)
            if values.size != 49 * 10:
                raise ValueError(f"invalid MLPerf Tiny KWS input size: {path}")
            inputs.append(
                (values.astype(np.float32).reshape(1, 49, 10) - zero_point) * scale
            )
            labels.append(int(row[2]))
    if len(inputs) != 1000:
        raise ValueError(
            f"MLPerf Tiny KWS accuracy set expected 1000 samples, found {len(inputs)}"
        )
    return torch.from_numpy(np.stack(inputs)), torch.tensor(labels, dtype=torch.long)
