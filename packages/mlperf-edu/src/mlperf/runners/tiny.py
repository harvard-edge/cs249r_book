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

from mlperf.assets import (
    MLPERF_TINY_ANOMALY_ARCHIVE_MD5,
    MLPERF_TINY_ANOMALY_COMMIT,
    MLPERF_TINY_ANOMALY_FLOAT_MODEL_SHA256,
    MLPERF_TINY_ANOMALY_INT8_MODEL_SHA256,
    MLPERF_TINY_ANOMALY_MEMBER_MANIFEST_SHA256,
    MLPERF_TINY_VWW_COMMIT,
    MLPERF_TINY_VWW_FLOAT_MODEL_SHA256,
    MLPERF_TINY_VWW_INT8_MODEL_SHA256,
    ensure_mlperf_tiny_anomaly,
    ensure_mlperf_tiny_kws,
    ensure_mlperf_tiny_vww,
    mlperf_tiny_anomaly_paths,
    mlperf_tiny_kws_paths,
    mlperf_tiny_vww_paths,
)
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    configured_seed,
    select_torch_device,
    synchronize_device,
)


def _canonical_config_int(
    workload: Workload,
    key: str,
    environment_variable: str,
    fallback: int,
) -> int:
    """Resolve a runner knob from an override or the canonical registry config."""
    canonical_config = (workload.raw.get("canonical_max_contract") or {}).get(
        "config"
    ) or {}
    return int(
        os.environ.get(environment_variable, str(canonical_config.get(key, fallback)))
    )


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
    device = select_torch_device()
    batch_size = _canonical_config_int(
        workload,
        "batch_size",
        "MLPERF_EDU_KEYWORD_SPOTTING_MAX_BATCH_SIZE",
        64,
    )
    warmup_repetitions = _canonical_config_int(
        workload,
        "warmup_repetitions",
        "MLPERF_EDU_KEYWORD_SPOTTING_MAX_WARMUP_REPETITIONS",
        1000,
    )
    repetitions = _canonical_config_int(
        workload,
        "repetitions",
        "MLPERF_EDU_KEYWORD_SPOTTING_MAX_REPETITIONS",
        2000,
    )
    if batch_size < 1 or warmup_repetitions < 1 or repetitions < 1:
        raise ValueError(
            "keyword spotting requires positive batch size, warmup repetitions, "
            "and measured repetitions"
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
        for _ in range(warmup_repetitions):
            for start_index in range(0, len(inputs), batch_size):
                model(inputs[start_index : start_index + batch_size])
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
            "warmup_repetitions": warmup_repetitions,
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


def run_anomaly_detection_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic smoke test of the MLPerf Tiny autoencoder graph."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.mlperf_tiny_anomaly import (
        MLPerfTinyAnomalyAutoencoder,
    )

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 4
    model = MLPerfTinyAnomalyAutoencoder().to(device).eval()
    inputs = torch.randn(batch_size, 640, device=device)

    start = time.perf_counter()
    with torch.inference_mode():
        reconstructions = model(inputs)
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
            "input_shape": list(inputs.shape),
            "reconstruction_shape": list(reconstructions.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": (
                "The min profile validates the pinned dense autoencoder graph only. "
                "The max profile owns the complete 248-recording ROC AUC contract."
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
        dataset_name="synthetic-deterministic-mel-windows",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_anomaly_detection_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Evaluate the complete MLPerf Tiny ToyCar accuracy set in PyTorch."""
    from sklearn.metrics import roc_auc_score

    from mlperf.reference.tiny.mlperf_tiny_anomaly import load_mlperf_tiny_anomaly

    root = find_project_root()
    seed = configured_seed()
    torch.manual_seed(seed)
    device = select_torch_device()
    batch_size = _canonical_config_int(
        workload,
        "batch_size",
        "MLPERF_EDU_ANOMALY_DETECTION_MAX_BATCH_SIZE",
        512,
    )
    warmup_repetitions = _canonical_config_int(
        workload,
        "warmup_repetitions",
        "MLPERF_EDU_ANOMALY_DETECTION_MAX_WARMUP_REPETITIONS",
        1,
    )
    repetitions = _canonical_config_int(
        workload,
        "repetitions",
        "MLPERF_EDU_ANOMALY_DETECTION_MAX_REPETITIONS",
        10,
    )
    if batch_size < 1 or warmup_repetitions < 1 or repetitions < 1:
        raise ValueError(
            "anomaly detection requires positive batch size, warmup repetitions, "
            "and measured repetitions"
        )

    asset = ensure_mlperf_tiny_anomaly(download=True)
    paths = mlperf_tiny_anomaly_paths()
    model = load_mlperf_tiny_anomaly(paths["float_model"]).to(device).eval()
    inputs, labels, names = _load_mlperf_tiny_anomaly_accuracy_set(asset.root)
    inputs = inputs.to(device)

    with torch.inference_mode():
        for warmup_size in sorted({batch_size, len(inputs) % batch_size} - {0}):
            model(inputs[:warmup_size])
        for _ in range(warmup_repetitions):
            for start_index in range(0, len(inputs), batch_size):
                model(inputs[start_index : start_index + batch_size])
    synchronize_device(device)
    first_errors: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for repetition in range(repetitions):
            for start_index in range(0, len(inputs), batch_size):
                batch = inputs[start_index : start_index + batch_size]
                reconstruction = model(batch)
                if repetition == 0:
                    first_errors.append(
                        torch.mean((reconstruction - batch) ** 2, dim=1).cpu()
                    )
    synchronize_device(device)
    duration = time.perf_counter() - start
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("anomaly-detection inference duration must be positive")

    errors = torch.cat(first_errors).reshape(len(labels), 196).mean(dim=1).numpy()
    roc_auc = float(roc_auc_score(labels, errors))
    per_id_auc = {
        machine_id: float(
            roc_auc_score(
                [
                    label
                    for label, name in zip(labels, names, strict=True)
                    if f"id_{machine_id}_" in name
                ],
                [
                    score
                    for score, name in zip(errors, names, strict=True)
                    if f"id_{machine_id}_" in name
                ],
            )
        )
        for machine_id in ("01", "02", "03", "04")
    }
    target = float(workload.quality_value or 0.85)
    target_met = roc_auc >= target
    n_params = sum(parameter.numel() for parameter in model.parameters())
    total_windows = len(inputs) * repetitions
    model_source = {
        "authority": "MLCommons MLPerf Tiny",
        "repository": "https://github.com/mlcommons/tiny",
        "commit": MLPERF_TINY_ANOMALY_COMMIT,
        "float32_tflite_sha256": f"sha256:{MLPERF_TINY_ANOMALY_FLOAT_MODEL_SHA256}",
        "int8_tflite_sha256": f"sha256:{MLPERF_TINY_ANOMALY_INT8_MODEL_SHA256}",
        "source_archive_md5": f"md5:{MLPERF_TINY_ANOMALY_ARCHIVE_MD5}",
        "selected_member_manifest_sha256": f"sha256:{MLPERF_TINY_ANOMALY_MEMBER_MANIFEST_SHA256}",
        "adapter": "fused-tflite-weights-to-pytorch-v1",
    }

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
        "model": "mlperf-tiny-toycar-autoencoder",
        "data_mode": "real-preprocessed-mlperf-tiny-accuracy-set",
        "dataset": {
            "name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
        },
        "model_source": model_source,
        "seed": seed,
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "batch_size": batch_size,
            "warmup_repetitions": warmup_repetitions,
            "repetitions": repetitions,
            "samples": len(labels),
            "windows_per_sample": 196,
            "input_shape": [640],
            "source_input_dtype": "float32-little-endian",
            "execution_dtype": "float32",
            "feature_extractor": "librosa-0.11.0-upstream-recipe",
            "adapter": "fused-tflite-weights-to-pytorch-v1",
        },
        "metrics": {
            "roc_auc": roc_auc,
            "per_machine_id_roc_auc": per_id_auc,
            "evaluation_samples": len(labels),
            "evaluation_windows": len(inputs),
            "duration_seconds": duration,
            "inference_seconds": duration,
            "samples": total_windows,
            "samples_per_second": total_windows / duration,
            "n_params": n_params,
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "roc_auc",
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


def run_visual_wake_words_min(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Run a deterministic smoke test of the MLPerf Tiny VWW graph."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.mlperf_tiny_vww import MLPerfTinyVWW

    seed = configured_seed()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    batch_size = 4
    model = MLPerfTinyVWW().to(device).eval()
    inputs = torch.randn(batch_size, 3, 96, 96, device=device)

    start = time.perf_counter()
    with torch.inference_mode():
        probabilities = model(inputs)
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
            "input_shape": list(inputs.shape),
            "probabilities_shape": list(probabilities.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": (
                "The min profile validates the pinned MobileNetV1 graph only. "
                "The max profile owns the official 1,000-example accuracy contract."
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
        dataset_name="synthetic-deterministic-rgb-images",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_visual_wake_words_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    """Evaluate the official MLPerf Tiny VWW model through its PyTorch adapter."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from mlperf.reference.tiny.mlperf_tiny_vww import load_mlperf_tiny_vww

    seed = configured_seed()
    torch.manual_seed(seed)
    device = select_torch_device()
    batch_size = _canonical_config_int(
        workload,
        "batch_size",
        "MLPERF_EDU_VISUAL_WAKE_WORDS_MAX_BATCH_SIZE",
        64,
    )
    warmup_repetitions = _canonical_config_int(
        workload,
        "warmup_repetitions",
        "MLPERF_EDU_VISUAL_WAKE_WORDS_MAX_WARMUP_REPETITIONS",
        5,
    )
    repetitions = _canonical_config_int(
        workload,
        "repetitions",
        "MLPERF_EDU_VISUAL_WAKE_WORDS_MAX_REPETITIONS",
        50,
    )
    if batch_size < 1 or warmup_repetitions < 1 or repetitions < 1:
        raise ValueError(
            "visual wake words requires positive batch size, warmup repetitions, "
            "and measured repetitions"
        )

    asset = ensure_mlperf_tiny_vww(download=True)
    paths = mlperf_tiny_vww_paths()
    model = load_mlperf_tiny_vww(paths["float_model"]).to(device).eval()
    inputs, labels = _load_mlperf_tiny_vww_accuracy_set(asset.root)
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.inference_mode():
        for warmup_size in sorted({batch_size, len(inputs) % batch_size} - {0}):
            model(inputs[:warmup_size])
        for _ in range(warmup_repetitions):
            for start_index in range(0, len(inputs), batch_size):
                model(inputs[start_index : start_index + batch_size])
    synchronize_device(device)
    outputs: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for repetition in range(repetitions):
            for start_index in range(0, len(inputs), batch_size):
                probabilities = model(inputs[start_index : start_index + batch_size])
                if repetition == 0:
                    outputs.append(probabilities.detach())
    synchronize_device(device)
    duration = time.perf_counter() - start
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(
            "visual-wake-words inference duration must be finite and positive"
        )

    predictions = torch.cat(outputs).argmax(dim=1)
    accuracy = float((predictions == labels).float().mean().item())
    target = float(workload.quality_value or 0.80)
    target_met = accuracy >= target
    n_params = sum(parameter.numel() for parameter in model.parameters())
    total_samples = len(inputs) * repetitions
    model_source = {
        "authority": "MLCommons MLPerf Tiny",
        "repository": "https://github.com/mlcommons/tiny",
        "commit": MLPERF_TINY_VWW_COMMIT,
        "float32_tflite_sha256": f"sha256:{MLPERF_TINY_VWW_FLOAT_MODEL_SHA256}",
        "int8_tflite_sha256": f"sha256:{MLPERF_TINY_VWW_INT8_MODEL_SHA256}",
        "adapter": "fused-tflite-weights-to-pytorch-v1",
        "input_scaling": "RGB float32 divided by 255",
    }

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
        "model": "mlperf-tiny-mobilenet-v1-0.25-vww",
        "data_mode": "real-preprocessed-mlperf-tiny-accuracy-set",
        "dataset": {
            "name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
        },
        "model_source": model_source,
        "seed": seed,
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "batch_size": batch_size,
            "warmup_repetitions": warmup_repetitions,
            "repetitions": repetitions,
            "samples": len(inputs),
            "input_shape": [3, 96, 96],
            "source_input_dtype": "uint8-jpeg",
            "execution_dtype": "float32",
            "input_scaling": "divide-by-255",
            "adapter": "fused-tflite-weights-to-pytorch-v1",
        },
        "metrics": {
            "top1_accuracy": accuracy,
            "correct_predictions": int((predictions == labels).sum().item()),
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


def _load_mlperf_tiny_vww_accuracy_set(
    dataset_root: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    from PIL import Image

    inputs: list[np.ndarray] = []
    labels: list[int] = []
    with (dataset_root / "y_labels.csv").open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 3:
                raise ValueError(f"invalid MLPerf Tiny VWW label row: {row}")
            stem = Path(row[0].strip()).stem
            image_path = dataset_root / "images" / f"{stem}.jpg"
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                if rgb.size != (96, 96):
                    raise ValueError(
                        f"invalid MLPerf Tiny VWW image dimensions: {image_path}"
                    )
                inputs.append(
                    np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
                )
            labels.append(int(row[2]))
    if len(inputs) != 1000 or sum(labels) != 500:
        raise ValueError(
            "MLPerf Tiny VWW accuracy set must contain 1,000 balanced samples"
        )
    return torch.from_numpy(np.stack(inputs)), torch.tensor(labels, dtype=torch.long)


def _load_mlperf_tiny_anomaly_accuracy_set(
    dataset_root: Path,
) -> tuple[torch.Tensor, list[int], list[str]]:
    inputs: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    with (dataset_root / "y_labels.csv").open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 5 or row[1] != "2" or row[3:] != ["2560", "512"]:
                raise ValueError(f"invalid MLPerf Tiny anomaly label row: {row}")
            path = dataset_root / row[0]
            values = np.fromfile(path, dtype="<f4")
            if values.size != 200 * 128:
                raise ValueError(f"invalid MLPerf Tiny anomaly input size: {path}")
            windows = np.lib.stride_tricks.sliding_window_view(values, 640)[::128]
            if windows.shape != (196, 640):
                raise ValueError(f"invalid MLPerf Tiny anomaly windows: {path}")
            inputs.append(windows.copy())
            labels.append(int(row[2]))
            names.append(row[0])
    if len(inputs) != 248 or labels.count(0) != 140 or labels.count(1) != 108:
        raise ValueError(
            "MLPerf Tiny anomaly set must contain 140 normal and 108 anomalous samples"
        )
    return torch.from_numpy(np.concatenate(inputs)), labels, names


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
