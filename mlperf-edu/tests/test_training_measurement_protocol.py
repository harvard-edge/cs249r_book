from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from mlperf.registry import load_registry
from mlperf.runners import tiny, vision


@pytest.mark.parametrize(
    ("workload_id", "runner_name", "model_module", "model_name", "env_prefix"),
    [
        (
            "resnet18-train",
            "run_resnet18_max",
            "mlperf.reference.edge.resnet_train",
            "ResNet18WhiteBox",
            "RESNET",
        ),
        (
            "mobilenetv2-train",
            "run_mobilenetv2_max",
            "mlperf.reference.mobile.mobilenet_core",
            "MobileNetV2Local",
            "MOBILENET",
        ),
    ],
)
def test_vision_training_measurement_uses_exact_synchronized_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workload_id: str,
    runner_name: str,
    model_module: str,
    model_name: str,
    env_prefix: str,
) -> None:
    shard_path = tmp_path / f"{workload_id}.pt"
    torch.save(
        {
            "train_images": torch.randn(4, 3, 32, 32),
            "train_labels": torch.tensor([0, 1, 2, 3]),
            "val_images": torch.randn(4, 3, 32, 32),
            "val_labels": torch.tensor([0, 1, 2, 3]),
        },
        shard_path,
    )
    output_dir = tmp_path / "out"
    events: list[str] = []

    original_load = vision._load_tensor_shard

    def recording_load(path: Path, batch_size: int):
        events.append("asset")
        return original_load(path, batch_size)

    reference_module = importlib.import_module(model_module)
    original_model = getattr(reference_module, model_name)

    def recording_model(*args: Any, **kwargs: Any):
        events.append("model")
        return original_model(*args, **kwargs)

    def recording_train(*args: Any, **kwargs: Any) -> tuple[float, int]:
        events.append("train")
        return 0.25, 2

    def recording_validate(*args: Any, **kwargs: Any) -> tuple[float, float, int]:
        events.append("validate")
        return 0.20, 0.90, 2

    def recording_sync(device: torch.device) -> None:
        assert device.type == "cpu"
        events.append("sync")

    original_save = torch.save

    def recording_save(obj: Any, path: Path) -> None:
        events.append("checkpoint")
        original_save(obj, path)

    original_write_text = Path.write_text

    def recording_write_text(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.parent == output_dir:
            events.append(f"write:{path.name}")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(vision, "_load_tensor_shard", recording_load)
    monkeypatch.setattr(reference_module, model_name, recording_model)
    monkeypatch.setattr(vision, "_train_epoch", recording_train)
    monkeypatch.setattr(vision, "_validate", recording_validate)
    monkeypatch.setattr(vision, "synchronize_device", recording_sync)
    monkeypatch.setattr(torch, "save", recording_save)
    monkeypatch.setattr(Path, "write_text", recording_write_text)
    monkeypatch.setenv("MLPERF_EDU_DEVICE", "cpu")
    monkeypatch.setenv(f"MLPERF_EDU_{env_prefix}_MAX_TENSOR_PATH", str(shard_path))
    monkeypatch.setenv(f"MLPERF_EDU_{env_prefix}_MAX_BATCH_SIZE", "2")
    monkeypatch.setenv(f"MLPERF_EDU_{env_prefix}_MAX_EPOCHS", "1")
    monkeypatch.setenv(f"MLPERF_EDU_{env_prefix}_MAX_BATCHES_PER_EPOCH", "1")
    monkeypatch.setenv(f"MLPERF_EDU_{env_prefix}_MAX_VAL_BATCHES", "1")
    monkeypatch.setenv(f"MLPERF_EDU_{env_prefix}_MAX_ACCURACY_TARGET", "0.0")

    workload = load_registry()[workload_id]
    report = getattr(vision, runner_name)(workload, output_dir)

    assert events == [
        "asset",
        "model",
        "sync",
        "train",
        "validate",
        "sync",
        "checkpoint",
        f"write:{workload_id}_max_report.json",
        f"write:{workload_id}_max.provd.json",
    ]
    duration = report["metrics"]["train_and_eval_seconds"]
    assert math.isfinite(duration) and duration > 0
    assert duration == report["metrics"]["duration_seconds"]
    assert report["measurement_protocol"] == workload.raw["measurement_protocol"]
    assert report["measurement_protocol"] is not workload.raw["measurement_protocol"]


def test_anomaly_training_measurement_uses_exact_synchronized_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard_path = tmp_path / "anomaly.pt"
    torch.save(
        {
            "train": torch.rand(8, 784),
            "val": torch.rand(4, 784),
        },
        shard_path,
    )
    output_dir = tmp_path / "out"
    events: list[str] = []

    original_load = tiny._load_tensor_shard

    def recording_load(path: Path, batch_size: int):
        events.append("asset")
        return original_load(path, batch_size)

    reference_module = importlib.import_module(
        "mlperf.reference.tiny.anomaly_detection_ae"
    )
    original_model = reference_module.AnomalyDetectionAE

    def recording_model(*args: Any, **kwargs: Any):
        events.append("model")
        return original_model(*args, **kwargs)

    def recording_train(*args: Any, **kwargs: Any) -> tuple[float, int]:
        events.append("train")
        return 0.25, 4

    def recording_validate(
        *args: Any, **kwargs: Any
    ) -> tuple[float, float, float, int]:
        events.append("validate")
        return 0.20, float("nan"), float("nan"), 4

    def recording_sync(device: torch.device) -> None:
        assert device.type == "cpu"
        events.append("sync")

    original_save = torch.save

    def recording_save(obj: Any, path: Path) -> None:
        events.append("checkpoint")
        original_save(obj, path)

    original_write_text = Path.write_text

    def recording_write_text(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.parent == output_dir:
            events.append(f"write:{path.name}")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(tiny, "_load_tensor_shard", recording_load)
    monkeypatch.setattr(reference_module, "AnomalyDetectionAE", recording_model)
    monkeypatch.setattr(tiny, "_train_epoch", recording_train)
    monkeypatch.setattr(tiny, "_evaluate_anomaly", recording_validate)
    monkeypatch.setattr(tiny, "synchronize_device", recording_sync)
    monkeypatch.setattr(torch, "save", recording_save)
    monkeypatch.setattr(Path, "write_text", recording_write_text)
    monkeypatch.setenv("MLPERF_EDU_DEVICE", "cpu")
    monkeypatch.setenv("MLPERF_EDU_ANOMALY_MAX_TENSOR_PATH", str(shard_path))
    monkeypatch.setenv("MLPERF_EDU_ANOMALY_MAX_BATCH_SIZE", "4")
    monkeypatch.setenv("MLPERF_EDU_ANOMALY_MAX_EPOCHS", "1")
    monkeypatch.setenv("MLPERF_EDU_ANOMALY_MAX_BATCHES_PER_EPOCH", "1")
    monkeypatch.setenv("MLPERF_EDU_ANOMALY_MAX_VAL_BATCHES", "1")

    workload = load_registry()["anomaly-ae-train"]
    report = tiny.run_anomaly_ae_max(workload, output_dir)

    assert events == [
        "asset",
        "model",
        "sync",
        "train",
        "validate",
        "sync",
        "checkpoint",
        "write:anomaly-ae-train_max_report.json",
        "write:anomaly-ae-train_max.provd.json",
    ]
    duration = report["metrics"]["train_and_eval_seconds"]
    assert math.isfinite(duration) and duration > 0
    assert duration == report["metrics"]["duration_seconds"]
    assert report["measurement_protocol"] == workload.raw["measurement_protocol"]
    assert report["measurement_protocol"] is not workload.raw["measurement_protocol"]
