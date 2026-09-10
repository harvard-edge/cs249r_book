from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mlperf.assets import ensure_ettm1
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.reference.timeseries.patchtst import PatchTST_backbone
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    TrainingProgress,
    configured_seed,
    select_torch_device,
    synchronize_device,
)


PATCHTST_COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
ETT_DATASET_COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
PATCHTST_BACKBONE_SHA256 = (
    "df67173153787c2356bdfb6491159cd754332ef7382986efe879e1fbea8ebf26"
)
PATCHTST_LAYERS_SHA256 = (
    "21c06c70a90c60ee2a269b5c600c702834dea22cdfd72915e6b0f8b4a28db3f6"
)
PATCHTST_REVIN_SHA256 = (
    "e64c0ccded9228b347134e7368420d3fb10f70c75145b5ff2d8bdd8c3af59df6"
)
PATCHTST_SCRIPT_SHA256 = (
    "6b4fd17c3da9471f29556b92f7c788e667a108548ebb0919d24ecea4b3d36aa5"
)
PATCHTST_EXPERIMENT_SHA256 = (
    "de76924d0dc15c50a9b199a094f1362bdc4975566d3bb2a854ed2298185086cc"
)
PATCHTST_DATA_LOADER_SHA256 = (
    "1d49e2207ae1fa5ae7c1ddbe4be0f38e9b2a73ebc851a16e02681f26f843905e"
)


class ETTm1WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Official PatchTST ETTm1 window and split semantics."""

    def __init__(
        self,
        values: np.ndarray,
        *,
        split: str,
        context_length: int,
        prediction_length: int,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown ETTm1 split: {split}")
        train_end = 12 * 30 * 24 * 4
        validation_end = train_end + 4 * 30 * 24 * 4
        test_end = train_end + 8 * 30 * 24 * 4
        borders = {
            "train": (0, train_end),
            "validation": (train_end - context_length, validation_end),
            "test": (validation_end - context_length, test_end),
        }
        start, end = borders[split]
        self.values = torch.from_numpy(values[start:end].astype(np.float32, copy=False))
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self) -> int:
        return len(self.values) - self.context_length - self.prediction_length + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        context_end = index + self.context_length
        prediction_end = context_end + self.prediction_length
        return self.values[index:context_end], self.values[context_end:prediction_end]


def load_official_ettm1_splits(
    csv_path: Path,
    *,
    context_length: int,
    prediction_length: int,
) -> dict[str, ETTm1WindowDataset]:
    frame = pd.read_csv(csv_path)
    if list(frame.columns) != [
        "date",
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT",
    ]:
        raise ValueError("pinned ETTm1 CSV does not have the canonical eight columns")
    raw = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    train_end = 12 * 30 * 24 * 4
    train = raw[:train_end]
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    if not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError("ETTm1 training split produced an invalid standard deviation")
    standardized = (raw - mean) / scale
    return {
        split: ETTm1WindowDataset(
            standardized,
            split=split,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        for split in ("train", "validation", "test")
    }


def build_patchtst(
    *,
    channels: int,
    context_length: int,
    prediction_length: int,
    layers: int,
    heads: int,
    d_model: int,
    d_ff: int,
    dropout: float,
    fc_dropout: float,
    head_dropout: float,
    patch_length: int,
    stride: int,
) -> nn.Module:
    return PatchTST_backbone(
        c_in=channels,
        context_window=context_length,
        target_window=prediction_length,
        patch_len=patch_length,
        stride=stride,
        n_layers=layers,
        d_model=d_model,
        n_heads=heads,
        d_ff=d_ff,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=dropout,
        act="gelu",
        res_attention=True,
        pre_norm=False,
        pe="zeros",
        learn_pe=True,
        fc_dropout=fc_dropout,
        head_dropout=head_dropout,
        padding_patch="end",
        individual=False,
        revin=True,
        affine=False,
        subtract_last=False,
    )


def run_time_series_forecasting_min(
    workload: Workload, output_dir: Path
) -> dict[str, Any]:
    root = find_project_root()
    seed = configured_seed(default=2021)
    torch.manual_seed(seed)
    model = build_patchtst(
        channels=3,
        context_length=32,
        prediction_length=8,
        layers=1,
        heads=4,
        d_model=32,
        d_ff=64,
        dropout=0.0,
        fc_dropout=0.0,
        head_dropout=0.0,
        patch_length=8,
        stride=4,
    )
    inputs = torch.randn(2, 32, 3)
    targets = torch.randn(2, 8, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs.permute(0, 2, 1)).permute(0, 2, 1)
    loss = nn.functional.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()
    duration = time.perf_counter() - start

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
            "loss": float(loss.item()),
            "duration_seconds": duration,
            "n_params": sum(parameter.numel() for parameter in model.parameters()),
            "input_shape": list(inputs.shape),
            "output_shape": list(outputs.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "The min profile validates the official PatchTST operator stack only. It does not use ETTm1 or support a forecasting-quality claim.",
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
        dataset_name="synthetic-deterministic-time-series",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_time_series_forecasting_max(
    workload: Workload, output_dir: Path
) -> dict[str, Any]:
    root = find_project_root()
    seed = configured_seed(default=2021)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = select_torch_device()
    config = {
        "context_length": 336,
        "prediction_length": int(os.environ.get("MLPERF_EDU_TIMESERIES_HORIZON", 96)),
        "channels": 7,
        "layers": 3,
        "heads": 16,
        "d_model": 128,
        "d_ff": 256,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "head_dropout": 0.0,
        "patch_length": 16,
        "stride": 8,
        "epochs": int(os.environ.get("MLPERF_EDU_TIMESERIES_MAX_EPOCHS", 100)),
        "patience": int(os.environ.get("MLPERF_EDU_TIMESERIES_PATIENCE", 20)),
        "batch_size": 128,
        "learning_rate": 1e-4,
        "one_cycle_pct_start": 0.4,
        "num_workers": int(os.environ.get("MLPERF_EDU_TIMESERIES_WORKERS", 10)),
    }
    if config["prediction_length"] not in {96, 192, 336, 720}:
        raise ValueError("PatchTST ETTm1 horizon must be one of 96, 192, 336, or 720")
    asset = ensure_ettm1(download=True)
    datasets = load_official_ettm1_splits(
        asset.files[0],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=config["num_workers"],
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=config["num_workers"],
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=True,
            num_workers=config["num_workers"],
        ),
    }
    model = build_patchtst(
        channels=config["channels"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        layers=config["layers"],
        heads=config["heads"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        fc_dropout=config["fc_dropout"],
        head_dropout=config["head_dropout"],
        patch_length=config["patch_length"],
        stride=config["stride"],
    ).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_batch_limit = _optional_positive_int(
        "MLPERF_EDU_TIMESERIES_MAX_TRAIN_BATCHES"
    )
    eval_batch_limit = _optional_positive_int("MLPERF_EDU_TIMESERIES_MAX_EVAL_BATCHES")
    steps_per_epoch = min(
        len(loaders["train"]), train_batch_limit or len(loaders["train"])
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        steps_per_epoch=steps_per_epoch,
        pct_start=config["one_cycle_pct_start"],
        epochs=config["epochs"],
        max_lr=config["learning_rate"],
    )

    training_losses: list[float] = []
    validation_mses: list[float] = []
    test_mses: list[float] = []
    epoch_seconds: list[float] = []
    best_validation_mse = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    progress = TrainingProgress(workload.id, int(config["epochs"]), unit="epoch")
    synchronize_device(device)
    measured_start = time.perf_counter()
    for epoch in range(config["epochs"]):
        epoch_start = time.perf_counter()
        model.train()
        batch_losses: list[float] = []
        for batch_index, (past, future) in enumerate(loaders["train"]):
            if train_batch_limit is not None and batch_index >= train_batch_limit:
                break
            past = past.to(device)
            future = future.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(past.permute(0, 2, 1)).permute(0, 2, 1)
            loss = nn.functional.mse_loss(prediction, future)
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_losses.append(float(loss.detach().cpu()))
        validation = _evaluate(model, loaders["validation"], device, eval_batch_limit)
        test = _evaluate(model, loaders["test"], device, eval_batch_limit)
        synchronize_device(device)
        training_losses.append(float(np.mean(batch_losses)))
        validation_mses.append(validation["mse"])
        test_mses.append(test["mse"])
        epoch_seconds.append(time.perf_counter() - epoch_start)
        if validation["mse"] < best_validation_mse:
            best_validation_mse = validation["mse"]
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        progress.update(
            epoch + 1,
            loss=training_losses[-1],
            val_mse=validation["mse"],
            best=best_validation_mse,
            stale=stale_epochs,
        )
        # `stale_epochs` guards the comparison so this stays equivalent to the
        # original in-else break even when patience is overridden to 0: an
        # epoch that improved the best score never triggers early stopping.
        if stale_epochs and stale_epochs >= config["patience"]:
            break
    synchronize_device(device)
    train_and_eval_seconds = time.perf_counter() - measured_start
    progress.close(f"best validation MSE {best_validation_mse:.4f} at epoch {best_epoch}")
    if best_state is None:
        raise RuntimeError("PatchTST training produced no best checkpoint")
    model.load_state_dict(best_state)
    final_test = _evaluate(model, loaders["test"], device, eval_batch_limit)

    target = float(workload.quality_value or 0.290)
    tolerance = float((workload.raw.get("quality_target") or {}).get("tolerance", 0.0))
    target_met = final_test["mse"] <= target + tolerance
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_max_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_max.provd.json").resolve()
    checkpoint_path = (output_dir / f"{workload.id}_max_checkpoint.pt").resolve()
    torch.save(best_state, checkpoint_path)
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "model": "patchtst-supervised",
        "data_mode": "real",
        "dataset": {
            "name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
            "split": "official-ETTm1-12-4-4-month",
            "train_windows": len(datasets["train"]),
            "validation_windows": len(datasets["validation"]),
            "test_windows": len(datasets["test"]),
        },
        "model_source": {
            "repository": "https://github.com/yuqinie98/PatchTST",
            "commit": PATCHTST_COMMIT,
            "ett_dataset_commit": ETT_DATASET_COMMIT,
            "upstream_sha256": {
                "backbone": f"sha256:{PATCHTST_BACKBONE_SHA256}",
                "layers": f"sha256:{PATCHTST_LAYERS_SHA256}",
                "revin": f"sha256:{PATCHTST_REVIN_SHA256}",
                "ettm1_script": f"sha256:{PATCHTST_SCRIPT_SHA256}",
                "experiment": f"sha256:{PATCHTST_EXPERIMENT_SHA256}",
                "data_loader": f"sha256:{PATCHTST_DATA_LOADER_SHA256}",
            },
        },
        "seed": seed,
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "random_seed": seed,
            "context_length": config["context_length"],
            "prediction_length": config["prediction_length"],
            "channels": config["channels"],
            "layers": config["layers"],
            "attention_heads": config["heads"],
            "model_width": config["d_model"],
            "feed_forward_width": config["d_ff"],
            "dropout": config["dropout"],
            "fully_connected_dropout": config["fc_dropout"],
            "head_dropout": config["head_dropout"],
            "patch_length": config["patch_length"],
            "stride": config["stride"],
            "padding_patch": "end",
            "reversible_instance_normalization": True,
            "reversible_instance_normalization_affine": False,
            "residual_attention": True,
            "epochs": config["epochs"],
            "patience": config["patience"],
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "optimizer": "Adam",
            "learning_rate": config["learning_rate"],
            "scheduler": "OneCycleLR",
            "one_cycle_pct_start": config["one_cycle_pct_start"],
        },
        "development_limits": {
            "train_batches": train_batch_limit,
            "evaluation_batches": eval_batch_limit,
        },
        "metrics": {
            "test_mse": final_test["mse"],
            "test_mae": final_test["mae"],
            "best_validation_mse": best_validation_mse,
            "best_epoch": best_epoch,
            "epochs_completed": len(training_losses),
            "duration_seconds": train_and_eval_seconds,
            "train_and_eval_seconds": train_and_eval_seconds,
            "n_params": n_params,
            "training_losses": training_losses,
            "validation_mses": validation_mses,
            "test_mses": test_mses,
            "epoch_seconds": epoch_seconds,
            "test_samples": final_test["samples"],
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "test_mse",
            "target": target,
            "tolerance": tolerance,
            "direction": "lower",
            "target_met": target_met,
            "quality_required": True,
            "override": False,
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


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    batch_limit: int | None,
) -> dict[str, float | int]:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    elements = 0
    samples = 0
    with torch.inference_mode():
        for batch_index, (past, future) in enumerate(loader):
            if batch_limit is not None and batch_index >= batch_limit:
                break
            past = past.to(device)
            future = future.to(device)
            prediction = model(past.permute(0, 2, 1)).permute(0, 2, 1)
            residual = prediction - future
            squared_error += float(residual.square().sum().detach().cpu())
            absolute_error += float(residual.abs().sum().detach().cpu())
            elements += residual.numel()
            samples += residual.shape[0]
    if elements == 0:
        raise RuntimeError("PatchTST evaluation produced no samples")
    return {
        "mse": squared_error / elements,
        "mae": absolute_error / elements,
        "samples": samples,
    }


def _optional_positive_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed
