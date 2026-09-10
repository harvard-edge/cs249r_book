from __future__ import annotations

import json
import math
import os
import time
from unittest.mock import patch
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mlperf.assets import ensure_ogbn_arxiv
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    TrainingProgress,
    configured_seed,
    select_torch_device,
    synchronize_device,
)


OGB_COMMIT = "61e9784ca76edeaa6e259ba0f836099608ff0586"
OGB_GCN_SOURCE_SHA256 = (
    "050b6b7a0fc86ef99b237438f1506f2868a442833d963cfe03efa3419e60d365"
)


class OGBGCN(torch.nn.Module):
    """Direct PyTorch Geometric transcription of the official OGB GCN."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels, hidden_channels, cached=True)]
        )
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_channels)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for index, convolution in enumerate(self.convs[:-1]):
            x = convolution(x, edge_index)
            x = self.bns[index](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index).log_softmax(dim=-1)


def run_graph_node_classification_min(
    workload: Workload, output_dir: Path
) -> dict[str, Any]:
    from torch_geometric.utils import to_undirected

    root = find_project_root()
    seed = configured_seed()
    torch.manual_seed(seed)
    nodes = 32
    features = torch.randn(nodes, 8)
    edge_index = torch.stack(
        [torch.arange(nodes), torch.roll(torch.arange(nodes), shifts=-1)]
    )
    edge_index = to_undirected(edge_index)
    labels = torch.arange(nodes) % 4
    model = OGBGCN(8, 16, 4, 3, 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    start = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(features, edge_index)
    loss = F.nll_loss(output, labels)
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
            "nodes": nodes,
            "edges": int(edge_index.shape[1]),
            "n_params": sum(parameter.numel() for parameter in model.parameters()),
            "logits_shape": list(output.shape),
        },
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": False,
            "note": "The min profile validates the official GCN operator stack only. It does not use ogbn-arxiv or support a quality claim.",
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
        dataset_name="synthetic-deterministic-ring-graph",
        dataset_files=[],
        rng_seed=seed,
        torch_state_bytes=torch.get_rng_state().numpy().tobytes(),
        repo_root=root,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def run_graph_node_classification_max(
    workload: Workload, output_dir: Path
) -> dict[str, Any]:
    from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
    from torch_geometric.utils import to_undirected

    root = find_project_root()
    seed = configured_seed()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = select_torch_device()
    epochs = int(os.environ.get("MLPERF_EDU_GRAPH_MAX_EPOCHS", 500))
    hidden_channels = int(os.environ.get("MLPERF_EDU_GRAPH_MAX_HIDDEN_CHANNELS", 256))
    num_layers = int(os.environ.get("MLPERF_EDU_GRAPH_MAX_LAYERS", 3))
    dropout = float(os.environ.get("MLPERF_EDU_GRAPH_MAX_DROPOUT", 0.5))
    lr = float(os.environ.get("MLPERF_EDU_GRAPH_MAX_LR", 0.01))

    asset = ensure_ogbn_arxiv(download=True)
    # OGB 1.3.6 predates PyTorch's weights_only=True default. This processed
    # object was just derived locally from the pinned, hashed OGB archive.
    original_torch_load = torch.load

    def trusted_ogb_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    with patch("torch.load", trusted_ogb_load):
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(asset.root))
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    split = dataset.get_idx_split()
    data = data.to(device)
    split = {name: indices.to(device) for name, indices in split.items()}
    train_index = split["train"]
    evaluator = Evaluator(name="ogbn-arxiv")
    model = OGBGCN(
        data.num_features,
        hidden_channels,
        dataset.num_classes,
        num_layers,
        dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(parameter.numel() for parameter in model.parameters())

    losses: list[float] = []
    train_accuracies: list[float] = []
    validation_accuracies: list[float] = []
    test_accuracies: list[float] = []
    epoch_times: list[float] = []
    best_validation = -1.0
    test_at_best_validation = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    progress = TrainingProgress(workload.id, epochs, unit="epoch")
    synchronize_device(device)
    start = time.perf_counter()
    for _epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output[train_index], data.y.squeeze(1)[train_index])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(data.x, data.edge_index).argmax(dim=-1, keepdim=True)
        accuracies = {
            name: float(
                evaluator.eval(
                    {
                        "y_true": data.y[indices].detach().cpu(),
                        "y_pred": prediction[indices].detach().cpu(),
                    }
                )["acc"]
            )
            for name, indices in split.items()
        }
        losses.append(float(loss.item()))
        train_accuracies.append(accuracies["train"])
        validation_accuracies.append(accuracies["valid"])
        test_accuracies.append(accuracies["test"])
        if accuracies["valid"] > best_validation:
            best_validation = accuracies["valid"]
            test_at_best_validation = accuracies["test"]
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        epoch_times.append(time.perf_counter() - epoch_start)
        progress.update(
            _epoch + 1,
            loss=losses[-1],
            val_acc=accuracies["valid"],
            best_val=best_validation,
        )
    synchronize_device(device)
    duration = time.perf_counter() - start
    progress.close(f"test accuracy at best validation {test_at_best_validation:.4f}")
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("ogbn-arxiv training duration must be finite and positive")

    target = float(workload.quality_value or 0.7251)
    tolerance = float(workload.quality_tolerance or 0.0)
    target_met = test_at_best_validation + tolerance >= target

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_max_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_max.provd.json").resolve()
    checkpoint_path = (output_dir / f"{workload.id}_max_checkpoint.pt").resolve()
    torch.save(best_state or model.state_dict(), checkpoint_path)
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "max",
        "status": "passed" if target_met else "quality_failed",
        "backend": f"pytorch-{device.type}",
        "model": "official-ogb-gcn",
        "data_mode": "real",
        "dataset": {
            "name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
            "split": "official-time-split",
        },
        "model_source": {
            "repository": "https://github.com/snap-stanford/ogb",
            "commit": OGB_COMMIT,
            "gnn_py_sha256": f"sha256:{OGB_GCN_SOURCE_SHA256}",
        },
        "seed": seed,
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "epochs": epochs,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "symmetric_edges": True,
            "cached_gcn_normalization": True,
        },
        "metrics": {
            "loss": losses[-1],
            "best_validation_accuracy": best_validation,
            "test_accuracy": test_at_best_validation,
            "duration_seconds": duration,
            "train_and_eval_seconds": duration,
            "nodes": int(data.num_nodes),
            "edges": int(data.edge_index.shape[1]),
            "n_params": n_params,
            "epoch_times": epoch_times,
            "losses": losses,
            "train_accuracies": train_accuracies,
            "validation_accuracies": validation_accuracies,
            "test_accuracies": test_accuracies,
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "test_accuracy",
            "target": target,
            "tolerance": tolerance,
            "direction": "higher",
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
