"""MLPerf Training v0.5 recommendation: Neural Collaborative Filtering.

Thin PyTorch adapter for the retired MLPerf v0.5 recommendation benchmark. The
model, dataset, leave-one-out split, evaluator, and 0.635 HR@10 target are
inherited unchanged; this module adds execution, measurement boundaries,
per-epoch curves, and provenance.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from mlperf.assets import ensure_movielens_20m, movielens_20m_paths
from mlperf.fingerprint import detect_hardware
from mlperf.manifest import build_provd
from mlperf.registry import Workload, find_project_root
from mlperf.runners.common import (
    TrainingProgress,
    configured_seed,
    select_torch_device,
    synchronize_device,
)

MLPERF_V05_REFERENCE = (
    "https://github.com/mlcommons/training/tree/master/retired_benchmarks/recommendation"
)


class NeuMF(nn.Module):
    """NCF as specified by MLPerf v0.5: parallel GMF and MLP towers."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        factors: int,
        layer_sizes: list[int],
    ):
        super().__init__()
        mlp_dim = layer_sizes[0] // 2
        self.gmf_user = nn.Embedding(n_users, factors)
        self.gmf_item = nn.Embedding(n_items, factors)
        self.mlp_user = nn.Embedding(n_users, mlp_dim)
        self.mlp_item = nn.Embedding(n_items, mlp_dim)

        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(factors + layer_sizes[-1], 1)

        for embedding in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(embedding.weight, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user(users) * self.gmf_item(items)
        mlp = self.mlp(
            torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1)
        )
        return self.head(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)


def _load_leave_one_out(
    ratings_csv: Path, negatives_per_user: int, seed: int
) -> dict[str, Any]:
    """Build the v0.5 leave-one-out split with sampled evaluation negatives.

    Implicit feedback: every observed rating is a positive. The chronologically
    last interaction per user is held out for evaluation, scored against
    `negatives_per_user` items the user never interacted with.
    """
    import csv

    users: list[int] = []
    items: list[int] = []
    stamps: list[int] = []
    with ratings_csv.open() as handle:
        reader = csv.reader(handle)
        next(reader)  # header
        for row in reader:
            users.append(int(row[0]))
            items.append(int(row[1]))
            stamps.append(int(row[3]))

    user_array = np.asarray(users, dtype=np.int64)
    item_array = np.asarray(items, dtype=np.int64)
    stamp_array = np.asarray(stamps, dtype=np.int64)

    # 20M boxed Python ints per list is gigabytes the arrays now hold compactly.
    # Releasing them here keeps the peak inside a laptop's memory budget.
    del users, items, stamps

    # Reindex to dense ids so embedding tables stay tight.
    unique_users, user_idx = np.unique(user_array, return_inverse=True)
    unique_items, item_idx = np.unique(item_array, return_inverse=True)
    n_users = int(unique_users.size)
    n_items = int(unique_items.size)

    # Held-out positive = latest interaction per user.
    order = np.lexsort((stamp_array, user_idx))
    sorted_users = user_idx[order]
    sorted_items = item_idx[order]
    last_positions = np.searchsorted(sorted_users, np.arange(n_users), side="right") - 1
    test_items = sorted_items[last_positions]

    is_test = np.zeros(sorted_users.size, dtype=bool)
    is_test[last_positions] = True
    train_users = sorted_users[~is_test]
    train_items = sorted_items[~is_test]

    # The rows are already grouped by user, so each user's interactions are one
    # contiguous slice. Reading membership from those slices avoids building
    # 138k Python sets over 20M boxed ints, which dominated both the wall clock
    # and the peak memory of split construction.
    user_starts = np.searchsorted(sorted_users, np.arange(n_users), side="left")
    user_ends = np.searchsorted(sorted_users, np.arange(n_users), side="right")

    rng = np.random.default_rng(seed)
    negatives = np.empty((n_users, negatives_per_user), dtype=np.int64)
    for user in range(n_users):
        blocked = np.unique(sorted_items[user_starts[user] : user_ends[user]])
        # Rejection sampling cannot terminate for a user who has interacted
        # with all but a few items, and the loop below would spin forever with
        # no diagnostic. The candidate count is part of the metric, so the
        # honest response is to fail rather than quietly return fewer.
        available = n_items - blocked.size
        if available < negatives_per_user:
            raise ValueError(
                f"user {user} has only {available} unseen items but the "
                f"contract requires {negatives_per_user} evaluation negatives. "
                "The candidate count defines the metric and cannot be reduced "
                "to fit the data."
            )
        # Draws and acceptance order are unchanged from the per-element form:
        # the same batches are drawn in the same sequence and survivors keep
        # their positions, so the sampled negatives are identical.
        drawn: list[int] = []
        while len(drawn) < negatives_per_user:
            batch = rng.integers(0, n_items, size=negatives_per_user * 2)
            keep = batch[~np.isin(batch, blocked, assume_unique=False)]
            drawn.extend(keep.tolist())
        negatives[user] = drawn[:negatives_per_user]

    return {
        "n_users": n_users,
        "n_items": n_items,
        "train_users": train_users,
        "train_items": train_items,
        "test_items": test_items,
        "eval_negatives": negatives,
        # `seen` is deliberately not returned. Nothing downstream reads it, and
        # holding several gigabytes of per-user Python sets alive for the whole
        # training run is the difference between a laptop that trains and a
        # laptop that swaps.
        "n_interactions": int(sorted_users.size),
    }


def _hit_rate_at_10(
    model: NeuMF, split: dict[str, Any], device: torch.device, batch_users: int = 2048
) -> float:
    """HR@10 over the held-out positive against its sampled negatives."""
    model.eval()
    n_users = split["n_users"]
    negatives = torch.from_numpy(split["eval_negatives"])
    positives = torch.from_numpy(split["test_items"])
    hits = 0
    with torch.no_grad():
        for start in range(0, n_users, batch_users):
            stop = min(start + batch_users, n_users)
            chunk = stop - start
            candidates = torch.cat(
                [positives[start:stop].unsqueeze(1), negatives[start:stop]], dim=1
            ).to(device)
            n_candidates = candidates.shape[1]
            users = (
                torch.arange(start, stop, device=device)
                .unsqueeze(1)
                .expand(chunk, n_candidates)
            )
            scores = model(users.reshape(-1), candidates.reshape(-1)).view(
                chunk, n_candidates
            )
            top = scores.topk(10, dim=1).indices
            hits += int((top == 0).any(dim=1).sum().item())
    return hits / float(n_users)


LR_SCHEDULES = ("constant", "cosine", "step")


def _optimizer_lr_override(contract: dict[str, Any]) -> float:
    """Base learning rate, from the contract unless a research override is set.

    Fails closed on a malformed or non-positive value rather than silently
    falling back to the contract, because a typo that quietly reproduces the
    baseline would make an ablation look like a null result.
    """
    contract_lr = float(contract.get("learning_rate", 0.0005))
    raw = os.environ.get("MLPERF_EDU_NCF_LEARNING_RATE")
    if raw is None or raw == "":
        return contract_lr
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"MLPERF_EDU_NCF_LEARNING_RATE must be a number, got {raw!r}"
        ) from exc
    if not value > 0.0:
        raise ValueError(
            f"MLPERF_EDU_NCF_LEARNING_RATE must be positive, got {value!r}"
        )
    return value


def _lr_schedule_override() -> str:
    """Learning-rate schedule for the run. Defaults to the contract's constant rate."""
    raw = os.environ.get("MLPERF_EDU_NCF_LR_SCHEDULE")
    if raw is None or raw == "":
        return "constant"
    value = raw.strip().lower()
    if value not in LR_SCHEDULES:
        raise ValueError(
            f"MLPERF_EDU_NCF_LR_SCHEDULE must be one of {', '.join(LR_SCHEDULES)}, "
            f"got {raw!r}"
        )
    return value


def _build_lr_scheduler(
    optimizer: "torch.optim.Optimizer", schedule: str, epochs: int
) -> Any:
    """Return a per-epoch scheduler, or None for the contract's constant rate."""
    if schedule == "constant":
        return None
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if schedule == "step":
        # Halve at each third of the budget: a coarse anneal that does not
        # depend on the epoch count being large.
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=0.5
        )
    raise ValueError(f"unhandled schedule {schedule!r}")


def run_recommendation_max(workload: Workload, output_dir: Path) -> dict[str, Any]:
    root = find_project_root()
    seed = configured_seed()
    torch.manual_seed(seed)
    device = select_torch_device()

    contract = (workload.raw.get("canonical_max_contract") or {}).get("config") or {}
    factors = int(contract.get("predictive_factors", 64))
    layer_sizes = [int(v) for v in contract.get("mlp_layer_sizes", [256, 256, 128, 64])]
    train_negatives = int(contract.get("negatives_per_positive_train", 4))
    eval_negatives = int(contract.get("negatives_per_user_eval", 100))
    batch_size = int(contract.get("batch_size", 2048))
    # The epoch budget is part of the contract, not a runner default, so the
    # registry stays the single source of truth for what "max" costs.
    epochs = int(
        os.environ.get("MLPERF_EDU_NCF_MAX_EPOCHS", contract.get("max_epochs", 7))
    )
    # Research overrides for the `pro` envelope. Both default to the contract,
    # so an unset environment reproduces the canonical max path exactly. They
    # exist because the contract fixes Adam at a constant rate while the
    # upstream reference may anneal, and that hypothesis is untestable without
    # a way to vary the schedule from outside the registry.
    lr = _optimizer_lr_override(contract)
    lr_schedule = _lr_schedule_override()

    asset = ensure_movielens_20m(download=True)
    ratings_csv = movielens_20m_paths()["ratings"]

    # Split construction is excluded from the measured region by contract.
    split = _load_leave_one_out(ratings_csv, eval_negatives, seed)
    model = NeuMF(split["n_users"], split["n_items"], factors, layer_sizes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = _build_lr_scheduler(optimizer, lr_schedule, epochs)
    lr_by_epoch: list[float] = []
    loss_fn = nn.BCEWithLogitsLoss()

    train_users = torch.from_numpy(split["train_users"])
    train_items = torch.from_numpy(split["train_items"])
    n_positive = train_users.numel()
    rng = np.random.default_rng(seed)

    losses: list[float] = []
    hit_rates: list[float] = []
    epoch_times: list[float] = []
    best_hr = 0.0
    best_state = None
    target = float(workload.quality_value or 0.635)
    tolerance = float(workload.quality_tolerance or 0.0)

    progress = TrainingProgress(workload.id, epochs, unit="epoch")
    synchronize_device(device)
    started = time.perf_counter()
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        lr_by_epoch.append(float(optimizer.param_groups[0]["lr"]))
        model.train()
        # Fresh negatives each epoch, as in the v0.5 reference.
        neg_items = torch.from_numpy(
            rng.integers(0, split["n_items"], size=n_positive * train_negatives)
        )
        users = torch.cat([train_users, train_users.repeat(train_negatives)])
        items = torch.cat([train_items, neg_items])
        labels = torch.cat(
            [torch.ones(n_positive), torch.zeros(n_positive * train_negatives)]
        )
        order = torch.randperm(users.numel())
        users, items, labels = users[order], items[order], labels[order]

        running = 0.0
        batches = 0
        for start in range(0, users.numel(), batch_size):
            stop = start + batch_size
            batch_users = users[start:stop].to(device)
            batch_items = items[start:stop].to(device)
            batch_labels = labels[start:stop].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_users, batch_items), batch_labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            batches += 1

        if scheduler is not None:
            scheduler.step()
        hit_rate = _hit_rate_at_10(model, split, device)
        losses.append(running / max(1, batches))
        hit_rates.append(hit_rate)
        epoch_times.append(time.perf_counter() - epoch_start)
        if hit_rate > best_hr:
            best_hr = hit_rate
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        progress.update(
            epoch + 1, loss=losses[-1], hr10=hit_rate, best=best_hr, target=target
        )
        if best_hr >= target - tolerance:
            break
    synchronize_device(device)
    duration = time.perf_counter() - started
    progress.close(f"best HR@10 {best_hr:.4f} against target {target:.4f}")

    target_met = best_hr + tolerance >= target
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
        "model": "mlperf-v0.5-ncf",
        "data_mode": "real",
        "dataset": {
            "name": asset.name,
            "source": asset.source,
            "root": str(asset.root),
            "sha256": asset.sha256,
            "n_bytes": asset.n_bytes,
            "split": "leave-one-out-100-negatives",
        },
        "model_source": {"reference_implementation": MLPERF_V05_REFERENCE},
        "seed": seed,
        "measurement_protocol": workload.raw.get("measurement_protocol", {}),
        "config": {
            "predictive_factors": factors,
            "mlp_layer_sizes": layer_sizes,
            "negatives_per_positive_train": train_negatives,
            "negatives_per_user_eval": eval_negatives,
            "learning_rate": lr,
            "learning_rate_schedule": lr_schedule,
            "learning_rate_by_epoch": lr_by_epoch,
            "learning_rate_overridden": lr
            != float(contract.get("learning_rate", 0.0005)),
            "batch_size": batch_size,
            "epochs_requested": epochs,
        },
        "metrics": {
            "hit_rate_at_10": best_hr,
            "loss": losses[-1] if losses else None,
            "duration_seconds": duration,
            "train_and_eval_seconds": duration,
            "epochs_completed": len(hit_rates),
            "n_params": n_params,
            "n_users": split["n_users"],
            "n_items": split["n_items"],
            "n_interactions": split["n_interactions"],
            "epoch_times": epoch_times,
            "losses": losses,
            "hit_rates": hit_rates,
        },
        "quality": {
            "metric": workload.quality_metric,
            "metric_key": "hit_rate_at_10",
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
