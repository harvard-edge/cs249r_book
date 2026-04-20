#!/usr/bin/env python3
"""
MLPerf EDU: Training Verification Script

Runs each classical workload for a controlled number of epochs on real data,
captures train/val loss curves, and checks for overfitting.

This produces the baseline results that go into the paper.

Usage:
    python scripts/verify_training.py [--model MODEL_NAME] [--epochs N]
"""

import sys
import os
import json
import time
import argparse

# Ensure repo root is on the path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn.functional as F
import torch.optim as optim

from reference.dataset_factory import get_dataloaders


# ---------------------------------------------------------------------------
# Model configs: (module_path, class_name, kwargs, lr, batch_size, epochs)
# ---------------------------------------------------------------------------

CLASSICAL_MODELS = {
    "nanogpt-12m": {
        "module": "reference.cloud.nanogpt_train",
        "class": "NanoGPTWhiteBox",
        "kwargs": {},
        "lr": 3e-4,
        "batch_size": 16,
        "epochs": 20,
        "batches_per_epoch": 100,
    },
    "nano-moe-12m": {
        "module": "reference.cloud.nano_moe",
        "class": "NanoMoEWhiteBox",
        "kwargs": {},
        "lr": 3e-4,
        "batch_size": 16,
        "epochs": 20,
        "batches_per_epoch": 100,
    },
    "resnet18": {
        "module": "reference.edge.resnet_train",
        "class": "ResNet18WhiteBox",
        "kwargs": {"num_classes": 100},
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 15,
        "batches_per_epoch": 100,
    },
    "micro-dlrm-1m": {
        "module": "reference.cloud.micro_dlrm",
        "class": "MicroDLRMWhiteBox",
        "kwargs": {},
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 20,
        "batches_per_epoch": 50,
    },
    "micro-dlrm-dram-1m": {
        "module": "reference.cloud.micro_dlrm_dram",
        "class": "MicroDLRMDRAM",
        "kwargs": {},
        "lr": 1e-3,
        "batch_size": 1024,
        "epochs": 20,
        "batches_per_epoch": 50,
    },
    "micro-diffusion-32px": {
        "module": "reference.cloud.micro_diffusion",
        "class": "MicroDiffusionUNet",
        "kwargs": {},
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 15,
        "batches_per_epoch": 100,
    },
    "dscnn-kws": {
        "module": "reference.tiny.dscnn_kws",
        "class": "DSCNN",
        "kwargs": {"num_classes": 12},
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 15,
        "batches_per_epoch": 100,
    },
    "anomaly-ae": {
        "module": "reference.tiny.anomaly_detection_ae",
        "class": "AnomalyDetectionAE",
        "kwargs": {"input_dim": 784, "bottleneck_dim": 8},
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 20,
        "batches_per_epoch": 50,
    },
}


def load_model(config, device):
    """Dynamically import and instantiate a model."""
    import importlib
    mod = importlib.import_module(config["module"])
    cls = getattr(mod, config["class"])
    model = cls(**config["kwargs"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def train_one_epoch(model, model_name, train_loader, optimizer, device, max_batches):
    """Train for one epoch, return average loss."""
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_batches:
            break

        optimizer.zero_grad()

        if "dlrm" in model_name:
            dense, sparse_idx, sparse_off, labels = batch
            dense = dense.to(device)
            sparse_idx = [s.to(device) for s in sparse_idx]
            sparse_off = [s.to(device) for s in sparse_off]
            labels = labels.to(device)
            outputs = model(dense, sparse_idx, sparse_off)
            loss = F.binary_cross_entropy(outputs, labels)
        elif "resnet" in model_name:
            data_batch, target_batch = batch
            outputs = model(data_batch.to(device))
            loss = F.cross_entropy(outputs, target_batch.to(device))
        elif "diffusion" in model_name:
            data_batch, _ = batch
            data_batch = data_batch.to(device)
            outputs = model(data_batch)
            loss = F.mse_loss(outputs, data_batch)
        elif "dscnn" in model_name or "kws" in model_name:
            data_batch, target_batch = batch
            _, loss = model(data_batch.to(device), targets=target_batch.to(device))
        elif "anomaly" in model_name:
            data_batch, _ = batch  # labels not used for AE training
            _, loss = model(data_batch.to(device))
        else:
            # Language models
            data_batch, target_batch = batch
            _, loss = model(data_batch.to(device), targets=target_batch.to(device))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return sum(losses) / len(losses) if losses else float("inf")


@torch.no_grad()
def validate(model, model_name, val_loader, device, max_batches=20):
    """Evaluate on validation set, return average loss and accuracy (if applicable)."""
    model.eval()
    losses = []
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        if "dlrm" in model_name:
            dense, sparse_idx, sparse_off, labels = batch
            dense = dense.to(device)
            sparse_idx = [s.to(device) for s in sparse_idx]
            sparse_off = [s.to(device) for s in sparse_off]
            labels = labels.to(device)
            outputs = model(dense, sparse_idx, sparse_off)
            loss = F.binary_cross_entropy(outputs, labels)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
        elif "resnet" in model_name:
            data_batch, target_batch = batch
            data_batch = data_batch.to(device)
            target_batch = target_batch.to(device)
            outputs = model(data_batch)
            loss = F.cross_entropy(outputs, target_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == target_batch).sum().item()
            total += target_batch.numel()
        elif "diffusion" in model_name:
            data_batch, _ = batch
            data_batch = data_batch.to(device)
            outputs = model(data_batch)
            loss = F.mse_loss(outputs, data_batch)
        elif "dscnn" in model_name or "kws" in model_name:
            data_batch, target_batch = batch
            data_batch = data_batch.to(device)
            target_batch = target_batch.to(device)
            logits, loss = model(data_batch, targets=target_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == target_batch).sum().item()
            total += target_batch.numel()
        elif "anomaly" in model_name:
            data_batch, _ = batch
            _, loss = model(data_batch.to(device))
        else:
            data_batch, target_batch = batch
            _, loss = model(data_batch.to(device), targets=target_batch.to(device))

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    accuracy = correct / total if total > 0 else None
    return avg_loss, accuracy


def train_model(model_name, config, device):
    """Full training loop for one model."""
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    # Load model
    model, n_params = load_model(config, device)
    print(f"  Parameters: {n_params/1e6:.2f}M")
    print(f"  Device: {device}")

    # Load data
    train_loader, val_loader = get_dataloaders(model_name, batch_size=config["batch_size"])
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")

    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2)

    # LR scheduler for smooth convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.1
    )

    # Training loop
    results = {
        "model_name": model_name,
        "n_params": n_params,
        "device": str(device),
        "train_losses": [],
        "val_losses": [],
        "val_accuracies": [],
        "epoch_times": [],
    }

    start_time = time.time()

    for epoch in range(config["epochs"]):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(
            model, model_name, train_loader, optimizer, device,
            max_batches=config["batches_per_epoch"]
        )
        val_loss, val_acc = validate(model, model_name, val_loader, device)
        scheduler.step()

        epoch_time = time.perf_counter() - t0

        results["train_losses"].append(train_loss)
        results["val_losses"].append(val_loss)
        results["val_accuracies"].append(val_acc)
        results["epoch_times"].append(epoch_time)

        # Print progress
        acc_str = f" | Acc: {val_acc:.3f}" if val_acc is not None else ""
        gap = val_loss - train_loss
        overfit = " ⚠️ OVERFIT" if gap > 0.5 * train_loss else ""
        print(f"  Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{acc_str} | "
              f"Gap: {gap:.4f}{overfit} | {epoch_time:.1f}s")

    total_time = time.time() - start_time
    results["total_time_s"] = total_time
    results["final_train_loss"] = results["train_losses"][-1]
    results["final_val_loss"] = results["val_losses"][-1]

    # Overfitting check
    gap = results["final_val_loss"] - results["final_train_loss"]
    if gap > 0.5 * results["final_train_loss"]:
        print(f"  ⚠️  OVERFITTING: val-train gap = {gap:.4f}")
    else:
        print(f"  ✅ Healthy: val-train gap = {gap:.4f}")

    print(f"  Total time: {total_time:.1f}s")

    # Save checkpoint
    ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "instructor_baseline.pt"))

    # Save results
    results_path = os.path.join(ckpt_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="MLPerf EDU Training Verification")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to train (or 'all')")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"🚀 MLPerf EDU Training Verification")
    print(f"   Device: {device}")

    models_to_train = {}
    if args.model and args.model != "all":
        if args.model not in CLASSICAL_MODELS:
            print(f"❌ Unknown model: {args.model}")
            print(f"   Available: {list(CLASSICAL_MODELS.keys())}")
            sys.exit(1)
        models_to_train[args.model] = CLASSICAL_MODELS[args.model]
    else:
        models_to_train = CLASSICAL_MODELS

    if args.epochs:
        for cfg in models_to_train.values():
            cfg["epochs"] = args.epochs

    all_results = {}
    for model_name, config in models_to_train.items():
        results = train_model(model_name, config, device)
        all_results[model_name] = results

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<25s} {'Params':>8s} {'Train':>8s} {'Val':>8s} {'Gap':>8s} {'Time':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, r in all_results.items():
        gap = r["final_val_loss"] - r["final_train_loss"]
        print(f"  {name:<25s} {r['n_params']/1e6:>7.1f}M "
              f"{r['final_train_loss']:>8.4f} {r['final_val_loss']:>8.4f} "
              f"{gap:>8.4f} {r['total_time_s']:>7.1f}s")


if __name__ == "__main__":
    main()
