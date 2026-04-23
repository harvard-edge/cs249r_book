#!/usr/bin/env python3
"""
MLPerf EDU: Lab 1 — Systems Optimization Challenge
====================================================

Students receive a "broken baseline" ResNet-18 configuration and must
apply systems optimizations to maximize accuracy within a fixed wall-clock budget.

Starter: 5% accuracy in 30s (bad hyperparams, no parallelism)
Target:  >50% accuracy in 30s (after optimizations)

INSTRUCTIONS:
    1. Run the baseline: python examples/lab1_optimization.py
    2. Apply optimizations one at a time (see TODOs below)
    3. Measure the impact of each change
    4. Submit your results: python scripts/compliance_checker.py --workload resnet18
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from reference.dataset_factory import get_dataloaders

# ── Device setup ──────────────────────────────────────────────────────────────
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"🖥️  Device: {device}")

# ── Load the model ────────────────────────────────────────────────────────────
from reference.edge.resnet_core import ResNet18Local
model = ResNet18Local(num_classes=100).to(device)
print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── BROKEN BASELINE (intentionally poor configuration) ────────────────────────
# TODO: Fix each of these settings to improve performance

batch_size = 8        # TODO 1: Increase to 64 or 128 for better GPU utilization
num_workers = 0       # TODO 2: Set to 4 for parallel data loading
learning_rate = 0.1   # TODO 3: Use a schedule (CosineAnnealingLR)
use_augmentation = False  # TODO 4: Add RandomCrop + HorizontalFlip

# Load data with broken config
train_ld, val_ld = get_dataloaders("resnet18", bs=batch_size)

# ── Training loop ─────────────────────────────────────────────────────────────
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # TODO 3: Add momentum=0.9
# TODO 3: scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

BUDGET_SECONDS = 30.0
epoch = 0
t_start = time.time()

print(f"\n🏋️ Training with {BUDGET_SECONDS}s wall-clock budget...")
print(f"{'Epoch':>5} {'Loss':>8} {'Acc':>7} {'Time':>8}")
print("-" * 35)

while time.time() - t_start < BUDGET_SECONDS:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in train_ld:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        if time.time() - t_start >= BUDGET_SECONDS:
            break
    # TODO 3: scheduler.step()
    epoch += 1
    elapsed = time.time() - t_start
    acc = correct / max(total, 1)
    print(f"{epoch:5d} {total_loss/len(train_ld):8.4f} {acc:7.1%} {elapsed:7.1f}s")

# ── Validation ────────────────────────────────────────────────────────────────
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in val_ld:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

val_acc = correct / total
elapsed = time.time() - t_start

print(f"\n{'='*35}")
print(f"⏱️  Total time: {elapsed:.1f}s")
print(f"📈 Final val accuracy: {val_acc:.1%}")
print(f"🏁 Epochs completed: {epoch}")

if val_acc > 0.50:
    print("🏆 TARGET HIT! Accuracy > 50%")
elif val_acc > 0.36:
    print("✅ Minimum MLPerf quality target met (>36%)")
else:
    print("❌ Below quality target. Apply optimizations!")

# ── Diagnostic: What to optimize ──────────────────────────────────────────────
print(f"""
📋 Optimization Checklist:
   [ ] Batch size: {batch_size} → 64 (4× throughput)
   [ ] DataLoader workers: {num_workers} → 4 (parallel CPU preprocessing)
   [ ] Learning rate schedule: constant → CosineAnnealing
   [ ] Data augmentation: {'off' if not use_augmentation else 'on'} → RandomCrop + HFlip
   [ ] Optimizer: SGD(lr=0.1) → SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)
""")
