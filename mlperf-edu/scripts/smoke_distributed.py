#!/usr/bin/env python3
"""
Iter-10 smoke test: two-process DDP via Gloo on localhost.

Single dimensionless gate (Dean's iter-10 spec Q4):
  | loss_ddp(step=50) - loss_gradacc(step=50) | / loss_gradacc(step=50) < 0.02

This catches the three most common DDP bugs in one number:
  - DistributedSampler seed wrong
  - Gradient averaging wrong (mean vs sum)
  - LR scaling wrong

Run: python3 scripts/smoke_distributed.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from reference.distributed.ddp_runner import run_ddp, run_gradacc_baseline


def main() -> int:
    print("Iter-10: DDP-vs-grad-accumulation loss equivalence")
    print("-" * 60)
    N_STEPS = 50
    MICRO_BATCH = 64
    WORLD_SIZE = 2

    print(f"world_size={WORLD_SIZE}, n_steps={N_STEPS}, micro_batch={MICRO_BATCH}")
    print(f"global batch (DDP) = grad-acc batch = {WORLD_SIZE * MICRO_BATCH}")
    print()

    print("Running grad-accumulation baseline (single proc)...", flush=True)
    base = run_gradacc_baseline(N_STEPS, MICRO_BATCH, WORLD_SIZE)
    print(f"  baseline final_loss: {base['final_loss']:.6f}")

    print(f"Spawning {WORLD_SIZE} DDP workers (Gloo on 127.0.0.1)...", flush=True)
    ddp = run_ddp(N_STEPS, MICRO_BATCH, WORLD_SIZE)
    if "error" in ddp:
        print(f"  DDP failed: {ddp['error']}")
        return 1
    print(f"  DDP rank-0 final_loss: {ddp['final_loss']:.6f}")
    print(f"  DDP avg AllReduce time per step: {ddp['allreduce_time_per_step_ms']:.3f} ms")
    print()

    if base["final_loss"] == 0:
        print("ITER-10 SMOKE: SKIP (baseline loss zero — degenerate)")
        return 0
    delta = abs(ddp["final_loss"] - base["final_loss"]) / base["final_loss"]
    print(f"|loss_ddp - loss_baseline| / baseline = {delta:.4f}  (gate < 0.02)")

    if delta < 0.02:
        print("ITER-10 SMOKE: PASS")
        print(f'  Headline: "DDP and gradient accumulation are mathematically equivalent. '
              f'Two ranks of micro-batch=64 produce the same loss as one process accumulating '
              f'2x64=128. The {ddp["allreduce_time_per_step_ms"]:.1f} ms/step AllReduce overhead '
              f'is what you pay for the parallelism."')
        return 0
    print("ITER-10 SMOKE: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
