#!/usr/bin/env python3
"""
Iter-7 smoke test: LoRA fine-tuning workload.

Per Han's iter-7 spec. Three falsifiable assertions:
  1. trainable_param_ratio in (0.001, 0.01)  — the parameterization is correct
  2. base_grad_norm == 0.0                    — base is frozen
  3. lora_grad_norm > 0.0 AND loss descends   — adapter is learning

Gate: trainable_param_ratio in (0.001, 0.01).
The other two are sanity assertions, not the gate.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from reference.cloud.nanogpt_train import NanoGPTWhiteBox, GPT2_SMALL_CONFIG
from reference.cloud.lora import inject_lora, base_grad_norm, lora_grad_norm
from mlperf.roofline import measure_roofline


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    device = _device()
    print(f"Device: {device}")

    # Build GPT-2-Small backbone (random weights — we measure systems cost,
    # not task accuracy, per Han's iter-7 honesty note).
    model = NanoGPTWhiteBox(**GPT2_SMALL_CONFIG).to(device)
    base_params = sum(p.numel() for p in model.parameters())
    print(f"Base model: {base_params/1e6:.1f}M params")

    # Inject LoRA adapters (rank=8, alpha=16, target c_attn).
    n_injected, n_trainable = inject_lora(model, rank=8, alpha=16, target="c_attn")
    print(f"Injected {n_injected} LoRA adapters; {n_trainable:,} trainable params "
          f"({n_trainable/base_params*100:.3f}% of base)")

    ratio = n_trainable / base_params
    print()

    # Quick training loop: 50 steps, batch=8, ctx=256 (smaller than spec's 100/8/512
    # to keep smoke ≤2 min on M5 Max).
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=3e-4)
    vocab = model.config["vocab_size"]
    n_steps = 30
    losses: list[float] = []

    print(f"Running {n_steps} LoRA training steps (batch=8, ctx=256)...", flush=True)
    t0 = time.perf_counter()
    with measure_roofline(
        "nano-lora-finetune",
        analytic_flops=lambda: 2 * base_params * 8 * 256 * n_steps,  # forward+backward of base
        analytic_bytes=lambda: (base_params * 4 + 8 * 256 * 768 * 4 * 12) * n_steps,
        n_iter=n_steps,
    ):
        for step in range(n_steps):
            ids = torch.randint(0, vocab, (8, 256), device=device)
            targets = torch.randint(0, vocab, (8, 256), device=device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(ids, targets=targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    elapsed = time.perf_counter() - t0

    # Capture grad norms from the LAST step (already computed; need a fresh
    # backward to populate gradients we can inspect).
    optimizer.zero_grad(set_to_none=True)
    ids = torch.randint(0, vocab, (8, 256), device=device)
    targets = torch.randint(0, vocab, (8, 256), device=device)
    _, loss = model(ids, targets=targets)
    loss.backward()
    bgn = base_grad_norm(model)
    lgn = lora_grad_norm(model)

    print(f"  wall-clock:        {elapsed:.2f} s ({elapsed/n_steps*1000:.0f} ms/step)")
    print(f"  loss step 0:       {losses[0]:.4f}")
    print(f"  loss step {n_steps-1}: {losses[-1]:.4f}")
    print(f"  base_grad_norm:    {bgn:.6f}  (must be 0.0)")
    print(f"  lora_grad_norm:    {lgn:.6f}  (must be > 0)")
    print(f"  trainable ratio:   {ratio*100:.3f}%  (gate: 0.1% < r < 1%)")
    print()

    # Gate + sanity assertions.
    failures = []
    if not (0.001 < ratio < 0.01):
        failures.append(f"GATE: trainable_param_ratio {ratio:.5f} not in (0.001, 0.01)")
    if bgn != 0.0:
        failures.append(f"SANITY: base_grad_norm = {bgn} ≠ 0.0 (base is not actually frozen)")
    if lgn <= 0.0:
        failures.append(f"SANITY: lora_grad_norm = {lgn} ≤ 0 (adapter is not receiving gradient)")
    if losses[-1] >= losses[0]:
        failures.append(f"SANITY: loss did not descend ({losses[0]:.3f} → {losses[-1]:.3f}) "
                        "— may indicate broken init or untrained-data noise")

    if failures:
        print("ITER-7 SMOKE: FAIL")
        for f in failures:
            print(f"  {f}")
        return 1
    print("ITER-7 SMOKE: PASS")
    print(f'  Headline: LoRA trains {n_trainable:,} parameters ({ratio*100:.2f}% of {base_params/1e6:.0f}M)')
    print(f'            and the base model is genuinely frozen (grad norm exactly 0.0).')
    return 0


if __name__ == "__main__":
    sys.exit(main())
