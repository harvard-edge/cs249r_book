#!/usr/bin/env python3
"""
MLPerf EDU: Lab 3 — Architecture Comparison
=============================================

Students train NanoGPT and Nano-MoE side-by-side to understand
the systems implications of dense vs. sparse architectures.

Key Questions:
    - Why does MoE converge faster with fewer active parameters?
    - How does expert routing affect memory access patterns?
    - What is the throughput/quality tradeoff?

Run: python examples/lab3_arch_comparison.py
"""

import time
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from reference.dataset_factory import get_dataloaders

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


def train_one_epoch(model, train_ld, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss, n_batches = 0.0, 0
    for x, y in train_ld:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        # Models return (logits, loss) or just logits
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=" * 60)
    print("  MLPerf EDU: Lab 3 — Dense vs. Sparse Architecture")
    print("=" * 60)

    # ── Load models ───────────────────────────────────────────
    from reference.cloud.nanogpt_train import NanoGPTWhiteBox
    from reference.cloud.nano_moe import NanoMoEWhiteBox

    nanogpt = NanoGPTWhiteBox(
        vocab_size=65, n_embd=128, n_head=4, n_layer=4
    ).to(device)

    nanomoe = NanoMoEWhiteBox(
        vocab_size=65, d_model=128, n_heads=4, n_layers=4,
    ).to(device)

    gpt_params, _ = count_params(nanogpt)
    moe_params, _ = count_params(nanomoe)

    print(f"\n📊 Model Parameters:")
    print(f"   NanoGPT:  {gpt_params:>10,}")
    print(f"   Nano-MoE: {moe_params:>10,}")
    print(f"   Ratio:    MoE is {moe_params/gpt_params:.1f}× GPT")

    # ── Load data ─────────────────────────────────────────────
    train_ld_gpt, _ = get_dataloaders("nanogpt-12m", bs=32)
    train_ld_moe, _ = get_dataloaders("nano-moe", bs=32)

    # ── Train both for 5 epochs ───────────────────────────────
    opt_gpt = torch.optim.AdamW(nanogpt.parameters(), lr=3e-4)
    opt_moe = torch.optim.AdamW(nanomoe.parameters(), lr=3e-4)

    N_EPOCHS = 5
    print(f"\n🏋️ Training both models for {N_EPOCHS} epochs...")
    print(f"{'Epoch':>5} {'GPT Loss':>10} {'GPT Time':>10} {'MoE Loss':>10} {'MoE Time':>10}")
    print("-" * 50)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        gpt_loss = train_one_epoch(nanogpt, train_ld_gpt, opt_gpt, device)
        gpt_time = time.time() - t0

        t0 = time.time()
        moe_loss = train_one_epoch(nanomoe, train_ld_moe, opt_moe, device)
        moe_time = time.time() - t0

        print(f"{epoch:5d} {gpt_loss:10.4f} {gpt_time:9.1f}s {moe_loss:10.4f} {moe_time:9.1f}s")

    # ── Analysis ──────────────────────────────────────────────
    print(f"""
{'='*50}
📋 Discussion Questions:

1. MoE has {moe_params-gpt_params:,} MORE total parameters than GPT,
   but only activates 2/8 experts per token (25%).
   How many "active" parameters does MoE use per forward pass?

2. If MoE converges to a lower loss, why not always use MoE?
   What are the memory and communication costs?

3. Measure GPU memory usage for each model:
   torch.cuda.max_memory_allocated() or activity monitor

4. Advanced: Modify top_k from 2 to 1 or 4.
   How does expert utilization affect convergence?
""")


if __name__ == "__main__":
    main()
