"""
Iter-8 (Han): MobileNetV2 compression composition primitives.

Three primitives that COMPOSE on plain PyTorch+MPS:
  - prune_unstructured(model, sparsity): mask-based magnitude pruning.
  - prune_2of4(model): mask-based 2:4 structured sparsity.
  - fake_quantize_int8(model): per-tensor symmetric quant -> dequant
    (no speedup; effective_bits=8 for byte accounting).

Per Han's iter-8 negative-result discipline: none of these accelerate
inference on PyTorch+MPS. They are *algorithmic* primitives that
compose; the runtime gap to a fused-kernel stack IS the lesson.
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def prune_unstructured(model: nn.Module, sparsity: float = 0.5) -> dict:
    """Magnitude-prune `sparsity` fraction of weights in every Conv/Linear.

    Stores the mask as a buffer so the pruning is reproducible and the
    effective_bits accounting can read it back.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    n_pruned = 0
    n_total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            k = int(sparsity * w.numel())
            if k == 0:
                continue
            threshold = w.abs().flatten().kthvalue(k).values
            mask = (w.abs() > threshold).to(w.dtype)
            m.weight.data.mul_(mask)
            m.register_buffer("_prune_mask", mask)
            n_pruned += int((mask == 0).sum().item())
            n_total += w.numel()
    return {"sparsity_actual": n_pruned / max(n_total, 1), "n_pruned": n_pruned, "n_total": n_total}


@torch.no_grad()
def prune_2of4(model: nn.Module) -> dict:
    """Structured 2:4 sparsity: zero 2 of every 4 contiguous weights."""
    n_pruned = 0
    n_total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            flat = w.flatten()
            n = (flat.numel() // 4) * 4  # round down
            if n == 0:
                continue
            blocks = flat[:n].view(-1, 4)
            # Keep top-2 by magnitude in each block of 4.
            _, top_idx = blocks.abs().topk(2, dim=1)
            mask = torch.zeros_like(blocks)
            mask.scatter_(1, top_idx, 1.0)
            new_flat = flat.clone()
            new_flat[:n] = (blocks * mask).flatten()
            w.data.copy_(new_flat.view_as(w))
            full_mask = torch.ones_like(w)
            full_mask.flatten()[:n] = mask.flatten()
            m.register_buffer("_2of4_mask", full_mask)
            n_pruned += int(((full_mask == 0)).sum().item())
            n_total += w.numel()
    return {"sparsity_actual": n_pruned / max(n_total, 1), "n_pruned": n_pruned, "n_total": n_total}


@torch.no_grad()
def fake_quantize_int8(model: nn.Module) -> dict:
    """Per-tensor symmetric INT8 quantize -> dequantize (no speedup)."""
    n_quantized = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            scale = w.abs().max() / 127.0
            if scale == 0:
                continue
            q = torch.clamp(torch.round(w / scale), -128, 127)
            w.data.copy_(q * scale)  # dequantized; bits-equivalent stored as fp32
            m.register_buffer("_quant_scale", scale.unsqueeze(0))
            n_quantized += w.numel()
    return {"n_quantized_params": n_quantized}


def effective_param_bytes(model: nn.Module) -> int:
    """Effective bytes if we packed pruning + quantization for storage.

    Accounts for masks (1 bit per zero) and per-tensor INT8 bits where
    `_quant_scale` is registered. A real production runtime would also
    pack masks via 2:4 indices etc; this is an honest approximation.
    """
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight
            n = w.numel()
            quant = hasattr(m, "_quant_scale")
            mask = None
            if hasattr(m, "_prune_mask"):
                mask = m._prune_mask
            elif hasattr(m, "_2of4_mask"):
                mask = m._2of4_mask
            bits_per_param = 8 if quant else 32
            n_kept = int((mask != 0).sum().item()) if mask is not None else n
            total += n_kept * (bits_per_param // 8)
            if mask is not None:
                # Bookkeeping: 1 bit per param for the mask layout.
                total += (n + 7) // 8
            # Bias is usually small; count as fp32 always.
            if hasattr(m, "bias") and m.bias is not None:
                total += m.bias.numel() * 4
    # Add non-Conv/Linear params (BN, embeddings) in fp32.
    for name, p in model.named_parameters():
        owner = name.rsplit(".", 1)[0]
        try:
            mod = dict(model.named_modules())[owner]
        except KeyError:
            mod = None
        if not isinstance(mod, (nn.Conv2d, nn.Linear)):
            total += p.numel() * 4
    return total
