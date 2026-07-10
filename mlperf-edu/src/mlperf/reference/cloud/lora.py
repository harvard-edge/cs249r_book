"""
MLPerf EDU iter-7: LoRA adapter and injection helper.

Pure PyTorch, no `peft` library, no HF dependency. Targets the fused
`c_attn` (QKV) projection in `CausalSelfAttention`.

Per Han's iter-7 spec:
  - rank=8, alpha=16 (scale = alpha/rank = 2.0)
  - target_modules: c_attn only
  - Trainable params for GPT-2-Small (12 layers, d_model=768):
    12 * (768*8 + 8*3*768) = 12 * (6144 + 18432) = 294,912 params
    = 0.33% of the 88M base.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """Low-rank adapter A @ B injected as a residual on a Linear layer.

    Forward: y = base(x) + (alpha/rank) * (x @ A) @ B

    Where:
      base   : the frozen Linear (d_in -> d_out) we are adapting.
      A      : Linear(d_in -> rank), Kaiming-initialized, trainable.
      B      : Linear(rank -> d_out), zero-initialized, trainable.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRAAdapter expects nn.Linear, got {type(base)}")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Freeze base.
        for p in self.base.parameters():
            p.requires_grad = False

        # Init A with Kaiming, B with zero so initial delta = 0 (training
        # starts at the base model's behavior — standard LoRA convention).
        # Build the adapters on the same device + dtype as the base layer
        # so injection works regardless of where the base lives (MPS/CUDA/CPU).
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Linear(base.in_features, rank, bias=False).to(device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False).to(device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

    def merge(self) -> nn.Linear:
        """Return a new nn.Linear with W' = W_base + scale * B @ A baked in.

        Used for deployment-time inference: drop the LoRA branch entirely.
        Critical for the merged-vs-unmerged latency comparison.
        """
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                            bias=self.base.bias is not None)
        with torch.no_grad():
            delta = self.scale * (self.lora_B.weight @ self.lora_A.weight)
            merged.weight.copy_(self.base.weight + delta)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged.to(self.base.weight.device).to(self.base.weight.dtype)


def inject_lora(model: nn.Module, rank: int = 8, alpha: int = 16,
                 target: str = "c_attn") -> tuple[int, int]:
    """Walk the model, replace every `target`-named Linear with a LoRAAdapter.

    Returns (n_adapters_injected, total_trainable_lora_params).

    After injection, freeze every non-LoRA parameter. The model's `forward`
    path is unchanged because LoRAAdapter shares the base layer's call shape.
    """
    n_injected = 0
    n_trainable = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name == target and isinstance(child, nn.Linear):
                adapter = LoRAAdapter(child, rank=rank, alpha=alpha)
                setattr(module, child_name, adapter)
                n_injected += 1
                n_trainable += sum(p.numel() for p in adapter.lora_A.parameters())
                n_trainable += sum(p.numel() for p in adapter.lora_B.parameters())

    # Freeze every parameter that is NOT inside a LoRAAdapter.
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return n_injected, n_trainable


def base_grad_norm(model: nn.Module) -> float:
    """Sum of squared gradient norms over base (non-LoRA) parameters."""
    total = 0.0
    for name, p in model.named_parameters():
        if "lora_" in name:
            continue
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def lora_grad_norm(model: nn.Module) -> float:
    """Sum of squared gradient norms over LoRA-only parameters."""
    total = 0.0
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5
