#!/usr/bin/env python3
"""Real PyTorch output for Lecture 5 (Neural Computation), run on this machine.

Two runs, written to data/torch_runs.txt together with the versions and the
machine, so the slides quote output that exists outside the deck:

  1. Autograd keeps the forward values. The 2-2-1 network of Napkin Math 5.2
     (x = [1.0, 0.5], biases zero, squared-error loss, target 1.0), built from
     plain tensors. Prints what each grad_fn saved (ReluBackward0 keeps its
     output; MmBackward0 keeps its left operand), the loss, the gradients,
     and w11 after one SGD step with learning rate 0.1. See the PyTorch
     autograd notes, "Saved tensors" (pytorch.org/docs/stable/notes/autograd.html).

  2. Where the forward pass goes. torch.profiler over the running network,
     784-128-64-10 with ReLU (nn.Linear layers), batch 32, FP32, one CPU
     thread, 200 forward passes after 50 warm-up passes. Prints the
     key_averages table (the PyTorch profiler recipe,
     pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

Needs torch (not in the MLSysBook venv; the system python3 has it):
  python3 torch_runs.py
"""
import platform
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

OUT = Path(__file__).resolve().parent / "data" / "torch_runs.txt"
lines = []


def emit(s=""):
    print(s)
    lines.append(str(s))


cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                     capture_output=True, text=True).stdout.strip() or platform.processor()
emit(f"torch {torch.__version__}; Python {platform.python_version()}; {cpu}; "
     f"{platform.platform()}")

# ---------------------------------------------------------------------------
# 1. Autograd keeps the forward values (Napkin Math 5.2)
emit("\n== 1. saved tensors, 2-2-1 network ==")
x = torch.tensor([[1.0, 0.5]])
W1 = torch.tensor([[0.5, -0.3], [0.8, 0.2]], requires_grad=True)
W2 = torch.tensor([[0.6], [-0.4]], requires_grad=True)
z = x @ W1
a = torch.relu(z)
y = a @ W2
loss = 0.5 * (y - 1.0) ** 2
emit(">>> a.grad_fn")
emit(a.grad_fn.name())
emit(">>> a.grad_fn._saved_result")
emit(a.grad_fn._saved_result)
emit(">>> y.grad_fn._saved_self")
emit(y.grad_fn._saved_self)
emit(">>> z.grad_fn._saved_self")
emit(z.grad_fn._saved_self)
emit(">>> loss")
emit(loss)
loss.backward()
emit(">>> W2.grad")
emit(W2.grad)
emit(">>> W1.grad")
emit(W1.grad)
with torch.no_grad():
    W1 -= 0.1 * W1.grad
emit(">>> W1[0, 0] after one SGD step, lr 0.1")
emit(f"{W1[0, 0].item():.4f}")

# ---------------------------------------------------------------------------
# 2. Profile the forward pass of the running network at batch 32
emit("\n== 2. torch.profiler, 784-128-64-10, batch 32, FP32, 1 thread ==")
torch.manual_seed(0)
torch.set_num_threads(1)
net = nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Linear(64, 10))
xb = torch.randn(32, 784)
with torch.no_grad():
    for _ in range(50):
        net(xb)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(200):
            net(xb)
emit(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT}")
