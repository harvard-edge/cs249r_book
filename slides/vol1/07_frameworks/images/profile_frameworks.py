#!/usr/bin/env python3
"""Real runs behind six slides of Lecture 7 (ML Frameworks).

Nothing here is typed by hand: every file in data/ is printed by PyTorch on
this machine. Each run writes one file in data/:

  graph_break.txt  torch.compile on the deck's branching function, with
                   TORCH_LOGS="graph_breaks,recompiles" (the log lines the
                   compile slide quotes), run in a subprocess.
  tape.txt         z.grad_fn and next_functions for the tape slide's
                   three-line program.
  inplace_error.txt
                   The diagnose slide's in-place edit, run for its real
                   RuntimeError, and torch.jit.trace's warnings on the
                   predict slide's branching function.
  resnet50_memory.csv
                   ResNet-50 (torchvision, random init), batch 64,
                   224x224, FP32, Adam, two training steps on the Apple GPU
                   (MPS). torch.mps.current_allocated_memory() is sampled
                   after every leaf module's forward, after every backward
                   node of those modules, and after zero_grad and step.
  mnist_step.txt   The chapter's two-layer MNIST network (784-128-10) with
                   Adam on the Apple GPU: mean wall time of an eager training
                   step at batch 32 and batch 1024 (300 steps each, after
                   20 warm-up steps, synchronized; 7 interleaved trials,
                   median reported).
  resnet50_peak_b128.txt
                   The same ResNet-50 at batch 128, one forward and backward,
                   peak allocated bytes.

Needs torch and torchvision (not in the MLSysBook venv):
  uv venv tvenv && uv pip install --python tvenv/bin/python torch torchvision
  tvenv/bin/python profile_frameworks.py
"""
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)
CPU = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                     capture_output=True, text=True).stdout.strip() or platform.processor()
HEAD = f"# torch {torch.__version__}, Python {platform.python_version()}, {CPU}, macOS {platform.mac_ver()[0]}\n"

# -- 1. Graph breaks and recompiles -------------------------------------------
BRANCH = '''
import torch
@torch.compile
def f(x):
    if x.sum() > 0: return x * 2
    return x * 3
print(f(torch.tensor([1.0])))
print(f(torch.tensor([-1.0])))
print(f(torch.tensor([1., 2.])))
'''
if "--child" not in sys.argv:
    env = dict(os.environ, TORCH_LOGS="graph_breaks,recompiles")
    r = subprocess.run([sys.executable, "-c", BRANCH], env=env, capture_output=True, text=True)
    keep = []
    for line in (r.stderr + r.stdout).splitlines():
        # drop the logger prefix (timestamp, pid, file:line), keep the message
        msg = line.split("] ", 2)[-1] if "[__" in line else line
        keep.append(msg)
    (DATA / "graph_break.txt").write_text(
        HEAD + '# command: TORCH_LOGS="graph_breaks,recompiles" python -c <BRANCH above>\n'
        + "\n".join(keep) + "\n")
    print("graph_break.txt:", len(keep), "lines")

# -- 2. The autograd tape -------------------------------------------------------
x = torch.ones(1, requires_grad=True)
y = x * 2
z = y + 1
lines = [f"z.grad_fn = {type(z.grad_fn).__name__}",
         f"z.grad_fn.next_functions = {tuple((type(n).__name__ if n else None, i) for n, i in z.grad_fn.next_functions)}",
         f"y.grad_fn.next_functions = {tuple((type(n).__name__ if n else None, i) for n, i in y.grad_fn.next_functions)}"]
z.backward()
lines.append(f"x.grad = {x.grad}")
(DATA / "tape.txt").write_text(HEAD + "\n".join(lines) + "\n")
print("\n".join(lines))

# -- 2b. The in-place error of the diagnose slide, and jit.trace's warnings ----
import warnings  # noqa: E402

torch.manual_seed(0)
xi = torch.randn(4, 3)
w = torch.randn(3, 2, requires_grad=True)
h = xi @ w
a = torch.sigmoid(h)
a += 1
try:
    a.sum().backward()
    err = "no error"
except RuntimeError as e:
    err = f"RuntimeError: {e}"


def branch(x):
    if x.sum() > 0:
        return x * 2
    return x * 3


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    traced = torch.jit.trace(branch, torch.tensor([1.0]))
    out_neg = traced(torch.tensor([-1.0]))
seen = []
for m in caught:
    line = f"{m.category.__name__}: {m.message}"
    if line not in seen:
        seen.append(line)
(DATA / "inplace_error.txt").write_text(
    HEAD + "# x: 4x3, w: 3x2 (requires grad); h = x @ w; a = sigmoid(h); a += 1; a.sum().backward()\n"
    + err + "\n\n# torch.jit.trace(branch, tensor([1.0])); traced(tensor([-1.0]))\n"
    + "\n".join(seen) + f"\nresult: {out_neg}\n")
print(err[:120])

dev = torch.device("mps")

# -- 3. ResNet-50 training memory over two steps -------------------------------
import torchvision  # noqa: E402

torch.manual_seed(0)
model = torchvision.models.resnet50(weights=None).to(dev).train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
n_params = sum(p.numel() for p in model.parameters())
xb = torch.randn(64, 3, 224, 224, device=dev)
yb = torch.randint(0, 1000, (64,), device=dev)
rows = []
step = [0]


def mem():
    return torch.mps.current_allocated_memory()


def rec(phase, name):
    rows.append((len(rows), step[0], phase, name, mem()))


def fwd_hook(mod, inp, out):
    rec("forward", mod._name)
    if isinstance(out, torch.Tensor) and out.grad_fn is not None:
        out.grad_fn.register_hook(lambda gi, go, n=mod._name: rec("backward", n))


for name, m in model.named_modules():
    if len(list(m.children())) == 0:
        m._name = name
        m.register_forward_hook(fwd_hook)

torch.mps.synchronize()
for s in (1, 2):
    step[0] = s
    rec("start", "")
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    rec("after_backward", "")
    opt.step()
    rec("after_step", "")
    opt.zero_grad(set_to_none=True)
    rec("after_zero_grad", "")
    torch.mps.synchronize()
with open(DATA / "resnet50_memory.csv", "w") as fh:
    fh.write(HEAD + f"# ResNet-50 (torchvision {torchvision.__version__}, random init), {n_params} parameters, "
             "batch 64, 3x224x224, FP32, Adam, Apple GPU (MPS)\n"
             "# bytes = torch.mps.current_allocated_memory() after each event\n")
    fh.write("event,step,phase,module,bytes\n")
    for r in rows:
        fh.write(",".join(str(v) for v in r) + "\n")
print("resnet50_memory.csv:", len(rows), "samples; peak", max(r[4] for r in rows) / 1e9, "GB")
# One more step at batch 128, peak only, to see how the activations scale.
del opt, xb, yb
torch.mps.empty_cache()
model.zero_grad(set_to_none=True)
xb = torch.randn(128, 3, 224, 224, device=dev)
yb = torch.randint(0, 1000, (128,), device=dev)
rows.clear()
step[0] = 128
torch.mps.synchronize()
rec("start", "")
F.cross_entropy(model(xb), yb).backward()
rec("after_backward", "")
torch.mps.synchronize()
peak128 = max(r[4] for r in rows)
(DATA / "resnet50_peak_b128.txt").write_text(
    HEAD + "# ResNet-50, batch 128, one forward and backward (no optimizer state), Apple GPU (MPS)\n"
    f"start bytes: {rows[0][4]}\npeak bytes: {peak128}\n")
print("batch 128 peak", peak128 / 1e9, "GB")
del model, xb, yb
torch.mps.empty_cache()


# -- 4. The MNIST training step: wall time and profile -------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def train_step(model, opt, x, y):
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()


out = [HEAD, "# MNIST MLP 784-128-10, FP32, Adam (default foreach update), eager, Apple GPU (MPS)\n",
       "# mean wall time per step over 300 synchronized steps after 20 warm-up steps;\n",
       "# 7 trials, batch sizes interleaved; the median trial is the reported value\n"]
setups = {}
for batch in (32, 1024):
    torch.manual_seed(0)
    model = MLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch, 784, device=dev)
    y = torch.randint(0, 10, (batch,), device=dev)
    for _ in range(20):
        train_step(model, opt, x, y)
    setups[batch] = (model, opt, x, y)
trials = {32: [], 1024: []}
for _ in range(7):
    for batch, (model, opt, x, y) in setups.items():
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(300):
            train_step(model, opt, x, y)
        torch.mps.synchronize()
        trials[batch].append((time.perf_counter() - t0) / 300 * 1e6)
for batch, ts in trials.items():
    ts = sorted(ts)
    out.append(f"batch {batch}: median step {ts[3]:.1f} us (trials {', '.join(f'{t:.0f}' for t in ts)})\n")
model = setups[32][0]
out.append(f"parameters: {sum(p.numel() for p in model.parameters())}\n")
print("".join(out))
(DATA / "mnist_step.txt").write_text("".join(out))
