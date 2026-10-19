#!/usr/bin/env python3
"""Real PyTorch transcripts for two Lecture 10 frames (not a mock).

Each ">>>" line below is executed as written and its result printed the way
the Python prompt prints it, so the slides can quote the transcript exactly.

  prune: torch.nn.utils.prune.l1_unstructured on the chapter's listing-1
         weights (Sec. 10.2.1), then the bytes of the dense and CSR forms.
  fuse:  torch.ao.quantization.fuse_modules on conv, batch norm, ReLU.

Run (the MLSysBook venv has no PyTorch; the mlperf-edu venv does):
  ../../../../mlperf-edu/.venv/bin/python torch_demos.py
torch_demos.txt keeps a copy of the output.
"""

import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.ao.quantization import fuse_modules

warnings.filterwarnings("ignore")
torch.manual_seed(0)
torch.set_printoptions(precision=2)


def nbytes(*ts):
    """Bytes held by the given tensors."""
    return sum(t.numel() * t.element_size() for t in ts)


ENV = dict(torch=torch, nn=nn, prune=prune, fuse_modules=fuse_modules, nbytes=nbytes)
lines = []


def run(src):
    lines.append(">>> " + src)
    try:
        val = eval(compile(src, "<demo>", "eval"), ENV)
        if val is not None:
            lines.append(repr(val))
    except SyntaxError:
        exec(compile(src, "<demo>", "exec"), ENV)


lines.append(f"# PyTorch {torch.__version__}")
lines.append("## prune")
# LISTING_1: the chapter's listing-1 weights (Sec. 10.2.1)
ENV["LISTING_1"] = torch.tensor([[0.8, 0.1, -0.7], [0.05, -0.9, 0.03], [-0.6, 0.02, 0.4]])
run("layer = nn.Linear(3, 3, bias=False)")
run("layer.weight.data = LISTING_1.clone()")
run("_ = prune.l1_unstructured(layer, 'weight', amount=5)")
run("layer.weight")
run("layer.weight.shape, list(dict(layer.named_buffers()))")
run("c = layer.weight.detach().to_sparse_csr()")
run("nbytes(layer.weight)")
run("nbytes(c.values(), c.col_indices(), c.crow_indices())")
lines.append("## fuse")
run("conv, bn = nn.Conv2d(16, 16, 3), nn.BatchNorm2d(16)")
run("_ = bn.running_var.uniform_(0.5, 2)")
run("m = nn.Sequential(conv, bn, nn.ReLU()).eval()")
run("[type(x).__name__ for x in m]")
run("f = fuse_modules(m, [['0', '1', '2']])")
run("[type(x).__name__ for x in f]")
run("x = torch.randn(1, 16, 28, 28)")
run("(f(x) - m(x)).abs().max().item()")

out = "\n".join(lines) + "\n"
print(out, end="")
(Path(__file__).parent / "torch_demos.txt").write_text(out)
