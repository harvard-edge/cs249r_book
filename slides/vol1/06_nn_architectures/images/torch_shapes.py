#!/usr/bin/env python3
"""Real PyTorch output shown on three slides of Lecture 6 (Part 1).

Each block prints exactly what the slide quotes:
  1. nn.Linear(784, 128): the MNIST layer's weight shape and parameter count
     (dense frame).
  2. F.unfold on the 3x3 input 1..9 with a 2x2 window: the im2col patch
     matrix the slide draws by hand (unfold frame).
  3. nn.LSTM(100, 128): the fused gate weights, (4 x hidden, input) and
     (4 x hidden, hidden) (LSTM frame).

Needs torch (not in the MLSysBook venv):
  uv venv tvenv && uv pip install --python tvenv/bin/python torch
  tvenv/bin/python torch_shapes.py
Run for the deck with PyTorch 2.14.1 on CPU, 2026-10-18.
"""

import torch
import torch.nn.functional as F
from torch import nn

print("torch", torch.__version__)

print("--- dense")
layer = nn.Linear(784, 128)
print(layer.weight.shape)
print(sum(p.numel() for p in layer.parameters()))

print("--- unfold")
x = torch.arange(1., 10.).reshape(1, 1, 3, 3)
print(F.unfold(x, kernel_size=2))

print("--- lstm")
lstm = nn.LSTM(input_size=100, hidden_size=128)
print(lstm.weight_ih_l0.shape)
print(lstm.weight_hh_l0.shape)
