#!/usr/bin/env python3
"""Real TorchInductor output for one Conv-BatchNorm-ReLU block (inference).

Run: TORCH_LOGS=output_code python3 fused_kernel.py 2> log.txt
(PyTorch 2.13.0, Apple M5 Max CPU; data/fused_kernel_output.txt is log.txt
with the logger prefix stripped.)
Prints the code Inductor generates; the slide quotes the fused BN+ReLU kernel.
"""
import platform
import torch
import torch.nn as nn

torch.manual_seed(0)
block = nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
).eval()
x = torch.randn(1, 64, 56, 56)  # ResNet-50 stage-2 map, batch one
with torch.no_grad():
    compiled = torch.compile(block)
    y = compiled(x)
    ref = block(x)
print("torch", torch.__version__, platform.processor(), platform.machine())
print("max abs diff vs eager", (y - ref).abs().max().item())
