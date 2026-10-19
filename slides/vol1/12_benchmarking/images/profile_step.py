#!/usr/bin/env python3
"""One ResNet-50 training step under torch.profiler, run for Lecture 12.

Run: python3 profile_step.py > profile_step_output.txt
(CPU only, Apple M5 Max; torch 2.13.0, torchvision 0.28.0; batch 16.)
The slide quotes the first three columns of the top rows of the table.
"""
import torch
import torchvision
from torch.profiler import ProfilerActivity, profile

torch.manual_seed(0)
model = torchvision.models.resnet50()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x, t = torch.randn(16, 3, 224, 224), torch.randint(0, 1000, (16,))


def step():
    opt.zero_grad()
    torch.nn.functional.cross_entropy(model(x), t).backward()
    opt.step()


for _ in range(3):
    step()                                    # warm-up
with profile(activities=[ProfilerActivity.CPU]) as prof:
    step()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
