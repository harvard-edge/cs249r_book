#!/usr/bin/env python3
"""Measure MobileNetV2 latency on this machine's CPU (Lecture 10, The System).

Real tool output, produced by running PyTorch here (not a mock). Three
variants of torchvision's MobileNetV2 at batch one, 3 x 224 x 224, one CPU
thread (the MobileNetV3 paper's phone numbers also use one large core):

  1. width 1.0, dense
  2. width 1.0 with 80 percent of conv and linear weights zeroed by
     torch.nn.utils.prune.global_unstructured (L1), still run by dense kernels
  3. width 0.75 (the width multiplier, a structured change)

FLOPs come from torch.utils.flop_counter.FlopCounterMode. Weights are the
random initialization: dense kernels take the same time for any values.

Run (the MLSysBook venv has no PyTorch; the mlperf-edu venv does):
  ../../../../mlperf-edu/.venv/bin/python measure_cpu.py
Writes cpu_measure.json beside this file; make_figs.py reads it.
"""

import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
from torch.utils.flop_counter import FlopCounterMode

torch.manual_seed(0)
torch.set_num_threads(1)
x = torch.randn(1, 3, 224, 224)


def build(width, sparsity=0.0):
    m = torchvision.models.mobilenet_v2(weights=None, width_mult=width).eval()
    if sparsity:
        params = [(mod, "weight") for mod in m.modules()
                  if isinstance(mod, (nn.Conv2d, nn.Linear))]
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured,
                                  amount=sparsity)
        for mod, name in params:
            prune.remove(mod, name)  # bake the zeros into a plain dense tensor
    return m


def zeros_share(m):
    ws = [mod.weight for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
    return sum(int((w == 0).sum()) for w in ws) / sum(w.numel() for w in ws)


def flops(m):
    with FlopCounterMode(display=False) as fc, torch.inference_mode():
        m(x)
    return fc.get_total_flops()


def timed(m, runs):
    ts = []
    with torch.inference_mode():
        for _ in range(runs):
            t0 = time.perf_counter()
            m(x)
            ts.append((time.perf_counter() - t0) * 1e3)
    return ts


variants = [("width 1.0, dense", 1.0, 0.0),
            ("width 1.0, 80% zeros", 1.0, 0.8),
            ("width 0.75, dense", 0.75, 0.0)]
models = [build(w, s) for _, w, s in variants]
for m in models:
    timed(m, 30)  # warm up
# Interleave: 20 rounds of 50 runs per variant, so drift in clock speed or
# background load hits every variant alike. Report the median of all runs.
samples = [[] for _ in models]
for _ in range(20):
    for i, m in enumerate(models):
        samples[i] += timed(m, 50)
rows = []
for (name, w, s), m, ts in zip(variants, models, samples):
    rows.append(dict(name=name, width=w, target_sparsity=s,
                     zeros_share=zeros_share(m),
                     params=sum(p.numel() for p in m.parameters()),
                     flops=flops(m), median_ms=statistics.median(ts),
                     p10_ms=statistics.quantiles(ts, n=10)[0],
                     p90_ms=statistics.quantiles(ts, n=10)[-1], runs=len(ts)))
    r = rows[-1]
    print(f"{name:22s} params {r['params']:>9,d}  zeros {r['zeros_share']:.3f}  "
          f"FLOPs {r['flops'] / 1e6:7.1f} M  median {r['median_ms']:.2f} ms "
          f"(p10 {r['p10_ms']:.2f}, p90 {r['p90_ms']:.2f}, {r['runs']} runs)")

cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                     capture_output=True, text=True).stdout.strip() or platform.processor()
out = dict(cpu=cpu, torch=torch.__version__, torchvision=torchvision.__version__,
           threads=torch.get_num_threads(), batch=1, input="1x3x224x224",
           date=time.strftime("%Y-%m-%d"), rows=rows)
(Path(__file__).parent / "cpu_measure.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"{cpu}, PyTorch {torch.__version__}, torchvision {torchvision.__version__}, "
      f"{torch.get_num_threads()} thread")
