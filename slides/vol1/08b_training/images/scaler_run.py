#!/usr/bin/env python3
"""Real loss-scale trace for the Diagnose slide of Lecture 8b.

A small FP16 mixed-precision run with torch.amp.GradScaler at its default
settings (init_scale 2^16, growth_factor 2, backoff_factor 0.5,
growth_interval 2,000). The model is a two-layer MLP (64-256-10) trained
with SGD on one fixed random batch of 512 examples; it exists only to
produce real gradients, so the scale's behavior is what the slide reads.
After every step, scaler.get_scale() is written to data/scaler_trace.csv
with a flag for steps the scaler skipped (inf or NaN found in the scaled
gradients, which is when the scale halves).

Needs torch (not in the MLSysBook venv). Run with the mlperf-edu venv:
  ../../../../mlperf-edu/.venv/bin/python scaler_run.py
Device: the Apple GPU (MPS) if present, else CPU; the device and torch
version are written on the first line of the CSV.
"""
import csv
import platform
from pathlib import Path

import torch

STEPS = 28000
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.ReLU(),
                            torch.nn.Linear(256, 10)).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
scaler = torch.amp.GradScaler(dev)  # defaults
x = torch.randn(512, 64, device=dev)
y = torch.randint(0, 10, (512,), device=dev)

rows = []
for step in range(1, STEPS + 1):
    before = scaler.get_scale()
    with torch.autocast(dev, dtype=torch.float16):
        loss = torch.nn.functional.cross_entropy(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    after = scaler.get_scale()
    rows.append((step, after, int(after < before)))

out = Path(__file__).parent / "data" / "scaler_trace.csv"
with out.open("w", newline="") as f:
    f.write(f"# torch {torch.__version__}, device {dev}, {platform.platform()}, "
            f"GradScaler defaults (init 65536, growth 2, backoff 0.5, interval 2000)\n")
    w = csv.writer(f)
    w.writerow(["step", "scale_after_step", "skipped"])
    w.writerows(rows)
skips = [r[0] for r in rows if r[2]]
print(f"wrote {out}: {len(rows)} steps, {len(skips)} skipped steps, first at {skips[:1]}")
print("distinct scales:", sorted({r[1] for r in rows}))
