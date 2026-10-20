#!/usr/bin/env python3
"""Measure one exact vector scan (a query scored against every stored vector)
on the machine that runs this script, and write scan_bench.csv.

This is the deck's own measurement, shown as measured data on the slide
"An exact scan on a laptop runs near its bandwidth floor". The chapter's
Napkin Math 6.2 uses FP16; numpy has no fast FP16 matrix-vector product, so
this run stores FP32 (4 bytes per component) and says so on the slide.

  - d = 768 components, as in Napkin Math 6.2.
  - Scan time: median of 7 runs of one matrix-vector product, after a warm-up.
  - Achieved bandwidth: bytes streamed divided by scan time, per row.

Run from this folder: python3 scan_bench.py   (takes about a minute)
"""

import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

D = 768
B = 4  # bytes per FP32 component
COUNTS = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
out = Path(__file__).parent


def median_time(fn, reps=7):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


rng = np.random.default_rng(0)

rows = []
for n in COUNTS:
    x = rng.standard_normal((n, D), dtype=np.float32)
    q = rng.standard_normal(D, dtype=np.float32)
    t = median_time(lambda: x @ q)
    rows.append((n, n * D * B, t))
    print(f"N={n:>9,}  bytes={n * D * B / 1e9:7.3f} GB  scan={t * 1e3:8.3f} ms  "
          f"achieved={n * D * B / t / 1e9:6.1f} GB/s  {2 * n * D / t / 1e9:6.1f} GFLOP/s")
    del x

try:
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
except OSError:
    chip = platform.processor()
with open(out / "scan_bench.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["# machine", chip, "numpy", np.__version__, "date", time.strftime("%Y-%m-%d"),
                "threads", "numpy default"])
    w.writerow(["vectors", "bytes", "scan_s"])
    for r in rows:
        w.writerow([r[0], r[1], f"{r[2]:.6f}"])
print(f"machine: {chip}")
