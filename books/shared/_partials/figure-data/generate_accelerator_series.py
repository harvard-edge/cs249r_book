#!/usr/bin/env python3
"""Emit the canonical NVIDIA data center accelerator series used by the
memory-wall family of figures.

Every value comes from the MLSysIM hardware registry, which carries the
vendor datasheet provenance for each part. Nothing here is hand-typed:
if a spec changes in the registry, re-running this script updates every
chapter that plots it, so the volumes cannot disagree with each other.

Consumers (one CSV written per chapter that plots the series):
    vol2/conclusion, vol2/inference, vol2/performance_engineering,
    vol2/compute_infrastructure, vol4/placement

Usage:  python3 generate_accelerator_series.py
"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENTS = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(CONTENTS, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "mlsysim"))

from mlsysim.hardware.registry import CloudHardware  # noqa: E402

# The registry's NVIDIA data center line, in release order. Registry coverage
# begins at Volta; extending the series earlier requires adding those parts to
# MLSysIM first (with datasheet provenance), never a chapter-local literal.
SERIES = ["V100", "A100", "H100", "B200"]

CONSUMERS = {
    "vol2/conclusion": "gpu_memory_wall.csv",
    "vol2/inference": "gpu_memory_wall.csv",
    "vol2/performance_engineering": "memory_wall_trend.csv",
    "vol2/compute_infrastructure": "gpu_scaling_trends.csv",
    "vol4/placement": "gpu_memory_wall.csv",
}


def _mag(q, unit):
    """Pint quantity -> float in `unit`."""
    return float(q.to(unit).magnitude)


def rows():
    out = []
    for key in SERIES:
        h = getattr(CloudHardware, key)
        prov = getattr(h.metadata, "provenance", None)
        out.append(
            {
                "Year": h.release_year,
                "Accelerator": key,
                "Peak_TFLOPS": round(_mag(h.compute.peak_flops, "TFLOPs/s"), 1),
                "Memory_Bandwidth_GBps": round(_mag(h.memory.bandwidth, "GB/s"), 1),
                "Memory_Capacity_GiB": round(_mag(h.memory.capacity, "GiB"), 0),
                "TDP_W": round(_mag(h.tdp, "W"), 0),
                "Source": getattr(prov, "url", "") or "",
            }
        )
    return out


def main():
    data = rows()
    fields = list(data[0].keys())
    for chapter, fname in CONSUMERS.items():
        out_dir = os.path.join(CONTENTS, chapter, "data")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, fname)
        with open(path, "w", newline="\n", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
        print(f"wrote {os.path.relpath(path, CONTENTS)}")

    c0, c1 = data[0]["Peak_TFLOPS"], data[-1]["Peak_TFLOPS"]
    b0, b1 = data[0]["Memory_Bandwidth_GBps"], data[-1]["Memory_Bandwidth_GBps"]
    print(
        f"\n{data[0]['Year']}-{data[-1]['Year']}: "
        f"compute x{c1/c0:.1f}, bandwidth x{b1/b0:.1f}, divergence x{(c1/c0)/(b1/b0):.2f}"
    )


if __name__ == "__main__":
    main()
