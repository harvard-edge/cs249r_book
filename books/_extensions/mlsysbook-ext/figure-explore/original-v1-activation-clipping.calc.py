#!/usr/bin/env python3
"""Provenance and numeric verification for original-v1-activation-clipping.js.

Computes exact UINT8 scale, clipped share, quantization RMSE, clipping RMSE, and
total RMSE from the canonical Wu et al. (2020) ResNet-50 layer1.0.conv2 post-ReLU
activation profile (books/vol1/10_model_compression/data/wu2020_resnet50_activation_profile.csv)
and verifies MLSysIM's affine UINT8 scale formula (Scenarios.Uint8RangeMinusOneToThree).
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
CSV_PATH = REPO / "books/vol1/10_model_compression/data/wu2020_resnet50_activation_profile.csv"
JS_PATH = HERE / "original-v1-activation-clipping.js"

sys.path.insert(0, str(REPO / "books/shared/python"))
sys.path.insert(0, str(REPO / "mlsysim/src"))
sys.path.insert(0, str(REPO / "mlsysim"))

from mlsysim import Scenarios  # noqa: E402

ALPHA = 0.0
Q_MIN = 0
Q_MAX = 255
THRESHOLDS = [2.0922, 1.3940, 1.1221, 0.8000, 0.5000]


def load_profile() -> tuple[list[dict[str, float | int]], dict[str, float]]:
    with CSV_PATH.open(encoding="utf-8") as handle:
        raw = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    bins: list[dict[str, float | int]] = []
    for row in raw:
        bins.append(
            {
                "left": round(float(row["bin_left"]), 4),
                "right": round(float(row["bin_right"]), 4),
                "mid": round(float(row["bin_center"]), 4),
                "count": int(row["count"]),
            }
        )
    markers = {
        "p9999": round(float(raw[0]["p9999_threshold"]), 4),
        "entropy": round(float(raw[0]["entropy_threshold"]), 4),
        "max": round(float(raw[0]["max_threshold"]), 4),
    }
    return bins, markers


def evaluate_threshold(bins: list[dict[str, float | int]], beta: float) -> dict[str, float | int]:
    ref = Scenarios.Uint8RangeMinusOneToThree.variant(alpha=ALPHA, beta=beta, q_min=Q_MIN, q_max=Q_MAX)
    scale = float(ref.scale)
    assert math.isclose(scale, (beta - ALPHA) / (Q_MAX - Q_MIN), rel_tol=1e-12)
    assert int(ref.zero_point) == 0

    total = sum(int(b["count"]) for b in bins)
    clipped = 0
    sq_clip = 0.0
    sq_quant = 0.0
    for b in bins:
        cnt = int(b["count"])
        if cnt == 0:
            continue
        x = float(b["mid"])
        if x > beta:
            clipped += cnt
            sq_clip += cnt * ((x - beta) ** 2)
        else:
            q = min(Q_MAX, max(Q_MIN, round(x / scale)))
            x_hat = q * scale
            sq_quant += cnt * ((x - x_hat) ** 2)
    rmse_quant = math.sqrt(sq_quant / total)
    rmse_clip = math.sqrt(sq_clip / total)
    rmse_total = math.sqrt((sq_quant + sq_clip) / total)
    return {
        "beta": round(beta, 4),
        "scale": round(scale, 6),
        "clipped": clipped,
        "clipped_pct": round(100.0 * clipped / total, 4),
        "rmse_quant": round(rmse_quant, 6),
        "rmse_clip": round(rmse_clip, 6),
        "rmse_total": round(rmse_total, 6),
    }


def build_payload() -> dict[str, object]:
    bins, markers = load_profile()
    total = sum(int(b["count"]) for b in bins)
    presets = [evaluate_threshold(bins, beta) for beta in THRESHOLDS]
    return {
        "total": total,
        "markers": markers,
        "axis": {
            "png_w": 3315,
            "png_h": 2234,
            "plot_x0": 372,
            "plot_x1": 3255,
            "plot_y0": 59,
            "plot_y1": 1908,
            "xlim": [-0.03, 2.25],
            "ylim": [0.42, 1.1e7],
        },
        "presets": presets,
        "bins": [[b["mid"], b["count"]] for b in bins if int(b["count"]) > 0],
    }


def main() -> int:
    payload = build_payload()
    if "--check" in sys.argv:
        js = JS_PATH.read_text(encoding="utf-8")
        match = re.search(r"const PROFILE = (\{.*?\});", js)
        if not match:
            raise AssertionError("Missing const PROFILE = {...}; in original-v1-activation-clipping.js")
        embedded = json.loads(match.group(1))
        assert embedded == payload, f"Embedded PROFILE drifted from CSV/MLSysIM:\n{embedded}\n!=\n{payload}"
        print("original-v1-activation-clipping.calc.py: OK")
        return 0
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
