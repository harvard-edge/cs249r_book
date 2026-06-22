#!/usr/bin/env python3
"""Drift-guard for the @fig-ai-performance TikZ figure (hw_acceleration).

Why this exists
---------------
``@fig-ai-performance`` is a hand-authored TikZ/pgfplots figure. TikZ cannot
import mlsysim at render time, so its NVIDIA accelerator throughput numbers are
necessarily hardcoded literals in the ``.qmd``. This check pins those literals
to the mlsysim registry so they cannot silently drift: if a registry spec
changes (or the figure label is mistyped), this fails and forces a review.

Only the registry-backed *modern* accelerators are guarded. The historical
parts (K20X, M40, P40) are documented literals with no registry home and are
intentionally exempt (reported as INFO).

Metric mapping (figure label  ->  registry field x sparsity)
-----------------------------------------------------------
  V100  125  TFLOP/s FP16  ->  V100.compute.peak_flops            (dense FP16)
  A100  1,248 TOPS INT8    ->  A100.precision_flops['int8'] x 2   (2:4 sparse)
  H100  4,000 TFLOP/s FP8  ->  H100.precision_flops['fp8']  x 2   (2:4 sparse)
  B200  9,000 TFLOP/s FP8  ->  B200.precision_flops['fp8']  x 2   (2:4 sparse)

The book plots each generation's *headline* (sparse) marketing throughput, so
the 2x sparsity factor is applied to the registry's dense figure. A 5% tolerance
absorbs vendor rounding (e.g. H100 1979x2 = 3958, shown as 4,000).

Usage:  python3 book/tools/audit/check_fig_ai_performance_drift.py
Exit:   0 = figure matches registry, 1 = drift detected.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from mlsysim import Hardware, TFLOPs, TOPS, second

QMD = (
    Path(__file__).resolve().parents[2]
    / "quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd"
)
TOL = 0.05  # 5% tolerance for vendor rounding

# chip -> (registry getter, sparsity factor, expected label substring/unit)
SPECS = {
    "V100": (lambda c: c.compute.peak_flops, 1.0, "TFLOP/s FP16"),
    "A100": (lambda c: c.compute.precision_flops["int8"], 2.0, "TOPS INT8"),
    "H100": (lambda c: c.compute.precision_flops["fp8"], 2.0, "TFLOP/s FP8"),
    "B200": (lambda c: c.compute.precision_flops["fp8"], 2.0, "TFLOP/s FP8"),
}
HISTORICAL = ["K20X", "M40", "P40"]  # no registry home — documented literals


def tnum(q) -> float:
    """Magnitude of a throughput Quantity in T-scale (TFLOP/s or TOPS)."""
    for unit in (TFLOPs / second, TOPS):
        try:
            return float(q.m_as(unit))
        except Exception:
            pass
    return float(getattr(q, "magnitude", q))


def fig_block(text: str) -> str:
    """Isolate the fig-ai-performance tikz block."""
    start = text.index("#fig-ai-performance")
    end = text.index(":::", text.index("```", start))
    return text[start:end]


def label_value(block: str, chip: str) -> float | None:
    """Extract the numeric throughput from the chip's node label."""
    # e.g.  {\textbf{H100} \\ {\color{black!60}4,000 TFLOP/s FP8}}
    m = re.search(
        r"\\textbf\{" + re.escape(chip) + r"\}\s*\\\\\s*\{\\color\{[^}]*\}\s*"
        r"([0-9][0-9,\.]*)",
        block,
    )
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def main() -> int:
    text = QMD.read_text(encoding="utf-8")
    block = fig_block(text)

    print(f"Drift-guard: {QMD.name} @fig-ai-performance\n")
    print(f"{'chip':6} {'figure':>9} {'registry x f':>14} {'Δ':>7}  status")
    print("-" * 50)

    failures = []
    for chip, (getter, factor, unit_hint) in SPECS.items():
        node = getattr(Hardware.Cloud, chip, None)
        if node is None:
            failures.append(f"{chip}: absent from registry")
            print(f"{chip:6} {'?':>9} {'ABSENT':>14}      FAIL")
            continue
        expected = tnum(getter(node)) * factor
        shown = label_value(block, chip)
        if shown is None:
            failures.append(f"{chip}: label not found in figure")
            print(f"{chip:6} {'NOT FOUND':>9} {expected:>14.0f}      FAIL")
            continue
        rel = abs(shown - expected) / expected
        ok = rel <= TOL and unit_hint in block
        status = "ok" if ok else "FAIL"
        print(f"{chip:6} {shown:>9.0f} {expected:>14.0f} {rel*100:>6.1f}%  {status}")
        if not ok:
            failures.append(
                f"{chip}: figure {shown:.0f} vs registry-derived {expected:.0f} "
                f"({rel*100:.1f}% > {TOL*100:.0f}% tol) or unit '{unit_hint}' missing"
            )

    print("\nINFO (no registry home, exempt):", ", ".join(HISTORICAL))

    if failures:
        print("\nDRIFT DETECTED:")
        for f in failures:
            print(f"  - {f}")
        print(
            "\nFix: update the figure label/coordinate in hw_acceleration.qmd to "
            "match the registry, or update the registry spec."
        )
        return 1
    print("\nOK: all registry-backed figure values match mlsysim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
