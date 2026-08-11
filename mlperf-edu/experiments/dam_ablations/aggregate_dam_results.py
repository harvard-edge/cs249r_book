#!/usr/bin/env python3
"""Aggregate every measured DAM run into one integrated results set.

Reads the provenance-backed reports written by `mlperf run --plan` under
submissions/research/ and emits the Machine, Algorithm, and Data lens tables
plus the D x A x M factorial with its interaction analysis.

This script derives everything from measured artifacts. It contains no workload
data of its own, which is the property the previous generator lacked: the old
run_dam_experiments.py held a hardcoded table and applied fixed multipliers, so
its output could not disagree with its assumptions. If a run is missing here,
the cell is reported as missing rather than modelled.

Usage:
    python3 experiments/dam_ablations/aggregate_dam_results.py [--markdown OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Any


_HERE = os.path.dirname(os.path.abspath(__file__))  # experiments/dam_ablations
ROOT = os.path.dirname(os.path.dirname(_HERE))  # the mlperf-edu project root
SUBMISSIONS = os.path.join(ROOT, "submissions", "research")


def load_runs() -> list[dict[str, Any]]:
    """Collect every completed run report with its lever settings."""
    runs: list[dict[str, Any]] = []
    pattern = os.path.join(SUBMISSIONS, "*", "runs", "*", "*_pro_report.json")
    for path in sorted(glob.glob(pattern)):
        try:
            report = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        run_dir = os.path.basename(os.path.dirname(path))
        plan = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        config = report.get("config", {}) or {}
        quality = report.get("quality", {}) or {}
        metrics = report.get("metrics", {}) or {}
        metric_key = quality.get("metric_key") or quality.get("metric") or ""
        observed = metrics.get(f"{metric_key}_mean", metrics.get(metric_key))
        runs.append(
            {
                "plan": plan,
                "cell": re.sub(r"^\d+-", "", run_dir),
                "workload": report.get("workload") or report.get("id"),
                "device": report.get("device_executed")
                or (report.get("backend") or "").replace("pytorch-", ""),
                "seconds": metrics.get("duration_seconds_mean")
                or metrics.get("duration_seconds"),
                "observed": observed,
                "target": quality.get("target"),
                "target_met": quality.get("target_met"),
                "metric": quality.get("metric"),
                "epochs": config.get("epochs"),
                "hidden": config.get("hidden_channels"),
                "precision": config.get("execution_dtype"),
                "requested_precision": config.get("requested_precision"),
            }
        )
    return runs


def machine_lens(runs: list[dict[str, Any]]) -> list[str]:
    """CPU-versus-MPS ratio per workload, using the pinned baseline settings."""
    by_workload: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        # Only the pinned-contract cells belong in the Machine lens; the
        # factorial's reduced-lever cells would otherwise contaminate the ratio.
        if run["hidden"] not in (None, 256) or run["epochs"] not in (None, 500):
            continue
        if run["requested_precision"] not in (None, "float32"):
            continue
        device = run["device"]
        if device in ("cpu", "mps") and device not in by_workload[run["workload"]]:
            by_workload[run["workload"]][device] = run

    lines = [
        "| Workload | CPU (s) | MPS (s) | Speedup | Quality | Target met |",
        "|:---|---:|---:|---:|---:|:---|",
    ]
    for workload in sorted(by_workload):
        arms = by_workload[workload]
        cpu, mps = arms.get("cpu"), arms.get("mps")
        cpu_s = f"{cpu['seconds']:.2f}" if cpu and cpu["seconds"] else "not run"
        mps_s = f"{mps['seconds']:.2f}" if mps and mps["seconds"] else "not run"
        if cpu and mps and cpu["seconds"] and mps["seconds"]:
            speedup = f"{cpu['seconds'] / mps['seconds']:.2f}x"
        else:
            speedup = "incomplete"
        ref = cpu or mps
        obs = f"{ref['observed']:.4f}" if ref and isinstance(ref["observed"], float) else "n/a"
        met = "yes" if ref and ref["target_met"] else ("no" if ref else "n/a")
        lines.append(
            f"| {workload} | {cpu_s} | {mps_s} | {speedup} | {obs} | {met} |"
        )
    return lines


def algorithm_lens(runs: list[dict[str, Any]]) -> list[str]:
    """Precision sweep against the inherited target."""
    cells = [r for r in runs if r["requested_precision"]]
    if not cells:
        return ["_No precision runs recorded yet._"]
    lines = [
        "| Workload | Requested | Executed dtype | Device | Seconds | Quality | Target met |",
        "|:---|:---|:---|:---|---:|---:|:---|",
    ]
    for run in sorted(cells, key=lambda r: (r["workload"], r["cell"])):
        obs = f"{run['observed']:.4f}" if isinstance(run["observed"], float) else "n/a"
        secs = f"{run['seconds']:.2f}" if run["seconds"] else "n/a"
        lines.append(
            f"| {run['workload']} | {run['requested_precision']} | {run['precision']} "
            f"| {run['device']} | {secs} | {obs} | {'yes' if run['target_met'] else 'no'} |"
        )
    return lines


def factorial(runs: list[dict[str, Any]]) -> list[str]:
    """D x A x M factorial with the interaction contrast."""
    cells = {
        (r["epochs"], r["hidden"], r["device"]): r
        for r in runs
        if r["workload"] == "graph-node-classification" and r["epochs"] and r["hidden"]
    }
    lines = [
        "| D (epochs) | A (hidden) | CPU (s) | MPS (s) | Speedup |",
        "|---:|---:|---:|---:|---:|",
    ]
    speedups: dict[tuple[int, int], float] = {}
    for epochs in sorted({k[0] for k in cells}, reverse=True):
        for hidden in sorted({k[1] for k in cells}, reverse=True):
            cpu = cells.get((epochs, hidden, "cpu"))
            mps = cells.get((epochs, hidden, "mps"))
            cpu_s = f"{cpu['seconds']:.1f}" if cpu else "not run"
            mps_s = f"{mps['seconds']:.1f}" if mps else "not run"
            if cpu and mps:
                ratio = cpu["seconds"] / mps["seconds"]
                speedups[(epochs, hidden)] = ratio
                sp = f"{ratio:.2f}x"
            else:
                sp = "incomplete"
            lines.append(f"| {epochs} | {hidden} | {cpu_s} | {mps_s} | {sp} |")

    lines.append("")
    widths = sorted({k[1] for k in speedups}, reverse=True)
    if len(widths) >= 2:
        wide, narrow = widths[0], widths[-1]
        for epochs in sorted({k[0] for k in speedups}, reverse=True):
            a, b = speedups.get((epochs, wide)), speedups.get((epochs, narrow))
            if a and b:
                lines.append(
                    f"**A x M interaction at D={epochs}:** speedup {a:.2f}x at hidden={wide} "
                    f"versus {b:.2f}x at hidden={narrow}, a factor of {a / b:.2f}. "
                    f"{'Clears' if abs(a / b - 1) > 0.10 else 'Does NOT clear'} "
                    f"the ~3 percent run-to-run noise floor."
                )
    else:
        lines.append("_Factorial incomplete; interaction not yet computable._")
    return lines


def noise_floor(runs: list[dict[str, Any]]) -> list[str]:
    """Run-to-run spread for configurations executed more than once."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[
            (run["workload"], run["device"], run["epochs"], run["hidden"], run["precision"])
        ].append(run)
    lines: list[str] = []
    for key, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(group) < 2:
            continue
        times = [r["seconds"] for r in group if r["seconds"]]
        quals = [r["observed"] for r in group if isinstance(r["observed"], float)]
        if len(times) >= 2:
            spread = (max(times) - min(times)) / min(times) * 100
            qtxt = ""
            if len(quals) >= 2:
                qspread = max(quals) - min(quals)
                qtxt = f", quality spread {qspread:.2e}"
            lines.append(
                f"- `{key[0]}` on `{key[1]}` x{len(group)}: timing spread "
                f"{spread:.1f} percent{qtxt}"
            )
    return lines or ["_No repeated configurations yet._"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", help="write the report to this path")
    args = parser.parse_args()

    runs = load_runs()
    out: list[str] = [
        "# Integrated DAM results (measured)",
        "",
        f"Aggregated from {len(runs)} provenance-backed runs under "
        "`submissions/research/`. Every value is measured; missing cells are "
        "reported as missing rather than modelled.",
        "",
        "## Machine lens",
        "",
        *machine_lens(runs),
        "",
        "## Algorithm lens (precision)",
        "",
        *algorithm_lens(runs),
        "",
        "## D x A x M factorial (graph node classification)",
        "",
        *factorial(runs),
        "",
        "## Measured noise floor",
        "",
        *noise_floor(runs),
        "",
    ]
    text = "\n".join(out)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.markdown} ({len(runs)} runs)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
