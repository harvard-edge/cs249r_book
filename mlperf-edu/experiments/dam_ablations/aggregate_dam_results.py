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


def _resolve_seconds(metrics: dict[str, Any]) -> float | None:
    """Find a run's wall-clock, whatever the runner chose to call it.

    Runners name their timing metric after the phase they measure:
    duration_seconds for in-process inference, train_and_eval_seconds for
    training loops, self_play_and_training_seconds for the containerized
    reinforcement runner, wall_time_seconds as a fallback. Keying only on
    duration_seconds silently dropped MiniGo from the Machine lens, which
    looked like a missing run rather than a naming difference.
    """
    preferred = (
        "duration_seconds_mean",
        "duration_seconds",
        "train_and_eval_seconds_mean",
        "self_play_and_training_seconds_mean",
        "generation_and_evaluation_seconds_mean",
        "inference_seconds",
        "wall_time_seconds",
    )
    for key in preferred:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    # Last resort: any mean-aggregated seconds metric, chosen deterministically.
    for key in sorted(metrics):
        if key.endswith("_seconds_mean") and isinstance(metrics[key], (int, float)):
            return float(metrics[key])
    return None


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
        # Some contracts can import a prior generation instead of executing it
        # (MLPERF_EDU_EDM_IMPORT_PACKET, MLPERF_EDU_BFCL_IMPORT_PACKET). An
        # imported cell is a valid artifact but it is not a measurement made
        # here, so it must never enter a results table unlabelled.
        source = str(config.get("generation_source", "") or "")
        imported = "import" in source.lower()
        # image-generation swaps sampler and state precision on MPS, so its
        # accelerator cell is a variant of the reference path rather than the
        # official one. The runner records this; carry it forward.
        state_precision = str(config.get("state_precision", "") or "")
        variant_path = "mps" in state_precision.lower()
        runs.append(
            {
                "plan": plan,
                "cell": re.sub(r"^\d+-", "", run_dir),
                "imported": imported,
                "generation_source": source,
                "variant_path": variant_path,
                "state_precision": state_precision,
                "workload": report.get("workload") or report.get("id"),
                "device": report.get("device_executed")
                or (report.get("backend") or "").replace("pytorch-", ""),
                "seconds": _resolve_seconds(metrics),
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
    # Draw only from plans whose purpose is Machine-lens measurement. Without
    # this the table silently changed as unrelated runs accumulated: adding the
    # precision sweep moved text-classification from 44.08/10.09 (4.37x) to
    # 76.73/6.25 (12.28x), because the selector took whichever cell it saw
    # first. Same workload, same config, headline number 2.8x apart. A ratio is
    # only meaningful between runs taken under the same conditions, so the lens
    # pins its source plans and names them in the output.
    machine_plans = ("dam-machine-lens-fast", "dam-machine-lens-gcn",
                     "dam-machine-lens-basis")
    by_workload: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        if run["plan"] not in machine_plans:
            continue
        # Only the pinned-contract cells belong in the Machine lens.
        if run["hidden"] not in (None, 256) or run["epochs"] not in (None, 500):
            continue
        if run["requested_precision"] not in (None, "float32"):
            continue
        device = run["device"]
        if device in ("cpu", "mps") and device not in by_workload[run["workload"]]:
            by_workload[run["workload"]][device] = run

    lines = [
        "| Workload | CPU (s) | MPS (s) | Speedup | Quality | Target met | Source plan |",
        "|:---|---:|---:|---:|---:|:---|:---|",
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
        plans = sorted({r["plan"] for r in (cpu, mps) if r})
        lines.append(
            f"| {workload} | {cpu_s} | {mps_s} | {speedup} | {obs} | {met} "
            f"| {', '.join(p.replace('dam-machine-lens-', '') for p in plans)} |"
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
    # Prefer cells from the factorial plans. A speedup contrast is only valid
    # between runs executed under the same plan and host conditions; mixing in
    # the standalone machine-lens run shifted the D=500 A=256 ratio from 2.38x
    # to 2.30x purely because it was a different invocation.
    candidates = [
        r
        for r in runs
        if r["workload"] == "graph-node-classification" and r["epochs"] and r["hidden"]
    ]
    factorial_runs = [r for r in candidates if "factorial" in r["plan"]]
    preferred = factorial_runs or candidates
    cells: dict[tuple, dict[str, Any]] = {}
    for run in preferred:
        cells.setdefault((run["epochs"], run["hidden"], run["device"]), run)
    # Backfill any cell the factorial plans have not produced yet.
    for run in candidates:
        cells.setdefault((run["epochs"], run["hidden"], run["device"]), run)
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


def provenance_flags(runs: list[dict[str, Any]]) -> list[str]:
    """Surface cells that are not straightforward local measurements."""
    lines: list[str] = []
    for run in runs:
        if run.get("imported"):
            lines.append(
                f"- **IMPORTED, not executed here:** `{run['workload']}` / "
                f"`{run['cell']}` (`generation_source={run['generation_source']}`). "
                "Valid evidence, but it is not a measurement made by this run."
            )
        if run.get("variant_path"):
            lines.append(
                f"- **Variant execution path:** `{run['workload']}` / "
                f"`{run['cell']}` ran `{run['state_precision']}`. This is not the "
                "official reference path, so it is not like-for-like with a CPU "
                "cell or with the published reference."
            )
    return lines or ["_All cells executed locally on the official path._"]


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
        "## Provenance flags",
        "",
        *provenance_flags(runs),
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
