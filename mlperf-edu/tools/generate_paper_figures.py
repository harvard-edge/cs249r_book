#!/usr/bin/env python3
"""Generate the paper's figures from committed evidence and run reports.

Four figures, each answering a question a reviewer will ask:

  fig_quality_vs_target   Did the inherited contracts reproduce? Shows every
                          workload's observed value normalised against its own
                          published target, so passes and misses sit on one axis
                          despite having incomparable metrics.

  fig_runtime             Does it fit a class period? Runtime per workload on a
                          log axis, with the suite total annotated.

  fig_training_curves     Does it converge, and how fast? Per-epoch quality for
                          every workload that trains, with the target line.

Reads the registry, the evidence index, and any run reports passed on the
command line. Writes PDF and PNG into paper/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlperf.registry import load_registry  # noqa: E402

OUT = ROOT / "paper" / "figures"
COMMITTED_RUNS = ROOT / "paper" / "evidence" / "runs"
PASS_COLOR = "#2b7a3d"
MISS_COLOR = "#b3402f"
NEUTRAL = "#4a6fa5"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        # Matplotlib stamps a creation timestamp into the file, so regenerating
        # unchanged figures would show up as a diff and make the committed
        # copies look stale on every run. Suppressing it keeps regeneration
        # byte-identical, which is what lets the figures be checked in at all.
        metadata = {"CreationDate": None} if suffix == "pdf" else {}
        fig.savefig(OUT / f"{name}.{suffix}", metadata=metadata)
    plt.close(fig)
    print(f"wrote {OUT / f'{name}.pdf'}")


def load_reports(paths: list[Path]) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for path in paths:
        for report in sorted(path.rglob("*_max_report.json")):
            try:
                payload = json.loads(report.read_text())
            except json.JSONDecodeError:
                continue
            if "workload" in payload:
                reports[payload["workload"]] = payload
    return reports


def load_evidence() -> dict[str, dict]:
    index_path = ROOT / "provisional_results" / "index.json"
    index = json.loads(index_path.read_text())
    cases = index if isinstance(index, list) else index.get("cases", [])
    out: dict[str, dict] = {}
    for entry in cases:
        record = json.loads((ROOT / entry["path"]).read_text())
        record["_entry"] = entry
        out.setdefault(entry["workload"], []).append(record)
    return out


def fig_quality_vs_target(workloads, reports, evidence) -> None:
    """Observed / target, so metrics with different scales share one axis."""
    rows = []
    for wid, workload in sorted(workloads.items()):
        gate = (workload.raw.get("canonical_max_contract") or {}).get("quality") or {}
        target, direction = gate.get("target"), gate.get("direction")
        observed = None
        if wid in reports:
            observed = (reports[wid].get("metrics") or {}).get(gate.get("metric_key"))
        if observed is None:
            for record in evidence.get(wid, []):
                quality = record.get("quality") or {}
                if quality.get("aggregate"):
                    observed = quality["aggregate"]["median"]
                    break
        if observed is None:
            # Contracts that ran and missed are held out of the evidence index
            # by the fail-closed rule, but they were still executed. Omitting
            # them would let the figure show a higher pass rate than the text
            # reports, which is the impression the paper works to avoid.
            recorded = (
                workload.raw.get("canonical_max_contract") or {}
            ).get("measured_evidence") or {}
            observed = recorded.get("score", recorded.get("best_score"))
        if observed is None or target in (None, 0):
            continue
        # Ratio >= 1 always means "met", whichever way the metric points.
        ratio = (
            float(target) / float(observed)
            if direction == "lower"
            else float(observed) / float(target)
        )
        rows.append((wid, ratio))

    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    ratios = [r[1] for r in rows]
    colors = [PASS_COLOR if r >= 1.0 else MISS_COLOR for r in ratios]

    fig, ax = plt.subplots(figsize=(6.0, 0.28 * len(rows) + 1.1))
    ax.barh(names, ratios, color=colors, height=0.62)
    ax.axvline(1.0, color="black", linewidth=1.1)
    ax.set_xlabel("observed / published target   (≥ 1.0 meets the contract)")
    ax.set_title("Reproduction of inherited quality targets")
    lo = min(0.9, min(ratios) - 0.03)
    ax.set_xlim(lo, max(1.05, max(ratios) + 0.02))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for name, ratio in zip(names, ratios):
        ax.text(
            ratio + 0.004,
            name,
            f"{ratio:.3f}",
            va="center",
            fontsize=6.5,
            color="#333333",
        )
    save(fig, "fig_quality_vs_target")


def fig_runtime(workloads, reports, evidence) -> None:
    rows = []
    for wid in sorted(workloads):
        seconds = None
        if wid in reports:
            metrics = reports[wid].get("metrics") or {}
            for key in ("train_and_eval_seconds", "duration_seconds"):
                if isinstance(metrics.get(key), (int, float)):
                    seconds = float(metrics[key])
                    break
        if seconds is None:
            for record in evidence.get(wid, []):
                measurement = record.get("measurement") or {}
                metric = str(measurement.get("primary_metric") or "")
                if metric.endswith("_seconds"):
                    seconds = float(measurement["aggregate"]["median"])
                    break
        if seconds:
            rows.append((wid, seconds))

    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.0, 0.28 * len(rows) + 1.1))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=NEUTRAL, height=0.62)
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock seconds (log scale)")
    ax.set_title("Measured runtime of the canonical max path")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for name, seconds in rows:
        label = f"{seconds:.1f}s" if seconds < 90 else f"{seconds / 60:.1f}m"
        ax.text(seconds * 1.1, name, label, va="center", fontsize=6.5, color="#333333")
    ax.text(
        0.98,
        0.03,
        f"suite total {sum(r[1] for r in rows) / 60:.0f} min",
        transform=ax.transAxes,
        ha="right",
        fontsize=7,
        style="italic",
    )
    save(fig, "fig_runtime")


def fig_training_curves(workloads, reports) -> None:
    """Per-epoch quality for everything that trains, against its own target."""
    curves = []
    for wid, report in sorted(reports.items()):
        metrics = report.get("metrics") or {}
        for key in (
            "hit_rates",
            "test_accuracies",
            "test_mses",
            "validation_accuracies",
            "validation_mses",
        ):
            series = metrics.get(key)
            if isinstance(series, list) and len(series) > 1:
                gate = (
                    (workloads[wid].raw.get("canonical_max_contract") or {}).get(
                        "quality"
                    )
                    or {}
                )
                curves.append((wid, key, series, gate.get("target")))
                break

    # Curves recorded in the registry rather than in a run report. The
    # recommendation study is the clearest example: it is what established that
    # the workload peaks early and then degrades, which is why its contract
    # caps the epoch budget. Omitting it would drop the most informative curve
    # the suite has.
    for wid, workload in sorted(workloads.items()):
        if any(existing[0] == wid for existing in curves):
            continue
        contract = workload.raw.get("canonical_max_contract") or {}
        recorded = contract.get("measured_evidence") or {}
        for key, label in (("hit_rate_curve", "hit rate at 10"),):
            series = recorded.get(key)
            if isinstance(series, list) and len(series) > 1:
                gate = contract.get("quality") or {}
                curves.append((wid, label, series, gate.get("target")))
                break

    if not curves:
        print("no multi-epoch curves available yet; skipping fig_training_curves")
        return

    cols = min(3, len(curves))
    rows_n = (len(curves) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows_n, cols, figsize=(2.3 * cols, 2.0 * rows_n), squeeze=False
    )
    for index, (wid, key, series, target) in enumerate(curves):
        ax = axes[index // cols][index % cols]
        ax.plot(range(1, len(series) + 1), series, marker="o", ms=2.5, color=NEUTRAL)
        if isinstance(target, (int, float)):
            ax.axhline(target, color=MISS_COLOR, linestyle="--", linewidth=1)
            ax.text(
                0.98,
                target,
                " target",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=6,
                color=MISS_COLOR,
            )
        ax.set_title(wid, fontsize=7.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key.replace("_", " ") if "_" in key else key)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for index in range(len(curves), rows_n * cols):
        axes[index // cols][index % cols].axis("off")
    title = ("Convergence of the training workloads" if len(curves) > 1
             else f"Convergence of {curves[0][0]}")
    fig.suptitle(title, fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, "fig_training_curves")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "report_dirs",
        nargs="*",
        type=Path,
        help=(
            "directories to scan for *_max_report.json; defaults to the "
            "committed paper/evidence/runs so the figures reproduce from a "
            "clean checkout"
        ),
    )
    args = ap.parse_args()

    # Every figure the paper ships must be regenerable by a reviewer who only
    # has the repository. Defaulting to committed evidence is what makes that
    # true; passing a scratch directory is the exception, not the norm.
    report_dirs = args.report_dirs or [COMMITTED_RUNS]

    style()
    workloads = load_registry(ROOT / "registry")
    reports = load_reports(report_dirs)
    evidence = load_evidence()
    print(f"registry {len(workloads)} | run reports {len(reports)} | evidence {len(evidence)}")

    fig_quality_vs_target(workloads, reports, evidence)
    fig_runtime(workloads, reports, evidence)
    fig_training_curves(workloads, reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
