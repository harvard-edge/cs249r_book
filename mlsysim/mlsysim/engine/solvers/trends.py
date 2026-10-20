"""Trend solvers: growth factors fitted to published run records.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

from ..results import DomainComputeGapResult, ScalingSeriesResult, TrainingGrowthResult
from .base import ForwardModel


def _decimal_year(s):
    d = dt.date.fromisoformat(s)
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


def _annual_factor(rows, value):
    """10**slope of a least-squares fit of log10(value(row)) on publication date."""
    return 10 ** np.polyfit([_decimal_year(r["date"]) for r in rows], np.log10([value(r) for r in rows]), 1)[0]


class TrainingGrowthModel(ForwardModel):
    """
    Splits training-compute growth into accelerator count, duration, and rate.

    Training compute is count x hours x delivered rate per accelerator, so the
    annual growth factors of the three multiply to the compute trend. Fits are
    least squares of log10 value on publication date over runs that report all
    three. The fleet's share is the part of the exponent from count and duration.
    """
    requires = ("series",)
    produces = TrainingGrowthResult

    def solve(self, rows, first_year: str, last_year: str, origin_model: str = "AlexNet") -> TrainingGrowthResult:
        window = [r for r in rows if r["training_compute_flop"] and first_year <= r["date"][:4] <= last_year]
        runs = [r for r in window if r["hardware_quantity"] and r["training_time_hours"]]
        compute = _annual_factor(runs, lambda r: r["training_compute_flop"])
        count = _annual_factor(runs, lambda r: r["hardware_quantity"])
        duration = _annual_factor(runs, lambda r: r["training_time_hours"])
        rate = _annual_factor(runs, lambda r: r["training_compute_flop"]
                              / (r["hardware_quantity"] * r["training_time_hours"] * 3600))
        origin = next(r for r in window if r["model"] == origin_model)["training_compute_flop"]
        return TrainingGrowthResult(
            run_count=len(runs), compute_factor=compute, count_factor=count,
            duration_factor=duration, rate_factor=rate,
            fleet_share=(math.log(count) + math.log(duration)) / math.log(compute),
            span_orders=math.log10(max(r["training_compute_flop"] for r in window) / origin),
        )


# --- Training compute by domain (added 2026-10-20) --------------------------

class DomainComputeGapModel(ForwardModel):
    """
    How far one model's training compute sits below the largest model of another domain that year.

    Rows are the Epoch AI models-by-domain series (``domain_group``,
    ``publication_date``, ``training_flop``). The gap is the largest
    ``reference_group`` compute published in the model's year over the
    model's own compute.
    """
    requires = ("series",)
    produces = DomainComputeGapResult

    def solve(self, rows, model: str, reference_group: str = "Language") -> DomainComputeGapResult:
        row = next(r for r in rows if r["model"] == model)
        year = str(row["publication_date"])[:4]
        ref = max((r for r in rows if r["domain_group"] == reference_group and r["training_flop"]
                   and str(r["publication_date"]).startswith(year)),
                  key=lambda r: r["training_flop"])
        return DomainComputeGapResult(
            model=model, year=year, model_flop=row["training_flop"],
            reference_model=ref["model"], reference_flop=ref["training_flop"],
            gap=ref["training_flop"] / row["training_flop"],
            domain_models_with_compute=sum(1 for r in rows if r["domain_group"] == row["domain_group"]
                                           and r["training_flop"]),
        )


class ScalingSeriesModel(ForwardModel):
    """
    Relative scaling efficiency of benchmark series measured at several sizes.

    A series is one submitter's same-round runs of one benchmark on one
    accelerator. Its efficiency is ``(T0 N0) / (T N)`` between the smallest
    configuration ``(N0, T0)`` and the largest ``(N, T)``: 1 when time falls in
    proportion to accelerators, below 1 when the larger run wastes some of them.
    Added 2026-10-20.
    """
    requires = ("series",)
    produces = ScalingSeriesResult

    def solve(self, rows, specs) -> ScalingSeriesResult:
        labels, series, effs = [], [], []
        for label, (rnd, submitter, benchmark, accelerator) in specs:
            pts = sorted((r for r in rows if r["round"] == rnd and r["submitter"].startswith(submitter)
                          and r["benchmark"] == benchmark and accelerator in r["accelerator"]),
                         key=lambda r: r["accelerators"])
            a, z = pts[0], pts[-1]
            labels.append(label)
            series.append(list(pts))
            effs.append((a["score_minutes"] * a["accelerators"]) / (z["score_minutes"] * z["accelerators"]))
        return ScalingSeriesResult(labels=labels, series=series, efficiencies=effs)


class MemoryWallModel(ForwardModel):
    """
    Training-state demand against single-accelerator memory, and where they cross.

    Demand is each model's parameter count times the bytes of training state
    per parameter; capacity is each accelerator's memory at its release year
    (vendors label 80 GiB as 80 GB, so capacity is read in GiB). The crossover is
    the first model whose demand exceeds the largest accelerator available in
    its publication year. Added 2026-10-20.
    """
    requires = ("models", "accelerators")

    def solve(self, models, accelerators, bytes_per_param):
        from ..results import MemoryWallResult
        from ...hardware.registry import Hardware
        from ...literature.registry import Literature
        rows = []
        for label, name in models:
            r = next(m for m in Literature.Series.NotableModels.entries if m["model"] == name)
            rows.append({"Year": int(r["date"][:4]), "Date": r["date"], "Name": label, "Type": "Model",
                         "Memory_GB": (r["parameters"] * bytes_per_param).to("GB").magnitude})
        for key in accelerators:
            h = getattr(Hardware.Cloud, key)
            rows.append({"Year": int(h.release_year), "Date": None, "Name": key, "Type": "GPU",
                         "Memory_GB": h.memory.capacity.to("GiB").magnitude})
        gpus = sorted((r for r in rows if r["Type"] == "GPU"), key=lambda r: r["Year"])
        demand = sorted((r for r in rows if r["Type"] == "Model"), key=lambda r: r["Year"])
        cross = next(r for r in demand
                     if r["Memory_GB"] > max((g["Memory_GB"] for g in gpus if g["Year"] <= r["Year"]),
                                         default=float("inf")))  # no accelerator yet: no crossover
        largest = max(g["Memory_GB"] for g in gpus if g["Year"] <= cross["Year"])
        return MemoryWallResult(
            rows=rows,
            capacity_growth=max(g["Memory_GB"] for g in gpus) / min(g["Memory_GB"] for g in gpus),
            demand_orders=math.log10(max(r["Memory_GB"] for r in demand) / min(r["Memory_GB"] for r in demand)),
            crossover_name=cross["Name"], crossover_year=cross["Year"],
            crossover_gap=cross["Memory_GB"] / largest,
        )
