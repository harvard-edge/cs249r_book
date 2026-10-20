"""Trend solvers: growth factors fitted to published run records.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

from ..results import TrainingGrowthResult
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
