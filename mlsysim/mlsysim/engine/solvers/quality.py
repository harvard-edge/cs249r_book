"""Prediction-quality solvers: measured accuracy loss under shift, and proxy drift.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic
stays reviewable. Added 2026-10-20 for the silent-degradation examples
(ImageNetV2, Google Flu Trends).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..results import (
    ShiftedTestSetResult,
)
from ...physics.statistics import calc_spearman_rank_correlation
from .base import ForwardModel


class ShiftedTestSetModel(ForwardModel):
    """
    Accuracy of the same models on an original test set and on a re-collected one.

    Each row is one model: its accuracy on the original set, on the new set,
    and the published gap (percentage points). Models whose name starts with
    ``baseline_prefix`` are the non-neural baselines, reported separately. The
    fit is ordinary least squares of new accuracy on original accuracy; a slope
    near 1 means a model that started higher ended higher by the same amount.
    """
    requires = ("rows",)
    produces = ShiftedTestSetResult

    def solve(self, rows, original_key: str, new_key: str, gap_key: Optional[str],
              baseline_prefix: Optional[str]) -> ShiftedTestSetResult:
        orig = np.array([r[original_key] for r in rows], dtype=float)
        new = np.array([r[new_key] for r in rows], dtype=float)
        # A series without a published gap column: gap = original - new (added 2026-10-20).
        gap = (lambda r: r[gap_key]) if gap_key else (lambda r: r[original_key] - r[new_key])
        gaps = [gap(r) for r in rows]
        networks = [r for r in rows
                    if not (baseline_prefix and str(r["model"]).startswith(baseline_prefix))]
        nets = [gap(r) for r in networks]
        slope, intercept = np.polyfit(orig, new, 1)
        return ShiftedTestSetResult(
            model_count=len(rows), baseline_count=len(rows) - len(nets),
            gap_min=min(gaps), gap_max=max(gaps),
            network_gap_min=min(nets), network_gap_max=max(nets),
            mean_gap=sum(gaps) / len(gaps),
            gap_median=float(np.median(gaps)),
            slope=float(slope), intercept=float(intercept),
            network_rank_correlation=calc_spearman_rank_correlation(
                [r[original_key] for r in networks], [r[new_key] for r in networks]),
        )


def season_weeks(rows, start_year: int, start_week: int, end_week: int):
    """Rows of the season that opens in week ``start_week`` of ``start_year``
    and closes in week ``end_week`` of the next year (epiweek = year*100 + week)."""
    def in_season(ew):
        y, w = divmod(int(ew), 100)
        return (y == start_year and w >= start_week) or (y == start_year + 1 and w <= end_week)
    return [r for r in rows if in_season(r["epiweek"])]


def season_label(start_year: int) -> str:
    """'12-13' for the season that opens in 2012."""
    return f"{start_year % 100:02d}-{(start_year + 1) % 100:02d}"


