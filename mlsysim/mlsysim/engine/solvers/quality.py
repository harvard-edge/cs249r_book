"""Prediction-quality solvers: measured accuracy loss under shift, and proxy drift.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic
stays reviewable. Added 2026-10-20 for the silent-degradation examples
(ImageNetV2, Google Flu Trends).
"""

from __future__ import annotations

import numpy as np

from ..results import ProxyDriftResult, ShiftedTestSetResult
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

    def solve(self, rows, original_key: str, new_key: str, gap_key: str,
              baseline_prefix: str) -> ShiftedTestSetResult:
        orig = np.array([r[original_key] for r in rows], dtype=float)
        new = np.array([r[new_key] for r in rows], dtype=float)
        gaps = [r[gap_key] for r in rows]
        nets = [r[gap_key] for r in rows if not str(r["model"]).startswith(baseline_prefix)]
        slope, intercept = np.polyfit(orig, new, 1)
        return ShiftedTestSetResult(
            model_count=len(rows), baseline_count=len(rows) - len(nets),
            gap_min=min(gaps), gap_max=max(gaps),
            network_gap_min=min(nets), network_gap_max=max(nets),
            mean_gap=sum(gaps) / len(gaps),
            slope=float(slope), intercept=float(intercept),
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


class ProxyDriftModel(ForwardModel):
    """
    A proxy estimate against the reference it stands in for, season by season.

    The season ratio is the summed weekly proxy over the summed weekly
    reference. Before drift it stays near 1; the largest deviation over the
    seasons before ``first_drift_year`` is the stable band. In the peak season
    the model reports both peaks, their ratio, and the largest single week.
    """
    requires = ("rows",)
    produces = ProxyDriftResult

    def solve(self, rows, proxy_key: str, reference_key: str, first_year: int, last_year: int,
              start_week: int, end_week: int, first_drift_year: int, peak_year: int) -> ProxyDriftResult:
        both = [r for r in rows if r[proxy_key] is not None and r[reference_key] is not None]
        ratios = {}
        for y in range(first_year, last_year + 1):
            ws = season_weeks(both, y, start_week, end_week)
            ratios[season_label(y)] = sum(r[proxy_key] for r in ws) / sum(r[reference_key] for r in ws)
        stable = [ratios[season_label(y)] for y in range(first_year, first_drift_year)]
        peak = season_weeks(both, peak_year, start_week, end_week)
        p_proxy = max(r[proxy_key] for r in peak)
        p_ref = max(r[reference_key] for r in peak)
        return ProxyDriftResult(
            season_ratios=ratios,
            stable_max_deviation=max(abs(x - 1) for x in stable),
            peak_season=season_label(peak_year),
            peak_proxy=p_proxy, peak_reference=p_ref, peak_ratio=p_proxy / p_ref,
            max_week_ratio=max(r[proxy_key] / r[reference_key] for r in peak),
        )
