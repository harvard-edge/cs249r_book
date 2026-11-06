"""Typed result models for statistics calculations."""

from __future__ import annotations

from typing import Dict, Tuple
from ...core.types import Quantity
from .base import SolverResult

__all__ = [
    "BinomialIntervalsResult",
    "ProportionPlanningResult",
    "PairedComparisonResult",
    "TaskMixConsistencyResult",
    "CappedSelectionCostResult",
]


class BinomialIntervalsResult(SolverResult):
    """Result from BinomialIntervalsModel: Wald and Wilson intervals for several counts at one size."""

    __module__ = "mlsysim.engine.results"

    z_critical: float
    wald: Dict[int, Tuple[float, float]]  # successes -> (low, high), unclipped
    wilson: Dict[int, Tuple[float, float]]  # successes -> (low, high)


class ProportionPlanningResult(SolverResult):
    """Result from ProportionPlanningModel: one-rate precision and two-arm detection counts."""

    __module__ = "mlsysim.engine.results"

    z_critical: float
    half_width: float  # normal approximation at the planning size
    wilson_half_width: float  # half the Wilson width at the same size and rate
    samples_for_target: int  # draws for the target half-width
    per_arm: Dict[float, float]  # detectable difference -> tasks per arm (unrounded)


class PairedComparisonResult(SolverResult):
    """Result from PairedComparisonModel: a candidate against a baseline on the same tasks."""

    __module__ = "mlsysim.engine.results"

    pairs: int
    candidate_rate: float
    baseline_rate: float
    difference: float  # candidate minus baseline
    lower: float  # Newcombe method-10 interval
    upper: float
    noninferior: bool  # lower > -tolerance
    superior: bool  # lower > 0
    exact_p: float  # McNemar, two-sided
    lower_width: float  # difference - lower
    clear_width: float  # difference + tolerance, the half-width the gate needs
    pairs_if_gain_holds: int  # pairs that center the expected lower limit on the margin
    pairs_if_no_gain: int  # pairs that fit the lower half-width inside the tolerance
    discordant_fraction: float
    paired_variance: float  # per pair, d - delta^2
    independent_variance: float  # per task, two independent arms
    paired_share: float  # paired / independent
    paired_half_width: float  # normal approximation
    independent_multiple: float  # independent / paired


class TaskMixConsistencyResult(SolverResult):
    """Result from TaskMixConsistencyModel: repeating one task against drawing new tasks."""

    __module__ = "mlsysim.engine.results"

    mean_rate: float
    same_task_all: float  # E[p_i^k]
    new_tasks_all: float  # (E[p_i])^k


class CappedSelectionCostResult(SolverResult):
    """Result from CappedSelectionCostModel: batch and sequential selection under one candidate cap."""

    __module__ = "mlsysim.engine.results"

    availability: float  # 1 - (1 - p)^k, either policy
    expected_candidates: float  # sequential, availability / p
    batch_trial_cost: Quantity
    sequential_trial_cost: Quantity
    batch_per_accepted: Quantity
    sequential_per_accepted: Quantity
    batch_premium: float  # batch / sequential per accepted - 1
    doubled_rate_per_accepted: Quantity  # sequential with the pass probability doubled
