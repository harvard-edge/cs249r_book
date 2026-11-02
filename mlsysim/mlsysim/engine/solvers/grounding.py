"""Grounded-intent solvers: how long a learned target stays true, and the lease that bounds it.

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). The equations live in
``physics.robotics`` (the evidence horizons, the stopping distance) and
``physics.statistics`` (the conformal rank, the chi-square quantile); these
models compose them for one stated target, arm, and aisle, so a chapter, a
slide deck, and a lab read the same arithmetic (added 2026-10-20).
"""

from __future__ import annotations

import math
from typing import Sequence

from ..results import (
    ConfidenceRegionResult,
    DynamicReachResult,
    ForwardOccupancyResult,
    IntentCadenceResult,
    IntentLeaseExpiryResult,
    ReferentSetResult,
)
from ...core.types import Quantity
from ...core.units import ureg
from ...physics.robotics import calc_linear_invalidation_horizon, calc_tracked_evidence_horizon
from ...physics.robotics.stopping import calc_safe_stopping_distance
from ...physics.statistics import calc_chi_square_quantile, calc_split_conformal_rank
from .base import ForwardModel

_MM, _MS = ureg.millimeter, ureg.millisecond


def _ratio(a, b) -> float:
    return float((a / b).to("").magnitude)


class IntentCadenceModel(ForwardModel):
    """
    A moving target's evidence horizons against a slow proposer's cadence.

    Untracked, the target leaves its tolerance after ``(r_tol - e_0) / v``;
    tracked, only residual slip remains, ``sqrt(2 (r_tol - e_0) / a_slip)``.
    A proposer that renews the target from its own inference computes it from
    a frame captured one inference earlier and holds it one period, so the
    oldest evidence behind a model-only target is ``period + inference``, and
    the drift it can follow is the margin over that age. A stall of ``T``
    leaves the target ``e_0 + v T`` (untracked) or ``e_0 + a_slip T^2 / 2``
    (tracked) off; a stop begins one tick, one bus cycle, and one brake onset
    after expiry.
    """
    requires = ("target", "cadence", "stop_terms")
    produces = IntentCadenceResult

    def solve(self, tolerance: Quantity, initial_error: Quantity, target_speed: Quantity,
              slip_acceleration: Quantity, intent_rate: Quantity, inference_latency: Quantity,
              stall: Quantity, tick: Quantity, bus: Quantity, brake_onset: Quantity) -> IntentCadenceResult:
        margin = (tolerance - initial_error).to(_MM)
        untracked = calc_linear_invalidation_horizon(initial_error, target_speed, tolerance).to(_MS)
        tracked = calc_tracked_evidence_horizon(tolerance, initial_error, slip_acceleration).to(_MS)
        period = (1 / intent_rate).to(_MS)
        err_period = (initial_error + target_speed * period).to(_MM)
        age = (period + inference_latency).to(_MS)
        ceiling = (margin / age).to(_MM / ureg.second)
        onset = (tick + bus + brake_onset).to(_MS)
        return IntentCadenceResult(
            margin=margin,
            untracked_horizon=untracked,
            tracked_horizon=tracked,
            intent_period=period,
            error_at_period=err_period,
            excess_at_period=(err_period - tolerance).to(_MM),
            inference_overrun=(inference_latency - untracked).to(_MS),
            inference_over_horizon=_ratio(inference_latency, untracked),
            renewal_age=age,
            renewal_speed_ceiling=ceiling,
            renewal_share=_ratio(ceiling, target_speed),
            stall=stall.to(_MS),
            stall_error_untracked=(initial_error + target_speed * stall).to(_MM),
            stall_error_tracked=(initial_error + 0.5 * slip_acceleration * stall ** 2).to(_MM),
            stop_onset=onset,
            untracked_expiry_to_onset=(untracked + onset).to(_MS),
            tracked_expiry_to_onset=(tracked + onset).to(_MS),
        )


class DynamicReachModel(ForwardModel):
    """
    The transit lower bound: no trajectory covers ``d`` faster than ``d / v_max``.

    A terminal move is admissible only if that floor fits the evidence window;
    otherwise the reach is staged on at least ``ceil(floor / window)`` fresh
    captures. The reach inside a window at the speed ceiling is ``v_max T``.
    """
    requires = ("speed_ceiling", "distance", "windows")
    produces = DynamicReachResult

    def solve(self, speed_ceiling: Quantity, target_distance: Quantity, tracked_horizon: Quantity,
              untracked_horizon: Quantity) -> DynamicReachResult:
        floor = (target_distance / speed_ceiling).to(_MS)
        return DynamicReachResult(
            transit_floor=floor,
            reach_tracked=(speed_ceiling * tracked_horizon).to(_MM),
            reach_untracked=(speed_ceiling * untracked_horizon).to(_MM),
            shortfall=(floor - tracked_horizon).to(_MS),
            fits=floor <= tracked_horizon,
            fresh_captures=math.ceil(_ratio(floor, tracked_horizon)),
        )


class ForwardOccupancyModel(ForwardModel):
    """
    Growth term of a person's forward occupancy set, ``v_h dt``, against an aisle.

    Over a screening horizon ``dt`` the set centered on the last measurement
    grows by ``v_h dt`` (before the body envelope and tracking error); past
    ``w / v_h`` one person's growth term alone spans an aisle of width ``w``.
    """
    requires = ("approach_speed", "aisle_width", "horizons")
    produces = ForwardOccupancyResult

    def solve(self, approach_speed: Quantity, aisle_width: Quantity, untracked_horizon: Quantity,
              tracked_horizon: Quantity, forecast_horizon: Quantity) -> ForwardOccupancyResult:
        forecast = (approach_speed * forecast_horizon).to(ureg.meter)
        return ForwardOccupancyResult(
            growth_untracked=(approach_speed * untracked_horizon).to(_MM),
            growth_tracked=(approach_speed * tracked_horizon).to(_MM),
            growth_forecast=forecast,
            forecast_over_aisle=_ratio(forecast, aisle_width),
            aisle_span_horizon=(aisle_width / approach_speed).to(_MS),
        )


class IntentLeaseExpiryModel(ForwardModel):
    """
    The intent lease's expiry: the earliest of evidence, host silence, and the stopping deadline.

    The credible deceleration is the arm's worst-case stop averaged over a stop
    from its velocity limit; at the admitted speed ceiling the stop takes
    ``v t_onset + v^2 / (2 a)``, and the stopping deadline is the time the
    remaining clearance allows at that speed. With all three countdowns started
    together, the lease expires at the minimum.
    """
    requires = ("evidence", "administrative_lease", "stop")
    produces = IntentLeaseExpiryResult

    def solve(self, untracked_horizon: Quantity, tracked_horizon: Quantity, administrative_lease: Quantity,
              stop_onset: Quantity, speed_ceiling: Quantity, velocity_limit: Quantity,
              worst_stop_time: Quantity, clearance: Quantity) -> IntentLeaseExpiryResult:
        a = (velocity_limit / worst_stop_time).to(ureg.meter / ureg.second ** 2)
        d_stop = calc_safe_stopping_distance(speed_ceiling, stop_onset, a).to(_MM)
        deadline = ((clearance - d_stop) / speed_ceiling).to(_MS)

        def bound(evidence):
            terms = (("evidence", evidence.to(_MS)), ("administrative", administrative_lease.to(_MS)),
                     ("stopping", deadline))
            return min(terms, key=lambda t: t[1].magnitude)

        bu, bt = bound(untracked_horizon), bound(tracked_horizon)
        return IntentLeaseExpiryResult(
            credible_deceleration=a,
            stop_distance=d_stop,
            stopping_deadline=deadline,
            expiry_untracked=bu[1],
            expiry_tracked=bt[1],
            binding_untracked=bu[0],
            binding_tracked=bt[0],
        )


class ReferentSetModel(ForwardModel):
    """
    Open-vocabulary candidates normalized and screened by a split-conformal threshold.

    Raw per-box scores are normalized over the retained candidates; the set
    keeps every candidate whose normalized score is at least ``1 - qhat``,
    where ``qhat`` is the calibration score at rank ``ceil((n + 1)(1 - alpha))``.
    A set of any size other than one is refused.
    """
    requires = ("scores", "calibration")
    produces = ReferentSetResult

    def solve(self, scores: Sequence[float], calibration_count: int, alpha: float,
              threshold_score: float) -> ReferentSetResult:
        total = sum(scores)
        p = [s / total for s in scores]
        threshold = 1 - threshold_score
        size = sum(x >= threshold for x in p)
        return ReferentSetResult(
            normalized=p,
            entropy=0.0 - sum(x * math.log(x) for x in p if x > 0),
            max_entropy=math.log(len(p)),
            rank=calc_split_conformal_rank(calibration_count, alpha),
            threshold=threshold,
            set_size=size,
            refuse=size != 1,
        )


class ConfidenceRegionModel(ForwardModel):
    """
    A Gaussian estimate's confidence region compared with a tolerance, axis by axis.

    The region holding probability ``coverage`` has principal semi-axes
    ``r sigma_i``, where ``r^2`` is the chi-square quantile for one degree of
    freedom per axis.
    """
    requires = ("sigmas", "coverage", "tolerance")
    produces = ConfidenceRegionResult

    def solve(self, sigmas: Sequence[Quantity], coverage: float, tolerance: Quantity) -> ConfidenceRegionResult:
        q = calc_chi_square_quantile(len(sigmas), coverage)
        r = math.sqrt(q)
        axes = [(r * s).to(_MM) for s in sigmas]
        return ConfidenceRegionResult(
            quantile=q,
            radius=r,
            semi_axes=axes,
            axes_within=sum(ax <= tolerance for ax in axes),
            worst_over_tolerance=max(_ratio(ax, tolerance) for ax in axes),
        )
