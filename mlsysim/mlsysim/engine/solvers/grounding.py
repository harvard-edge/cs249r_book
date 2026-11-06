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
        """A moving target's evidence horizons against a slow proposer's cadence.

        Parameters
        ----------
        tolerance : Quantity
            Positive allowed spatial error, length.
        initial_error : Quantity
            Nonnegative initial spatial error, length.
        target_speed : Quantity
            Positive untracked target drift speed, length/time.
        slip_acceleration : Quantity
            Positive residual acceleration under tracking, length/time**2.
        intent_rate : Quantity
            Positive proposal frequency, inverse time.
        inference_latency : Quantity
            Nonnegative proposal computation time.
        stall : Quantity
            Nonnegative proposal outage duration, time.
        tick : Quantity
            Nonnegative stop-dispatch tick delay, time.
        bus : Quantity
            Nonnegative bus-cycle delay, time.
        brake_onset : Quantity
            Nonnegative actuator onset delay, time.

        Returns
        -------
        IntentCadenceResult
            Spatial margins and errors in millimeters, horizons and delays in
            milliseconds, renewal speed in millimeters per second, and
            dimensionless horizon and speed ratios.

        See Also
        --------
        calc_linear_invalidation_horizon : Composed numerical primitive.
        calc_tracked_evidence_horizon : Composed numerical primitive.

        Notes
        -----
        Equation: Untracked, the target leaves its tolerance after ``(r_tol - e_0) / v``;
        tracked, only residual slip remains, ``sqrt(2 (r_tol - e_0) / a_slip)``.
        A proposer that renews the target from its own inference computes it from
        a frame captured one inference earlier and holds it one period, so the
        oldest evidence behind a model-only target is ``period + inference``, and
        the drift it can follow is the margin over that age. A stall of ``T``
        leaves the target ``e_0 + v T`` (untracked) or ``e_0 + a_slip T^2 / 2``
        (tracked) off; a stop begins one tick, one bus cycle, and one brake onset
        after expiry.

        Assumptions: Positive tolerance margin; untracked motion uses constant target
        speed and tracked residual motion uses constant slip acceleration.

        Excludes: Tracking resets within the interval, time-varying slip, and
        physical stop-distance verification.

        References
        ----------
        Derivation: Solve margin=``v*t`` and margin=``a*t*t/2`` for evidence horizons;
        renewal age adds proposal period and inference latency. Stop
        onset adds tick, bus, and brake-onset delays.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.grounding import IntentCadenceModel
        >>> result = IntentCadenceModel().solve(tolerance=Q_('10 mm'), initial_error=Q_('1 mm'), target_speed=Q_('1 mm/s'), slip_acceleration=Q_('1 mm/s**2'), intent_rate=Q_('1 Hz'), inference_latency=Q_('10 ms'), stall=Q_('1 s'), tick=Q_('1 ms'), bus=Q_('1 ms'), brake_onset=Q_('1 ms'))
        >>> bool(result.untracked_horizon > result.stop_onset)
        True
        """
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
        """The transit lower bound: no trajectory covers ``d`` faster than ``d / v_max``.

        Parameters
        ----------
        speed_ceiling : Quantity
            Positive admitted speed limit, length/time.
        target_distance : Quantity
            Nonnegative target travel distance, length.
        tracked_horizon : Quantity
            Positive remaining evidence window with tracking, time.
        untracked_horizon : Quantity
            Positive remaining evidence window without tracking, time.

        Returns
        -------
        DynamicReachResult
            Transit floor and shortfall in milliseconds, reach distances in
            millimeters, a fits boolean, and integer fresh-capture count.

        Notes
        -----
        Equation: A terminal move is admissible only if that floor fits the evidence window;
        otherwise the reach is staged on at least ``ceil(floor / window)`` fresh
        captures. The reach inside a window at the speed ceiling is ``v_max T``.

        Assumptions: The speed ceiling can be sustained and each positive evidence
        window can be renewed when a staged reach needs another capture.

        Excludes: Acceleration and braking ramps, capture latency, obstacles, and
        proof that a staged trajectory is executable.

        References
        ----------
        Derivation: Distance divided by maximum speed bounds transit from below;
        ceiling the ratio of transit time to evidence window counts the
        minimum windows needed under this ideal staging convention.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.grounding import DynamicReachModel
        >>> result = DynamicReachModel().solve(speed_ceiling=Q_('1 m/s'), target_distance=Q_('1 m'), tracked_horizon=Q_('2 s'), untracked_horizon=Q_('1 s'))
        >>> bool(result.fits)
        True
        """
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
        """Growth term of a person's forward occupancy set, ``v_h dt``, against an aisle.

        Parameters
        ----------
        approach_speed : Quantity
            Positive constant approach speed, length/time.
        aisle_width : Quantity
            Positive aisle width, length.
        untracked_horizon : Quantity
            Positive remaining evidence window without tracking, time.
        tracked_horizon : Quantity
            Positive remaining evidence window with tracking, time.
        forecast_horizon : Quantity
            Nonnegative forecast duration, time.

        Returns
        -------
        ForwardOccupancyResult
            Growth distances in millimeters (forecast growth in meters),
            aisle-span horizon in milliseconds, and forecast-to-aisle ratio.

        Notes
        -----
        Equation: Over a screening horizon ``dt`` the set centered on the last measurement
        grows by ``v_h dt`` (before the body envelope and tracking error); past
        ``w / v_h`` one person's growth term alone spans an aisle of width ``w``.

        Assumptions: A constant positive approach speed expands the forward set
        linearly from the last measurement.

        Excludes: Body envelope, tracking error, lateral approach geometry, and
        acceleration; growth alone is not a complete occupancy set.

        References
        ----------
        Derivation: Integrate constant speed over each horizon; solve ``speed*time``
        equals aisle width for the time when growth alone spans it.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.grounding import ForwardOccupancyModel
        >>> result = ForwardOccupancyModel().solve(approach_speed=Q_('1 m/s'), aisle_width=Q_('1 m'), untracked_horizon=Q_('1 s'), tracked_horizon=Q_('2 s'), forecast_horizon=Q_('3 s'))
        >>> bool(result.growth_tracked > result.growth_untracked)
        True
        """
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
        """The intent lease's expiry: the earliest of evidence, host silence, and the stopping deadline.

        Parameters
        ----------
        untracked_horizon : Quantity
            Positive remaining evidence window without tracking, time.
        tracked_horizon : Quantity
            Positive remaining evidence window with tracking, time.
        administrative_lease : Quantity
            Nonnegative host-silence lease, time.
        stop_onset : Quantity
            Nonnegative combined delay before braking, time.
        speed_ceiling : Quantity
            Positive admitted speed limit, length/time.
        velocity_limit : Quantity
            Positive speed at which the stop-time bound was obtained, length/time.
        worst_stop_time : Quantity
            Positive worst-case stopping duration, time.
        clearance : Quantity
            Nonnegative available separation, length.

        Returns
        -------
        IntentLeaseExpiryResult
            Credible deceleration in meters per second squared, stop distance
            in millimeters, signed deadlines and expiries in milliseconds,
            and the name of each binding constraint.

        See Also
        --------
        calc_safe_stopping_distance : Composed numerical primitive.

        Notes
        -----
        Equation: The credible deceleration is the arm's worst-case stop averaged over a stop
        from its velocity limit; at the admitted speed ceiling the stop takes
        ``v t_onset + v^2 / (2 a)``, and the stopping deadline is the time the
        remaining clearance allows at that speed. With all three countdowns started
        together, the lease expires at the minimum.

        Assumptions: Countdowns share an origin; averaged worst-stop deceleration
        applies at the admitted speed. Negative deadlines remain visible.

        Excludes: Stop variability beyond the supplied bound, jerk-dependent
        transients, and certification of the resulting lease as safe.

        References
        ----------
        Derivation: Infer a=velocity_limit/worst_stop_time and compute stopping
        distance ``v*onset+v*v/(2*a)``. Divide remaining clearance by v,
        then take the earliest of that deadline and the other leases.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.grounding import IntentLeaseExpiryModel
        >>> result = IntentLeaseExpiryModel().solve(untracked_horizon=Q_('1 s'), tracked_horizon=Q_('2 s'), administrative_lease=Q_('0.5 s'), stop_onset=Q_('1 ms'), speed_ceiling=Q_('1 m/s'), velocity_limit=Q_('1 m/s'), worst_stop_time=Q_('1 s'), clearance=Q_('10 m'))
        >>> bool(result.binding_tracked == "administrative")
        True
        """
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
        """Open-vocabulary candidates normalized and screened by a split-conformal threshold.

        Parameters
        ----------
        scores : sequence of float
            Nonnegative candidate scores with positive total; not assumed calibrated probabilities.
        calibration_count : int
            Positive exchangeable calibration-set size.
        alpha : float
            Miscoverage probability strictly between zero and one.
        threshold_score : float
            Supplied calibrated nonconformity threshold in [0, 1].

        Returns
        -------
        ReferentSetResult
            Normalized scores, entropy in natural-log units, conformal rank,
            threshold, integer admitted-set size, and a refusal boolean.

        See Also
        --------
        calc_split_conformal_rank : Composed numerical primitive.

        Notes
        -----
        Equation: Raw per-box scores are normalized over the retained candidates; the set
        keeps every candidate whose normalized score is at least ``1 - qhat``,
        where ``qhat`` is the calibration score at rank ``ceil((n + 1)(1 - alpha))``.
        A set of any size other than one is refused.

        Assumptions: Nonnegative retained scores have positive total; the supplied
        threshold was calibrated for these scores under exchangeability.

        Excludes: Calibration fitting, proof of exchangeability, and interpreting
        normalized detector scores as calibrated correctness probabilities.

        References
        ----------
        Derivation: Normalize each score by their sum and count those at least
        1-threshold_score; refuse every count except one. The rank
        ``ceil((n+1)*(1-alpha))`` identifies the intended calibration order statistic.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.grounding import ReferentSetModel
        >>> result = ReferentSetModel().solve(scores=[0.9, 0.1], calibration_count=100, alpha=0.1, threshold_score=0.2)
        >>> bool(not result.refuse)
        True
        """
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
        """A Gaussian estimate's confidence region compared with a tolerance, axis by axis.

        Parameters
        ----------
        sigmas : sequence of Quantity
            Positive principal-axis standard deviations, length.
        coverage : float
            Gaussian-region probability strictly between zero and one.
        tolerance : Quantity
            Positive allowed spatial error, length.

        Returns
        -------
        ConfidenceRegionResult
            Dimensionless chi-square quantile and radius multiplier, principal
            semi-axes in millimeters, count inside tolerance, and worst
            axis-to-tolerance ratio.

        See Also
        --------
        calc_chi_square_quantile : Composed numerical primitive.

        Notes
        -----
        Equation: The region holding probability ``coverage`` has principal semi-axes
        ``r sigma_i``, where ``r^2`` is the chi-square quantile for one degree of
        freedom per axis.

        Assumptions: A jointly Gaussian estimate is expressed in principal axes with
        the supplied standard deviations and dimension equal to their count.

        Excludes: Non-Gaussian tails, uncertain covariance estimates, and obstacle
        geometry beyond the scalar per-axis tolerance comparison.

        References
        ----------
        Derivation: Whiten the Gaussian error so squared radius follows chi-square
        with one degree of freedom per axis; multiply each standard
        deviation by the square root of its coverage quantile.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.grounding import ConfidenceRegionModel
        >>> result = ConfidenceRegionModel().solve(sigmas=[Q_('1 mm'), Q_('1 mm')], coverage=0.95, tolerance=Q_('10 mm'))
        >>> bool(result.axes_within == 2)
        True
        """
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
