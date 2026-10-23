"""Spatial-memory solvers: state estimation, estimator consistency, drift, and belief expiry (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers``. The equations live in
``physics/robotics/estimation.py``, ``physics/robotics/perception.py``,
``physics/robotics/sensing.py``, and ``physics/statistics.py``; these models
compose them for one stated estimator, sensor, benchmark, or belief, so a
chapter, a slide deck, and a lab read the same arithmetic.
"""

from __future__ import annotations

import math
import statistics
from typing import Optional, Sequence

from ..results import (
    BeliefExpiryResult,
    EpochGapResult,
    EstimatorConsistencyResult,
    FreeSpaceExpiryResult,
    FrozenTransformResult,
    InertialCoastResult,
    OdometryDriftResult,
    OdometryLeaderboardResult,
    RangeFusionResult,
    ReplayBufferResult,
    SensorBiasDriftResult,
    StaleTransformResult,
)
from ...core.types import Quantity
from ...core.units import ureg
from ...physics.robotics.estimation import (
    calc_belief_error_bound,
    calc_inertial_coast_errors,
    calc_quadratic_invalidation_horizon,
    calc_scalar_kalman_update,
)
from ...physics.robotics.perception import calc_free_space_expiry
from ...physics.robotics.sensing import calc_lever_arm_error
from ...physics.statistics import calc_chi2_gate, calc_chi2_mean_band, calc_noncentral_gate_detection
from .base import ForwardModel

_MM = ureg.millimeter
_MS = ureg.millisecond


def _ratio(a, b) -> float:
    return float((a / b).to("").magnitude)


class RangeFusionModel(ForwardModel):
    """
    One scalar predict and two updates of a range, and the same second update misdated.

    Prediction adds ``(sigma_v dt)^2``; each update applies
    ``calc_scalar_kalman_update``. A reading paired with the wrong pose is
    off by ``misdating_bias``; the update moves the estimate ``g b`` while the
    reported spread, which uses only the modeled noise, is unchanged. Its
    innovation's NIS has mean ``1 + b^2 / S``. ``comparison_scale`` reruns the
    second update with that sensor's spread scaled.
    """
    requires = ("prior_sigma", "speed_sigma", "interval", "first_sigma", "second_sigma")
    produces = RangeFusionResult

    def solve(self, prior_sigma: Quantity, speed_sigma: Quantity, interval: Quantity,
              first_sigma: Quantity, second_sigma: Quantity, misdating_bias: Quantity,
              comparison_scale: float = 2.0) -> RangeFusionResult:
        mm2 = _MM ** 2
        process = (speed_sigma * interval).to(_MM)
        p_pred = (prior_sigma.to(_MM) ** 2 + process ** 2).to(mm2)
        g1, p1 = calc_scalar_kalman_update(p_pred, first_sigma.to(_MM) ** 2)
        r2 = second_sigma.to(_MM) ** 2
        s2 = (p1 + r2).to(mm2)
        g2, p2 = calc_scalar_kalman_update(p1, r2)
        fused = p2 ** 0.5
        shift = (g2 * misdating_bias).to(_MM)
        lam = _ratio(misdating_bias.to(_MM) ** 2, s2)
        gc, pc = calc_scalar_kalman_update(p1, (comparison_scale * second_sigma.to(_MM)) ** 2)
        return RangeFusionResult(
            process_sigma=process, predicted_sigma=(p_pred ** 0.5).to(_MM), first_gain=g1,
            first_sigma=(p1 ** 0.5).to(_MM), second_sensor_sigma=second_sigma.to(_MM), second_gain=g2,
            fused_sigma=fused.to(_MM), innovation_variance=s2, misdating_bias=misdating_bias.to(_MM),
            misdated_shift=shift, misdated_shift_sigmas=_ratio(shift, fused),
            nis_expected=1 + lam, noncentrality=lam, comparison_gain=gc, comparison_sigma=(pc ** 0.5).to(_MM),
        )


class EstimatorConsistencyModel(ForwardModel):
    """
    The band a campaign's mean NEES or NIS must fall in, and what single-update gates catch.

    ``calc_chi2_mean_band`` gives the band for ``runs`` statistics of
    dimension ``dimension``; each gate is ``calc_chi2_gate`` at its confidence,
    and a biased update with mean NIS ``nis_expected`` exceeds it with
    ``calc_noncentral_gate_detection`` (noncentrality ``nis_expected - dimension``).
    """
    requires = ("runs", "dimension", "confidence", "nis_expected")
    produces = EstimatorConsistencyResult

    def solve(self, runs: int, dimension: int, confidence: float, gate_confidences: Sequence[float],
              nis_expected: float) -> EstimatorConsistencyResult:
        lo, hi = calc_chi2_mean_band(runs, dimension, confidence)
        gates = [calc_chi2_gate(c, dimension) for c in gate_confidences]
        lam = nis_expected - dimension
        return EstimatorConsistencyResult(
            band_low=lo, band_high=hi, gates=gates,
            detection=[calc_noncentral_gate_detection(g, dimension, lam) for g in gates],
            nis_expected=nis_expected,
        )


class ReplayBufferModel(ForwardModel):
    """Slots of a rewind buffer sampled at ``sample_rate`` that reach back ``horizon``: ``ceil(horizon / dt)``."""
    requires = ("horizon", "sample_rate")
    produces = ReplayBufferResult

    def solve(self, horizon: Quantity, sample_rate: Quantity) -> ReplayBufferResult:
        period = (1 / sample_rate).to(_MS)
        return ReplayBufferResult(sample_period=period, slots=math.ceil(_ratio(horizon, period)))


class InertialCoastModel(ForwardModel):
    """
    An IMU integrated open loop for ``duration``, with the heading offset seen at ``target_range``.

    ``calc_inertial_coast_errors`` gives the four terms; the heading offset
    moves a point at range sideways by ``calc_lever_arm_error``. The position
    offset is priced against ``reference`` (a localization bound).
    """
    requires = ("duration", "target_range", "reference")
    produces = InertialCoastResult

    def solve(self, gyro_noise_density: Quantity, gyro_bias: Quantity, accel_noise_density: Quantity,
              accel_bias: Quantity, duration: Quantity, target_range: Quantity,
              reference: Quantity) -> InertialCoastResult:
        e = calc_inertial_coast_errors(gyro_noise_density, gyro_bias, accel_noise_density, accel_bias, duration)
        return InertialCoastResult(
            heading_noise=e["heading_noise"], heading_bias=e["heading_bias"],
            lateral_bias=calc_lever_arm_error(target_range, e["heading_bias"]),
            position_noise=e["position_noise"], position_bias=e["position_bias"],
            bias_over_reference=_ratio(e["position_bias"], reference),
            bias_over_noise=_ratio(e["position_bias"], e["position_noise"]),
        )


class OdometryDriftModel(ForwardModel):
    """
    A benchmark's return error over its path as a drift rate, before and after calibration.

    The rate ``eps = E / L`` spends a bound ``delta`` after ``delta / eps`` of
    travel, which takes ``delta / (eps v)`` at the operating speed; over a
    segment of length ``s`` the error is ``eps s``.
    """
    requires = ("return_error_uncalibrated", "return_error_calibrated", "path_length", "bound")
    produces = OdometryDriftResult

    def solve(self, return_error_uncalibrated: Quantity, return_error_calibrated: Quantity,
              path_length: Quantity, bound: Quantity, speed: Quantity, segment: Quantity,
              test_speed: Quantity) -> OdometryDriftResult:
        eu = _ratio(return_error_uncalibrated, path_length)
        ec = _ratio(return_error_calibrated, path_length)
        du, dc = (bound / eu).to(ureg.meter), (bound / ec).to(ureg.meter)
        su, sc = (eu * segment).to(_MM), (ec * segment).to(_MM)
        return OdometryDriftResult(
            path_length=path_length.to(ureg.meter), rate_uncalibrated=eu, rate_calibrated=ec,
            distance_uncalibrated=du, distance_calibrated=dc,
            time_uncalibrated=(du / speed).to(ureg.second), time_calibrated=(dc / speed).to(ureg.second),
            segment_error_uncalibrated=su, segment_error_calibrated=sc,
            segment_over_bound_uncalibrated=_ratio(su, bound),
            calibration_gain=_ratio(return_error_uncalibrated, return_error_calibrated),
            speed_ratio=_ratio(speed, test_speed),
        )


class OdometryLeaderboardModel(ForwardModel):
    """
    Ranked odometry methods' translation error (percent of distance) against a bound.

    A rate ``eps`` spends ``bound`` after ``bound / eps``; the rate that spends
    it in exactly one ``segment`` is ``bound / segment``.
    """
    requires = ("entries", "bound", "segment")
    produces = OdometryLeaderboardResult

    def solve(self, entries: Sequence[dict], bound: Quantity, segment: Quantity) -> OdometryLeaderboardResult:
        err = [r["translation_err_pct"] for r in entries]
        st = [r["translation_err_pct"] for r in entries if r["sensor_group"] == "stereo"]
        la = [r["translation_err_pct"] for r in entries if r["sensor_group"] == "laser"]
        best = min(min(st), min(la))
        st_med, la_med = statistics.median(st), statistics.median(la)

        def dist(pct):
            return (bound / (pct / 100)).to(ureg.meter)

        thr = 100 * _ratio(bound, segment)
        return OdometryLeaderboardResult(
            count=len(entries), stereo_count=len(st), laser_count=len(la), best_pct=best,
            stereo_median_pct=st_med, laser_median_pct=la_med, distance_best=dist(best),
            distance_stereo_median=dist(st_med), distance_laser_median=dist(la_med),
            segment_threshold_pct=thr, count_below_threshold=sum(1 for e in err if e < thr),
        )


class BeliefExpiryModel(ForwardModel):
    """
    A belief's error bound ``E(dt) = E_0 + v dt + a dt^2 / 2``, its invalidation horizon, and an admission check.

    The horizon is ``calc_quadratic_invalidation_horizon``; dropping the
    acceleration term gives the constant-velocity horizon, at which the true
    bound has already overrun the tolerance. With ``use_age``, the consumer's
    age bound is ``use_age + age_bound`` (the clock-conversion bound), and the
    belief is admitted if the error there is within tolerance; the latest
    admissible use age is the horizon less ``age_bound``.
    """
    requires = ("initial_error", "growth_rate", "tolerance")
    produces = BeliefExpiryResult

    def solve(self, initial_error: Quantity, growth_rate: Quantity, tolerance: Quantity,
              growth_acceleration: Optional[Quantity] = None, use_age: Optional[Quantity] = None,
              age_bound: Optional[Quantity] = None) -> BeliefExpiryResult:
        a = growth_acceleration if growth_acceleration is not None else 0 * _MM / ureg.second ** 2
        horizon = calc_quadratic_invalidation_horizon(initial_error, growth_rate, a, tolerance)
        t_lin = calc_quadratic_invalidation_horizon(initial_error, growth_rate, 0 * a, tolerance)
        e_lin = calc_belief_error_bound(initial_error, growth_rate, t_lin, a)
        extra = {}
        if use_age is not None:
            bound = age_bound if age_bound is not None else 0 * _MS
            upper = (use_age + bound).to(_MS)
            e_use = calc_belief_error_bound(initial_error, growth_rate, upper, a)
            extra = dict(use_age_upper=upper, error_at_use=e_use, margin_at_use=(tolerance - e_use).to(_MM),
                         time_left=(horizon - upper).to(_MS), latest_use_age=(horizon - bound).to(_MS))
        return BeliefExpiryResult(
            initial_error=initial_error.to(_MM), growth_rate=growth_rate.to(_MM / ureg.second),
            growth_acceleration=a.to(_MM / ureg.second ** 2), tolerance=tolerance.to(_MM),
            horizon=horizon, constant_velocity_horizon=t_lin, error_at_constant_velocity_horizon=e_lin,
            overshoot=_ratio(e_lin, tolerance) - 1, overrun=(t_lin - horizon).to(_MS), **extra,
        )


class StaleTransformModel(ForwardModel):
    """
    A cached sensor-to-target edge, one ``interval`` old, composed with the current wrist pose.

    The finger error is the initial error, plus the unobserved drift over the
    interval ``v dt``, plus the lever-arm swing ``l omega dt`` of the rotation
    the cached edge never saw. Composing the edge with the wrist pose of its
    own epoch removes the lever term and leaves the belief's own bound.
    """
    requires = ("belief", "lever", "angular_rate", "interval")
    produces = StaleTransformResult

    def solve(self, belief: BeliefExpiryResult, lever: Quantity, angular_rate: Quantity,
              interval: Quantity) -> StaleTransformResult:
        dt = interval.to(_MS)
        drift = (belief.growth_rate * dt).to(_MM)
        swing = calc_lever_arm_error(lever, (angular_rate * dt).to(ureg.radian))
        total = (belief.initial_error + drift + swing).to(_MM)
        return StaleTransformResult(
            interval=dt, drift_term=drift, lever_term=swing, total=total,
            over_tolerance=_ratio(total, belief.tolerance) - 1, lever_over_drift=_ratio(swing, drift),
            same_epoch_error=belief.error_after(dt),
        )


class FreeSpaceExpiryModel(ForwardModel):
    """
    Occluded free space eroding inward at the approach speed ``v_h``, ahead and abreast of an occluder.

    Each horizon is ``calc_free_space_expiry``; the observation age spends
    part of it before memory receives the frame. A walker approaching during a
    stop at ``speed`` covers ``v_h (tau_delay + v / a_brake)``.
    """
    requires = ("depth_ahead", "depth_abreast", "loc_bound", "protective_clearance", "approach_speed")
    produces = FreeSpaceExpiryResult

    def solve(self, depth_ahead: Quantity, depth_abreast: Quantity, loc_bound: Quantity,
              protective_clearance: Quantity, approach_speed: Quantity, observation_age: Quantity,
              frame_period: Quantity, speed: Quantity, braking: Quantity, pre_brake_delay: Quantity,
              uniform_timeout: Optional[Quantity] = None) -> FreeSpaceExpiryResult:
        ahead = calc_free_space_expiry(depth_ahead, loc_bound, protective_clearance, approach_speed)
        abreast = calc_free_space_expiry(depth_abreast, loc_bound, protective_clearance, approach_speed)
        left = (ahead - observation_age).to(_MS)
        stop = (speed / braking).to(ureg.second)
        def certified(depth):  # free depth left after the up-front terms, floored at zero
            return max((depth - loc_bound - protective_clearance).to(_MM), 0 * _MM)

        return FreeSpaceExpiryResult(
            certified_depth_ahead=certified(depth_ahead),
            certified_depth_abreast=certified(depth_abreast),
            horizon_ahead=ahead, horizon_abreast=abreast, time_left_at_dispatch=left,
            frames_bridged=_ratio(left, frame_period),
            depth_eroded_by_age=(approach_speed * observation_age).to(_MM),
            time_per_depth=(1 / approach_speed).to(_MS / _MM),
            walker_intrusion=(approach_speed * (pre_brake_delay + stop)).to(ureg.meter),
            timeout_refused=None if uniform_timeout is None else (ahead - uniform_timeout).to(_MS),
        )


class EpochGapModel(ForwardModel):
    """
    Travel across the gap between two acquisition epochs, ``speed x |t2 - t1|``.

    With ``separation``, the slack is the separation less the ``committed``
    terms (localization bound and protective clearance), which the travel
    across the gap must fit inside.
    """
    requires = ("first_epoch", "second_epoch", "speed")
    produces = EpochGapResult

    def solve(self, first_epoch: Quantity, second_epoch: Quantity, speed: Quantity,
              separation: Optional[Quantity] = None, committed: Sequence[Quantity] = ()) -> EpochGapResult:
        gap = abs(second_epoch - first_epoch).to(_MS)
        slack = None
        if separation is not None:
            slack = separation
            for c in committed:
                slack = slack - c
            slack = slack.to(_MM)
        return EpochGapResult(gap=gap, speed=speed.to(ureg.meter / ureg.second),
                              travel=(speed * gap).to(_MM), slack=slack)


class SensorBiasDriftModel(ForwardModel):
    """A bias drifting at ``drift_rate`` for ``span`` under a steady ``reading``: the true value is the reading less the shift."""
    requires = ("reading", "drift_rate", "span")
    produces = SensorBiasDriftResult

    def solve(self, reading: Quantity, drift_rate: Quantity, span: Quantity) -> SensorBiasDriftResult:
        shift = (drift_rate * span).to(reading.units)
        return SensorBiasDriftResult(shift=shift, true_value=(reading - shift).to(reading.units),
                                     lost_share=_ratio(shift, reading))


class FrozenTransformModel(ForwardModel):
    """
    A last transform republished every ``republish_period`` with fresh stamps while the object moves.

    Every copy passes a publish-time ``freshness_threshold``; the object has
    moved ``speed x window`` from the frozen pose.
    """
    requires = ("speed", "window", "tolerance", "republish_period", "freshness_threshold")
    produces = FrozenTransformResult

    def solve(self, speed: Quantity, window: Quantity, tolerance: Quantity, republish_period: Quantity,
              freshness_threshold: Quantity) -> FrozenTransformResult:
        err = calc_belief_error_bound(0 * _MM, speed, window)
        return FrozenTransformResult(
            republish_rate=(1 / republish_period).to(ureg.hertz), error=err,
            passes_freshness=bool(republish_period < freshness_threshold), exceeds_tolerance=bool(err > tolerance),
        )


