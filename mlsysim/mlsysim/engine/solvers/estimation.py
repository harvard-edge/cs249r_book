"""Spatial-memory solvers: state estimation, estimator consistency, drift, and belief expiry (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers``. The equations live in
``physics/robotics/estimation.py``, ``physics/robotics/perception.py``,
``physics/robotics/sensing.py``, and ``physics/statistics.py``; these models
compose them for one stated estimator, sensor, benchmark, or belief, so a
chapter, a slide deck, and a lab read the same arithmetic.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from ..results import (
    BeliefExpiryResult,
    EpochGapResult,
    EstimatorConsistencyResult,
    FreeSpaceExpiryResult,
    FrozenTransformResult,
    InertialCoastResult,
    OdometryDriftResult,
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
        """One scalar predict and two updates of a range, and the same second update misdated.

        Parameters
        ----------
        prior_sigma : Quantity
            Prior range standard deviation, positive length.
        speed_sigma : Quantity
            Speed standard deviation, nonnegative length/time.
        interval : Quantity
            Elapsed time, positive.
        first_sigma : Quantity
            First sensor standard deviation, positive length.
        second_sigma : Quantity
            Second sensor standard deviation, positive length.
        misdating_bias : Quantity
            Signed range offset caused by using the wrong epoch, length.
        comparison_scale : float
            Positive dimensionless multiplier for the second sensor standard deviation; default 2.

        Returns
        -------
        RangeFusionResult
            Posterior range standard deviations and bias shift in millimeters,
            innovation variance in square millimeters, and dimensionless gains,
            normalized shifts, and expected NIS.

        See Also
        --------
        calc_scalar_kalman_update : Composed numerical primitive.

        Notes
        -----
        Equation: Prediction adds ``(sigma_v dt)^2``; each update applies
        ``calc_scalar_kalman_update``. A reading paired with the wrong pose is
        off by ``misdating_bias``; the update moves the estimate ``g b`` while the
        reported spread, which uses only the modeled noise, is unchanged. Its
        innovation's NIS has mean ``1 + b^2 / S``. ``comparison_scale`` reruns the
        second update with that sensor's spread scaled.

        Assumptions: Scalar linear observations with independent Gaussian noises; velocity
        uncertainty contributes independent variance during prediction.

        Excludes: Correlated sensors, nonlinear range geometry, and changes in bias
        during the update sequence.

        References
        ----------
        Derivation: Add independent variances for prediction, then apply K=P/(P+R)
        and P_post=(1-K)P. A deterministic observation bias shifts the
        posterior by Kb and adds b squared over innovation variance to NIS.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.estimation import RangeFusionModel
        >>> result = RangeFusionModel().solve(prior_sigma=Q_('2 mm'), speed_sigma=Q_('1 mm/s'), interval=Q_('1 s'), first_sigma=Q_('1 mm'), second_sigma=Q_('1 mm'), misdating_bias=Q_('1 mm'))
        >>> bool(result.second_gain > 0)
        True
        """
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
        """The band a campaign's mean NEES or NIS must fall in, and what single-update gates catch.

        Parameters
        ----------
        runs : int
            Positive independent trial count.
        dimension : int
            Positive state or innovation dimension.
        confidence : float
            Confidence probability strictly between zero and one.
        gate_confidences : sequence of float
            Gate coverage probabilities strictly between zero and one.
        nis_expected : float
            Expected dimensionless NIS, at least dimension.

        Returns
        -------
        EstimatorConsistencyResult
            Dimensionless lower and upper bounds on mean NEES/NIS, per-update
            chi-square gates, and probabilities of detecting the supplied bias.

        See Also
        --------
        calc_chi2_gate : Composed numerical primitive.
        calc_chi2_mean_band : Composed numerical primitive.
        calc_noncentral_gate_detection : Composed numerical primitive.

        Notes
        -----
        Equation: ``calc_chi2_mean_band`` gives the band for ``runs`` statistics of
        dimension ``dimension``; each gate is ``calc_chi2_gate`` at its confidence,
        and a biased update with mean NIS ``nis_expected`` exceeds it with
        ``calc_noncentral_gate_detection`` (noncentrality ``nis_expected - dimension``).

        Assumptions: Independent campaign statistics with the stated chi-square dimension;
        biased single-update statistics have the stated noncentrality.

        Excludes: Temporal correlation, distribution misspecification, and guarantees
        that a finite campaign detects every inconsistent estimator.

        References
        ----------
        Derivation: Independent chi-square statistics add their degrees of freedom.
        Divide sum-statistic quantiles by runs for the mean band; evaluate
        noncentral chi-square upper tails for biased gate exceedance.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.estimation import EstimatorConsistencyModel
        >>> result = EstimatorConsistencyModel().solve(runs=100, dimension=1, confidence=0.95, gate_confidences=[0.95], nis_expected=2)
        >>> bool(result.band_low < result.band_high)
        True
        """
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
        """Count slots of a rewind buffer sampled at ``sample_rate`` that reach back ``horizon``: ``ceil(horizon / dt)``.

        Parameters
        ----------
        horizon : Quantity
            Nonnegative retained time span.
        sample_rate : Quantity
            Positive sampling frequency, inverse time.

        Returns
        -------
        ReplayBufferResult
            Sampling period in milliseconds and integer retained sample slots.

        Notes
        -----
        Equation: Slots of a rewind buffer sampled at ``sample_rate`` that reach back ``horizon``: ``ceil(horizon / dt)``.

        Assumptions: Periodic sampling at a constant rate and a horizon measured from the
        newest sample; slot count follows the implemented ceiling convention.

        Excludes: Per-record byte size, indexing overhead, and scheduling jitter.

        References
        ----------
        Derivation: A sample occupies one period 1/r. Covering horizon H requires
        ceil(H/(1/r)) slots under this retention convention.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.estimation import ReplayBufferModel
        >>> result = ReplayBufferModel().solve(horizon=Q_('1 s'), sample_rate=Q_('10 Hz'))
        >>> bool(result.slots == 10)
        True
        """
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
        """An IMU integrated open loop for ``duration``, with the heading offset seen at ``target_range``.

        Parameters
        ----------
        gyro_noise_density : Quantity
            Angular random-walk density, radians / sqrt(second).
        gyro_bias : Quantity
            Constant angular-rate bias, radians/second.
        accel_noise_density : Quantity
            Acceleration noise density, meters/second**1.5.
        accel_bias : Quantity
            Constant acceleration bias, length/time**2.
        duration : Quantity
            Positive open-loop integration time.
        target_range : Quantity
            Nonnegative lever distance to target, length.
        reference : Quantity
            Positive position-error comparison bound, length.

        Returns
        -------
        InertialCoastResult
            Angular noise and bias in radians, lateral and position errors in
            length units returned by the physics functions, and dimensionless
            position-bias ratios against the reference and noise.

        See Also
        --------
        calc_inertial_coast_errors : Composed numerical primitive.
        calc_lever_arm_error : Composed numerical primitive.

        Notes
        -----
        Equation: ``calc_inertial_coast_errors`` gives the four terms; the heading offset
        moves a point at range sideways by ``calc_lever_arm_error``. The position
        offset is priced against ``reference`` (a localization bound).

        Assumptions: Constant gyro and accelerometer biases, white independent noise, and
        small-angle lever-arm displacement during open-loop integration.

        Excludes: Orientation-dependent gravity leakage, sensor cross-correlations,
        and external corrections during coasting.

        References
        ----------
        Derivation: Integrate constant angular-rate bias once and acceleration bias
        twice. White-noise integrals produce the time powers in
        calc_inertial_coast_errors; a small angular error sweeps range
        times angle laterally.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.estimation import InertialCoastModel
        >>> result = InertialCoastModel().solve(gyro_noise_density=Q_('0.01 rad/s**0.5'), gyro_bias=Q_('0.01 rad/s'), accel_noise_density=Q_('0.01 m/s**1.5'), accel_bias=Q_('0.01 m/s**2'), duration=Q_('1 s'), target_range=Q_('1 m'), reference=Q_('1 mm'))
        >>> bool(result.bias_over_reference > 0)
        True
        """
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
        """A benchmark's return error over its path as a drift rate, before and after calibration.

        Parameters
        ----------
        return_error_uncalibrated : Quantity
            Positive benchmark return error before calibration, length.
        return_error_calibrated : Quantity
            Positive benchmark return error after calibration, length.
        path_length : Quantity
            Positive benchmark travel distance, length.
        bound : Quantity
            Positive allowed position error, length.
        speed : Quantity
            Positive constant operating speed, length/time.
        segment : Quantity
            Nonnegative segment travel distance, length.
        test_speed : Quantity
            Positive benchmark speed, length/time.

        Returns
        -------
        OdometryDriftResult
            Dimensionless drift rates and calibration gain, allowed travel in
            meters, operating times in seconds, and segment errors in millimeters.

        Notes
        -----
        Equation: The rate ``eps = E / L`` spends a bound ``delta`` after ``delta / eps`` of
        travel, which takes ``delta / (eps v)`` at the operating speed; over a
        segment of length ``s`` the error is ``eps s``.

        Assumptions: Benchmark return-error per path length is extrapolated as a constant
        linear drift rate at the supplied operating speed.

        Excludes: Path-dependent cancellation, nonlinear drift, and evidence that a
        calibration gain transfers to different surfaces or motion regimes.

        References
        ----------
        Derivation: Divide return error by benchmark path length to estimate error
        per unit distance; solve bound=``rate*distance`` and distance=``speed*time``.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.estimation import OdometryDriftModel
        >>> result = OdometryDriftModel().solve(return_error_uncalibrated=Q_('2 mm'), return_error_calibrated=Q_('1 mm'), path_length=Q_('1 m'), bound=Q_('10 mm'), speed=Q_('1 m/s'), segment=Q_('1 m'), test_speed=Q_('1 m/s'))
        >>> bool(result.calibration_gain > 1)
        True
        """
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
        """A belief's error bound ``E(dt) = E_0 + v dt + a dt^2 / 2``, its invalidation horizon, and an admission check.

        Parameters
        ----------
        initial_error : Quantity
            Nonnegative initial spatial error, length.
        growth_rate : Quantity
            Positive error-growth rate, length/time.
        tolerance : Quantity
            Positive allowed spatial error, length.
        growth_acceleration : Quantity, optional
            Nonnegative error-growth acceleration, length/time**2; None means zero.
        use_age : Quantity, optional
            Optional nonnegative consumer age, time; None omits use-time fields.
        age_bound : Quantity, optional
            Nonnegative additive clock-age bound, time; None means zero.

        Returns
        -------
        BeliefExpiryResult
            Initial error and tolerance in millimeters, growth rates in millimeters
            per second, expiry times in milliseconds, and dimensionless overshoot.
            Use-time fields remain absent when use_age is None.

        See Also
        --------
        calc_belief_error_bound : Composed numerical primitive.
        calc_quadratic_invalidation_horizon : Composed numerical primitive.

        Notes
        -----
        Equation: The horizon is ``calc_quadratic_invalidation_horizon``; dropping the
        acceleration term gives the constant-velocity horizon, at which the true
        bound has already overrun the tolerance. With ``use_age``, the consumer's
        age bound is ``use_age + age_bound`` (the clock-conversion bound), and the
        belief is admitted if the error there is within tolerance; the latest
        admissible use age is the horizon less ``age_bound``.

        Assumptions: Nonnegative error-growth terms bound the full interval; optional
        clock uncertainty is an additive upper bound on observation age.

        Excludes: Stochastic cancellation, abrupt changes in motion, and measurement
        updates that would reset the initial error.

        References
        ----------
        Derivation: Integrate a constant growth rate and acceleration to obtain
        ``E0+v*t+a*t*t/2``; solve its first crossing of tolerance and subtract
        the clock-age bound for latest admissible use.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.estimation import BeliefExpiryModel
        >>> result = BeliefExpiryModel().solve(initial_error=Q_('1 mm'), growth_rate=Q_('1 mm/s'), tolerance=Q_('2 mm'))
        >>> bool(result.horizon.to("s").magnitude == 1)
        True
        """
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
        """A cached sensor-to-target edge, one ``interval`` old, composed with the current wrist pose.

        Parameters
        ----------
        belief : BeliefExpiryResult
            Previously computed belief with error, growth rate, and tolerance quantities.
        lever : Quantity
            Nonnegative distance from rotation axis, length.
        angular_rate : Quantity
            Angular rate, radians/time.
        interval : Quantity
            Elapsed time, positive.

        Returns
        -------
        StaleTransformResult
            Drift, lever swing, total error and same-epoch error in millimeters;
            interval in milliseconds and dimensionless tolerance/drift ratios.

        See Also
        --------
        calc_lever_arm_error : Composed numerical primitive.

        Notes
        -----
        Equation: The finger error is the initial error, plus the unobserved drift over the
        interval ``v dt``, plus the lever-arm swing ``l omega dt`` of the rotation
        the cached edge never saw. Composing the edge with the wrist pose of its
        own epoch removes the lever term and leaves the belief's own bound.

        Assumptions: The cached transform misses a constant-rate small-angle rotation.
        The stale total uses the belief initial error and linear growth term.

        Excludes: Higher-order rotational geometry and acceleration in the stale
        linear sum; same_epoch_error uses the belief full error function.

        References
        ----------
        Derivation: Add initial error, ``v*dt`` drift, and ``lever*angular_rate*dt``.
        Comparing with the belief evaluated at dt separates stale-frame
        lever error from same-epoch belief growth.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.estimation import StaleTransformModel
        >>> from mlsysim.engine.solvers.estimation import BeliefExpiryModel
        >>> result = StaleTransformModel().solve(belief=BeliefExpiryModel().solve(Q_('1 mm'), Q_('1 mm/s'), Q_('10 mm')), lever=Q_('1 m'), angular_rate=Q_('0.01 rad/s'), interval=Q_('1 s'))
        >>> bool(result.total > result.same_epoch_error)
        True
        """
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
        """Occluded free space eroding inward at the approach speed ``v_h``, ahead and abreast of an occluder.

        Parameters
        ----------
        depth_ahead : Quantity
            Nonnegative observed free depth ahead, length.
        depth_abreast : Quantity
            Nonnegative observed free depth beside the occluder, length.
        loc_bound : Quantity
            Nonnegative localization-error bound, length.
        protective_clearance : Quantity
            Nonnegative reserved separation, length.
        approach_speed : Quantity
            Positive constant approach speed, length/time.
        observation_age : Quantity
            Nonnegative age since acquisition, time.
        frame_period : Quantity
            Positive sampling period, time.
        speed : Quantity
            Positive constant operating speed, length/time.
        braking : Quantity
            Positive deceleration magnitude, length/time**2.
        pre_brake_delay : Quantity
            Nonnegative delay before braking, time.
        uniform_timeout : Quantity, optional
            Optional nonnegative timeout, time; None omits timeout comparison.

        Returns
        -------
        FreeSpaceExpiryResult
            Remaining depths and erosion in millimeters, horizons in milliseconds,
            walker intrusion in meters, and dimensionless bridged-frame count.
            The optional timeout comparison is a signed time margin.

        See Also
        --------
        calc_free_space_expiry : Composed numerical primitive.

        Notes
        -----
        Equation: Each horizon is ``calc_free_space_expiry``; the observation age spends
        part of it before memory receives the frame. A walker approaching during a
        stop at ``speed`` covers ``v_h (tau_delay + v / a_brake)``.

        Assumptions: Constant approach speed erodes free depth; localization and
        protective clearance are reserved before assigning an expiry time.

        Excludes: Occluder shape changes, approach acceleration, and safety
        certification from a single free-space observation.

        References
        ----------
        Derivation: Subtract localization and clearance from observed depth, floor
        remaining depth at zero, then divide by approach speed. During
        robot delay plus speed/braking stop time, a walker advances at
        the supplied approach speed.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.estimation import FreeSpaceExpiryModel
        >>> result = FreeSpaceExpiryModel().solve(depth_ahead=Q_('2 m'), depth_abreast=Q_('1 m'), loc_bound=Q_('10 mm'), protective_clearance=Q_('10 mm'), approach_speed=Q_('1 m/s'), observation_age=Q_('1 ms'), frame_period=Q_('10 ms'), speed=Q_('1 m/s'), braking=Q_('1 m/s**2'), pre_brake_delay=Q_('10 ms'))
        >>> bool(result.horizon_ahead > result.horizon_abreast)
        True
        """
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
        """Travel across the gap between two acquisition epochs, ``speed x |t2 - t1|``.

        Parameters
        ----------
        first_epoch : Quantity
            First acquisition timestamp, time on a common clock.
        second_epoch : Quantity
            Second acquisition timestamp on the same clock, time.
        speed : Quantity
            Positive constant operating speed, length/time.
        separation : Quantity, optional
            Optional nonnegative available separation, length; None omits slack.
        committed : sequence of Quantity
            Reserved length terms subtracted from separation; defaults to no reservations.

        Returns
        -------
        EpochGapResult
            Absolute epoch gap in milliseconds, speed in meters per second,
            travel and optional remaining separation slack in millimeters.

        Notes
        -----
        Equation: With ``separation``, the slack is the separation less the ``committed``
        terms (localization bound and protective clearance), which the travel
        across the gap must fit inside.

        Assumptions: Both acquisition timestamps use a common clock and speed bounds
        travel across their absolute difference.

        Excludes: Clock offset uncertainty, acceleration, and the actual path
        geometry between acquisitions.

        References
        ----------
        Derivation: Distance over an epoch gap is ``speed*abs(t2-t1)``; subtract the
        committed separation terms to obtain the comparison slack.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.estimation import EpochGapModel
        >>> result = EpochGapModel().solve(first_epoch=Q_('1 s'), second_epoch=Q_('2 s'), speed=Q_('1 m/s'))
        >>> bool(result.travel.to("m").magnitude == 1)
        True
        """
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
        """A bias drifting at ``drift_rate`` for ``span`` under a steady ``reading``: the true value is the reading less the shift.

        Parameters
        ----------
        reading : Quantity
            Nonzero observed quantity whose units define the result.
        drift_rate : Quantity
            Signed sensor-bias change per unit time, reading units/time.
        span : Quantity
            Nonnegative elapsed drift time.

        Returns
        -------
        SensorBiasDriftResult
            Bias shift and inferred true value in reading units, plus a signed
            dimensionless shift-to-reading ratio.

        Notes
        -----
        Equation: A bias drifting at ``drift_rate`` for ``span`` under a steady ``reading``: the true value is the reading less the shift.

        Assumptions: A constant signed bias-drift rate acts for the supplied span while
        the observed reading remains steady and nonzero.

        Excludes: Random drift, sensor saturation, and any separately changing
        physical process that also contributes to the reading.

        References
        ----------
        Derivation: Integrate constant drift to get shift=``rate*span``; infer the
        underlying value by subtracting that shift from the reading.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.estimation import SensorBiasDriftModel
        >>> result = SensorBiasDriftModel().solve(reading=Q_('10 mm'), drift_rate=Q_('1 mm/s'), span=Q_('1 s'))
        >>> bool(result.true_value.to("mm").magnitude == 9)
        True
        """
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
        """A last transform republished every ``republish_period`` with fresh stamps while the object moves.

        Parameters
        ----------
        speed : Quantity
            Positive constant operating speed, length/time.
        window : Quantity
            Nonnegative interval since the frozen acquisition, time.
        tolerance : Quantity
            Positive allowed spatial error, length.
        republish_period : Quantity
            Positive period between repeated publications, time.
        freshness_threshold : Quantity
            Positive allowed publish age, time.

        Returns
        -------
        FrozenTransformResult
            Republish frequency in hertz, displacement error in millimeters, and
            booleans for publish-age acceptance and spatial-tolerance violation.

        See Also
        --------
        calc_belief_error_bound : Composed numerical primitive.

        Notes
        -----
        Equation: Every copy passes a publish-time ``freshness_threshold``; the object has
        moved ``speed x window`` from the frozen pose.

        Assumptions: One acquired pose stays frozen while its timestamp is refreshed;
        the target moves at constant speed over the acquisition window.

        Excludes: Actual pose updates, clock uncertainty, and acceleration.

        References
        ----------
        Derivation: Publication age depends on the republish period, whereas spatial
        error depends on speed times elapsed acquisition age. Comparing
        the two separately exposes why a fresh stamp cannot refresh data.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.solvers.estimation import FrozenTransformModel
        >>> result = FrozenTransformModel().solve(speed=Q_('1 mm/s'), window=Q_('10 s'), tolerance=Q_('1 mm'), republish_period=Q_('1 ms'), freshness_threshold=Q_('2 ms'))
        >>> bool(result.passes_freshness and result.exceeds_tolerance)
        True
        """
        err = calc_belief_error_bound(0 * _MM, speed, window)
        return FrozenTransformResult(
            republish_rate=(1 / republish_period).to(ureg.hertz), error=err,
            passes_freshness=bool(republish_period < freshness_threshold), exceeds_tolerance=bool(err > tolerance),
        )


