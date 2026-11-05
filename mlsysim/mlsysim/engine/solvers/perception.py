"""Sensor-perception solvers: capture timing, observation age, and measured error scales (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). The equations live in
``physics/robotics/sensing.py``, ``physics/robotics/control.py``, and
``physics/statistics.py``; these models compose them for one stated camera,
lidar, stereo rig, or loop, so a chapter, a slide deck, and a lab read the
same arithmetic.
"""

from __future__ import annotations

import math
from typing import Optional

from ..results import (
    ClockConversionResult,
    ConformalInsetResult,
    DelayPhaseMarginResult,
    LidarSweepAgeResult,
    ObservationAgeResult,
    RollingShutterSkewResult,
    StereoDepthErrorResult,
    ThermalExpansionResult,
    TimingAttitudeErrorResult,
    VisibilityRangeResult,
)
from ...core.types import Quantity
from ...core.units import ureg
from ...physics.robotics.control import calc_delay_phase_margin_ceiling
from ...physics.robotics.sensing import (
    calc_contrast_visibility_range,
    calc_lever_arm_error,
    calc_scan_sector_angle,
    calc_stereo_baseline_bias,
    calc_stereo_depth_sigma,
    calc_stereo_range_for_sigma,
    calc_thermal_expansion,
)
from ...physics.statistics import calc_split_conformal_min_count, calc_split_conformal_rank
from .base import ForwardModel

_MM = ureg.millimeter
_MS = ureg.millisecond


def _ratio(a, b) -> float:
    return float((a / b).to("").magnitude)


class ObservationAgeModel(ForwardModel):
    """
    A camera observation's age, stage by stage, from the exposure midpoint to dispatch.

    The capture epoch is the exposure midpoint, so the age starts at half the
    exposure, then adds readout, link transport, DMA and queueing, the image
    signal processor, the vision backbone, and dispatch. A stamp taken at DMA
    completion (arrival) misdates the frame by the age up to that stage; a
    trigger stamp at exposure start misdates it by half the exposure. Batching
    ``batch_frames`` successive frames holds the oldest ``n - 1`` frame
    periods. Every interval is also priced as travel at ``speed``.
    """
    requires = ("exposure", "readout", "frame_rate", "speed")
    produces = ObservationAgeResult

    def solve(self, exposure: Quantity, readout: Quantity, transport: Quantity, dma: Quantity,
              isp: Quantity, backbone: Quantity, dispatch: Quantity, frame_rate: Quantity,
              speed: Quantity, batch_frames: int = 1) -> ObservationAgeResult:
        """Accumulate observation age from exposure midpoint to dispatch.

        Parameters
        ----------
        exposure : Quantity
            Nonnegative exposure duration, in time units.
        readout : Quantity
            Nonnegative sensor readout duration, in time units.
        transport : Quantity
            Nonnegative link-transfer duration, in time units.
        dma : Quantity
            Nonnegative DMA and queue delay, in time units.
        isp : Quantity
            Nonnegative image-signal-processing duration, in time units.
        backbone : Quantity
            Nonnegative vision-model execution duration, in time units.
        dispatch : Quantity
            Nonnegative dispatch duration, in time units.
        frame_rate : Quantity
            Positive capture frequency, convertible to hertz.
        speed : Quantity
            Travel speed, convertible to meters per second.
        batch_frames : int, optional
            Number of successive frames accumulated; at least one. The default
            one-frame batch adds no frame-collection wait.

        Returns
        -------
        ObservationAgeResult
            Stage and cumulative ages in milliseconds; travel in millimeters;
            dimensionless age and stage shares. These are timing accounts, not
            a prediction of perception accuracy.

        Notes
        -----
        Equation: age = exposure/2 + readout + transport + dma + isp + backbone
        + dispatch. The oldest frame in an n-frame batch waits (n-1)/frame_rate;
        distance traveled during any interval t is ``speed*t``.

        Assumptions: Stages execute serially with the supplied durations. Exposure
        midpoint represents capture time; frame arrivals are evenly spaced. Inputs
        must describe nonnegative intervals and a positive total age.

        Excludes: Pipeline overlap, jitter distributions, variable frame spacing,
        object motion, and a collision-safety determination.

        References
        ----------
        Derivation: Elapsed time along a serial path is the sum of its intervals.
        Changing the timestamp from the capture midpoint to arrival hides the elapsed
        capture/readout/transport/DMA intervals; batching adds arrival periods.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> durations = [Q_("2 ms")] * 7
        >>> one = ObservationAgeModel().solve(*durations, Q_("10 Hz"), Q_("1 m/s"))
        >>> two = ObservationAgeModel().solve(*durations, Q_("10 Hz"), Q_("1 m/s"), batch_frames=2)
        >>> two.batched_age - one.age == Q_("100 ms")
        True
        """
        if batch_frames < 1:
            raise ValueError("batch_frames must be at least 1")
        steps = [("exposure midpoint", exposure / 2), ("readout", readout), ("link transport", transport),
                 ("DMA and queue", dma), ("ISP", isp), ("vision backbone", backbone), ("dispatch", dispatch)]
        stages, age = [], 0 * _MS
        for name, t in steps:
            age = (age + t).to(_MS)
            stages.append((name, t.to(_MS), age))
        arrival = stages[3][2]
        top = stages[1][2]
        middle = (exposure / 2 + readout / 2).to(_MS)
        period = (1 / frame_rate).to(_MS)
        waits = [((batch_frames - k) * period).to(_MS) for k in range(1, batch_frames)]
        wait = waits[0] if waits else 0 * _MS
        batched = (age + wait).to(_MS)

        def travel(t):
            return (speed * t).to(_MM)

        return ObservationAgeResult(
            stages=stages, age=age, age_travel=travel(age),
            arrival_offset=arrival, arrival_travel=travel(arrival),
            trigger_offset=(exposure / 2).to(_MS), trigger_travel=travel(exposure / 2),
            exposure_travel=travel(exposure),
            capture_before_compute=top, sensor_age_middle_row=middle,
            top_row_excess=(top - period).to(_MS),
            capture_share=_ratio(top, age), backbone_share=_ratio(backbone, age),
            frame_period=period, frame_travel=travel(period),
            batch_frames=batch_frames, batch_waits=waits, batch_wait=wait,
            batch_wait_travel=travel(wait), batched_age=batched, batched_age_ratio=_ratio(batched, age),
        )


class RollingShutterSkewModel(ForwardModel):
    """
    Row timing and skew of a rolling-shutter camera on a moving base.

    Rows are read one after another over the readout time, so the last row
    sees the scene from ``v t_readout`` farther along, and a turn at
    ``omega`` shifts a target at range ``d`` sideways by ``d omega t_readout``.
    Each row also blurs over its own exposure, ``v t_exp``.
    """
    requires = ("rows", "readout", "exposure", "speed")
    produces = RollingShutterSkewResult

    def solve(self, rows: int, readout: Quantity, exposure: Quantity, speed: Quantity,
              yaw_rate: Quantity, target_range: Quantity, along_allowance: Quantity,
              side_clearance: Quantity) -> RollingShutterSkewResult:
        """Translate rolling-shutter readout time into motion skew.

        Parameters
        ----------
        rows : int
            Positive number of uniformly timed sensor rows.
        readout : Quantity
            Nonnegative full-image readout time.
        exposure : Quantity
            Nonnegative per-row exposure duration.
        speed : Quantity
            Translational speed, convertible to meters per second.
        yaw_rate : Quantity
            Angular velocity, convertible to radians per second.
        target_range : Quantity
            Nonnegative distance to the target.
        along_allowance : Quantity
            Positive longitudinal error allowance, in length units.
        side_clearance : Quantity
            Positive lateral clearance used to normalize angular skew.

        Returns
        -------
        RollingShutterSkewResult
            Row interval in microseconds, exposure-midpoint lag in milliseconds,
            motion displacements in millimeters, and dimensionless allowance shares.

        Notes
        -----
        Equation: row_interval = readout/rows; longitudinal skew = ``speed*readout``;
        lateral skew = ``target_range*yaw_rate*readout``, with yaw in radians.
        Exposure travel is ``speed*exposure``.

        Assumptions: Constant translation and yaw during capture, uniform row timing,
        and small angular displacement. Allowances are supplied comparisons.

        Excludes: Acceleration, lens distortion, rolling-shutter correction,
        scene-depth variation, and safety certification.

        References
        ----------
        Derivation: Uniformly partition the readout interval among rows, integrate
        constant translational velocity over that interval, and apply the small-angle
        arc-length relation ``d*theta`` to yaw motion.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = RollingShutterSkewModel().solve(100, Q_("10 ms"), Q_("2 ms"), Q_("1 m/s"), Q_("0 rad/s"), Q_("2 m"), Q_("20 mm"), Q_("20 mm"))
        >>> r.skew == Q_("10 mm") and r.lateral_skew == Q_("0 mm")
        True
        """
        skew = (speed * readout).to(_MM)
        omega = yaw_rate.to(ureg.radian / ureg.second).magnitude / ureg.second
        lateral = (target_range * omega * readout).to(_MM)
        return RollingShutterSkewResult(
            row_interval=(readout / rows).to(ureg.microsecond), midpoint_lag=(exposure / 2).to(_MS),
            skew=skew, skew_over_allowance=_ratio(skew, along_allowance),
            lateral_skew=lateral, lateral_share=_ratio(lateral, side_clearance),
            exposure_travel=(speed * exposure).to(_MM),
        )


class TimingAttitudeErrorModel(ForwardModel):
    """
    Position errors from a timing error and an angle error, for a target at ``distance``.

    A timestamp off by ``dt`` places the target ``v dt`` along the travel and,
    while turning at ``omega``, misorients the view by ``omega dt``, which at
    range is ``d omega dt`` sideways. An angle error ``theta`` moves the target
    ``d theta`` sideways (``calc_lever_arm_error``, small angle).
    """
    requires = ("distance",)
    produces = TimingAttitudeErrorResult

    def solve(self, speed: Quantity, yaw_rate: Quantity, distance: Quantity, timing_error: Quantity,
              angle_error: Quantity, reference: Optional[Quantity] = None) -> TimingAttitudeErrorResult:
        """Separate translation and angular displacement caused by timestamp error.

        Parameters
        ----------
        speed : Quantity
            Translation speed, convertible to meters per second.
        yaw_rate : Quantity
            Angular velocity, convertible to radians per second.
        distance : Quantity
            Nonnegative target range, in length units.
        timing_error : Quantity
            Signed timestamp offset, in time units.
        angle_error : Quantity
            Signed small orientation error, convertible to radians.
        reference : Quantity, optional
            Positive length used to normalize the angle-induced displacement;
            omit to return no normalized share.

        Returns
        -------
        TimingAttitudeErrorResult
            Translation and lateral errors in millimeters, timestamp-induced
            orientation error in milliradians, and an optional dimensionless share.

        See Also
        --------
        mlsysim.physics.robotics.sensing.calc_lever_arm_error : Small-angle displacement.

        Notes
        -----
        Equation: translation = ``speed*timing_error``; misorientation = yaw_rate*
        timing_error; timing lateral error = ``distance*misorientation``; independent
        angle error produces ``distance*angle_error``.

        Assumptions: Translation and yaw remain constant over the timestamp offset.
        Angular errors are small; signs of supplied errors are retained.

        Excludes: Error covariance, sensor fusion, acceleration, and a combined
        probabilistic error bound.

        References
        ----------
        Derivation: Integrate constant velocity and angular velocity over the time
        offset, then multiply each angular error in radians by its lever arm.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = TimingAttitudeErrorModel().solve(Q_("1 m/s"), Q_("0 rad/s"), Q_("2 m"), Q_("10 ms"), Q_("0 rad"))
        >>> r.timing_travel == Q_("10 mm") and r.angle_lateral == Q_("0 mm")
        True
        """
        omega = yaw_rate.to(ureg.radian / ureg.second)
        misorientation = (omega * timing_error).to(ureg.milliradian)
        lateral = calc_lever_arm_error(distance, angle_error)
        return TimingAttitudeErrorResult(
            timing_travel=(speed * timing_error).to(_MM), misorientation=misorientation,
            timing_lateral=(distance * misorientation.to(ureg.radian).magnitude).to(_MM),
            angle_lateral=lateral,
            angle_lateral_share=None if reference is None else _ratio(lateral, reference),
        )


class LidarSweepAgeModel(ForwardModel):
    """
    One spinning-lidar sweep's motion distortion, and a forward sector's age.

    A sweep lasts one rotation, during which the base travels ``v T`` and
    turns ``omega T``. A sector of width ``w`` at distance ``d`` is swept in
    ``T phi / 360 deg``; streaming it as it arrives ages it by that sector time
    plus the sensor's data latency and its time-sync bound, waiting for the
    full sweep by the whole rotation plus the same two terms.
    """
    requires = ("scan_rate",)
    produces = LidarSweepAgeResult

    def solve(self, scan_rate: Quantity, data_latency: Quantity, sync_accuracy: Quantity,
              sector_width: Quantity, sector_distance: Quantity, speed: Quantity,
              yaw_rate: Quantity, reference_age: Quantity) -> LidarSweepAgeResult:
        """Compare full-sweep and sector-streaming observation ages.

        Parameters
        ----------
        scan_rate : Quantity
            Positive rotational scan frequency, convertible to hertz.
        data_latency : Quantity
            Nonnegative sensor delivery latency, in time units.
        sync_accuracy : Quantity
            Nonnegative time-synchronization uncertainty bound.
        sector_width : Quantity
            Positive lateral width of the sector, in length units.
        sector_distance : Quantity
            Positive distance to the center of the sector.
        speed : Quantity
            Platform translation speed, convertible to meters per second.
        yaw_rate : Quantity
            Platform angular velocity, convertible to radians per second.
        reference_age : Quantity
            Nonnegative observation-age budget, in time units.

        Returns
        -------
        LidarSweepAgeResult
            Sweep and sector times and ages in milliseconds, travel in millimeters,
            rotation in milliradians, sector angle in degrees, and streamed-age slack.
            Negative slack means the supplied budget is exceeded.

        See Also
        --------
        mlsysim.physics.robotics.sensing.calc_scan_sector_angle : Sector geometry.

        Notes
        -----
        Equation: T = 1/scan_rate; phi = ``2*atan(sector_width/(2*sector_distance))``;
        sector_time = ``T*phi/(2*pi)``. Streamed age adds data_latency and sync_accuracy
        to sector_time; full age adds those terms to T.

        Assumptions: Uniform rotation, constant platform speed and yaw, and serial
        addition of the stated latency and synchronization bounds.

        Excludes: Scan deskewing, nonuniform firing patterns, dropped returns,
        network-delay distributions, and object motion.

        References
        ----------
        Derivation: A uniformly rotating scanner spends the angular fraction
        ``phi/(2*pi)`` of one revolution on the sector. Age and travel follow by adding
        serial intervals and integrating constant speed.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = LidarSweepAgeModel().solve(Q_("10 Hz"), Q_("1 ms"), Q_("1 ms"), Q_("1 m"), Q_("5 m"), Q_("1 m/s"), Q_("0 rad/s"), Q_("20 ms"))
        >>> r.age_streamed < r.age_full and r.sweep_period == Q_("100 ms")
        True
        """
        period = (1 / scan_rate).to(_MS)
        angle = calc_scan_sector_angle(sector_width, sector_distance)
        sector = (period * _ratio(angle, 360 * ureg.degree)).to(_MS)
        streamed = (sector + data_latency + sync_accuracy).to(_MS)
        full = (period + data_latency + sync_accuracy).to(_MS)
        return LidarSweepAgeResult(
            sweep_period=period, sweep_travel=(speed * period).to(_MM),
            sweep_rotation=(yaw_rate * period).to(ureg.milliradian),
            sector_angle=angle, sector_time=sector, age_streamed=streamed, age_full=full,
            travel_streamed=(speed * streamed).to(_MM), travel_full=(speed * full).to(_MM),
            slack_streamed=(reference_age - streamed).to(_MS),
        )


class StereoDepthErrorModel(ForwardModel):
    """
    One-sigma stereo depth error against range, and where it spends each clearance.

    ``sigma_z = z^2 sigma_d / (f b)`` grows with squared range; the range at
    which one sigma equals a clearance C is ``sqrt(C f b / sigma_d)``. Adding
    one sigma to a clearance kept along the viewing ray is the shortcut inset;
    an unmodeled baseline change ``db`` biases depth by ``z db / b``.
    """
    requires = ("baseline", "focal_length_px", "disparity_sigma_px")
    produces = StereoDepthErrorResult

    def solve(self, baseline: Quantity, focal_length_px: float, disparity_sigma_px: float,
              near_range: Quantity, far_range: Quantity, target_range: Quantity,
              ray_clearance: Quantity, loc_allowance: Quantity, side_clearance: Quantity,
              baseline_change: Quantity, bias_range: Quantity) -> StereoDepthErrorResult:
        """Propagate stereo disparity uncertainty into range-dependent depth error.

        Parameters
        ----------
        baseline : Quantity
            Positive camera separation, in length units.
        focal_length_px : float
            Positive rectified-camera focal length, in pixels.
        disparity_sigma_px : float
            Positive disparity standard deviation, in pixels.
        near_range : Quantity
            Positive near comparison depth, in length units.
        far_range : Quantity
            Positive far comparison depth, in length units.
        target_range : Quantity
            Positive depth of the target whose uncertainty is reported.
        ray_clearance : Quantity
            Nonnegative clearance added to one standard deviation along the ray.
        loc_allowance : Quantity
            Positive localization allowance used to invert the uncertainty equation.
        side_clearance : Quantity
            Positive second error allowance used for range comparisons.
        baseline_change : Quantity
            Nonnegative magnitude of unmodeled baseline change.
        bias_range : Quantity
            Positive depth at which baseline-change bias is evaluated.

        Returns
        -------
        StereoDepthErrorResult
            One-sigma errors, clearance-plus-sigma insets, and baseline bias in
            millimeters; inverse-error ranges in meters; dimensionless growth ratios.
            One sigma is not a guaranteed containment interval.

        See Also
        --------
        mlsysim.physics.robotics.sensing.calc_stereo_depth_sigma : Disparity propagation.
        mlsysim.physics.robotics.sensing.calc_stereo_range_for_sigma : Inverse error relation.
        mlsysim.physics.robotics.sensing.calc_stereo_baseline_bias : Baseline sensitivity.

        Notes
        -----
        Equation: sigma_z = ``z**2*sigma_d/(f*b)``; z_at_allowance =
        ``sqrt(allowance*f*b/sigma_d)``; baseline bias = ``bias_range*baseline_change/b``.
        The near and far insets add ray_clearance to their respective sigma_z.

        Assumptions: Rectified pinhole stereo, small disparity errors, fixed f and b,
        and the same disparity variance across all evaluated depths.

        Excludes: Occlusions, matching outliers, lens/calibration covariance,
        correlated errors, and any claimed collision-risk probability.

        References
        ----------
        Derivation: Differentiate z=``f*b/disparity`` with respect to disparity to
        propagate one standard deviation; invert that expression for a chosen error
        allowance. Baseline enters linearly at fixed disparity.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = StereoDepthErrorModel().solve(Q_("0.1 m"), 1000, 1, Q_("1 m"), Q_("2 m"), Q_("1 m"), Q_("1 mm"), Q_("10 mm"), Q_("20 mm"), Q_("0 mm"), Q_("1 m"))
        >>> r.sigma_far == 4 * r.sigma_near and r.baseline_bias == Q_("0 mm")
        True
        """
        f, sd = focal_length_px, disparity_sigma_px
        near = calc_stereo_depth_sigma(near_range, f, baseline, sd)
        far = calc_stereo_depth_sigma(far_range, f, baseline, sd)
        return StereoDepthErrorResult(
            sigma_near=near, sigma_far=far, sigma_target=calc_stereo_depth_sigma(target_range, f, baseline, sd),
            range_ratio=_ratio(far_range, near_range), sigma_growth=_ratio(far, near),
            inset_near=(ray_clearance + near).to(_MM), inset_far=(ray_clearance + far).to(_MM),
            range_at_loc_allowance=calc_stereo_range_for_sigma(loc_allowance, f, baseline, sd),
            range_at_side_clearance=calc_stereo_range_for_sigma(side_clearance, f, baseline, sd),
            range_at_side_clearance_half_baseline=calc_stereo_range_for_sigma(side_clearance, f, baseline / 2, sd),
            baseline_bias=calc_stereo_baseline_bias(bias_range, baseline, baseline_change),
        )


class DelayPhaseMarginModel(ForwardModel):
    """
    The delay a loop's phase margin can absorb, and the share a stated delay spends.

    A pure delay removes ``omega_c tau`` of phase at crossover, so the margin
    is gone at ``tau_max = PM / omega_c`` (``calc_delay_phase_margin_ceiling``).
    """
    requires = ("phase_margin", "crossover", "delay")
    produces = DelayPhaseMarginResult

    def solve(self, phase_margin: Quantity, crossover: Quantity, delay: Quantity) -> DelayPhaseMarginResult:
        """Compute the pure-delay budget implied by a phase margin.

        Parameters
        ----------
        phase_margin : Quantity
            Positive available phase margin, convertible to radians.
        crossover : Quantity
            Positive loop crossover angular frequency, in radians per second.
        delay : Quantity
            Nonnegative additional pure delay, in time units.

        Returns
        -------
        DelayPhaseMarginResult
            Phase margin in radians, crossover in hertz, delay and delay ceiling in
            milliseconds, and the dimensionless fraction of the ceiling consumed.

        See Also
        --------
        mlsysim.physics.robotics.control.calc_delay_phase_margin_ceiling : Pure-delay ceiling.

        Notes
        -----
        Equation: delay_ceiling = phase_margin/crossover, after expressing margin
        in radians and crossover in radians per second. share_used = delay/ceiling;
        crossover_frequency = ``crossover/(2*pi)``.

        Assumptions: A fixed crossover and a pure transport-delay phase contribution.
        The margin refers to the loop before adding this delay.

        Excludes: Changes to gain crossover, other plant dynamics, sampling and
        jitter distributions, and a complete closed-loop stability proof.

        References
        ----------
        Derivation: The frequency response of a delay is ``exp(-j*omega*tau)``, whose
        phase is ``-omega*tau``. Set that phase loss equal to the available phase margin.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = DelayPhaseMarginModel().solve(Q_("1 rad"), Q_("10 rad/s"), Q_("50 ms"))
        >>> r.delay_ceiling == Q_("100 ms") and r.share_used == 0.5
        True
        """
        ceiling = calc_delay_phase_margin_ceiling(phase_margin, crossover)
        wc = crossover.to(ureg.radian / ureg.second).magnitude
        return DelayPhaseMarginResult(
            phase_margin=phase_margin.to(ureg.radian),
            crossover_frequency=(wc / (2 * math.pi) * ureg.hertz),
            delay_ceiling=ceiling, delay=delay.to(_MS), share_used=_ratio(delay, ceiling),
        )


class ConformalInsetModel(ForwardModel):
    """
    The calibration rank behind a split-conformal bound on residuals.

    The bound is the ``ceil((n + 1)(1 - alpha))``-th smallest of ``n``
    residuals, which exists only from ``n = ceil((1 - alpha) / alpha)``.
    """
    requires = ("alpha", "calibration_count")
    produces = ConformalInsetResult

    def solve(self, alpha: float, calibration_count: int) -> ConformalInsetResult:
        """Locate the finite-sample order statistic for a split-conformal bound.

        Parameters
        ----------
        alpha : float
            Desired miscoverage fraction, strictly between zero and one.
        calibration_count : int
            Positive number of held-out calibration residuals; large enough for
            the requested order statistic to exist.

        Returns
        -------
        ConformalInsetResult
            Dimensionless miscoverage, calibration count, minimum count, one-based
            ascending rank, and one-based rank counted down from the largest residual.
            No residual values are inspected and no physical inset is calculated.

        See Also
        --------
        mlsysim.physics.statistics.calc_split_conformal_rank : Finite-sample rank.
        mlsysim.physics.statistics.calc_split_conformal_min_count : Minimum sample count.

        Notes
        -----
        Equation: rank = ``ceil((n+1)*(1-alpha))``; min_count = ceil((1-alpha)/alpha);
        rank_from_top = n-rank+1.

        Assumptions: Calibration and future examples are exchangeable and use the same
        fixed score. A finite observed rank must not exceed the calibration count.

        Excludes: Distribution-shift protection, score construction, conditional
        coverage guarantees, and computation of the residual quantile itself.

        References
        ----------
        Derivation: The split-conformal finite-sample rank uses the n+1 possible
        positions of a future exchangeable score. Requiring ``ceil((n+1)*(1-alpha))``
        <= n gives the minimum calibration-count expression. The rank itself is
        implemented by the referenced statistics helpers.

        Examples
        --------
        >>> r = ConformalInsetModel().solve(0.1, 19)
        >>> r.rank + r.rank_from_top == r.calibration_count + 1
        True
        """
        rank = calc_split_conformal_rank(calibration_count, alpha)
        return ConformalInsetResult(
            alpha=alpha, calibration_count=calibration_count,
            min_count=calc_split_conformal_min_count(alpha), rank=rank,
            rank_from_top=calibration_count - rank + 1,
        )


class ClockConversionModel(ForwardModel):
    """
    A free-running clock's drift between syncs, and the dispatch limit a conversion bound sets.

    Drift accumulates as ``rate x interval``; a producer refuses to dispatch
    an observation older than its budgeted age plus the conversion bound,
    whose own uncertainty is travel at ``speed``.
    """
    requires = ("drift_rate", "sync_interval", "conversion_bound")
    produces = ClockConversionResult

    def solve(self, drift_rate: Quantity, sync_interval: Quantity, conversion_bound: Quantity,
              observation_age: Quantity, speed: Quantity) -> ClockConversionResult:
        """Convert clock drift and conversion uncertainty into age and travel budgets.

        Parameters
        ----------
        drift_rate : Quantity
            Clock offset accumulation rate, such as microseconds per second.
        sync_interval : Quantity
            Nonnegative time between clock synchronizations.
        conversion_bound : Quantity
            Nonnegative timestamp-conversion uncertainty, in time units.
        observation_age : Quantity
            Nonnegative nominal observation-age budget, in time units.
        speed : Quantity
            Translation speed, convertible to meters per second.

        Returns
        -------
        ClockConversionResult
            Accumulated drift in microseconds, dispatch age limit in milliseconds,
            and travel associated with the conversion bound in millimeters.

        Notes
        -----
        Equation: drift_per_interval = ``drift_rate*sync_interval``;
        dispatch_age_limit = observation_age+conversion_bound;
        bound_travel = ``speed*conversion_bound``.

        Assumptions: Constant drift rate and speed over the supplied interval. The
        conversion bound is an externally justified input, not estimated here.

        Excludes: Oscillator-temperature variation, synchronization protocol
        failures, stochastic timestamp error, and an enforced dispatch policy.

        References
        ----------
        Derivation: Integrate a constant offset rate over time, add an uncertainty
        allowance to the nominal age budget, and convert that time allowance to
        travel by multiplying by speed.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = ClockConversionModel().solve(Q_("10 us/s"), Q_("2 s"), Q_("1 ms"), Q_("5 ms"), Q_("1 m/s"))
        >>> r.drift_per_interval == Q_("20 us") and r.bound_travel == Q_("1 mm")
        True
        """
        return ClockConversionResult(
            drift_per_interval=(drift_rate * sync_interval).to(ureg.microsecond),
            dispatch_age_limit=(observation_age + conversion_bound).to(_MS),
            bound_travel=(speed * conversion_bound).to(_MM),
        )


class VisibilityRangeModel(ForwardModel):
    """Range at which fog with extinction ``beta`` drops a black target's contrast to the threshold (Koschmieder)."""
    requires = ("extinction",)
    produces = VisibilityRangeResult

    def solve(self, extinction: Quantity, contrast_threshold: float) -> VisibilityRangeResult:
        """Find the distance at which atmospheric attenuation reaches a contrast threshold.

        Parameters
        ----------
        extinction : Quantity
            Positive homogeneous extinction coefficient, in inverse length units.
        contrast_threshold : float
            Detection contrast fraction, strictly between zero and one.

        Returns
        -------
        VisibilityRangeResult
            Modeled visibility range in meters for a black target against the
            assumed background. This is not a sensor detection guarantee.

        See Also
        --------
        mlsysim.physics.robotics.sensing.calc_contrast_visibility_range : Contrast attenuation.

        Notes
        -----
        Equation: visibility_range = log(1/contrast_threshold)/extinction,
        from contrast(d) = ``exp(-extinction*d)``.

        Assumptions: Homogeneous attenuation, unit initial contrast of a black target,
        and a fixed detection threshold supplied by the caller.

        Excludes: Illumination variation, finite target size, sensor noise,
        scattering geometry, and learned-detector performance.

        References
        ----------
        Derivation: Solve the exponential contrast-attenuation relation for distance
        at the chosen contrast threshold; the sensing helper documents the
        Koschmieder model underlying that relation.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> a = VisibilityRangeModel().solve(Q_("0.1 / m"), 0.1)
        >>> b = VisibilityRangeModel().solve(Q_("0.2 / m"), 0.1)
        >>> a.visibility_range == 2 * b.visibility_range
        True
        """
        return VisibilityRangeResult(visibility_range=calc_contrast_visibility_range(extinction, contrast_threshold))


class ThermalExpansionModel(ForwardModel):
    """Length change of a uniformly heated member, ``alpha L dT``."""
    requires = ("coefficient", "length", "temperature_rise")
    produces = ThermalExpansionResult

    def solve(self, coefficient: Quantity, length: Quantity, temperature_rise: Quantity) -> ThermalExpansionResult:
        """Calculate linear expansion of a uniformly heated member.

        Parameters
        ----------
        coefficient : Quantity
            Linear expansion coefficient, in inverse kelvin or inverse temperature
            difference units; treated as constant over the interval.
        length : Quantity
            Positive initial member length, in length units.
        temperature_rise : Quantity
            Signed temperature difference, in kelvin or delta-degrees Celsius;
            not an absolute Celsius temperature.

        Returns
        -------
        ThermalExpansionResult
            Signed length change in millimeters. Positive values denote expansion;
            negative values denote contraction under the supplied coefficient.

        See Also
        --------
        mlsysim.physics.robotics.sensing.calc_thermal_expansion : Linear thermal expansion.

        Notes
        -----
        Equation: expansion = ``coefficient*length*temperature_rise``.

        Assumptions: Uniform temperature change, unconstrained expansion, constant
        linear coefficient, and small strain.

        Excludes: Thermal gradients, anisotropy, mechanical restraint, temperature-
        dependent coefficients, stress, and buckling.

        References
        ----------
        Derivation: The linear thermal-expansion coefficient is fractional length
        change per temperature change; multiplying it by initial length and the
        uniform temperature increment gives the first-order length change.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> a = ThermalExpansionModel().solve(Q_("1e-5 / K"), Q_("1 m"), Q_("10 K"))
        >>> b = ThermalExpansionModel().solve(Q_("1e-5 / K"), Q_("2 m"), Q_("10 K"))
        >>> b.expansion == 2 * a.expansion
        True
        """
        return ThermalExpansionResult(expansion=calc_thermal_expansion(coefficient, length, temperature_rise))
