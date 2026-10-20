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
        omega = yaw_rate.to(ureg.radian / ureg.second)
        mis = (omega * timing_error).to(ureg.milliradian)
        lateral = calc_lever_arm_error(distance, angle_error)
        return TimingAttitudeErrorResult(
            timing_travel=(speed * timing_error).to(_MM), misorientation=mis,
            timing_lateral=(distance * mis.to(ureg.radian).magnitude).to(_MM),
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
        return VisibilityRangeResult(visibility_range=calc_contrast_visibility_range(extinction, contrast_threshold))


class ThermalExpansionModel(ForwardModel):
    """Length change of a uniformly heated member, ``alpha L dT``."""
    requires = ("coefficient", "length", "temperature_rise")
    produces = ThermalExpansionResult

    def solve(self, coefficient: Quantity, length: Quantity, temperature_rise: Quantity) -> ThermalExpansionResult:
        return ThermalExpansionResult(expansion=calc_thermal_expansion(coefficient, length, temperature_rise))
