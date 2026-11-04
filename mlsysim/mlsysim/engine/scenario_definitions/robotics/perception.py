"""Physical-machine scenarios for perception."""

from __future__ import annotations
from typing import Any, Optional
from ....core.types import Quantity
from ..base import BaseScenario


class ObservationAgeScenario(BaseScenario):
    """``site.machine``'s navigation-camera observation age at ``speed``, with a
    ``batch_frames`` comparison.

    ``exposure`` replaces the camera's own exposure time (for example, a dim
    aisle that lengthens it); the other stages are unchanged (added 2026-10-20).

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.ObservationAgeModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.ObservationAgeModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.ObservationAgeModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseNavCameraAge
    >>> round(float(scenario.exposure_time.magnitude), 4)
    16.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    speed: Quantity
    batch_frames: int = 1
    exposure: Optional[Quantity] = None  # default: the navigation camera's exposure time

    @property
    def exposure_time(self) -> Quantity:
        """Camera exposure duration retaining its quantity units; use the explicit override or the
        camera registry value when no override is supplied.

        Equation: ``site.machine.nav_camera.exposure_time if exposure is None else exposure``."""
        return self.site.machine.nav_camera.exposure_time if self.exposure is None else self.exposure

    @property
    def result(self):
        """Return the modeled ObservationAge result from
        ``mlsysim.solvers.ObservationAgeModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import ObservationAgeModel

        s, cam = self.site, self.site.machine.nav_camera
        return ObservationAgeModel().solve(
            self.exposure_time,
            cam.readout_time,
            s.t_transport,
            s.t_dma,
            s.t_isp,
            s.t_backbone,
            s.t_ipc,
            cam.frame_rate,
            self.speed,
            self.batch_frames,
        )


class RollingShutterScenario(BaseScenario):
    """``site.machine``'s rolling-shutter navigation camera at ``speed``, turning at the site's
    yaw rate toward the rack end.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.RollingShutterSkewModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.RollingShutterSkewModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.RollingShutterSkewModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseNavCameraRollingShutter
    >>> round(float(scenario.result.row_interval.magnitude), 4)
    5.4605"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    speed: Quantity

    @property
    def result(self):
        """Return the modeled RollingShutter result from
        ``mlsysim.solvers.RollingShutterSkewModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import RollingShutterSkewModel

        s, cam = self.site, self.site.machine.nav_camera
        return RollingShutterSkewModel().solve(
            cam.resolution_height,
            cam.readout_time,
            cam.exposure_time,
            self.speed,
            s.yaw_rate_turn,
            s.d_clear,
            s.delta_loc,
            s.side_clearance,
        )


class TimingAttitudeErrorScenario(BaseScenario):
    """A ``timing_error`` and an ``angle_error`` on a target at ``distance``, priced against
    ``reference``.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.TimingAttitudeErrorModel.solve`` with the stored fields and any
    derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.TimingAttitudeErrorModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.TimingAttitudeErrorModel.solve`` with the stored fields and any
    derived properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseLidarExtrinsicShift
    >>> round(float(scenario.result.timing_travel.magnitude), 4)
    1.5"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    speed: Quantity
    yaw_rate: Quantity
    distance: Quantity
    timing_error: Quantity
    angle_error: Quantity
    reference: Optional[Quantity] = None

    @property
    def result(self):
        """Return the modeled TimingAttitudeError result from
        ``mlsysim.solvers.TimingAttitudeErrorModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import TimingAttitudeErrorModel

        return TimingAttitudeErrorModel().solve(
            self.speed, self.yaw_rate, self.distance, self.timing_error, self.angle_error, self.reference
        )


class LidarSweepAgeScenario(BaseScenario):
    """``site.machine``'s spinning lidar at ``speed``: sweep motion and the age of the aisle
    sector at the rack end.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.LidarSweepAgeModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.LidarSweepAgeModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.LidarSweepAgeModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseRackEndLidarAge
    >>> round(float(scenario.result.sweep_period.magnitude), 4)
    50.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    speed: Quantity

    @property
    def result(self):
        """Return the modeled LidarSweepAge result from
        ``mlsysim.solvers.LidarSweepAgeModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import LidarSweepAgeModel

        s, lidar = self.site, self.site.machine.lidar
        return LidarSweepAgeModel().solve(
            lidar.scan_rate,
            lidar.nominal_latency,
            lidar.time_sync_accuracy,
            s.aisle_width,
            s.d_clear,
            self.speed,
            s.yaw_rate_turn,
            s.perception_age,
        )


class StereoDepthErrorScenario(BaseScenario):
    """A stereo rig's depth error at two ranges and at ``site``'s rack end, against the site's
    clearances.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.StereoDepthErrorModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.StereoDepthErrorModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.StereoDepthErrorModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseStereoRigDepthError
    >>> round(float(scenario.result.sigma_near.magnitude), 4)
    12.5"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    baseline: Quantity
    focal_length_px: float
    disparity_sigma_px: float
    near_range: Quantity
    far_range: Quantity
    ray_clearance: Quantity
    baseline_change: Quantity
    bias_range: Quantity

    @property
    def result(self):
        """Return the modeled StereoDepthError result from
        ``mlsysim.solvers.StereoDepthErrorModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import StereoDepthErrorModel

        s = self.site
        return StereoDepthErrorModel().solve(
            self.baseline,
            self.focal_length_px,
            self.disparity_sigma_px,
            self.near_range,
            self.far_range,
            s.d_clear,
            self.ray_clearance,
            s.delta_loc,
            s.side_clearance,
            self.baseline_change,
            self.bias_range,
        )


class DelayPhaseMarginScenario(BaseScenario):
    """A loop with ``phase_margin`` at ``crossover``, closed through a pure ``delay``.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.DelayPhaseMarginModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.DelayPhaseMarginModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.DelayPhaseMarginModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.VisualServoDelayPhaseMargin
    >>> round(float(scenario.result.phase_margin.magnitude), 4)
    0.7854"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    phase_margin: Quantity
    crossover: Quantity
    delay: Quantity

    @property
    def result(self):
        """Return the modeled DelayPhaseMargin result from
        ``mlsysim.solvers.DelayPhaseMarginModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import DelayPhaseMarginModel

        return DelayPhaseMarginModel().solve(self.phase_margin, self.crossover, self.delay)


class ConformalInsetScenario(BaseScenario):
    """A split-conformal bound at miss rate ``alpha`` from ``calibration_count`` residuals.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.ConformalInsetModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.ConformalInsetModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.ConformalInsetModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ClearanceInsetConformalRank
    >>> round(float(scenario.result.alpha), 4)
    0.01"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    alpha: float
    calibration_count: int

    @property
    def result(self):
        """Return the modeled ConformalInset result from
        ``mlsysim.solvers.ConformalInsetModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import ConformalInsetModel

        return ConformalInsetModel().solve(self.alpha, self.calibration_count)


class ClockConversionScenario(BaseScenario):
    """A sensor clock drifting at ``drift_rate`` between syncs every ``sync_interval``,
    converted under ``site``'s bound.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.ClockConversionModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.ClockConversionModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.ClockConversionModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseLidarClockConversion
    >>> round(float(scenario.result.drift_per_interval.magnitude), 4)
    10.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    drift_rate: Quantity
    sync_interval: Quantity
    speed: Quantity

    @property
    def result(self):
        """Return the modeled ClockConversion result from
        ``mlsysim.solvers.ClockConversionModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import ClockConversionModel

        s = self.site
        return ClockConversionModel().solve(
            self.drift_rate, self.sync_interval, s.clock_conversion_bound, s.perception_age, self.speed
        )


class VisibilityRangeScenario(BaseScenario):
    """Fog of extinction coefficient ``extinction`` against a contrast threshold.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.VisibilityRangeModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.VisibilityRangeModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.VisibilityRangeModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.DenseFogVisibility
    >>> round(float(scenario.result.visibility_range.magnitude), 4)
    59.9146"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    extinction: Quantity
    contrast_threshold: float

    @property
    def result(self):
        """Return the modeled VisibilityRange result from
        ``mlsysim.solvers.VisibilityRangeModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import VisibilityRangeModel

        return VisibilityRangeModel().solve(self.extinction, self.contrast_threshold)


class ThermalExpansionScenario(BaseScenario):
    """A member of ``length`` and expansion ``coefficient`` warmed by ``temperature_rise``.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.ThermalExpansionModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.ThermalExpansionModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.ThermalExpansionModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.AluminumLinkWarming
    >>> round(float(scenario.result.expansion.magnitude), 4)
    0.231"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    coefficient: Quantity
    length: Quantity
    temperature_rise: Quantity

    @property
    def result(self):
        """Return the modeled ThermalExpansion result from
        ``mlsysim.solvers.ThermalExpansionModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import ThermalExpansionModel

        return ThermalExpansionModel().solve(self.coefficient, self.length, self.temperature_rise)
