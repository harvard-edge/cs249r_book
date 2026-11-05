"""Physical-machine scenarios for estimation."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from ....core.types import Quantity
from ..base import BaseScenario


class RangeFusionScenario(BaseScenario):
    """The range to ``site``'s rack-end upright, predicted over one lidar sweep and updated by
    the lidar, then ``stereo``; the stereo update misdated by ``age``'s arrival stamp at
    ``speed``.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.RangeFusionModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.RangeFusionModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes ``mlsysim.solvers.RangeFusionModel.solve``
    with the stored fields and any derived properties described below. Individual property
    contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseRackEndRangeFusion
    >>> round(float(scenario.result.process_sigma.magnitude), 4)
    5.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    stereo: Any  # StereoDepthErrorScenario
    age: Any  # ObservationAgeScenario
    prior_sigma: Quantity
    speed_sigma: Quantity
    comparison_scale: float = 2.0

    @property
    def result(self):
        """Return the modeled RangeFusion result from ``mlsysim.solvers.RangeFusionModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import RangeFusionModel

        lidar = self.site.machine.lidar
        return RangeFusionModel().solve(
            self.prior_sigma,
            self.speed_sigma,
            1 / lidar.scan_rate,
            lidar.range_precision_max,
            self.stereo.result.sigma_target,
            self.age.result.arrival_travel,
            self.comparison_scale,
        )


class EstimatorConsistencyScenario(BaseScenario):
    """A qualification campaign of ``runs`` scalar estimates, and single-update gates against
    ``fusion``'s misdated update.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.EstimatorConsistencyModel.solve`` with the stored fields and any
    derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.EstimatorConsistencyModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.EstimatorConsistencyModel.solve`` with the stored fields and any
    derived properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RackEndMisdatedUpdateConsistency
    >>> round(float(scenario.result.band_low), 4)
    0.6471"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fusion: Any  # RangeFusionScenario
    runs: int
    dimension: int = 1
    confidence: float = 0.95
    gate_confidences: List[float] = [0.95, 0.99]

    @property
    def result(self):
        """Return the modeled EstimatorConsistency result from
        ``mlsysim.solvers.EstimatorConsistencyModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import EstimatorConsistencyModel

        return EstimatorConsistencyModel().solve(
            self.runs, self.dimension, self.confidence, self.gate_confidences, self.fusion.result.nis_expected
        )


class ReplayBufferScenario(BaseScenario):
    """A rewind buffer at ``site.machine``'s IMU rate reaching back one observation age.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.ReplayBufferModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.ReplayBufferModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes ``mlsysim.solvers.ReplayBufferModel.solve``
    with the stored fields and any derived properties described below. Individual property
    contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseObservationReplayBuffer
    >>> round(float(scenario.result.sample_period.magnitude), 4)
    1.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario

    @property
    def result(self):
        """Return the modeled ReplayBuffer result from ``mlsysim.solvers.ReplayBufferModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import ReplayBufferModel

        return ReplayBufferModel().solve(self.site.perception_age, self.site.machine.imu.sample_rate)


class InertialCoastScenario(BaseScenario):
    """``site.machine``'s IMU coasting ``duration`` without a fix, seen at the rack-end
    distance.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.InertialCoastModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.InertialCoastModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.InertialCoastModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseImuOneSecondCoast
    >>> round(float(scenario.result.heading_noise.magnitude), 4)
    0.014"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    duration: Quantity

    @property
    def result(self):
        """Return the modeled InertialCoast result from
        ``mlsysim.solvers.InertialCoastModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import InertialCoastModel

        imu, s = self.site.machine.imu, self.site
        return InertialCoastModel().solve(
            imu.gyro_noise_density,
            imu.gyro_zero_rate_offset,
            imu.accel_noise_density,
            imu.accel_zero_g_offset,
            self.duration,
            s.d_clear,
            s.delta_loc,
        )


class OdometryDriftScenario(BaseScenario):
    """A square-path odometry benchmark's return errors, priced against ``site``'s localization
    bound at its aisle speed and length.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.OdometryDriftModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.OdometryDriftModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.OdometryDriftModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseUmbmarkOdometryDrift
    >>> round(float(scenario.path_length.magnitude), 4)
    16.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    square_side: Quantity
    legs: int
    return_error_uncalibrated: Quantity
    return_error_calibrated: Quantity
    run_sigma: Quantity
    test_speed: Quantity

    @property
    def path_length(self) -> Quantity:
        """Total driven path length, retaining the square-side units and counting every leg.

        Equation: ``legs * square_side``."""
        return self.legs * self.square_side

    @property
    def result(self):
        """Return the modeled OdometryDrift result from
        ``mlsysim.solvers.OdometryDriftModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import OdometryDriftModel

        s = self.site
        return OdometryDriftModel().solve(
            self.return_error_uncalibrated,
            self.return_error_calibrated,
            self.path_length,
            s.delta_loc,
            s.v_aisle,
            s.aisle_length,
            self.test_speed,
        )


class BeliefExpiryScenario(BaseScenario):
    """A belief with initial error, growth law, and task tolerance; optionally admitted at
    ``use_age`` under ``age_bound``.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.BeliefExpiryModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.BeliefExpiryModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes ``mlsysim.solvers.BeliefExpiryModel.solve``
    with the stored fields and any derived properties described below. Individual property
    contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ConveyorMugHandleBelief
    >>> round(float(scenario.result.initial_error.magnitude), 4)
    3.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    initial_error: Quantity
    growth_rate: Quantity
    tolerance: Quantity
    growth_acceleration: Optional[Quantity] = None
    use_age: Optional[Quantity] = None
    age_bound: Optional[Quantity] = None

    @property
    def result(self):
        """Return the modeled BeliefExpiry result from ``mlsysim.solvers.BeliefExpiryModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import BeliefExpiryModel

        return BeliefExpiryModel().solve(
            self.initial_error, self.growth_rate, self.tolerance, self.growth_acceleration, self.use_age, self.age_bound
        )


class StaleTransformScenario(BaseScenario):
    """``belief``'s cached camera edge composed with the current wrist pose, one
    ``site.machine`` wrist-camera frame old.

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.StaleTransformModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.StaleTransformModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.StaleTransformModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchSlotStaleWristTransform
    >>> round(float(scenario.result.interval.magnitude), 4)
    16.6667"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    belief: Any  # BeliefExpiryScenario
    lever: Quantity
    angular_rate: Quantity

    @property
    def result(self):
        """Return the modeled StaleTransform result from
        ``mlsysim.solvers.StaleTransformModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ...solvers import StaleTransformModel

        return StaleTransformModel().solve(
            self.belief.result, self.lever, self.angular_rate, 1 / self.site.machine.wrist_camera.frame_rate
        )


class FreeSpaceExpiryScenario(BaseScenario):
    """``site``'s rack end: free space ahead (clear distance) and abreast (side clearance) eroding at the walking speed.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Free-space evidence shrinks at constant human approach speed with fixed localization and margin allowances.

    Excludes: Occlusion changes, human acceleration, and a certified collision-avoidance controller.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``FreeSpaceExpiryModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseRackEndFreeSpace
    >>> value = case.result
    >>> print(f"{value.horizon_ahead:.3f~P}")
    593.750 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    uniform_timeout: Optional[Quantity] = None

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.FreeSpaceExpiryModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = FreeSpaceExpiryModel().solve(...)`` using the fields shown below."""
        from ...solvers import FreeSpaceExpiryModel

        s, rm = self.site, self.site.machine
        return FreeSpaceExpiryModel().solve(
            s.d_clear,
            s.side_clearance,
            s.delta_loc,
            s.delta_margin,
            s.v_human_approach,
            s.perception_age,
            1 / rm.nav_camera.frame_rate,
            rm.base.max_velocity,
            s.a_brake,
            s.tau_delay,
            self.uniform_timeout,
        )


class EpochGapScenario(BaseScenario):
    """Named acquisition ``epochs`` in one cycle; the ``pair`` to fuse is projected across its gap at ``speed``.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Sensor epochs use a common clock and displacement across their gap is speed times elapsed time.

    Excludes: Clock-conversion error, acceleration, and fusion uncertainty.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``EpochGapModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ArmSensorEpochGap
    >>> value = case.result
    >>> print(f"{value.travel:.3f~P}")
    4.250 mm"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    epochs: Dict[str, Quantity]
    pair: List[str]
    speed: Quantity

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.EpochGapModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = EpochGapModel().solve(...)`` using the fields shown below."""
        from ...solvers import EpochGapModel

        a, b = self.pair
        return EpochGapModel().solve(self.epochs[a], self.epochs[b], self.speed)


class SkewedRecordsScenario(BaseScenario):
    """Two records ``skew`` apart, compared at ``site``: the base at its drive limit and a person at the walking speed close on each other.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: The base and person close at their declared constant speed limits; spatial margins add.

    Excludes: Acceleration, directional geometry, and probabilistic occupancy.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``EpochGapModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseSkewedRecordPair
    >>> value = case.result
    >>> print(f"{value.travel:.3f~P}")
    62.000 mm"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    skew: Quantity
    separation: Quantity

    @property
    def closing_speed(self) -> Quantity:
        """Closing speed in length/time, adding base and human approach speeds.

        Equation:
            return self.site.machine.base.max_velocity + self.site.v_human_approach"""
        return self.site.machine.base.max_velocity + self.site.v_human_approach

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.EpochGapModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = EpochGapModel().solve(...)`` using the fields shown below."""
        from ...solvers import EpochGapModel

        s = self.site
        return EpochGapModel().solve(
            0 * self.skew, self.skew, self.closing_speed, self.separation, (s.delta_loc, s.delta_margin)
        )


class SensorBiasDriftScenario(BaseScenario):
    """A steady ``reading`` whose sensor bias drifts at ``drift_rate`` for ``span``, while the estimator reports ``reported_sigma``.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Bias drifts at a constant supplied rate for the specified span.

    Excludes: A full estimator update; reported_sigma and excitation do not enter result.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``SensorBiasDriftModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WristForceBiasDrift
    >>> value = case.result
    >>> print(f"{value.shift:.3f~P}")
    3.000 N"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    reading: Quantity
    reported_sigma: Quantity
    drift_rate: Quantity
    span: Quantity
    excitation: Optional[Quantity] = None  # a deliberate contact break that would expose the bias

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.SensorBiasDriftModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = SensorBiasDriftModel().solve(...)`` using the fields shown below."""
        from ...solvers import SensorBiasDriftModel

        return SensorBiasDriftModel().solve(self.reading, self.drift_rate, self.span)


class FrozenTransformScenario(BaseScenario):
    """``site``'s conveyor object tracked by a camera lost at ``loss_time``; its last transform republished every ``republish_period``.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Conveyor speed is constant and republishing changes timestamp freshness without adding observations.

    Excludes: Prediction correction, transport jitter, and loss_time as a computational offset.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``FrozenTransformModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ConveyorMugFrozenTransform
    >>> value = case.result
    >>> print(f"{value.error:.3f~P}")
    20.000 mm"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    republish_period: Quantity
    freshness_threshold: Quantity
    loss_time: Quantity
    window: Quantity

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.FrozenTransformModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = FrozenTransformModel().solve(...)`` using the fields shown below."""
        from ...solvers import FrozenTransformModel

        s = self.site
        return FrozenTransformModel().solve(
            s.v_conveyor, self.window, s.grasp_tolerance, self.republish_period, self.freshness_threshold
        )


class ClearanceQueryScenario(BaseScenario):
    """Sphere-hull clearance queries on ``site.machine``'s permission controller, and a snapshot of the stopping corridor.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: The corridor is a rectangular slab and declared hull/grid parameters define the query workload.

    Excludes: Measured cache locality, worst-case bus arbitration, and safety certification.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``ClearanceQueryModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehousePermissionClearanceQuery
    >>> value = case.result
    >>> print(f"{value.query_time.m_as('microsecond'):.3f} microsecond")
    2.000 microsecond"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    collision_allocation: Quantity
    hulls: int
    fewest_hulls: int
    hull_radius: Quantity
    esdf_cell_bytes: Quantity
    occupancy_cell_bytes: Quantity
    map_edge: Quantity
    slab_height: Quantity
    pitches: List[Quantity]
    trilinear_reads: int

    @property
    def slab_volume(self) -> Quantity:
        """Rectangular stopping-corridor volume in cubic meters.

        Equation:
            return (self.site.aisle_width * self.site.d_clear * self.slab_height).to('``m**3``')"""
        return (self.site.aisle_width * self.site.d_clear * self.slab_height).to("m**3")

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.ClearanceQueryModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = ClearanceQueryModel().solve(...)`` using the fields shown below."""
        from ...solvers import ClearanceQueryModel

        s, rm = self.site, self.site.machine
        return ClearanceQueryModel().solve(
            rm.permission_rate,
            self.collision_allocation,
            self.hulls,
            self.hull_radius,
            s.delta_margin,
            rm.permission_mcu.memory.sram_bandwidth,
            self.esdf_cell_bytes,
            self.occupancy_cell_bytes,
            self.map_edge,
            self.slab_volume,
            self.pitches,
            self.trilinear_reads,
            s.enforcer_wcet,
        )


class HostMapSizingScenario(BaseScenario):
    """A workcell of ``extent`` mapped at ``pitch``: dense grid against a hashed band of ``band_half_width`` around ``surface_area``.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Extent, grid pitch, voxel size, and hashed surface band are supplied geometry.

    Excludes: Compression, hash-table metadata beyond the solver accounting, and surface reconstruction quality.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``HostMapSizingModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WorkcellHostMapSizing
    >>> value = case.result
    >>> print(f"{value.hashed_bytes:.3f~P}")
    76.800 MB"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    extent: List[Quantity]
    pitch: Quantity
    voxel_bytes: Quantity
    surface_area: Quantity
    band_half_width: Quantity
    block_edge: int

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.HostMapSizingModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = HostMapSizingModel().solve(...)`` using the fields shown below."""
        from ...solvers import HostMapSizingModel

        return HostMapSizingModel().solve(
            self.extent, self.pitch, self.voxel_bytes, self.surface_area, self.band_half_width, self.block_edge
        )


class RaymarchTrafficScenario(BaseScenario):
    """One depth stream ray-marched into a map on ``site.machine``'s host SoC.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Every depth pixel follows the same ray length and step pitch at a fixed frame rate.

    Excludes: Early termination, sparsity, cache reuse, and competing host traffic.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``RaymarchTrafficModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseDepthRaymarchTraffic
    >>> value = case.result
    >>> print(f"{value.traffic:.3f~P}")
    7.373 GB/s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    width_px: int
    height_px: int
    frame_rate: Quantity
    ray_length: Quantity
    pitch: Quantity
    bytes_per_step: Quantity

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.RaymarchTrafficModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = RaymarchTrafficModel().solve(...)`` using the fields shown below."""
        from ...solvers import RaymarchTrafficModel

        return RaymarchTrafficModel().solve(
            self.width_px,
            self.height_px,
            self.frame_rate,
            self.ray_length,
            self.pitch,
            self.bytes_per_step,
            self.site.machine.brain_soc.memory.bandwidth,
        )


class EncoderRateScenario(BaseScenario):
    """An encoder of ``flop_per_frame`` on every ``site.machine`` navigation frame, against the host SoC's dense FP16 peak.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Every camera frame incurs the supplied encoder FLOPs at the registered dense FP16 peak.

    Excludes: Memory limits, effective kernel efficiency, and keyframe-rate fields in this result.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``EncoderRateModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseFoundationEncoderAtCameraRate
    >>> value = case.result
    >>> print(f"{value.required:.3f~P}")
    180.000 TFLOPs/s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario
    flop_per_frame: Quantity
    keyframe_rate_low: Optional[Quantity] = None
    keyframe_rate_high: Optional[Quantity] = None

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.EncoderRateModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = EncoderRateModel().solve(...)`` using the fields shown below."""
        from ...solvers import EncoderRateModel

        rm = self.site.machine
        return EncoderRateModel().solve(
            self.flop_per_frame, rm.nav_camera.frame_rate, rm.brain_soc.compute.precision_flops["fp16"]
        )
