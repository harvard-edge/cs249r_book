"""Physical-machine scenarios for plant."""

from __future__ import annotations
from typing import Any, List, Optional, Tuple
from ....core.types import Quantity
from ..base import BaseScenario


class FreshnessProfileScenario(BaseScenario):
    """A target moving at ``site``'s conveyor speed, grasped through a staged capture-to-onset
    pipeline.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.FreshnessProfileModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.FreshnessProfileModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.FreshnessProfileModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.FreshnessProfileModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WristCameraGraspAgeProfile
    >>> type(case.result).__name__
    'FreshnessProfileResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario (conveyor speed, tolerance, initial error)
    stages: List[Tuple[str, Quantity, Quantity, Quantity]]  # (stage, P50, P99, max)
    focus_stage: str  # the stage a benchmark reports
    system_p99: Quantity
    system_p999: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.FreshnessProfileModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ...solvers import FreshnessProfileModel

        return FreshnessProfileModel().solve(
            self.site.v_conveyor,
            self.site.grasp_tolerance,
            self.site.grasp_initial_error,
            self.stages,
            self.focus_stage,
            self.system_p999,
        )


class BrakingCeilingsScenario(BaseScenario):
    """The ceilings on ``site``'s credible deceleration for its loaded machine at the drive
    limit.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.BrakingCeilingsModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.BrakingCeilingsModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.BrakingCeilingsModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.BrakingCeilingsModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseBaseBrakingCeilings
    >>> type(case.result).__name__
    'BrakingCeilingsResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.BrakingCeilingsModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...solvers import BrakingCeilingsModel

        rm = self.site.machine
        return BrakingCeilingsModel().solve(
            rm.loaded_mass,
            self.site.a_brake,
            rm.base.max_velocity,
            rm.drive_unit,
            rm.drive_wheel_count,
            rm.drive_current_limit,
            rm.drive_wheel_load_share_max,
            self.site.mu_dry,
            self.site.mu_inspected_floor,
            rm.drive_bus_battery_low,
            rm.cg_height,
            rm.tip_arm_stowed,
        )


class HubDriveElectricalScenario(BaseScenario):
    """Current rise and phase-short braking of ``site.machine``'s hub drives on a depleted bus.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.HubDriveElectricalModel``.

    Assumptions: The stated fields are inputs to
    ``mlsysim.solvers.HubDriveElectricalModel``; its documented validity regime, units, and
    modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.HubDriveElectricalModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.HubDriveElectricalModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseBaseHubDriveElectrical
    >>> type(case.result).__name__
    'HubDriveElectricalResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.HubDriveElectricalModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ...solvers import HubDriveElectricalModel

        rm = self.site.machine
        return HubDriveElectricalModel().solve(
            rm.drive_unit, rm.drive_current_limit, rm.drive_bus_battery_low, self.site.t_brake_onset
        )


class GearedRotorScenario(BaseScenario):
    """A rotor seen through a ``gear_ratio`` reduction, beside a link of ``load_inertia``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.GearedRotorModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.GearedRotorModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.GearedRotorModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.GearedRotorModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.GearedRotorFlywheel
    >>> type(case.result).__name__
    'GearedRotorResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    gear_ratio: float
    rotor_mass: Quantity
    rotor_inertia: Quantity
    flywheel_radius: Quantity  # radius of the comparison solid flywheel
    load_inertia: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.GearedRotorModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...solvers import GearedRotorModel

        return GearedRotorModel().solve(
            self.gear_ratio, self.rotor_mass, self.rotor_inertia, self.flywheel_radius, self.load_inertia
        )


class GearRatioCostScenario(BaseScenario):
    """``motor`` in ``joint`` holding ``output_torque`` through each of ``ratios``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.GearRatioCostModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.GearRatioCostModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.GearRatioCostModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.GearRatioCostModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.PandaClassJointGearRatios
    >>> type(case.result).__name__
    'GearRatioCostResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    joint: Any  # ActuatorJoint (loss resistance, input inertia, gear ratio)
    motor: Any  # FramelessMotor
    output_torque: Quantity
    gear_efficiency: float
    ratios: List[int]

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.GearRatioCostModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...solvers import GearRatioCostModel

        return GearRatioCostModel().solve(
            self.motor,
            self.joint.internal_resistance,
            self.joint.rotor_inertia,
            self.output_torque,
            self.gear_efficiency,
            self.ratios,
        )


class ThermalBurstScenario(BaseScenario):
    """A ``burst_factor`` times rated-current burst in ``joint``, first-order lumped, fixed
    resistance.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.motors.calc_first_order_temperature``,
    ``mlsysim.solvers.ThermalBurstModel``.

    Assumptions: The winding is a first-order lumped thermal body with fixed resistance
    during each burst.

    Excludes: Spatial gradients, resistance-temperature feedback, and unmodeled cooling
    changes are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.motors.calc_first_order_temperature``,
    ``mlsysim.solvers.ThermalBurstModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.PandaClassJointBurst
    >>> type(case.result).__name__
    'ThermalBurstResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    joint: Any  # ActuatorJoint (loss resistance)
    motor: Any  # FramelessMotor (rated current)
    thermal_capacitance: Quantity
    thermal_resistance: Quantity
    ambient: Quantity
    limit: Quantity  # insulation rating
    ceiling: Quantity  # working ceiling below the rating
    burst_factor: float
    burst_on: Quantity

    def temperature(self, elapsed: Quantity, start: Quantity) -> Quantity:
        """Winding temperature after an elapsed constant-current burst from ``start``. Equation:
        ``mlsysim.physics.motors.calc_first_order_temperature`` using the result's asymptote and
        time constant. Units: temperature; resistance and lumped thermal parameters remain
        fixed.
        """
        from ....physics.motors import calc_first_order_temperature

        r = self.result
        return calc_first_order_temperature(elapsed, start, r.asymptote, r.time_constant)

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.ThermalBurstModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...solvers import ThermalBurstModel

        return ThermalBurstModel().solve(
            self.motor.rated_current,
            self.joint.internal_resistance,
            self.thermal_capacitance,
            self.thermal_resistance,
            self.ambient,
            self.limit,
            self.ceiling,
            self.burst_factor,
            self.burst_on,
        )


class StallDeratingScenario(BaseScenario):
    """``joint`` holding ``hold_torque`` at stall with copper's resistance rising with
    temperature.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.StallDeratingModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.StallDeratingModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.StallDeratingModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.StallDeratingModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.PandaClassElbowStall
    >>> type(case.result).__name__
    'StallDeratingResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    joint: Any  # ActuatorJoint (gear ratio, loss resistance)
    motor: Any  # FramelessMotor (torque constant)
    hold_torque: Quantity
    gear_efficiency: float
    thermal_resistance: Quantity
    ambient: Quantity
    limit: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.StallDeratingModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...solvers import StallDeratingModel

        return StallDeratingModel().solve(
            self.hold_torque,
            self.joint.gear_ratio,
            self.gear_efficiency,
            self.motor.torque_constant,
            self.joint.internal_resistance,
            self.thermal_resistance,
            self.ambient,
            self.limit,
        )


class ContinuousDeratingScenario(BaseScenario):
    """A catalog continuous rating carried to a hotter ambient and a worse heat path.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.ContinuousDeratingModel``.

    Assumptions: The stated fields are inputs to
    ``mlsysim.solvers.ContinuousDeratingModel``; its documented validity regime, units, and
    modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.ContinuousDeratingModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.ContinuousDeratingModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SealedJointCatalogDerating
    >>> type(case.result).__name__
    'ContinuousDeratingResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    ceiling: Quantity
    catalog_ambient: Quantity
    installed_ambient: Quantity
    thermal_resistance_factor: float  # installed theta relative to the catalog's

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.ContinuousDeratingModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ...solvers import ContinuousDeratingModel

        return ContinuousDeratingModel().solve(
            self.ceiling, self.catalog_ambient, self.installed_ambient, self.thermal_resistance_factor
        )


class DriveBusScenario(BaseScenario):
    """``site.machine``'s shared drive bus: a coincident load step, and a stop from the drive
    limit into it.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.DriveBusModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.DriveBusModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.DriveBusModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.DriveBusModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseDriveBusStop
    >>> type(case.result).__name__
    'DriveBusResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario (load step, deceleration)
    link_capacitance: Quantity
    regen_fraction: float  # share of kinetic energy returned to the bus
    candidate_resistor: Quantity
    candidate_power: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.DriveBusModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...solvers import DriveBusModel

        rm = self.site.machine
        return DriveBusModel().solve(
            rm.loaded_mass,
            rm.base.max_velocity,
            self.site.a_brake,
            self.site.i_rail_transient,
            rm.drive_bus_resistance,
            rm.drive_bus_full_charge,
            rm.drive_bus_nominal,
            rm.drive_bus_battery_low,
            rm.drive_bus_uvlo,
            rm.brain_soc.tdp,
            rm.drive_bus_converter_efficiency,
            self.link_capacitance,
            rm.drive_bus_clamp,
            self.regen_fraction,
        )


class ZeroFailureScenario(BaseScenario):
    """``trials`` failure-free trials, bounded at one-sided ``confidence``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.ZeroFailureModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.ZeroFailureModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.ZeroFailureModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.ZeroFailureModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.CleanLoadedStopCampaign
    >>> type(case.result).__name__
    'ZeroFailureResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    trials: int
    confidence: float
    target_rate: Optional[float] = None  # a per-trial rate the campaign is meant to support

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.ZeroFailureModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...solvers import ZeroFailureModel

        return ZeroFailureModel().solve(self.trials, self.confidence, self.target_rate)


class ThrottledStageScenario(BaseScenario):
    """``site``'s vision backbone at ``clock_ratio`` of full clock, paid as travel at
    ``speed``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.ThrottledStageModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.ThrottledStageModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.ThrottledStageModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.ThrottledStageModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ThrottledNavigationBackbone
    >>> type(case.result).__name__
    'ThrottledStageResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario (backbone time)
    clock_ratio: float
    speed: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.ThrottledStageModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...solvers import ThrottledStageModel

        return ThrottledStageModel().solve(self.site.t_backbone, self.clock_ratio, self.speed)
