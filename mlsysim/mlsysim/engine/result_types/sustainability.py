"""Typed result models for sustainability calculations."""

from __future__ import annotations

from typing import Dict, List, Optional
from ...core.types import Quantity
from .base import SolverResult

__all__ = [
    "SustainabilityResult",
    "PowerTransientResult",
    "CoolantFlowResult",
    "OutageCostResult",
    "FacilityPowerResult",
    "WaterUseResult",
    "RunFootprintResult",
    "PueRetrofitResult",
    "GridPlacementResult",
    "EmbodiedPaybackResult",
    "ServiceLifeExtensionResult",
    "RepeatedWorkEnergyResult",
    "ServingCrossoverResult",
    "CascadeEnergyResult",
    "DutyCycledPowerResult",
    "DataMovementEnergyResult",
    "DemandReboundResult",
    "DevelopmentSweepResult",
    "CarbonPricedPlacementResult",
    "TemporalShiftResult",
]


class SustainabilityResult(SolverResult):
    """Result from SustainabilityModel: energy, carbon, and water footprint."""

    __module__ = "mlsysim.engine.results"

    it_energy_kwh: Quantity
    total_energy_kwh: Quantity
    carbon_footprint_kg: float
    water_usage_liters: float
    pue: float
    region_name: str
    embodied_carbon_kg: float = 0.0

    @property
    def it_energy(self) -> Quantity:
        """Return it energy before facility overhead with units.

        Returns
        -------
        Quantity
            IT energy before facility overhead as a Pint quantity, conventionally in kWh.

        Notes
        -----
        Equation: it_energy = it_energy_kwh, with no arithmetic or unit conversion.

        Assumptions: The producing solver supplied it_energy_kwh as a time, energy, or
        memory quantity appropriate to this result.

        Excludes: Re-evaluation of the model and any new feasibility or unit-dimension
        check. The original quantity and its unit are returned unchanged.

        References
        ----------
        Derivation: This compatibility property returns the existing it_energy_kwh field
        directly; its value is defined by that field rather than a separate formula.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import SustainabilityResult
        >>> r = SustainabilityResult(it_energy_kwh=Q_(1, "kWh"),
        ...     total_energy_kwh=Q_(2, "kWh"), carbon_footprint_kg=3,
        ...     water_usage_liters=4, pue=2, region_name="example", embodied_carbon_kg=1)
        >>> r.it_energy is r.it_energy_kwh
        True
        """
        return self.it_energy_kwh

    @property
    def total_energy(self) -> Quantity:
        """Return facility energy after pue overhead with units.

        Returns
        -------
        Quantity
            Facility energy after PUE overhead as a Pint quantity, conventionally in kWh.

        Notes
        -----
        Equation: total_energy = total_energy_kwh, with no arithmetic or unit conversion.

        Assumptions: The producing solver supplied total_energy_kwh as a time, energy, or
        memory quantity appropriate to this result.

        Excludes: Re-evaluation of the model and any new feasibility or unit-dimension
        check. The original quantity and its unit are returned unchanged.

        References
        ----------
        Derivation: This compatibility property returns the existing total_energy_kwh field
        directly; its value is defined by that field rather than a separate formula.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import SustainabilityResult
        >>> r = SustainabilityResult(it_energy_kwh=Q_(1, "kWh"),
        ...     total_energy_kwh=Q_(2, "kWh"), carbon_footprint_kg=3,
        ...     water_usage_liters=4, pue=2, region_name="example", embodied_carbon_kg=1)
        >>> r.total_energy is r.total_energy_kwh
        True
        """
        return self.total_energy_kwh

    @property
    def carbon_footprint(self) -> Quantity:
        """Return stored carbon footprint with units.

        Returns
        -------
        Quantity
            Stored carbon footprint as a Pint quantity, conventionally in kg CO2-equivalent. The
            attached mass unit does not distinguish greenhouse gases.

        Notes
        -----
        Equation: carbon_footprint = Q_(carbon_footprint_kg, "kg").

        Assumptions: The stored scalar carbon_footprint_kg is expressed in kg CO2-equivalent
        and retains the accounting boundary chosen by the producing solver.

        Excludes: Recomputing the underlying footprint or checking consistency with the
        other result fields.

        References
        ----------
        Derivation: This accessor attaches the documented unit kg to the stored scalar
        carbon_footprint_kg; it performs no additional physical calculation.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import SustainabilityResult
        >>> r = SustainabilityResult(it_energy_kwh=Q_(1, "kWh"),
        ...     total_energy_kwh=Q_(2, "kWh"), carbon_footprint_kg=3,
        ...     water_usage_liters=4, pue=2, region_name="example", embodied_carbon_kg=1)
        >>> r.carbon_footprint == Q_(r.carbon_footprint_kg, "kg")
        True
        """
        from ...core.units import Q_

        return Q_(self.carbon_footprint_kg, "kg")

    @property
    def water_usage(self) -> Quantity:
        """Return stored cooling-water use with units.

        Returns
        -------
        Quantity
            Stored cooling-water use as a Pint quantity, conventionally in liters.

        Notes
        -----
        Equation: water_usage = Q_(water_usage_liters, "liter").

        Assumptions: The stored scalar water_usage_liters is expressed in liters and retains
        the accounting boundary chosen by the producing solver.

        Excludes: Recomputing the underlying footprint or checking consistency with the
        other result fields.

        References
        ----------
        Derivation: This accessor attaches the documented unit liter to the stored scalar
        water_usage_liters; it performs no additional physical calculation.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import SustainabilityResult
        >>> r = SustainabilityResult(it_energy_kwh=Q_(1, "kWh"),
        ...     total_energy_kwh=Q_(2, "kWh"), carbon_footprint_kg=3,
        ...     water_usage_liters=4, pue=2, region_name="example", embodied_carbon_kg=1)
        >>> r.water_usage == Q_(r.water_usage_liters, "liter")
        True
        """
        from ...core.units import Q_

        return Q_(self.water_usage_liters, "liter")

    @property
    def embodied_carbon(self) -> Quantity:
        """Return manufacturing carbon footprint with units.

        Returns
        -------
        Quantity
            Manufacturing carbon footprint as a Pint quantity, conventionally in kg
            CO2-equivalent.

        Notes
        -----
        Equation: embodied_carbon = Q_(embodied_carbon_kg, "kg").

        Assumptions: The stored scalar embodied_carbon_kg is expressed in kg CO2-equivalent
        and retains the accounting boundary chosen by the producing solver.

        Excludes: Recomputing the underlying footprint or checking consistency with the
        other result fields.

        References
        ----------
        Derivation: This accessor attaches the documented unit kg to the stored scalar
        embodied_carbon_kg; it performs no additional physical calculation.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import SustainabilityResult
        >>> r = SustainabilityResult(it_energy_kwh=Q_(1, "kWh"),
        ...     total_energy_kwh=Q_(2, "kWh"), carbon_footprint_kg=3,
        ...     water_usage_liters=4, pue=2, region_name="example", embodied_carbon_kg=1)
        >>> r.embodied_carbon == Q_(r.embodied_carbon_kg, "kg")
        True
        """
        from ...core.units import Q_

        return Q_(self.embodied_carbon_kg, "kg")


class PowerTransientResult(SolverResult):
    """Result from PowerTransientModel: one device's load step and a cluster's ramp."""

    __module__ = "mlsysim.engine.results"

    power_step: Quantity
    current_step: Quantity
    droop: Dict[str, Quantity]  # keyed by rise time label
    droop_share: Dict[str, float]
    cluster_power_step: Quantity
    ramp_rate: Quantity


class CoolantFlowResult(SolverResult):
    """Result from CoolantFlowModel: coolant flow that carries away a heat load."""

    __module__ = "mlsysim.engine.results"

    mass_flow: Quantity
    volume_flow: Quantity


class OutageCostResult(SolverResult):
    """Result from OutageCostModel: idle cost of a failure domain going down."""

    __module__ = "mlsysim.engine.results"

    direct_cost: float  # dollars, failed devices only
    job_cost: float  # dollars, a synchronous job idles every device
    job_cost_with_rollback: Dict[str, float]


class FacilityPowerResult(SolverResult):
    """Result from FacilityPowerModel: accelerator, IT, and facility power."""

    __module__ = "mlsysim.engine.results"

    accelerator_power: Quantity
    it_power: Quantity
    facility_power: Quantity
    support_power: Quantity  # IT minus accelerators (hosts, switches, conversion)
    overhead_power: Quantity  # facility minus IT (PUE overhead)


class WaterUseResult(SolverResult):
    """Result from WaterUseModel: water consumed at each named WUE."""

    __module__ = "mlsysim.engine.results"

    water: Dict[str, Quantity]


class RunFootprintResult(SolverResult):
    """Result from RunFootprintModel: one run's energy, carbon, embodied carbon, and water."""

    __module__ = "mlsysim.engine.results"

    power: FacilityPowerResult
    duration: Quantity  # device-hours / devices (every device running throughout)
    accelerator_energy: Quantity
    it_energy: Quantity
    facility_energy: Quantity
    carbon: Dict[str, Quantity]  # operational, per named grid
    embodied_manufactured: Quantity  # every device, at manufacture
    embodied_allocated: Quantity  # the run's time share of a service life
    embodied_share: Dict[str, float]  # allocated / (operational + allocated), per grid
    water: Dict[str, Quantity]  # per named WUE


class PueRetrofitResult(SolverResult):
    """Result from PueRetrofitModel: facility power before and after, and what the change saves."""

    __module__ = "mlsysim.engine.results"

    facility_power_before: Quantity
    facility_power_after: Quantity
    saved_power: Quantity
    saved_energy: Quantity
    saved_cost: Quantity


class GridPlacementResult(SolverResult):
    """Result from GridPlacementModel: emissions of one energy on each grid."""

    __module__ = "mlsysim.engine.results"

    emissions: Dict[str, Quantity]
    emissions_after_cut: Dict[str, Quantity]
    lowest: str
    highest: str
    spread: float  # highest / lowest intensity


class EmbodiedPaybackResult(SolverResult):
    """Result from EmbodiedPaybackModel: carbon a replacement avoids per day and its payback time."""

    __module__ = "mlsysim.engine.results"

    saved_power: Quantity
    saved_energy_per_day: Quantity
    avoided_per_day: Dict[str, Quantity]
    payback: Dict[str, Quantity]


class ServiceLifeExtensionResult(SolverResult):
    """Result from ServiceLifeExtensionModel: cut in embodied carbon charged per year."""

    __module__ = "mlsysim.engine.results"

    annual_charge_reduction: float


class RepeatedWorkEnergyResult(SolverResult):
    """Result from RepeatedWorkEnergyModel: energy repeated per named rollback."""

    __module__ = "mlsysim.engine.results"

    energy: Dict[str, Quantity]


class ServingCrossoverResult(SolverResult):
    """Result from ServingCrossoverModel: daily serving energy and the day it passes a one-time energy."""

    __module__ = "mlsysim.engine.results"

    daily_energy: Quantity
    crossover_days: float


class CascadeEnergyResult(SolverResult):
    """Result from CascadeEnergyModel: mean energy per input of an edge-first cascade."""

    __module__ = "mlsysim.engine.results"

    escalated_energy: Quantity
    mean_energy: Quantity
    cloud_only_energy: Quantity
    saving: float


class DutyCycledPowerResult(SolverResult):
    """Result from DutyCycledPowerModel: each stage's average contribution and the total."""

    __module__ = "mlsysim.engine.results"

    contributions: Dict[str, Quantity]
    average_power: Quantity
    reduction: float  # reference power / average power


class DataMovementEnergyResult(SolverResult):
    """Result from DataMovementEnergyModel: arithmetic against data-movement energy per weight."""

    __module__ = "mlsysim.engine.results"

    compute_energy: Quantity
    movement_energy: Quantity
    movement_share: float
    movement_over_compute: float
    intensity: float  # FLOP/byte of the access pattern
    crossover_intensity: float  # FLOP/byte at which the two energies match
    balance_batch: float
    operand_energy: Quantity
    operand_over_flop: float


class DemandReboundResult(SolverResult):
    """Result from DemandReboundModel: fleet energy multiple after efficiency and volume change."""

    __module__ = "mlsysim.engine.results"

    net_factor: float


class DevelopmentSweepResult(SolverResult):
    """Result from DevelopmentSweepModel: final-run equivalents a sweep consumes."""

    __module__ = "mlsysim.engine.results"

    final_run_equivalents: float


class CarbonPricedPlacementResult(SolverResult):
    """Result from CarbonPricedPlacementModel: region ranking with and without a carbon price."""

    __module__ = "mlsysim.engine.results"

    duration_days: float
    best_region: str
    best_region_untaxed: str
    objective_value: float
    objective: Dict[str, float]
    objective_untaxed: Dict[str, float]


class TemporalShiftResult(SolverResult):
    """Result from TemporalShiftModel: best deadline-feasible start and the carbon it avoids."""

    __module__ = "mlsysim.engine.results"

    immediate_emissions: Quantity  # starting at release; hypothetical if immediate_feasible is False
    immediate_feasible: bool  # start-now has capacity for the whole run
    best_start: Quantity  # wait from release to the chosen start
    expected_completion: Quantity  # best_start + duration
    best_run_emissions: Quantity  # run emissions at the chosen start
    shift_overhead_emissions: Quantity  # carbon caused by waiting
    best_net_emissions: Quantity  # run plus overhead at the chosen start
    net_saving: Quantity  # immediate minus best net; negative only if capacity forces it
    saving_fraction: Optional[float]  # net_saving / immediate_emissions; None if immediate is zero
    shifted: bool  # best_start > 0
    candidate_starts: int  # starts that meet the deadline and have capacity
    net_emissions_by_start: List[Optional[Quantity]]  # indexed by wait in steps; None if infeasible
    min_beneficial_wait: Optional[Quantity]  # shortest wait that saves carbon
