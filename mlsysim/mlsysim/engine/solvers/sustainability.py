"""Sustainability solvers: facility power, run footprints, carbon placement, and demand.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic
stays reviewable. Added 2026-10-20.

Every model here works on a rated-power (TDP) basis unless a caller passes a
measured power: these are upper-bound planning accounts, not meter readings.
The power chain is accelerator power, times the rack's IT-to-accelerator ratio
(hosts, switches, conversion inside the IT boundary), times PUE (facility
overhead, applied once at the facility boundary). Carbon is facility energy
times a grid carbon intensity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..results import (
    CarbonPricedPlacementResult,
    CascadeEnergyResult,
    DataMovementEnergyResult,
    DemandReboundResult,
    DevelopmentSweepResult,
    DutyCycledPowerResult,
    EmbodiedPaybackResult,
    FacilityPowerResult,
    GridPlacementResult,
    PueRetrofitResult,
    RepeatedWorkEnergyResult,
    RunFootprintResult,
    ServiceLifeExtensionResult,
    ServingCrossoverResult,
    WaterUseResult,
)
from ...core.types import Quantity
from ...core.units import Q_ as _Q
from ...physics.quantities import carbon_from_energy
from .base import ForwardModel


_DAY = _Q(1, "day")


def _ratio(a, b) -> float:
    return (a / b).to("").magnitude


class FacilityPowerModel(ForwardModel):
    """
    The power chain from accelerators to the facility meter.

    Accelerator power is ``devices x device_power``; the IT load multiplies it
    by the rack's IT-to-accelerator ratio (hosts, switches, network adapters,
    and conversion losses inside the IT boundary); the facility draws the IT
    load times PUE. The facility figure is what the grid connection must
    supply at every instant.
    """
    requires = ()
    produces = FacilityPowerResult

    def solve(self, devices: float, device_power: Quantity, it_ratio: float, pue: float) -> FacilityPowerResult:
        acc = (devices * device_power).to("MW")
        it = (acc * it_ratio).to("MW")
        fac = (it * pue).to("MW")
        return FacilityPowerResult(accelerator_power=acc, it_power=it, facility_power=fac,
                                   support_power=(it - acc).to("MW"), overhead_power=(fac - it).to("MW"))


class WaterUseModel(ForwardModel):
    """Water consumed for an IT energy at a water usage effectiveness: ``W = E_IT x WUE``."""
    requires = ()
    produces = WaterUseResult

    def solve(self, it_energy: Quantity, wue: Dict[str, Quantity]) -> WaterUseResult:
        return WaterUseResult(water={k: (it_energy * w).to("liter") for k, w in wue.items()})


class RunFootprintModel(ForwardModel):
    """
    One training run's energy, operational carbon, embodied carbon, and water.

    Energy follows the power chain over the run's device-hours: accelerator
    energy ``device_hours x device_power``, IT energy times the rack ratio,
    facility energy times PUE. Operational carbon applies each named grid's
    intensity to facility energy. Embodied carbon is ``devices x
    embodied_per_device`` at manufacture; the run is charged its time share,
    ``device_hours x embodied_per_device / service_life``. Water applies each
    WUE to IT energy (WUE's denominator is IT energy).
    """
    requires = ()
    produces = RunFootprintResult

    def solve(self, device_hours: Quantity, devices: float, device_power: Quantity, it_ratio: float,
              pue: float, intensities: Dict[str, Quantity], embodied_per_device: Quantity,
              service_life: Quantity, wue: Optional[Dict[str, Quantity]] = None) -> RunFootprintResult:
        power = FacilityPowerModel().solve(devices, device_power, it_ratio, pue)
        acc_e = (device_hours * device_power).to("MWh")
        it_e = (acc_e * it_ratio).to("MWh")
        fac_e = (it_e * pue).to("MWh")
        carbon = {k: carbon_from_energy(fac_e, g).to("metric_ton") for k, g in intensities.items()}
        made = (devices * embodied_per_device).to("metric_ton")
        allocated = (device_hours * embodied_per_device / service_life).to("metric_ton")
        share = {k: _ratio(allocated, c + allocated) for k, c in carbon.items()}
        water = WaterUseModel().solve(it_e, wue).water if wue else {}
        return RunFootprintResult(
            power=power, duration=(device_hours / devices).to("hour"),
            accelerator_energy=acc_e, it_energy=it_e, facility_energy=fac_e,
            carbon=carbon, embodied_manufactured=made, embodied_allocated=allocated,
            embodied_share=share, water=water,
        )


class PueRetrofitModel(ForwardModel):
    """Energy and cost saved by lowering PUE under a fixed IT load over a period."""
    requires = ()
    produces = PueRetrofitResult

    def solve(self, it_power: Quantity, pue_before: float, pue_after: float, period: Quantity,
              electricity_price: Quantity) -> PueRetrofitResult:
        before = (it_power * pue_before).to("MW")
        after = (it_power * pue_after).to("MW")
        saved = (before - after).to("MW")
        energy = (saved * period).to("MWh")
        return PueRetrofitResult(facility_power_before=before, facility_power_after=after,
                                 saved_power=saved, saved_energy=energy,
                                 saved_cost=(energy * electricity_price).to("USD"))




class GridPlacementModel(ForwardModel):
    """
    One facility energy placed on several grids, optionally after an energy cut.

    Emissions are ``E x CI`` per grid; ``energy_cut`` is a fractional energy
    reduction applied before the grid factor (an algorithmic saving), so a
    placement and an efficiency gain can be compared at one energy.
    """
    requires = ()
    produces = GridPlacementResult

    def solve(self, energy: Quantity, intensities: Dict[str, Quantity], energy_cut: float = 0.0) -> GridPlacementResult:
        emissions = {k: carbon_from_energy(energy, g).to("metric_ton") for k, g in intensities.items()}
        cut = {k: (e * (1 - energy_cut)).to("metric_ton") for k, e in emissions.items()}
        lowest = min(intensities, key=lambda k: intensities[k].to("g/kWh").magnitude)
        highest = max(intensities, key=lambda k: intensities[k].to("g/kWh").magnitude)
        return GridPlacementResult(emissions=emissions, emissions_after_cut=cut, lowest=lowest,
                                   highest=highest, spread=_ratio(intensities[highest], intensities[lowest]))


class EmbodiedPaybackModel(ForwardModel):
    """
    Time for a more efficient replacement to repay its manufacturing carbon.

    The replacement does the old device's work with ``saved_fraction`` less
    energy, running continuously at the old device's power; the carbon it
    avoids per day is the saved energy times the grid intensity, and payback
    is the embodied carbon divided by that daily saving.
    """
    requires = ()
    produces = EmbodiedPaybackResult

    def solve(self, embodied: Quantity, device_power: Quantity, saved_fraction: float,
              intensities: Dict[str, Quantity]) -> EmbodiedPaybackResult:
        saved_power = (device_power * saved_fraction).to("kW")
        per_day = (saved_power * _DAY).to("kWh")
        avoided = {k: (per_day * g).to("kg") for k, g in intensities.items()}
        payback = {k: (embodied / a * _DAY).to("day") for k, a in avoided.items()}
        return EmbodiedPaybackResult(saved_power=saved_power, saved_energy_per_day=per_day,
                                     avoided_per_day=avoided, payback=payback)


class ServiceLifeExtensionModel(ForwardModel):
    """Cut in embodied carbon charged per year when a service life grows: ``1 - short / long``."""
    requires = ()
    produces = ServiceLifeExtensionResult

    def solve(self, short_life: Quantity, long_life: Quantity) -> ServiceLifeExtensionResult:
        return ServiceLifeExtensionResult(annual_charge_reduction=1 - _ratio(short_life, long_life))




class RepeatedWorkEnergyModel(ForwardModel):
    """Accelerator energy repeated when a fleet rolls back by each named interval."""
    requires = ()
    produces = RepeatedWorkEnergyResult

    def solve(self, devices: float, device_power: Quantity, rollbacks: Dict[str, Quantity]) -> RepeatedWorkEnergyResult:
        return RepeatedWorkEnergyResult(energy={k: (devices * device_power * t).to("MWh") for k, t in rollbacks.items()})


class ServingCrossoverModel(ForwardModel):
    """Days of serving at a request volume whose cumulative energy equals a one-time (training) energy."""
    requires = ()
    produces = ServingCrossoverResult

    def solve(self, one_time_energy: Quantity, requests_per_day: float, energy_per_request: Quantity) -> ServingCrossoverResult:
        daily = (requests_per_day * energy_per_request).to("MWh")
        return ServingCrossoverResult(daily_energy=daily, crossover_days=_ratio(one_time_energy, daily))


class CascadeEnergyModel(ForwardModel):
    """
    Edge-first cascade: every input runs a small edge model; a fraction
    ``escalation`` is sent (transmit) to a large cloud model. Mean energy per
    input ``E_edge + p (E_tx + E_cloud)`` against always sending to the cloud.
    """
    requires = ()
    produces = CascadeEnergyResult

    def solve(self, edge_energy: Quantity, transmit_energy: Quantity, cloud_energy: Quantity,
              escalation: float) -> CascadeEnergyResult:
        cloud_only = (transmit_energy + cloud_energy).to("mJ")
        escalated = (escalation * cloud_only).to("mJ")
        mean = (edge_energy + escalated).to("mJ")
        return CascadeEnergyResult(escalated_energy=escalated, mean_energy=mean, cloud_only_energy=cloud_only,
                                   saving=1 - _ratio(mean, cloud_only))


class DutyCycledPowerModel(ForwardModel):
    """Average power of stages each active a fraction of the time, against a reference always-on power."""
    requires = ()
    produces = DutyCycledPowerResult

    def solve(self, powers: Dict[str, Quantity], duty_cycles: Dict[str, float], reference_power: Quantity) -> DutyCycledPowerResult:
        contributions = {k: (p * duty_cycles[k]).to("uW") for k, p in powers.items()}
        avg = sum(contributions.values(), 0 * contributions[next(iter(contributions))])
        return DutyCycledPowerResult(contributions=contributions, average_power=avg.to("uW"),
                                     reduction=_ratio(reference_power, avg))


class DataMovementEnergyModel(ForwardModel):
    """
    Modeled dynamic energy of using one weight: arithmetic against moving it.

    A weight of ``bytes_per_param`` read once for ``flops_per_param`` FLOPs
    costs ``bytes x E_byte`` to move and ``FLOPs x E_flop`` to compute. The
    energy crossover ``E_byte / E_flop`` is the arithmetic intensity at which
    the two match; ``operand_bytes`` prices one operand fetch against one FLOP.
    """
    requires = ()
    produces = DataMovementEnergyResult

    def solve(self, energy_per_flop: Quantity, energy_per_byte: Quantity, flops_per_param: float,
              bytes_per_param: Quantity, operand_bytes: Quantity) -> DataMovementEnergyResult:
        e_flop = energy_per_flop.to("pJ/flop").magnitude
        e_byte = energy_per_byte.to("pJ/byte").magnitude
        nbytes = bytes_per_param.to("byte").magnitude
        compute = flops_per_param * e_flop
        move = nbytes * e_byte
        operand = operand_bytes.to("byte").magnitude * e_byte
        crossover = e_byte / e_flop
        intensity = flops_per_param / nbytes
        return DataMovementEnergyResult(
            compute_energy=_Q(compute, "pJ"), movement_energy=_Q(move, "pJ"),
            movement_share=move / (move + compute), movement_over_compute=move / compute,
            intensity=intensity, crossover_intensity=crossover, balance_batch=crossover / intensity,
            operand_energy=_Q(operand, "pJ"), operand_over_flop=operand / e_flop,
        )


class DemandReboundModel(ForwardModel):
    """Fleet energy after a per-unit efficiency change and a volume change: ``e x v``."""
    requires = ()
    produces = DemandReboundResult

    def solve(self, per_unit_factor: float, volume_factor: float) -> DemandReboundResult:
        return DemandReboundResult(net_factor=per_unit_factor * volume_factor)


class DevelopmentSweepModel(ForwardModel):
    """Final-run equivalents a development sweep consumes: candidates x relative cost."""
    requires = ()
    produces = DevelopmentSweepResult

    def solve(self, candidates: float, relative_cost: float) -> DevelopmentSweepResult:
        return DevelopmentSweepResult(final_run_equivalents=candidates * relative_cost)




class CarbonPricedPlacementModel(ForwardModel):
    """
    Placement of one run among regions with and without a carbon price.

    The run's duration is the days the fleet needs to spend ``it_energy`` at
    ``mfu`` (EconomicsModel's IT energy per day is region-independent); the
    PlacementOptimizer then ranks regions by total cost, once with the carbon
    price and once without.
    """
    requires = ("fleet",)
    produces = CarbonPricedPlacementResult

    def solve(self, fleet, it_energy: Quantity, regions: List[str], carbon_price_per_ton: float,
              mfu: float, reference_grid) -> CarbonPricedPlacementResult:
        from .economics import EconomicsModel, PlacementOptimizer
        per_day = EconomicsModel().solve(fleet, duration_days=1, grid=reference_grid, mfu=mfu).it_energy_kwh.to("MWh")
        days = _ratio(it_energy, per_day)
        opt = PlacementOptimizer()
        taxed = opt.solve(fleet, duration_days=days, regions=regions, carbon_tax_per_ton=carbon_price_per_ton, mfu=mfu)
        untaxed = opt.solve(fleet, duration_days=days, regions=regions, carbon_tax_per_ton=0.0, mfu=mfu)
        return CarbonPricedPlacementResult(
            duration_days=days, best_region=taxed.best_region, best_region_untaxed=untaxed.best_region,
            objective_value=taxed.objective_value,
            objective={c["region"]: c["objective"] for c in taxed.top_candidates},
            objective_untaxed={c["region"]: c["objective"] for c in untaxed.top_candidates},
        )

