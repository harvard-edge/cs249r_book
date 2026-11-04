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

from typing import Dict, List, Optional

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
        """The power chain from accelerators to the facility meter.

        Parameters
        ----------
        devices : float
            Accelerator count, positive and dimensionless.
        device_power : Quantity
            Power per accelerator in watts; nonnegative rated or supplied measured power.
        it_ratio : float
            Dimensionless IT-to-accelerator power ratio, at least one for this boundary.
        pue : float
            Dimensionless facility-to-IT energy ratio, at least one.

        Returns
        -------
        FacilityPowerResult
            Accelerator, IT, facility, support, and facility-overhead powers in MW.

        Notes
        -----
        Equation: P_acc = devices * device_power; P_IT = P_acc * it_ratio; P_facility
        = P_IT * PUE.

        Assumptions: All devices draw the same supplied power; ratios describe nested
        boundaries.
        Excludes: Transient peaks, redundancy margins, and capacity admission.

        References
        ----------
        Derivation: Apply each boundary ratio once; subtract accelerator from IT and
        IT from facility to isolate overhead.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = FacilityPowerModel().solve(2, Q_("100 W"), 1.2, 1.1)
        >>> result.facility_power > result.it_power
        True
        """
        acc = (devices * device_power).to("MW")
        it = (acc * it_ratio).to("MW")
        fac = (it * pue).to("MW")
        return FacilityPowerResult(
            accelerator_power=acc,
            it_power=it,
            facility_power=fac,
            support_power=(it - acc).to("MW"),
            overhead_power=(fac - it).to("MW"),
        )


class WaterUseModel(ForwardModel):
    """Water consumed for an IT energy at a water usage effectiveness: ``W = E_IT x WUE``."""

    requires = ()
    produces = WaterUseResult

    def solve(self, it_energy: Quantity, wue: Dict[str, Quantity]) -> WaterUseResult:
        """Water consumed for an IT energy at a water usage effectiveness: ``W = E_IT x WUE``.

        Parameters
        ----------
        it_energy : Quantity
            IT energy in kWh or compatible units; nonnegative.
        wue : Dict[str, Quantity]
            Named water-use factors in liters per kWh of IT energy; nonnegative.

        Returns
        -------
        WaterUseResult
            Mapping of site names to water quantities in liters.

        Notes
        -----
        Equation: water[site] = IT_energy * WUE[site], converted to liters.

        Assumptions: Every WUE uses IT energy as its denominator and remains constant.
        Excludes: Supply-chain water and facility-energy WUE conventions.

        References
        ----------
        Derivation: Multiply liters per IT kWh by IT kWh; the energy unit cancels.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = WaterUseModel().solve(Q_("1 kWh"), {"site": Q_("1 liter/kWh")})
        >>> result.water["site"] == Q_("1 liter")
        True
        """
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

    def solve(
        self,
        device_hours: Quantity,
        devices: float,
        device_power: Quantity,
        it_ratio: float,
        pue: float,
        intensities: Dict[str, Quantity],
        embodied_per_device: Quantity,
        service_life: Quantity,
        wue: Optional[Dict[str, Quantity]] = None,
    ) -> RunFootprintResult:
        """One training run's energy, operational carbon, embodied carbon, and water.

        Parameters
        ----------
        device_hours : Quantity
            Aggregate accelerator operating time in hours; count is dimensionless.
        devices : float
            Accelerator count, positive and dimensionless.
        device_power : Quantity
            Power per accelerator in watts; nonnegative rated or supplied measured power.
        it_ratio : float
            Dimensionless IT-to-accelerator power ratio, at least one for this boundary.
        pue : float
            Dimensionless facility-to-IT energy ratio, at least one.
        intensities : Dict[str, Quantity]
            Nonempty mapping of grid names to positive carbon intensities in g/kWh.
        embodied_per_device : Quantity
            Manufacturing carbon per accelerator in kg CO2e; nonnegative.
        service_life : Quantity
            Positive service lifetime in hours or compatible time units.
        wue : Optional[Dict[str, Quantity]]
            Named water-use factors in liters per kWh of IT energy; nonnegative.

        Returns
        -------
        RunFootprintResult
            Power breakdown, duration in hours, energy in MWh, operational and embodied
            carbon in metric tons, embodied fractions, and optional water in liters.

        Notes
        -----
        Equation: E_acc = device_hours * device_power; E_IT = E_acc * it_ratio;
        E_facility = E_IT * PUE. Allocated embodied carbon = device_hours *
        embodied_per_device / service_life.

        Assumptions: Device-hours aggregate equal-power accelerators; embodied carbon
        is allocated uniformly over the supplied service life.
        Excludes: Manufacturing timing, retirement credits, and variation of grid
        intensity within the run.

        References
        ----------
        Derivation: Integrate device power over aggregate hours, apply the two
        overhead ratios, and charge manufacturing carbon in proportion to consumed
        service hours.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = RunFootprintModel().solve(Q_("2 hour"), 2, Q_("100 W"), 1.2, 1.1, {"site": Q_("100 g/kWh")}, Q_("1 kg"), Q_("1 year"))
        >>> result.facility_energy > result.it_energy
        True
        """
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
            power=power,
            duration=(device_hours / devices).to("hour"),
            accelerator_energy=acc_e,
            it_energy=it_e,
            facility_energy=fac_e,
            carbon=carbon,
            embodied_manufactured=made,
            embodied_allocated=allocated,
            embodied_share=share,
            water=water,
        )


class PueRetrofitModel(ForwardModel):
    """Energy and cost saved by lowering PUE under a fixed IT load over a period."""

    requires = ()
    produces = PueRetrofitResult

    def solve(
        self, it_power: Quantity, pue_before: float, pue_after: float, period: Quantity, electricity_price: Quantity
    ) -> PueRetrofitResult:
        """Energy and cost saved by lowering PUE under a fixed IT load over a period.

        Parameters
        ----------
        it_power : Quantity
            Constant IT load in watts; nonnegative.
        pue_before : float
            Initial dimensionless PUE, at least one.
        pue_after : float
            Replacement dimensionless PUE, at least one.
        period : Quantity
            Duration in hours or compatible time units; nonnegative.
        electricity_price : Quantity
            Energy price in USD/kWh; nonnegative.

        Returns
        -------
        PueRetrofitResult
            Before/after facility powers and saved power in MW, saved energy in MWh, and
            saved electricity cost in USD.

        Notes
        -----
        Equation: saved_power = IT_power * (PUE_before - PUE_after); saved_energy =
        saved_power * period; saved_cost = saved_energy * price.

        Assumptions: IT load and electricity price stay fixed across both PUE cases.
        Excludes: Retrofit capital cost, downtime, and changes in IT demand.

        References
        ----------
        Derivation: Subtract the facility-power products before integrating over the
        period; price only the energy difference.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = PueRetrofitModel().solve(Q_("1 kW"), 1.4, 1.2, Q_("1 day"), Q_("0.1 USD/kWh"))
        >>> result.saved_energy > Q_("0 kWh")
        True
        """
        before = (it_power * pue_before).to("MW")
        after = (it_power * pue_after).to("MW")
        saved = (before - after).to("MW")
        energy = (saved * period).to("MWh")
        return PueRetrofitResult(
            facility_power_before=before,
            facility_power_after=after,
            saved_power=saved,
            saved_energy=energy,
            saved_cost=(energy * electricity_price).to("USD"),
        )


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
        """One facility energy placed on several grids, optionally after an energy cut.

        Parameters
        ----------
        energy : Quantity
            Facility energy in kWh; nonnegative.
        intensities : Dict[str, Quantity]
            Nonempty mapping of grid names to positive carbon intensities in g/kWh.
        energy_cut : float
            Fractional energy reduction in [0, 1]; default zero.

        Returns
        -------
        GridPlacementResult
            Emissions before and after the cut in metric tons, lowest/highest intensity
            grid names, and their dimensionless intensity ratio.

        Notes
        -----
        Equation: emissions[grid] = energy * intensity[grid]; reduced_emissions =
        emissions * (1 - energy_cut).

        Assumptions: All regions serve identical energy demand and all intensities are
        positive.
        Excludes: Regional latency, electricity cost, marginal dispatch, and placement
        feasibility.

        References
        ----------
        Derivation: At fixed energy, carbon ordering equals intensity ordering; scale
        both by the same fractional energy saving.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = GridPlacementModel().solve(Q_("1 kWh"), {"low": Q_("10 g/kWh"), "high": Q_("100 g/kWh")})
        >>> result.lowest == "low"
        True
        """
        emissions = {k: carbon_from_energy(energy, g).to("metric_ton") for k, g in intensities.items()}
        cut = {k: (e * (1 - energy_cut)).to("metric_ton") for k, e in emissions.items()}
        lowest = min(intensities, key=lambda k: intensities[k].to("g/kWh").magnitude)
        highest = max(intensities, key=lambda k: intensities[k].to("g/kWh").magnitude)
        return GridPlacementResult(
            emissions=emissions,
            emissions_after_cut=cut,
            lowest=lowest,
            highest=highest,
            spread=_ratio(intensities[highest], intensities[lowest]),
        )


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

    def solve(
        self, embodied: Quantity, device_power: Quantity, saved_fraction: float, intensities: Dict[str, Quantity]
    ) -> EmbodiedPaybackResult:
        """Time for a more efficient replacement to repay its manufacturing carbon.

        Parameters
        ----------
        embodied : Quantity
            Replacement manufacturing carbon in kg CO2e; nonnegative.
        device_power : Quantity
            Power per accelerator in watts; nonnegative rated or supplied measured power.
        saved_fraction : float
            Fraction of baseline power avoided, in (0, 1].
        intensities : Dict[str, Quantity]
            Nonempty mapping of grid names to positive carbon intensities in g/kWh.

        Returns
        -------
        EmbodiedPaybackResult
            Saved power in kW, daily energy in kWh, daily avoided carbon in kg, and per-
            grid payback times in days.

        Notes
        -----
        Equation: saved_power = device_power * saved_fraction; avoided_carbon_per_day
        = saved_power * one_day * intensity; payback = embodied /
        avoided_carbon_per_day.

        Assumptions: Replacement provides equal useful work continuously; power saving
        and positive grid intensities stay fixed.
        Excludes: Idle periods, changing demand, old-device disposal, and discounting
        future emissions.

        References
        ----------
        Derivation: Divide the one-time manufacturing carbon by avoided carbon per day
        to obtain a carbon-balance time.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = EmbodiedPaybackModel().solve(Q_("1 kg"), Q_("1 kW"), 0.5, {"site": Q_("100 g/kWh")})
        >>> result.payback["site"] > Q_("0 day")
        True
        """
        saved_power = (device_power * saved_fraction).to("kW")
        per_day = (saved_power * _DAY).to("kWh")
        avoided = {k: (per_day * g).to("kg") for k, g in intensities.items()}
        payback = {k: (embodied / a * _DAY).to("day") for k, a in avoided.items()}
        return EmbodiedPaybackResult(
            saved_power=saved_power, saved_energy_per_day=per_day, avoided_per_day=avoided, payback=payback
        )


class ServiceLifeExtensionModel(ForwardModel):
    """Cut in embodied carbon charged per year when a service life grows: ``1 - short / long``."""

    requires = ()
    produces = ServiceLifeExtensionResult

    def solve(self, short_life: Quantity, long_life: Quantity) -> ServiceLifeExtensionResult:
        """Cut in embodied carbon charged per year when a service life grows: ``1 - short / long``.

        Parameters
        ----------
        short_life : Quantity
            Positive original service life in time units.
        long_life : Quantity
            Positive extended service life in compatible time units.

        Returns
        -------
        ServiceLifeExtensionResult
            Dimensionless reduction in annual embodied-carbon allocation.

        Notes
        -----
        Equation: annual_charge_reduction = 1 - short_life / long_life.

        Assumptions: Identical embodied carbon is spread uniformly over either
        positive lifetime.
        Excludes: Replacement operating energy and changes in useful work.

        References
        ----------
        Derivation: Annual charges are C/short and C/long; their relative reduction
        cancels C and equals 1-short/long.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = ServiceLifeExtensionModel().solve(Q_("1 year"), Q_("2 year"))
        >>> result.annual_charge_reduction > 0
        True
        """
        return ServiceLifeExtensionResult(annual_charge_reduction=1 - _ratio(short_life, long_life))


class RepeatedWorkEnergyModel(ForwardModel):
    """Accelerator energy repeated when a fleet rolls back by each named interval."""

    requires = ()
    produces = RepeatedWorkEnergyResult

    def solve(self, devices: float, device_power: Quantity, rollbacks: Dict[str, Quantity]) -> RepeatedWorkEnergyResult:
        """Accelerator energy repeated when a fleet rolls back by each named interval.

        Parameters
        ----------
        devices : float
            Accelerator count, positive and dimensionless.
        device_power : Quantity
            Power per accelerator in watts; nonnegative rated or supplied measured power.
        rollbacks : Dict[str, Quantity]
            Named nonnegative rollback intervals in time units.

        Returns
        -------
        RepeatedWorkEnergyResult
            Named rollback energy expenditures in MWh.

        Notes
        -----
        Equation: energy[interval] = devices * device_power * rollback_time[interval].

        Assumptions: All devices repeat work at constant supplied power for each
        interval.
        Excludes: Recovery I/O, idle power, and facility overhead.

        References
        ----------
        Derivation: Integrate fleet accelerator power over each repeated-work
        interval.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = RepeatedWorkEnergyModel().solve(2, Q_("100 W"), {"short": Q_("1 hour"), "long": Q_("2 hour")})
        >>> result.energy["long"] == 2 * result.energy["short"]
        True
        """
        return RepeatedWorkEnergyResult(
            energy={k: (devices * device_power * t).to("MWh") for k, t in rollbacks.items()}
        )


class ServingCrossoverModel(ForwardModel):
    """Calculate serving days needed to match a one-time energy expenditure."""

    requires = ()
    produces = ServingCrossoverResult

    def solve(
        self, one_time_energy: Quantity, requests_per_day: float, energy_per_request: Quantity
    ) -> ServingCrossoverResult:
        """Calculate serving days needed to match a one-time energy expenditure.

        Parameters
        ----------
        one_time_energy : Quantity
            One-time energy expenditure in joules or compatible units.
        requests_per_day : float
            Positive request count per calendar day, supplied as a scalar.
        energy_per_request : Quantity
            Positive energy per request in joules; request count is dimensionless.

        Returns
        -------
        ServingCrossoverResult
            Daily serving energy in MWh and dimensionless elapsed day count at
            crossover.

        Notes
        -----
        Equation: daily_energy = requests_per_day * energy_per_request; crossover_days
        = one_time_energy / daily_energy.

        Assumptions: Positive constant daily request volume and per-request energy.
        Excludes: Load growth, idle infrastructure, and latency constraints.

        References
        ----------
        Derivation: Divide one-time energy by daily serving energy; each quotient unit
        denotes one day of demand.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = ServingCrossoverModel().solve(Q_("1 kWh"), 100, Q_("1 J"))
        >>> result.crossover_days > 0
        True
        """
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

    def solve(
        self, edge_energy: Quantity, transmit_energy: Quantity, cloud_energy: Quantity, escalation: float
    ) -> CascadeEnergyResult:
        """Edge-first cascade: every input runs a small edge model; a fraction.

        Parameters
        ----------
        edge_energy : Quantity
            Energy spent locally on every input, in joules.
        transmit_energy : Quantity
            Transmission energy for an escalated input, in joules.
        cloud_energy : Quantity
            Cloud processing energy for an escalated input, in joules.
        escalation : float
            Fraction of inputs sent to the cloud, in [0, 1].

        Returns
        -------
        CascadeEnergyResult
            Escalated, mean, and cloud-only energies in mJ, plus dimensionless relative
            saving.

        Notes
        -----
        Equation: cloud_only = transmit_energy + cloud_energy; mean = edge_energy +
        escalation * cloud_only; saving = 1 - mean / cloud_only.

        Assumptions: Every input pays edge cost; a fixed fraction also pays
        transmission and cloud costs.
        Excludes: Accuracy changes, queueing, retries, and load-dependent power.

        References
        ----------
        Derivation: Use the expectation of a two-branch decision: fixed edge cost plus
        escalation probability times remote cost.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = CascadeEnergyModel().solve(Q_("1 mJ"), Q_("1 mJ"), Q_("10 mJ"), 0.1)
        >>> result.mean_energy < result.cloud_only_energy
        True
        """
        cloud_only = (transmit_energy + cloud_energy).to("mJ")
        escalated = (escalation * cloud_only).to("mJ")
        mean = (edge_energy + escalated).to("mJ")
        return CascadeEnergyResult(
            escalated_energy=escalated,
            mean_energy=mean,
            cloud_only_energy=cloud_only,
            saving=1 - _ratio(mean, cloud_only),
        )


class DutyCycledPowerModel(ForwardModel):
    """Average power of stages each active a fraction of the time, against a reference always-on power."""

    requires = ()
    produces = DutyCycledPowerResult

    def solve(
        self, powers: Dict[str, Quantity], duty_cycles: Dict[str, float], reference_power: Quantity
    ) -> DutyCycledPowerResult:
        """Average power of stages each active a fraction of the time, against a reference always-on power.

        Parameters
        ----------
        powers : Dict[str, Quantity]
            Nonempty mapping of stage names to nonnegative power quantities.
        duty_cycles : Dict[str, float]
            Stage-name mapping to active fractions in [0, 1]; must cover powers.
        reference_power : Quantity
            Positive always-on comparison power in watts.

        Returns
        -------
        DutyCycledPowerResult
            Stage contributions and average power in microwatts, and reference-to-
            average power ratio.

        Notes
        -----
        Equation: average_power = sum(stage_power * stage_duty); reduction =
        reference_power / average_power.

        Assumptions: Stage powers are additive; duty fractions describe the same
        observation window.
        Excludes: Sleep leakage, switching energy, and correlated peak-power
        constraints.

        References
        ----------
        Derivation: Time-averaging each rectangular active-power contribution gives
        power times duty fraction; add the contributions.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = DutyCycledPowerModel().solve({"stage": Q_("10 W")}, {"stage": 0.5}, Q_("10 W"))
        >>> result.average_power == Q_("5 W")
        True
        """
        contributions = {k: (p * duty_cycles[k]).to("uW") for k, p in powers.items()}
        avg = sum(contributions.values(), 0 * contributions[next(iter(contributions))])
        return DutyCycledPowerResult(
            contributions=contributions, average_power=avg.to("uW"), reduction=_ratio(reference_power, avg)
        )


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

    def solve(
        self,
        energy_per_flop: Quantity,
        energy_per_byte: Quantity,
        flops_per_param: float,
        bytes_per_param: Quantity,
        operand_bytes: Quantity,
    ) -> DataMovementEnergyResult:
        """Modeled dynamic energy of using one weight: arithmetic against moving it.

        Parameters
        ----------
        energy_per_flop : Quantity
            Positive dynamic arithmetic energy in pJ/flop.
        energy_per_byte : Quantity
            Positive transfer energy in pJ/byte.
        flops_per_param : float
            Positive arithmetic operation count per weight use.
        bytes_per_param : Quantity
            Positive bytes transferred for one weight.
        operand_bytes : Quantity
            Positive bytes transferred for one operand fetch.

        Returns
        -------
        DataMovementEnergyResult
            Compute, transfer, and operand energies in pJ; dimensionless energy ratios;
            arithmetic intensities in FLOP per byte; and a dimensionless balance batch.

        Notes
        -----
        Equation: compute_energy = flops_per_param * energy_per_flop; movement_energy
        = bytes_per_param * energy_per_byte; crossover_intensity = energy_per_byte /
        energy_per_flop.

        Assumptions: Each weight is transferred once per modeled use; energy per
        operation and byte is fixed.
        Excludes: Caches, static power, communication protocols, and precision-
        dependent kernel effects.

        References
        ----------
        Derivation: Price the operation and byte counts separately; equating their
        energies gives the energy ridge point E_byte/E_flop.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = DataMovementEnergyModel().solve(Q_("1 pJ/flop"), Q_("10 pJ/byte"), 2, Q_("2 byte"), Q_("2 byte"))
        >>> result.movement_energy > result.compute_energy
        True
        """
        e_flop = energy_per_flop.to("pJ/flop").magnitude
        e_byte = energy_per_byte.to("pJ/byte").magnitude
        nbytes = bytes_per_param.to("byte").magnitude
        compute = flops_per_param * e_flop
        move = nbytes * e_byte
        operand = operand_bytes.to("byte").magnitude * e_byte
        crossover = e_byte / e_flop
        intensity = flops_per_param / nbytes
        return DataMovementEnergyResult(
            compute_energy=_Q(compute, "pJ"),
            movement_energy=_Q(move, "pJ"),
            movement_share=move / (move + compute),
            movement_over_compute=move / compute,
            intensity=intensity,
            crossover_intensity=crossover,
            balance_batch=crossover / intensity,
            operand_energy=_Q(operand, "pJ"),
            operand_over_flop=operand / e_flop,
        )


class DemandReboundModel(ForwardModel):
    """Fleet energy after a per-unit efficiency change and a volume change: ``e x v``."""

    requires = ()
    produces = DemandReboundResult

    def solve(self, per_unit_factor: float, volume_factor: float) -> DemandReboundResult:
        """Fleet energy after a per-unit efficiency change and a volume change: ``e x v``.

        Parameters
        ----------
        per_unit_factor : float
            Nonnegative new-to-old energy ratio per unit of work.
        volume_factor : float
            Nonnegative new-to-old workload volume ratio.

        Returns
        -------
        DemandReboundResult
            Dimensionless total-energy ratio after both changes.

        Notes
        -----
        Equation: net_factor = per_unit_factor * volume_factor.

        Assumptions: Per-unit energy and total volume factors multiply independently.
        Excludes: Causal prediction of induced demand and changes in workload mix.

        References
        ----------
        Derivation: Total energy is energy per unit times unit count; dividing new by
        old cancels the baseline units.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = DemandReboundModel().solve(0.5, 2)
        >>> result.net_factor == 1
        True
        """
        return DemandReboundResult(net_factor=per_unit_factor * volume_factor)


class DevelopmentSweepModel(ForwardModel):
    """Final-run equivalents a development sweep consumes: candidates x relative cost."""

    requires = ()
    produces = DevelopmentSweepResult

    def solve(self, candidates: float, relative_cost: float) -> DevelopmentSweepResult:
        """Final-run equivalents a development sweep consumes: candidates x relative cost.

        Parameters
        ----------
        candidates : float
            Nonnegative number of development candidates.
        relative_cost : float
            Nonnegative cost per candidate relative to a final run.

        Returns
        -------
        DevelopmentSweepResult
            Dimensionless development cost measured in final-run equivalents.

        Notes
        -----
        Equation: final_run_equivalents = candidates * relative_cost.

        Assumptions: Every candidate has the same supplied relative cost.
        Excludes: Failed runs, shared preprocessing, and variable candidate budgets.

        References
        ----------
        Derivation: Sum equal candidate costs after normalizing each by final-run
        cost.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = DevelopmentSweepModel().solve(4, 0.5)
        >>> result.final_run_equivalents == 2
        True
        """
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

    def solve(
        self, fleet, it_energy: Quantity, regions: List[str], carbon_price_per_ton: float, mfu: float, reference_grid
    ) -> CarbonPricedPlacementResult:
        """Placement of one run among regions with and without a carbon price.

        Parameters
        ----------
        fleet : object
            Fleet supplying accelerator count, power, price, and environment defaults.
        it_energy : Quantity
            IT energy in kWh or compatible units; nonnegative.
        regions : List[str]
            Nonempty list of registered grid names; unrecognized names are skipped.
        carbon_price_per_ton : float
            Nonnegative carbon price in USD per metric ton CO2e.
        mfu : float
            Dimensionless model FLOPs utilization in [0, 1].
        reference_grid : object
            Grid profile used to determine the region-independent IT energy per day.

        Returns
        -------
        CarbonPricedPlacementResult
            Duration in days, chosen regions, and taxed/untaxed objective mappings in
            USD.

        Notes
        -----
        Equation: duration_days = IT_energy / modeled_IT_energy_per_day;
        objective[region] = TCO[region] + carbon_tonnes[region] *
        carbon_price_per_ton.

        Assumptions: EconomicsModel gives region-independent IT energy per day;
        candidates share fleet and MFU.
        Excludes: Region-specific runtime changes, data-transfer bills, and deployment
        eligibility.

        References
        ----------
        Derivation: Convert the fixed IT-energy budget to duration, then enumerate
        EconomicsModel totals with and without the carbon charge.

        Examples
        --------
        >>> from mlsysim import Q_, Systems, Infrastructure
        >>> result = CarbonPricedPlacementModel().solve(Systems.Clusters.Research_256, Q_("1 MWh"), ["Quebec"], 100, 0.5, Infrastructure.Grids.Quebec)
        >>> result.best_region == "Quebec"
        True
        """
        from .economics import EconomicsModel, PlacementOptimizer

        per_day = EconomicsModel().solve(fleet, duration_days=1, grid=reference_grid, mfu=mfu).it_energy_kwh.to("MWh")
        days = _ratio(it_energy, per_day)
        opt = PlacementOptimizer()
        taxed = opt.solve(fleet, duration_days=days, regions=regions, carbon_tax_per_ton=carbon_price_per_ton, mfu=mfu)
        untaxed = opt.solve(fleet, duration_days=days, regions=regions, carbon_tax_per_ton=0.0, mfu=mfu)
        return CarbonPricedPlacementResult(
            duration_days=days,
            best_region=taxed.best_region,
            best_region_untaxed=untaxed.best_region,
            objective_value=taxed.objective_value,
            objective={c["region"]: c["objective"] for c in taxed.top_candidates},
            objective_untaxed={c["region"]: c["objective"] for c in untaxed.top_candidates},
        )
