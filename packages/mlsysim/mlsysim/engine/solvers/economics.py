"""Sustainability, economics, responsible-engineering, and placement solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..results import (
    SustainabilityResult,
    EconomicsResult,
    ResponsibleEngineeringResult,
    PlacementOptimizerResult,
)
from ...core.units import Q_
from ...infrastructure.registry import Infrastructure
from .. import calibration as cal
from ...core.types import Quantity
from ...systems.types import Fleet
from ...infrastructure.types import Datacenter
from .base import BaseOptimizer, ForwardModel

class SustainabilityModel(ForwardModel):
    """
    Calculates Datacenter-scale Sustainability metrics.

    Handles Power Usage Effectiveness (PUE), Carbon Intensity,
    and Water Usage Effectiveness (WUE) across different regional grids.
    This model simulates the 'Infrastructure Tax' — the energy spent on
    cooling and power delivery rather than on neural computation.

    Literature Source:
    1. Patterson et al. (2021), "Carbon Emissions and Large Neural Network
       Training."
    2. Belkhir & Elmeligi (2018), "Assessing ICT Global Emissions Footprint."
    3. Wu et al. (2022), "Sustainable AI: Environmental Implications,
       Challenges and Opportunities."

    Formula contract:
    - per-device power = idle_fraction * TDP + dynamic_fraction * TDP * MFU.
    - IT energy = device power * accelerator count * duration.
    - facility energy applies PUE; operational carbon applies grid carbon
      intensity to facility energy; water applies WUE to facility energy.
    """
    requires = ("fleet",)
    produces = SustainabilityResult

    def solve(self, fleet: Fleet, duration_days: float, datacenter: Optional[Datacenter] = None,
              mfu: float = 1.0, embodied_carbon_per_device: float = 0.0) -> SustainabilityResult:
        """
        Calculate energy, carbon, and water footprint for a fleet operation.

        Implements the formula contract in the class docstring: per-device
        power is an energy-proportionality model
        (``idle_fraction * TDP + dynamic_fraction * TDP * MFU``), scaled to
        the fleet, integrated over the duration, then taxed by PUE for
        facility energy. Carbon and water apply the regional grid's carbon
        intensity and WUE to facility (post-PUE) energy.

        Parameters
        ----------
        fleet : Fleet
            The cluster; supplies accelerator count, per-device TDP, and the
            default datacenter/region when ``datacenter`` is not given.
        duration_days : float
            Operation duration in days.
        datacenter : Datacenter, optional
            Override for the fleet's datacenter (or a bare GridProfile);
            supplies PUE, carbon intensity, and WUE. Falls back to the fleet
            region and finally to the US average grid.
        mfu : float
            Model FLOPs Utilization in [0, 1]; drives the dynamic-power term.
        embodied_carbon_per_device : float
            Manufacturing/shipping carbon per accelerator in kg CO2e
            (default 0 = operational carbon only).

        Returns
        -------
        SustainabilityResult
            IT and facility energy (kWh), total carbon (kg CO2e, operational
            plus embodied), water usage (liters), and the PUE applied.
        """
        # 1. Resolve Environment
        dc = datacenter or fleet.datacenter

        # Flexibly handle if dc is already a GridProfile or a Datacenter
        if hasattr(dc, 'grid'):
            region = dc.grid
        else:
            region = dc or fleet.region

        if not region:
             from ...infrastructure.registry import Grids
             region = Grids.US_Avg

        from ...core._validation import validate_range, validate_nonnegative
        validate_range(mfu, 0.0, 1.0, "mfu")
        validate_nonnegative(embodied_carbon_per_device, "embodied_carbon_per_device")

        duration_hours = duration_days * 24

        # 2. Power — fallback TDP comes from the registry (H100 reference
        # accelerator), not a magic literal (SSOT, audit fix 2026-06-06).
        if fleet.node.accelerator.tdp:
            base_tdp = fleet.node.accelerator.tdp
        else:
            from ...hardware.registry import Cloud as _CloudHardware
            base_tdp = _CloudHardware.H100.tdp
        # Energy proportionality: Idle power is ~30% of TDP. Dynamic power scales with compute utilization (MFU).
        idle_power = base_tdp * cal.ENERGY_IDLE_FRACTION
        dynamic_power = base_tdp * cal.ENERGY_DYNAMIC_FRACTION * mfu
        effective_power_per_chip = idle_power + dynamic_power
        it_power_w = effective_power_per_chip * fleet.total_accelerators

        # 3. Energy Consumption
        it_energy_kwh = (it_power_w * Q_(duration_hours, "hour")).to("kWh")

        # Apply PUE
        pue = getattr(dc, 'pue', fleet.effective_pue)
        total_energy_kwh = it_energy_kwh * pue

        # 4. Carbon Footprint (use total facility energy, PUE already applied)
        # Prefer the grid's own carbon_kg() helper; otherwise apply the raw
        # intensity, with /1000 converting gCO2e/kWh to kgCO2e/kWh.
        carbon_kg = region.carbon_kg(total_energy_kwh.magnitude) if hasattr(region, 'carbon_kg') else total_energy_kwh.magnitude * (region.carbon_intensity_g_kwh / 1000.0)

        # 5. Water Usage
        # Resolve WUE from dc.grid, dc, or region
        if hasattr(dc, 'grid') and dc.grid:
            wue = dc.grid.wue
        elif hasattr(dc, 'wue'):
            wue = dc.wue
        else:
            wue = region.wue

        water_liters = total_energy_kwh.magnitude * wue

        # 6. Embodied Carbon (manufacturing, shipping, end-of-life)
        # Source: Gupta et al. (2022), "ACT: Designing Sustainable Computer Systems
        #         with an Architectural Carbon Modeling Tool"
        n_devices = fleet.total_accelerators
        embodied_kg = embodied_carbon_per_device * n_devices
        total_carbon_kg = carbon_kg + embodied_kg

        return SustainabilityResult(
            it_energy_kwh=it_energy_kwh,
            total_energy_kwh=total_energy_kwh,
            carbon_footprint_kg=total_carbon_kg,
            water_usage_liters=water_liters,
            pue=pue,
            region_name=region.name,
            embodied_carbon_kg=embodied_kg,
        )

class EconomicsModel(ForwardModel):
    """
    Calculates Total Cost of Ownership (TCO) including Capex and Opex.

    Combines hardware costs, energy consumption, and maintenance
    into a single financial model for the fleet.

    Literature Source:
    1. Barroso et al. (2018), "The Datacenter as a Computer: An Introduction
       to the Design of Warehouse-Scale Machines."
    2. Patterson (2004), "Latent Bugs in Common-Case Software." (TCO Foundations)
    3. Meta (2024), "Sustainable AI Infrastructure at Meta Scale."

    Formula contract:
    - energy OpEx delegates to SustainabilityModel so cost, carbon, and water
      share the same PUE and grid assumptions.
    - hardware CapEx = unit cost * accelerator count * infrastructure multiplier.
    - period CapEx and maintenance are prorated by duration over the
      amortization window.
    """
    requires = ("fleet",)
    produces = EconomicsResult
    _fallacies = {
        "Cheaper hardware is always more cost-effective": "Reality: slower hardware may cost more in electricity and total time than expensive hardware. TCO = CapEx + OpEx; a 2x cheaper GPU that takes 3x longer has higher TCO.",
        "GPU cost is the dominant expense": "Reality: networking, cooling, facility, and staff costs are 50-150% of GPU CapEx. Use infrastructure_multiplier=2.0-2.5 for realistic TCO.",
        "Cloud is always more expensive than on-prem": "Reality: for bursty or short-duration workloads, cloud spot instances can be 3-10x cheaper than amortized on-prem hardware sitting idle.",
    }

    def solve(self, fleet: Fleet, duration_days: float, kwh_price: Optional[float] = None, datacenter: Optional[Any] = None, grid: Optional[Any] = None, mfu: float = 1.0, amortization_years: float = 3.0, infrastructure_multiplier: float = cal.DEFAULT_INFRASTRUCTURE_MULTIPLIER) -> EconomicsResult:
        """
        Calculates the TCO for a fleet over a specified duration.

        Parameters
        ----------
        fleet : Fleet
            The hardware cluster configuration.
        duration_days : float
            Operation duration in days.
        kwh_price : float, optional
            Price of electricity per kWh.
        datacenter : Datacenter, optional
            A specific datacenter profile.
        grid : GridProfile, optional
            A specific grid profile.
        mfu : float, optional
            Model FLOPs Utilization (0.0 to 1.0) impacting energy footprint.
        amortization_years : float, optional
            Straight-line CapEx depreciation horizon (default 3 years). The
            period is charged ``total_capex / amortization_years *
            duration_days / 365`` — a 365-day year, matching the book's
            8,760 h/yr convention.
        infrastructure_multiplier : float, optional
            CapEx multiplier for networking/cooling/facility/staff (default
            1.0 = hardware only; 2.0-2.5 approximates full datacenter TCO).

        Returns
        -------
        EconomicsResult
            Financial metrics including CapEx, OpEx, and total TCO. Energy
            OpEx is facility-level (PUE-loaded via SustainabilityModel);
            maintenance accrues on FULL CapEx, not the amortized slice.
        """
        sust_model = SustainabilityModel()
        energy_result = sust_model.solve(fleet, duration_days, datacenter=datacenter or grid, mfu=mfu)

        price = kwh_price
        if price is None:
            # Electricity price resolution chain: explicit argument > grid >
            # datacenter > fleet's own datacenter/region > registry default.
            # Mirrors how the energy model resolved its grid, so price and
            # carbon come from the same place when possible.
            target = grid or datacenter or fleet.datacenter or fleet.region
            price = getattr(target, 'kwh_price', None)
            if price is None:
                price = Infrastructure.Pricing.Cloud.ElectricityPerKwh.rate.magnitude

        opex_energy = energy_result.total_energy_kwh.magnitude * price

        unit_cost = fleet.node.accelerator.unit_cost
        if unit_cost is None:
            # Unknown hardware price should not silently inherit an H100 cost.
            # Registries with sourced pricing populate unit_cost explicitly.
            unit_cost = Q_("0 USD")
        total_capex_hardware = unit_cost.magnitude * fleet.total_accelerators
        # Apply infrastructure multiplier for networking, cooling, facility, staff costs
        # Default 1.0 (hardware only). Set 2.0-2.5x for full datacenter TCO.
        total_capex = total_capex_hardware * infrastructure_multiplier
        # Amortize CapEx over deployment period (default 3-year depreciation
        # schedule): annual share of the purchase, prorated to the run's days.
        capex_for_period = (total_capex / amortization_years) * (duration_days / 365.0)

        # Maintenance scales with the FULL CapEx (you maintain the whole asset,
        # not the amortized slice), prorated to the period.
        annual_maintenance_ratio = Infrastructure.Pricing.Capital.AnnualMaintenanceRatio.rate
        opex_maintenance = total_capex * annual_maintenance_ratio * (duration_days / 365.0)

        # Compose economics + sustainability into single result
        return EconomicsResult(
            capex_usd=capex_for_period,
            opex_energy_usd=opex_energy,
            opex_maintenance_usd=opex_maintenance,
            total_opex_usd=opex_energy + opex_maintenance,
            tco_usd=capex_for_period + opex_energy + opex_maintenance,
            it_energy_kwh=energy_result.it_energy_kwh,
            total_energy_kwh=energy_result.total_energy_kwh,
            carbon_footprint_kg=energy_result.carbon_footprint_kg,
            water_usage_liters=energy_result.water_usage_liters,
            pue=energy_result.pue,
            region_name=energy_result.region_name,
        )

class ResponsibleEngineeringModel(ForwardModel):
    """
    Models the computational cost of responsible AI practices (Wall 20: Safety).

    This model quantifies the 'Safety Tax' — the additional compute and data
    required for differential privacy or fairness guarantees.

    Literature Source:
    1. Abadi et al. (2016), "Deep Learning with Differential Privacy."
    2. Anil et al. (2022), "Large-Scale Differentially Private BERT."
    """
    requires = ("training_time",)
    produces = ResponsibleEngineeringResult

    def solve(self, base_training_time: Quantity,
              epsilon: float = 1.0, delta: float = 1e-5,
              min_subgroup_prevalence: float = 0.01) -> ResponsibleEngineeringResult:
        """
        Calculate the compute and data overhead of responsible-AI guarantees.

        Two first-order cost models:

        - **Privacy (DP-SGD slowdown)**:
          ``slowdown = 1 + coefficient / epsilon`` — tighter privacy budgets
          (smaller epsilon) require more noise and per-sample clipping work,
          so training time grows hyperbolically as epsilon -> 0. The
          coefficient is a pedagogical calibration constant
          (``DP_SGD_SLOWDOWN_COEFFICIENT`` in ``engine/calibration.py``).
        - **Fairness (data requirement)**:
          ``additional_data = 1 / min_subgroup_prevalence`` — to collect N
          samples of a subgroup seen at prevalence p, you must collect ~N/p
          samples overall.

        Parameters
        ----------
        base_training_time : Quantity
            Training time without privacy guarantees (any time unit; the
            result keeps the same unit).
        epsilon : float
            Differential-privacy budget (> 0); smaller is stricter. Clamped
            below at 0.01 to avoid a singular slowdown.
        delta : float
            DP failure probability (reported, not used in the slowdown model).
        min_subgroup_prevalence : float
            Prevalence in (0, 1] of the rarest subgroup that must be
            adequately represented.

        Returns
        -------
        ResponsibleEngineeringResult
            DP slowdown factor (dimensionless), effective training time, and
            the data-collection multiplier.
        """
        dp_slowdown = 1.0 + (cal.DP_SGD_SLOWDOWN_COEFFICIENT / max(epsilon, 0.01))
        additional_data_factor = 1.0 / max(min_subgroup_prevalence, 1e-6)
        effective_time = base_training_time * dp_slowdown

        return ResponsibleEngineeringResult(
            dp_slowdown_factor=dp_slowdown,
            effective_training_time=effective_time.to(base_training_time.units),
            additional_data_requirement=additional_data_factor,
            epsilon=epsilon,
            delta=delta,
            min_subgroup_prevalence=min_subgroup_prevalence,
            privacy_cost_ratio=dp_slowdown,
            fairness_data_ratio=additional_data_factor,
        )

class PlacementOptimizer(BaseOptimizer):
    """
    Finds the datacenter region that minimizes TCO plus a carbon tax.

    Sweeps a list of grid regions, runs ``EconomicsModel`` for each, and
    ranks them by the combined objective
    ``TCO_usd + carbon_tons * carbon_tax_per_ton``. This makes the
    carbon/cost trade-off explicit: a cheap-but-dirty grid can lose to a
    pricier low-carbon one once carbon is priced in.
    """
    requires = ("fleet", "duration_days")
    produces = PlacementOptimizerResult

    def solve(self, fleet: Fleet, duration_days: float,
              regions: List[str] = ["US_Avg", "Quebec", "Iowa"],
              carbon_tax_per_ton: float = 100.0, mfu: float = 1.0) -> PlacementOptimizerResult:
        """
        Determine the datacenter region minimizing TCO plus carbon tax.

        Parameters
        ----------
        fleet : Fleet
            The cluster to place.
        duration_days : float
            Operation duration in days.
        regions : List[str]
            Candidate grid names looked up on ``Infrastructure.Grids``;
            unknown names are skipped silently.
        carbon_tax_per_ton : float
            Carbon price in USD per metric ton CO2e (default 100, a common
            mid-range social-cost-of-carbon teaching value).
        mfu : float
            Model FLOPs Utilization in [0, 1]; passed through to the
            energy model.

        Returns
        -------
        PlacementOptimizerResult
            Best region with its TCO (USD), carbon footprint (metric tons),
            PUE, and the full ranked candidate list.
        """
        from ...infrastructure.registry import Infrastructure
        econ_model = EconomicsModel()

        candidates = []

        for region_name in regions:
            grid = getattr(Infrastructure.Grids, region_name, None)
            if not grid: continue

            res = econ_model.solve(fleet, duration_days=duration_days, grid=grid, mfu=mfu)

            # Objective: TCO + Carbon Tax. Pricing carbon converts the
            # environmental externality into the same currency as TCO, so a
            # cheap-but-dirty grid can lose to a pricier low-carbon one.
            carbon_tons = res.carbon_footprint_kg / 1000.0  # kg -> metric tons
            total_cost = res.tco_usd + (carbon_tons * carbon_tax_per_ton)

            candidates.append({
                "region": region_name,
                "tco": res.tco_usd,
                "carbon": carbon_tons,
                "pue": res.pue,
                "objective": total_cost
            })

        if not candidates:
            raise ValueError("No valid regions found for optimization.")

        best = min(candidates, key=lambda x: x["objective"])
        top_n = sorted(candidates, key=lambda x: x["objective"])

        return PlacementOptimizerResult(
            objective_value=best["objective"],
            best_config={"region": best["region"]},
            best_region=best["region"],
            lowest_tco=best["tco"],
            carbon_footprint=best["carbon"],
            pue=best["pue"],
            total_searched=len(candidates),
            top_candidates=top_n
        )

__all__ = [
    "SustainabilityModel",
    "EconomicsModel",
    "ResponsibleEngineeringModel",
    "PlacementOptimizer",
]
