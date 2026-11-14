"""Sustainability, economics, responsible-engineering, and placement solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

import math
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
      intensity to facility energy; water applies on-site WUE to IT energy.
    """

    requires = ("fleet",)
    produces = SustainabilityResult

    def solve(
        self,
        fleet: Fleet,
        duration_days: float,
        datacenter: Optional[Datacenter] = None,
        mfu: float = 1.0,
        embodied_carbon_per_device: float = 0.0,
    ) -> SustainabilityResult:
        """Calculate energy, carbon, and water footprint for a fleet operation.

        Implements the formula contract in the class docstring: per-device
        power is an energy-proportionality model
        (``idle_fraction * TDP + dynamic_fraction * TDP * MFU``), scaled to
        the fleet, integrated over the duration, then taxed by PUE for
        facility energy. Operational carbon applies the regional grid's carbon
        intensity to facility (post-PUE) energy, whereas water applies on-site
        WUE (L / kWh_IT) to pre-PUE IT equipment energy.

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

        Notes
        -----
        Equation: P = TDP * (idle_fraction + dynamic_fraction * MFU); E_IT = P * devices * duration;
        E_facility = PUE * E_IT. Carbon is E_facility * grid intensity plus the full supplied
        embodied amount. Water is ``E_IT * WUE`` (The Green Grid standard: liters per kWh of IT
        equipment energy, matching ``WaterUseModel``).

        Assumptions: Constant utilization, PUE, and regional intensities over the run. Embodied carbon is charged fully rather than amortized.
        Excludes: Measured power traces, host power beyond the assumed accelerator-based account, off-site source-electricity water (EWIF), and lifetime allocation of embodied carbon.

        References
        ----------
        Derivation: Integrate the assumed idle-plus-utilization power over the fleet
        duration, apply PUE to facility electricity/carbon, and multiply IT energy by
        the regional on-site WUE (L / kWh_IT).

        Examples
        --------
        >>> from mlsysim import Q_, Systems
        >>> result = SustainabilityModel().solve(Systems.Clusters.Research_256, duration_days=1)
        >>> result.total_energy_kwh >= result.it_energy_kwh
        True
        """
        # 1. Resolve Environment
        dc = datacenter or fleet.datacenter

        # Flexibly handle if dc is already a GridProfile or a Datacenter
        if hasattr(dc, "grid"):
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
            from ...hardware.registry import Hardware

            base_tdp = Hardware.Cloud.H100.tdp
        # Energy proportionality: Idle power is ~30% of TDP. Dynamic power scales with compute utilization (MFU).
        idle_power = base_tdp * cal.ENERGY_IDLE_FRACTION
        dynamic_power = base_tdp * cal.ENERGY_DYNAMIC_FRACTION * mfu
        effective_power_per_chip = idle_power + dynamic_power
        it_power_w = effective_power_per_chip * fleet.total_accelerators

        # 3. Energy Consumption
        it_energy_kwh = (it_power_w * Q_(duration_hours, "hour")).to("kWh")

        # Apply PUE
        pue = getattr(dc, "pue", fleet.effective_pue)
        total_energy_kwh = it_energy_kwh * pue

        # 4. Carbon Footprint (use total facility energy, PUE already applied)
        # Prefer the grid's own carbon_kg() helper; otherwise apply the raw
        # intensity, with /1000 converting gCO2e/kWh to kgCO2e/kWh.
        carbon_kg = (
            region.carbon_kg(total_energy_kwh.magnitude)
            if hasattr(region, "carbon_kg")
            else total_energy_kwh.magnitude * (region.carbon_intensity_g_kwh / 1000.0)
        )

        # 5. Water Usage (The Green Grid WUE standard: L / kWh of IT equipment energy)
        # Resolve WUE from dc.grid, dc, or region
        if hasattr(dc, "grid") and dc.grid:
            wue = dc.grid.wue
        elif hasattr(dc, "wue"):
            wue = dc.wue
        else:
            wue = region.wue

        water_liters = it_energy_kwh.magnitude * wue

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

    def solve(
        self,
        fleet: Fleet,
        duration_days: float,
        kwh_price: Optional[float] = None,
        datacenter: Optional[Any] = None,
        grid: Optional[Any] = None,
        mfu: float = 1.0,
        amortization_years: float = 3.0,
        infrastructure_multiplier: float = cal.DEFAULT_INFRASTRUCTURE_MULTIPLIER,
    ) -> EconomicsResult:
        """Calculate the TCO for a fleet over a specified duration.

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

        Notes
        -----
        Equation: TCO = amortized CapEx + energy OpEx + maintenance. Period CapEx = unit_cost * accelerator_count * infrastructure_multiplier * duration_days / (365 * amortization_years). Maintenance uses full CapEx; energy uses SustainabilityModel.

        Assumptions: Constant prices and straight-line amortization over a positive horizon; missing accelerator price is treated as zero and must be supplied for procurement use.
        Excludes: Financing, resale, tax, demand charges, and measured fleet billing.

        References
        ----------
        Derivation: Allocate purchase cost uniformly over the amortization horizon,
        charge maintenance on the full asset value, and add energy purchased during
        the period. All cost fields are in USD.

        Examples
        --------
        >>> from mlsysim import Q_, Systems
        >>> result = EconomicsModel().solve(Systems.Clusters.Research_256, duration_days=1)
        >>> result.tco_usd >= result.opex_energy_usd
        True
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
            price = getattr(target, "kwh_price", None)
            if price is None:
                price = Infrastructure.Pricing.Cloud.ElectricityPerKwh.rate.to("USD/kWh").magnitude
        price_val = price.to("USD/kWh").magnitude if hasattr(price, "to") else float(price)

        opex_energy = energy_result.total_energy_kwh.to("kWh").magnitude * price_val

        unit_cost = fleet.node.accelerator.unit_cost
        if unit_cost is None:
            # Unknown hardware price should not silently inherit an H100 cost.
            # Registries with sourced pricing populate unit_cost explicitly.
            unit_cost = Q_("0 USD")
        total_capex_hardware = unit_cost.to("USD").magnitude * fleet.total_accelerators
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

    requires = ("base_training_time",)
    produces = ResponsibleEngineeringResult

    def solve(
        self,
        base_training_time: Quantity,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        min_subgroup_prevalence: float = 0.01,
        sample_rate: float = 0.001,
        steps: int = 10000,
        clipping_step_overhead: float = 0.0,
    ) -> ResponsibleEngineeringResult:
        """Calculate DP-SGD Gaussian noise scale, step overhead, and subgroup sampling multiplier.

        Parameters
        ----------
        base_training_time : Quantity
            Positive training duration without DP-SGD per-example clipping (any time unit;
            the returned ``effective_training_time`` preserves this unit).
        epsilon : float, default: 1.0
            Positive differential-privacy budget ``epsilon``; smaller values require a
            larger Gaussian noise multiplier ``sigma``.
        delta : float, default: 1e-5
            Differential-privacy failure probability in ``(0, 1)``.
        min_subgroup_prevalence : float, default: 0.01
            Population prevalence ``p`` in ``(0, 1]`` of the rarest subgroup.
        sample_rate : float, default: 0.001
            Poisson subsampling rate ``q = L / N`` in ``(0, 1]`` per DP-SGD step.
        steps : int, default: 10000
            Positive count ``T`` of DP-SGD optimization steps.
        clipping_step_overhead : float, default: 0.0
            Nonnegative fractional per-step compute/memory overhead from per-example
            gradient clipping (e.g., ``0.25`` for a 1.25x step-time multiplier; ``0.0``
            when unmeasured). Note that privacy budget ``epsilon`` governs ``sigma``,
            whereas per-step wall-clock overhead comes from per-example clipping kernels.

        Returns
        -------
        ResponsibleEngineeringResult
            Gaussian mechanism noise multiplier ``noise_multiplier``, step-time multiplier
            ``dp_slowdown_factor``, ``effective_training_time``, and subgroup sampling
            multiplier ``additional_data_requirement``.

        Notes
        -----
        Equation: ``sigma = q * sqrt(2 * T * ln(1.25 / delta)) / epsilon``;
        ``dp_slowdown_factor = 1 + clipping_step_overhead``;
        ``effective_training_time = base_training_time * dp_slowdown_factor``;
        ``additional_data_requirement = 1 / min_subgroup_prevalence``.

        Assumptions: Subsampled Gaussian mechanism under moments-accountant / Gaussian
        composition scaling with subsampling ratio ``q`` and ``T`` steps; independent
        random sampling from a population where the target subgroup has prevalence ``p``.

        Excludes: Tight numerical Renyi-DP / PRV accountant constants, convergence-step
        increases needed to recover utility at high ``sigma``, and measured kernel traces.

        References
        ----------
        Dwork and Roth (2014), "The Algorithmic Foundations of Differential Privacy,"
        Theorem A.1. https://doi.org/10.1561/0400000042 (bibliography key: dwork2014).
        Abadi et al. (2016), "Deep Learning with Differential Privacy," Theorem 1.
        https://doi.org/10.1145/2976749.2978318 (bibliography key: abadi2016).

        Examples
        --------
        >>> from mlsysim import Q_
        >>> result = ResponsibleEngineeringModel().solve(
        ...     Q_("1 hour"), epsilon=1.0, clipping_step_overhead=0.25
        ... )
        >>> result.effective_training_time > Q_("1 hour") and result.noise_multiplier > 0
        True
        """
        from ...core._validation import (
            validate_at_least,
            validate_nonnegative,
            validate_positive,
            validate_range,
        )

        validate_positive(base_training_time, "base_training_time")
        validate_positive(epsilon, "epsilon")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if not (0.0 < min_subgroup_prevalence <= 1.0):
            raise ValueError(
                f"min_subgroup_prevalence must be in (0, 1], got {min_subgroup_prevalence}"
            )
        if not (0.0 < sample_rate <= 1.0):
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        validate_at_least(steps, 1, "steps")
        validate_nonnegative(clipping_step_overhead, "clipping_step_overhead")
        validate_range(sample_rate, 1e-12, 1.0, "sample_rate")

        noise_multiplier = (
            sample_rate * math.sqrt(2.0 * steps * math.log(1.25 / delta)) / epsilon
        )
        dp_slowdown = 1.0 + clipping_step_overhead
        additional_data_factor = 1.0 / min_subgroup_prevalence
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
            noise_multiplier=noise_multiplier,
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

    def solve(
        self,
        fleet: Fleet,
        duration_days: float,
        regions: Optional[List[str]] = None,
        carbon_tax_per_ton: float = 100.0,
        mfu: float = 1.0,
    ) -> PlacementOptimizerResult:
        """Determine the datacenter region minimizing TCO plus carbon tax.

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

        Notes
        -----
        Equation: objective = TCO_USD + carbon_kg / 1000 * carbon_tax_per_ton. Evaluate EconomicsModel independently for each recognized candidate and return the minimum.

        Assumptions: Identical fleet, duration, and MFU across candidates; at least one region must resolve. The candidate list defines the search space.
        Excludes: Network latency, data sovereignty, capacity availability, and provisioning constraints.

        References
        ----------
        Derivation: Enumerate the supplied regions and minimize the sum of their
        modeled financial cost and carbon mass priced in USD per metric ton. This is
        finite candidate search, not a continuous geographic optimum.

        Examples
        --------
        >>> from mlsysim import Q_, Systems
        >>> result = PlacementOptimizer().solve(Systems.Clusters.Research_256, duration_days=1, regions=["Quebec"])
        >>> result.best_region == "Quebec"
        True
        """
        from ...infrastructure.registry import Infrastructure

        if regions is None:
            regions = ["US_Avg", "Quebec", "Iowa"]
        econ_model = EconomicsModel()

        candidates = []

        for region_name in regions:
            grid = getattr(Infrastructure.Grids, region_name, None)
            if not grid:
                continue

            res = econ_model.solve(fleet, duration_days=duration_days, grid=grid, mfu=mfu)

            # Objective: TCO + Carbon Tax. Pricing carbon converts the
            # environmental externality into the same currency as TCO, so a
            # cheap-but-dirty grid can lose to a pricier low-carbon one.
            carbon_tons = res.carbon_footprint_kg / 1000.0  # kg -> metric tons
            total_cost = res.tco_usd + (carbon_tons * carbon_tax_per_ton)

            candidates.append(
                {
                    "region": region_name,
                    "tco": res.tco_usd,
                    "carbon": carbon_tons,
                    "pue": res.pue,
                    "objective": total_cost,
                }
            )

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
            top_candidates=top_n,
        )


__all__ = [
    "SustainabilityModel",
    "EconomicsModel",
    "ResponsibleEngineeringModel",
    "PlacementOptimizer",
]
