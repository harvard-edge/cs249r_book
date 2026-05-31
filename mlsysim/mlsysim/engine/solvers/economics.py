"""Sustainability, economics, responsible-engineering, and placement solvers.

These implementations live outside ``engine.solver`` so the public compatibility
module can stay small while domain logic remains easier to review.
"""

# ruff: noqa: F401
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Type

from ..engine import Engine, PerformanceProfile
from ..results import (
    SolverResult,
    DistributedResult,
    ReliabilityResult,
    CheckpointResult,
    SustainabilityResult,
    ServingResult,
    TrainingMemoryResult,
    ServingCapacityResult,
    MoERoutingResult,
    ContinuousBatchingResult,
    WeightStreamingResult,
    TailLatencyResult,
    EconomicsResult,
    DataResult,
    TopologyResult,
    EfficiencyResult,
    TransformationResult,
    ScalingResult,
    CompressionResult,
    SynthesisResult,
    OrchestrationResult,
    InferenceScalingResult,
    SensitivityResult,
    ResponsibleEngineeringResult,
    ParallelismOptimizerResult,
    BatchingOptimizerResult,
    PlacementOptimizerResult,
)
from ...physics import (
    calc_ring_allreduce_time,
    calc_hierarchical_allreduce_time,
    calc_all_to_all_time,
    calc_bottleneck,
    calc_mtbf_cluster,
    calc_mtbf_node,
    calc_young_daly_interval,
    calc_failure_probability,
    calc_pipeline_bubble,
)
from ...core.constants import ureg, Q_, resolve_precision
from ...infrastructure.registry import Infrastructure
from ...literature.registry import Literature
from ...systems.reliability import Reliability
from .. import calibration as cal
from ...core.types import Quantity
from ...models.types import Workload, TransformerWorkload, SparseTransformerWorkload
from ...hardware.types import HardwareNode
from ...systems.types import Fleet, NetworkFabric, Node
from ...infrastructure.types import Datacenter
from .base import BaseModel, BaseOptimizer, BaseResolver, BaseSolver, ForwardModel
from .utils import _inter_node_latency, _intra_node_latency

class SustainabilityModel(BaseModel):
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
        Calculates energy, carbon, and water footprint for a fleet operation.
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

        # 2. Power
        base_tdp = fleet.node.accelerator.tdp if fleet.node.accelerator.tdp else (700 * ureg.watt)
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

class EconomicsModel(BaseModel):
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

        Returns
        -------
        Dict[str, Any]
            Financial metrics including CapEx, OpEx, and total TCO.
        """
        sust_model = SustainabilityModel()
        energy_result = sust_model.solve(fleet, duration_days, datacenter=datacenter or grid, mfu=mfu)

        price = kwh_price
        if price is None:
            # Try to resolve from grid/datacenter or default
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
        # Amortize CapEx over deployment period (default 3-year depreciation schedule)
        capex_for_period = (total_capex / amortization_years) * (duration_days / 365.0)

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

class ResponsibleEngineeringModel(BaseModel):
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
        Calculates the overhead of responsible engineering practices.
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
    Finds the optimal datacenter location to minimize TCO and Carbon.
    """
    requires = ("fleet", "duration_days")
    produces = PlacementOptimizerResult

    def solve(self, fleet: Fleet, duration_days: float,
              regions: List[str] = ["US_Avg", "Quebec", "Iowa"],
              carbon_tax_per_ton: float = 100.0, mfu: float = 1.0) -> PlacementOptimizerResult:
        """
        Determines the optimal data center location to minimize TCO and carbon taxes.
        """
        from ...infrastructure.registry import Infrastructure
        econ_model = EconomicsModel()

        candidates = []

        for region_name in regions:
            grid = getattr(Infrastructure.Grids, region_name, None)
            if not grid: continue

            res = econ_model.solve(fleet, duration_days=duration_days, grid=grid, mfu=mfu)

            # Objective: TCO + Carbon Tax
            carbon_tons = res.carbon_footprint_kg / 1000.0
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
