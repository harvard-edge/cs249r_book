# simulations.py
"""
MLSys Analytical Simulations
============================
This module provides domain-specific analytical solvers for lab simulations.
Each simulation class implements the 'Physics' of a specific engineering domain.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional
from ..core.constants import ureg, Q_, HOURS_PER_DAY
from .ledger import SystemLedger, PerformanceMetrics, SustainabilityMetrics, EconomicMetrics, ReliabilityMetrics
from .personas import Persona, Personas
from ..core.scenarios import ApplicationScenario, ClusterScenario
from ..core.engine import Engine
from ..core.systems import SystemArchetype, Systems
from ..core.datacenters import Datacenters

@dataclass
class SimulationResult:
    """The complete outcome of a simulation evaluation, including raw data for plots."""
    ledger: SystemLedger
    plots: Dict[str, Any]

class BaseSimulation:
    """
    Abstract Base Class for all Analytical Simulations.
    Provides the standard 'evaluate' interface for student choice processing.
    """
    def __init__(self, scenario: Union[ApplicationScenario, ClusterScenario], persona: Persona):
        """Initializes the simulation with a static scenario and a persona.
        
        Args:
            scenario: The base model + hardware setup.
            persona: The scaling and narrative context.
        """
        self.scenario = scenario
        self.persona = persona

    def _get_system_archetype(self) -> SystemArchetype:
        """Helper to unify Application and Cluster scenarios for the Engine.
        
        Returns:
            A SystemArchetype object compatible with Engine.solve().
        """
        if hasattr(self.scenario, "system"):
            return self.scenario.system
        
        if hasattr(self.scenario, "cluster"):
            cluster = self.scenario.cluster
            return SystemArchetype(
                name=f"Virtual Node ({cluster.node.name})",
                hardware=cluster.node.accelerator,
                tier=Systems.Cloud.tier,
                network_bw=cluster.fabric.bandwidth,
                power_budget=cluster.node.node_tdp or Q_("700 watt")
            )
        
        return Systems.Cloud

    def evaluate(self, choice: Dict[str, Any]) -> SystemLedger:
        """Processes a student's choice and returns a Ledger.
        
        Args:
            choice: A dictionary of parameters from the UI (e.g., {'region': 'Quebec'}).
            
        Returns:
            A SystemLedger containing the multi-dimensional results.
        """
        raise NotImplementedError

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE SIMULATION: Sustainability, Carbon, and TCO
# ─────────────────────────────────────────────────────────────────────────────

class ResourceSimulation(BaseSimulation):
    """
    Analyzes energy consumption, carbon footprint, and economic TCO.
    
    This simulation handles regional grid math and fleet-wide power scaling.
    """
    def evaluate(self, choice: Dict[str, Any]) -> SystemLedger:
        # 1. BASE PERFORMANCE
        system = self._get_system_archetype()
        perf_base = Engine.solve(self.scenario.model, system)
        mfu_val = (perf_base.latency_compute / perf_base.latency).to_base_units().m
        
        # 2. EXTRACT USER CHOICES
        region_name = choice.get("region", "US_Avg")
        grid = getattr(Datacenters.Grids, region_name, Datacenters.Grids.US_Avg)
        duration_days = float(choice.get("duration_days", 365.0))
        
        # 3. SCALE TO FLEET (Persona Context)
        scale = self.persona.scale_factor
        
        # IT Energy (kWh) = Power(W) * Time(h) / 1000
        it_power_w = (perf_base.energy / perf_base.latency).to(ureg.watt).m
        total_hours = duration_days * HOURS_PER_DAY
        it_energy_kwh = (it_power_w * total_hours * scale) / 1000.0
        
        # 4. APPLY PHYSICAL INVARIANTS (Sustainability)
        total_energy_kwh = it_energy_kwh * grid.pue
        total_carbon_kg = grid.carbon_kg(it_energy_kwh)
        
        # 5. ECONOMIC MATH
        electricity_cost = total_energy_kwh * 0.12 
        hw_cost_per_unit = 10.0 if system.tier.name == "Tiny" else 30000.0
        total_capex = hw_cost_per_unit * scale
        
        # 6. ASSEMBLE UNIVERSAL LEDGER
        ledger = SystemLedger(
            mission_name="Global Efficiency Challenge",
            track_name=self.persona.name,
            choice_summary=f"Region: {grid.name}, Duration: {duration_days} days",
            performance=PerformanceMetrics(
                latency=perf_base.latency,
                throughput=perf_base.throughput * scale,
                mfu=mfu_val,
                hfu=mfu_val * 1.1, 
                bottleneck=perf_base.bottleneck
            ),
            sustainability=SustainabilityMetrics(
                energy=total_energy_kwh * ureg.kilowatt_hour,
                carbon_kg=total_carbon_kg,
                pue=grid.pue,
                water_liters=total_energy_kwh * grid.wue
            ),
            economics=EconomicMetrics(
                capex=total_capex,
                opex=electricity_cost,
                tco=total_capex + electricity_cost,
                cost_per_million=(electricity_cost / (perf_base.throughput.m * total_hours * scale * 3600)) * 1e6
            ),
            reliability=ReliabilityMetrics(
                mttf=Q_("100 hours"),
                goodput=0.95,
                recovery_time=Q_("15 minutes")
            )
        )
        ledger.validate()
        return ledger
