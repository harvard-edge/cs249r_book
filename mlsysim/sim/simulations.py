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
from ..core.scenarios import Scenario
from ..core.engine import Engine
from ..core.solver import SustainabilitySolver
from ..hardware.types import HardwareNode
from ..systems.types import Fleet
from ..infra.registry import Infra

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
    def __init__(self, scenario: Scenario, persona: Persona):
        """Initializes the simulation with a static scenario and a persona.
        
        Args:
            scenario: The base model + hardware setup.
            persona: The scaling and narrative context.
        """
        self.scenario = scenario
        self.persona = persona

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
        # 1. BASE PERFORMANCE (Single Node)
        workload = self.scenario.workload
        hardware = self.scenario.system.node.accelerator if isinstance(self.scenario.system, Fleet) else self.scenario.system
        
        perf_base = Engine.solve(workload, hardware)
        mfu_val = (perf_base.latency_compute / perf_base.latency).to_base_units().magnitude
        
        # 2. EXTRACT USER CHOICES
        region_name = choice.get("region", "US_Avg")
        grid = getattr(Infra.Grids, region_name, Infra.Grids.US_Avg)
        duration_days = float(choice.get("duration_days", 365.0))
        
        # 3. SCALE TO FLEET (Persona Context)
        # Handle scaling factor (e.g. 1000 sensors or 100 clusters)
        scale = self.persona.scale_factor
        
        # Create a virtual fleet for the solver if scenario is single-node
        if isinstance(self.scenario.system, Fleet):
             sim_fleet = self.scenario.system
        else:
             from ..systems.types import Node, Fleet
             from ..systems.registry import Fabrics
             dummy_node = Node(name="Standard", accelerator=hardware, accelerators_per_node=1, intra_node_bw="50 GB/s")
             sim_fleet = Fleet(name="SimFleet", node=dummy_node, count=int(scale), fabric=Fabrics.Ethernet_10G)

        # 4. SUSTAINABILITY MATH
        sust_solver = SustainabilitySolver()
        impact = sust_solver.solve(sim_fleet, duration_days=duration_days, datacenter=grid)
        
        # 5. ECONOMIC MATH
        electricity_cost = impact["total_energy_kwh"].magnitude * 0.12 
        hw_cost_per_unit = hardware.unit_cost.magnitude if hardware.unit_cost else 30000.0
        total_capex = hw_cost_per_unit * sim_fleet.total_accelerators
        
        # 6. ASSEMBLE UNIVERSAL LEDGER
        ledger = SystemLedger(
            mission_name=self.scenario.name,
            track_name=self.persona.name,
            choice_summary=f"Region: {grid.name}, Duration: {duration_days} days",
            performance=PerformanceMetrics(
                latency=perf_base.latency,
                throughput=perf_base.throughput * sim_fleet.total_accelerators,
                mfu=mfu_val,
                hfu=mfu_val * 1.1, 
                bottleneck=perf_base.bottleneck
            ),
            sustainability=SustainabilityMetrics(
                energy=impact["total_energy_kwh"],
                carbon_kg=impact["carbon_footprint_kg"],
                pue=impact["pue"],
                water_liters=impact["water_usage_liters"]
            ),
            economics=EconomicMetrics(
                capex=total_capex,
                opex=electricity_cost,
                tco=total_capex + electricity_cost,
                cost_per_million=(electricity_cost / (perf_base.throughput.magnitude * duration_days * 24 * 3600 * sim_fleet.total_accelerators + 1e-9)) * 1e6
            ),
            reliability=ReliabilityMetrics(
                mttf=Q_("50000 hours") / sim_fleet.total_accelerators,
                goodput=0.95,
                recovery_time=Q_("15 minutes")
            )
        )
        return ledger
