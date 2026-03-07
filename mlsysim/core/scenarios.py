from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional, Union, Dict, Any, List
from .constants import ureg, Q_
from .types import Quantity
from ..models.types import Workload, TransformerWorkload
from ..hardware.types import HardwareNode
from ..systems.types import Fleet, Node
from .exceptions import OOMError, SLAViolation
from .evaluation import SystemEvaluation, EvaluationLevel

class Scenario(BaseModel):
    """
    A Narrative Bundle tying a Workload, a System, and Performance Constraints.
    This is the primary entry point for student labs and textbook case studies.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    workload: Workload
    system: Union[Fleet, HardwareNode]
    
    # Constraints (SLAs)
    sla_latency: Optional[Quantity] = None
    target_accuracy: Optional[float] = None
    power_budget: Optional[Quantity] = None
    
    @property
    def is_distributed(self) -> bool:
        return isinstance(self.system, Fleet)

    def evaluate(self, batch_size: int = 1, precision: str = "fp16") -> SystemEvaluation:
        """
        Runs a full multi-level evaluation of the scenario.
        """
        from .engine import Engine
        from .solver import DistributedSolver, SustainabilitySolver, EconomicsSolver
        
        # 1. Resolve Hardware
        hardware = self.system.node.accelerator if self.is_distributed else self.system
        
        # --- LEVEL 1: FEASIBILITY ---
        weights = self.workload.size_in_bytes()
        feasible = weights <= hardware.memory.capacity
        f_status = "PASS" if feasible else "FAIL"
        
        # Dynamic unit scaling for summary
        unit = "MB" if weights < Q_("1 GB") else "GB"
        f_summary = f"Model fits in memory ({weights.to(unit):.1f} / {hardware.memory.capacity.to(unit):.1f})" if feasible else f"OOM: Requires {weights.to(unit):.1f} but only has {hardware.memory.capacity.to(unit):.1f}"
        
        l1 = EvaluationLevel(
            level_name="Feasibility", 
            status=f_status, 
            summary=f_summary,
            metrics={"weight_size": weights, "capacity": hardware.memory.capacity}
        )

        # --- LEVEL 2: PERFORMANCE ---
        if self.is_distributed:
            solver = DistributedSolver()
            perf = solver.solve(self.workload, self.system, batch_size=batch_size, precision=precision)
            actual_latency = perf["step_latency_total"]
            throughput = perf["effective_throughput"]
            perf_metrics = {
                "latency": actual_latency, 
                "throughput": throughput, 
                "scaling_eff": perf["scaling_efficiency"],
                "sla_latency": self.sla_latency
            }
        else:
            perf = Engine.solve(self.workload, self.system, batch_size=batch_size, precision=precision)
            actual_latency = perf.latency
            throughput = perf.throughput
            perf_metrics = {
                "latency": actual_latency, 
                "throughput": throughput, 
                "bottleneck": perf.bottleneck,
                "sla_latency": self.sla_latency
            }

        p_status = "PASS"
        if self.sla_latency and actual_latency > self.sla_latency:
            p_status = "FAIL"
        
        p_summary = f"Latency: {actual_latency:.2f} (Target: {self.sla_latency or 'N/A'})"
        l2 = EvaluationLevel(level_name="Performance", status=p_status, summary=p_summary, metrics=perf_metrics)

        # --- LEVEL 3: MACRO ---
        # Scale to 1 year operation for macro view
        if self.is_distributed:
             sim_fleet = self.system
        else:
             from ..systems.types import Node, Fleet
             from ..systems.registry import Fabrics
             dummy_node = Node(name="Standard", accelerator=hardware, accelerators_per_node=1, intra_node_bw="50 GB/s")
             sim_fleet = Fleet(name="SimFleet", node=dummy_node, count=1, fabric=Fabrics.Ethernet_10G)

        sust = SustainabilitySolver().solve(sim_fleet, duration_days=365)
        econ = EconomicsSolver().solve(sim_fleet, duration_days=365)
        
        m_summary = f"Annual Carbon: {sust['carbon_footprint_kg']:.1f} kg | TCO: ${econ['tco_usd']:,.0f}"
        l3 = EvaluationLevel(
            level_name="Macro", 
            status="PASS", 
            summary=m_summary,
            metrics={"carbon_kg": sust['carbon_footprint_kg'], "tco_usd": econ['tco_usd']}
        )

        return SystemEvaluation(
            scenario_name=self.name,
            feasibility=l1,
            performance=l2,
            macro=l3
        )

    def validate_scenario(self, batch_size: int = 1, precision: str = "fp16") -> Dict[str, Any]:
        """
        Comprehensive validation of the scenario's physical and performance feasibility.
        """
        from .engine import Engine
        from .solver import ServingSolver, DistributedSolver
        
        # 1. Resolve Hardware for memory check
        hardware = self.system.node.accelerator if self.is_distributed else self.system
        
        # 2. Memory Feasibility Check
        weights = self.workload.size_in_bytes()
        # For transformers, also check KV cache at a reasonable context (e.g., 512)
        if isinstance(self.workload, TransformerWorkload):
            kv_cache = self.workload.get_kv_cache_size(seq_len=512, batch_size=batch_size)
            total_mem = weights + kv_cache
        else:
            total_mem = weights
            
        if total_mem > hardware.memory.capacity:
            raise OOMError(
                f"Physical Failure: {self.name} requires {total_mem.to('GB')} but {hardware.name} only has {hardware.memory.capacity.to('GB')}.",
                required_bytes=total_mem,
                available_bytes=hardware.memory.capacity
            )

        # 3. Performance / SLA Check
        if self.is_distributed:
            solver = DistributedSolver()
            perf = solver.solve(self.workload, self.system, batch_size=batch_size, precision=precision)
            actual_latency = perf["step_latency_total"]
        else:
            perf = Engine.solve(self.workload, self.system, batch_size=batch_size, precision=precision)
            actual_latency = perf.latency

        if self.sla_latency and actual_latency > self.sla_latency:
            raise SLAViolation(
                f"SLA Violation: {self.name} actual latency {actual_latency} exceeds target {self.sla_latency}."
            )

        return {
            "status": "Validated",
            "memory_utilization": (total_mem / hardware.memory.capacity).to_base_units().magnitude,
            "performance": perf
        }

class Scenarios:
    """
    The Lighthouse Archetypes used throughout Volume 1 and Volume 2.
    """
    from ..models.registry import Models
    from ..hardware.registry import Hardware
    from ..systems.registry import Clusters, Nodes
    
    # --- TINYML WORLD ---
    SmartDoorbell = Scenario(
        name="Smart Doorbell",
        description="Identifying humans at the door using a sub-watt microcontroller.",
        workload=Models.Tiny.WakeVision,
        system=Hardware.Tiny.ESP32_S3,
        sla_latency=Q_("200 ms")
    )

    # --- EDGE WORLD ---
    AutonomousVehicle = Scenario(
        name="Autonomous Vehicle",
        description="Real-time object detection for safe urban navigation.",
        workload=Models.Vision.ResNet50,
        system=Hardware.Edge.JetsonOrinNX,
        sla_latency=Q_("10 ms")
    )

    # --- WORKSTATION WORLD ---
    LocalTraining = Scenario(
        name="Local LLM Fine-tuning",
        description="Fine-tuning a Llama-3 model on a high-end student workstation.",
        workload=Models.Language.Llama3_8B,
        system=Hardware.Workstation.MacBookM3Max,
        sla_latency=Q_("100 ms")
    )

    # --- CLOUD WORLD ---
    FrontierTraining = Scenario(
        name="Frontier LLM Training",
        description="Pre-training a 70B parameter foundation model on a massive fleet.",
        workload=Models.Language.Llama3_70B,
        system=Clusters.Frontier_8K,
        sla_latency=Q_("500 ms") # Per-step target
    )

class Applications:
    Doorbell = Scenarios.SmartDoorbell
    AutoDrive = Scenarios.AutonomousVehicle
    Workstation = Scenarios.LocalTraining
    Frontier = Scenarios.FrontierTraining
