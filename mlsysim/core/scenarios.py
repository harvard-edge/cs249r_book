from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional, Union, Dict, Any, List
from .constants import ureg, Q_
from .types import Quantity
from ..models.types import Workload, TransformerWorkload
from ..hardware.types import HardwareNode
from ..systems.types import Fleet, Node
from .exceptions import OOMError, SLAViolation
from .evaluation import SystemEvaluation, EvaluationLevel
from typing import Optional, Union, Dict, Any, List, ClassVar

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

    _legacy_aliases: ClassVar[dict[str, Any]] = {
        "mission_goal": lambda self: self.description,
        "critical_constraint": lambda self:
            self.sla_latency
            if self.sla_latency is not None
            else self.target_accuracy
            if self.target_accuracy is not None
            else self.power_budget,
        "ram": lambda self:
            self.system.node.accelerator.ram
            if isinstance(self.system, Fleet)
            else self.system.ram,
        "hardware": lambda self:
            self.system.node.accelerator
            if isinstance(self.system, Fleet)
            else self.system,
    }

    def __getattr__(self, name):
        aliases = type(self)._legacy_aliases
        if name in aliases:
            value = aliases[name](self)
            if value is not None:
                return value
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    @property
    def is_distributed(self) -> bool:
        return isinstance(self.system, Fleet)

    def evaluate(self, batch_size: int = 1, precision: str = "fp16") -> SystemEvaluation:
        """
        Runs a full multi-level evaluation of the scenario.
        """
        from .engine import Engine
        from .solver import DistributedModel, EconomicsModel
        
        # 1. Resolve Hardware
        hardware = self.system.node.accelerator if self.is_distributed else self.system
        
        # --- LEVEL 1: FEASIBILITY ---
        from .solver import DataModel
        weights = self.workload.size_in_bytes()
        mem_feasible = weights <= hardware.memory.capacity
        
        # Data Pipeline Check
        data_status = "PASS"
        data_summary = ""
        if self.workload.data_rate:
            ds = DataModel().solve(self.workload.data_rate, hardware)
            if ds.is_stalled:
                data_status = "FAIL"
                data_summary = f" | Data Wall: Pipeline Stalled ({ds.utilization:.1f}x capacity)"
            else:
                data_summary = f" | Data Pipeline: OK ({ds.utilization*100:.1f}%)"

        feasible = mem_feasible and (data_status == "PASS")
        f_status = "PASS" if feasible else "FAIL"
        
        # Dynamic unit scaling for summary
        unit = "MB" if weights < Q_("1 GB") else "GB"
        mem_summary = f"Model fits in memory ({weights.to(unit):.1f} / {hardware.memory.capacity.to(unit):.1f})" if mem_feasible else f"OOM: Requires {weights.to(unit):.1f} but only has {hardware.memory.capacity.to(unit):.1f}"
        
        l1 = EvaluationLevel(
            level_name="Feasibility", 
            status=f_status, 
            summary=mem_summary + data_summary,
            metrics={"weight_size": weights, "capacity": hardware.memory.capacity, "data_stalled": data_status == "FAIL"}
        )

        # --- LEVEL 2: PERFORMANCE ---
        if self.is_distributed:
            solver = DistributedModel()
            perf = solver.solve(self.workload, self.system, batch_size=batch_size, precision=precision)
            actual_latency = perf.step_latency_total
            throughput = perf.effective_throughput
            perf_metrics = {
                "latency": actual_latency,
                "throughput": throughput,
                "scaling_eff": perf.scaling_efficiency,
                "sla_latency": self.sla_latency
            }
        else:
            from .solver import SingleNodeModel
            perf = SingleNodeModel().solve(self.workload, self.system, batch_size=batch_size, precision=precision)
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

        # EconomicsModel internally delegates to SustainabilityModel and merges
        # its results, so we only need one call (avoids computing energy twice).
        econ = EconomicsModel().solve(sim_fleet, duration_days=365)

        m_summary = f"Annual Carbon: {econ.carbon_footprint_kg:.1f} kg | TCO: ${econ.tco_usd:,.0f}"
        l3 = EvaluationLevel(
            level_name="Macro",
            status="PASS",
            summary=m_summary,
            metrics={"carbon_kg": econ.carbon_footprint_kg, "tco_usd": econ.tco_usd}
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
        from .solver import ServingModel, DistributedModel, SingleNodeModel
        
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
            solver = DistributedModel()
            perf = solver.solve(self.workload, self.system, batch_size=batch_size, precision=precision)
            actual_latency = perf.step_latency_total
        else:
            perf = SingleNodeModel().solve(self.workload, self.system, batch_size=batch_size, precision=precision)
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
        sla_latency=Q_("200 ms"),
        power_budget=Q_("100 mW"),
    )

    TinySensor = Scenario(
        name="Anomaly Sensor",
        description="Low-power vibration monitoring for industrial predictive maintenance.",
        workload=Models.Tiny.AnomalyDetector,
        system=Hardware.Tiny.ESP32_S3,
        sla_latency=Q_("10 ms"),
        power_budget=Q_("50 mW"),
    )

    # --- EDGE WORLD ---
    AutonomousVehicle = Scenario(
        name="Autonomous Vehicle",
        description="Real-time object detection for safe urban navigation.",
        workload=Models.Vision.ResNet50,
        system=Hardware.Edge.JetsonOrinNX,
        sla_latency=Q_("10 ms")
    )

    AutonomousVehicle_Waymo = Scenario(
        name="Waymo AV Data Pipeline",
        description="High-throughput data ingestion for autonomous fleet training.",
        workload=Models.Vision.ResNet50.model_copy(update={"name": "Waymo (High)", "data_rate": Q_("19 TB/hour")}),
        system=Hardware.Edge.JetsonOrinNX,
        sla_latency=Q_("10 ms")
    )

    # --- MOBILE WORLD ---
    MobileHealth = Scenario(
        name="Mobile Health",
        description="On-device medical image analysis for remote diagnostics.",
        workload=Models.Vision.MobileNetV2,
        system=Hardware.Mobile.iPhone15Pro,
        sla_latency=Q_("30 ms")
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

    # --- SERVING WORLD ---
    ChatbotServing = Scenario(
        name="Chatbot Serving",
        description="Serving a Llama-3 8B chatbot on a single H100 with latency SLA.",
        workload=Models.Language.Llama3_8B,
        system=Hardware.Cloud.H100,
        sla_latency=Q_("500 ms"),  # TTFT target
    )

    # --- TINYML: KEYWORD SPOTTING ---
    KeywordSpotting = Scenario(
        name="Keyword Spotting",
        description="Always-on wake-word detection on a microcontroller (MLPerf Tiny benchmark).",
        workload=Models.Tiny.DS_CNN,
        system=Hardware.Tiny.ESP32_S3,
        sla_latency=Q_("30 ms"),
        power_budget=Q_("1 mW"),
    )

class Applications:
    Doorbell = Scenarios.SmartDoorbell
    AutoDrive = Scenarios.AutonomousVehicle
    Workstation = Scenarios.LocalTraining
    Frontier = Scenarios.FrontierTraining
    Chatbot = Scenarios.ChatbotServing
    KWS = Scenarios.KeywordSpotting

class Archetypes:
    """
    Backward-compatible namespace for older textbook notebooks/qmd files.
    """

    # Legacy system/application archetype aliases
    Cloud_V100 = Scenarios.FrontierTraining
    TinyML_M7 = Scenarios.SmartDoorbell

    # Optional convenience aliases
    Frontier = Applications.Frontier
    AutoDrive = Applications.AutoDrive
    Doorbell = Applications.Doorbell
    Workstation = Applications.Workstation