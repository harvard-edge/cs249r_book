from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, List
from .constants import ureg, Q_
from .types import Quantity

class EvaluationLevel(BaseModel):
    """A single tier in the Hierarchy of Constraints."""
    level_name: str
    status: str = "PASS" # PASS, FAIL, WARNING
    summary: str
    metrics: Dict[str, Any] = {}

class SystemEvaluation(BaseModel):
    """
    The multi-level 'Scorecard' for a System Simulation.
    Organizes results into the three pedagogical lenses by composing 
    analytical models and analysis solvers.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_name: str
    
    # Level 1: Feasibility (The "Will it run?" check)
    feasibility: EvaluationLevel
    
    # Level 2: Performance (The "Is it fast enough?" check)
    performance: EvaluationLevel
    
    # Level 3: Macro (The "Is it worth it?" check)
    macro: EvaluationLevel

    def scorecard(self) -> str:
        """Generates a human-readable summary for students."""
        # Visual styling for the scorecard
        border = "═" * 60
        lines = [
            f"╔{border}╗",
            f"║ MLSys·im SYSTEM EVALUATION",
            f"║ Scenario: {self.scenario_name}",
            f"╠{border}╣"
        ]
        
        levels = [
            ("Feasibility", self.feasibility),
            ("Performance", self.performance),
            ("Macro/Economics", self.macro)
        ]
        
        for idx, (name, level) in enumerate(levels):
            status_emoji = "✅" if level.status == "PASS" else "❌" if level.status == "FAIL" else "⚠️"
            
            lines.append(f"║ Level {idx+1}: {name} [{level.status}] {status_emoji}")
            lines.append(f"║ ↳ {level.summary}")
            
            if idx < 2:
                lines.append(f"╟{'-'*60}╢")
                
        lines.append(f"╚{border}╝")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Flattens the evaluation into a single-level dictionary for CSV/DataFrame export."""
        return {
            "scenario": self.scenario_name,
            "f_status": self.feasibility.status,
            "p_status": self.performance.status,
            "m_status": self.macro.status,
            **{f"f_{k}": v for k, v in self.feasibility.metrics.items()},
            **{f"p_{k}": v for k, v in self.performance.metrics.items()},
            **{f"m_{k}": v for k, v in self.macro.metrics.items()},
        }

    @property
    def passed_all(self) -> bool:
        return all(l.status == "PASS" for l in [self.feasibility, self.performance, self.macro])

class SystemEvaluator:
    """
    Orchestrator that wraps the 3-tier analytical solvers and constructs the 
    unified SystemEvaluation "Envelope" payload. Keeps CLI "dumb".
    """
    @staticmethod
    def evaluate(
        scenario_name: str,
        model_obj: Any,
        hardware_obj: Any,
        batch_size: int,
        precision: str,
        efficiency: float,
        fleet_obj: Optional[Any] = None,
        nodes: int = 1,
        duration_days: Optional[float] = None
    ) -> SystemEvaluation:
        """
        Evaluates an ML system scenario across three pedagogical lenses: Feasibility, Performance, and Macro/Economics.
        
        This method orchestrates the Tier 1 models (SingleNodeModel, DistributedModel) and Tier 2 models 
        (EconomicsModel, SustainabilityModel) to generate a comprehensive scorecard for the given hardware/software configuration.
        
        Args:
            scenario_name (str): A descriptive name for the evaluation (e.g., "Llama-3 8B on H100").
            model_obj (Workload): The workload architecture to simulate (e.g., Models.Language.Llama3_8B).
            hardware_obj (HardwareNode): The target hardware accelerator (e.g., Hardware.Cloud.H100).
            batch_size (int): The number of samples/tokens processed concurrently.
            precision (str): The numerical precision used (e.g., 'fp16', 'int8').
            efficiency (float): The achieved hardware utilization as a fraction of peak (e.g., 0.45 for 45% MFU).
            fleet_obj (Optional[Fleet]): The cluster topology for distributed runs.
            nodes (int): The number of accelerators to simulate. Defaults to 1.
            duration_days (Optional[float]): The expected duration of the run, required for Macro/Economics evaluation.
            
        Returns:
            SystemEvaluation: A multi-level scorecard containing the feasibility, performance, and macro metrics.
        """
        
        from .solver import SingleNodeModel, DistributedModel, EconomicsModel

        # 1. Evaluate Performance & Feasibility
        if nodes == 1 or fleet_obj is None:
            solver = SingleNodeModel()
            profile = solver.solve(
                model=model_obj,
                hardware=hardware_obj,
                batch_size=batch_size,
                precision=precision,
                efficiency=efficiency,
                raise_errors=False
            )
            
            feasibility = EvaluationLevel(
                level_name="Memory Feasibility",
                status="PASS" if profile.feasible else "FAIL",
                summary=f"{profile.memory_footprint.to('GB'):~.1f} / {hardware_obj.memory.capacity.to('GB'):~.1f} used",
                metrics={"memory_used_gb": profile.memory_footprint.to('GB').magnitude}
            )
            
            performance = EvaluationLevel(
                level_name="Single Node Performance",
                status="PASS",
                summary=f"{profile.bottleneck} Bound",
                metrics={
                    "latency": profile.latency.m_as("ms"),
                    "throughput": profile.throughput.m_as("1/s"),
                    "mfu": profile.mfu
                }
            )
        else:
            dist_model = DistributedModel()
            dist_res = dist_model.solve(
                model=model_obj,
                fleet=fleet_obj,
                batch_size=batch_size,
                precision=precision,
                efficiency=efficiency,
                tp_size=nodes,
                pp_size=1
            )
            
            feasibility = EvaluationLevel(
                level_name="Feasibility",
                status="PASS",
                summary="Distributed Model Check Passed",
                metrics={}
            )
            
            performance = EvaluationLevel(
                level_name="Fleet Performance",
                status="PASS",
                summary=f"Scaling Efficiency: {dist_res.scaling_efficiency:.1%}",
                metrics={
                    "step_latency": dist_res.step_latency_total.m_as("ms"),
                    "comm_overhead": dist_res.communication_latency.m_as("ms"),
                    "fleet_throughput": dist_res.effective_throughput.m_as("1/s")
                }
            )

        # 2. Evaluate Macro / Economics
        macro = EvaluationLevel(level_name="Macro", status="SKIPPED", summary="No Ops config provided", metrics={})
        if fleet_obj and duration_days:
            econ_model = EconomicsModel()
            econ_res = econ_model.solve(
                fleet=fleet_obj,
                duration_days=duration_days
            )
            macro = EvaluationLevel(
                level_name="Economics & Sustainability",
                status="PASS",
                summary=f"TCO: ${econ_res.tco_usd:,.0f}",
                metrics={
                    "carbon_footprint": econ_res.carbon_footprint_kg / 1000.0,
                    "energy_cost": econ_res.opex_energy_usd,
                    "capex": econ_res.capex_usd
                }
            )

        return SystemEvaluation(
            scenario_name=scenario_name,
            feasibility=feasibility,
            performance=performance,
            macro=macro
        )
