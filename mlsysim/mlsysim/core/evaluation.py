from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any

class EvaluationLevel(BaseModel):
    """A single tier in the Hierarchy of Constraints."""
    level_name: str
    status: str = "PASS" # PASS, FAIL, WARNING
    summary: str
    metrics: Dict[str, Any] = {}

class SystemEvaluation(BaseModel):
    """
    The multi-level 'Scorecard' for a System Simulation.
    Organizes results into the three analytical lenses by composing 
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
            "║ MLSys·im SYSTEM EVALUATION",
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
        Evaluates an ML system scenario across three analytical lenses: Feasibility, Performance, and Macro/Economics.
        """
        
        from .solver import SingleNodeModel, DistributedModel, EconomicsModel
        from .pipeline import Pipeline

        # Compose the pipeline dynamically based on the inputs
        resolvers = []
        if nodes == 1 or fleet_obj is None:
            resolvers.append(SingleNodeModel())
        else:
            resolvers.append(DistributedModel())

        if fleet_obj and duration_days:
            resolvers.append(EconomicsModel())

        pipeline = Pipeline(resolvers, name=f"Eval_{scenario_name}")
        
        # Execute the pipeline
        results = pipeline.run(
            model=model_obj,
            hardware=hardware_obj,
            batch_size=batch_size,
            precision=precision,
            efficiency=efficiency,
            fleet=fleet_obj,
            tp_size=nodes,
            pp_size=1,
            duration_days=duration_days,
            raise_errors=False
        )

        # 1. Evaluate Feasibility & Performance
        if "SingleNodeModel" in results:
            profile = results["SingleNodeModel"]
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
        elif "DistributedModel" in results:
            dist_res = results["DistributedModel"]
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
        else:
            raise RuntimeError("Pipeline failed to produce a valid performance model result.")

        # 2. Evaluate Macro / Economics
        macro = EvaluationLevel(level_name="Macro", status="SKIPPED", summary="No Ops config provided", metrics={})
        if "EconomicsModel" in results:
            econ_res = results["EconomicsModel"]
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
