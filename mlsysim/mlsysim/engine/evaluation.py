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
    The multi-level scorecard for an analytical system evaluation.
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
            lines.append(f"║ Level {idx+1}: {name} [{level.status}]")
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
        return all(l.status in {"PASS", "SKIPPED"} for l in [self.feasibility, self.performance, self.macro])

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
        Evaluate an ML system scenario across three analytical lenses.

        Composes a pipeline from the inputs and maps its results onto the
        three-level scorecard:

        - **Feasibility** ("will it run?"): memory footprint vs HBM capacity
          from ``SingleNodeModel`` (single node only; the distributed path
          currently reports a pass-through check);
        - **Performance** ("is it fast enough?"): latency/throughput/MFU from
          ``SingleNodeModel``, or step latency / scaling efficiency from
          ``DistributedModel`` when ``nodes > 1`` and a fleet is given;
        - **Macro** ("is it worth it?"): TCO/carbon/energy from
          ``EconomicsModel``, included only when both ``fleet_obj`` and
          ``duration_days`` are provided (otherwise SKIPPED).

        Parameters
        ----------
        scenario_name : str
            Label echoed into the scorecard.
        model_obj, hardware_obj : Any
            Workload and accelerator registry objects.
        batch_size : int
            Samples per step.
        precision : str
            Numerical precision ('fp16', 'fp32', ...).
        efficiency : float
            Software efficiency factor in (0, 1].
        fleet_obj : Any, optional
            Fleet object; enables the distributed and economics paths.
        nodes : int
            Node count; > 1 with a fleet selects ``DistributedModel``.
        duration_days : float, optional
            Operation duration for the economics lens.

        Returns
        -------
        SystemEvaluation
            The three-level scorecard (metric units: memory in GB, latency in
            ms, throughput in 1/s, TCO in USD, carbon in metric tons).
        """
        
        from .solvers import SingleNodeModel, DistributedModel, EconomicsModel
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
            tp_size=1,
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
                metrics={
                    "memory_used_gb": profile.memory_footprint.to('GB').magnitude,
                    "memory_capacity_gb": hardware_obj.memory.capacity.to('GB').magnitude,
                    "memory_utilization": (profile.memory_footprint / hardware_obj.memory.capacity).to_base_units().magnitude,
                }
            )
            performance = EvaluationLevel(
                level_name="Single Node Performance",
                status="PASS" if profile.feasible else "FAIL",
                summary=f"{profile.bottleneck} Bound" if profile.feasible else f"{profile.bottleneck} Bound (infeasible)",
                metrics={
                    "latency": profile.latency.m_as("ms"),
                    "throughput": profile.throughput.m_as("1/s"),
                    "mfu": profile.mfu
                }
            )
        elif "DistributedModel" in results:
            dist_res = results["DistributedModel"]
            # Fleet-level MFU compounds two losses: per-node software efficiency
            # (node MFU) and the distributed tax (communication + bubbles).
            effective_mfu = dist_res.node_profile.mfu * dist_res.scaling_efficiency
            # The distributed path does not run a memory-feasibility check here:
            # correctly sizing per-accelerator training memory needs the ZeRO
            # stage and gradient-accumulation (microbatch) assumptions this entry
            # point does not thread through to the model. Report SKIPPED rather
            # than a no-op PASS, so the scorecard never claims "will run" without
            # having checked. To evaluate it, thread zero_stage +
            # gradient_accumulation_steps into evaluate() and call
            # TrainingMemoryModel for the distributed feasibility lens.
            feasibility = EvaluationLevel(
                level_name="Feasibility",
                status="SKIPPED",
                summary="Memory feasibility not evaluated for the distributed path "
                        "(requires ZeRO/accumulation assumptions; size it with TrainingMemoryModel)",
                metrics={}
            )
            performance = EvaluationLevel(
                level_name="Fleet Performance",
                status="PASS",
                summary=f"Scaling Efficiency: {dist_res.scaling_efficiency:.1%}",
                metrics={
                    "step_latency": dist_res.step_latency_total.m_as("ms"),
                    "comm_overhead": dist_res.communication_latency.m_as("ms"),
                    "fleet_throughput": dist_res.effective_throughput.m_as("1/s"),
                    "mfu": effective_mfu,
                    "node_mfu": dist_res.node_profile.mfu,
                    "scaling_efficiency": dist_res.scaling_efficiency,
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
                    "tco_usd": econ_res.tco_usd,
                    "carbon_footprint": econ_res.carbon_footprint_kg / 1000.0,  # kg -> metric tons
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
