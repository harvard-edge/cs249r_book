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
        """Check whether every scorecard level is passed or skipped.

        Parameters
        ----------

        Returns
        -------
        bool
            True when every level has status PASS or SKIPPED; false for WARNING, FAIL, or any other status. True does not certify that feasibility was evaluated.

        Notes
        -----
        Equation: passed_all = all(level.status in {PASS, SKIPPED} for feasibility, performance, macro).

        Assumptions: Status strings use the scorecard conventions; each producer is responsible for the checks behind its status.

        Excludes: Recomputing model results, checking omitted constraints, or converting skipped feasibility into evidence of capacity.

        References
        ----------
        Derivation: This is the conjunction of the accepted status predicate across the three scorecard levels.

        Examples
        --------
        >>> level = EvaluationLevel(level_name="example", summary="not evaluated", status="SKIPPED")
        >>> report = SystemEvaluation(scenario_name="example", feasibility=level, performance=level, macro=level)
        >>> report.passed_all
        True
        """
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
        """Compose performance and optional economics into a system scorecard.

        Parameters
        ----------
        scenario_name : str
            Display label carried into the result.
        model_obj : Any
            Workload registry object accepted by the selected solver.
        hardware_obj : Any
            Accelerator registry object used by the single-node path.
        batch_size : int
            Positive global samples or sequences per modeled step, as defined by the selected
            solver.
        precision : str
            Registered numerical precision such as fp16.
        efficiency : float
            Declared compute efficiency in (0, 1].
        fleet_obj : Any, optional
            Fleet definition. A supplied fleet with nodes greater than one selects
            DistributedModel; its own node count determines the modeled hardware.
        nodes : int, optional
            Path selector, default one. This argument does not resize fleet_obj.
        duration_days : float, optional
            Nonnegative modeled operating duration in days. Economics runs only when both
            fleet_obj and duration_days are truthy.

        Returns
        -------
        SystemEvaluation
            Feasibility and performance levels plus optional macro costs. Memory uses GB,
            latency ms, throughput 1/s, TCO USD, and carbon metric tons. Distributed memory
            feasibility is SKIPPED; performance PASS is not an SLA comparison.

        See Also
        --------
        mlsysim.engine.solvers.performance.SingleNodeModel.solve : Evaluate a single accelerator.
        mlsysim.engine.solvers.distributed.DistributedModel.solve : Evaluate distributed training.

        Notes
        -----
        Single-node memory utilization = footprint/capacity. Distributed effective MFU =
        node_mfu * scaling_efficiency. Macro carbon tons = EconomicsModel carbon_kg / 1000;
        other outputs are selected solver fields.

        Solver defaults apply, including TP=1 and PP=1 for the distributed path. The pipeline
        connects outputs by argument names; model and hardware must be compatible.

        The method does not provide sLA thresholds, distributed memory feasibility, custom ZeRO/accumulation settings, and
        measured deployment validation. Solver exceptions propagate.

        Examples
        --------
        >>> from mlsysim import Models, Hardware
        >>> r = SystemEvaluator.evaluate("small", Models.Vision.ResNet50,
        ...     Hardware.Cloud.H100, 1, "fp16", 0.5)
        >>> r.feasibility.status == "PASS" and r.macro.status == "SKIPPED"
        True
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
