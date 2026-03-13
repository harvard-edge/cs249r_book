import typer
from pathlib import Path
import yaml
from typing import Optional
from mlsysim.cli.schemas import EvalNodeSchema, MlsysPlanSchema
from mlsysim.cli.exceptions import ExitCode, exit_with_code, error_shield
from mlsysim.cli.renderers import render_scorecard, print_warning, print_error
from mlsysim.core.solver import SingleNodeModel, DistributedModel, EconomicsModel
from mlsysim.core.evaluation import SystemEvaluation, EvaluationLevel

def evaluate_main(
    ctx: typer.Context,
    target: str = typer.Argument(..., help="Path to mlsys.yaml OR Model name (e.g. Llama3_8B)"),
    hardware: Optional[str] = typer.Argument(None, help="Hardware name (e.g. H100) - Required if target is not a YAML file"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size"),
    precision: str = typer.Option("fp16", "--precision", "-p", help="Precision (fp16, fp8, etc)"),
    efficiency: float = typer.Option(0.5, "--efficiency", "-e", help="Target Model FLOPs Utilization (0.0 to 1.0)")
):
    """[Tier 1] Evaluate the analytical physics of an ML system (via YAML or CLI flags)."""
    
    # Context should be initialized by main.py
    output_format = ctx.obj.get("output_format", "text") if ctx.obj else "text"
    is_json = output_format == "json"
    
    with error_shield(output_format=output_format):
        # Branch 1: Deep Evaluation (YAML / IaC)
        if target.endswith(".yaml") or target.endswith(".yml"):
            config_file = Path(target)
            if not config_file.exists():
                raise ValueError(f"File not found: {config_file}")
                
            with open(config_file, "r") as f:
                raw_data = yaml.safe_load(f)
                
            # Gate 1 & 2: Schema Validation
            schema = MlsysPlanSchema(**raw_data)
            
            if schema.hardware.nodes == 1:
                # Single Node Evaluation
                solver = SingleNodeModel()
                profile = solver.solve(
                    model=schema.model_obj,
                    hardware=schema.hardware_obj,
                    batch_size=schema.workload.batch_size,
                    precision=schema.hardware.precision,
                    efficiency=schema.hardware.efficiency,
                    raise_errors=True
                )
                
                perf_level = EvaluationLevel(
                    level_name="Single Node Performance",
                    status="PASS",
                    summary=f"{profile.bottleneck} Bound",
                    metrics={
                        "latency": f"{profile.latency:~.2f}",
                        "throughput": f"{profile.throughput:~.1f}",
                        "mfu": round(profile.mfu, 2)
                    }
                )
                feasibility_summary = f"{profile.memory_footprint.to('GB'):~.1f} / {schema.hardware_obj.memory.capacity.to('GB'):~.1f} used"
                
            else:
                # Distributed Fleet Evaluation
                dist_model = DistributedModel()
                dist_res = dist_model.solve(
                    model=schema.model_obj,
                    fleet=schema.fleet_obj,
                    batch_size=schema.workload.batch_size,
                    precision=schema.hardware.precision,
                    efficiency=schema.hardware.efficiency,
                    tp_size=schema.hardware.nodes, # Simplify for prototype: TP across all nodes
                    pp_size=1
                )
                
                perf_level = EvaluationLevel(
                    level_name="Fleet Performance",
                    status="PASS",
                    summary=f"Scaling Efficiency: {dist_res.scaling_efficiency:.1%}",
                    metrics={
                        "step_latency": f"{dist_res.step_latency_total:~.2f}",
                        "comm_overhead": f"{dist_res.communication_latency:~.2f}",
                        "fleet_throughput": f"{dist_res.effective_throughput:~.1f}"
                    }
                )
                feasibility_summary = "Distributed Model Check Passed"
                
            # Build Macro Level if OpsConfig exists
            macro_level = EvaluationLevel(level_name="Macro", status="SKIPPED", summary="No Ops config provided")
            if schema.ops and schema.fleet_obj:
                econ_model = EconomicsModel()
                econ_res = econ_model.solve(
                    fleet=schema.fleet_obj,
                    duration_days=schema.ops.duration_days
                )
                macro_level = EvaluationLevel(
                    level_name="Economics & Sustainability",
                    status="PASS",
                    summary=f"TCO: ${econ_res.tco_usd:,.0f}",
                    metrics={
                        "carbon_footprint": f"{econ_res.carbon_footprint_kg/1000.0:,.1f} tons",
                        "energy_cost": f"${econ_res.opex_energy_usd:,.0f}",
                        "capex": f"${econ_res.capex_usd:,.0f}"
                    }
                )

            eval_obj = SystemEvaluation(
                scenario_name=schema.name,
                feasibility=EvaluationLevel(
                    level_name="Feasibility",
                    status="PASS",
                    summary=feasibility_summary,
                ),
                performance=perf_level,
                macro=macro_level
            )
            
            # Check Assertions (Gate 5)
            assertion_failures = []
            if schema.constraints and schema.constraints.asserts:
                all_metrics = {}
                for k, v in eval_obj.feasibility.metrics.items():
                    all_metrics[f"feasibility.{k}"] = v
                for k, v in eval_obj.performance.metrics.items():
                    all_metrics[f"performance.{k}"] = v
                for k, v in eval_obj.macro.metrics.items():
                    all_metrics[f"macro.{k}"] = v
                    
                def _parse_val(val):
                    if isinstance(val, float) or isinstance(val, int): return val
                    if isinstance(val, str):
                        try:
                            import re
                            return float(re.sub(r'[^\d.]', '', val))
                        except:
                            return 0.0
                    return 0.0
                    
                for assertion in schema.constraints.asserts:
                    metric_val_str = all_metrics.get(assertion.metric)
                    if metric_val_str is not None:
                        metric_val = _parse_val(metric_val_str)
                        if assertion.max is not None and metric_val > assertion.max:
                            assertion_failures.append(f"{assertion.metric} ({metric_val}) exceeds max ({assertion.max})")
                        if assertion.min is not None and metric_val < assertion.min:
                            assertion_failures.append(f"{assertion.metric} ({metric_val}) is below min ({assertion.min})")
            
            if assertion_failures:
                render_scorecard(eval_obj, output_format=output_format)
                msg = "\n".join([f"❌ {f}" for f in assertion_failures])
                print_error("SLA / Constraint Violation", msg, output_format=output_format)
                exit_with_code(ExitCode.SLA_FAIL)
            
            render_scorecard(eval_obj, output_format=output_format)

        # Branch 2: Quick Evaluation (CLI Flags)
        else:
            if not hardware:
                raise ValueError("Hardware argument is required when not providing a YAML file. Example: mlsysim eval Llama3_8B H100")
                
            schema = EvalNodeSchema(
                model_name=target, 
                hardware_name=hardware, 
                batch_size=batch_size, 
                precision=precision, 
                efficiency=efficiency
            )

            if output_format == "text" and efficiency > 0.60:
                print_warning(f"Efficiency {efficiency:.0%} is extremely high for realistic workloads. Ensure this matches empirical traces.")

            solver = SingleNodeModel()
            profile = solver.solve(
                model=schema.model_obj,
                hardware=schema.hardware_obj,
                batch_size=schema.batch_size,
                precision=schema.precision,
                efficiency=schema.efficiency,
                raise_errors=True 
            )

            eval_obj = SystemEvaluation(
                scenario_name=f"{schema.model_name} on {schema.hardware_name}",
                feasibility=EvaluationLevel(
                    level_name="Memory Feasibility",
                    status="PASS" if profile.feasible else "FAIL",
                    summary=f"{profile.memory_footprint.to('GB'):~.1f} / {schema.hardware_obj.memory.capacity.to('GB'):~.1f} used",
                ),
                performance=EvaluationLevel(
                    level_name="Single Node Performance",
                    status="PASS",
                    summary=f"{profile.bottleneck} Bound",
                    metrics={
                        "latency": f"{profile.latency:~.2f}",
                        "throughput": f"{profile.throughput:~.1f}",
                        "mfu": round(profile.mfu, 2)
                    }
                ),
                macro=EvaluationLevel(level_name="Macro", status="SKIPPED", summary="No Ops config provided")
            )
            
            render_scorecard(eval_obj, output_format=output_format)
            
    exit_with_code(ExitCode.SUCCESS)
