import typer
from pathlib import Path
import yaml
from typing import Optional
from mlsysim.cli.schemas import EvalNodeSchema, MlsysPlanSchema
from mlsysim.cli.exceptions import ExitCode, exit_with_code, error_shield
from mlsysim.cli.renderers import render_scorecard, print_warning, print_error
from mlsysim.core.evaluation import SystemEvaluator

def evaluate_main(
    ctx: typer.Context,
    target: str = typer.Argument(..., help="Path to `mlsys.yaml` OR Model name (e.g. `Llama3_8B`)"),
    hardware: Optional[str] = typer.Argument(None, help="Hardware name (e.g. `H100`) - Required if target is not a YAML file"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size (for single node evaluation)"),
    precision: str = typer.Option("fp16", "--precision", "-p", help="Numerical precision (`fp32`, `fp16`, `fp8`, `int8`, `int4`)"),
    efficiency: float = typer.Option(0.5, "--efficiency", "-e", help="Target Model FLOPs Utilization (`0.0` to `1.0`)")
):
    """
    **[Tier 1] Evaluate the analytical physics of an ML system.**
    
    This command can evaluate a single node via direct CLI flags, or a massive distributed fleet via a declarative YAML configuration.
    
    **Examples:**
    
    Quick single node check:
    ```bash
    mlsysim eval Llama3_8B H100 --batch-size 32
    ```
    
    Deep cluster simulation:
    ```bash
    mlsysim eval my_cluster.yaml
    ```
    """
    
    # Context should be initialized by main.py
    output_format = ctx.obj.get("output_format", "text") if ctx.obj else "text"

    with error_shield(output_format=output_format):
        # Determine if target is a Cluster YAML or a Workload (Model) YAML/String
        is_cluster_yaml = False
        raw_data = None
        if target.endswith(".yaml") or target.endswith(".yml"):
            config_file = Path(target)
            if not config_file.exists():
                raise ValueError(f"File not found: {config_file}")
            with open(config_file, "r") as f:
                raw_data = yaml.safe_load(f)
            # A full cluster definition must have a hardware block
            if raw_data and "hardware" in raw_data and "workload" in raw_data:
                is_cluster_yaml = True

        # Branch 1: Deep Evaluation (YAML / IaC Cluster Plan)
        if is_cluster_yaml:
            # Gate 1 & 2: Schema Validation
            schema = MlsysPlanSchema(**raw_data)
            
            # Use SystemEvaluator to decouple orchestration from CLI
            eval_obj = SystemEvaluator.evaluate(
                scenario_name=schema.name,
                model_obj=schema.model_obj,
                hardware_obj=schema.hardware_obj,
                batch_size=schema.workload.batch_size,
                precision=schema.hardware.precision,
                efficiency=schema.hardware.efficiency,
                fleet_obj=schema.fleet_obj,
                nodes=schema.hardware.nodes,
                duration_days=schema.ops.duration_days if schema.ops else None
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
                        except (ValueError, TypeError):
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

            # Use SystemEvaluator to decouple orchestration from CLI
            eval_obj = SystemEvaluator.evaluate(
                scenario_name=f"{schema.model_name} on {schema.hardware_name}",
                model_obj=schema.model_obj,
                hardware_obj=schema.hardware_obj,
                batch_size=schema.batch_size,
                precision=schema.precision,
                efficiency=schema.efficiency,
                nodes=1
            )
            
            render_scorecard(eval_obj, output_format=output_format)
            
    exit_with_code(ExitCode.SUCCESS)
