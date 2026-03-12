import typer
from pathlib import Path
import yaml
from mlsysim.cli.schemas import MlsysPlanSchema
from mlsysim.cli.exceptions import ExitCode, exit_with_code, error_shield
from mlsysim.cli.renderers import render_optimization
from mlsysim.core.solver import ParallelismOptimizer, BatchingOptimizer, PlacementOptimizer

optimize_app = typer.Typer(help="[Tier 3] Search the design space for optimal configurations.", no_args_is_help=True)

@optimize_app.command("parallelism")
def optimize_parallelism(
    ctx: typer.Context,
    config_file: Path = typer.Argument(..., help="Path to the mlsys.yaml configuration file")
):
    """Find the optimal (TP, PP, DP) split to maximize MFU."""
    output_format = ctx.obj.get("output_format", "text") if ctx.obj else "text"
    
    with error_shield(output_format=output_format):
        if not config_file.exists():
            raise ValueError(f"File not found: {config_file}")
            
        with open(config_file, "r") as f:
            raw_data = yaml.safe_load(f)
            
        schema = MlsysPlanSchema(**raw_data)
        
        if not schema.fleet_obj:
            raise ValueError("Parallelism optimization requires a fleet with nodes > 1.")
            
        optimizer = ParallelismOptimizer()
        result = optimizer.solve(
            model=schema.model_obj,
            fleet=schema.fleet_obj,
            batch_size=schema.workload.batch_size,
            precision=schema.hardware.precision,
            efficiency=schema.hardware.efficiency
        )
        
        is_json = output_format == "json"
        render_optimization("3D Parallelism", result, is_json)
        
    exit_with_code(ExitCode.SUCCESS)

@optimize_app.command("batching")
def optimize_batching(
    ctx: typer.Context,
    config_file: Path = typer.Argument(..., help="Path to the mlsys.yaml configuration file"),
    sla_ms: float = typer.Option(..., "--sla-ms", help="P99 Latency SLA in milliseconds"),
    qps: float = typer.Option(..., "--qps", help="Arrival rate in Queries Per Second")
):
    """Find the maximum safe batch size for a given latency SLA."""
    output_format = ctx.obj.get("output_format", "text") if ctx.obj else "text"
    
    with error_shield(output_format=output_format):
        if not config_file.exists():
            raise ValueError(f"File not found: {config_file}")
            
        with open(config_file, "r") as f:
            raw_data = yaml.safe_load(f)
            
        schema = MlsysPlanSchema(**raw_data)
        
        optimizer = BatchingOptimizer()
        result = optimizer.solve(
            model=schema.model_obj,
            hardware=schema.hardware_obj,
            seq_len=schema.workload.seq_len,
            sla_latency_ms=sla_ms,
            arrival_rate_qps=qps,
            num_replicas=schema.hardware.nodes,
            precision=schema.hardware.precision,
            efficiency=schema.hardware.efficiency
        )
        
        is_json = output_format == "json"
        render_optimization("Batching vs. SLA", result, is_json)
        
    exit_with_code(ExitCode.SUCCESS)

@optimize_app.command("placement")
def optimize_placement(
    ctx: typer.Context,
    config_file: Path = typer.Argument(..., help="Path to the mlsys.yaml configuration file"),
    carbon_tax: float = typer.Option(100.0, "--carbon-tax", help="Carbon tax penalty in $/ton")
):
    """Find the optimal datacenter region to minimize TCO and carbon footprint."""
    output_format = ctx.obj.get("output_format", "text") if ctx.obj else "text"
    
    with error_shield(output_format=output_format):
        if not config_file.exists():
            raise ValueError(f"File not found: {config_file}")
            
        with open(config_file, "r") as f:
            raw_data = yaml.safe_load(f)
            
        schema = MlsysPlanSchema(**raw_data)
        
        if not schema.fleet_obj:
            raise ValueError("Placement optimization requires a fleet with nodes > 1.")
            
        duration = schema.ops.duration_days if schema.ops else 30.0
            
        optimizer = PlacementOptimizer()
        result = optimizer.solve(
            fleet=schema.fleet_obj,
            duration_days=duration,
            carbon_tax_per_ton=carbon_tax,
            mfu=schema.hardware.efficiency
        )
        
        is_json = output_format == "json"
        render_optimization(f"Global Placement (Tax: ${carbon_tax}/ton)", result, is_json)
        
    exit_with_code(ExitCode.SUCCESS)
