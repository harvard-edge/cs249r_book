import typer
import json
from typing import Optional
from mlsysim.cli.context import OUTPUT_FORMAT_HELP, resolve_output_format
from mlsysim.cli.schemas import MlsysPlanSchema
from mlsysim.hardware.types import HardwareNode
from mlsysim.models.types import Workload
from mlsysim.cli.exceptions import ExitCode, error_shield, exit_with_code

def schema_main(
    ctx: typer.Context,
    type: str = typer.Option("plan", "--type", "-t", help="Which schema to export ('plan', 'hardware', 'workload')"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help=f"{OUTPUT_FORMAT_HELP}; schema always emits JSON"),
):
    """Export the JSON Schema for mlsysim configuration files (for IDE autocompletion)."""
    output_format = resolve_output_format(ctx, output, supported={"text", "json"})

    with error_shield(output_format=output_format):
        schema_type = type.lower()
        if schema_type == "hardware":
            schema_json = HardwareNode.model_json_schema()
        elif schema_type == "workload":
            schema_json = Workload.model_json_schema()
        elif schema_type == "plan":
            schema_json = MlsysPlanSchema.model_json_schema()
        else:
            raise ValueError(
                f"Unknown schema type: '{type}'. Valid options are 'plan', 'hardware', or 'workload'."
            )

        print(json.dumps(schema_json, indent=2, allow_nan=False))
    exit_with_code(ExitCode.SUCCESS)
