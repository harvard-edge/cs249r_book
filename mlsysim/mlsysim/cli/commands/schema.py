import typer
import json
from mlsysim.cli.schemas import MlsysPlanSchema
from mlsysim.hardware.types import HardwareNode
from mlsysim.models.types import Workload
from mlsysim.cli.exceptions import ExitCode, exit_with_code

def schema_main(
    type: str = typer.Option("plan", "--type", "-t", help="Which schema to export ('plan', 'hardware', 'workload')")
):
    """Export the JSON Schema for mlsysim configuration files (for IDE autocompletion)."""
    if type == "hardware":
        schema_json = HardwareNode.model_json_schema()
    elif type == "workload":
        schema_json = Workload.model_json_schema()
    else:
        schema_json = MlsysPlanSchema.model_json_schema()
        
    print(json.dumps(schema_json, indent=2))
    exit_with_code(ExitCode.SUCCESS)
