import typer
import json
from mlsysim.cli.schemas import MlsysPlanSchema
from mlsysim.cli.exceptions import ExitCode, exit_with_code

schema_app = typer.Typer(help="Export the JSON Schema for the mlsys.yaml configuration file (for AI agents & IDEs).", no_args_is_help=True)

@schema_app.callback(invoke_without_command=True)
def schema_main():
    """Export the JSON Schema for the mlsys.yaml configuration file (for AI agents & IDEs)."""
    schema_json = MlsysPlanSchema.model_json_schema()
    print(json.dumps(schema_json, indent=2))
    exit_with_code(ExitCode.SUCCESS)
