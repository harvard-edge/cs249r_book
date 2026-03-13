import typer
import json
from mlsysim.cli.schemas import MlsysPlanSchema
from mlsysim.cli.exceptions import ExitCode, exit_with_code

def schema_main():
    """Export the JSON Schema for the mlsys.yaml configuration file (for AI agents & IDEs)."""
    schema_json = MlsysPlanSchema.model_json_schema()
    print(json.dumps(schema_json, indent=2))
    exit_with_code(ExitCode.SUCCESS)
