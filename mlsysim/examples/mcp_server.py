# mlsysim/examples/mcp_server.py
"""
MLSys·im Model Context Protocol (MCP) Server
============================================
This script runs a fast, native MCP server that exposes the MLSys·im engine
to any compatible AI Agent (e.g., Claude Desktop, Cursor, or custom agents).

It exposes two primary tools:
1. `get_schemas`: Returns the exact JSON schema required to formulate an MLSysPlan.
2. `evaluate_cluster_yaml`: Takes a YAML string, evaluates it, and returns the strict JSON physics profile.

To use with Claude Desktop, add to your config:
```json
"mcpServers": {
  "mlsysim": {
    "command": "python3",
    "args": ["/path/to/MLSysBook/mlsysim/examples/mcp_server.py"]
  }
}
```
"""

import sys
import json
import yaml
from pathlib import Path
from pydantic import ValidationError

try:
    from mcp.server import FastMCP
except ImportError:
    print("Error: The 'mcp' library is required to run the MCP server.", file=sys.stderr)
    print("Install it with: pip install mcp", file=sys.stderr)
    sys.exit(1)

from mlsysim.cli.schemas import MlsysPlanSchema
from mlsysim.core.evaluation import SystemEvaluator

# Initialize the MCP Server
mcp = FastMCP("MLSys·im")

@mcp.tool()
def get_schemas() -> str:
    """
    Returns the strict JSON schemas required to interact with the MLSys·im engine.
    Always read these schemas before attempting to generate a cluster YAML.
    """
    return json.dumps({
        "MlsysPlanSchema": MlsysPlanSchema.model_json_schema()
    }, indent=2)

@mcp.tool()
def evaluate_cluster_yaml(yaml_config_string: str) -> str:
    """
    Evaluates an MLSys·im cluster configuration against the Iron Law of ML Performance.
    
    Args:
        yaml_config_string: A complete YAML string adhering to MlsysPlanSchema.
        
    Returns:
        A JSON string containing the physics-based evaluation (feasibility, performance, economics).
        If the configuration violates physical laws (e.g., OOM), the JSON will detail the failure.
    """
    try:
        raw_data = yaml.safe_load(yaml_config_string)
        if not raw_data:
            return json.dumps({"error": "Empty YAML provided."})
            
        # 1. Gate: Schema Validation
        try:
            schema = MlsysPlanSchema(**raw_data)
        except ValidationError as e:
            return json.dumps({
                "status": "VALIDATION_FAILED",
                "message": "The provided YAML did not match the required schema.",
                "errors": e.errors()
            })
            
        # 2. Gate: Physics Evaluation
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
        
        # 3. Output standard JSON payload
        return json.dumps(eval_obj.to_dict(), indent=2)
        
    except Exception as e:
        import traceback
        return json.dumps({
            "status": "CRITICAL_ERROR",
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        })

if __name__ == "__main__":
    mcp.run()
