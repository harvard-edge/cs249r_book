import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from mlsysim.cli.main import app
from mlsysim.cli.schemas import MlsysPlanSchema


ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_eval_json_sla_failure_is_single_json_object():
    result = runner.invoke(
        app,
        ["--output", "json", "eval", str(ROOT / "examples/yaml/test_assert_plan.yaml")],
    )

    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["status"] == "sla_failed"
    assert payload["violations"]
    assert "m_tco_usd" in payload


def test_optimize_parallelism_json_is_serializable():
    result = runner.invoke(
        app,
        ["--output", "json", "optimize", "parallelism", str(ROOT / "examples/yaml/test_fleet_plan.yaml")],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["best_config"]
    assert isinstance(payload["top_candidates"], list)


def test_eval_accepts_command_local_output_option():
    result = runner.invoke(
        app,
        ["eval", "Llama3_8B", "H100", "-o", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["scenario"] == "Llama3_8B on H100"
    assert payload["f_status"] == "PASS"


def test_eval_rejects_unknown_precision():
    result = runner.invoke(
        app,
        ["eval", "Llama3_8B", "H100", "--precision", "fp6", "-o", "json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "precision 'fp6' is not supported" in payload["reason"]


def test_plan_builds_fleet_from_explicit_topology():
    schema = MlsysPlanSchema.model_validate(
        {
            "version": "1.0",
            "name": "Topology test",
            "workload": {"name": "Llama3_8B", "batch_size": 16},
            "hardware": {
                "name": "H100",
                "accelerators": 16,
                "accelerators_per_node": 4,
                "intra_node_bw": "400 GB/s",
                "fabric_bandwidth": "200 Gbit/s",
            },
        }
    )

    assert schema.fleet_obj.total_accelerators == 16
    assert schema.fleet_obj.count == 4
    assert schema.fleet_obj.node.accelerators_per_node == 4
    assert schema.fleet_obj.node.intra_node_bw.m_as("GB/s") == pytest.approx(400)
    assert schema.fleet_obj.fabric.bandwidth.m_as("Gbit/s") == pytest.approx(200)


def test_plan_builds_fleet_from_node_count_topology():
    schema = MlsysPlanSchema.model_validate(
        {
            "version": "1.0",
            "name": "Node count topology test",
            "workload": {"name": "Llama3_8B", "batch_size": 16},
            "hardware": {
                "name": "H100",
                "node_count": 2,
                "accelerators_per_node": 8,
                "fabric_bandwidth": "400 Gbit/s",
            },
        }
    )

    assert schema.hardware.total_accelerators == 16
    assert schema.fleet_obj.count == 2
    assert schema.fleet_obj.total_accelerators == 16


def test_plan_rejects_legacy_nodes_field():
    with pytest.raises(ValidationError, match="Extra inputs"):
        MlsysPlanSchema.model_validate(
            {
                "version": "1.0",
                "name": "Legacy nodes topology",
                "workload": {"name": "Llama3_8B"},
                "hardware": {
                    "name": "H100",
                    "nodes": 16,
                },
            }
        )


def test_plan_rejects_non_divisible_topology():
    with pytest.raises(ValidationError, match="total accelerators must be divisible"):
        MlsysPlanSchema.model_validate(
            {
                "version": "1.0",
                "name": "Bad topology",
            "workload": {"name": "Llama3_8B"},
            "hardware": {
                "name": "H100",
                "accelerators": 10,
                "accelerators_per_node": 4,
            },
        }
        )


def test_plan_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs|extra_forbidden"):
        MlsysPlanSchema.model_validate(
            {
                "version": "1.0",
                "name": "Bad plan",
                "workload": {"name": "Llama3_8B", "stray_field": True},
                "hardware": {"name": "H100"},
            }
        )


def test_zoo_accepts_command_local_output_option():
    result = runner.invoke(app, ["zoo", "hardware", "-o", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "hardware" in payload
    assert payload["hardware"]


def test_zoo_models_accepts_polymorphic_workloads():
    result = runner.invoke(app, ["zoo", "models", "-o", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "models" in payload
    assert any(row["name"] == "Stable Diffusion v1.5" for row in payload["models"])
    dlrm = next(row for row in payload["models"] if row["name"] == "DLRM")
    assert dlrm["parameters"] is None


def test_audit_json_is_single_json_object():
    result = runner.invoke(app, ["audit", "-o", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["workload"] == "Llama3_8B"
    assert payload["reference_hardware"]
    assert "environment" in payload


def test_audit_rejects_unknown_workload():
    result = runner.invoke(app, ["audit", "--workload", "NoSuchModel", "-o", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "NoSuchModel" in payload["reason"]


def test_invalid_output_format_is_rejected():
    result = runner.invoke(app, ["eval", "Llama3_8B", "H100", "-o", "xml"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "unsupported output format" in result.stderr


def test_serve_rejects_unsupported_html_output():
    result = runner.invoke(app, ["serve", "Llama3_8B", "H100", "-o", "html"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "not supported by this command" in result.stderr


def test_serve_markdown_output_is_markdown_table():
    result = runner.invoke(app, ["serve", "Llama3_8B", "H100", "-o", "markdown"])

    assert result.exit_code == 0
    assert result.stdout.startswith("## Serving Analysis")
    assert "| TTFT (prefill) |" in result.stdout


def test_schema_accepts_command_local_output_option():
    result = runner.invoke(app, ["schema", "-o", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["type"] == "object"
    assert "model_obj" not in payload.get("properties", {})
    assert "hardware_obj" not in payload.get("properties", {})
    assert "fleet_obj" not in payload.get("properties", {})


def test_schema_rejects_markdown_output():
    result = runner.invoke(app, ["schema", "-o", "markdown"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "not supported by this command" in result.stderr


def test_schema_rejects_unknown_type():
    result = runner.invoke(app, ["schema", "--type", "nope", "-o", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "Unknown schema type" in payload["reason"]


def test_zoo_without_category_lists_both():
    # 2026-06-06 UX fix: the CATEGORY argument is genuinely optional now
    # (--help always marked it optional, but the command used to error).
    # Omitting it renders both registries and exits 0.
    result = runner.invoke(app, ["zoo"])

    assert result.exit_code == 0
    assert "Hardware Zoo" in result.stdout
    assert "Models Zoo" in result.stdout
