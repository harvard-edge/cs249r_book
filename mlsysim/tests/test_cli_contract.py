import json
from pathlib import Path

from typer.testing import CliRunner

from mlsysim.cli.main import app


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


def test_zoo_accepts_command_local_output_option():
    result = runner.invoke(app, ["zoo", "hardware", "-o", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "hardware" in payload
    assert payload["hardware"]


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


def test_zoo_requires_category():
    result = runner.invoke(app, ["zoo", "-o", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "Zoo category is required" in payload["reason"]
