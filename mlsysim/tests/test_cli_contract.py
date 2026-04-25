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
