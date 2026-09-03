from __future__ import annotations

from pathlib import Path

import pytest

from mlperf import edu_cli
from mlperf.manifest import verify_provd
from mlperf.registry import find_project_root, load_registry


FUNCTIONAL_WORKLOADS = (
    "code-generation",
    "function-calling",
    "recommendation",
    "image-generation",
    "reinforcement-learning",
)


@pytest.mark.parametrize("workload_id", FUNCTIONAL_WORKLOADS)
def test_functional_spiral_min_runner_is_reviewable(
    workload_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLPERF_EDU_DEVICE", "cpu")
    workload = load_registry()[workload_id]

    report = edu_cli.run_workload(workload, "min", tmp_path)

    assert report["status"] == "passed"
    assert report["profile"] == "min"
    assert report["data_mode"] == "synthetic-deterministic-functional-probe"
    assert report["quality"]["quality_required"] is False
    assert report["quality"]["target_met"] is None
    readiness = report["functional_readiness"]
    assert readiness["schema"] == "mlperf-edu-functional-readiness/0.1"
    assert readiness["stage"] == "functional"
    assert readiness["probe"]
    assert readiness["end_to_end_execution"] is True
    assert readiness["authoritative_quality_contract_executed"] is False
    assert readiness["repeatability_verified"] is False
    assert readiness["promotion_eligible"] is False
    assert readiness["next_stage"] == "quality-conformance"
    assert report["metrics"]["duration_seconds"] > 0
    assert report["metrics"]["functional_check"]

    report_path = Path(report["artifacts"]["report"])
    manifest_path = Path(report["artifacts"]["provenance"])
    assert report_path.is_file()
    assert manifest_path.is_file()
    verification = verify_provd(manifest_path, repo_root=find_project_root())
    assert verification.all_ok, verification.checks


def test_function_calling_probe_connects_generation_to_ast_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLPERF_EDU_DEVICE", "cpu")
    report = edu_cli.run_workload(load_registry()["function-calling"], "min", tmp_path)

    metrics = report["metrics"]
    assert metrics["generated_tokens"] == metrics["grammar_constraint_steps"]
    assert metrics["ast_fixture_valid"] is True
    assert metrics["functional_check"] == (
        "grammar-constrained-generation-and-ast-evaluator-completed"
    )
