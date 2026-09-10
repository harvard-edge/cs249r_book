from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from tools.build_wheel import verify_wheel
from tools.check_reference_claims import count_claim_pattern

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(PROJECT_ROOT / "src")
    if environment.get("PYTHONPATH"):
        source_root = f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
    environment["PYTHONPATH"] = source_root
    return environment


@pytest.mark.parametrize(
    ("script", "marker"),
    (
        ("lab1_optimization.py", "LAB 1 SMOKE PASS"),
        ("lab2_inference_sut.py", "LAB 2 SMOKE PASS"),
        ("lab3_arch_comparison.py", "LAB 3 SMOKE PASS"),
    ),
)
def test_lab_smoke_entrypoint(script: str, marker: str, tmp_path: Path) -> None:
    output_path = tmp_path / f"{Path(script).stem}.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "examples" / script),
            "--smoke",
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        env=subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert marker in completed.stdout
    result = json.loads(output_path.read_text())
    assert result["status"] == "passed"
    assert result["result_scope"] == "functional-smoke"
    assert result["canonical_result"] is False
    assert result["functional_check"]["passed"] is True


def test_tutorial_smoke_runs_and_verifies_provenance(tmp_path: Path) -> None:
    output_dir = tmp_path / "tutorial-01"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tutorials" / "smoke_first_benchmark.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "TUTORIAL 01 SMOKE PASS" in completed.stdout

    report_path = output_dir / "time-series-forecasting_min_report.json"
    manifest_path = output_dir / "time-series-forecasting_min.provd.json"
    assert report_path.is_file()
    assert report_path.with_suffix(".html").is_file()
    assert report_path.with_suffix(".csv").is_file()
    assert manifest_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["status"] == "passed"
    assert report["metrics"]
    assert report["run_fingerprint"]


def test_lab1_enforces_an_explicit_classroom_target(tmp_path: Path) -> None:
    output_path = tmp_path / "lab1-target.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "examples" / "lab1_optimization.py"),
            "--smoke",
            "--target-accuracy",
            "1.0",
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        env=subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 2
    result = json.loads(output_path.read_text())
    assert result["status"] == "quality-failed"
    assert result["functional_check"]["passed"] is True
    assert result["classroom_target"]["passed"] is False


def test_lab_guide_does_not_advertise_removed_interfaces() -> None:
    guide = (PROJECT_ROOT / "examples" / "GETTING_STARTED.md").read_text()
    tutorial_guide = (PROJECT_ROOT / "tutorials" / "README.md").read_text()
    combined = f"{guide}\n{tutorial_guide}"
    assert "mlperf cloud" not in combined
    assert "mlperf config" not in combined
    assert "harvard-edge/mlperf-edu" not in combined
    assert "(planned)" not in tutorial_guide


def test_obsolete_and_fabricated_examples_are_removed() -> None:
    removed = (
        "configs/lab2_dense.yaml",
        "configs/nanogpt_train.yaml",
        "configs/resnet_lab1_broken.yaml",
        "practitioner_workflows.py",
        "closed_resnet_optimized.py",
        "open_resnet_quantized.py",
    )
    for relative_path in removed:
        assert not (PROJECT_ROOT / "examples" / relative_path).exists()


def test_obsolete_parallel_product_surfaces_are_removed() -> None:
    removed = (
        "TODAY_IMPLEMENTATION_PLAN.md",
        "src/mlperf/cli.py",
        "src/mlperf/core.py",
        "src/mlperf/datasets.py",
        "src/mlperf/grader.py",
        "src/mlperf/hardware.py",
        "src/mlperf/plotting.py",
        "src/mlperf/provenance.py",
        "src/mlperf/report.py",
        "src/mlperf/reference/cloud/nanogpt_core.py",
        "src/mlperf/reference/cloud/nanogpt_infer.py",
        "src/mlperf/reference/mobile/mobilenet_infer.py",
        "src/mlperf_edu/core.py",
        "tools/measure_all_workloads.py",
        "tools/migrate_taxonomy_iter4.py",
        "tools/sync_yaml_from_sidecars.py",
        "scripts/compliance_checker.py",
        "scripts/generate_all_curves.py",
        "scripts/generate_all_curves_v2.py",
        "scripts/orchestration/auto_trainer.py",
        "scripts/orchestration/llm_researcher.py",
        "scripts/autograder/grade_all.py",
        "scripts/generate_shards.py",
        "scripts/orchestration/data_fetcher.py",
        "scripts/setup_micro_datasets.sh",
        "scripts/smoke_data_quality.py",
        "labs/data_quality/inject.py",
        "scripts/smoke_distributed.py",
        "scripts/smoke_dlrm_dram.py",
        "scripts/smoke_iter6_serving.py",
        "scripts/smoke_iter8_compression.py",
        "scripts/smoke_lora.py",
        "scripts/smoke_nanogpt_phases.py",
        "scripts/smoke_roofline_provenance.py",
        "scripts/verify_training.py",
        "examples/student_optimizations/README.md",
        "examples/student_optimizations/stub_custom_numerics_infer.py",
        "examples/student_optimizations/stub_custom_optimizer_train.py",
        "src/mlperf/reference/agent_datasets.py",
        "src/mlperf/reference/cloud/micro_bert.py",
        "src/mlperf/reference/cloud/micro_diffusion.py",
        "src/mlperf/reference/cloud/micro_dlrm.py",
        "src/mlperf/reference/cloud/micro_gnn.py",
        "src/mlperf/reference/cloud/micro_lstm.py",
        "src/mlperf/reference/cloud/micro_rl.py",
        "src/mlperf/reference/cloud/nano_codegen_agent.py",
        "src/mlperf/reference/cloud/nano_rag_agent.py",
        "src/mlperf/reference/cloud/nano_react_agent.py",
        "src/mlperf/reference/cloud/nano_toolcall_agent.py",
        "src/mlperf/reference/cloud/nanogpt_decode_spec.py",
        "src/mlperf/reference/distributed/ddp_runner.py",
        "src/mlperf/reference/mobile/mobilenet_compress.py",
        "src/mlperf/reference/tiny/anomaly_detection_ae.py",
        "src/mlperf/reference/tiny/dscnn_kws.py",
        "src/mlperf/reference/tiny/wake_vision_vww.py",
    )
    for relative_path in removed:
        assert not (PROJECT_ROOT / relative_path).exists(), relative_path


def test_mlperf_edu_workflows_derive_current_case_and_workload_closure() -> None:
    workflow_root = PROJECT_ROOT.parent / ".github" / "workflows"
    dev = (workflow_root / "mlperf-edu-validate-dev.yml").read_text()
    release = (workflow_root / "mlperf-edu-release-validation.yml").read_text()
    combined = f"{dev}\n{release}"

    for stale_claim in (
        "ten-case",
        "ten-path",
        'summary_count"] == 10',
        'len(data["workloads"]) == 7',
    ):
        assert stale_claim not in combined
    assert "expected_cases" in dev
    assert 'grade["passed"] == len(expected)' in dev
    assert 'int(grade.get("passed", -1)) != len(expected)' in release
    assert "pro-research" not in release  # selected by the product validation plan
    assert '"pro": research_count' in release
    assert "benchmarks/tiny/anomaly-detection.html" in dev
    assert "benchmarks/tiny/visual-wake-words.html" in dev


def test_publication_gates_derive_portfolio_closure_from_registry() -> None:
    paper_generator = (
        PROJECT_ROOT / "paper" / "generate_registry_snapshot.py"
    ).read_text()
    claim_checker = (PROJECT_ROOT / "tools" / "check_reference_claims.py").read_text()

    for stale_constant in ("EXPECTED_WORKLOADS", "EXPECTED_CASES", "SOURCE_SHA"):
        assert stale_constant not in paper_generator
    assert "evidence.expected_cases()" in paper_generator
    assert "verify_current=False" in paper_generator
    assert "promotion_workloads" in paper_generator
    assert '{record["entry"]["workload"] for record in records}' in paper_generator
    assert "count_claim_pattern(len(registry_workload_ids)" in claim_checker
    assert 'len(evidence_workload_ids), "workload"' in claim_checker
    assert "count_claim_pattern(len(records)" in claim_checker


@pytest.mark.parametrize(
    ("count", "noun", "claim"),
    (
        (14, "workload", "fourteen workloads"),
        (14, "workload", "14 workloads"),
        (12, "evidence case", "twelve evidence cases"),
        (12, "evidence case", "12 evidence cases"),
    ),
)
def test_publication_count_claims_accept_words_and_digits(
    count: int, noun: str, claim: str
) -> None:
    assert count_claim_pattern(count, noun).search(claim)
    assert count_claim_pattern(count + 1, noun).search(claim) is None


def test_wheel_guard_rejects_a_retired_module(tmp_path: Path) -> None:
    wheel_path = tmp_path / "stale.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("mlperf_edu/workloads.yaml", "suites: {}\n")
        archive.writestr("mlperf/core.py", "retired = True\n")

    with pytest.raises(RuntimeError, match="retired module"):
        verify_wheel(wheel_path)


def _write_minimal_indexed_wheel(
    path: Path, *, stale_member: str | None = None
) -> None:
    mirror = PROJECT_ROOT / "src" / "mlperf_edu"
    index = json.loads((mirror / "provisional_results" / "index.json").read_text())
    members = {
        "mlperf_edu/workloads.yaml": mirror / "workloads.yaml",
        "mlperf_edu/datasets.yaml": mirror / "datasets.yaml",
        "mlperf_edu/provisional_results/index.json": (
            mirror / "provisional_results" / "index.json"
        ),
        "mlperf_edu/provisional_results/source_lock.json": (
            mirror / "provisional_results" / "source_lock.json"
        ),
    }
    for entry in index["cases"]:
        members[f"mlperf_edu/{entry['path']}"] = mirror / entry["path"]
    with zipfile.ZipFile(path, "w") as archive:
        for member, source in members.items():
            archive.writestr(member, source.read_bytes())
        if stale_member:
            archive.writestr(stale_member, "{}\n")


def test_wheel_guard_rejects_unindexed_result_data(tmp_path: Path) -> None:
    wheel_path = tmp_path / "stale-result.whl"
    _write_minimal_indexed_wheel(
        wheel_path,
        stale_member="mlperf_edu/reference_results/unindexed.json",
    )

    with pytest.raises(RuntimeError, match="stale unindexed reference result"):
        verify_wheel(wheel_path)
