from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from tools.build_wheel import verify_wheel

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

    report_path = output_dir / "micro-lstm-train_min_report.json"
    manifest_path = output_dir / "micro-lstm-train_min.provd.json"
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
    )
    for relative_path in removed:
        assert not (PROJECT_ROOT / relative_path).exists(), relative_path


def test_wheel_guard_rejects_a_retired_module(tmp_path: Path) -> None:
    wheel_path = tmp_path / "stale.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("mlperf_edu/workloads.yaml", "suites: {}\n")
        archive.writestr("mlperf_edu/slm_quality_prompts.json", "{}\n")
        archive.writestr("mlperf/core.py", "retired = True\n")

    with pytest.raises(RuntimeError, match="retired module"):
        verify_wheel(wheel_path)


def test_wheel_guard_rejects_corrupt_reference_evidence(tmp_path: Path) -> None:
    wheel_path = tmp_path / "corrupt-evidence.whl"
    index_path = PROJECT_ROOT / "reference_results" / "index.json"
    index = json.loads(index_path.read_text())
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr(
            "mlperf_edu/workloads.yaml",
            (PROJECT_ROOT / "src/mlperf_edu/workloads.yaml").read_bytes(),
        )
        archive.writestr(
            "mlperf_edu/slm_quality_prompts.json",
            (PROJECT_ROOT / "src/mlperf_edu/slm_quality_prompts.json").read_bytes(),
        )
        archive.writestr(
            "mlperf_edu/reference_results/index.json", index_path.read_bytes()
        )
        archive.writestr(
            "mlperf_edu/reference_results/source_lock.json",
            (PROJECT_ROOT / "reference_results/source_lock.json").read_bytes(),
        )
        for entry in index["summaries"]:
            archive.writestr(f"mlperf_edu/{entry['path']}", b"wrong")

    with pytest.raises(RuntimeError, match="digest mismatch"):
        verify_wheel(wheel_path)
