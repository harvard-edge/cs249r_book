import argparse
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "run_reference_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_reference_sweep", SCRIPT)
assert SPEC and SPEC.loader
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def test_parse_seeds_rejects_empty_and_duplicate_values():
    assert sweep.parse_seeds("0, 1,2") == [0, 1, 2]
    with pytest.raises(argparse.ArgumentTypeError):
        sweep.parse_seeds("")
    with pytest.raises(argparse.ArgumentTypeError):
        sweep.parse_seeds("0,1,0")


def test_default_evidence_root_is_outside_source_checkout():
    assert not sweep.DEFAULT_OUTPUT_DIR.is_relative_to(sweep.ROOT)


def test_sweep_environment_removes_higher_priority_seed_overrides(monkeypatch):
    monkeypatch.setenv("MLPERF_EDU_SEED", "999")
    monkeypatch.setenv("MLPERF_EDU_SLM_SEED", "998")
    monkeypatch.setenv("MLPERF_EDU_DEVICE", "mps")
    env = sweep.sweep_environment(3, "cpu")
    assert "MLPERF_EDU_SEED" not in env
    assert "MLPERF_EDU_SLM_SEED" not in env
    assert env["MLPERF_EDU_MAX_SEED"] == "3"
    assert env["MLPERF_EDU_DEVICE"] == "cpu"


def _write_fake_lineage_package(path: Path) -> None:
    report = {
        "workload": "nanogpt-train",
        "profile": "max",
        "status": "passed",
        "data_mode": "real",
        "quality": {"quality_required": True, "target_met": True},
    }
    manifest = {
        "workload": "nanogpt-train",
        "leaves": {
            "measurement": {"report_path": "../report/train.json"},
            "weights": {"path": "../weights/model.pt"},
        },
    }
    payloads = {
        "manifest/train.provd.json": json.dumps(manifest).encode(),
        "report/train.json": json.dumps(report).encode(),
        "weights/model.pt": b"checkpoint",
    }
    included = [
        {
            "role": name.split("/", 1)[0],
            "path": name,
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "n_bytes": len(payload),
        }
        for name, payload in payloads.items()
    ]
    index = {
        "schema": "mlperf-edu-package/0.2",
        "workload": "nanogpt-train",
        "manifest": "manifest/train.provd.json",
        "source_manifest": "manifest/train.provd.json",
        "included_files": included,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("package_index.json", json.dumps(index))
        for name, payload in payloads.items():
            zf.writestr(name, payload)


def test_lineage_package_is_safely_staged_and_returns_exact_environment(
    tmp_path, monkeypatch
):
    package_path = tmp_path / "training.mlperf-edu.zip"
    _write_fake_lineage_package(package_path)
    monkeypatch.setattr(
        sweep,
        "_verify_package_checks",
        lambda _path: [("clean_extraction", True, "verified")],
    )
    monkeypatch.setattr(
        sweep,
        "_verify_provenance_checks",
        lambda _path: [("provenance", True, "verified")],
    )

    validation = sweep.validate_nanogpt_lineage_package(package_path)
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    staged = sweep.stage_nanogpt_lineage_package(validation, attempt_dir)

    assert (
        staged["package_sha256"]
        == "sha256:" + hashlib.sha256(package_path.read_bytes()).hexdigest()
    )
    assert staged["paths"]["checkpoint"].read_bytes() == b"checkpoint"
    assert staged["paths"]["report"].is_file()
    assert staged["paths"]["manifest"].is_file()
    assert staged["environment"] == {
        "MLPERF_EDU_NANOGPT_CHECKPOINT": str(staged["paths"]["checkpoint"].resolve()),
        "MLPERF_EDU_NANOGPT_TRAIN_REPORT": str(staged["paths"]["report"].resolve()),
        "MLPERF_EDU_NANOGPT_TRAIN_MANIFEST": str(staged["paths"]["manifest"].resolve()),
    }
    for path in staged["paths"].values():
        assert path.resolve().is_relative_to(attempt_dir.resolve())


def test_lineage_package_rejects_traversal_before_verification(tmp_path, monkeypatch):
    package_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(package_path, "w") as zf:
        zf.writestr("package_index.json", "{}")
        zf.writestr("../escape", b"bad")
    verifier_called = False

    def verify(_path):
        nonlocal verifier_called
        verifier_called = True
        return []

    monkeypatch.setattr(sweep, "_verify_package_checks", verify)
    with pytest.raises(sweep.LineagePackageError, match="unsafe.*member path"):
        sweep.validate_nanogpt_lineage_package(package_path)
    assert verifier_called is False
    assert not (tmp_path / "escape").exists()


def test_performance_acceptance_requires_every_functional_gate():
    rows = [
        {"evidence_valid": True, "quality_target_met": True},
        {"evidence_valid": True, "quality_target_met": True},
    ]
    assert sweep.performance_acceptance(rows, "serving check")["passed"] is True
    rows[1]["quality_target_met"] = False
    assert sweep.performance_acceptance(rows, "serving check")["passed"] is False


def test_build_row_rejects_report_or_manifest_seed_mismatch():
    row = sweep.build_row(
        {
            "requested_seed": 3,
            "report_recorded_seed": 42,
            "manifest_recorded_seed": 3,
            "execution_ok": True,
            "evidence_valid": True,
            "quality_value": 0.8,
        }
    )
    assert row["seed_match"] is False
    assert row["evidence_valid"] is False


def fake_result(seed, value, *, data_mode="real"):
    return {
        "requested_seed": seed,
        "report_recorded_seed": seed,
        "manifest_recorded_seed": seed,
        "execution_ok": True,
        "evidence_valid": True,
        "status": "passed",
        "quality_metric_declared": "top1_accuracy",
        "quality_metric_key": "final_accuracy",
        "quality_value": value,
        "quality_target_met": True,
        "wall_seconds": 1.0,
        "backend": "pytorch-cpu",
        "hardware_backend": "MPS",
        "fingerprint_backends": ["pytorch-cpu"],
        "chip": "test",
        "data_mode": data_mode,
        "manifest_verified": True,
        "grade": {"passed": True},
        "artifacts": [],
        "invalid_reasons": [],
        "reproduce": {"env": {"MLPERF_EDU_MAX_SEED": str(seed)}},
    }


def test_main_writes_create_once_valid_evidence_and_digest(tmp_path, monkeypatch):
    values = {0: 0.86, 1: 0.87, 2: 0.88, 3: 0.89, 4: 0.90}

    def run_one_seed(_bootstrap, **kwargs):
        return fake_result(kwargs["seed"], values[kwargs["seed"]])

    monkeypatch.setattr(sweep, "run_one_seed", run_one_seed)
    monkeypatch.setattr(
        sweep,
        "source_snapshot",
        lambda: {
            "git_sha": "abc",
            "git_dirty": False,
            "tool_path": "tools/run_reference_sweep.py",
        },
    )
    code = sweep.main(
        [
            "--workload",
            "resnet18-train",
            "--profile",
            "max",
            "--seeds",
            "0,1,2,3,4",
            "--output-dir",
            str(tmp_path),
            "--timeout-seconds",
            "1",
            "--evidence-tier",
            "public-candidate",
        ]
    )
    assert code == 0
    summaries = list(tmp_path.glob("*/evidence_summary.json"))
    assert len(summaries) == 1
    summary_path = summaries[0]
    summary = json.loads(summary_path.read_text())
    assert summary["schema"] == "mlperf-edu-reference-evidence/0.3"
    assert summary["status"] == "valid"
    assert summary["primary_metric"] == {
        "name": "top1_accuracy",
        "role": "quality",
    }
    assert summary["aggregate"]["primary_metric"] == summary["aggregate"]["quality"]
    assert summary["functional_gate"] is None
    assert summary["repeatability"] is None
    assert summary["eligible_for_public_baseline"] is True
    assert summary["seed_sensitivity"]["verdict"] == "sensitive"
    assert (
        summary["basis"]["reference_protocol"]["seed_interface"]
        == "MLPERF_EDU_MAX_SEED"
    )
    assert summary["rerun_policy"]["mode"] == "full-sweep-only"
    assert summary["timeout_seconds_per_seed"] == 1
    assert summary["runs"][0]["backend"] == "pytorch-cpu"
    assert summary["runs"][0]["hardware_backend"] == "MPS"
    assert summary["runs"][0]["fingerprint_backends"] == ["pytorch-cpu"]
    sidecar = summary_path.with_suffix(".json.sha256")
    digest, filename = sidecar.read_text().strip().split("  ")
    assert filename == summary_path.name
    assert digest == hashlib.sha256(summary_path.read_bytes()).hexdigest()


def test_main_returns_nonzero_for_identical_or_nonreal_public_evidence(
    tmp_path, monkeypatch
):
    def run_one_seed(_bootstrap, **kwargs):
        return fake_result(kwargs["seed"], 0.80, data_mode="synthetic-deterministic")

    monkeypatch.setattr(sweep, "run_one_seed", run_one_seed)
    monkeypatch.setattr(
        sweep, "source_snapshot", lambda: {"git_sha": "abc", "git_dirty": False}
    )
    code = sweep.main(
        [
            "--workload",
            "resnet18-train",
            "--profile",
            "max",
            "--seeds",
            "0,1,2,3,4",
            "--output-dir",
            str(tmp_path),
            "--evidence-tier",
            "public-candidate",
        ]
    )
    assert code == 1
    summary_path = next(tmp_path.glob("*/evidence_summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "invalid"
    assert summary["eligible_for_public_baseline"] is False
    assert summary["seed_sensitivity"]["verdict"] == "identical"
    assert any("score-bearing" in reason for reason in summary["invalid_reasons"])


def test_main_accepts_performance_metric_with_all_run_functional_gate(
    tmp_path, monkeypatch
):
    def run_one_seed(_bootstrap, **kwargs):
        result = fake_result(
            kwargs["seed"],
            100.0 + kwargs["seed"],
            data_mode="local-prompt",
        )
        result.update(
            {
                "quality_metric_declared": "output_tokens_per_sec",
                "quality_metric_key": "output_tokens_per_sec",
                "functional_metric_declared": "generated_tokens",
                "reference_metric_role": "performance",
            }
        )
        return result

    monkeypatch.setattr(sweep, "run_one_seed", run_one_seed)
    monkeypatch.setattr(
        sweep, "source_snapshot", lambda: {"git_sha": "abc", "git_dirty": False}
    )
    code = sweep.main(
        [
            "--workload",
            "slm-decode",
            "--profile",
            "max",
            "--seeds",
            "0,1,2,3,4",
            "--output-dir",
            str(tmp_path),
            "--evidence-tier",
            "public-candidate",
        ]
    )
    assert code == 0
    summary = json.loads(next(tmp_path.glob("*/evidence_summary.json")).read_text())
    assert summary["eligible_for_public_baseline"] is True
    assert summary["reference_metric_role"] == "performance"
    assert summary["quality_metric"] == "output_tokens_per_sec"
    assert summary["acceptance"]["statistic"] == "all_runs"
    assert summary["acceptance"]["passed"] is True


def test_public_nanogpt_inference_requires_lineage_package(tmp_path, capsys):
    code = sweep.main(
        [
            "--workload",
            "nanogpt-prefill",
            "--profile",
            "max",
            "--output-dir",
            str(tmp_path),
            "--evidence-tier",
            "public-candidate",
        ]
    )
    assert code == 2
    assert "requires --nanogpt-lineage-package" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_main_injects_staged_lineage_and_records_only_relative_paths(
    tmp_path, monkeypatch
):
    captured_environments = []

    def validate(_path):
        return {"validated": True}

    def stage(_validation, attempt_dir):
        stage_root = attempt_dir / "inputs" / "nanogpt-training"
        paths = {
            "checkpoint": stage_root / "weights" / "model.pt",
            "report": stage_root / "report" / "train.json",
            "manifest": stage_root / "manifest" / "train.provd.json",
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test")
        environment = {
            sweep.NANOGPT_LINEAGE_ENV[role]: str(path.resolve())
            for role, path in paths.items()
        }
        return {
            "package_sha256": "sha256:" + "a" * 64,
            "package_schema": "mlperf-edu-package/0.2",
            "source_workload": "nanogpt-train",
            "stage_root": stage_root,
            "paths": paths,
            "environment": environment,
            "verification_check_count": 12,
        }

    def run_one_seed(_bootstrap, **kwargs):
        captured_environments.append(kwargs["environment_overrides"])
        result = fake_result(
            kwargs["seed"],
            200.0 + kwargs["seed"],
            data_mode="checkpoint-backed",
        )
        result.update(
            {
                "quality_metric_declared": "prefill_tokens_per_sec",
                "quality_metric_key": "prefill_tokens_per_sec",
                "functional_metric_declared": "prefill_tokens_per_sec",
                "reference_metric_role": "performance",
            }
        )
        return result

    monkeypatch.setattr(sweep, "validate_nanogpt_lineage_package", validate)
    monkeypatch.setattr(sweep, "stage_nanogpt_lineage_package", stage)
    monkeypatch.setattr(sweep, "run_one_seed", run_one_seed)
    monkeypatch.setattr(
        sweep, "source_snapshot", lambda: {"git_sha": "abc", "git_dirty": False}
    )
    code = sweep.main(
        [
            "--workload",
            "nanogpt-prefill",
            "--profile",
            "max",
            "--seeds",
            "0,1,2,3,4",
            "--output-dir",
            str(tmp_path),
            "--evidence-tier",
            "public-candidate",
            "--nanogpt-lineage-package",
            str(tmp_path / "input.zip"),
        ]
    )
    assert code == 0
    assert len(captured_environments) == 5
    assert all(
        set(environment) == set(sweep.NANOGPT_LINEAGE_ENV.values())
        for environment in captured_environments
    )
    summary = json.loads(next(tmp_path.glob("*/evidence_summary.json")).read_text())
    lineage = summary["nanogpt_training_lineage"]
    assert lineage["required"] is True
    assert lineage["status"] == "staged"
    assert lineage["package_sha256"] == "sha256:" + "a" * 64
    assert lineage["staged_root"] == "inputs/nanogpt-training"
    assert lineage["source_training_report"].startswith("inputs/")
    assert lineage["source_training_manifest"].startswith("inputs/")
    assert lineage["source_training_checkpoint"].startswith("inputs/")
    assert not any(
        Path(lineage[key]).is_absolute()
        for key in (
            "staged_root",
            "source_training_report",
            "source_training_manifest",
            "source_training_checkpoint",
        )
    )
