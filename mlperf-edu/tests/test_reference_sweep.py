import argparse
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

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
    monkeypatch.setenv("MLPERF_EDU_RESNET_MAX_EPOCHS", "1")
    monkeypatch.setenv("MLPERF_EDU_MAX_QUALITY_TARGET", "999")
    monkeypatch.setenv("MLPERF_EDU_SLM_MODEL_ID", "unapproved/model")
    monkeypatch.setenv("MLPERF_EDU_DATA_DIR", "/tmp/noncanonical-data")
    env = sweep.sweep_environment(3, "cpu")
    assert "MLPERF_EDU_SEED" not in env
    assert "MLPERF_EDU_SLM_SEED" not in env
    assert env["MLPERF_EDU_MAX_SEED"] == "3"
    assert env["MLPERF_EDU_DEVICE"] == "cpu"
    assert "MLPERF_EDU_RESNET_MAX_EPOCHS" not in env
    assert "MLPERF_EDU_MAX_QUALITY_TARGET" not in env
    assert "MLPERF_EDU_SLM_MODEL_ID" not in env
    assert "MLPERF_EDU_DATA_DIR" not in env

    with pytest.raises(ValueError, match="unsupported reference sweep"):
        sweep.sweep_environment(3, "cpu", {"MLPERF_EDU_RESNET_MAX_EPOCHS": "1"})


def test_outer_execution_policy_stabilizes_all_timed_public_candidates():
    policy = sweep.build_outer_execution_policy(
        public_status="performance-bearing",
        evidence_tier="public-candidate",
        seeds=[0, 1, 2, 3, 4],
        configured_cooldown_seconds=2.5,
    )
    assert policy["applies"] is True
    assert policy["process_execution_count"] == 5
    assert policy["execution_unit"] == "one fresh Python subprocess per requested seed"
    assert [execution["execution_index"] for execution in policy["executions"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert [execution["seed"] for execution in policy["executions"]] == [0, 1, 2, 3, 4]
    assert [
        execution["cooldown_before_seconds"] for execution in policy["executions"]
    ] == [0.0, 2.5, 2.5, 2.5, 2.5]

    score_policy = sweep.build_outer_execution_policy(
        public_status="score-bearing",
        evidence_tier="public-candidate",
        seeds=[0, 1],
        configured_cooldown_seconds=2.5,
    )
    assert score_policy["applies"] is True
    assert [
        execution["cooldown_before_seconds"] for execution in score_policy["executions"]
    ] == [0.0, 2.5]

    for public_status, evidence_tier in (("performance-bearing", "development"),):
        not_applied = sweep.build_outer_execution_policy(
            public_status=public_status,
            evidence_tier=evidence_tier,
            seeds=[0, 1],
            configured_cooldown_seconds=2.5,
        )
        assert not_applied["applies"] is False
        assert [
            execution["cooldown_before_seconds"]
            for execution in not_applied["executions"]
        ] == [0.0, 0.0]


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "300.1"])
def test_main_rejects_invalid_inter_execution_cooldowns(value):
    with pytest.raises(SystemExit) as exc_info:
        sweep.main(
            [
                "--workload",
                "resnet18-train",
                "--inter-execution-cooldown-seconds",
                value,
            ]
        )
    assert exc_info.value.code == 2


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


def fake_result(seed, value, *, data_mode="real", primary_metric_value=None):
    if primary_metric_value is None:
        primary_metric_value = 10.0 + seed
    return {
        "requested_seed": seed,
        "report_recorded_seed": seed,
        "manifest_recorded_seed": seed,
        "execution_ok": True,
        "evidence_valid": True,
        "status": "passed",
        "primary_metric_declared": "train_and_eval_seconds",
        "primary_metric_key": "train_and_eval_seconds",
        "primary_metric_value": primary_metric_value,
        "quality_metric_declared": "top1_accuracy",
        "quality_metric_key": "final_accuracy",
        "quality_value": value,
        "quality_target_met": True,
        "comparison_fingerprint_sha256": "a" * 64,
        "scenario": "training",
        "manifest_scenario": "training",
        "registry_scenario": "training",
        "reference_metric_role": "performance",
        "wall_seconds": 1.0,
        "backend": "pytorch-cpu",
        "hardware_backend": "MPS",
        "fingerprint_backends": ["pytorch-cpu"],
        "chip": "test",
        "data_mode": data_mode,
        "manifest_verified": True,
        "grade": {
            "passed": True,
            "status": "passed",
            "target_met": True,
            "metric": "top1_accuracy",
            "value": value,
            "target": 0.85,
        },
        "artifacts": [],
        "invalid_reasons": [],
        "reproduce": {"env": {"MLPERF_EDU_MAX_SEED": str(seed)}},
    }


def test_run_one_seed_cooldown_precedes_mocked_subprocess_and_never_sleeps_zero(
    tmp_path, monkeypatch
):
    events = []
    subprocess_environments = []
    monkeypatch.setenv("MLPERF_EDU_MAX_QUALITY_TARGET", "999")

    def fake_sleep(seconds):
        events.append(("sleep", seconds))

    def fake_run(command, **kwargs):
        events.append(("subprocess", Path(command[1]).name))
        subprocess_environments.append(kwargs["env"])
        args = json.loads(Path(command[-2]).read_text())
        Path(command[-1]).write_text(
            json.dumps(fake_result(args["seed"], 100.0 + args["seed"]))
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sweep.time, "sleep", fake_sleep)
    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    bootstrap = tmp_path / "child.py"
    bootstrap.write_text("# subprocess is mocked\n")

    common = {
        "workload_id": "slm-decode",
        "variant": None,
        "profile": "max",
        "device": "cpu",
        "attempt_dir": tmp_path,
        "timeout_seconds": 1.0,
        "evidence_tier": "public-candidate",
        "allowed_data_modes": sweep.PERFORMANCE_PUBLIC_DATA_MODES,
    }
    sweep.run_one_seed(bootstrap, seed=0, cooldown_before_seconds=0.0, **common)
    sweep.run_one_seed(bootstrap, seed=1, cooldown_before_seconds=2.5, **common)

    assert events == [
        ("subprocess", "child.py"),
        ("sleep", 2.5),
        ("subprocess", "child.py"),
    ]
    assert all(
        "MLPERF_EDU_MAX_QUALITY_TARGET" not in env for env in subprocess_environments
    )


def test_main_writes_create_once_valid_evidence_and_digest(tmp_path, monkeypatch):
    values = {0: 0.86, 1: 0.87, 2: 0.88, 3: 0.89, 4: 0.90}

    def run_one_seed(_bootstrap, **kwargs):
        seed = kwargs["seed"]
        return fake_result(
            seed,
            values[seed],
            primary_metric_value=10.0 + (0.05 * seed),
        )

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
    assert summary["schema"] == "mlperf-edu-reference-evidence/0.4"
    assert summary["status"] == "valid"
    assert summary["primary_metric"] == {
        "name": "train_and_eval_seconds",
        "role": "performance",
    }
    assert summary["reference_metric_role"] == "performance"
    assert summary["quality_metric"] == "top1_accuracy"
    assert summary["aggregate"]["primary_metric"]["median"] == 10.1
    assert summary["aggregate"]["quality"]["median"] == 0.88
    assert summary["aggregate"]["primary_metric"] != summary["aggregate"]["quality"]
    assert summary["acceptance"]["all_runs_passed"] is True
    assert summary["acceptance"]["passed_runs"] == 5
    assert summary["quality_gate"] == {
        "metric": "top1_accuracy",
        "target": 0.85,
        "direction": "higher",
        "tolerance": 0.0,
        "all_runs_must_pass": True,
    }
    assert summary["functional_gate"] is None
    assert summary["repeatability"] is None
    assert summary["primary_metric_repeatability"]["passed"] is True
    assert summary["eligible_for_public_baseline"] is True
    assert summary["seed_sensitivity"]["verdict"] == "sensitive"
    assert (
        summary["basis"]["reference_protocol"]["seed_interface"]
        == "MLPERF_EDU_MAX_SEED"
    )
    assert summary["rerun_policy"]["mode"] == "full-sweep-only"
    assert summary["timeout_seconds_per_seed"] == 1
    assert summary["inter_execution_stabilization"]["applies"] is True
    assert [
        execution["cooldown_before_seconds"]
        for execution in summary["inter_execution_stabilization"]["executions"]
    ] == [0.0, 5.0, 5.0, 5.0, 5.0]
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


def test_score_evidence_rejects_high_variance_primary_timing(tmp_path, monkeypatch):
    values = {0: 0.86, 1: 0.87, 2: 0.88, 3: 0.89, 4: 0.90}

    def run_one_seed(_bootstrap, **kwargs):
        seed = kwargs["seed"]
        return fake_result(seed, values[seed], primary_metric_value=10.0 + seed)

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
    summary = json.loads(next(tmp_path.glob("*/evidence_summary.json")).read_text())
    assert summary["primary_metric_repeatability"]["passed"] is False
    assert any(
        "primary performance repeatability exceeds" in reason
        for reason in summary["invalid_reasons"]
    )


def test_score_evidence_requires_every_quality_run_and_primary_metric(
    tmp_path, monkeypatch
):
    def run_one_seed(_bootstrap, **kwargs):
        result = fake_result(kwargs["seed"], 0.90)
        if kwargs["seed"] == 2:
            result["quality_value"] = 0.84
            result["quality_target_met"] = False
        if kwargs["seed"] == 3:
            result["primary_metric_value"] = None
        return result

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
    summary = json.loads(next(tmp_path.glob("*/evidence_summary.json")).read_text())
    assert summary["acceptance"]["passed"] is False
    assert summary["acceptance"]["all_runs_passed"] is False
    assert summary["acceptance"]["passed_runs"] == 4
    assert any(
        "primary metric value is not finite and positive" in reason
        for reason in summary["invalid_reasons"]
    )


def test_main_accepts_performance_metric_with_all_run_functional_gate(
    tmp_path, monkeypatch
):
    calls = []

    def run_one_seed(_bootstrap, **kwargs):
        calls.append(kwargs)
        result = fake_result(
            kwargs["seed"],
            100.0 + kwargs["seed"],
            data_mode="local-prompt",
        )
        result.update(
            {
                "primary_metric_declared": "output_tokens_per_sec",
                "primary_metric_key": "output_tokens_per_sec",
                "primary_metric_value": 100.0 + kwargs["seed"],
                "quality_metric_declared": None,
                "quality_metric_key": None,
                "quality_value": None,
                "functional_metric_declared": "generated_tokens",
                "functional_metric_key": "generated_tokens",
                "functional_metric_value": 16.0,
                "reference_metric_role": "performance",
                "grade": {
                    "passed": True,
                    "status": "passed",
                    "target_met": True,
                    "metric": "generated_tokens",
                    "value": 16.0,
                    "target": 8,
                },
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
            "--inter-execution-cooldown-seconds",
            "2.5",
        ]
    )
    assert code == 0
    summary = json.loads(next(tmp_path.glob("*/evidence_summary.json")).read_text())
    assert summary["eligible_for_public_baseline"] is True
    assert summary["reference_metric_role"] == "performance"
    assert summary["primary_metric"] == {
        "name": "output_tokens_per_sec",
        "role": "performance",
    }
    assert summary["quality_metric"] is None
    assert summary["aggregate"]["quality"] is None
    assert summary["functional_gate"]["metric"] == "generated_tokens"
    assert summary["acceptance"]["statistic"] == "all_runs"
    assert summary["acceptance"]["passed"] is True
    assert [call["cooldown_before_seconds"] for call in calls] == [
        0.0,
        2.5,
        2.5,
        2.5,
        2.5,
    ]
    stabilization = summary["inter_execution_stabilization"]
    assert stabilization["applies"] is True
    assert stabilization["configured_cooldown_seconds"] == 2.5
    assert summary["runs"][0]["outer_process_execution"] == {
        "execution_index": 1,
        "seed": 0,
        "fresh_process": True,
        "cooldown_before_seconds": 0.0,
    }
    reproduction = summary["runs"][1]["reproduce"]["reference_sweep"]
    assert reproduction["outer_process_execution"]["execution_index"] == 2
    assert reproduction["inter_execution_cooldown"] == {
        "cli_option": "--inter-execution-cooldown-seconds",
        "configured_seconds": 2.5,
        "applied_before_this_execution_seconds": 2.5,
        "applies": True,
    }


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
                "primary_metric_declared": "prefill_tokens_per_sec",
                "primary_metric_key": "prefill_tokens_per_sec",
                "primary_metric_value": 200.0 + kwargs["seed"],
                "quality_metric_declared": None,
                "quality_metric_key": None,
                "quality_value": None,
                "functional_metric_declared": "prefill_tokens_per_sec",
                "functional_metric_key": "prefill_tokens_per_sec",
                "functional_metric_value": 200.0 + kwargs["seed"],
                "reference_metric_role": "performance",
                "grade": {
                    "passed": True,
                    "status": "passed",
                    "target_met": True,
                    "metric": "prefill_tokens_per_sec",
                    "value": 200.0 + kwargs["seed"],
                    "target": 0,
                },
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
