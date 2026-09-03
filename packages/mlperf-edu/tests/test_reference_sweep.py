import argparse
import ast
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


def stable_power_record():
    snapshot = {
        "schema": sweep.HOST_POWER_STATE_SCHEMA,
        "platform": "Darwin",
        "captured_at": "2026-07-14T12:00:00+00:00",
        "provider": "macos-pmset-sysctl",
        "supported": True,
        "source": "external",
        "source_raw": "AC Power",
        "battery_percent": 100,
        "battery_status": "charged",
        "power_mode": 0,
        "low_power_mode": False,
        "last_sleep_epoch": 100,
        "last_wake_epoch": 101,
        "suspend_clock_offset_seconds": None,
        "query_errors": [],
    }
    return {
        "policy": dict(sweep.POWER_STABILITY_POLICY),
        "promotion_conditions_required": True,
        "before": dict(snapshot),
        "after": dict(snapshot),
        "stable": True,
        "invalid_reasons": [],
    }


def _bootstrap_artifact_index():
    tree = ast.parse(sweep._CHILD_BOOTSTRAP)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.FunctionDef)
        and node.name in {"sha256_file", "artifact_index"}
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<bootstrap>", "exec"),
        namespace,
    )
    return namespace["artifact_index"]


def test_child_artifact_index_retains_external_runner_artifacts(tmp_path):
    artifact_index = _bootstrap_artifact_index()
    run_dir = tmp_path / "attempt" / "run_001"
    cache_dir = tmp_path / "cache"
    run_dir.mkdir(parents=True)
    cache_dir.mkdir()
    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run.provd.json"
    report_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")
    external = cache_dir / "model weights.bin"
    external.write_bytes(b"fixed upstream model bytes")
    report = {"artifacts": {"source weights": str(external)}}

    claims = artifact_index(report, report_path, manifest_path, {})
    by_role = {claim["role"]: claim for claim in claims}
    retained = Path(by_role["source weights"]["path"])

    assert retained.parent == run_dir.parent / "retained_artifacts"
    assert retained.read_bytes() == external.read_bytes()
    assert by_role["source weights"]["sha256"] == (
        "sha256:" + hashlib.sha256(external.read_bytes()).hexdigest()
    )
    assert by_role["source weights"]["n_bytes"] == external.stat().st_size
    assert external.is_file()

    second_run = run_dir.parent / "run_002"
    second_run.mkdir()
    second_report = second_run / "report.json"
    second_manifest = second_run / "run.provd.json"
    second_report.write_text("{}\n", encoding="utf-8")
    second_manifest.write_text("{}\n", encoding="utf-8")
    second_claims = artifact_index(report, second_report, second_manifest, {})
    second_by_role = {claim["role"]: claim for claim in second_claims}

    assert second_by_role["source weights"]["path"] == str(retained)
    assert len(list(retained.parent.iterdir())) == 1


def test_parse_seeds_rejects_empty_and_duplicate_values():
    assert sweep.parse_seeds("0, 1,2") == [0, 1, 2]
    with pytest.raises(argparse.ArgumentTypeError):
        sweep.parse_seeds("")
    with pytest.raises(argparse.ArgumentTypeError):
        sweep.parse_seeds("0,1,0")


def test_parse_run_count_and_canonical_seed():
    assert sweep.parse_run_count("5") == 5
    with pytest.raises(argparse.ArgumentTypeError):
        sweep.parse_run_count("0")
    assert sweep.parse_preconditioning_run_count("0") == 0
    assert sweep.parse_preconditioning_run_count("2") == 2
    with pytest.raises(argparse.ArgumentTypeError):
        sweep.parse_preconditioning_run_count("-1")
    from mlperf.registry import load_registry

    registry = load_registry()
    assert sweep.canonical_seed(registry["causal-language-modeling"]) == 1337
    assert sweep.canonical_seed(registry["time-series-forecasting"]) == 2021
    assert sweep.canonical_seed(registry["image-classification"]) == 42


def test_execution_result_roles_are_case_specific():
    from mlperf.registry import load_registry

    registry = load_registry()
    assert (
        sweep.execution_result_role(
            registry["causal-language-modeling"], mode="training", phase=None
        )
        == "score-bearing"
    )
    for phase in ("full", "prefill", "decode"):
        assert (
            sweep.execution_result_role(
                registry["causal-language-modeling"],
                mode="inference",
                phase=phase,
            )
            == "performance-bearing"
        )
    assert (
        sweep.execution_result_role(
            registry["image-classification"], mode="inference", phase=None
        )
        == "score-bearing"
    )


def test_default_evidence_root_is_outside_source_checkout():
    assert not sweep.DEFAULT_OUTPUT_DIR.is_relative_to(sweep.ROOT)


def test_pmset_power_state_parsing_and_promotion_gate():
    battery = """Now drawing from 'AC Power'\n -InternalBattery-0\t87%; charging; 0:20 remaining"""
    settings = """Battery Power:\n powermode 1\nAC Power:\n powermode 0\n"""
    parsed = sweep._parse_pmset_battery(battery)
    assert parsed == {
        "source": "external",
        "source_raw": "AC Power",
        "battery_percent": 87,
        "battery_status": "charging",
    }
    assert sweep._parse_pmset_power_mode(settings, "AC Power") == 0
    assert sweep._parse_pmset_power_mode(settings, "Battery Power") == 1

    record = stable_power_record()
    assert (
        sweep.assess_power_stability(
            record["before"],
            record["after"],
            require_promotion_conditions=True,
        )
        == []
    )
    changed = dict(record["after"])
    changed["source"] = "battery"
    reasons = sweep.assess_power_stability(
        record["before"], changed, require_promotion_conditions=True
    )
    assert "host power source changed during execution" in reasons
    assert "promotion evidence requires external power throughout execution" in reasons


def test_promotion_power_gate_rejects_low_power_and_sleep():
    record = stable_power_record()
    changed = dict(record["after"])
    changed["power_mode"] = 1
    changed["low_power_mode"] = True
    changed["last_sleep_epoch"] = 102
    reasons = sweep.assess_power_stability(
        record["before"], changed, require_promotion_conditions=True
    )
    assert "host power mode changed during execution" in reasons
    assert "host entered sleep during execution" in reasons
    assert "promotion evidence requires Low Power Mode to remain disabled" in reasons


def test_child_bootstrap_records_devices_before_fingerprinting():
    bootstrap = sweep._CHILD_BOOTSTRAP
    import_position = bootstrap.index("            annotate_execution_device,")
    run_position = bootstrap.index("        report = run_workload(")
    annotation_position = bootstrap.index("        annotate_execution_device(report)")
    fingerprint_position = bootstrap.index("        attach_run_fingerprints(report)")

    assert import_position < run_position
    assert run_position < annotation_position < fingerprint_position
    assert '"device_requested": report_requested_device' in bootstrap
    assert '"device_executed": report_executed_device' in bootstrap


def test_sweep_environment_removes_higher_priority_seed_overrides(monkeypatch):
    monkeypatch.setenv("MLPERF_EDU_SEED", "999")
    monkeypatch.setenv("MLPERF_EDU_DEVICE", "mps")
    monkeypatch.setenv("MLPERF_EDU_KEYWORD_SPOTTING_MAX_REPETITIONS", "1")
    monkeypatch.setenv("MLPERF_EDU_KEYWORD_SPOTTING_MAX_WARMUP_REPETITIONS", "1")
    monkeypatch.setenv("MLPERF_EDU_MAX_QUALITY_TARGET", "999")
    monkeypatch.setenv("MLPERF_EDU_UNAPPROVED_OVERRIDE", "unapproved")
    monkeypatch.setenv("MLPERF_EDU_DATA_DIR", "/tmp/noncanonical-data")
    env = sweep.sweep_environment(3, "cpu")
    assert "MLPERF_EDU_SEED" not in env
    assert env["MLPERF_EDU_MAX_SEED"] == "3"
    assert env["MLPERF_EDU_DEVICE"] == "cpu"
    assert "MLPERF_EDU_KEYWORD_SPOTTING_MAX_REPETITIONS" not in env
    assert "MLPERF_EDU_KEYWORD_SPOTTING_MAX_WARMUP_REPETITIONS" not in env
    assert "MLPERF_EDU_MAX_QUALITY_TARGET" not in env
    assert "MLPERF_EDU_UNAPPROVED_OVERRIDE" not in env
    assert "MLPERF_EDU_DATA_DIR" not in env

    with pytest.raises(ValueError, match="unsupported reference sweep"):
        sweep.sweep_environment(3, "cpu", {"MLPERF_EDU_UNAPPROVED_OVERRIDE": "1"})


def test_outer_execution_policy_stabilizes_all_timed_public_candidates():
    policy = sweep.build_outer_execution_policy(
        public_status="performance-bearing",
        evidence_tier="public-candidate",
        seeds=[0, 1, 2, 3, 4],
        configured_cooldown_seconds=2.5,
    )
    assert policy["applies"] is True
    assert policy["process_execution_count"] == 5
    assert policy["execution_unit"] == "one fresh Python subprocess per repetition"
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


def test_preconditioning_policy_is_complete_retained_and_aggregate_excluded():
    policy = sweep.build_preconditioning_policy(seed=1337, execution_count=2)

    assert policy["applies"] is True
    assert policy["process_execution_count"] == 2
    assert [execution["execution_index"] for execution in policy["executions"]] == [
        1,
        2,
    ]
    assert [execution["seed"] for execution in policy["executions"]] == [1337, 1337]
    assert all(
        execution["output_group"] == "preconditioning"
        for execution in policy["executions"]
    )
    assert "excluded from all evidence aggregates" in policy["timing_scope"]


def test_declared_cooldown_is_finite_bounded_and_explicit():
    assert (
        sweep.declared_inter_execution_cooldown_seconds(
            {"outer_inter_execution_cooldown_seconds": 30}
        )
        == 30.0
    )
    for protocol in (
        {},
        {"outer_inter_execution_cooldown_seconds": True},
        {"outer_inter_execution_cooldown_seconds": -1},
        {"outer_inter_execution_cooldown_seconds": 301},
        {"outer_inter_execution_cooldown_seconds": float("nan")},
    ):
        with pytest.raises(ValueError, match="outer_inter_execution_cooldown"):
            sweep.declared_inter_execution_cooldown_seconds(protocol)


def test_declared_preconditioning_count_is_nonnegative_and_explicit():
    assert sweep.declared_preconditioning_runs({"outer_preconditioning_runs": 0}) == 0
    assert sweep.declared_preconditioning_runs({"outer_preconditioning_runs": 2}) == 2
    for protocol in (
        {},
        {"outer_preconditioning_runs": True},
        {"outer_preconditioning_runs": -1},
        {"outer_preconditioning_runs": 1.0},
    ):
        with pytest.raises(ValueError, match="outer_preconditioning_runs"):
            sweep.declared_preconditioning_runs(protocol)


def test_run_one_seed_rejects_unrecognized_output_group(tmp_path):
    with pytest.raises(ValueError, match="unsupported reference-sweep run group"):
        sweep.run_one_seed(
            tmp_path / "child.py",
            workload_id="image-classification",
            variant=None,
            profile="max",
            seed=42,
            execution_index=1,
            mode="inference",
            phase=None,
            device="cpu",
            attempt_dir=tmp_path,
            timeout_seconds=1.0,
            evidence_tier="development",
            allowed_data_modes=frozenset({"real"}),
            run_group="../escape",
        )


def test_promotion_rejects_preconditioning_override_that_differs_from_protocol():
    with pytest.raises(SystemExit) as exc_info:
        sweep.main(
            [
                "--workload",
                "image-classification",
                "--preconditioning-runs",
                "-1",
            ]
        )
    assert exc_info.value.code == 2

    assert (
        sweep.main(
            [
                "--workload",
                "image-classification",
                "--preconditioning-runs",
                "1",
                "--evidence-tier",
                "promotion-candidate",
            ]
        )
        == 2
    )


def test_promotion_rejects_cooldown_override_that_differs_from_protocol():
    assert (
        sweep.main(
            [
                "--workload",
                "image-classification",
                "--inter-execution-cooldown-seconds",
                "1",
                "--evidence-tier",
                "promotion-candidate",
            ]
        )
        == 2
    )


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "300.1"])
def test_main_rejects_invalid_inter_execution_cooldowns(value):
    with pytest.raises(SystemExit) as exc_info:
        sweep.main(
            [
                "--workload",
                "image-classification",
                "--inter-execution-cooldown-seconds",
                value,
            ]
        )
    assert exc_info.value.code == 2


def _write_fake_lineage_package(path: Path) -> None:
    report = {
        "workload": "causal-language-modeling",
        "profile": "max",
        "mode": "training",
        "status": "passed",
        "data_mode": "real",
        "quality": {"quality_required": True, "target_met": True},
    }
    manifest = {
        "workload": "causal-language-modeling",
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
        "workload": "causal-language-modeling",
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


def test_causal_prefill_sweep_uses_performance_bearing_semantics():
    from mlperf.registry import load_registry

    workload = load_registry()["causal-language-modeling"]
    selected = sweep.execution_contract(workload, mode="inference", phase="prefill")
    rows = []
    for execution_index in range(1, 6):
        rows.append(
            {
                "execution_index": execution_index,
                "requested_seed": 1337,
                "evidence_valid": True,
                "result_role": "performance-bearing",
                "reference_metric_role": "performance",
                "primary_metric_declared": "prefill_tokens_per_sec",
                "primary_metric_key": "prefill_tokens_per_sec",
                "primary_metric_value": 28_000.0 + execution_index,
                "quality_metric_key": "prefill_tokens_per_sec",
                "functional_metric_declared": "prefill_tokens_per_sec",
                "functional_metric_value": 28_000.0 + execution_index,
                "quality_target_met": True,
                "host_power": stable_power_record(),
                "comparison_fingerprint_sha256": "a" * 64,
                "data_mode": "checkpoint-backed",
                "grade": {"target": 0.0},
            }
        )
    acceptance = sweep.performance_acceptance(rows, "canonical functional gate")
    errors = sweep.validate_sweep(
        workload=workload,
        result_role="performance-bearing",
        seeds=[1337] * 5,
        mode="inference",
        phase="prefill",
        rows=rows,
        sensitivity={},
        acceptance=acceptance,
        evidence_tier="promotion-candidate",
    )
    assert errors == []

    basis = sweep.build_basis(
        workload=workload,
        result_role="performance-bearing",
        selected_contract=selected,
        profile="max",
        rows=rows,
        primary_aggregate=sweep.aggregate(
            [float(row["primary_metric_value"]) for row in rows]
        ),
        primary_metric_name="prefill_tokens_per_sec",
        quality_aggregate=None,
        quality_metric_name=None,
        dataset_mode=None,
        eligible=True,
        evidence_tier="promotion-candidate",
    )
    assert basis["result_role"] == "performance-bearing"
    assert basis["quality_target"] is None
    assert basis["functional_check"] == {
        "metric": "prefill_tokens_per_sec",
        "metric_key": "prefill_tokens_per_sec",
        "condition": "Every run must pass the canonical functional gate.",
        "target": 0.0,
    }


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


def test_build_row_preserves_requested_and_executed_devices():
    row = sweep.build_row(
        {
            "requested_seed": 42,
            "report_recorded_seed": 42,
            "manifest_recorded_seed": 42,
            "execution_ok": True,
            "evidence_valid": True,
            "device_requested": "cpu",
            "device_executed": "cpu",
        }
    )

    assert row["device_requested"] == "cpu"
    assert row["device_executed"] == "cpu"


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
        "quality_target": 0.85,
        "quality_direction": "higher",
        "result_role": "score-bearing",
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
        "host_power": stable_power_record(),
        "reproduce": {"env": {"MLPERF_EDU_MAX_SEED": str(seed)}},
    }


def test_preconditioning_is_retained_but_excluded_from_aggregates(
    tmp_path, monkeypatch
):
    calls = []

    def run_one_seed(_bootstrap, **kwargs):
        calls.append(kwargs)
        result = fake_result(
            kwargs["seed"],
            0.90,
            primary_metric_value=10.0,
        )
        result["execution_index"] = kwargs["execution_index"]
        result["primary_metric_declared"] = "inference_and_evaluation_seconds"
        result["primary_metric_key"] = "inference_and_evaluation_seconds"
        return result

    monkeypatch.setattr(sweep, "run_one_seed", run_one_seed)
    monkeypatch.setattr(
        sweep, "source_snapshot", lambda: {"git_sha": "abc", "git_dirty": False}
    )

    code = sweep.main(
        [
            "--workload",
            "image-classification",
            "--profile",
            "max",
            "--seeds",
            "0,1",
            "--preconditioning-runs",
            "1",
            "--output-dir",
            str(tmp_path),
            "--evidence-tier",
            "development",
        ]
    )

    assert code == 0
    assert len(calls) == 3
    assert calls[0]["seed"] == 42
    assert calls[0]["run_group"] == "preconditioning"
    assert "run_group" not in calls[1]
    assert "run_group" not in calls[2]
    summary = json.loads(next(tmp_path.glob("*/evidence_summary.json")).read_text())
    assert summary["preconditioning"]["process_execution_count"] == 1
    assert len(summary["preconditioning"]["runs"]) == 1
    assert len(summary["runs"]) == 2
    assert summary["aggregate"]["primary_metric"]["count"] == 2


def test_failed_preconditioning_skips_all_measured_repetitions(tmp_path, monkeypatch):
    calls = []

    def run_one_seed(_bootstrap, **kwargs):
        calls.append(kwargs)
        result = fake_result(kwargs["seed"], 0.90, primary_metric_value=10.0)
        result["execution_index"] = kwargs["execution_index"]
        result["evidence_valid"] = False
        result["invalid_reasons"] = ["injected preparation failure"]
        return result

    monkeypatch.setattr(sweep, "run_one_seed", run_one_seed)
    monkeypatch.setattr(
        sweep, "source_snapshot", lambda: {"git_sha": "abc", "git_dirty": False}
    )

    code = sweep.main(
        [
            "--workload",
            "image-classification",
            "--profile",
            "max",
            "--seeds",
            "0,1",
            "--preconditioning-runs",
            "2",
            "--output-dir",
            str(tmp_path),
            "--evidence-tier",
            "development",
        ]
    )

    assert code == 1
    assert len(calls) == 1
    assert calls[0]["run_group"] == "preconditioning"
    summary = json.loads(next(tmp_path.glob("*/evidence_summary.json")).read_text())
    assert summary["status"] == "invalid"
    assert len(summary["preconditioning"]["runs"]) == 1
    assert summary["runs"] == []


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
            "image-classification",
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
            "image-classification",
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
