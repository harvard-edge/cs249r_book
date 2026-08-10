import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

from mlperf import assets, edu_cli
from mlperf.edu_cli import (
    default_collection_for,
    enrich_report_for_display,
    package_dataset_policy_issue,
)
from mlperf.manifest import build_provd
from mlperf.registry import load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args, cwd=None, env_extra=None):
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "MLPERF_EDU_NO_BROWSER": "1",
    }
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "mlperf_edu.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )


def replacement_pending(workload_id):
    baseline = load_registry()[workload_id].raw.get("verified_baseline") or {}
    return baseline.get("replacement_required") is True


def test_cli_help():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "usage: mlperf" in result.stdout
    assert "MLPerf EDU" in result.stdout
    assert "Defaults to the mlperf-edu suite" in result.stdout
    assert "Common user path: init, health, list, fetch, run, report." in result.stdout
    assert "health" in result.stdout
    assert "validate" in result.stdout
    assert "--maturity" not in result.stdout


def test_profile_help_describes_contract_depth_not_max_as_the_full_suite():
    result = run_cli("run", "--help")

    assert result.returncode == 0
    normalized = " ".join(result.stdout.split())
    assert (
        "max=complete quality evaluation for the selected workload(s)" in normalized
    )
    assert "max=full suite" not in normalized


def test_keyboard_interrupt_returns_shell_interrupt_status(monkeypatch, capsys):
    def interrupted(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(edu_cli, "cmd_doctor", interrupted)

    assert edu_cli.main(["doctor"]) == 130
    output = capsys.readouterr().out
    assert "Interrupted" in output
    assert "resumable progress" in output


def test_list_help_explains_workload_filter():
    result = run_cli("list", "--help")
    assert result.returncode == 0
    assert "Filter by workload id or canonical workload" in result.stdout
    assert "Workload id for variant listing" not in result.stdout


def test_execution_commands_do_not_advertise_unimplemented_model_override():
    for command in ("init", "fetch", "run", "show", "validate"):
        result = run_cli(command, "--help")
        assert result.returncode == 0, result.stderr
        assert "--model" not in result.stdout

    info = run_cli("info", "--help")
    assert info.returncode == 0, info.stderr
    assert "--model" in info.stdout


def test_run_validate_and_health_advertise_device_selection():
    for command in ("run", "validate", "health"):
        result = run_cli(command, "--help")
        assert result.returncode == 0, result.stderr
        assert "--device {auto,cpu,cuda,mps}" in result.stdout


def test_run_dashboard_stays_closed_by_default_and_can_be_opened():
    parser = edu_cli.build_parser()

    default_args = parser.parse_args(
        ["run", "--workload", "causal-language-modeling", "--profile", "min"]
    )
    enabled_args = parser.parse_args(
        [
            "run",
            "--workload",
            "causal-language-modeling",
            "--profile",
            "min",
            "--open-report",
        ]
    )

    assert default_args.open_report is False
    assert enabled_args.open_report is True
    help_result = run_cli("run", "--help")
    assert "Open the generated HTML dashboard." in help_result.stdout
    assert "--no-open-report" in help_result.stdout


def test_health_dashboard_stays_closed_by_default_and_can_be_opened():
    parser = edu_cli.build_parser()

    default_args = parser.parse_args(["health"])
    enabled_args = parser.parse_args(["health", "--open-report"])

    assert default_args.open_report is False
    assert enabled_args.open_report is True
    assert default_args.output_dir == "submissions/health"


def test_browser_opening_can_be_suppressed_for_automation(
    tmp_path, monkeypatch, capsys
):
    dashboard = tmp_path / "dashboard.html"
    dashboard.write_text("<!doctype html>")
    monkeypatch.setenv("MLPERF_EDU_NO_BROWSER", "1")

    def unexpected_open(_uri):
        raise AssertionError("webbrowser.open must not run in automation")

    monkeypatch.setattr(edu_cli.webbrowser, "open", unexpected_open)

    assert edu_cli.open_report_path(dashboard) is False
    assert "browser opening suppressed" in capsys.readouterr().out


def test_doctor_passes():
    result = run_cli("doctor")
    assert result.returncode == 0
    assert "mlperf-edu" in result.stdout
    assert "registry" in result.stdout


def test_audit_json_exposes_draft_evidence_and_quality_margin():
    result = run_cli("audit", "--workload", "keyword-spotting", "--format", "json")
    payload = json.loads(result.stdout)

    assert payload["schema"] == "mlperf-edu-public-contract-audit/0.2"
    source = payload["draft_evidence_source"]
    assert source["source_git_sha"] == "163d42ee3df54ab122543469ccf2b6b3bd119455"
    assert source["claim_scope"] in {
        "current-source",
        "historical-draft",
        "unverified-installed-artifact",
    }
    evidence = payload["workloads"][0]["draft_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["integrity_ok"] is True
    assert evidence[0]["evidence_class"] == "five-run-verified"
    assert evidence[0]["run_count"] == 5
    assert evidence[0]["quality"]["nominal_headroom"] == pytest.approx(0.002, abs=1e-7)
    assert evidence[0]["repeatability"]["passed"] is True
    conformance = payload["workloads"][0]["adapter_conformance"]
    assert conformance["status"] == "quality-preserving-nonidentical"
    assert conformance["promotion_eligible"] is False
    assert any(
        "public promotion is blocked" in issue["issue"] for issue in payload["issues"]
    )


def test_installed_draft_evidence_is_not_compared_with_unrelated_git_checkout(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(edu_cli, "find_project_root", lambda: tmp_path)

    def reject_git_probe(*args, **kwargs):
        raise AssertionError(
            "an installed artifact must not probe the current Git tree"
        )

    monkeypatch.setattr(edu_cli.subprocess, "run", reject_git_probe)

    source = edu_cli.draft_evidence_source_status()

    assert source["source_git_sha"] == "163d42ee3df54ab122543469ccf2b6b3bd119455"
    assert source["current_git_sha"] is None
    assert source["current_git_dirty"] is None
    assert source["current_revision_match"] is None
    assert source["claim_scope"] == "unverified-installed-artifact"


def test_verify_rejects_a_report_without_a_traceback(tmp_path):
    report = tmp_path / "workload_report.json"
    report.write_text(json.dumps({"workload": "example", "status": "passed"}))

    result = run_cli("verify", str(report))

    assert result.returncode == 1
    assert "Invalid provenance manifest" in result.stdout
    assert "expected a .provd.json file" in result.stdout
    assert "Traceback" not in result.stderr


def test_run_workload_records_resolved_mode_and_phase(tmp_path, monkeypatch):
    workloads = load_registry()
    observed = {}

    def inference_runner(_workload, _output_dir):
        return {"status": "passed"}

    monkeypatch.setattr(
        edu_cli, "load_runner", lambda _workload, _profile: inference_runner
    )
    report = edu_cli.run_workload(
        workloads["image-classification"],
        "max",
        tmp_path,
        mode="inference",
    )
    assert report["mode"] == "inference"
    assert report["phase"] is None

    def phased_runner(_workload, _output_dir, *, mode, phase):
        observed.update(mode=mode, phase=phase)
        return {"status": "passed"}

    monkeypatch.setattr(
        edu_cli, "load_runner", lambda _workload, _profile: phased_runner
    )
    report = edu_cli.run_workload(
        workloads["causal-language-modeling"],
        "max",
        tmp_path,
        mode="inference",
        phase="prefill",
    )
    assert observed == {"mode": "inference", "phase": "prefill"}
    assert report["mode"] == "inference"
    assert report["phase"] == "prefill"


def test_pro_profile_repeats_max_runner_and_emits_reviewable_artifacts(
    tmp_path, monkeypatch
):
    workload = load_registry()["image-classification"]
    calls = []

    def max_runner(_workload, output_dir):
        calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subrun_report = output_dir / "image-classification_max_report.json"
        subrun_report.write_text('{"status": "passed"}\n')
        return {
            "status": "passed",
            "backend": "cpu",
            "data_mode": "real",
            "mode": "inference",
            "model": {"id": "resnet8", "revision": "fixture-revision"},
            "model_source": {
                "repository": "https://example.test/model",
                "revision": "fixture-revision",
            },
            "config": {"batch_size": 1, "precision": "float32"},
            "metrics": {"inference_seconds": 1.0, "top1_accuracy": 0.87},
            "quality": {
                "metric": "top1_accuracy",
                "target": 0.85,
                "quality_required": True,
                "target_met": True,
            },
            "artifacts": {"report": str(subrun_report)},
        }

    def fake_load_runner(_workload, profile):
        return max_runner if profile == "max" else None

    monkeypatch.setattr(edu_cli, "load_runner", fake_load_runner)
    monkeypatch.setenv("MLPERF_EDU_PRO_REPETITIONS", "2")

    report = edu_cli.run_workload(
        workload,
        "pro",
        tmp_path,
        mode="inference",
    )

    assert len(calls) == 2
    assert calls[0].relative_to(tmp_path).parts == (
        ".pro_evidence",
        "image-classification",
        "rep1",
    )
    assert report["status"] == "passed"
    assert report["profile"] == "pro"
    assert report["mode"] == "inference"
    assert report["pro_policy"]["repetitions"] == 2
    assert report["metrics"]["repetitions"] == 2
    assert report["metrics"]["inference_seconds_mean"] == 1.0
    assert report["model"] == {"id": "resnet8", "revision": "fixture-revision"}
    assert report["config"] == {"batch_size": 1, "precision": "float32"}
    assert report["artifacts"]["subrun_1_report"] == str(
        calls[0] / "image-classification_max_report.json"
    )
    assert report["artifacts"]["subrun_2_report"] == str(
        calls[1] / "image-classification_max_report.json"
    )
    assert report["readiness_stage"] == "quality"
    assert report["quality"]["quality_required"] is True
    assert report["quality"]["target_met"] is True
    assert len(report["subruns"]) == 2
    report_path = tmp_path / "image-classification_pro_report.json"
    manifest_path = tmp_path / "image-classification_pro.provd.json"
    assert json.loads(report_path.read_text())["status"] == "passed"
    assert json.loads(manifest_path.read_text())["workload"] == "image-classification"
    report["experiment_run"] = {
        "plan_id": "fixture-plan",
        "plan_source_sha256": "sha256:fixture",
        "index": 1,
        "name": "fixture-condition",
        "role": "baseline",
        "device": "cpu",
        "repetitions": 2,
        "environment": {},
    }
    edu_cli.export_workload_reports([report], {workload.id: workload})
    stored_report = json.loads(report_path.read_text())
    assert stored_report["experiment_run"]["name"] == "fixture-condition"
    assert edu_cli.verify_provd(
        manifest_path, repo_root=edu_cli.find_project_root()
    ).all_ok
    assert report_path.with_suffix(".html").is_file()
    assert report_path.with_suffix(".csv").is_file()
    pro_html = report_path.with_suffix(".html").read_text()
    assert "Quality Results" in pro_html
    assert "Target attainment" in pro_html
    assert "102.4%" in pro_html
    assert "image-classification · fixture-condition (baseline)" in pro_html
    assert str(tmp_path) not in pro_html
    grade = edu_cli.grade_manifest(manifest_path)
    assert grade["verified"] is True
    assert grade["passed"] is True
    assert grade["quality_ready"] is True


def test_pro_profile_rejects_runner_quality_contract_drift(tmp_path, monkeypatch):
    workload = load_registry()["image-classification"]

    def max_runner(_workload, _output_dir):
        return {
            "status": "passed",
            "metrics": {"top1_accuracy": 0.99},
            "quality": {
                "metric": "top1_accuracy",
                "target": 0.99,
                "quality_required": True,
                "target_met": True,
            },
        }

    monkeypatch.setattr(
        edu_cli,
        "load_runner",
        lambda _workload, profile: max_runner if profile == "max" else None,
    )

    with pytest.raises(ValueError, match="differs from the registry"):
        edu_cli.run_workload(workload, "pro", tmp_path)


def test_pro_profile_preserves_functional_only_max_readiness(tmp_path, monkeypatch):
    workload = load_registry()["image-generation"]

    def functional_max_runner(_workload, _output_dir):
        return {
            "status": "passed",
            "backend": "cpu",
            "data_mode": "synthetic-deterministic-functional-probe",
            "metrics": {"duration_seconds": 0.1},
            "quality": {
                "metric": "fid",
                "target": 1.79,
                "quality_required": False,
                "target_met": None,
            },
            "functional_readiness": {
                "stage": "functional",
                "authoritative_quality_contract_executed": False,
                "promotion_eligible": False,
                "next_stage": "quality-conformance",
            },
            "artifacts": {},
        }

    monkeypatch.setattr(
        edu_cli,
        "load_runner",
        lambda _workload, profile: functional_max_runner if profile == "max" else None,
    )

    report = edu_cli.run_workload(workload, "pro", tmp_path)

    assert report["status"] == "passed"
    assert report["readiness_stage"] == "functional"
    assert report["quality"]["quality_required"] is False
    assert report["quality"]["target_met"] is None
    assert report["functional_readiness"] == {
        "schema": "mlperf-edu-functional-readiness/0.1",
        "stage": "functional",
        "end_to_end_execution": True,
        "authoritative_quality_contract_executed": False,
        "repeatability_verified": False,
        "promotion_eligible": False,
        "next_stage": "quality-conformance",
    }
    assert report["subruns"][0]["quality_required"] is False
    assert report["subruns"][0]["target_met"] is None

    edu_cli.export_workload_reports([report], {workload.id: workload})
    pro_html = (tmp_path / "image-generation_pro_report.html").read_text()
    assert "Functional Readiness" in pro_html
    assert "Any max target shown is context only" in pro_html

    grade = edu_cli.grade_manifest(tmp_path / "image-generation_pro.provd.json")
    assert grade["verified"] is True
    assert grade["passed"] is True
    assert grade["quality_ready"] is False


def test_quality_metric_lookup_supports_generic_pro_aggregates():
    metrics = {
        "humaneval_plus_pass_at_1_mean": 0.573,
        "humaneval_plus_pass_at_1_min": 0.573,
        "humaneval_plus_pass_at_1_max": 0.573,
    }

    assert (
        edu_cli.metric_key_for_quality("humaneval_plus_pass_at_1", metrics)
        == "humaneval_plus_pass_at_1_mean"
    )


def test_run_summary_distinguishes_quality_and_functional_passes(tmp_path, capsys):
    report_path = tmp_path / "aggregate.json"
    exports = {"html": tmp_path / "aggregate.html", "csv": tmp_path / "aggregate.csv"}
    reports = [
        {
            "status": "passed",
            "quality": {"quality_required": True, "target_met": True},
        },
        {
            "status": "passed",
            "quality": {"quality_required": False, "target_met": None},
        },
    ]

    status = edu_cli.print_run_summary("max", reports, report_path, exports)

    assert status == 0
    output = capsys.readouterr().out
    assert "1 quality-passed" in output
    assert "1 functional-passed" in output


def test_doctor_json_marks_bad_selection_as_failure():
    result = run_cli("doctor", "--workload", "does-not-exist", "--format", "json")
    assert result.returncode == 1
    data = json.loads(result.stdout)
    checks = {check["name"]: check for check in data["checks"]}
    assert checks["registry"]["status"] == "ok"
    assert checks["selection"]["status"] == "fail"
    assert "does-not-exist" in checks["selection"]["detail"]
    assert data["selected_workloads"] == []


def test_doctor_does_not_gate_recommendation_on_the_retired_dlrm_environment():
    """Recommendation trains locally since the contract moved to NCF.

    The preflight used to demand Criteo terms acceptance and DLRM paths, so it
    reported a gated environment for a workload that runs, and told the reader
    to set variables nothing consults. Preflight and runner must agree.
    """
    result = run_cli(
        "doctor",
        "--workload",
        "recommendation",
        "--profile",
        "max",
        env_extra={
            "MLPERF_EDU_CRITEO_TERMS_ACCEPTED": "",
            "MLPERF_EDU_DLRM_DATA_DIR": "",
            "MLPERF_EDU_DLRM_CHECKPOINT": "",
        },
    )

    assert "research environment is gated" not in result.stdout
    assert "MLPERF_EDU_DLRM_DATA_DIR" not in result.stdout
    assert "MLPERF_EDU_CRITEO_TERMS_ACCEPTED" not in result.stdout


def test_doctor_emits_no_environment_handoffs():
    """No workload hands off to another environment any more.

    Recommendation left this set when its contract moved to NCF on
    MovieLens-20M. Reinforcement learning left it when the PyTorch adapter
    replaced the CUDA and TensorFlow 1.x MiniGo container. A handoff appearing
    here again means a workload stopped running locally.
    """
    for workload_id in ("reinforcement-learning", "recommendation"):
        result = run_cli(
            "doctor", "--workload", workload_id, "--profile", "max", "--format", "json"
        )
        checks = json.loads(result.stdout)["checks"]
        assert not any("handoff" in check for check in checks), workload_id


def test_list_default_contains_canonical_language_modeling():
    result = run_cli("list")
    assert result.returncode == 0
    assert "causal-language-modeling" in result.stdout
    assert "Public" in result.stdout
    assert "experimental" in result.stdout


def test_list_discovery_subjects():
    suites = run_cli("list", "suites")
    assert suites.returncode == 0, suites.stdout + suites.stderr
    assert "MLPerf EDU Suites" in suites.stdout
    assert "language" in suites.stdout

    profiles = run_cli("list", "profiles")
    assert profiles.returncode == 0, profiles.stdout + profiles.stderr
    assert "MLPerf EDU Profiles" in profiles.stdout
    assert "min" in profiles.stdout
    assert "max" in profiles.stdout
    assert "pro" in profiles.stdout

    profiles_json = run_cli("list", "profiles", "--format", "json")
    assert profiles_json.returncode == 0, profiles_json.stdout + profiles_json.stderr
    profile_counts = {
        row["profile"]: row["workloads"]
        for row in json.loads(profiles_json.stdout)["profiles"]
    }
    assert profile_counts == {"min": 4, "max": 14, "pro": 14}


def test_info_profile_shows_default_selection():
    result = run_cli("info", "--profile", "min")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Profile: min" in result.stdout
    assert "Selected 4 workload(s) for profile min (default)." in result.stdout
    assert "List details: mlperf list --profile min" in result.stdout


def test_default_collection_for_profile_defaults():
    assert (
        default_collection_for(
            Namespace(collection=None, suite=None, workload=None, profile="min")
        )
        == "starter"
    )
    assert (
        default_collection_for(
            Namespace(collection=None, suite=None, workload=None, profile="max")
        )
        == "all"
    )
    assert (
        default_collection_for(
            Namespace(collection=None, suite=None, workload=None, profile="pro")
        )
        == "research"
    )
    assert (
        default_collection_for(
            Namespace(collection=None, suite="vision", workload=None, profile="max")
        )
        is None
    )
    assert (
        default_collection_for(
            Namespace(collection="all", suite=None, workload=None, profile="min")
        )
        == "all"
    )


def test_explicit_collection_overrides_profile_default():
    result = run_cli(
        "run",
        "--profile",
        "min",
        "--collection",
        "all",
        "--dry-run",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Selected 14 workload(s) for profile min (collection:all)." in result.stdout
    assert (
        "Suite coverage: graph=1, language=5, recommendation=1, reinforcement=1, "
        "timeseries=1, tiny=3, vision=2"
    ) in result.stdout


def test_report_enrichment_defaults_quality_required_from_public_contract():
    workloads = load_registry()

    report = {"workload": "causal-language-modeling", "quality": {}}
    enrich_report_for_display(report, workloads)
    assert report["quality"]["quality_required"] is False
    assert report["quality"]["target_basis"] == "literature"
    assert report["quality"]["target_kind"] == "published_reference_reproduction"
    assert "reference_protocol" not in report["quality"]
    assert "gated" not in report["quality"]

    explicit_not_required = {
        "workload": "causal-language-modeling",
        "quality": {"quality_required": False},
    }
    enrich_report_for_display(explicit_not_required, workloads)
    assert explicit_not_required["quality"]["quality_required"] is False
    assert "gated" not in explicit_not_required["quality"]


def test_dashboard_derives_functional_readiness_from_report_semantics():
    report = {
        "workloads": [
            {
                "workload": "setup-a",
                "status": "passed",
                "quality": {
                    "quality_required": False,
                    "metric": "accuracy",
                    "metric_key": "accuracy",
                    "target": 0.8,
                    "direction": "higher",
                    "target_kind": "inherited_acceptance_gate",
                },
                "metrics": {"accuracy": 0.5},
            },
            {
                "workload": "setup-b",
                "status": "passed",
                "quality": {
                    "quality_required": False,
                    "metric": "loss",
                    "metric_key": "loss",
                    "target": 1.0,
                    "direction": "lower",
                    "target_kind": "published_reference_reproduction",
                },
                "metrics": {"loss": 4.0},
            },
        ]
    }

    html = edu_cli.quality_dashboard_html(report)

    assert "Functional Readiness" in html
    assert "2 of 2 functional paths passed; no quality claim" in html
    assert "Functional paths passed" in html
    assert "2 / 2" in html
    assert html.count("Path passed") == 2
    assert html.count("Setup and execution") == 2
    assert "Diagnostic: accuracy 50.00%" in html
    assert "Any max target shown is context only" in html
    assert "Max target: Target ≥ 80.00% · inherited acceptance gate" in html
    assert "Target met" not in html


def test_dashboard_derives_quality_results_and_target_attainment():
    report = {
        "workloads": [
            {
                "workload": "quality-pass",
                "status": "passed",
                "quality": {
                    "quality_required": True,
                    "metric": "accuracy",
                    "metric_key": "accuracy",
                    "target": 0.8,
                    "direction": "higher",
                    "target_met": True,
                },
                "metrics": {"accuracy": 0.9},
            },
            {
                "workload": "quality-fail",
                "status": "quality_failed",
                "quality": {
                    "quality_required": True,
                    "metric": "fid",
                    "metric_key": "fid",
                    "target": 1.79,
                    "direction": "lower",
                    "target_met": False,
                },
                "metrics": {"fid": 1.801554},
            },
        ]
    }

    html = edu_cli.quality_dashboard_html(report)

    assert "Quality Results" in html
    assert "1 of 2 quality targets met" in html
    assert "Quality targets met" in html
    assert "1 / 2" in html
    assert "Target attainment" in html
    assert "112.5%" in html
    assert "99.4%" in html
    assert "Target not met" in html
    assert "Run failed" not in html
    assert "Functional Readiness" not in html


def test_dashboard_keeps_nonpass_states_distinct():
    report = {
        "workloads": [
            {
                "workload": "environment-gated",
                "status": "not_implemented",
                "max_execution": "environment-gated-quality-conformance",
                "quality": {"quality_required": True, "metric": "accuracy"},
                "metrics": {},
            },
            {
                "workload": "skipped",
                "status": "skipped",
                "quality": {"quality_required": True, "metric": "accuracy"},
                "metrics": {},
            },
            {
                "workload": "unsupported",
                "status": "unsupported",
                "quality": {"quality_required": True, "metric": "accuracy"},
                "metrics": {},
            },
            {
                "workload": "execution-failed",
                "status": "execution_failed",
                "quality": {"quality_required": True, "metric": "accuracy"},
                "metrics": {},
            },
        ]
    }

    html = edu_cli.quality_dashboard_html(report)

    assert "Environment gated" in html
    assert "Skipped" in html
    assert "Unsupported" in html
    assert "Run failed" in html


def test_dashboard_uses_separate_meters_for_mixed_results():
    report = {
        "workloads": [
            {
                "workload": "functional",
                "status": "passed",
                "quality": {"quality_required": False},
                "metrics": {"duration_seconds": 0.1},
            },
            {
                "workload": "quality",
                "status": "passed",
                "quality": {
                    "quality_required": True,
                    "metric": "accuracy",
                    "metric_key": "accuracy",
                    "target": 0.8,
                    "direction": "higher",
                    "target_met": True,
                },
                "metrics": {"accuracy": 0.9},
            },
        ]
    }

    html = edu_cli.quality_dashboard_html(report)

    assert "Benchmark Results" in html
    assert "1 of 1 quality targets met · 1 of 1 functional paths passed" in html
    assert "Quality targets met" in html
    assert "Functional paths passed" in html


def test_functional_result_table_does_not_pair_diagnostic_with_quality_target(
    tmp_path,
):
    report = {
        "schema": "mlperf-edu-report/0.1",
        "profile": "min",
        "workloads": [
            {
                "workload": "code-generation",
                "suite": "language",
                "profile": "min",
                "status": "passed",
                "metrics": {"generated_tokens": 8},
                "quality": {
                    "quality_required": False,
                    "metric": "humaneval_plus_pass_at_1",
                    "metric_key": "generated_tokens",
                    "target": 0.573,
                    "direction": "higher",
                },
            }
        ],
    }
    output = tmp_path / "functional.html"

    edu_cli.write_html_report(report, output, source_path=tmp_path / "report.json")

    detail_table = output.read_text().split("<h2>Detailed Results</h2>", 1)[1]
    assert "generated tokens: 8" in detail_table
    assert "Not evaluated in this run" in detail_table
    assert "0.573" not in detail_table


def test_quality_target_attainment_is_direction_and_tolerance_aware():
    assert edu_cli.quality_target_attainment(
        0.7145, 0.7174, "higher", 0.0029
    ) == pytest.approx(100.0)
    assert edu_cli.quality_target_attainment(1.801554, 1.79, "lower") == pytest.approx(
        99.3586, rel=1e-4
    )
    assert edu_cli.quality_target_attainment("0.9", 0.8, "higher") is None


def test_nanogpt_training_enrichment_excludes_prior_promoted_results():
    workloads = load_registry()
    workload = workloads["causal-language-modeling"]
    report = {
        "workload": workload.id,
        "profile": "max",
        "metrics": {"cross_entropy_loss": 2.1},
        "quality": {
            "metric": "cross_entropy_loss",
            "target": 2.3,
            "target_met": True,
            "variance_summary": {
                "median": 1.9,
                "evidence_id": "prior-training-evidence",
                "source_git_sha": "1" * 40,
            },
        },
        "verified_baseline": {
            "evidence_id": "prior-training-evidence",
            "source_git_sha": "1" * 40,
        },
    }

    enrich_report_for_display(report, workloads)

    assert report["metrics"]["cross_entropy_loss"] == 2.1
    assert report["quality"]["target_met"] is True
    assert report["quality"]["target_basis"] == "literature"
    assert "reference_protocol" not in report["quality"]
    serialized = json.dumps(report, sort_keys=True)
    assert "variance_summary" not in serialized
    assert "verified_baseline" not in serialized
    assert "prior-training-evidence" not in serialized
    assert "source_git_sha" not in serialized


def test_nanogpt_inference_lineage_excludes_source_promoted_results():
    workloads = load_registry()
    workload = workloads["causal-language-modeling"]
    report = {
        "workload": workload.id,
        "profile": "max",
        "metrics": {"prefill_tokens_per_sec": 123.0},
        "quality": {
            "metric": "prefill_tokens_per_sec",
            "target": 0.0,
            "target_met": True,
        },
        "checkpoint_provenance": {
            "checkpoint_sha256": f"sha256:{'2' * 64}",
            "source_report_sha256": f"sha256:{'3' * 64}",
            "source_manifest_sha256": f"sha256:{'4' * 64}",
            "source_quality_value": 2.1,
            "source_quality_target_met": True,
            "source_verified_baseline": {
                "evidence_id": "prior-source-evidence",
                "evidence_sha256": "5" * 64,
                "source_git_sha": "6" * 40,
                "median": 1.9,
            },
        },
    }

    enrich_report_for_display(report, workloads)

    lineage = report["checkpoint_provenance"]
    assert lineage["source_report_sha256"] == f"sha256:{'3' * 64}"
    assert lineage["source_manifest_sha256"] == f"sha256:{'4' * 64}"
    assert lineage["source_quality_value"] == 2.1
    assert "source_verified_baseline" not in lineage
    assert "performance_reference_protocol" not in report
    serialized = json.dumps(report, sort_keys=True)
    assert "prior-source-evidence" not in serialized
    assert "evidence_sha256" not in serialized
    assert "source_git_sha" not in serialized


def test_execution_lineage_distinguishes_pretrained_and_run_trained_models():
    workloads = load_registry()
    inferred = {
        "workload": "image-classification",
        "profile": "max",
        "quality": {},
    }
    enrich_report_for_display(inferred, workloads)
    assert inferred["mode"] == "inference"
    assert inferred["execution_lineage"]["mode"] == "inference"

    pretrained = {
        "workload": "code-generation",
        "profile": "max",
        "mode": "inference",
        "model_source": {
            "type": "huggingface-pinned",
            "repo_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "revision": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
        },
        "quality": {},
    }
    enrich_report_for_display(pretrained, workloads)

    pretrained_lineage = pretrained["execution_lineage"]
    assert pretrained_lineage["training"]["status"] == (
        "upstream-pretrained-checkpoint"
    )
    assert pretrained_lineage["checkpoint"]["role"] == (
        "upstream-pretrained-checkpoint"
    )
    assert pretrained_lineage["inference"]["status"] == "executed-in-this-run"
    assert pretrained_lineage["inference"]["adapter"].endswith(
        ":run_code_generation_max"
    )

    trained = {
        "workload": "causal-language-modeling",
        "profile": "max",
        "mode": "training",
        "artifacts": {"checkpoint": "/tmp/nanogpt.pt"},
        "quality": {},
    }
    enrich_report_for_display(trained, workloads)

    trained_lineage = trained["execution_lineage"]
    assert trained_lineage["training"]["status"] == "executed-in-this-run"
    assert trained_lineage["training"]["adapter"].endswith(
        ":run_causal_language_modeling_max"
    )
    assert trained_lineage["checkpoint"]["role"] == "produced-by-this-run"


def test_show_workload():
    result = run_cli("show", "causal-language-modeling")
    assert result.returncode == 0
    assert "Workload: causal-language-modeling" in result.stdout
    assert "min, max, pro" in result.stdout
    assert "public_status" in result.stdout
    assert "experimental" in result.stdout
    assert "evaluator" in result.stdout
    assert "cross_entropy_loss" in result.stdout
    assert "default_mode" in result.stdout
    assert "training" in result.stdout
    assert "default_phase" in result.stdout
    assert "full" in result.stdout
    assert "quality_direction" in result.stdout
    assert "lower" in result.stdout
    assert "quality_tolerance" in result.stdout
    assert "max_execution" in result.stdout
    assert "source_suite" not in result.stdout
    assert "maturity" not in result.stdout


def test_no_workload_is_environment_gated():
    """Every registered workload executes its contract on the target platform.

    Two workloads used to be gated. Recommendation left when its contract moved
    from DLRM on Criteo Terabyte to MLPerf v0.5 NCF on MovieLens-20M.
    Reinforcement learning left when the PyTorch adapter replaced the CUDA and
    TensorFlow 1.x MiniGo container. A workload reappearing here means the
    suite stopped being runnable as shipped.
    """
    import yaml

    for path in sorted((PROJECT_ROOT / "registry" / "suites").glob("*/*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        contract = spec.get("canonical_max_contract") or {}
        assert contract.get("execution_status") != (
            "environment-gated-quality-conformance"
        ), f"{spec['id']} is environment-gated"


def test_info_dataset_shows_asset_dossier():
    result = run_cli("info", "--dataset", "tinyshakespeare")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dataset: tinyshakespeare" in result.stdout
    assert "mit-repository-public-domain-text" in result.stdout
    assert "public-ok-fetch-only" in result.stdout
    assert "expected_download_bytes" in result.stdout
    assert "5600000" in result.stdout
    assert "causal-language-modeling" in result.stdout


def test_cache_list_and_verify_known_missing_workload(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    listed = run_cli(
        "cache",
        "list",
        "--workload",
        "causal-language-modeling",
        env_extra={"MLPERF_EDU_DATA_DIR": str(data_dir)},
    )
    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert "missing" in listed.stdout
    assert "mit-repository-public-domain-text" in listed.stdout
    assert "release=public-ok-fetch-only" in listed.stdout

    verified = run_cli(
        "cache",
        "verify",
        "--workload",
        "causal-language-modeling",
        env_extra={"MLPERF_EDU_DATA_DIR": str(data_dir)},
    )
    assert verified.returncode == 1, verified.stdout + verified.stderr
    assert "missing" in verified.stdout


def test_fetch_workload_dry_run():
    result = run_cli(
        "fetch",
        "--workload",
        "causal-language-modeling",
        "--profile",
        "min",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert (
        "Selected 1 workload(s) for profile min (causal-language-modeling)."
        in result.stdout
    )
    assert "Would fetch 1 workload" in result.stdout
    assert "causal-language-modeling: tinyshakespeare" in result.stdout
    assert "underlying Shakespeare text is public domain" in result.stdout
    assert "terms=mit-repository-public-domain-text" in result.stdout
    assert "release=public-ok-fetch-only" in result.stdout


def test_fetch_visual_wake_words_dry_run_discloses_exact_source():
    result = run_cli(
        "fetch",
        "--workload",
        "visual-wake-words",
        "--profile",
        "max",
        "--dry-run",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "mlperf-tiny-vww-eval" in result.stdout
    assert "vw_coco2014_96.tar.gz" in result.stdout
    assert "mlcommons-coco-review-required" in result.stdout
    assert "needs-release-decision" in result.stdout


def test_fetch_anomaly_detection_dry_run_discloses_selective_source():
    result = run_cli(
        "fetch",
        "--workload",
        "anomaly-detection",
        "--profile",
        "max",
        "--dry-run",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "mlperf-tiny-anomaly-eval" in result.stdout
    assert "dev_data_ToyCar.zip" in result.stdout
    assert "selective range fetch" in result.stdout
    assert "cc-by-4.0-mlcommons-attribution" in result.stdout


def test_fetch_new_quality_assets_dry_run_discloses_authoritative_sources():
    code = run_cli(
        "fetch",
        "--workload",
        "code-generation",
        "--profile",
        "max",
        "--dry-run",
    )
    assert code.returncode == 0, code.stdout + code.stderr
    assert "Qwen2.5-Coder-0.5B-Instruct@ea3f2471" in code.stdout
    assert "HumanEvalPlus.jsonl.gz" in code.stdout
    assert "evalplus/archive/899b2b31" in code.stdout
    assert "public-ok-fetch-only" in code.stdout

    functions = run_cli(
        "fetch",
        "--workload",
        "function-calling",
        "--profile",
        "max",
        "--dry-run",
    )
    assert functions.returncode == 0, functions.stdout + functions.stderr
    assert "Qwen3-1.7B@70d244cc" in functions.stdout
    assert "gorilla/archive/6ea57973" in functions.stdout
    assert "upstream-terms-review-required" in functions.stdout

    images = run_cli(
        "fetch",
        "--workload",
        "image-generation",
        "--profile",
        "max",
        "--dry-run",
    )
    assert images.returncode == 0, images.stdout + images.stderr
    assert "edm-cifar10-32x32-cond-vp.pkl" in images.stdout
    assert "cifar10-32x32.npz" in images.stdout
    assert "MLPerf Tiny model/index" not in images.stdout


def test_fetch_manual_quality_assets_returns_actionable_nonzero_status():
    reinforcement = run_cli(
        "fetch", "--workload", "reinforcement-learning", "--profile", "max"
    )
    assert reinforcement.returncode == 2
    assert "MANUAL ACTION REQUIRED" in " ".join(reinforcement.stdout.split())
    assert "professional-move inputs" in reinforcement.stdout


def test_fetch_min_profile_uses_consolidated_workload_identity():
    result = run_cli("fetch", "--profile", "min", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Selected 4 workload(s) for profile min (default)." in result.stdout
    assert "causal-language-modeling: tinyshakespeare" in result.stdout
    assert "nanogpt-prefill" not in result.stdout
    assert "nanogpt-decode" not in result.stdout


def test_std_profile_is_not_a_public_alias(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "causal-language-modeling",
        "--profile",
        "std",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode != 0
    assert "invalid choice: 'std'" in result.stderr
    assert not any(tmp_path.iterdir())


def test_set_is_not_a_public_selector(tmp_path):
    result = run_cli(
        "run",
        "--set",
        "starter",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode != 0
    assert "unrecognized arguments: --set starter" in result.stderr
    assert not any(tmp_path.iterdir())


def test_init_min_runs_smoke_validation_and_reports(tmp_path):
    output_dir = tmp_path / "init_smoke"
    result = run_cli(
        "init",
        "--profile",
        "min",
        "--output-dir",
        str(output_dir),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MLPerf EDU Local Paths" in result.stdout
    assert "data cache" in result.stdout
    assert "model cache" in result.stdout
    assert "Next commands" in result.stdout
    assert "mlperf health" in result.stdout
    assert "mlperf show image-classification" in result.stdout
    assert "mlperf fetch --workload image-classification --profile max" in result.stdout
    assert "Running min-profile smoke validation" in result.stdout
    assert "min run complete" in result.stdout

    reports = list(output_dir.glob("mlperf_edu_min_*.json"))
    csv_reports = list(output_dir.glob("mlperf_edu_min_*.csv"))
    html_reports = list(output_dir.glob("mlperf_edu_min_*.html"))
    assert len(reports) == 1
    assert len(csv_reports) == 1
    assert len(html_reports) == 1
    data = json.loads(reports[0].read_text())
    assert data["mlperf_suite"] == "mlperf-edu"
    assert data["profile"] == "min"
    assert "set" not in data
    assert data["selection"] == {"kind": "default", "name": "default"}
    assert len(data["workloads"]) == 4


def test_validate_coverage_dry_run_lists_all_min_suites(tmp_path):
    result = run_cli(
        "validate",
        "coverage",
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MLPerf EDU Validation: coverage" in result.stdout
    assert "min-all" in result.stdout
    assert "dry-run complete" in result.stdout
    assert not any(tmp_path.iterdir())


def test_health_dry_run_uses_all_registered_min_paths(tmp_path):
    result = run_cli(
        "health",
        "--dry-run",
        "--no-open-report",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "min-all" in result.stdout
    assert "all workloads" in result.stdout
    assert not any(tmp_path.iterdir())


def test_health_success_points_to_authoritative_quality_journey(monkeypatch, capsys):
    monkeypatch.setattr(edu_cli, "cmd_validate", lambda _args: 0)
    args = edu_cli.build_parser().parse_args(["health"])

    assert edu_cli.cmd_health(args) == 0

    output = " ".join(capsys.readouterr().out.split())
    assert "Next: choose a workload" in output
    assert "mlperf show <workload>" in output
    assert "mlperf run --profile max" in output


def test_suite_filtered_health_uses_the_selected_suite(tmp_path):
    result = run_cli(
        "health",
        "--suite",
        "recommendation",
        "--dry-run",
        "--no-open-report",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "min-recommendation" in result.stdout
    assert "suite:recommendation" in result.stdout


def test_validation_runner_exception_still_writes_failure_report(tmp_path, monkeypatch):
    def fail_run(_args):
        raise RuntimeError("missing licensed benchmark environment")

    monkeypatch.setattr(edu_cli, "cmd_run", fail_run)
    args = Namespace(
        registry=None,
        preset="coverage",
        preset_option=None,
        legacy_level=None,
        suite=["recommendation"],
        output_dir=str(tmp_path),
        skip_doctor=True,
        skip_grade=False,
        keep_going=True,
        dry_run=False,
        open_report=False,
        device="cpu",
    )

    assert edu_cli.cmd_validate(args) == 1
    reports = list(tmp_path.glob("mlperf_validate_coverage_*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text())
    assert payload["status"] == "failed"
    assert payload["validations"][0]["status"] == "run_failed"
    assert payload["validations"][0]["error_type"] == "RuntimeError"
    assert (
        "missing licensed benchmark environment" in payload["validations"][0]["error"]
    )
    html = reports[0].with_suffix(".html").read_text()
    assert "Needs attention" in html
    assert "missing licensed benchmark environment" in html
    assert "overflow-wrap:anywhere" in html


def test_validate_release_dry_run_includes_min_max_and_research_pro(tmp_path):
    result = run_cli(
        "validate",
        "release",
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MLPerf EDU Validation: release" in result.stdout
    assert "min-all" in result.stdout
    assert "max-all" in result.stdout
    assert "pro-research" in result.stdout
    assert not any(tmp_path.iterdir())


def test_validate_pro_dry_run_lists_research_collection(tmp_path):
    result = run_cli(
        "validate",
        "pro",
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MLPerf EDU Validation: pro" in result.stdout
    assert "pro-research" in result.stdout
    assert "research workloads" in result.stdout
    assert "dry-run complete" in result.stdout
    assert not any(tmp_path.iterdir())


def test_validate_release_suite_dry_run_includes_all_three_profiles(tmp_path):
    result = run_cli(
        "validate",
        "release",
        "--suite",
        "tiny",
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "min-tiny" in result.stdout
    assert "max-tiny" in result.stdout
    assert "pro-tiny" in result.stdout
    assert not any(tmp_path.iterdir())


def test_validate_max_dry_run_lists_product_max_suites(tmp_path):
    result = run_cli(
        "validate",
        "max",
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MLPerf EDU Validation: max" in result.stdout
    assert "max-all" in result.stdout
    assert "all workloads" in result.stdout
    assert "dry-run complete" in result.stdout
    assert not any(tmp_path.iterdir())


def test_validation_seed_environment_preserves_per_workload_defaults(monkeypatch):
    for name in ("MLPERF_EDU_SEED", "MLPERF_EDU_MAX_SEED"):
        monkeypatch.delenv(name, raising=False)

    selection = edu_cli.validation_seed_environment("max")

    assert selection == {
        "seed": None,
        "source": "per_workload_canonical_default",
        "set_max_seed": False,
    }


def test_validation_seed_environment_preserves_explicit_seed(monkeypatch):
    monkeypatch.setenv("MLPERF_EDU_SEED", "3")

    selection = edu_cli.validation_seed_environment("max")

    assert selection == {
        "seed": 3,
        "source": "MLPERF_EDU_SEED",
        "set_max_seed": False,
    }


def test_validate_legacy_level_alias_maps_to_coverage(tmp_path):
    result = run_cli(
        "validate",
        "--level",
        "min",
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MLPerf EDU Validation: coverage" in result.stdout
    assert "min-all" in result.stdout
    assert not any(tmp_path.iterdir())


def test_run_with_power_writes_aggregate_power_report(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "causal-language-modeling",
        "--profile",
        "min",
        "--power",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    aggregate = next(tmp_path.glob("mlperf_edu_min_*.json"))
    data = json.loads(aggregate.read_text())
    assert data["power"]["source"] == "estimated_nominal"
    assert data["power"]["average_watts"] > 0
    assert data["power"]["energy_joules"] >= 0

    with aggregate.with_suffix(".csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]["power_average_watts"]) > 0
    assert float(rows[0]["energy_joules"]) >= 0

    html = aggregate.with_suffix(".html").read_text()
    assert "Average Watts" in html
    assert "Energy Joules" in html

    summary = run_cli("report", str(aggregate))
    assert summary.returncode == 0, summary.stdout + summary.stderr
    assert "power_average_watts:" in summary.stdout


def test_multi_workload_power_stays_aggregate_level(tmp_path):
    result = run_cli(
        "run",
        "--profile",
        "min",
        "--power",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    aggregate = next(tmp_path.glob("mlperf_edu_min_*.json"))
    data = json.loads(aggregate.read_text())
    assert data["power"]["average_watts"] > 0
    assert len(data["workloads"]) > 1

    with aggregate.with_suffix(".csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(data["workloads"]) + 1
    workload_rows = [row for row in rows if row["workload"] != "__aggregate__"]
    aggregate_rows = [row for row in rows if row["workload"] == "__aggregate__"]
    assert len(aggregate_rows) == 1
    assert {row["power_average_watts"] for row in workload_rows} == {""}
    assert {row["energy_joules"] for row in workload_rows} == {""}
    assert float(aggregate_rows[0]["power_average_watts"]) > 0
    assert float(aggregate_rows[0]["energy_joules"]) >= 0


def test_report_command_exports_json_csv_html(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "causal-language-modeling",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dashboard:" in result.stdout
    assert "browser opening suppressed" not in result.stdout

    report_path = tmp_path / "causal-language-modeling_training_min_report.json"
    manifest_path = tmp_path / "causal-language-modeling_training_min.provd.json"
    assert report_path.with_suffix(".html").is_file()
    assert report_path.with_suffix(".csv").is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["dataset"] == "tinyshakespeare"
    assert report["dataset_asset"]["id"] == "tinyshakespeare"
    assert (
        report["dataset_asset"]["license_status"] == "mit-repository-public-domain-text"
    )
    assert report["dataset_asset"]["public_release_status"] == "public-ok-fetch-only"
    assert "score-bearing candidate" in report["dataset_asset"]["public_result_use"]
    assert report["quality"]["target_basis"] == "literature"
    assert report["quality"]["target_kind"] == "published_reference_reproduction"
    assert "reference_protocol" not in report["quality"]
    assert report["quality"]["quality_required"] is False
    assert report["execution_lineage"]["training"]["status"] == ("executed-in-this-run")
    assert report["execution_lineage"]["checkpoint"]["role"] == "runtime-model"
    assert report["run_fingerprint"]["schema"] == "mlperf-edu-run-fingerprint/0.1"
    assert report["run_fingerprint"]["hardware"]["fingerprint_hash"]
    assert report["run_fingerprint"]["software"]["python"]
    assert (
        report["run_fingerprint"]["execution"]["workload"] == "causal-language-modeling"
    )
    assert report["run_fingerprint"]["execution"]["profile"] == "min"
    assert report["run_fingerprint"]["execution"]["data_modes"] == [
        "synthetic-deterministic"
    ]

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr

    json_path = tmp_path / "manual.json"
    csv_path = tmp_path / "manual.csv"
    html_path = tmp_path / "manual.html"

    json_result = run_cli(
        "report", str(report_path), "--format", "json", "--output", str(json_path)
    )
    assert json_result.returncode == 0, json_result.stdout + json_result.stderr
    manual_json = json.loads(json_path.read_text())
    assert manual_json["workload"] == "causal-language-modeling"
    assert (
        manual_json["run_fingerprint"]["execution"]["workload"]
        == "causal-language-modeling"
    )

    csv_result = run_cli(
        "report", str(report_path), "--format", "csv", "--output", str(csv_path)
    )
    assert csv_result.returncode == 0, csv_result.stdout + csv_result.stderr
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["workload"] == "causal-language-modeling"
    assert rows[0]["suite"] == "language"
    assert rows[0]["profile"] == "min"
    assert rows[0]["mode"] == "training"
    assert rows[0]["phase"] == ""
    assert rows[0]["scenario"] == "training"
    assert rows[0]["status"] == "passed"
    assert rows[0]["backend"] == "pytorch-cpu"
    assert rows[0]["device_requested"] == "auto"
    assert rows[0]["device_executed"] == "cpu"
    assert rows[0]["data_mode"] == "synthetic-deterministic"
    assert rows[0]["dataset"] == "tinyshakespeare"
    assert rows[0]["dataset_license_status"] == "mit-repository-public-domain-text"
    assert rows[0]["dataset_public_release_status"] == "public-ok-fetch-only"
    assert "score-bearing candidate" in rows[0]["dataset_public_use"]
    assert rows[0]["dataset_release_next_step"].startswith("Keep the pinned commit")
    assert rows[0]["metric"] == "loss"
    assert rows[0]["target"] == "1.4697"
    assert rows[0]["target_kind"] == "published_reference_reproduction"
    assert rows[0]["target_basis"] == "literature"
    assert rows[0]["reference_runs"] == ""
    assert rows[0]["acceptance_runs"] == "1"
    assert rows[0]["reference_statistic"] == ""
    assert rows[0]["reference_protocol"] == ""
    assert rows[0]["direction"] == "lower"
    assert rows[0]["quality_required"] == "False"
    assert "gated" not in rows[0]
    assert float(rows[0]["value"]) > 0
    assert float(rows[0]["duration_seconds"]) >= 0
    assert float(rows[0]["throughput"]) > 0

    html_result = run_cli(
        "report", str(report_path), "--format", "html", "--output", str(html_path)
    )
    assert html_result.returncode == 0, html_result.stdout + html_result.stderr
    html = html_path.read_text()
    assert "MLPerf EDU Report: causal-language-modeling" in html
    assert "Functional Readiness" in html
    assert "Run Configuration" in html
    assert "Model Lineage" in html
    assert "Provenance" in html
    assert "Training" in html
    assert "Checkpoint" in html
    assert "Inference" in html
    assert "Evaluation" in html
    assert "Path passed" in html
    assert "Max target: Target ≤ 1.4697 · published reference reproduction" in html
    assert "loss" in html
    assert "literature" in html
    assert "fingerprint_hash" in html
    assert "Assets and Provenance" in html
    assert "Quality Decision" in html
    assert "Not evaluated in this run" in html
    assert "Gated" not in html
    assert "mit-repository-public-domain-text" in html
    assert "public-ok-fetch-only" in html
    assert "passed" in html


def test_package_and_grade_verified_manifest(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "image-classification",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    manifest_path = tmp_path / "image-classification_min.provd.json"
    report_path = tmp_path / "image-classification_min_report.json"
    package_path = tmp_path / "submission.zip"
    grade_path = tmp_path / "grade.json"

    assert report_path.with_suffix(".html").is_file()
    assert report_path.with_suffix(".csv").is_file()

    package = run_cli("package", str(manifest_path), "--output", str(package_path))
    assert package.returncode == 0, package.stdout + package.stderr
    assert package_path.is_file()
    with zipfile.ZipFile(package_path) as zf:
        names = set(zf.namelist())
        index = json.loads(zf.read("package_index.json"))
        packaged_manifest = json.loads(zf.read(f"manifest/{manifest_path.name}"))
        packaged_report = json.loads(zf.read(f"report/{report_path.name}"))
        packaged_html = zf.read(
            f"report/{report_path.with_suffix('.html').name}"
        ).decode()
        packaged_csv = zf.read(
            f"report/{report_path.with_suffix('.csv').name}"
        ).decode()
        assert names - {"package_index.json"} == {
            item["path"] for item in index["included_files"]
        }
        for item in index["included_files"]:
            payload = zf.read(item["path"])
            assert item["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()
            assert item["n_bytes"] == len(payload)
    assert "package_index.json" in names
    assert f"manifest/{manifest_path.name}" in names
    assert f"report/{report_path.name}" in names
    assert f"report/{report_path.with_suffix('.html').name}" in names
    assert f"report/{report_path.with_suffix('.csv').name}" in names
    assert index["schema"] == "mlperf-edu-package/0.2"
    assert index["workload"] == "image-classification"
    assert all(check["ok"] for check in index["verification"])
    assert index["source_manifest"] == f"manifest/{manifest_path.name}"
    assert all(not os.path.isabs(item["path"]) for item in index["included_files"])
    assert "signature" not in packaged_manifest
    assert packaged_manifest["integrity"]["authenticated"] is False
    leaves = packaged_manifest["leaves"]
    assert not os.path.isabs(leaves["measurement"]["report_path"])
    assert not os.path.isabs(packaged_report["artifacts"]["report"])
    assert not os.path.isabs(packaged_report["artifacts"]["provenance"])
    packaged_report_json = json.dumps(packaged_report)
    assert str(tmp_path) not in packaged_report_json
    assert str(Path.cwd()) not in packaged_report_json
    assert str(tmp_path) not in packaged_html
    assert str(tmp_path) not in packaged_csv

    grade = run_cli("grade", str(tmp_path), "--output", str(grade_path))
    assert grade.returncode == 0, grade.stdout + grade.stderr
    assert "Grade summary: 1 passed, 0 failed" in grade.stdout
    summary = json.loads(grade_path.read_text())
    assert summary["schema"] == "mlperf-edu-grade/0.1"
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["warning_count"] == 0
    assert summary["results"][0]["workload"] == "image-classification"
    assert summary["results"][0]["verified"] is True
    assert summary["results"][0]["quality_required"] is False
    assert "gated" not in summary["results"][0]
    assert summary["results"][0]["target_met"] == ""
    assert summary["results"][0]["warning_count"] == 0
    assert summary["results"][0]["warnings"] == []

    assignment_path = tmp_path / "assignment.yaml"
    package_grade_path = tmp_path / "package-grade.json"
    assignment_path.write_text(
        """\
schema: mlperf-edu-assignment/0.1
id: image-classification-portable-readiness-lab
requirements:
  - workload: image-classification
    profile: min
    count: 1
    quality:
      required: false
"""
    )
    package_grade = run_cli(
        "grade",
        str(package_path),
        "--assignment",
        str(assignment_path),
        "--output",
        str(package_grade_path),
    )
    assert package_grade.returncode == 0, package_grade.stdout + package_grade.stderr
    assert (
        "Assignment image-classification-portable-readiness-lab: passed"
        in package_grade.stdout
    )
    package_summary = json.loads(package_grade_path.read_text())
    assert package_summary["assignment"]["passed"] is True
    assert package_summary["results"][0]["package_verified"] is True
    assert package_summary["results"][0]["manifest"].startswith(str(package_path))


def test_report_baseline_comparison_separates_quality_and_performance(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    output_path = tmp_path / "comparison.json"
    base_report = {
        "schema": "mlperf-edu-report/0.1",
        "workload": "image-classification",
        "suite": "vision",
        "profile": "max",
        "mode": "inference",
        "phase": None,
        "scenario": "offline",
        "status": "passed",
        "backend": "pytorch-cpu",
        "data_mode": "real",
        "dataset": {
            "name": "cifar10",
            "split": "MLPerf-Tiny-200-sample-accuracy-set",
            "sha256": "sha256:dataset",
        },
        "evaluator": {
            "repository": "https://github.com/mlcommons/tiny",
            "revision": "pinned",
        },
        "execution_lineage": {
            "mode": "inference",
            "checkpoint": {"revision": "pinned", "sha256": "sha256:model"},
        },
        "config": {"batch_size": 32, "repetitions": 50},
        "metrics": {"top1_accuracy": 0.87, "samples_per_second": 100.0},
        "quality": {
            "metric": "top1_accuracy",
            "target": 0.85,
            "tolerance": 0.0,
            "direction": "higher",
            "target_kind": "inherited_acceptance_gate",
            "quality_required": True,
            "target_met": True,
        },
    }
    current_report = json.loads(json.dumps(base_report))
    current_report["metrics"]["top1_accuracy"] = 0.88
    current_report["metrics"]["samples_per_second"] = 120.0
    baseline_path.write_text(json.dumps(base_report))
    current_path.write_text(json.dumps(current_report))

    compare = run_cli(
        "report",
        str(current_path),
        "--baseline",
        str(baseline_path),
        "--format",
        "json",
        "--output",
        str(output_path),
    )
    assert compare.returncode == 0, compare.stdout + compare.stderr
    comparison = json.loads(output_path.read_text())["baseline_comparison"]
    result = comparison["results"][0]
    assert result["quality_compatible"] is True
    assert result["performance_compatible"] is True
    assert result["quality"]["current_margin"] == pytest.approx(0.03)
    assert result["performance"]["improvement_percent"] == pytest.approx(20.0)

    html_path = tmp_path / "comparison.html"
    html_compare = run_cli(
        "report",
        str(current_path),
        "--baseline",
        str(baseline_path),
        "--format",
        "html",
        "--output",
        str(html_path),
    )
    assert html_compare.returncode == 0, html_compare.stdout + html_compare.stderr
    html = html_path.read_text()
    assert "Baseline Comparison" in html
    assert "Quality Structurally Compatible" in html
    assert "exploratory" in html
    assert "comparison-bars" in html

    current_report["config"]["batch_size"] = 64
    current_path.write_text(json.dumps(current_report))
    incompatible = run_cli(
        "report",
        str(current_path),
        "--baseline",
        str(baseline_path),
        "--format",
        "json",
        "--output",
        str(output_path),
    )
    assert incompatible.returncode == 0, incompatible.stdout + incompatible.stderr
    result = json.loads(output_path.read_text())["baseline_comparison"]["results"][0]
    assert result["quality_compatible"] is True
    assert result["performance_compatible"] is False
    assert "performance comparison fingerprint differs" in result["reasons"]


def test_package_verification_rejects_traversal_before_extraction(tmp_path):
    package_path = tmp_path / "unsafe.zip"
    payload = b"escape"
    index = {
        "schema": "mlperf-edu-package/0.2",
        "manifest": "../escape.provd.json",
        "included_files": [
            {
                "role": "manifest",
                "path": "../escape.provd.json",
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "n_bytes": len(payload),
            }
        ],
    }
    with zipfile.ZipFile(package_path, "w") as zf:
        zf.writestr("package_index.json", json.dumps(index))
        zf.writestr("../escape.provd.json", payload)

    checks = edu_cli.verify_package_archive(package_path, repo_root=PROJECT_ROOT)

    assert not all(ok for _, ok, _ in checks)
    assert any(
        name == "archive.path:../escape.provd.json" and not ok for name, ok, _ in checks
    )


def test_package_verification_rejects_manifest_path_escape(tmp_path):
    package_path = tmp_path / "unsafe-manifest.zip"
    manifest_name = "manifest/submission.provd.json"
    manifest_payload = json.dumps(
        {
            "leaves": {
                "measurement": {"report_path": "../../../escape.json"},
                "weights": {},
                "dataset": {"files": []},
            }
        }
    ).encode()
    index = {
        "schema": "mlperf-edu-package/0.2",
        "manifest": manifest_name,
        "included_files": [
            {
                "role": "manifest",
                "path": manifest_name,
                "sha256": "sha256:" + hashlib.sha256(manifest_payload).hexdigest(),
                "n_bytes": len(manifest_payload),
            }
        ],
    }
    with zipfile.ZipFile(package_path, "w") as zf:
        zf.writestr("package_index.json", json.dumps(index))
        zf.writestr(manifest_name, manifest_payload)

    checks = edu_cli.verify_package_archive(package_path, repo_root=PROJECT_ROOT)

    assert not all(ok for _, ok, _ in checks)
    assert any(
        name == "clean_extraction.manifest_paths" and not ok for name, ok, _ in checks
    )


def test_package_policy_refuses_unresolved_canonical_dataset_bytes():
    manifest = {
        "leaves": {
            "dataset": {
                "name": "cifar10",
                "files": [
                    {"path": "/tmp/test.parquet", "sha256": "sha256:placeholder"}
                ],
            }
        }
    }
    issue = package_dataset_policy_issue(manifest)
    assert issue is not None
    assert "needs-release-decision" in issue
    assert "avoid redistributing" in issue


def test_package_policy_allows_open_or_artifact_free_datasets():
    fashion = {
        "leaves": {
            "dataset": {
                "name": "fashion-mnist",
                "files": [
                    {"path": "/tmp/train-images", "sha256": "sha256:placeholder"}
                ],
            }
        }
    }
    unresolved_without_bytes = {"leaves": {"dataset": {"name": "cifar10", "files": []}}}
    assert package_dataset_policy_issue(fashion) is None
    assert package_dataset_policy_issue(unresolved_without_bytes) is None

    fetch_only = {
        "leaves": {
            "dataset": {
                "name": "tinyshakespeare",
                "files": [{"path": "/tmp/input.txt", "sha256": "sha256:fixture"}],
            }
        }
    }
    assert "public-ok-fetch-only" in package_dataset_policy_issue(fetch_only)


def test_package_carries_all_manifest_dependencies_and_survives_source_removal(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    report_path = source / "toy_report.json"
    manifest_path = source / "toy.provd.json"
    checkpoint_path = source / "toy.pt"
    dataset_root = source / "data"
    dataset_root.mkdir()
    dataset_path = dataset_root / "dataset.bin"
    roofline_path = source / "roofline.json"
    metadata_path = source / "metadata.json"
    checkpoint_path.write_bytes(b"checkpoint")
    dataset_path.write_bytes(b"dataset")
    roofline_path.write_text('{"flops": 1}\n')
    metadata_path.write_text('{"model": "toy"}\n')
    report = {
        "schema": "mlperf-edu-report/0.1",
        "workload": "toy-workload",
        "profile": "max",
        "status": "passed",
        "seed": 7,
        "data_mode": "real",
        "metrics": {"accuracy": 1.0},
        "quality": {
            "metric": "accuracy",
            "target": 0.9,
            "direction": "higher",
            "quality_required": True,
            "target_met": True,
        },
        "dataset_asset": {
            "root": str(dataset_root),
            "hashes": {"files": [{"path": str(dataset_path), "sha256": "fixture"}]},
        },
        "run_fingerprint": {
            "software": {
                "python_executable": str(tmp_path / "venv" / "bin" / "python")
            },
        },
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
            "checkpoint": str(checkpoint_path),
            "model_metadata": str(metadata_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload="toy-workload",
        scenario="train",
        division="open",
        hardware_fingerprint={"platform": "test"},
        report=report,
        report_path=report_path,
        weights_path=checkpoint_path,
        dataset_name="toy-data",
        dataset_files=[dataset_path],
        rng_seed=7,
        roofline_sidecar_path=roofline_path,
        repo_root=source,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    package_path = tmp_path / "toy.zip"

    package = run_cli("package", str(manifest_path), "--output", str(package_path))
    assert package.returncode == 0, package.stdout + package.stderr
    extraction = tmp_path / "extraction"
    with zipfile.ZipFile(package_path) as zf:
        index = json.loads(zf.read("package_index.json"))
        roles = {item["role"] for item in index["included_files"]}
        assert {"manifest", "report", "weights", "dataset", "roofline_sidecar"} <= roles
        assert "report_artifact:model_metadata" in roles
        zf.extractall(extraction)

    shutil.rmtree(source)
    extracted_manifest = extraction / index["manifest"]
    packaged_manifest = json.loads(extracted_manifest.read_text())
    packaged_report_path = (
        extracted_manifest.parent
        / packaged_manifest["leaves"]["measurement"]["report_path"]
    )
    packaged_report = json.loads(packaged_report_path.read_text())
    assert packaged_report["dataset_asset"]["root"] == "../dataset"
    assert packaged_report["dataset_asset"]["hashes"]["files"][0]["path"].startswith(
        "../dataset/"
    )
    assert (
        packaged_report["run_fingerprint"]["software"]["python_executable"]
        == "local-environment:python"
    )
    assert str(tmp_path) not in json.dumps(packaged_report)
    verify = run_cli("verify", str(extracted_manifest))
    assert verify.returncode == 0, verify.stdout + verify.stderr

    dataset_relative = packaged_manifest["leaves"]["dataset"]["files"][0]["path"]
    packaged_dataset = extracted_manifest.parent / dataset_relative
    packaged_dataset.write_bytes(b"tampered")
    tampered = run_cli("verify", str(extracted_manifest))
    assert tampered.returncode == 1
    assert "dataset.files[0].sha256" in tampered.stdout


def test_grade_uses_quality_required_not_legacy_gated(tmp_path):
    report_path = tmp_path / "toy_report.json"
    report = {
        "schema": "mlperf-edu-report/0.1",
        "workload": "toy-workload",
        "profile": "min",
        "status": "passed",
        "metrics": {"accuracy": 0.1, "duration_seconds": 0.01},
        "quality": {
            "metric": "accuracy",
            "target": 0.9,
            "quality_required": True,
            "gated": False,
            "target_met": False,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload="toy-workload",
        scenario="offline",
        division="open",
        hardware_fingerprint={"platform": "test"},
        report=report,
        report_path=report_path,
        repo_root=tmp_path,
    )
    manifest_path = tmp_path / "toy.provd.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    grade = run_cli("grade", str(tmp_path), "--output", str(tmp_path / "grade.json"))
    assert grade.returncode == 1, grade.stdout + grade.stderr
    summary = json.loads((tmp_path / "grade.json").read_text())
    assert summary["passed"] == 0
    assert summary["failed"] == 1
    assert summary["results"][0]["quality_required"] is True
    assert "gated" not in summary["results"][0]
    assert summary["results"][0]["target_met"] is False


def test_grade_rejects_verified_report_with_lowered_registry_target(tmp_path):
    report_path = tmp_path / "image-classification_max_report.json"
    manifest_path = tmp_path / "image-classification_max.provd.json"
    report = {
        "schema": "mlperf-edu-report/0.1",
        "workload": "image-classification",
        "profile": "max",
        "mode": "inference",
        "status": "passed",
        "metrics": {"top1_accuracy": 0.80},
        "quality": {
            "metric": "top1_accuracy",
            "target": 0.80,
            "direction": "higher",
            "tolerance": 0.0,
            "quality_required": True,
            "target_met": True,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload="image-classification",
        scenario="offline",
        division="open",
        hardware_fingerprint={"platform": "test"},
        report=report,
        report_path=report_path,
        repo_root=PROJECT_ROOT,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    grade = edu_cli.grade_manifest(manifest_path)

    assert grade["verified"] is True
    assert grade["canonical_quality_verified"] is False
    assert grade["passed"] is False
    assert grade["quality_ready"] is False
    assert "differs from the registry" in grade["warnings"][0]


def test_nanogpt_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "causal-language-modeling",
        "--mode",
        "training",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr

    report_path = tmp_path / "causal-language-modeling_training_min_report.json"
    manifest_path = tmp_path / "causal-language-modeling_training_min.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["workload"] == "causal-language-modeling"
    assert report["mode"] == "training"
    assert report["status"] == "passed"
    assert report["backend"] == "pytorch-cpu"
    assert report["metrics"]["tokens"] == 32
    assert report["metrics"]["logits_shape"] == [2, 16, 128]

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "verified" in verify.stdout


def test_causal_language_modeling_rejects_phase_for_training(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "causal-language-modeling",
        "--mode",
        "training",
        "--phase",
        "prefill",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode != 0
    assert "phase" in (result.stdout + result.stderr).lower()


def test_nanogpt_prefill_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "causal-language-modeling",
        "--mode",
        "inference",
        "--phase",
        "prefill",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = (
        tmp_path / "causal-language-modeling_inference_prefill_min_report.json"
    )
    manifest_path = (
        tmp_path / "causal-language-modeling_inference_prefill_min.provd.json"
    )
    report = json.loads(report_path.read_text())
    assert report["workload"] == "causal-language-modeling"
    assert report["mode"] == "inference"
    assert report["phase"] == "prefill"
    assert report["status"] == "passed"
    assert report["metrics"]["context_length"] == 32
    assert report["metrics"]["prefill_tokens_per_sec"] > 0

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_nanogpt_decode_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "causal-language-modeling",
        "--mode",
        "inference",
        "--phase",
        "decode",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "causal-language-modeling_inference_decode_min_report.json"
    manifest_path = (
        tmp_path / "causal-language-modeling_inference_decode_min.provd.json"
    )
    report = json.loads(report_path.read_text())
    assert report["workload"] == "causal-language-modeling"
    assert report["mode"] == "inference"
    assert report["phase"] == "decode"
    assert report["status"] == "passed"
    assert report["metrics"]["prefill_ctx"] == 16
    assert report["metrics"]["decode_steps"] == 4
    assert report["metrics"]["output_tokens_per_sec"] > 0

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_image_classification_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "image-classification",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "image-classification_min_report.json"
    manifest_path = tmp_path / "image-classification_min.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["workload"] == "image-classification"
    assert report["status"] == "passed"
    assert report["metrics"]["samples"] == 2
    assert report["metrics"]["logits_shape"] == [2, 10]
    assert report["metrics"]["samples_per_second"] > 0
    assert report["device_requested"] == "auto"
    assert report["device_executed"] == "cpu"

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_run_device_option_overrides_environment(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "image-classification",
        "--profile",
        "min",
        "--device",
        "cpu",
        "--output-dir",
        str(tmp_path),
        env_extra={"MLPERF_EDU_DEVICE": "not-a-device"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "image-classification_min_report.json").read_text())
    assert report["device_requested"] == "cpu"
    assert report["device_executed"] == "cpu"


def test_run_auto_device_option_clears_environment_override(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "image-classification",
        "--profile",
        "min",
        "--device",
        "auto",
        "--output-dir",
        str(tmp_path),
        env_extra={"MLPERF_EDU_DEVICE": "not-a-device"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "image-classification_min_report.json").read_text())
    assert report["device_requested"] == "auto"
    assert report["device_executed"] == "cpu"


def test_run_rejects_unsupported_device_without_traceback(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "image-classification",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
        env_extra={"MLPERF_EDU_DEVICE": "not-a-device"},
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "not supported by MLPerf EDU" in output
    assert "MLPERF_EDU_DEVICE" in output
    assert "mlperf doctor" in output
    assert "Traceback" not in output


def test_device_validation_rejects_unavailable_cuda(monkeypatch):
    import torch

    monkeypatch.setenv("MLPERF_EDU_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable in this PyTorch environment"):
        edu_cli.validate_requested_torch_device()


def test_image_classification_max_run_writes_verifiable_artifacts(tmp_path):
    required_assets = (
        PROJECT_ROOT / "data/cifar10/plain_text/test-00000-of-00001.parquet",
        PROJECT_ROOT / "data/mlperf-tiny-image/pretrainedResnet.tflite",
        PROJECT_ROOT / "data/mlperf-tiny-image/perf_samples_idxs.npy",
    )
    if not all(path.is_file() for path in required_assets):
        pytest.skip("canonical MLPerf Tiny image assets are not cached")
    output_dir = tmp_path / "out"
    result = run_cli(
        "run",
        "--workload",
        "image-classification",
        "--profile",
        "max",
        "--output-dir",
        str(output_dir),
        env_extra={"MLPERF_EDU_DEVICE": "cpu"},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = output_dir / "image-classification_max_report.json"
    manifest_path = output_dir / "image-classification_max.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["profile"] == "max"
    assert report["status"] == "passed"
    assert report["data_mode"] == "real"
    assert report["quality"]["quality_required"] is True
    assert report["quality"]["override"] is False
    assert report["quality"]["target_met"] is True
    assert report["metrics"]["top1_accuracy"] == pytest.approx(0.87)
    assert report["metrics"]["evaluation_samples"] == 200
    assert report["metrics"]["samples"] == 10_000
    assert report["config"]["repetitions"] == 50
    assert report["device_requested"] == "cpu"
    assert report["device_executed"] == "cpu"
    assert report["metrics"]["correct"] == 174
    assert report["evaluator"]["name"] == "mlperf-tiny-top1-accuracy"
    assert report["evaluator"]["revision"] == assets.MLPERF_TINY_COMMIT
    assert report["evaluator"]["source_sha256"] == (
        f"sha256:{assets.MLPERF_TINY_IMAGE_EVALUATOR_SHA256}"
    )
    assert Path(report["artifacts"]["weights"]).name == "pretrainedResnet.tflite"
    assert (
        Path(report["artifacts"]["performance_indices"]).name == "perf_samples_idxs.npy"
    )

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_keyword_spotting_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "keyword-spotting",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "keyword-spotting_min_report.json"
    manifest_path = tmp_path / "keyword-spotting_min.provd.json"
    report = json.loads(report_path.read_text())
    assert report["workload"] == "keyword-spotting"
    assert report["status"] == "passed"
    assert report["metrics"]["logits_shape"] == [4, 12]
    assert report["metrics"]["samples_per_second"] > 0

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_visual_wake_words_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "visual-wake-words",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "visual-wake-words_min_report.json"
    manifest_path = tmp_path / "visual-wake-words_min.provd.json"
    report = json.loads(report_path.read_text())
    assert report["workload"] == "visual-wake-words"
    assert report["status"] == "passed"
    assert report["metrics"]["probabilities_shape"] == [4, 2]
    assert report["metrics"]["n_params"] == 210_850

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_anomaly_detection_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "anomaly-detection",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "anomaly-detection_min_report.json"
    manifest_path = tmp_path / "anomaly-detection_min.provd.json"
    report = json.loads(report_path.read_text())
    assert report["workload"] == "anomaly-detection"
    assert report["status"] == "passed"
    assert report["metrics"]["reconstruction_shape"] == [4, 640]
    assert report["metrics"]["n_params"] == 265_864

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_container_probe_matches_the_code_generation_runner():
    """The preflight probe is duplicated on purpose; keep the two in step.

    Importing the runner into the CLI would pull code_generation.py into the
    measurement source-lock closure, and preflight is not a measurement input.
    The cost of that separation is two definitions, so this pins them together:
    if they ever disagree, doctor and the runner would give a user opposite
    answers about whether a valid code-generation run is possible.
    """
    from mlperf import edu_cli
    from mlperf.runners import code_generation

    assert edu_cli.container_engine_available() == code_generation.docker_available()
