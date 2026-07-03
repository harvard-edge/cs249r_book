import csv
import json
import os
import subprocess
import sys
import zipfile
from argparse import Namespace

import torch

from mlperf.edu_cli import default_collection_for, enrich_report_for_display
from mlperf.manifest import build_provd
from mlperf.registry import load_registry


def run_cli(*args, cwd=None, env_extra=None):
    env = {**os.environ, "PYTHONPATH": "src"}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "mlperf_edu.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )


def write_tiny_movielens(root, *, n_users=12, n_items=16, n_ratings=96):
    dataset = root / "movielens" / "ml-100k"
    dataset.mkdir(parents=True)

    with (dataset / "u.user").open("w") as f:
        for user_id in range(1, n_users + 1):
            f.write(f"{user_id}|{20 + user_id}|M|student|00000\n")

    with (dataset / "u.item").open("w", encoding="latin-1") as f:
        for item_id in range(1, n_items + 1):
            genres = ["1" if idx == item_id % 19 else "0" for idx in range(19)]
            fields = [
                str(item_id),
                f"Movie {item_id}",
                "01-Jan-1995",
                "",
                f"http://example.com/{item_id}",
                *genres,
            ]
            f.write("|".join(fields) + "\n")

    with (dataset / "u.data").open("w") as f:
        for idx in range(n_ratings):
            user_id = (idx % n_users) + 1
            item_id = ((idx * 3) % n_items) + 1
            rating = 5 if (user_id + item_id) % 2 == 0 else 2
            f.write(f"{user_id}\t{item_id}\t{rating}\t{idx}\n")


def test_cli_help():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "usage: mlperf" in result.stdout
    assert "MLPerf EDU" in result.stdout
    assert "Defaults to the mlperf-edu suite" in result.stdout
    assert "Common user path: init, list, fetch, audit, run, report." in result.stdout
    assert "validate" in result.stdout
    assert "--maturity" not in result.stdout


def test_list_help_explains_workload_filter():
    result = run_cli("list", "--help")
    assert result.returncode == 0
    assert "Filter by workload id or canonical workload" in result.stdout
    assert "Workload id for variant listing" not in result.stdout


def test_doctor_passes():
    result = run_cli("doctor")
    assert result.returncode == 0
    assert "mlperf-edu" in result.stdout
    assert "registry" in result.stdout


def test_doctor_json_reports_selected_workloads():
    result = run_cli("doctor", "--profile", "min", "--format", "json")
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["schema"] == "mlperf-edu-doctor/0.1"
    assert data["profile"] == "min"
    assert len(data["selected_workloads"]) == 12
    checks = {check["name"]: check for check in data["checks"]}
    assert checks["selection"]["status"] == "ok"
    assert "12 workload(s) for profile min" in checks["selection"]["detail"]
    assert checks["data cache"]["status"] == "ok"
    assert checks["model cache"]["status"] == "ok"

    suite_result = run_cli("doctor", "--suite", "slm", "--profile", "pro", "--format", "json")
    assert suite_result.returncode == 0, suite_result.stdout + suite_result.stderr
    suite_data = json.loads(suite_result.stdout)
    assert suite_data["suite"] == "slm"
    assert {workload["id"] for workload in suite_data["selected_workloads"]} == {
        "slm-decode",
        "slm-quantized-decode",
        "slm-batched-decode",
        "slm-long-context-decode",
    }

    family_result = run_cli("doctor", "--workload", "smollm2-chat-inference", "--format", "json")
    assert family_result.returncode == 0, family_result.stdout + family_result.stderr
    family_data = json.loads(family_result.stdout)
    assert {workload["id"] for workload in family_data["selected_workloads"]} == {
        "slm-decode",
        "slm-quantized-decode",
        "slm-batched-decode",
        "slm-long-context-decode",
    }

    variant_result = run_cli(
        "doctor",
        "--workload",
        "smollm2-chat-inference",
        "--variant",
        "quantized-int8",
        "--format",
        "json",
    )
    assert variant_result.returncode == 0, variant_result.stdout + variant_result.stderr
    variant_data = json.loads(variant_result.stdout)
    assert [workload["id"] for workload in variant_data["selected_workloads"]] == ["slm-quantized-decode"]
    assert variant_data["selected_workloads"][0]["run_selector"] == "smollm2-chat-inference --variant quantized-int8"


def test_doctor_json_marks_bad_selection_as_failure():
    result = run_cli("doctor", "--workload", "does-not-exist", "--format", "json")
    assert result.returncode == 1
    data = json.loads(result.stdout)
    checks = {check["name"]: check for check in data["checks"]}
    assert checks["registry"]["status"] == "ok"
    assert checks["selection"]["status"] == "fail"
    assert "does-not-exist" in checks["selection"]["detail"]
    assert data["selected_workloads"] == []


def test_list_default_contains_nanogpt():
    result = run_cli("list")
    assert result.returncode == 0
    assert "nanogpt-train" in result.stdout
    assert "Public" in result.stdout
    assert "score-bearing" in result.stdout
    assert "performance-bearing" in result.stdout


def test_list_slm_contains_slm_decode():
    result = run_cli("list", "--suite", "slm")
    assert result.returncode == 0
    assert "Workload" in result.stdout
    assert "Internal ID" in result.stdout
    assert "slm-decode" in result.stdout
    assert "slm-quantized-decode" in result.stdout
    assert "slm-batched-decode" in result.stdout
    assert "slm-long-context-decode" in result.stdout
    assert "smollm2-chat-inference --variant baseline" in result.stdout
    assert "smollm2-chat-inference --variant quantized-int8" in result.stdout
    assert "smollm2-chat-inference --variant batched-b4" in result.stdout
    assert "smollm2-chat-inference --variant long-context" in result.stdout
    assert "nanogpt-decode-fp32-b16" not in result.stdout
    assert "performance-bearing" in result.stdout


def test_list_discovery_subjects():
    suites = run_cli("list", "suites")
    assert suites.returncode == 0, suites.stdout + suites.stderr
    assert "MLPerf EDU Suites" in suites.stdout
    assert "language" in suites.stdout
    assert "slm" in suites.stdout

    profiles = run_cli("list", "profiles")
    assert profiles.returncode == 0, profiles.stdout + profiles.stderr
    assert "MLPerf EDU Profiles" in profiles.stdout
    assert "min" in profiles.stdout
    assert "max" in profiles.stdout
    assert "pro" in profiles.stdout

    profiles_json = run_cli("list", "profiles", "--format", "json")
    assert profiles_json.returncode == 0, profiles_json.stdout + profiles_json.stderr
    profile_counts = {row["profile"]: row["workloads"] for row in json.loads(profiles_json.stdout)["profiles"]}
    assert profile_counts == {"min": 12, "max": 30, "pro": 12}


def test_info_profile_shows_default_selection():
    result = run_cli("info", "--profile", "min")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Profile: min" in result.stdout
    assert "Selected 12 workload(s) for profile min (default)." in result.stdout
    assert "Suite coverage: agent=1, distributed=1, graph=1, language=3" in result.stdout
    assert "List details: mlperf list --profile min" in result.stdout


def test_list_variants_for_canonical_workload():
    result = run_cli("list", "variants", "--workload", "smollm2-chat-inference")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "smollm2-chat-inference" in result.stdout
    assert "baseline" in result.stdout
    assert "quantized-int8" in result.stdout
    assert "batched-b4" in result.stdout
    assert "long-context" in result.stdout
    assert "slm-decode" in result.stdout
    assert "slm-quantized-decode" in result.stdout
    assert "slm-batched-decode" in result.stdout
    assert "slm-long-context-decode" in result.stdout


def test_list_variants_filters_by_suite():
    result = run_cli("list", "variants", "--suite", "slm")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "smollm2-chat-inference" in result.stdout
    assert "slm-quantized-decode" in result.stdout
    assert "slm-batched-decode" in result.stdout
    assert "slm-long-context-decode" in result.stdout
    assert "nanogpt-inference" not in result.stdout


def test_list_filters_by_public_status():
    result = run_cli("list", "--public-status", "score-bearing")
    assert result.returncode == 0
    assert "nanogpt-train" in result.stdout
    assert "resnet18-train" in result.stdout
    assert "slm-decode" not in result.stdout


def test_list_profile_filters_default_selection():
    min_result = run_cli("list", "--profile", "min", "--format", "json")
    assert min_result.returncode == 0, min_result.stdout + min_result.stderr
    min_data = json.loads(min_result.stdout)
    assert min_data["profile"] == "min"
    min_ids = {workload["id"] for workload in min_data["workloads"]}
    assert min_ids == {
        "anomaly-ae-train",
        "micro-dlrm-distributed",
        "micro-dlrm-train",
        "micro-gnn-train",
        "micro-lstm-train",
        "micro-rl-train",
        "nano-rag-agent",
        "nanogpt-decode",
        "nanogpt-prefill",
        "nanogpt-train",
        "resnet18-train",
        "slm-decode",
    }
    prefill = next(workload for workload in min_data["workloads"] if workload["id"] == "nanogpt-prefill")
    assert "prefill_tokens_per_sec" in prefill["functional_check"]

    max_result = run_cli("list", "--profile", "max", "--format", "json")
    assert max_result.returncode == 0, max_result.stdout + max_result.stderr
    max_workloads = json.loads(max_result.stdout)["workloads"]
    assert "maturity" not in max_workloads[0]
    assert len(max_workloads) == 30
    max_ids = {workload["id"] for workload in max_workloads}
    assert "slm-decode" in max_ids
    assert "nano-rag-agent" in max_ids
    assert "micro-bert-train" in max_ids
    slm = next(workload for workload in max_workloads if workload["id"] == "slm-decode")
    assert slm["workload"] == "smollm2-chat-inference"
    assert slm["internal_id"] == "slm-decode"
    assert slm["canonical_workload"] == "smollm2-chat-inference"
    assert slm["variant"] == "baseline"
    assert slm["run_selector"] == "smollm2-chat-inference --variant baseline"

    pro_result = run_cli("list", "--profile", "pro", "--format", "json")
    assert pro_result.returncode == 0, pro_result.stdout + pro_result.stderr
    pro_ids = {workload["id"] for workload in json.loads(pro_result.stdout)["workloads"]}
    assert "micro-bert-train" in pro_ids
    assert "slm-batched-decode" in pro_ids
    assert "slm-long-context-decode" in pro_ids
    assert "nanogpt-decode-spec" in pro_ids
    assert "nanogpt-train" not in pro_ids


def test_list_workload_filters_canonical_variants():
    result = run_cli("list", "--workload", "nanogpt-inference", "--format", "json")
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["schema"] == "mlperf-edu-list-workloads/0.1"
    assert data["workload"] == "nanogpt-inference"
    assert {workload["id"] for workload in data["workloads"]} == {
        "nanogpt-decode",
        "nanogpt-decode-fp16-b16",
        "nanogpt-decode-fp32-b16",
        "nanogpt-decode-spec",
        "nanogpt-prefill",
    }
    assert {workload["dataset"] for workload in data["workloads"]} == {"prompt-suite-local"}

    prefill_result = run_cli(
        "list",
        "--workload",
        "nanogpt-inference",
        "--variant",
        "prefill",
        "--format",
        "json",
    )
    assert prefill_result.returncode == 0, prefill_result.stdout + prefill_result.stderr
    prefill_data = json.loads(prefill_result.stdout)
    assert prefill_data["variant"] == "prefill"
    assert [workload["id"] for workload in prefill_data["workloads"]] == ["nanogpt-prefill"]

    min_result = run_cli("list", "--workload", "nanogpt-inference", "--profile", "min", "--format", "json")
    assert min_result.returncode == 0, min_result.stdout + min_result.stderr
    min_ids = {workload["id"] for workload in json.loads(min_result.stdout)["workloads"]}
    assert min_ids == {
        "nanogpt-decode",
        "nanogpt-decode-fp16-b16",
        "nanogpt-decode-fp32-b16",
        "nanogpt-decode-spec",
        "nanogpt-prefill",
    }

    suite_profile_result = run_cli("list", "matrix", "--suite", "language", "--profile", "min", "--format", "json")
    assert suite_profile_result.returncode == 0, suite_profile_result.stdout + suite_profile_result.stderr
    suite_profile_ids = {workload["workload"] for workload in json.loads(suite_profile_result.stdout)["workloads"]}
    assert suite_profile_ids == {
        "micro-bert-train",
        "nano-lora-finetune",
        "nano-moe-train",
        "nanogpt-decode",
        "nanogpt-decode-fp16-b16",
        "nanogpt-decode-fp32-b16",
        "nanogpt-decode-spec",
        "nanogpt-prefill",
        "nanogpt-train",
    }


def test_list_matrix_summarizes_roles_and_profiles():
    max_result = run_cli("list", "matrix", "--profile", "max", "--format", "json")
    assert max_result.returncode == 0, max_result.stdout + max_result.stderr
    max_data = json.loads(max_result.stdout)
    assert max_data["schema"] == "mlperf-edu-workload-matrix/0.1"
    slm = next(workload for workload in max_data["workloads"] if workload["workload"] == "slm-quantized-decode")
    assert slm["run_selector"] == "smollm2-chat-inference --variant quantized-int8"
    assert slm["role"] == "optimization"
    assert slm["default_profiles"] == "max, pro"
    assert slm["quality"] == "generated_tokens 8"

    slm_pro_result = run_cli("list", "matrix", "--suite", "slm", "--profile", "pro", "--format", "json")
    assert slm_pro_result.returncode == 0, slm_pro_result.stdout + slm_pro_result.stderr
    slm_pro_data = json.loads(slm_pro_result.stdout)
    assert {workload["workload"] for workload in slm_pro_data["workloads"]} == {
        "slm-decode",
        "slm-quantized-decode",
        "slm-batched-decode",
        "slm-long-context-decode",
    }
    slm_profiles = {workload["workload"]: workload["default_profiles"] for workload in slm_pro_data["workloads"]}
    assert slm_profiles == {
        "slm-decode": "min, max, pro",
        "slm-quantized-decode": "max, pro",
        "slm-batched-decode": "max, pro",
        "slm-long-context-decode": "max, pro",
    }

    pro_result = run_cli("list", "matrix", "--profile", "pro", "--format", "json")
    assert pro_result.returncode == 0, pro_result.stdout + pro_result.stderr
    pro_data = json.loads(pro_result.stdout)
    speculative = next(workload for workload in pro_data["workloads"] if workload["workload"] == "nanogpt-decode-spec")
    assert speculative["run_selector"] == "nanogpt-inference --variant speculative"
    assert speculative["role"] == "test-time-compute"
    assert speculative["default_profiles"] == "max, pro"

    nanogpt_inference_result = run_cli("list", "matrix", "--workload", "nanogpt-inference", "--format", "json")
    assert nanogpt_inference_result.returncode == 0, nanogpt_inference_result.stdout + nanogpt_inference_result.stderr
    nanogpt_inference = json.loads(nanogpt_inference_result.stdout)
    assert nanogpt_inference["workload"] == "nanogpt-inference"
    assert {workload["variant"] for workload in nanogpt_inference["workloads"]} == {
        "decode",
        "fp16-b16",
        "fp32-b16",
        "prefill",
        "speculative",
    }
    assert {workload["dataset"] for workload in nanogpt_inference["workloads"]} == {"prompt-suite-local"}

    nanogpt_prefill_result = run_cli(
        "list",
        "matrix",
        "--workload",
        "nanogpt-inference",
        "--variant",
        "prefill",
        "--format",
        "json",
    )
    assert nanogpt_prefill_result.returncode == 0, nanogpt_prefill_result.stdout + nanogpt_prefill_result.stderr
    nanogpt_prefill = json.loads(nanogpt_prefill_result.stdout)
    assert [workload["workload"] for workload in nanogpt_prefill["workloads"]] == ["nanogpt-prefill"]
    assert nanogpt_prefill["workloads"][0]["dataset"] == "prompt-suite-local"

    all_result = run_cli("list", "matrix", "--format", "json")
    assert all_result.returncode == 0, all_result.stdout + all_result.stderr
    all_data = json.loads(all_result.stdout)
    assert [workload["workload"] for workload in all_data["workloads"] if not workload["quality"]] == []

    table_result = run_cli("list", "matrix", "--suite", "slm")
    assert table_result.returncode == 0, table_result.stdout + table_result.stderr
    assert "MLPerf EDU Workload Matrix" in table_result.stdout
    assert "smollm2-chat-inference --variant baseline" in table_result.stdout


def test_default_collection_for_profile_defaults():
    assert default_collection_for(Namespace(collection=None, suite=None, workload=None, profile="min")) == "starter"
    assert default_collection_for(Namespace(collection=None, suite=None, workload=None, profile="max")) == "all"
    assert default_collection_for(Namespace(collection=None, suite=None, workload=None, profile="pro")) == "research"
    assert default_collection_for(Namespace(collection=None, suite="vision", workload=None, profile="max")) is None
    assert default_collection_for(Namespace(collection="all", suite=None, workload=None, profile="min")) == "all"


def test_report_enrichment_defaults_quality_required_from_public_contract():
    workloads = load_registry()

    report = {"workload": "nanogpt-train", "quality": {}}
    enrich_report_for_display(report, workloads)
    assert report["quality"]["quality_required"] is True
    assert report["quality"]["reference_protocol"]["profile"] == "max"
    assert report["quality"]["reference_protocol"]["seeds"] == [0, 1, 2, 3, 4]
    assert "gated" not in report["quality"]

    explicit_not_required = {"workload": "nanogpt-train", "quality": {"quality_required": False}}
    enrich_report_for_display(explicit_not_required, workloads)
    assert explicit_not_required["quality"]["quality_required"] is False
    assert "gated" not in explicit_not_required["quality"]


def test_show_workload():
    result = run_cli("show", "nanogpt-train")
    assert result.returncode == 0
    assert "Workload: nanogpt-train" in result.stdout
    assert "min, max, pro" in result.stdout
    assert "public_status" in result.stdout
    assert "score-bearing" in result.stdout
    assert "source_suite" not in result.stdout
    assert "maturity" not in result.stdout


def test_info_workload_alias():
    result = run_cli("info", "--workload", "smollm2-chat-inference")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Canonical workload" in result.stdout
    assert "quantized-int8" in result.stdout


def test_info_workload_variant_shows_resolved_slice():
    result = run_cli("info", "--workload", "smollm2-chat-inference", "--variant", "quantized-int8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Workload: slm-quantized-decode" in result.stdout
    assert "performance-bearing" in result.stdout
    assert "canonical_workload" in result.stdout
    assert "smollm2-chat-inference" in result.stdout
    assert "quantized-int8" in result.stdout


def test_info_dataset_shows_asset_dossier():
    result = run_cli("info", "--dataset", "tinyshakespeare")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dataset: tinyshakespeare" in result.stdout
    assert "public-domain-us" in result.stdout
    assert "public-ok-fetch-only" in result.stdout
    assert "expected_download_bytes" in result.stdout
    assert "5600000" in result.stdout
    assert "nanogpt-train" in result.stdout


def test_info_model_alias_shows_model_dossier():
    result = run_cli("info", "--model", "qwen3-0.6b")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Model: qwen3-0.6b" in result.stdout
    assert "Qwen/Qwen3-0.6B" in result.stdout
    assert "Apache-2.0" in result.stdout
    assert "selected_model_rationale" in result.stdout
    assert "optional model-family comparisons" in result.stdout
    assert "selection_rationale" in result.stdout
    assert "SmolLM2-135M-Instruct is the default" in result.stdout
    assert "backend_rationale" in result.stdout
    assert "slm-decode" in result.stdout
    assert "slm-quantized-decode" in result.stdout


def test_cache_list_and_verify_known_workload(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tinyshakespeare.txt").write_text("To be, or not to be:\n" * 512)

    fetch = run_cli(
        "fetch",
        "--workload",
        "nanogpt-train",
        "--profile",
        "max",
        env_extra={"MLPERF_EDU_DATA_DIR": str(data_dir)},
    )
    assert fetch.returncode == 0, fetch.stdout + fetch.stderr

    listed = run_cli(
        "cache",
        "list",
        "--workload",
        "nanogpt-train",
        env_extra={"MLPERF_EDU_DATA_DIR": str(data_dir)},
    )
    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert "present" in listed.stdout
    assert "public-domain-us" in listed.stdout
    assert "release=public-ok-fetch-only" in listed.stdout

    verified = run_cli(
        "cache",
        "verify",
        "--workload",
        "nanogpt-train",
        env_extra={"MLPERF_EDU_DATA_DIR": str(data_dir)},
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr


def test_cache_accepts_canonical_variant():
    result = run_cli("cache", "list", "--workload", "smollm2-chat-inference", "--variant", "quantized-int8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "slm-quantized-decode" in result.stdout
    assert "model" in result.stdout


def test_cache_canonical_workload_lists_all_variant_assets():
    result = run_cli("cache", "list", "--workload", "smollm2-chat-inference", "--format", "json")
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["selection"]["workload"] == "smollm2-chat-inference"
    assert payload["selection"]["workloads"] == 4
    assert {asset["workload"] for asset in payload["assets"]} == {
        "slm-decode",
        "slm-quantized-decode",
        "slm-batched-decode",
        "slm-long-context-decode",
    }


def test_cache_defaults_to_min_profile_and_accepts_max_profile():
    summary = run_cli("cache", "list")
    assert summary.returncode == 0, summary.stdout + summary.stderr
    assert "Selected 12 workload(s) for profile min." in summary.stdout
    assert "Use --profile max to inspect assets for the full suite." in summary.stdout

    min_result = run_cli("cache", "list", "--format", "json")
    assert min_result.returncode == 0, min_result.stdout + min_result.stderr
    min_payload = json.loads(min_result.stdout)
    assert min_payload["selection"]["profile"] == "min"
    assert min_payload["selection"]["workloads"] == 12
    min_workloads = {asset["workload"] for asset in min_payload["assets"]}
    assert len(min_workloads) == 12
    assert "slm-decode" in min_workloads
    assert "slm-long-context-decode" not in min_workloads

    max_result = run_cli("cache", "list", "--profile", "max", "--format", "json")
    assert max_result.returncode == 0, max_result.stdout + max_result.stderr
    max_payload = json.loads(max_result.stdout)
    assert max_payload["selection"]["profile"] == "max"
    assert max_payload["selection"]["workloads"] == 30
    max_workloads = {asset["workload"] for asset in max_payload["assets"]}
    assert len(max_workloads) == 30
    assert "slm-long-context-decode" in max_workloads


def test_audit_summary_passes():
    result = run_cli("audit")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MLPerf EDU Public Contract Audit" in result.stdout
    assert "public contract audit: passed" in result.stdout
    assert "score-bearing=5" in result.stdout
    assert "performance-bearing=4" in result.stdout
    assert "systems-only=21" in result.stdout
    assert "public warnings: 0" in result.stdout


def test_audit_json_passes_and_filters_by_suite():
    result = run_cli("audit", "--suite", "slm", "--format", "json")
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)

    assert data["schema"] == "mlperf-edu-public-contract-audit/0.1"
    assert data["status"] == "passed"
    assert data["policy"] == "development"
    assert data["suite"] == "slm"
    assert data["counts"] == {"performance-bearing": 2, "systems-only": 2}
    assert data["blocker_count"] == 0
    assert data["warning_count"] == 0
    assert data["issues"] == []
    assert data["warnings"] == []
    assert len(data["workloads"]) == 4
    assert all(not workload["issues"] for workload in data["workloads"])
    assert all(not workload["warnings"] for workload in data["workloads"])
    slm = next(workload for workload in data["workloads"] if workload["id"] == "slm-decode")
    assert slm["canonical_workload"] == "smollm2-chat-inference"
    assert slm["variant"] == "baseline"
    assert slm["run_selector"] == "smollm2-chat-inference --variant baseline"


def test_audit_profile_filters_default_selection():
    result = run_cli("audit", "--profile", "min", "--format", "json")
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["profile"] == "min"
    assert len(data["workloads"]) == 12
    assert {workload["id"] for workload in data["workloads"]} == {
        "anomaly-ae-train",
        "micro-dlrm-distributed",
        "micro-dlrm-train",
        "micro-gnn-train",
        "micro-lstm-train",
        "micro-rl-train",
        "nano-rag-agent",
        "nanogpt-decode",
        "nanogpt-prefill",
        "nanogpt-train",
        "resnet18-train",
        "slm-decode",
    }


def test_audit_json_suppresses_public_asset_warnings_by_default():
    result = run_cli("audit", "--status", "score-bearing", "--format", "json")
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)

    assert data["status"] == "passed"
    assert data["policy"] == "development"
    assert data["warning_count"] == 0
    assert data["warnings"] == []
    nanogpt = next(workload for workload in data["workloads"] if workload["id"] == "nanogpt-train")
    assert not nanogpt["issues"]
    assert nanogpt["warnings"] == []
    dlrm = next(workload for workload in data["workloads"] if workload["id"] == "micro-dlrm-train")
    assert dlrm["warnings"] == []
    anomaly = next(workload for workload in data["workloads"] if workload["id"] == "anomaly-ae-train")
    assert anomaly["warnings"] == []


def test_audit_public_policy_fails_on_unresolved_warnings():
    result = run_cli("audit", "--status", "score-bearing", "--policy", "public", "--format", "json")
    assert result.returncode == 1, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["status"] == "failed"
    assert data["policy"] == "public"
    assert data["blocker_count"] == 0
    assert data["warning_blocked"] is True
    assert data["warning_count"] > 0
    assert len(data["warnings"]) == data["warning_count"]


def test_audit_filters_by_canonical_workload_and_variant():
    canonical = run_cli("audit", "--workload", "smollm2-chat-inference", "--format", "json")
    assert canonical.returncode == 0, canonical.stdout + canonical.stderr
    canonical_data = json.loads(canonical.stdout)
    assert canonical_data["workload"] == "smollm2-chat-inference"
    assert canonical_data["counts"] == {"performance-bearing": 2, "systems-only": 2}
    assert {workload["id"] for workload in canonical_data["workloads"]} == {
        "slm-decode",
        "slm-quantized-decode",
        "slm-batched-decode",
        "slm-long-context-decode",
    }
    assert {workload["run_selector"] for workload in canonical_data["workloads"]} == {
        "smollm2-chat-inference --variant baseline",
        "smollm2-chat-inference --variant quantized-int8",
        "smollm2-chat-inference --variant batched-b4",
        "smollm2-chat-inference --variant long-context",
    }

    variant = run_cli(
        "audit",
        "--workload",
        "smollm2-chat-inference",
        "--variant",
        "quantized-int8",
        "--format",
        "json",
    )
    assert variant.returncode == 0, variant.stdout + variant.stderr
    variant_data = json.loads(variant.stdout)
    assert variant_data["variant"] == "quantized-int8"
    assert [workload["id"] for workload in variant_data["workloads"]] == ["slm-quantized-decode"]
    assert variant_data["workloads"][0]["run_selector"] == "smollm2-chat-inference --variant quantized-int8"


def test_fetch_workload_dry_run():
    result = run_cli("fetch", "--workload", "nanogpt-train", "--profile", "min", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "Selected 1 workload(s) for profile min (nanogpt-train)." in result.stdout
    assert "Would fetch 1 workload" in result.stdout
    assert "nanogpt-train: tinyshakespeare" in result.stdout
    assert "license=Public domain in the United States" in result.stdout
    assert "terms=public-domain-us" in result.stdout
    assert "release=public-ok-fetch-only" in result.stdout


def test_fetch_min_profile_explains_shared_checkpoints():
    result = run_cli("fetch", "--profile", "min", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Selected 12 workload(s) for profile min (default)." in result.stdout
    assert "nanogpt-prefill: shared checkpoint -> nanogpt-train" in result.stdout
    assert "nanogpt-decode: shared checkpoint -> nanogpt-train" in result.stdout
    assert "source=MLPerf EDU training workload" in result.stdout
    assert "prompt_fixture=prompt-suite-local" in result.stdout


def test_fetch_slm_model_dry_run_with_alias():
    result = run_cli(
        "fetch",
        "--workload",
        "slm-decode",
        "--profile",
        "max",
        "--model",
        "qwen3-0.6b",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert "Would fetch 1 workload" in result.stdout
    assert "huggingface model -> Qwen/Qwen3-0.6B" in result.stdout
    assert "license=Apache-2.0" in result.stdout
    assert "release=public-ok-with-attribution" in result.stdout


def test_fetch_canonical_variant_dry_run():
    result = run_cli(
        "fetch",
        "--workload",
        "smollm2-chat-inference",
        "--variant",
        "quantized-int8",
        "--profile",
        "max",
        "--model",
        "qwen3-0.6b",
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Would fetch 1 workload" in result.stdout
    assert "slm-quantized-decode" in result.stdout
    assert "huggingface model -> Qwen/Qwen3-0.6B" in result.stdout
    assert "release=public-ok-with-attribution" in result.stdout


def test_fetch_canonical_workload_expands_variants_without_running_them():
    result = run_cli(
        "fetch",
        "--workload",
        "smollm2-chat-inference",
        "--profile",
        "max",
        "--model",
        "qwen3-0.6b",
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Selected 4 workload(s) for profile max (smollm2-chat-inference)." in result.stdout
    assert "Would fetch 4 workload" in result.stdout
    assert "slm-decode" in result.stdout
    assert "slm-quantized-decode" in result.stdout
    assert "slm-batched-decode" in result.stdout
    assert "slm-long-context-decode" in result.stdout
    assert "huggingface model -> Qwen/Qwen3-0.6B" in result.stdout
    assert "release=public-ok-with-attribution" in result.stdout


def test_run_dry_run_previews_selection_without_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "nanogpt-inference",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Selected 5 workload(s) for profile min (nanogpt-inference)." in result.stdout
    assert "nanogpt-decode-spec | run as: nanogpt-inference --variant speculative" in result.stdout
    assert "dry-run complete" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_fetch_workload_materializes_existing_tinyshakespeare(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tinyshakespeare.txt").write_text("To be, or not to be:\n" * 512)

    result = run_cli(
        "fetch",
        "--workload",
        "nanogpt-train",
        "--profile",
        "max",
        env_extra={"MLPERF_EDU_DATA_DIR": str(data_dir)},
    )
    assert result.returncode == 0, result.stderr
    assert (data_dir / "tinyshakespeare_train.txt").is_file()
    assert (data_dir / "tinyshakespeare_val.txt").is_file()


def test_fetch_workload_materializes_existing_movielens(tmp_path):
    data_dir = tmp_path / "data"
    write_tiny_movielens(data_dir)

    result = run_cli(
        "fetch",
        "--workload",
        "micro-dlrm-train",
        "--profile",
        "max",
        env_extra={"MLPERF_EDU_DATA_DIR": str(data_dir)},
    )
    assert result.returncode == 0, result.stderr
    assert "movielens-100k" in result.stdout


def test_std_profile_is_not_a_public_alias(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "nano-rag-agent",
        "--profile",
        "std",
        "--output-dir",
        str(tmp_path),
        env_extra={"MLPERF_EDU_RAG_MAX_PASSAGES": "8"},
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
    assert "mlperf fetch --profile min --dry-run" in result.stdout
    assert f"mlperf report {output_dir.resolve()} --format html --open" in result.stdout
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
    assert len(data["workloads"]) == 12


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


def test_validate_release_dry_run_includes_all_min_and_max(tmp_path):
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


def test_validate_smoke_runs_starter_grades_and_summarizes(tmp_path):
    result = run_cli(
        "validate",
        "smoke",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validation summary: all checks passed" in result.stdout

    output_dir = tmp_path / "min-default"
    grade_path = output_dir / "grade.json"
    assert grade_path.is_file()
    summary = json.loads(grade_path.read_text())
    assert summary["passed"] == 12
    assert summary["failed"] == 0
    assert summary["warning_count"] == 0
    assert list(output_dir.glob("mlperf_edu_min_*.json"))
    assert list(output_dir.glob("mlperf_edu_min_*.html"))
    assert list(output_dir.glob("mlperf_edu_min_*.csv"))

    validation_reports = list(tmp_path.glob("mlperf_validate_smoke_*.json"))
    validation_html = list(tmp_path.glob("mlperf_validate_smoke_*.html"))
    validation_csv = list(tmp_path.glob("mlperf_validate_smoke_*.csv"))
    validation_workload_csv = list(tmp_path.glob("mlperf_validate_workloads_smoke_*.csv"))
    assert len(validation_reports) == 1
    assert len(validation_html) == 1
    assert len(validation_csv) == 1
    assert len(validation_workload_csv) == 1
    validation_report = json.loads(validation_reports[0].read_text())
    assert validation_report["schema"] == "mlperf-edu-validation/0.1"
    assert validation_report["preset"] == "smoke"
    assert validation_report["status"] == "passed"
    assert validation_report["totals"]["passed_manifests"] == 12
    assert validation_report["totals"]["failed_manifests"] == 0
    assert validation_report["totals"]["warning_count"] == 0
    assert validation_report["totals"]["workloads"] == 12
    assert validation_report["totals"]["validations"] == 1
    assert {row["workload"] for row in validation_report["workloads"]} == {
        "anomaly-ae-train",
        "micro-dlrm-distributed",
        "micro-dlrm-train",
        "micro-gnn-train",
        "micro-lstm-train",
        "micro-rl-train",
        "nano-rag-agent",
        "nanogpt-decode",
        "nanogpt-prefill",
        "nanogpt-train",
        "resnet18-train",
        "slm-decode",
    }
    assert validation_report["duration_seconds"] > 0
    assert validation_report["validations"][0]["report_html"].endswith(".html")
    assert validation_report["validations"][0]["report_csv"].endswith(".csv")
    assert validation_report["validations"][0]["warning_count"] == 0
    assert validation_report["validations"][0]["duration_seconds"] > 0

    html = validation_html[0].read_text()
    assert "Duration" in html
    assert "Warnings" in html
    assert "Workload Breakdown" in html
    assert "Quality Required" in html
    assert "HTML</a>" in html
    assert "Grade</a>" in html

    with validation_workload_csv[0].open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert {row["validation"] for row in rows} == {"min-default"}
    train_row = next(row for row in rows if row["workload"] == "nanogpt-train")
    assert train_row["dataset"] == "tinyshakespeare"
    assert train_row["dataset_license_status"] == "public-domain-us"
    assert train_row["dataset_public_release_status"] == "public-ok-fetch-only"
    assert train_row["reference_runs"] == "5"
    assert train_row["reference_statistic"] == "median"
    assert "profile=max" in train_row["reference_protocol"]
    assert train_row["quality_required"] == "False"
    prefill_row = next(row for row in rows if row["workload"] == "nanogpt-prefill")
    assert prefill_row["dataset"] == "prompt-suite-local"
    assert prefill_row["dataset_license_status"] == "bundled-project-asset"
    assert prefill_row["dataset_public_release_status"] == "public-ok-bundled"
    assert prefill_row["shared_checkpoint"] == "nanogpt-train"
    assert prefill_row["quality_dependency"] == "nanogpt-train"

    with validation_csv[0].open(newline="") as f:
        validation_rows = list(csv.DictReader(f))
    assert validation_rows[0]["warning_count"] == "0"

    validation_summary = run_cli("report", str(tmp_path))
    assert validation_summary.returncode == 0, validation_summary.stdout + validation_summary.stderr
    assert "mlperf_suite: mlperf-edu" in validation_summary.stdout
    assert "workloads: 12" in validation_summary.stdout


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


def test_min_run_writes_report(tmp_path):
    result = run_cli(
        "run",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "Selected 12 workload(s) for profile min (default)." in result.stdout
    assert "Suite coverage: agent=1, distributed=1, graph=1, language=3" in result.stdout
    assert "min run complete" in result.stdout
    assert "HTML:" in result.stdout
    assert "CSV:" in result.stdout

    reports = list(tmp_path.glob("mlperf_edu_min_*.json"))
    csv_reports = list(tmp_path.glob("mlperf_edu_min_*.csv"))
    html_reports = list(tmp_path.glob("mlperf_edu_min_*.html"))
    assert len(reports) == 1
    assert len(csv_reports) == 1
    assert len(html_reports) == 1

    data = json.loads(reports[0].read_text())
    assert data["mlperf_suite"] == "mlperf-edu"
    assert data["profile"] == "min"
    assert "set" not in data
    assert data["selection"] == {"kind": "default", "name": "default"}
    statuses = {w["id"]: w["status"] for w in data["workloads"]}
    assert statuses["nanogpt-train"] == "passed"
    assert statuses["nanogpt-prefill"] == "passed"
    assert statuses["nanogpt-decode"] == "passed"
    assert statuses["slm-decode"] == "passed"
    assert statuses["micro-dlrm-train"] == "passed"
    assert statuses["resnet18-train"] == "passed"
    assert statuses["anomaly-ae-train"] == "passed"
    assert statuses["nano-rag-agent"] == "passed"
    assert statuses["micro-dlrm-distributed"] == "passed"
    assert statuses["micro-gnn-train"] == "passed"
    assert statuses["micro-lstm-train"] == "passed"
    assert statuses["micro-rl-train"] == "passed"
    assert set(statuses.values()) == {"passed"}

    with csv_reports[0].open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert {row["workload"] for row in rows} == set(statuses)
    assert {row["status"] for row in rows} == {"passed"}

    html = html_reports[0].read_text()
    assert "MLPerf EDU Default Report: min" in html
    assert "nanogpt-train" in html
    assert "anomaly-ae-train" in html

    directory_summary = run_cli("report", str(tmp_path))
    assert directory_summary.returncode == 0, directory_summary.stdout + directory_summary.stderr
    assert "workloads: 12" in directory_summary.stdout

    directory_html = tmp_path / "latest.html"
    directory_html_result = run_cli("report", str(tmp_path), "--format", "html", "--output", str(directory_html))
    assert directory_html_result.returncode == 0, directory_html_result.stdout + directory_html_result.stderr
    assert directory_html.is_file()
    assert "MLPerf EDU Default Report: min" in directory_html.read_text()


def test_run_with_power_writes_aggregate_power_report(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "nanogpt-train",
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
        "nanogpt-train",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "nanogpt-train_min_report.json"
    manifest_path = tmp_path / "nanogpt-train_min.provd.json"
    assert report_path.with_suffix(".html").is_file()
    assert report_path.with_suffix(".csv").is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["dataset"] == "tinyshakespeare"
    assert report["dataset_asset"]["id"] == "tinyshakespeare"
    assert report["dataset_asset"]["license_status"] == "public-domain-us"
    assert report["dataset_asset"]["public_release_status"] == "public-ok-fetch-only"
    assert report["dataset_asset"]["public_result_use"].startswith("score-bearing")
    assert report["quality"]["reference_protocol"]["profile"] == "max"
    assert report["quality"]["reference_protocol"]["seeds"] == [0, 1, 2, 3, 4]
    assert report["quality"]["quality_required"] is False
    assert report["run_fingerprint"]["schema"] == "mlperf-edu-run-fingerprint/0.1"
    assert report["run_fingerprint"]["hardware"]["fingerprint_hash"]
    assert report["run_fingerprint"]["software"]["python"]
    assert report["run_fingerprint"]["execution"]["workload"] == "nanogpt-train"
    assert report["run_fingerprint"]["execution"]["profile"] == "min"
    assert report["run_fingerprint"]["execution"]["data_modes"] == ["synthetic-deterministic"]

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr

    json_path = tmp_path / "manual.json"
    csv_path = tmp_path / "manual.csv"
    html_path = tmp_path / "manual.html"

    json_result = run_cli("report", str(report_path), "--format", "json", "--output", str(json_path))
    assert json_result.returncode == 0, json_result.stdout + json_result.stderr
    manual_json = json.loads(json_path.read_text())
    assert manual_json["workload"] == "nanogpt-train"
    assert manual_json["run_fingerprint"]["execution"]["workload"] == "nanogpt-train"

    csv_result = run_cli("report", str(report_path), "--format", "csv", "--output", str(csv_path))
    assert csv_result.returncode == 0, csv_result.stdout + csv_result.stderr
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["workload"] == "nanogpt-train"
    assert rows[0]["suite"] == "language"
    assert rows[0]["profile"] == "min"
    assert rows[0]["status"] == "passed"
    assert rows[0]["backend"] == "pytorch-cpu"
    assert rows[0]["data_mode"] == "synthetic-deterministic"
    assert rows[0]["dataset"] == "tinyshakespeare"
    assert rows[0]["dataset_license_status"] == "public-domain-us"
    assert rows[0]["dataset_public_release_status"] == "public-ok-fetch-only"
    assert rows[0]["dataset_public_use"].startswith("score-bearing")
    assert rows[0]["dataset_release_next_step"].startswith("Keep generated-corpus recipe")
    assert rows[0]["metric"] == "loss"
    assert rows[0]["target"] == "2.3"
    assert rows[0]["target_basis"] == "reference_runs"
    assert rows[0]["reference_runs"] == "5"
    assert rows[0]["reference_statistic"] == "median"
    assert "profile=max" in rows[0]["reference_protocol"]
    assert rows[0]["direction"] == "lower"
    assert rows[0]["quality_required"] == "False"
    assert "gated" not in rows[0]
    assert float(rows[0]["value"]) > 0
    assert float(rows[0]["duration_seconds"]) >= 0
    assert float(rows[0]["throughput"]) > 0

    html_result = run_cli("report", str(report_path), "--format", "html", "--output", str(html_path))
    assert html_result.returncode == 0, html_result.stdout + html_result.stderr
    html = html_path.read_text()
    assert "MLPerf EDU Report: nanogpt-train" in html
    assert "loss" in html
    assert "reference_runs" in html
    assert "Reference Protocol" in html
    assert "profile=max" in html
    assert "fingerprint_hash" in html
    assert "Assets and Provenance" in html
    assert "Quality Required" in html
    assert "Gated" not in html
    assert "public-domain-us" in html
    assert "public-ok-fetch-only" in html
    assert "passed" in html


def test_package_and_grade_verified_manifest(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "nanogpt-train",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    manifest_path = tmp_path / "nanogpt-train_min.provd.json"
    report_path = tmp_path / "nanogpt-train_min_report.json"
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
    assert "package_index.json" in names
    assert f"manifest/{manifest_path.name}" in names
    assert f"report/{report_path.name}" in names
    assert f"report/{report_path.with_suffix('.html').name}" in names
    assert f"report/{report_path.with_suffix('.csv').name}" in names
    assert index["schema"] == "mlperf-edu-package/0.1"
    assert index["workload"] == "nanogpt-train"
    assert all(check["ok"] for check in index["verification"])

    grade = run_cli("grade", str(tmp_path), "--output", str(grade_path))
    assert grade.returncode == 0, grade.stdout + grade.stderr
    assert "Grade summary: 1 passed, 0 failed" in grade.stdout
    summary = json.loads(grade_path.read_text())
    assert summary["schema"] == "mlperf-edu-grade/0.1"
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["warning_count"] == 0
    assert summary["results"][0]["workload"] == "nanogpt-train"
    assert summary["results"][0]["verified"] is True
    assert summary["results"][0]["quality_required"] is False
    assert "gated" not in summary["results"][0]
    assert summary["results"][0]["target_met"] == ""
    assert summary["results"][0]["warning_count"] == 0
    assert summary["results"][0]["warnings"] == []


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
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")

    grade = run_cli("grade", str(tmp_path), "--output", str(tmp_path / "grade.json"))
    assert grade.returncode == 1, grade.stdout + grade.stderr
    summary = json.loads((tmp_path / "grade.json").read_text())
    assert summary["passed"] == 0
    assert summary["failed"] == 1
    assert summary["results"][0]["quality_required"] is True
    assert "gated" not in summary["results"][0]
    assert summary["results"][0]["target_met"] is False


def test_slm_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "slm-decode",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
        env_extra={
            "MLPERF_EDU_DEVICE": "cpu",
            "MLPERF_EDU_SLM_DECODE_TOKENS": "4",
            "MLPERF_EDU_SLM_TARGET_TOKENS": "4",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Selected 1 workload(s) for profile min (slm-decode)." in result.stdout
    assert "slm-decode | run as: smollm2-chat-inference --variant baseline | suite: slm" in result.stdout

    report_path = tmp_path / "slm-decode_min_report.json"
    manifest_path = tmp_path / "slm-decode_min.provd.json"
    metadata_path = tmp_path / "slm-decode_min_model.json"
    assert report_path.is_file()
    assert manifest_path.is_file()
    assert metadata_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["workload"] == "slm-decode"
    assert report["canonical_workload"] == "smollm2-chat-inference"
    assert report["variant"] == "baseline"
    assert report["run_selector"] == "smollm2-chat-inference --variant baseline"
    assert report["suite"] == "slm"
    assert report["profile"] == "min"
    assert report["status"] == "passed"
    assert report["backend"] == "transformers-cpu"
    assert report["model"]["id"] == "transformers:gpt2-tiny-random-local"
    assert report["model_asset"]["selected_model_rationale"].startswith("Default local SLM")
    assert "SmolLM2-135M-Instruct is the default" in report["model_asset"]["selection_rationale"]
    assert "135M parameters" in report["model_asset"]["size_rationale"]
    assert "Transformers/PyTorch" in report["model_asset"]["backend_rationale"]
    assert report["metrics"]["generated_tokens"] == 4
    assert report["metrics"]["requested_decode_tokens"] == 4
    assert report["metrics"]["target_generated_tokens"] == 4
    assert report["quality"]["target"] == 4
    assert report["quality"]["direction"] == "higher"
    assert report["quality"]["override"] is True
    assert "Functional serving check" in report["quality"]["note"]
    assert report["metrics"]["output_tokens_per_sec"] > 0
    assert report["metrics"]["time_to_first_token_s"] > 0
    assert report["metrics"]["inter_token_latency_s"] > 0
    assert report["metrics"]["prefill_tokens_per_sec"] > 0
    html = report_path.with_suffix(".html").read_text()
    assert "smollm2-chat-inference --variant baseline" in html
    assert "Serving Metrics" in html
    assert "generated_tokens" in html
    assert "time_to_first_token_s" in html
    assert "inter_token_latency_s" in html
    assert "prefill_tokens_per_sec" in html
    assert "Model Rationale" in html
    assert "Default local SLM" in html

    with report_path.with_suffix(".csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["model_rationale"].startswith("Default local SLM")

    summary = run_cli("report", str(report_path))
    assert summary.returncode == 0, summary.stdout + summary.stderr
    assert "time_to_first_token_s:" in summary.stdout

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_slm_max_default_decode_budget_has_functional_margin(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "slm-decode",
        "--profile",
        "max",
        "--output-dir",
        str(tmp_path),
        env_extra={
            "MLPERF_EDU_DEVICE": "cpu",
            "MLPERF_EDU_SLM_TINY": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report = json.loads((tmp_path / "slm-decode_max_report.json").read_text())
    assert report["status"] == "passed"
    assert report["metrics"]["requested_decode_tokens"] == 16
    assert report["metrics"]["generated_tokens"] == 16
    assert report["metrics"]["target_generated_tokens"] == 8
    assert report["quality"]["target"] == 8
    assert report["quality"]["direction"] == "higher"
    assert report["quality"]["override"] is False


def test_canonical_variant_run_resolves_to_current_registry_slice(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "smollm2-chat-inference",
        "--variant",
        "quantized-int8",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
        env_extra={
            "MLPERF_EDU_DEVICE": "cpu",
            "MLPERF_EDU_SLM_DECODE_TOKENS": "4",
            "MLPERF_EDU_SLM_TARGET_TOKENS": "4",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Selected 1 workload(s) for profile min (smollm2-chat-inference:quantized-int8)." in result.stdout
    assert "slm-quantized-decode | run as: smollm2-chat-inference --variant quantized-int8 | suite: slm" in result.stdout

    aggregate = next(tmp_path.glob("mlperf_edu_min_*.json"))
    data = json.loads(aggregate.read_text())
    assert data["workload"] == "smollm2-chat-inference"
    assert data["variant"] == "quantized-int8"
    assert data["selection"] == {"kind": "workload", "name": "smollm2-chat-inference:quantized-int8"}
    assert [item["workload"] for item in data["workloads"]] == ["slm-quantized-decode"]
    assert (tmp_path / "slm-quantized-decode_min_report.html").is_file()
    assert (tmp_path / "slm-quantized-decode_min_report.csv").is_file()

    grade = run_cli("grade", str(tmp_path), "--output", str(tmp_path / "grade.json"))
    assert grade.returncode == 0, grade.stdout + grade.stderr
    grade_data = json.loads((tmp_path / "grade.json").read_text())
    assert grade_data["results"][0]["canonical_workload"] == "smollm2-chat-inference"
    assert grade_data["results"][0]["variant"] == "quantized-int8"
    assert grade_data["results"][0]["run_selector"] == "smollm2-chat-inference --variant quantized-int8"


def test_canonical_workload_runs_all_variants_by_default(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "nanogpt-inference",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Selected 5 workload(s) for profile min (nanogpt-inference)." in result.stdout

    aggregate = next(tmp_path.glob("mlperf_edu_min_*.json"))
    data = json.loads(aggregate.read_text())
    assert data["workload"] == "nanogpt-inference"
    assert [item["workload"] for item in data["workloads"]] == [
        "nanogpt-decode-fp32-b16",
        "nanogpt-decode-fp16-b16",
        "nanogpt-decode-spec",
        "nanogpt-prefill",
        "nanogpt-decode",
    ]


def test_slm_max_tiny_mode_and_suite_reports(tmp_path):
    result = run_cli(
        "run",
        "--suite",
        "slm",
        "--profile",
        "max",
        "--output-dir",
        str(tmp_path),
        env_extra={
            "MLPERF_EDU_DEVICE": "cpu",
            "MLPERF_EDU_SLM_TINY": "1",
            "MLPERF_EDU_SLM_DECODE_TOKENS": "4",
            "MLPERF_EDU_SLM_TARGET_TOKENS": "4",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr

    aggregate = next(tmp_path.glob("mlperf_edu_max_*.json"))
    assert aggregate.with_suffix(".html").is_file()
    assert aggregate.with_suffix(".csv").is_file()
    aggregate_html = aggregate.with_suffix(".html").read_text()
    assert "Serving Metrics" in aggregate_html
    assert "slm-long-context-decode" in aggregate_html
    assert "total_context_tokens" in aggregate_html
    assert "requests_per_sec" in aggregate_html

    data = json.loads(aggregate.read_text())
    assert data["suite"] == "slm"
    assert [item["workload"] for item in data["workloads"]] == [
        "slm-decode",
        "slm-quantized-decode",
        "slm-batched-decode",
        "slm-long-context-decode",
    ]
    assert {item["status"] for item in data["workloads"]} == {"passed"}
    assert {item["metrics"]["generated_tokens"] for item in data["workloads"]} == {4}
    for item in data["workloads"]:
        metrics = item["metrics"]
        assert metrics["time_to_first_token_s"] > 0
        assert metrics["inter_token_latency_s"] > 0
        assert metrics["prefill_tokens_per_sec"] > 0
        assert metrics["total_context_tokens"] >= metrics["context_tokens"]
    batched = next(item for item in data["workloads"] if item["workload"] == "slm-batched-decode")
    assert batched["metrics"]["batch_size"] == 4
    assert batched["metrics"]["total_generated_tokens"] == 16
    assert batched["run_selector"] == "smollm2-chat-inference --variant batched-b4"
    long_context = next(item for item in data["workloads"] if item["workload"] == "slm-long-context-decode")
    assert long_context["metrics"]["configured_context_tokens"] == 96
    assert long_context["metrics"]["context_tokens"] == 96
    assert long_context["data_mode"] == "synthetic-tokenized-long-context"
    assert long_context["run_selector"] == "smollm2-chat-inference --variant long-context"
    quantized = next(item for item in data["workloads"] if item["workload"] == "slm-quantized-decode")
    assert quantized["backend"] == "transformers-cpu-dynamic-int8"
    assert quantized["metrics"]["model_state_bytes"] > 0

    summary = run_cli("report", str(tmp_path / "slm-decode_max_report.json"))
    assert summary.returncode == 0, summary.stdout + summary.stderr
    assert "generated_tokens: 4" in summary.stdout

    verify = run_cli("verify", str(tmp_path / "slm-decode_max.provd.json"))
    assert verify.returncode == 0, verify.stdout + verify.stderr
    quantized_verify = run_cli("verify", str(tmp_path / "slm-quantized-decode_max.provd.json"))
    assert quantized_verify.returncode == 0, quantized_verify.stdout + quantized_verify.stderr


def test_vision_min_suite_runs_domain_workloads(tmp_path):
    result = run_cli(
        "run",
        "--suite",
        "vision",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    aggregate = next(tmp_path.glob("mlperf_edu_min_*.json"))
    data = json.loads(aggregate.read_text())
    assert data["suite"] == "vision"
    assert [item["workload"] for item in data["workloads"]] == [
        "resnet18-train",
        "mobilenetv2-train",
        "mobilenet-cifar100-composed-fp16",
        "micro-diffusion-train",
    ]
    assert {item["status"] for item in data["workloads"]} == {"passed"}

    mobilenet_report = json.loads((tmp_path / "mobilenetv2-train_min_report.json").read_text())
    assert mobilenet_report["metrics"]["logits_shape"] == [2, 100]

    composed_report = json.loads((tmp_path / "mobilenet-cifar100-composed-fp16_min_report.json").read_text())
    assert composed_report["metrics"]["effective_compression_ratio"] > 1.0
    assert composed_report["metrics"]["sparsity_actual"] > 0.0

    grade = run_cli("grade", str(tmp_path), "--output", str(tmp_path / "grade.json"))
    assert grade.returncode == 0, grade.stdout + grade.stderr
    grade_data = json.loads((tmp_path / "grade.json").read_text())
    grade_metrics = {row["workload"]: row["metric"] for row in grade_data["results"]}
    assert grade_metrics["mobilenet-cifar100-composed-fp16"] == "effective_compression_ratio"

    for manifest in (
        tmp_path / "resnet18-train_min.provd.json",
        tmp_path / "mobilenetv2-train_min.provd.json",
        tmp_path / "mobilenet-cifar100-composed-fp16_min.provd.json",
        tmp_path / "micro-diffusion-train_min.provd.json",
    ):
        verify = run_cli("verify", str(manifest))
        assert verify.returncode == 0, verify.stdout + verify.stderr


def test_nanogpt_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "nanogpt-train",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr

    report_path = tmp_path / "nanogpt-train_min_report.json"
    manifest_path = tmp_path / "nanogpt-train_min.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["workload"] == "nanogpt-train"
    assert report["status"] == "passed"
    assert report["backend"] == "pytorch-cpu"
    assert report["metrics"]["tokens"] == 32
    assert report["metrics"]["logits_shape"] == [2, 16, 128]

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "verified" in verify.stdout


def test_nano_moe_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "nano-moe-train",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "nano-moe-train_min_report.json"
    manifest_path = tmp_path / "nano-moe-train_min.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["workload"] == "nano-moe-train"
    assert report["suite"] == "language"
    assert report["status"] == "passed"
    assert report["metrics"]["tokens"] == 32
    assert report["metrics"]["num_experts"] == 8
    assert report["metrics"]["top_k"] == 2
    assert report["metrics"]["active_expert_fraction"] == 0.25
    assert report["metrics"]["logits_shape"] == [2, 16, 128]

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_research_workloads_run_by_workload_min(tmp_path):
    workloads = (
        "micro-bert-train",
        "micro-diffusion-train",
        "micro-gnn-train",
        "micro-lstm-train",
        "micro-rl-train",
        "nano-lora-finetune",
        "nano-moe-train",
        "nanogpt-decode-fp16-b16",
        "nanogpt-decode-fp32-b16",
        "nanogpt-decode-spec",
    )
    for workload in workloads:
        result = run_cli(
            "run",
            "--workload",
            workload,
            "--profile",
            "min",
            "--output-dir",
            str(tmp_path),
        )
        assert result.returncode == 0, result.stdout + result.stderr

    bert = json.loads((tmp_path / "micro-bert-train_min_report.json").read_text())
    diffusion = json.loads((tmp_path / "micro-diffusion-train_min_report.json").read_text())
    gnn = json.loads((tmp_path / "micro-gnn-train_min_report.json").read_text())
    lstm = json.loads((tmp_path / "micro-lstm-train_min_report.json").read_text())
    rl = json.loads((tmp_path / "micro-rl-train_min_report.json").read_text())
    lora = json.loads((tmp_path / "nano-lora-finetune_min_report.json").read_text())
    decode_fp32 = json.loads((tmp_path / "nanogpt-decode-fp32-b16_min_report.json").read_text())
    decode_fp16 = json.loads((tmp_path / "nanogpt-decode-fp16-b16_min_report.json").read_text())
    decode_spec = json.loads((tmp_path / "nanogpt-decode-spec_min_report.json").read_text())
    assert bert["metrics"]["logits_shape"] == [4, 2]
    assert diffusion["metrics"]["output_shape"] == [2, 3, 32, 32]
    assert gnn["metrics"]["logits_shape"] == [16, 3]
    assert lstm["metrics"]["prediction_shape"] == [2, 4]
    assert rl["metrics"]["rollout_steps"] > 0
    assert lora["metrics"]["base_grad_norm"] == 0.0
    assert lora["metrics"]["lora_grad_norm"] > 0.0
    assert decode_fp32["metrics"]["dtype"] == "fp32"
    assert decode_fp32["metrics"]["decode_steps"] == 4
    assert decode_fp16["metrics"]["dtype"] == "fp16"
    assert decode_spec["metrics"]["gamma"] == 2

    for manifest in (
        tmp_path / "micro-bert-train_min.provd.json",
        tmp_path / "micro-diffusion-train_min.provd.json",
        tmp_path / "micro-gnn-train_min.provd.json",
        tmp_path / "micro-lstm-train_min.provd.json",
        tmp_path / "micro-rl-train_min.provd.json",
        tmp_path / "nano-lora-finetune_min.provd.json",
        tmp_path / "nano-moe-train_min.provd.json",
        tmp_path / "nanogpt-decode-fp16-b16_min.provd.json",
        tmp_path / "nanogpt-decode-fp32-b16_min.provd.json",
        tmp_path / "nanogpt-decode-spec_min.provd.json",
    ):
        verify = run_cli("verify", str(manifest))
        assert verify.returncode == 0, verify.stdout + verify.stderr


def test_research_workloads_run_by_workload_max_and_grade(tmp_path):
    workloads = (
        "micro-bert-train",
        "micro-diffusion-train",
        "micro-gnn-train",
        "micro-lstm-train",
        "micro-rl-train",
        "nano-lora-finetune",
        "nano-moe-train",
        "nanogpt-decode-fp16-b16",
        "nanogpt-decode-fp32-b16",
        "nanogpt-decode-spec",
    )
    for workload in workloads:
        result = run_cli(
            "run",
            "--workload",
            workload,
            "--profile",
            "max",
            "--output-dir",
            str(tmp_path),
        )
        assert result.returncode == 0, result.stdout + result.stderr

        report = json.loads((tmp_path / f"{workload}_max_report.json").read_text())
        assert report["status"] == "passed"
        assert report["profile"] == "max"
        assert report["data_mode"] == "synthetic-micro-shard"
        assert report["metrics"]["max_micro_shard"] is True

    grade = run_cli("grade", str(tmp_path), "--output", str(tmp_path / "grade.json"))
    assert grade.returncode == 0, grade.stdout + grade.stderr
    summary = json.loads((tmp_path / "grade.json").read_text())
    assert summary["passed"] == 10
    assert summary["failed"] == 0
    metrics = {row["workload"]: row["metric"] for row in summary["results"]}
    assert metrics["nano-lora-finetune"] == "base_grad_norm"
    assert metrics["nanogpt-decode-spec"] == "acceptance_rate"


def test_nanogpt_max_run_writes_verifiable_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tinyshakespeare.txt").write_text("First Citizen:\nBefore we proceed any further.\n" * 512)
    output_dir = tmp_path / "out"

    env = {
        "MLPERF_EDU_DATA_DIR": str(data_dir),
        "MLPERF_EDU_DEVICE": "cpu",
        "MLPERF_EDU_MAX_MODEL_SIZE": "tiny",
        "MLPERF_EDU_MAX_BATCH_SIZE": "2",
        "MLPERF_EDU_MAX_SEQ_LEN": "16",
        "MLPERF_EDU_MAX_EPOCHS": "1",
        "MLPERF_EDU_MAX_BATCHES_PER_EPOCH": "2",
        "MLPERF_EDU_MAX_VAL_BATCHES": "1",
        "MLPERF_EDU_MAX_QUALITY_TARGET": "10.0",
    }
    result = run_cli(
        "run",
        "--workload",
        "nanogpt-train",
        "--profile",
        "max",
        "--output-dir",
        str(output_dir),
        env_extra=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = output_dir / "nanogpt-train_max_report.json"
    manifest_path = output_dir / "nanogpt-train_max.provd.json"
    checkpoint_path = output_dir / "nanogpt-train_max_checkpoint.pt"
    assert report_path.is_file()
    assert manifest_path.is_file()
    assert checkpoint_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["profile"] == "max"
    assert report["status"] == "passed"
    assert report["data_mode"] == "real"
    assert report["quality"]["quality_required"] is True
    assert "gated" not in report["quality"]
    assert report["quality"]["override"] is True
    assert report["metrics"]["tokens"] == 64

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr

    summary = run_cli("report", str(report_path))
    assert summary.returncode == 0, summary.stdout + summary.stderr
    assert "workload: nanogpt-train" in summary.stdout
    assert "target_met: True" in summary.stdout
    assert "quality_required: True" in summary.stdout

    prefill_env = {
        **env,
        "MLPERF_EDU_PREFILL_MAX_CONTEXT": "16",
        "MLPERF_EDU_PREFILL_MAX_WARMUP": "1",
        "MLPERF_EDU_PREFILL_MAX_ITER": "2",
    }
    prefill = run_cli(
        "run",
        "--workload",
        "nanogpt-prefill",
        "--profile",
        "max",
        "--output-dir",
        str(output_dir),
        env_extra=prefill_env,
    )
    assert prefill.returncode == 0, prefill.stdout + prefill.stderr
    prefill_report = json.loads((output_dir / "nanogpt-prefill_max_report.json").read_text())
    assert prefill_report["status"] == "passed"
    assert prefill_report["metrics"]["context_length"] == 16
    assert prefill_report["metrics"]["prefill_tokens_per_sec"] > 0
    prefill_verify = run_cli("verify", str(output_dir / "nanogpt-prefill_max.provd.json"))
    assert prefill_verify.returncode == 0, prefill_verify.stdout + prefill_verify.stderr

    decode_env = {
        **env,
        "MLPERF_EDU_DECODE_MAX_PREFILL_CTX": "8",
        "MLPERF_EDU_DECODE_MAX_STEPS": "4",
    }
    decode = run_cli(
        "run",
        "--workload",
        "nanogpt-decode",
        "--profile",
        "max",
        "--output-dir",
        str(output_dir),
        env_extra=decode_env,
    )
    assert decode.returncode == 0, decode.stdout + decode.stderr
    decode_report = json.loads((output_dir / "nanogpt-decode_max_report.json").read_text())
    assert decode_report["status"] == "passed"
    assert decode_report["metrics"]["prefill_ctx"] == 8
    assert decode_report["metrics"]["decode_steps"] == 4
    assert decode_report["metrics"]["output_tokens_per_sec"] > 0
    decode_verify = run_cli("verify", str(output_dir / "nanogpt-decode_max.provd.json"))
    assert decode_verify.returncode == 0, decode_verify.stdout + decode_verify.stderr


def test_nanogpt_pro_profile_aggregates_max_runner(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tinyshakespeare.txt").write_text("First Citizen:\nBefore we proceed any further.\n" * 512)
    output_dir = tmp_path / "out"

    env = {
        "MLPERF_EDU_DATA_DIR": str(data_dir),
        "MLPERF_EDU_DEVICE": "cpu",
        "MLPERF_EDU_MAX_MODEL_SIZE": "tiny",
        "MLPERF_EDU_MAX_BATCH_SIZE": "2",
        "MLPERF_EDU_MAX_SEQ_LEN": "16",
        "MLPERF_EDU_MAX_EPOCHS": "1",
        "MLPERF_EDU_MAX_BATCHES_PER_EPOCH": "2",
        "MLPERF_EDU_MAX_VAL_BATCHES": "1",
        "MLPERF_EDU_MAX_QUALITY_TARGET": "10.0",
        "MLPERF_EDU_PRO_REPETITIONS": "1",
    }
    result = run_cli(
        "run",
        "--workload",
        "nanogpt-train",
        "--profile",
        "pro",
        "--output-dir",
        str(output_dir),
        env_extra=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = output_dir / "nanogpt-train_pro_report.json"
    manifest_path = output_dir / "nanogpt-train_pro.provd.json"
    subrun_dir = output_dir / ".pro_evidence" / "nanogpt-train" / "rep1"
    assert report_path.is_file()
    assert manifest_path.is_file()
    assert (subrun_dir / "nanogpt-train_max_report.json").is_file()
    assert (subrun_dir / "nanogpt-train_max.provd.json").is_file()
    assert (output_dir / "nanogpt-train_max_checkpoint.pt").is_file()

    report = json.loads(report_path.read_text())
    assert report["profile"] == "pro"
    assert report["status"] == "passed"
    assert report["pro_policy"]["mode"] == "max-repetition"
    assert report["metrics"]["repetitions"] == 1
    assert report["metrics"]["final_val_loss_mean"] > 0
    assert report["subruns"][0]["profile"] == "max"
    assert report["subruns"][0]["report_sha256"].startswith("sha256:")
    assert report["subruns"][0]["provenance_sha256"].startswith("sha256:")

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr

    grade = run_cli("grade", str(output_dir), "--output", str(output_dir / "grade.json"))
    assert grade.returncode == 0, grade.stdout + grade.stderr
    summary = json.loads((output_dir / "grade.json").read_text())
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["results"][0]["profile"] == "pro"


def test_nanogpt_prefill_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "nanogpt-prefill",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "nanogpt-prefill_min_report.json"
    manifest_path = tmp_path / "nanogpt-prefill_min.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["workload"] == "nanogpt-prefill"
    assert report["dataset"] == "prompt-suite-local"
    assert report["dataset_asset"]["id"] == "prompt-suite-local"
    assert report["dataset_asset"]["license_status"] == "bundled-project-asset"
    assert report["shared_checkpoint"] == "nanogpt-train"
    assert report["quality_dependency"] == "nanogpt-train"
    assert report["checkpoint_provenance"]["source_workload"] == "nanogpt-train"
    assert report["checkpoint_provenance"]["source_run_selector"] == "nanogpt-train"
    assert report["checkpoint_provenance"]["source_quality_metric"] == "cross_entropy_loss"
    assert report["checkpoint_provenance"]["source_quality_target"] == 2.3
    assert report["checkpoint_provenance"]["source_reference_runs"] == 5
    assert report["status"] == "passed"
    assert report["metrics"]["phase"] == "prefill"
    assert report["metrics"]["context_length"] == 32
    assert report["metrics"]["prefill_tokens_per_sec"] > 0
    with report_path.with_suffix(".csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["dataset"] == "prompt-suite-local"
    assert rows[0]["dataset_license_status"] == "bundled-project-asset"
    assert rows[0]["shared_checkpoint"] == "nanogpt-train"
    assert rows[0]["quality_dependency"] == "nanogpt-train"
    assert rows[0]["checkpoint_source_selector"] == "nanogpt-train"
    assert rows[0]["checkpoint_source_quality"] == "cross_entropy_loss lower 2.3 basis=reference_runs"
    assert rows[0]["checkpoint_artifact_policy"].startswith("Preserve the source training report")
    html = report_path.with_suffix(".html").read_text()
    assert "nanogpt-train" in html
    assert "Quality Dependency" in html
    assert "Checkpoint Source" in html
    assert "cross_entropy_loss lower 2.3 basis=reference_runs" in html

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_nanogpt_decode_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "nanogpt-decode",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "nanogpt-decode_min_report.json"
    manifest_path = tmp_path / "nanogpt-decode_min.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["workload"] == "nanogpt-decode"
    assert report["dataset"] == "prompt-suite-local"
    assert report["dataset_asset"]["id"] == "prompt-suite-local"
    assert report["dataset_asset"]["license_status"] == "bundled-project-asset"
    assert report["shared_checkpoint"] == "nanogpt-train"
    assert report["quality_dependency"] == "nanogpt-train"
    assert report["checkpoint_provenance"]["source_workload"] == "nanogpt-train"
    assert report["checkpoint_provenance"]["source_quality_metric"] == "cross_entropy_loss"
    assert report["status"] == "passed"
    assert report["metrics"]["phase"] == "decode"
    assert report["metrics"]["prefill_ctx"] == 16
    assert report["metrics"]["decode_steps"] == 4
    assert report["metrics"]["output_tokens_per_sec"] > 0

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_micro_dlrm_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "micro-dlrm-train",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "micro-dlrm-train_min_report.json"
    manifest_path = tmp_path / "micro-dlrm-train_min.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["workload"] == "micro-dlrm-train"
    assert report["status"] == "passed"
    assert report["metrics"]["samples"] == 8
    assert report["metrics"]["output_shape"] == [8, 1]
    assert report["metrics"]["samples_per_second"] > 0

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_micro_dlrm_dram_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "micro-dlrm-dram-train",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "micro-dlrm-dram-train_min_report.json"
    manifest_path = tmp_path / "micro-dlrm-dram-train_min.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["workload"] == "micro-dlrm-dram-train"
    assert report["status"] == "passed"
    assert report["metrics"]["samples"] == 8
    assert report["metrics"]["working_set_bytes"] == 16_384 * 32 * 4
    assert report["metrics"]["output_shape"] == [8, 1]

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_recommender_min_suite_runs_cache_and_dram_workloads(tmp_path):
    result = run_cli(
        "run",
        "--suite",
        "recommender",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
        env_extra={
            "MLPERF_EDU_DDP_STEPS": "1",
            "MLPERF_EDU_DDP_MICRO_BATCH": "2",
            "MLPERF_EDU_DDP_REL_LOSS_TARGET": "10.0",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr

    aggregate = next(tmp_path.glob("mlperf_edu_min_*.json"))
    data = json.loads(aggregate.read_text())
    assert data["suite"] == "recommender"
    assert [item["workload"] for item in data["workloads"]] == [
        "micro-dlrm-train",
        "micro-dlrm-dram-train",
    ]
    assert {item["status"] for item in data["workloads"]} == {"passed"}

    for manifest in (
        tmp_path / "micro-dlrm-train_min.provd.json",
        tmp_path / "micro-dlrm-dram-train_min.provd.json",
    ):
        verify = run_cli("verify", str(manifest))
        assert verify.returncode == 0, verify.stdout + verify.stderr


def test_agent_min_suite_runs_all_agent_patterns(tmp_path):
    result = run_cli(
        "run",
        "--suite",
        "agent",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
        env_extra={
            "MLPERF_EDU_CODEGEN_MIN_RETRIES": "2",
            "MLPERF_EDU_REACT_MIN_STEPS": "2",
            "MLPERF_EDU_TOOLCALL_MIN_QUERIES": "2",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr

    aggregate = next(tmp_path.glob("mlperf_edu_min_*.json"))
    data = json.loads(aggregate.read_text())
    assert data["suite"] == "agent"
    assert [item["workload"] for item in data["workloads"]] == [
        "nano-rag-agent",
        "nano-codegen-agent",
        "nano-react-agent",
        "nano-toolcall-agent",
    ]
    assert {item["status"] for item in data["workloads"]} == {"passed"}

    codegen = json.loads((tmp_path / "nano-codegen-agent_min_report.json").read_text())
    rag = json.loads((tmp_path / "nano-rag-agent_min_report.json").read_text())
    react = json.loads((tmp_path / "nano-react-agent_min_report.json").read_text())
    toolcall = json.loads((tmp_path / "nano-toolcall-agent_min_report.json").read_text())
    assert codegen["metrics"]["iterations"] == 2
    assert rag["metrics"]["retrieve_latency_ms"] >= 0
    assert react["metrics"]["steps"] == 2
    assert toolcall["metrics"]["total_queries"] == 2
    assert toolcall["metrics"]["valid_call_rate"] >= 0

    for manifest in (
        tmp_path / "nano-codegen-agent_min.provd.json",
        tmp_path / "nano-rag-agent_min.provd.json",
        tmp_path / "nano-react-agent_min.provd.json",
        tmp_path / "nano-toolcall-agent_min.provd.json",
    ):
        verify = run_cli("verify", str(manifest))
        assert verify.returncode == 0, verify.stdout + verify.stderr


def test_agent_max_suite_runs_all_agent_patterns(tmp_path):
    result = run_cli(
        "run",
        "--suite",
        "agent",
        "--profile",
        "max",
        "--output-dir",
        str(tmp_path),
        env_extra={
            "MLPERF_EDU_RAG_MAX_PASSAGES": "64",
            "MLPERF_EDU_CODEGEN_MAX_RETRIES": "3",
            "MLPERF_EDU_REACT_MAX_STEPS": "3",
            "MLPERF_EDU_TOOLCALL_MAX_QUERIES": "3",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr

    aggregate = next(tmp_path.glob("mlperf_edu_max_*.json"))
    data = json.loads(aggregate.read_text())
    assert data["suite"] == "agent"
    assert [item["workload"] for item in data["workloads"]] == [
        "nano-rag-agent",
        "nano-codegen-agent",
        "nano-react-agent",
        "nano-toolcall-agent",
    ]
    assert {item["status"] for item in data["workloads"]} == {"passed"}

    codegen = json.loads((tmp_path / "nano-codegen-agent_max_report.json").read_text())
    rag = json.loads((tmp_path / "nano-rag-agent_max_report.json").read_text())
    react = json.loads((tmp_path / "nano-react-agent_max_report.json").read_text())
    toolcall = json.loads((tmp_path / "nano-toolcall-agent_max_report.json").read_text())
    assert codegen["metrics"]["iterations"] == 3
    assert rag["metrics"]["n_passages"] == 64
    assert react["metrics"]["steps"] == 3
    assert toolcall["metrics"]["total_queries"] == 3

    for manifest in (
        tmp_path / "nano-codegen-agent_max.provd.json",
        tmp_path / "nano-rag-agent_max.provd.json",
        tmp_path / "nano-react-agent_max.provd.json",
        tmp_path / "nano-toolcall-agent_max.provd.json",
    ):
        verify = run_cli("verify", str(manifest))
        assert verify.returncode == 0, verify.stdout + verify.stderr


def test_movielens_text_occupations_are_encoded(tmp_path):
    from reference.dataset_factory import MovieLensRecommendationDataset

    dataset_dir = tmp_path / "movielens" / "ml-100k"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "u.user").write_text(
        "1|21|M|student|00000\n"
        "2|39|F|engineer|00000\n"
    )
    with (dataset_dir / "u.item").open("w", encoding="latin-1") as f:
        for item_id in range(1, 3):
            genres = ["1" if idx == item_id % 19 else "0" for idx in range(19)]
            fields = [
                str(item_id),
                f"Movie {item_id}",
                "01-Jan-1995",
                "",
                f"http://example.com/{item_id}",
                *genres,
            ]
            f.write("|".join(fields) + "\n")
    (dataset_dir / "u.data").write_text(
        "1\t1\t5\t0\n"
        "2\t2\t2\t1\n"
    )

    dataset = MovieLensRecommendationDataset(data_dir=str(dataset_dir))

    assert set(dataset.sparse_features[2].tolist()) == {0, 1}


def test_micro_dlrm_max_run_writes_verifiable_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    write_tiny_movielens(data_dir)
    output_dir = tmp_path / "out"

    env = {
        "MLPERF_EDU_DATA_DIR": str(data_dir),
        "MLPERF_EDU_DLRM_MAX_BATCH_SIZE": "4",
        "MLPERF_EDU_DLRM_MAX_EPOCHS": "1",
        "MLPERF_EDU_DLRM_MAX_BATCHES_PER_EPOCH": "2",
        "MLPERF_EDU_DLRM_MAX_VAL_BATCHES": "1",
        "MLPERF_EDU_DLRM_MAX_ACCURACY_TARGET": "0.0",
    }
    result = run_cli(
        "run",
        "--workload",
        "micro-dlrm-train",
        "--profile",
        "max",
        "--output-dir",
        str(output_dir),
        env_extra=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = output_dir / "micro-dlrm-train_max_report.json"
    manifest_path = output_dir / "micro-dlrm-train_max.provd.json"
    checkpoint_path = output_dir / "micro-dlrm-train_max_checkpoint.pt"
    assert report_path.is_file()
    assert manifest_path.is_file()
    assert checkpoint_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["profile"] == "max"
    assert report["status"] == "passed"
    assert report["data_mode"] == "real"
    assert report["quality"]["quality_required"] is True
    assert "gated" not in report["quality"]
    assert report["quality"]["override"] is True
    assert report["quality"]["metric_key"] == "best_accuracy"
    assert report["metrics"]["samples"] == 8
    assert report["metrics"]["best_epoch"] == 1
    assert report["metrics"]["best_accuracy"] == report["metrics"]["final_accuracy"]
    assert report["metrics"]["last_epoch_accuracy"] == report["metrics"]["final_accuracy"]
    assert 0.0 <= report["metrics"]["final_accuracy"] <= 1.0

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_micro_dlrm_dram_max_run_writes_verifiable_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    write_tiny_movielens(data_dir)
    output_dir = tmp_path / "out"

    env = {
        "MLPERF_EDU_DATA_DIR": str(data_dir),
        "MLPERF_EDU_DLRM_DRAM_MAX_BATCH_SIZE": "4",
        "MLPERF_EDU_DLRM_DRAM_MAX_EPOCHS": "1",
        "MLPERF_EDU_DLRM_DRAM_MAX_BATCHES_PER_EPOCH": "2",
        "MLPERF_EDU_DLRM_DRAM_MAX_VAL_BATCHES": "1",
        "MLPERF_EDU_DLRM_DRAM_MAX_M_SPA": "16",
        "MLPERF_EDU_DLRM_DRAM_MAX_VIRTUAL_TABLE_SIZE": "4096",
        "MLPERF_EDU_DLRM_DRAM_MAX_ACCURACY_TARGET": "0.0",
    }
    result = run_cli(
        "run",
        "--workload",
        "micro-dlrm-dram-train",
        "--profile",
        "max",
        "--output-dir",
        str(output_dir),
        env_extra=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = output_dir / "micro-dlrm-dram-train_max_report.json"
    manifest_path = output_dir / "micro-dlrm-dram-train_max.provd.json"
    checkpoint_path = output_dir / "micro-dlrm-dram-train_max_checkpoint.pt"
    assert report_path.is_file()
    assert manifest_path.is_file()
    assert checkpoint_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["profile"] == "max"
    assert report["status"] == "passed"
    assert report["data_mode"] == "real"
    assert report["config"]["virtual_table_size"] == 4096
    assert report["metrics"]["working_set_bytes"] == 4096 * 16 * 4
    assert report["quality"]["override"] is True

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_micro_dlrm_distributed_max_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "micro-dlrm-distributed",
        "--profile",
        "max",
        "--output-dir",
        str(tmp_path),
        env_extra={
            "MLPERF_EDU_DDP_MAX_STEPS": "1",
            "MLPERF_EDU_DDP_MAX_MICRO_BATCH": "2",
            "MLPERF_EDU_DDP_MAX_REL_LOSS_TARGET": "10.0",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "micro-dlrm-distributed_max_report.json"
    manifest_path = tmp_path / "micro-dlrm-distributed_max.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["profile"] == "max"
    assert report["status"] == "passed"
    assert report["quality"]["quality_required"] is True
    assert "gated" not in report["quality"]
    assert report["quality"]["override"] is True
    assert report["metrics"]["relative_loss_delta"] is not None

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_resnet18_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "resnet18-train",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "resnet18-train_min_report.json"
    manifest_path = tmp_path / "resnet18-train_min.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["workload"] == "resnet18-train"
    assert report["status"] == "passed"
    assert report["metrics"]["samples"] == 2
    assert report["metrics"]["logits_shape"] == [2, 100]
    assert report["metrics"]["samples_per_second"] > 0

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_resnet18_max_run_writes_verifiable_artifacts(tmp_path):
    shard_path = tmp_path / "resnet_shard.pt"
    torch.manual_seed(42)
    torch.save(
        {
            "train_images": torch.randn(4, 3, 32, 32),
            "train_labels": torch.tensor([0, 1, 2, 3]),
            "val_images": torch.randn(4, 3, 32, 32),
            "val_labels": torch.tensor([0, 1, 2, 3]),
        },
        shard_path,
    )
    output_dir = tmp_path / "out"
    env = {
        "MLPERF_EDU_DEVICE": "cpu",
        "MLPERF_EDU_RESNET_MAX_TENSOR_PATH": str(shard_path),
        "MLPERF_EDU_RESNET_MAX_BATCH_SIZE": "2",
        "MLPERF_EDU_RESNET_MAX_EPOCHS": "1",
        "MLPERF_EDU_RESNET_MAX_BATCHES_PER_EPOCH": "1",
        "MLPERF_EDU_RESNET_MAX_VAL_BATCHES": "1",
        "MLPERF_EDU_RESNET_MAX_ACCURACY_TARGET": "0.0",
    }
    result = run_cli(
        "run",
        "--workload",
        "resnet18-train",
        "--profile",
        "max",
        "--output-dir",
        str(output_dir),
        env_extra=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = output_dir / "resnet18-train_max_report.json"
    manifest_path = output_dir / "resnet18-train_max.provd.json"
    checkpoint_path = output_dir / "resnet18-train_max_checkpoint.pt"
    assert report_path.is_file()
    assert manifest_path.is_file()
    assert checkpoint_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["profile"] == "max"
    assert report["status"] == "passed"
    assert report["data_mode"] == "local-tensor-shard"
    assert report["quality"]["quality_required"] is True
    assert "gated" not in report["quality"]
    assert report["quality"]["override"] is True
    assert report["metrics"]["samples"] == 2

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_mobilenetv2_max_run_writes_verifiable_artifacts(tmp_path):
    shard_path = tmp_path / "mobilenet_shard.pt"
    torch.manual_seed(42)
    torch.save(
        {
            "train_images": torch.randn(4, 3, 32, 32),
            "train_labels": torch.tensor([0, 1, 2, 3]),
            "val_images": torch.randn(4, 3, 32, 32),
            "val_labels": torch.tensor([0, 1, 2, 3]),
        },
        shard_path,
    )
    output_dir = tmp_path / "out"
    env = {
        "MLPERF_EDU_DEVICE": "cpu",
        "MLPERF_EDU_MOBILENET_MAX_TENSOR_PATH": str(shard_path),
        "MLPERF_EDU_MOBILENET_MAX_BATCH_SIZE": "2",
        "MLPERF_EDU_MOBILENET_MAX_EPOCHS": "1",
        "MLPERF_EDU_MOBILENET_MAX_BATCHES_PER_EPOCH": "1",
        "MLPERF_EDU_MOBILENET_MAX_VAL_BATCHES": "1",
        "MLPERF_EDU_MOBILENET_MAX_ACCURACY_TARGET": "0.0",
    }
    result = run_cli(
        "run",
        "--workload",
        "mobilenetv2-train",
        "--profile",
        "max",
        "--output-dir",
        str(output_dir),
        env_extra=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = output_dir / "mobilenetv2-train_max_report.json"
    manifest_path = output_dir / "mobilenetv2-train_max.provd.json"
    checkpoint_path = output_dir / "mobilenetv2-train_max_checkpoint.pt"
    assert report_path.is_file()
    assert manifest_path.is_file()
    assert checkpoint_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["profile"] == "max"
    assert report["status"] == "passed"
    assert report["data_mode"] == "local-tensor-shard"
    assert report["quality"]["quality_required"] is True
    assert "gated" not in report["quality"]
    assert report["quality"]["override"] is True
    assert report["metrics"]["samples"] == 2
    assert 0.0 <= report["metrics"]["final_accuracy"] <= 1.0

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_anomaly_ae_min_run_writes_verifiable_artifacts(tmp_path):
    result = run_cli(
        "run",
        "--workload",
        "anomaly-ae-train",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = tmp_path / "anomaly-ae-train_min_report.json"
    manifest_path = tmp_path / "anomaly-ae-train_min.provd.json"
    assert report_path.is_file()
    assert manifest_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["workload"] == "anomaly-ae-train"
    assert report["status"] == "passed"
    assert report["metrics"]["samples"] == 4
    assert report["metrics"]["input_shape"] == [4, 784]
    assert report["metrics"]["reconstruction_shape"] == [4, 784]
    assert report["metrics"]["samples_per_second"] > 0

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_tiny_min_suite_runs_kws_and_wake_vision(tmp_path):
    result = run_cli(
        "run",
        "--suite",
        "tiny",
        "--profile",
        "min",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    aggregate = next(tmp_path.glob("mlperf_edu_min_*.json"))
    data = json.loads(aggregate.read_text())
    assert data["suite"] == "tiny"
    assert [item["workload"] for item in data["workloads"]] == [
        "anomaly-ae-train",
        "dscnn-kws-train",
        "wake-vision-vww",
    ]
    assert {item["status"] for item in data["workloads"]} == {"passed"}

    kws = json.loads((tmp_path / "dscnn-kws-train_min_report.json").read_text())
    assert kws["metrics"]["input_shape"] == [4, 1, 40, 101]
    assert kws["metrics"]["logits_shape"] == [4, 12]

    wake = json.loads((tmp_path / "wake-vision-vww_min_report.json").read_text())
    assert wake["metrics"]["input_shape"] == [4, 1, 96, 96]
    assert wake["metrics"]["logits_shape"] == [4, 2]

    for manifest in (
        tmp_path / "anomaly-ae-train_min.provd.json",
        tmp_path / "dscnn-kws-train_min.provd.json",
        tmp_path / "wake-vision-vww_min.provd.json",
    ):
        verify = run_cli("verify", str(manifest))
        assert verify.returncode == 0, verify.stdout + verify.stderr


def test_tiny_max_suite_runs_kws_and_wake_vision(tmp_path):
    anomaly_shard = tmp_path / "anomaly_shard.pt"
    torch.manual_seed(42)
    torch.save(
        {
            "train": torch.rand(16, 784),
            "val": torch.rand(8, 784),
        },
        anomaly_shard,
    )
    result = run_cli(
        "run",
        "--suite",
        "tiny",
        "--profile",
        "max",
        "--output-dir",
        str(tmp_path),
        env_extra={
            "MLPERF_EDU_ANOMALY_MAX_TENSOR_PATH": str(anomaly_shard),
            "MLPERF_EDU_ANOMALY_MAX_BATCH_SIZE": "4",
            "MLPERF_EDU_ANOMALY_MAX_EPOCHS": "1",
            "MLPERF_EDU_ANOMALY_MAX_BATCHES_PER_EPOCH": "2",
            "MLPERF_EDU_ANOMALY_MAX_VAL_BATCHES": "1",
            "MLPERF_EDU_ANOMALY_MAX_MSE_TARGET": "1.0",
            "MLPERF_EDU_DSCNN_MAX_BATCH_SIZE": "2",
            "MLPERF_EDU_DSCNN_MAX_BATCHES": "1",
            "MLPERF_EDU_DSCNN_MAX_VAL_BATCHES": "1",
            "MLPERF_EDU_WAKE_MAX_BATCH_SIZE": "2",
            "MLPERF_EDU_WAKE_MAX_BATCHES": "1",
            "MLPERF_EDU_WAKE_MAX_VAL_BATCHES": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr

    aggregate = next(tmp_path.glob("mlperf_edu_max_*.json"))
    data = json.loads(aggregate.read_text())
    assert data["suite"] == "tiny"
    assert [item["workload"] for item in data["workloads"]] == [
        "anomaly-ae-train",
        "dscnn-kws-train",
        "wake-vision-vww",
    ]
    assert {item["status"] for item in data["workloads"]} == {"passed"}

    anomaly = json.loads((tmp_path / "anomaly-ae-train_max_report.json").read_text())
    dscnn = json.loads((tmp_path / "dscnn-kws-train_max_report.json").read_text())
    wake = json.loads((tmp_path / "wake-vision-vww_max_report.json").read_text())
    assert anomaly["data_mode"] == "local-tensor-shard"
    assert anomaly["quality"]["target_met"] is True
    assert dscnn["data_mode"] == "synthetic-microshard"
    assert dscnn["metrics"]["samples"] == 2
    assert dscnn["quality"]["quality_required"] is False
    assert "gated" not in dscnn["quality"]
    assert wake["data_mode"] == "synthetic-microshard"
    assert wake["metrics"]["samples"] == 2
    assert wake["quality"]["quality_required"] is False
    assert "gated" not in wake["quality"]

    for manifest in (
        tmp_path / "anomaly-ae-train_max.provd.json",
        tmp_path / "dscnn-kws-train_max.provd.json",
        tmp_path / "wake-vision-vww_max.provd.json",
    ):
        verify = run_cli("verify", str(manifest))
        assert verify.returncode == 0, verify.stdout + verify.stderr


def test_anomaly_ae_max_run_writes_verifiable_artifacts(tmp_path):
    shard_path = tmp_path / "anomaly_shard.pt"
    torch.manual_seed(42)
    torch.save(
        {
            "train": torch.rand(16, 784),
            "val": torch.rand(8, 784),
        },
        shard_path,
    )
    output_dir = tmp_path / "out"
    env = {
        "MLPERF_EDU_ANOMALY_MAX_TENSOR_PATH": str(shard_path),
        "MLPERF_EDU_ANOMALY_MAX_BATCH_SIZE": "4",
        "MLPERF_EDU_ANOMALY_MAX_EPOCHS": "1",
        "MLPERF_EDU_ANOMALY_MAX_BATCHES_PER_EPOCH": "2",
        "MLPERF_EDU_ANOMALY_MAX_VAL_BATCHES": "1",
        "MLPERF_EDU_ANOMALY_MAX_MSE_TARGET": "1.0",
    }
    result = run_cli(
        "run",
        "--workload",
        "anomaly-ae-train",
        "--profile",
        "max",
        "--output-dir",
        str(output_dir),
        env_extra=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = output_dir / "anomaly-ae-train_max_report.json"
    manifest_path = output_dir / "anomaly-ae-train_max.provd.json"
    checkpoint_path = output_dir / "anomaly-ae-train_max_checkpoint.pt"
    assert report_path.is_file()
    assert manifest_path.is_file()
    assert checkpoint_path.is_file()

    report = json.loads(report_path.read_text())
    assert report["profile"] == "max"
    assert report["status"] == "passed"
    assert report["data_mode"] == "local-tensor-shard"
    assert report["quality"]["quality_required"] is True
    assert "gated" not in report["quality"]
    assert report["quality"]["override"] is True
    assert report["metrics"]["samples"] == 8

    verify = run_cli("verify", str(manifest_path))
    assert verify.returncode == 0, verify.stdout + verify.stderr
