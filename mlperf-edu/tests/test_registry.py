import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from importlib import resources
from pathlib import Path

import pytest

from mlperf.assets import asset_dossier, has_asset_dossier
from mlperf.registry import (
    EDU_SCENARIOS,
    PROFILES,
    PUBLIC_RESULT_SCENARIOS,
    PUBLIC_STATUSES,
    STARTER_WORKLOAD_ORDER,
    STARTER_WORKLOADS,
    RESEARCH_WORKLOAD_ORDER,
    RESEARCH_WORKLOADS,
    STANDARD_WORKLOAD_ORDER,
    STANDARD_WORKLOADS,
    baseline_lifecycle_issues,
    load_registry,
    public_contract_report,
    select_workloads,
    validate_registry,
)
from mlperf.runners.tiny import _canonical_config_int


def test_profiles_are_min_max_pro():
    assert PROFILES == ("min", "max", "pro")


def test_registry_loads_current_workloads():
    workloads = load_registry()
    assert len(workloads) == 14
    assert STARTER_WORKLOADS.issubset(workloads)
    assert STANDARD_WORKLOADS.issubset(workloads)
    assert RESEARCH_WORKLOADS.issubset(workloads)


def test_keyword_spotting_runner_uses_canonical_timing_defaults(monkeypatch):
    workload = load_registry()["keyword-spotting"]

    assert (
        _canonical_config_int(
            workload,
            "warmup_repetitions",
            "MLPERF_EDU_KEYWORD_SPOTTING_MAX_WARMUP_REPETITIONS",
            1,
        )
        == 1000
    )
    assert (
        _canonical_config_int(
            workload,
            "repetitions",
            "MLPERF_EDU_KEYWORD_SPOTTING_MAX_REPETITIONS",
            1,
        )
        == 2000
    )

    monkeypatch.setenv("MLPERF_EDU_KEYWORD_SPOTTING_MAX_REPETITIONS", "17")
    assert (
        _canonical_config_int(
            workload,
            "repetitions",
            "MLPERF_EDU_KEYWORD_SPOTTING_MAX_REPETITIONS",
            1,
        )
        == 17
    )


def test_packaged_registry_copy_matches_flat_registry():
    repo_root = Path(__file__).resolve().parents[1]
    flat = repo_root / "workloads.yaml"
    packaged = repo_root / "src" / "mlperf_edu" / "workloads.yaml"
    assert packaged.read_text() == flat.read_text()


def test_all_workloads_declare_min_and_max_runners():
    workloads = load_registry()

    assert workloads
    for workload in workloads.values():
        runner = workload.raw.get("runner", {})
        assert "min" in runner, workload.id
        assert "max" in runner, workload.id


def test_all_workloads_declare_public_contract_metadata():
    workloads = load_registry()
    counts = Counter(workload.public_status for workload in workloads.values())

    assert set(counts).issubset(PUBLIC_STATUSES)
    assert counts == {"experimental": 14}
    assert all(workload.public_rationale for workload in workloads.values())


def test_all_execution_contracts_declare_result_roles():
    workloads = load_registry()

    for workload in workloads.values():
        assert workload.raw["canonical_max_contract"]["result_role"] == "score-bearing"
        assert (
            workload.raw["canonical_max_contract"]["mode"]
            in workload.raw["implemented_modes"]
        )
        phases = (
            (workload.raw.get("mode_contracts") or {}).get("inference") or {}
        ).get("phases", {})
        for contract in phases.values():
            assert contract["result_role"] == "performance-bearing"


def test_all_measurement_contracts_pin_outer_execution_stabilization():
    workloads = load_registry()

    for workload in workloads.values():
        protocols = [workload.raw["measurement_protocol"]]
        phases = (
            (workload.raw.get("mode_contracts") or {}).get("inference") or {}
        ).get("phases", {})
        protocols.extend(
            contract["measurement_protocol"] for contract in phases.values()
        )
        for protocol in protocols:
            # Timing repeatability was reduced from five runs to one so the
            # timing protocol matches the single-run acceptance rule. The
            # already-measured five-run records are retained as data.
            assert protocol["outer_reference_runs"] == 1
            assert isinstance(protocol["outer_preconditioning_runs"], int)
            assert protocol["outer_preconditioning_runs"] >= 0
            assert 0 <= protocol["outer_inter_execution_cooldown_seconds"] <= 300


def test_all_quality_contracts_require_one_authoritative_acceptance_run():
    workloads = load_registry()

    assert all(workload.quality_acceptance_runs == 1 for workload in workloads.values())


def test_all_quality_contracts_classify_the_target_kind():
    workloads = load_registry()
    expected_counts = {
        "inherited_acceptance_gate": 6,
        "published_reference_reproduction": 7,
        "published_mean_with_tolerance": 1,
    }
    counts = Counter(workload.quality_target_kind for workload in workloads.values())

    assert counts == expected_counts


def test_functional_stage_workloads_separate_probe_from_quality_contract():
    workloads = load_registry()
    # Reinforcement learning is absent on purpose. It used to declare
    # environment-gated-quality-conformance because the only runner was a
    # CUDA and TensorFlow 1.x container. The PyTorch adapter executes the
    # contract locally, so it no longer carries an execution_status at all.
    expected_status = {
        "code-generation": "quality-audited-target-not-met",
        "function-calling": "quality-audited-target-not-met",
        "image-generation": "quality-audited-target-not-met",
    }

    for workload_id, status in expected_status.items():
        contract = workloads[workload_id].raw["canonical_max_contract"]
        assert contract["data_mode"] != "functional-setup-probe"
        assert contract["config"]
        assert contract["functional_probe"]
        assert contract["execution_status"] == status


def test_new_quality_contracts_pin_complete_evaluation_boundaries():
    workloads = load_registry()

    code = workloads["code-generation"].raw["canonical_max_contract"]
    assert code["config"]["evaluation_tasks"] == 164
    assert code["config"]["minimum_passing_tasks"] == 94
    assert code["config"]["prompt_format"] == "qwen2.5-coder-official-chatml"
    assert code["config"]["max_new_tokens"] == 2048
    assert "HumanEval-32" in code["config"]["evaluator_reference_self_check"]
    assert len(code["model_revision"]) == 40
    assert len(code["evaluator_revision"]) == 40
    assert len(code["generation_recipe_revision"]) == 40

    functions = workloads["function-calling"].raw["canonical_max_contract"]
    assert functions["config"]["evaluation_examples"] == 1150
    assert len(functions["config"]["categories"]) == 6
    assert len(functions["model_revision"]) == 40
    assert len(functions["dataset_revision"]) == 40
    assert len(functions["evaluator_revision"]) == 40

    recommendation = workloads["recommendation"].raw["canonical_max_contract"]
    assert recommendation["split"] == "leave-one-out-999-negatives"
    # The candidate count is part of the HR@10 definition, not a tuning knob.
    assert recommendation["config"]["negatives_per_user_eval"] == 999
    assert recommendation["config"]["predictive_factors"] == 64
    assert (
        workloads["recommendation"].raw["provenance"]["authority"]
        == "MLCommons MLPerf Training v0.5 recommendation"
    )

    images = workloads["image-generation"].raw["canonical_max_contract"]
    assert images["config"]["quality_trials"] == 3
    assert images["config"]["generated_images_per_trial"] == 50_000
    assert images["config"]["total_generated_images"] == 150_000
    assert images["config"]["trial_seeds"] == [
        "0-49999",
        "50000-99999",
        "100000-149999",
    ]
    assert images["config"]["network_evaluations_per_image"] == 35

    reinforcement = workloads["reinforcement-learning"].raw["canonical_max_contract"]
    assert reinforcement["config"]["self_play_games_per_generation"] == 2_000
    assert reinforcement["config"]["playoff_games"] == 100
    assert reinforcement["quality_gates"]["model_promotion_playoff"]["target"] == 0.55


def test_public_result_workloads_use_educational_mlcommons_scenarios():
    workloads = load_registry()

    assert {workload.scenario for workload in workloads.values()}.issubset(
        EDU_SCENARIOS
    )
    for workload in workloads.values():
        if workload.public_status in {"score-bearing", "performance-bearing"}:
            assert workload.scenario in PUBLIC_RESULT_SCENARIOS, workload.id
        if workload.public_status == "score-bearing":
            assert workload.scenario == "training", workload.id
        if workload.public_status == "performance-bearing":
            assert workload.scenario in {"single_stream", "offline", "server"}, (
                workload.id
            )


def test_candidates_do_not_claim_reference_evidence_before_admission():
    workloads = load_registry()
    selected = [
        workload
        for workload in workloads.values()
        if workload.public_status in {"score-bearing", "performance-bearing"}
    ]

    assert selected == []
    assert all(
        "verified_baseline" not in workload.raw for workload in workloads.values()
    )


def test_systems_only_rows_do_not_claim_uncommitted_verified_baselines():
    workloads = load_registry()

    for workload in workloads.values():
        if workload.public_status != "systems-only":
            continue
        assert "verified_baseline" not in workload.raw, workload.id
        calibration = workload.raw.get("calibration_observation")
        if calibration:
            assert calibration.get("evidence_status") in {
                "historical-unverified-no-committed-artifact",
                "bounded-local-methodology-check",
            }, workload.id


def test_public_contract_report_withholds_experimental_candidates():
    workloads = load_registry()
    issues = public_contract_report(workloads)

    experimental = {
        workload.id
        for workload in workloads.values()
        if workload.public_status == "experimental"
    }
    assert experimental
    for workload_id in experimental:
        expected = ["experimental workloads are not public-release-ready"]
        if workload_id == "keyword-spotting":
            expected.append(
                "adapter conformance is quality-preserving but nonidentical; "
                "public promotion is blocked"
            )
        assert issues[workload_id] == expected


def test_keyword_spotting_conformance_is_audited_and_fails_closed():
    workload = load_registry()["keyword-spotting"]
    conformance = workload.raw["adapter_conformance"]
    audit = conformance["audit"]
    root = Path(__file__).resolve().parents[1]
    artifact = root / audit["artifact"]
    artifact_bytes = artifact.read_bytes()
    payload = json.loads(artifact_bytes)

    assert conformance["status"] == "quality-preserving-nonidentical"
    assert conformance["promotion_eligible"] is False
    assert hashlib.sha256(artifact_bytes).hexdigest() == audit["sha256"].removeprefix(
        "sha256:"
    )
    assert payload["schema"] == audit["schema"]
    assert payload["source"]["git_sha"] == audit["source_git_sha"]
    assert payload["source"]["git_dirty"] is False
    assert payload["status"] == "failed"
    assert payload["workloads"]["image-classification"]["status"] == "passed"
    assert payload["workloads"]["keyword-spotting"]["status"] == "failed"
    assert payload["workloads"]["visual-wake-words"]["status"] == "passed"
    assert "model_path" not in artifact.read_text()
    packaged = root / "src" / "mlperf_edu" / audit["artifact"]
    assert packaged.read_bytes() == artifact_bytes
    packaged_resource = resources.files("mlperf_edu").joinpath(audit["artifact"])
    assert packaged_resource.is_file()


def test_nonidentical_adapter_cannot_be_marked_promotion_eligible():
    workloads = load_registry()
    keyword_spotting = workloads["keyword-spotting"]
    raw = deepcopy(keyword_spotting.raw)
    raw["adapter_conformance"]["promotion_eligible"] = True
    workloads["keyword-spotting"] = replace(keyword_spotting, raw=raw)

    with pytest.raises(ValueError, match="nonidentical adapter cannot be promotion"):
        validate_registry(workloads)


def test_verified_baseline_lifecycle_state_machine_fails_closed():
    status = "score-bearing"
    valid_historical = {
        "evidence_status": "committed-reference-summary",
        "review_eligible": False,
        "protocol_compatibility": "superseded",
        "replacement_required": True,
        "superseded_reason": "The protected measurement surface changed.",
    }
    assert baseline_lifecycle_issues(valid_historical, public_status=status) == [
        "score-bearing verified_baseline uses a superseded protocol and requires "
        "a replacement reference sweep"
    ]

    reviewable_history = {**valid_historical, "review_eligible": True}
    missing_replacement = {**valid_historical, "replacement_required": False}
    unexplained_history = {**valid_historical, "superseded_reason": ""}
    orphan_replacement = {
        "review_eligible": True,
        "replacement_required": True,
    }

    assert any(
        "review_eligible to false" in issue
        for issue in baseline_lifecycle_issues(reviewable_history, public_status=status)
    )
    assert any(
        "replacement_required to true" in issue
        for issue in baseline_lifecycle_issues(
            missing_replacement, public_status=status
        )
    )
    assert any(
        "superseded_reason" in issue
        for issue in baseline_lifecycle_issues(
            unexplained_history, public_status=status
        )
    )
    assert any(
        "only when protocol_compatibility is superseded" in issue
        for issue in baseline_lifecycle_issues(orphan_replacement, public_status=status)
    )


def test_public_asset_dossiers_include_size_and_hash_policy():
    tiny = asset_dossier("tinyshakespeare")
    cifar = asset_dossier("cifar10")
    fashion = asset_dossier("fashion-mnist")
    prompt = asset_dossier("prompt-suite-local")
    anomaly = asset_dossier("mlperf-tiny-anomaly-eval")
    vww = asset_dossier("mlperf-tiny-vww-eval")
    humaneval = asset_dossier("humaneval-plus")
    bfcl = asset_dossier("bfcl-v4-non-live-ast")
    movielens = asset_dossier("movielens-20m")
    minigo = asset_dossier("minigo-self-play")

    assert tiny["expected_download_bytes"] == 5_600_000
    assert tiny["expected_unpacked_bytes"] == 1_115_394
    assert tiny["hash_policy"]
    assert tiny["license_status"] == "mit-repository-public-domain-text"
    assert tiny["public_release_status"] == "public-ok-fetch-only"
    assert cifar["license_status"] == "source-citation-no-license"
    assert cifar["public_release_status"] == "needs-release-decision"
    assert fashion["license_status"] == "mit"
    assert fashion["license_spdx"] == "MIT"
    assert fashion["public_release_status"] == "public-ok-with-attribution"
    assert fashion["public_result_use"] == (
        "standalone educational lab asset; never a benchmark result"
    )
    assert anomaly["expected_download_bytes"] == 69_897_209
    assert anomaly["expected_unpacked_bytes"] == 25_408_204
    assert anomaly["license_spdx"] == "CC-BY-4.0"
    assert anomaly["public_release_status"] == "public-ok-fetch-only"
    assert prompt["expected_download_bytes"] == 0
    assert prompt["public_result_use"] == "performance-bearing functional check"
    assert prompt["public_release_status"] == "public-ok-bundled"
    assert vww["expected_download_bytes"] == 234_810_765
    assert vww["expected_unpacked_bytes"] == 2_747_212
    assert vww["public_release_status"] == "needs-release-decision"
    assert humaneval["expected_unpacked_bytes"] == 7_714_666
    assert humaneval["public_release_status"] == "public-ok-fetch-only"
    assert bfcl["version"].startswith("bfcl-")
    assert bfcl["public_release_status"] == "needs-release-decision"
    assert movielens["public_release_status"] == "fetch-only"
    assert movielens["expected_download_bytes"] == 198_702_078
    assert minigo["type"] == "generated-dataset"


def test_every_workload_dataset_has_a_structured_asset_dossier():
    workloads = load_registry()

    assert all(has_asset_dossier(workload.dataset) for workload in workloads.values())


def test_retired_dataset_adapters_are_not_public_assets():
    for dataset in ("movielens-100k", "mnist", "cifar100"):
        assert not has_asset_dossier(dataset)


def test_starter_selection_uses_workload_collection():
    workloads = load_registry()
    selected = select_workloads(workloads, collection="starter")
    selected_ids = {workload.id for workload in selected}

    assert selected_ids == STARTER_WORKLOADS
    assert [workload.id for workload in selected] == list(STARTER_WORKLOAD_ORDER)


def test_standard_selection_uses_workload_collection():
    workloads = load_registry()
    selected = select_workloads(workloads, collection="standard")

    assert {workload.id for workload in selected} == STANDARD_WORKLOADS
    assert [workload.id for workload in selected] == list(STANDARD_WORKLOAD_ORDER)


def test_research_selection_uses_workload_collection():
    workloads = load_registry()
    selected = select_workloads(workloads, collection="research")

    assert {workload.id for workload in selected} == RESEARCH_WORKLOADS
    assert [workload.id for workload in selected] == list(RESEARCH_WORKLOAD_ORDER)


def test_canonical_workloads_declare_exact_runners():
    workloads = load_registry()
    expected = {
        "causal-language-modeling": (
            "language",
            "mlperf.runners.nanogpt:run_causal_language_modeling_min",
            "mlperf.runners.nanogpt:run_causal_language_modeling_max",
        ),
        "text-classification": (
            "language",
            "mlperf.runners.text:run_text_classification_min",
            "mlperf.runners.text:run_text_classification_max",
        ),
        "information-retrieval": (
            "language",
            "mlperf.runners.retrieval:run_information_retrieval_min",
            "mlperf.runners.retrieval:run_information_retrieval_max",
        ),
        "image-classification": (
            "vision",
            "mlperf.runners.vision:run_image_classification_min",
            "mlperf.runners.vision:run_image_classification_max",
        ),
        "keyword-spotting": (
            "tiny",
            "mlperf.runners.tiny:run_keyword_spotting_min",
            "mlperf.runners.tiny:run_keyword_spotting_max",
        ),
        "anomaly-detection": (
            "tiny",
            "mlperf.runners.tiny:run_anomaly_detection_min",
            "mlperf.runners.tiny:run_anomaly_detection_max",
        ),
        "visual-wake-words": (
            "tiny",
            "mlperf.runners.tiny:run_visual_wake_words_min",
            "mlperf.runners.tiny:run_visual_wake_words_max",
        ),
        "graph-node-classification": (
            "graph",
            "mlperf.runners.graph:run_graph_node_classification_min",
            "mlperf.runners.graph:run_graph_node_classification_max",
        ),
        "time-series-forecasting": (
            "timeseries",
            "mlperf.runners.timeseries:run_time_series_forecasting_min",
            "mlperf.runners.timeseries:run_time_series_forecasting_max",
        ),
        "code-generation": (
            "language",
            "mlperf.runners.functional_setup:run_code_generation_min",
            "mlperf.runners.code_generation:run_code_generation_max",
        ),
        "function-calling": (
            "language",
            "mlperf.runners.functional_setup:run_function_calling_min",
            "mlperf.runners.function_calling:run_function_calling_max",
        ),
        "recommendation": (
            "recommendation",
            "mlperf.runners.functional_setup:run_recommendation_min",
            "mlperf.runners.ncf:run_recommendation_max",
        ),
        "image-generation": (
            "vision",
            "mlperf.runners.functional_setup:run_image_generation_min",
            "mlperf.runners.image_generation:run_image_generation_max",
        ),
        "reinforcement-learning": (
            "reinforcement",
            "mlperf.runners.functional_setup:run_reinforcement_learning_min",
            "mlperf.runners.minigo:run_reinforcement_learning_max",
        ),
    }

    assert set(workloads) == set(expected)
    for workload_id, (suite, min_runner, max_runner) in expected.items():
        workload = workloads[workload_id]
        assert workload.suite == suite
        assert workload.raw["runner"] == {"min": min_runner, "max": max_runner}


def test_functional_spiral_workloads_fail_closed_for_promotion():
    workloads = load_registry()
    functional = {
        "code-generation",
        "function-calling",
        "recommendation",
        "image-generation",
        "reinforcement-learning",
    }

    for workload_id in functional:
        workload = workloads[workload_id]
        assert workload.public_status == "experimental"
        assert workload.raw["promotion_scope"] is False
        spiral = workload.raw["spiral"]
        expected_stage = (
            "quality-conformance"
            if workload_id
            in {
                "code-generation",
                "function-calling",
                "image-generation",
                "recommendation",
                "reinforcement-learning",
            }
            else "functional"
        )
        assert spiral["stage"] == expected_stage
        assert spiral["functional_ready"] is True
        assert spiral["quality_conformant"] is False
        assert spiral["repeatability_verified"] is False
        assert spiral["promotion_ready"] is False
        assert spiral["next_gate"]
        # execution_status marks a contract that cannot complete as written on
        # the target platform. Reinforcement learning shed it when the PyTorch
        # adapter replaced the container, so it is no longer required of every
        # functional-spiral workload, only of those still blocked.
        contract = workload.raw["canonical_max_contract"]
        if workload_id != "reinforcement-learning":
            assert contract["execution_status"]


def test_retrieval_declares_one_complete_evaluation_protocol():
    workload = load_registry()["information-retrieval"]
    config = workload.raw["canonical_max_contract"]["config"]
    protocol = workload.raw["measurement_protocol"]

    assert "warmup_evaluations" not in config
    assert config["measurement_repetitions"] == 1
    assert config["performance_aggregate"] == "single-complete-evaluation"
    assert protocol["primary_metric"] == "inference_and_evaluation_seconds"
    assert "one complete" in protocol["timing_scope"]
    assert "warmup for each scoring batch shape" in " ".join(
        protocol["excluded_phases"]
    )


def test_causal_language_modeling_declares_modes_and_phases():
    workload = load_registry()["causal-language-modeling"]

    assert workload.raw["implemented_modes"] == ["training", "inference"]
    assert workload.raw["phases"]["inference"] == ["full", "prefill", "decode"]
    assert (
        workload.raw["mode_contracts"]["inference"]["dataset"] == "prompt-suite-local"
    )


def test_tiny_suite_contains_canonical_mlperf_tiny_workloads():
    workloads = load_registry()
    selected = select_workloads(workloads, suite="tiny")

    assert [workload.id for workload in selected] == [
        "keyword-spotting",
        "anomaly-detection",
        "visual-wake-words",
    ]


def test_unknown_workload_rejected():
    workloads = load_registry()
    try:
        select_workloads(workloads, workload_id="does-not-exist")
    except ValueError as exc:
        assert "unknown workload" in str(exc)
    else:
        raise AssertionError("unknown workload should fail")
