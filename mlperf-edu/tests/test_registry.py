from collections import Counter
from pathlib import Path

from mlperf.assets import asset_dossier
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
)
from mlperf.runners.tiny import _canonical_config_int


def test_profiles_are_min_max_pro():
    assert PROFILES == ("min", "max", "pro")


def test_registry_loads_current_workloads():
    workloads = load_registry()
    assert len(workloads) == 7
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
    assert counts == {"experimental": 7}
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
        assert issues[workload_id] == [
            "experimental workloads are not public-release-ready"
        ]


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
    movielens = asset_dossier("movielens-100k")
    mnist = asset_dossier("mnist")
    cifar = asset_dossier("cifar100")
    fashion = asset_dossier("fashion-mnist")
    prompt = asset_dossier("prompt-suite-local")

    assert tiny["expected_download_bytes"] == 5_600_000
    assert tiny["expected_unpacked_bytes"] == 1_115_394
    assert tiny["hash_policy"]
    assert tiny["license_status"] == "mit-repository-public-domain-text"
    assert tiny["public_release_status"] == "public-ok-fetch-only"
    assert movielens["expected_unpacked_bytes"] == 16_100_896
    assert movielens["license_status"] == "noncommercial-research-education"
    assert movielens["public_release_status"] == "restricted-needs-approval"
    assert mnist["license_status"] == "cc-by-sa-3.0"
    assert mnist["public_release_status"] == "public-ok-with-attribution"
    assert cifar["license_status"] == "source-citation-no-license"
    assert cifar["public_release_status"] == "needs-release-decision"
    assert fashion["license_status"] == "mit"
    assert fashion["license_spdx"] == "MIT"
    assert fashion["public_release_status"] == "public-ok-with-attribution"
    assert prompt["expected_download_bytes"] == 0
    assert prompt["public_result_use"] == "performance-bearing functional check"
    assert prompt["public_release_status"] == "public-ok-bundled"


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
    }

    assert set(workloads) == set(expected)
    for workload_id, (suite, min_runner, max_runner) in expected.items():
        workload = workloads[workload_id]
        assert workload.suite == suite
        assert workload.raw["runner"] == {"min": min_runner, "max": max_runner}


def test_retrieval_declares_steady_state_repetition_protocol():
    workload = load_registry()["information-retrieval"]
    config = workload.raw["canonical_max_contract"]["config"]
    protocol = workload.raw["measurement_protocol"]

    assert config["warmup_evaluations"] == 5
    assert config["measurement_repetitions"] == 3
    assert config["performance_aggregate"] == "median"
    assert protocol["primary_metric"] == "inference_and_evaluation_seconds"
    assert "three unchanged complete" in protocol["timing_scope"]
    assert "five untimed complete" in " ".join(protocol["excluded_phases"])


def test_causal_language_modeling_declares_modes_and_phases():
    workload = load_registry()["causal-language-modeling"]

    assert workload.raw["implemented_modes"] == ["training", "inference"]
    assert workload.raw["phases"]["inference"] == ["full", "prefill", "decode"]
    assert (
        workload.raw["mode_contracts"]["inference"]["dataset"] == "prompt-suite-local"
    )


def test_tiny_suite_contains_only_canonical_keyword_spotting():
    workloads = load_registry()
    selected = select_workloads(workloads, suite="tiny")

    assert [workload.id for workload in selected] == ["keyword-spotting"]


def test_unknown_workload_rejected():
    workloads = load_registry()
    try:
        select_workloads(workloads, workload_id="does-not-exist")
    except ValueError as exc:
        assert "unknown workload" in str(exc)
    else:
        raise AssertionError("unknown workload should fail")
