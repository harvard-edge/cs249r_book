import math

import pytest

from mlsysim.engine.constraint_explorer import (
    TRACKS,
    compare,
    evaluate,
    margins,
    monitoring_plan,
    population_compare,
    speedup,
    stress_compare,
)


def test_every_track_has_a_feasible_balanced_baseline_and_reachable_failure():
    for track_id in TRACKS:
        baseline = evaluate(track_id)
        failed = evaluate(track_id, demand_scale=2.0)

        assert baseline["feasible"], track_id
        assert not failed["feasible"], track_id
        assert failed["violations"]


def test_candidate_ladder_raises_quality_and_physical_costs_for_same_task():
    compact = evaluate("edge", candidate="compact")
    balanced = evaluate("edge", candidate="balanced")
    large = evaluate("edge", candidate="large")

    assert compact["quality_pct"] < balanced["quality_pct"] < large["quality_pct"]
    assert compact["memory_mb"] < balanced["memory_mb"] < large["memory_mb"]
    assert compact["compute_ms"] < balanced["compute_ms"] < large["compute_ms"]
    assert compact["movement_ms"] < balanced["movement_ms"] < large["movement_ms"]


def test_distribution_shift_changes_weighted_quality_with_fixed_model():
    easy_mix = evaluate("mobile", shift_pct=-10)
    hard_mix = evaluate("mobile", shift_pct=10)

    assert easy_mix["candidate"] == hard_mix["candidate"] == "balanced"
    assert easy_mix["quality_pct"] > hard_mix["quality_pct"]
    assert easy_mix["latency_ms"] == hard_mix["latency_ms"]
    assert easy_mix["easy_share_pct"] + easy_mix["difficult_share_pct"] == pytest.approx(100.0)
    assert easy_mix["candidate_scale"] == 1.0


def test_interventions_have_equal_first_cost_and_distinct_causal_effects():
    baseline = evaluate("edge")
    data = evaluate("edge", intervention="data")
    model = evaluate("edge", intervention="model")
    machine = evaluate("edge", intervention="machine")

    assert {data["intervention_cost"], model["intervention_cost"], machine["intervention_cost"]} == {1.0}
    assert data["quality_pct"] > baseline["quality_pct"]
    assert data["compute_ms"] == baseline["compute_ms"]
    assert model["effective_candidate"] == "large"
    assert model["memory_mb"] > baseline["memory_mb"]
    assert machine["compute_ms"] < baseline["compute_ms"]
    assert machine["energy_mj"] > baseline["energy_mj"]
    assert baseline["operational_ok"]
    assert baseline["inputs"]["intervention"] == "none"


def test_compare_keeps_same_nonintervention_baseline_and_helpers_report_units():
    result = compare("tinyml", intervention="machine", demand_scale=1.25)

    assert result["baseline"]["intervention_cost"] == 0.0
    assert result["baseline"]["movement_volume_mb"] == pytest.approx(0.1375)
    assert result["speedup"] == speedup(result["baseline"], result["result"])
    assert result["margins"] == margins(result["result"])
    assert result["delta"]["compute_ms"] < 0


def test_timing_is_derived_from_volume_bandwidth_ops_and_effective_rate():
    result = evaluate("edge", demand_scale=1.2)

    assert result["movement_ms"] == pytest.approx(
        result["movement_volume_mb"] / result["bandwidth_mb_per_ms"]
    )
    assert result["compute_ms"] == pytest.approx(
        result["ops_mflop"] / result["effective_rate_mflop_per_ms"]
    )
    assert result["latency_ms"] == pytest.approx(
        result["movement_ms"] + result["compute_ms"] + result["overhead_ms"]
    )


def test_tracks_teach_different_binding_terms_and_capability_changes_can_migrate_them():
    initial = {track_id: evaluate(track_id)["dominant_term"] for track_id in TRACKS}

    assert len(set(initial.values())) >= 3
    assert evaluate("tinyml", movement_scale=4)["dominant_term"] == "compute"
    assert evaluate("mobile", overhead_scale=4)["dominant_term"] == "compute"
    assert evaluate("edge", compute_scale=4)["dominant_term"] == "movement"
    assert evaluate("cloud", movement_scale=4)["dominant_term"] == "compute"


def test_large_candidate_can_be_made_feasible_by_a_looser_envelope():
    constrained = evaluate("edge", candidate="large")
    loose = evaluate("edge", candidate="large", budget_scale=1.4)

    assert not constrained["feasible"]
    assert loose["feasible"]


def test_stress_compare_preserves_cohort_shift_and_uses_an_unchanged_baseline():
    comparison = stress_compare("edge", shift_pct=8, intervention="data", condition="compute", scale=2)

    assert comparison["baseline"]["inputs"]["shift_pct"] == 8.0
    assert comparison["result"]["inputs"]["shift_pct"] == 8.0
    assert comparison["baseline"]["inputs"]["intervention"] == "data"
    assert comparison["baseline"]["inputs"]["compute_scale"] == 1.0
    assert comparison["result"]["inputs"]["compute_scale"] == 2.0
    assert comparison["result"]["compute_ms"] < comparison["baseline"]["compute_ms"]


def test_stress_compare_preserves_a_looser_deployment_envelope():
    comparison = stress_compare("edge", candidate="large", condition="compute", scale=2, budget_scale=1.4)

    assert comparison["baseline"]["inputs"]["budget_scale"] == 1.4
    assert comparison["result"]["inputs"]["budget_scale"] == 1.4
    assert comparison["baseline"]["feasible"]
    assert comparison["result"]["feasible"]


def test_population_compare_changes_only_cohort_quality_and_preserves_envelope():
    comparison = population_compare(
        "edge", shift_pct=4, shift_delta_pct=8, intervention="data", budget_scale=1.4
    )

    assert comparison["shift_delta_pct"] == 8.0
    assert comparison["baseline"]["inputs"]["shift_pct"] == 4.0
    assert comparison["result"]["inputs"]["shift_pct"] == 12.0
    assert comparison["baseline"]["inputs"]["budget_scale"] == 1.4
    assert comparison["result"]["inputs"]["budget_scale"] == 1.4
    assert comparison["result"]["quality_pct"] < comparison["baseline"]["quality_pct"]
    for field in ("movement_ms", "compute_ms", "overhead_ms", "latency_ms", "energy_mj", "memory_mb"):
        assert comparison["result"][field] == comparison["baseline"][field]


def test_monitoring_is_separate_from_model_quality():
    baseline = evaluate("cloud")
    plan = monitoring_plan("cloud", demand_scale=2)

    assert "quality_pct" not in plan
    assert plan["detection_delay_ms"] == plan["check_interval_ms"] / 2
    assert plan["check_effort_mj"] > 0
    frequent = monitoring_plan("cloud", check_interval_ms=50)
    assert frequent["detection_delay_ms"] < plan["detection_delay_ms"]
    assert frequent["total_check_effort_mj"] > plan["total_check_effort_mj"]
    assert evaluate("cloud")["quality_pct"] == baseline["quality_pct"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"track_id": "unknown"},
        {"track_id": "edge", "candidate": "huge"},
        {"track_id": "edge", "intervention": "magic"},
        {"track_id": "edge", "demand_scale": 0},
        {"track_id": "edge", "compute_scale": math.inf},
        {"track_id": "edge", "shift_pct": 100},
    ],
)
def test_invalid_or_out_of_bounds_inputs_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        evaluate(**kwargs)


@pytest.mark.parametrize("track_id", TRACKS)
def test_intro_candidates_keep_all_requirements_fixed_under_population_changes(track_id):
    limits = ("memory_limit_mb", "latency_limit_ms", "energy_limit_mj", "quality_floor_pct")
    baseline = evaluate(track_id)
    for candidate in ("compact", "balanced", "large"):
        result = evaluate(track_id, candidate=candidate, shift_pct=25)
        assert {key: result[key] for key in limits} == {key: baseline[key] for key in limits}


@pytest.mark.parametrize("term", ["movement", "compute", "overhead"])
def test_intro_one_term_counterfactual_preserves_other_phases_and_quality(term):
    baseline = evaluate("mobile")
    changed = evaluate("mobile", **{f"{term}_scale": 4})
    for phase in ("movement", "compute", "overhead"):
        expected = baseline[f"{phase}_ms"] / 4 if phase == term else baseline[f"{phase}_ms"]
        assert changed[f"{phase}_ms"] == pytest.approx(expected)
    assert changed["quality_pct"] == baseline["quality_pct"]
    assert changed["memory_mb"] == baseline["memory_mb"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_intro_task_factors_have_independent_supplied_quality(track_id):
    from mlsysim.engine.constraint_explorer import task_factors
    base = task_factors(track_id)
    fewer = task_factors(track_id, "data")
    algorithm = task_factors(track_id, "algorithm")
    machine = task_factors(track_id, "execution")
    unknown = task_factors(track_id, "unverified")
    assert fewer["duration_seconds"] == algorithm["duration_seconds"] == machine["duration_seconds"]
    assert fewer["quality_pct"] != algorithm["quality_pct"]
    assert not fewer["feasible"] and algorithm["feasible"] and machine["feasible"]
    assert unknown["quality_pct"] is None and not unknown["feasible"]
    assert machine["total_operations"] == base["total_operations"]
    assert algorithm["samples_per_task"] == base["samples_per_task"]
    assert fewer["operations_per_sample"] == base["operations_per_sample"]
    assert machine["duration_seconds"] == pytest.approx(machine["total_operations"] * machine["seconds_per_operation"])


def test_intro_task_replay_is_closed_and_capstone_preserves_missing_evidence():
    from mlsysim.engine.constraint_explorer import replay, task_factors
    from mlsysim.labs.experiments.v1_16_experiments import replay_evidence_arm
    result = task_factors("tinyml", "unverified")
    assert replay("constraint_explorer.task_factors.v1", result["inputs"]) == result
    assert replay_evidence_arm("constraint_explorer", "constraint_explorer.task_factors.v1", result) == result
    with pytest.raises(ValueError):
        replay("invented", result["inputs"])
    with pytest.raises(ValueError):
        task_factors("tinyml", "invented")


def test_intro_versioned_replay_rejects_a_changed_scenario_and_preserves_legacy_inputs():
    from mlsysim.engine.constraint_explorer import INTRO_SCENARIO_VERSION, replay, task_factors
    from mlsysim.engine.scenarios import Scenarios
    study = Scenarios.ControlledTaskFactors
    assert study.quality_floor_pct == 82.0
    assert study.provenance.kind.value == "illustrative"
    task = task_factors("tinyml")
    assert task["inputs"]["scenario_version"] == INTRO_SCENARIO_VERSION
    with pytest.raises(ValueError, match="version differs"):
        replay("constraint_explorer.task_factors.v1", {**task["inputs"], "scenario_version": "obsolete"})
    deployment = evaluate("mobile", scenario_version=INTRO_SCENARIO_VERSION)
    assert replay("constraint_explorer.evaluate", deployment["inputs"]) == deployment
    with pytest.raises(ValueError, match="version differs"):
        replay("constraint_explorer.evaluate", {**deployment["inputs"], "scenario_version": "obsolete"})
    legacy = evaluate("mobile")
    assert "scenario_version" not in legacy["inputs"]
    assert replay("constraint_explorer.evaluate", legacy["inputs"]) == legacy
