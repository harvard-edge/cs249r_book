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
