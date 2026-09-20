import pytest

from mlsysim.engine.v2_11_experiments import (
    TRACKS,
    client_records,
    compare_architectures,
    evaluate_adaptation,
    evaluate_architecture,
    evaluate_client_selection,
    evaluate_federation,
    evaluate_replay,
)


def test_every_track_has_an_admitted_method_and_a_reachable_memory_failure():
    for track_id in TRACKS:
        assert evaluate_adaptation(track_id, "bias", concurrent_with_foreground=False)["admitted"]
        failed = evaluate_adaptation(track_id, "full", memory_budget_scale=0.01)
        assert not failed["admitted"]
        assert "memory" in failed["violations"]


def test_each_admission_constraint_can_fail_independently():
    memory = evaluate_adaptation("edge", "adapter", concurrent_with_foreground=False, memory_budget_scale=0.01)
    energy = evaluate_adaptation("edge", "adapter", concurrent_with_foreground=False, energy_budget_scale=0)
    window = evaluate_adaptation("edge", "adapter", concurrent_with_foreground=False, window_scale=0)
    foreground = evaluate_adaptation("edge", "full", concurrent_with_foreground=True)

    assert memory["violations"] == ["memory"]
    assert energy["violations"] == ["energy"]
    assert window["violations"] == ["update window"]
    assert foreground["violations"] == ["foreground latency"]


def test_update_methods_separate_state_energy_and_foreground_pressure():
    full = evaluate_adaptation("mobile", "full")
    adapter = evaluate_adaptation("mobile", "adapter")
    bias = evaluate_adaptation("mobile", "bias")

    assert full["total_memory_mib"] > adapter["total_memory_mib"] > bias["total_memory_mib"]
    assert full["energy_j"] > adapter["energy_j"] > bias["energy_j"]
    assert full["foreground_delay_ms"] > adapter["foreground_delay_ms"] > bias["foreground_delay_ms"]
    assert (
        full["supplied_new_context_quality_pct"]
        > adapter["supplied_new_context_quality_pct"]
        > bias["supplied_new_context_quality_pct"]
    )


def test_scheduling_away_from_foreground_changes_only_latency_admission():
    concurrent = evaluate_adaptation("edge", "full", concurrent_with_foreground=True)
    background = evaluate_adaptation("edge", "full", concurrent_with_foreground=False)

    for key in ("total_memory_mib", "duration_s", "energy_j", "supplied_new_context_quality_pct"):
        assert background[key] == pytest.approx(concurrent[key])
    assert background["foreground_delay_ms"] == 0
    assert concurrent["foreground_delay_ms"] > 0


def test_batch_size_changes_activation_memory_but_not_fixed_update_work():
    small = evaluate_adaptation("mobile", "adapter", batch_size=1)
    large = evaluate_adaptation("mobile", "adapter", batch_size=8)

    assert large["activations_mib"] > small["activations_mib"]
    assert large["total_memory_mib"] > small["total_memory_mib"]
    assert large["operations_gflop"] == pytest.approx(small["operations_gflop"])
    assert large["duration_s"] == pytest.approx(small["duration_s"])
    assert large["energy_j"] == pytest.approx(small["energy_j"])


def test_replay_trades_old_context_retention_against_new_context_outcome():
    none = evaluate_replay("mobile", "adapter", 0.0)
    balanced = evaluate_replay("mobile", "adapter", 0.5)
    heavy = evaluate_replay("mobile", "adapter", 0.75)

    assert none["retained_examples"] < balanced["retained_examples"] < heavy["retained_examples"]
    assert none["old_context_quality_pct"] < balanced["old_context_quality_pct"] < heavy["old_context_quality_pct"]
    assert none["new_context_quality_pct"] > balanced["new_context_quality_pct"] > heavy["new_context_quality_pct"]
    assert balanced["balanced_quality_pct"] > none["balanced_quality_pct"]
    assert balanced["balanced_quality_pct"] > heavy["balanced_quality_pct"]


def test_replay_memory_is_finite_and_can_crowd_out_adaptation():
    roomy = evaluate_replay("mobile", "full", 0.0)
    crowded = evaluate_replay("mobile", "full", 0.75)

    assert roomy["adaptation_budget_mib"] > crowded["adaptation_budget_mib"]
    assert roomy["feasible"]
    assert not crowded["feasible"]


def test_local_epochs_have_initial_benefit_and_heterogeneity_exposes_reversal():
    high = [evaluate_federation("mobile", epochs, "high") for epochs in (1, 2, 4, 8)]

    assert high[1]["rounds_to_target"] < high[0]["rounds_to_target"]
    assert high[2]["total_communication_mib"] < high[1]["total_communication_mib"]
    assert high[3]["rounds_to_target"] > high[2]["rounds_to_target"]
    assert high[3]["total_communication_mib"] > high[2]["total_communication_mib"]
    assert {row["target_quality_pct"] for row in high} == {90.0}


def test_federation_physics_change_without_rewriting_supplied_convergence():
    base = evaluate_federation("edge", 4, "moderate")
    more_clients = evaluate_federation("edge", 4, "moderate", client_count_scale=1.5)

    assert base["rounds_to_target"] == more_clients["rounds_to_target"]
    assert base["target_quality_pct"] == more_clients["target_quality_pct"]
    assert more_clients["total_communication_mib"] > base["total_communication_mib"]
    assert more_clients["wall_time_s"] > base["wall_time_s"]


def test_architectures_preserve_distinct_data_boundaries():
    rows = {row["architecture"]: row for row in compare_architectures("tinyml")}

    assert rows["central"]["raw_data_leaves_device"]
    assert not rows["local"]["raw_data_leaves_device"]
    assert rows["local"]["fleet_transfer_mib"] == 0
    assert rows["federated"]["population_update"]
    assert not rows["federated"]["raw_data_leaves_device"]


def test_architecture_result_preserves_exact_replayable_arguments():
    result = evaluate_architecture("edge", "federated", local_epochs=4)

    replay = evaluate_architecture(**result["inputs"])
    assert replay == result


def test_no_action_architecture_is_explicit_and_contrasts_with_tested_option():
    no_action = evaluate_architecture("mobile", "none")
    tested = evaluate_architecture("mobile", "federated")

    assert no_action["fleet_transfer_mib"] == 0
    assert not no_action["population_update"]
    assert no_action["inputs"] != tested["inputs"]


def test_selection_reports_resource_failures_freshness_and_cohort_coverage():
    result = evaluate_client_selection("mobile", selection_policy="coverage", deadline_seconds=3)

    assert result["eligible_count"] > 0
    assert 0 < result["cohort_coverage_fraction"] <= 1
    assert result["failures"]
    assert "stale evidence" in result["failure_counts"]
    assert "deadline" in result["failure_counts"]


def test_all_track_endpoints_run_each_fleet_experiment():
    for track_id in TRACKS:
        assert evaluate_replay(track_id, "adapter", 0.25)["feasible"]
        assert evaluate_federation(track_id, 2, "low")["wall_time_s"] > 0
        selection = evaluate_client_selection(track_id, min_uplink_mbps=0, deadline_seconds=0)
        assert selection["eligible_count"] > 0
        assert selection["completion_seconds"]
        assert len(compare_architectures(track_id)) == 3


def test_results_capture_exact_replayable_evaluator_arguments():
    adaptation = evaluate_adaptation("mobile", "adapter", batch_size=4, concurrent_with_foreground=False)
    replay = evaluate_replay("edge", "bias", 0.5)
    federation = evaluate_federation("tinyml", 4, "high", method="bias", client_count_scale=1.5)
    selection = evaluate_client_selection("cloud", selection_policy="coverage", unavailable_domains=("site-a",))

    assert evaluate_adaptation(**adaptation["inputs"])["energy_j"] == pytest.approx(adaptation["energy_j"])
    assert evaluate_replay(**replay["inputs"])["retained_examples"] == replay["retained_examples"]
    assert evaluate_federation(**federation["inputs"])["wall_time_s"] == pytest.approx(federation["wall_time_s"])
    assert evaluate_client_selection(**selection["inputs"])["selected_ids"] == selection["selected_ids"]


def test_coverage_selection_can_cover_more_cohorts_than_freshness_ranking():
    freshest = evaluate_client_selection("edge", selection_policy="freshest", max_clients=3)
    coverage = evaluate_client_selection("edge", selection_policy="coverage", max_clients=3)

    assert coverage["cohort_coverage_fraction"] >= freshest["cohort_coverage_fraction"]
    assert coverage["mean_evidence_age_hours"] >= freshest["mean_evidence_age_hours"]


def test_failure_domain_outage_removes_correlated_clients():
    baseline = evaluate_client_selection("cloud", min_uplink_mbps=0, deadline_seconds=0)
    outage = evaluate_client_selection("cloud", min_uplink_mbps=0, deadline_seconds=0, unavailable_domains=("site-a",))

    assert outage["eligible_count"] < baseline["eligible_count"]
    assert outage["failure_counts"]["failure domain unavailable"] == 2


@pytest.mark.parametrize(
    "call",
    [
        lambda: evaluate_adaptation("unknown", "bias"),
        lambda: evaluate_adaptation("mobile", "magic"),
        lambda: evaluate_adaptation("mobile", "bias", memory_budget_scale=float("inf")),
        lambda: evaluate_replay("edge", "adapter", 0.3),
        lambda: evaluate_federation("edge", 3, "high"),
        lambda: evaluate_federation("edge", 2, "unknown"),
        lambda: evaluate_federation("edge", 2, "low", target_quality_pct=92),
        lambda: evaluate_client_selection("tinyml", max_clients=0),
        lambda: evaluate_client_selection("tinyml", selection_policy="random"),
    ],
)
def test_invalid_inputs_fail_explicitly(call):
    with pytest.raises(ValueError):
        call()


def test_mains_powered_tracks_bypass_charging_filter():
    mobile_charging = evaluate_client_selection("mobile", require_charging=True)
    mobile_no_charging = evaluate_client_selection("mobile", require_charging=False)
    assert mobile_charging["eligible_count"] < mobile_no_charging["eligible_count"]
    assert "not charging" in mobile_charging["failure_counts"]

    for track_id in ("edge", "cloud"):
        with_charging = evaluate_client_selection(track_id, require_charging=True)
        without_charging = evaluate_client_selection(track_id, require_charging=False)
        assert with_charging["eligible_count"] == without_charging["eligible_count"]
        assert "not charging" not in with_charging["failure_counts"]
        assert with_charging["mains_powered"]


def test_track_specific_rosters_reflect_deployment_domains():
    tinyml_records = client_records("tinyml")
    mobile_records = client_records("mobile")
    edge_records = client_records("edge")
    cloud_records = client_records("cloud")

    assert {c.cohort for c in tinyml_records} == {"industrial", "ambient", "acoustic", "structural"}
    assert {c.cohort for c in mobile_records} == {"urban", "rural", "night", "assistive"}
    assert {c.cohort for c in edge_records} == {"retail", "logistics", "clinic", "metro"}
    assert {c.cohort for c in cloud_records} == {"finance", "health", "analytics", "public"}

    assert min(c.uplink_mbps for c in cloud_records) >= 300
    assert max(c.uplink_mbps for c in mobile_records) <= 25
