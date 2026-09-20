from copy import deepcopy
import json
import math

import pytest

from mlsysim.engine.v2_07_experiments import (
    TRACKS,
    checkpoint_fault_fixture,
    checkpoint_tradeoff,
    get_track_scenario,
    replica_availability,
    select_restore_checkpoint,
    serving_capacity,
    training_exposure,
)


def test_all_tracks_have_honest_training_and_serving_fleet_semantics():
    for track_id in TRACKS:
        training = training_exposure(track_id)
        serving = serving_capacity(track_id)

        assert training["semantics"] == "coupled_training_job"
        assert training["expected_interruptions"] > 0
        assert 0 < training["interruption_probability"] < 1
        assert serving["semantics"] == "independent_serving_capacity"
        assert not serving["fleet_terminated"]
        assert 0 < serving["expected_available_devices"] < serving["fleet_size"]


def test_more_workers_and_correlated_domain_events_raise_training_exposure():
    small = training_exposure("cloud", workers=64, failure_domains=1)
    large = training_exposure("cloud", workers=2_048, failure_domains=256)
    independent_only = training_exposure("cloud", workers=2_048, failure_domains=1, domain_mttf_h=1e15)

    assert large["system_mtbf_h"] < small["system_mtbf_h"]
    assert large["expected_interruptions"] > small["expected_interruptions"]
    assert large["interruption_rate_per_h"] > independent_only["interruption_rate_per_h"]


def test_checkpoint_curve_penalizes_both_over_and_under_checkpointing():
    reference = checkpoint_tradeoff("cloud")
    optimum = reference["optimal_interval_min"]
    too_frequent = checkpoint_tradeoff("cloud", interval_min=optimum / 5)
    at_optimum = checkpoint_tradeoff("cloud", interval_min=optimum)
    too_rare = checkpoint_tradeoff("cloud", interval_min=optimum * 5)

    assert too_frequent["dominant_interval_cost"] == "checkpoint_writes"
    assert too_rare["dominant_interval_cost"] == "lost_work"
    assert at_optimum["total_waste_fraction"] < too_frequent["total_waste_fraction"]
    assert at_optimum["total_waste_fraction"] < too_rare["total_waste_fraction"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_each_track_has_valid_and_failed_recovery_contract_cases(track_id):
    baseline = checkpoint_tradeoff(track_id)
    failed = checkpoint_tradeoff(
        track_id,
        interval_min=baseline["requested_rpo_min"] * 2,
        requested_rto_s=baseline["recovery_s"] / 2,
    )

    assert baseline["rpo_ok"] and baseline["rto_ok"], track_id
    assert not failed["rpo_ok"] and not failed["rto_ok"], track_id


def test_recovery_phases_are_separate_and_only_recovery_tax_changes():
    baseline = checkpoint_tradeoff("edge", detection_s=60, restart_s=180, load_s=60, warmup_s=120)
    faster_detection = checkpoint_tradeoff("edge", detection_s=10, restart_s=180, load_s=60, warmup_s=120)

    assert faster_detection["recovery_s"] == baseline["recovery_s"] - 50
    assert faster_detection["recovery_tax"] < baseline["recovery_tax"]
    assert faster_detection["write_tax"] == baseline["write_tax"]
    assert faster_detection["rework_tax"] == baseline["rework_tax"]


def test_rpo_uses_maximum_rollback_window_and_rto_uses_only_recovery_phases():
    passing = checkpoint_tradeoff(
        "cloud", interval_min=20, requested_rpo_min=20, requested_rto_s=400
    )
    failed_rpo = checkpoint_tradeoff(
        "cloud", interval_min=21, requested_rpo_min=20, requested_rto_s=400
    )
    failed_rto = checkpoint_tradeoff(
        "cloud", interval_min=20, requested_rpo_min=20, requested_rto_s=300
    )

    assert passing["rpo_ok"] and passing["rto_ok"]
    assert not failed_rpo["rpo_ok"] and failed_rpo["rto_ok"]
    assert failed_rto["rpo_ok"] and not failed_rto["rto_ok"]
    assert passing["expected_lost_work_min"] == passing["interval_min"] / 2
    assert "steady-state" in passing["time_basis"]


def test_async_pause_changes_checkpoint_tax_without_changing_write_or_recovery():
    synchronous = checkpoint_tradeoff("mobile")
    staged = checkpoint_tradeoff("mobile", checkpoint_pause_s=5)

    assert staged["checkpoint_write_s"] == synchronous["checkpoint_write_s"]
    assert staged["write_tax"] < synchronous["write_tax"]
    assert staged["recovery_s"] == synchronous["recovery_s"]


def test_equal_bandwidth_repairs_change_only_the_allocated_path():
    baseline = checkpoint_tradeoff("edge", state_scale=5)
    write_path = checkpoint_tradeoff(
        "edge", state_scale=5, bandwidth_scale=4, bandwidth_allocation="write"
    )
    restore_path = checkpoint_tradeoff(
        "edge", state_scale=5, bandwidth_scale=4, bandwidth_allocation="restore"
    )

    assert write_path["additional_bandwidth_gbs"] == restore_path["additional_bandwidth_gbs"]
    assert write_path["write_tax"] < baseline["write_tax"]
    assert write_path["recovery_s"] == baseline["recovery_s"]
    assert restore_path["write_tax"] == baseline["write_tax"]
    assert restore_path["recovery_s"] < baseline["recovery_s"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_notebook_stress_has_failed_baseline_and_opposite_equal_bandwidth_outcomes(track_id):
    baseline = checkpoint_tradeoff(track_id, state_scale=20)
    write_path = checkpoint_tradeoff(
        track_id, state_scale=20, bandwidth_scale=12, bandwidth_allocation="write"
    )
    restore_path = checkpoint_tradeoff(
        track_id, state_scale=20, bandwidth_scale=12, bandwidth_allocation="restore"
    )

    assert not baseline["rto_ok"], track_id
    assert not write_path["rto_ok"], track_id
    assert restore_path["rto_ok"], track_id
    assert write_path["additional_bandwidth_gbs"] == restore_path["additional_bandwidth_gbs"]


def test_outside_first_order_domain_returns_diagnostics_without_fake_goodput():
    outside = checkpoint_tradeoff(
        "cloud",
        interval_min=1,
        checkpoint_pause_s=120,
        detection_s=20_000,
    )

    assert not outside["loss_approximation_valid"]
    assert outside["goodput_fraction"] is None
    assert outside["total_waste_fraction"] > 0
    assert "checkpoint pause is at least the checkpoint interval" in outside["validity_issues"]


def test_young_optimum_is_withheld_when_checkpoint_cost_is_not_small_vs_mtbf():
    outside = checkpoint_tradeoff(
        "cloud",
        interval_min=300,
        checkpoint_pause_s=10_000,
    )

    assert not outside["young_approximation_valid"]
    assert outside["optimal_interval_min"] is None
    assert outside["goodput_fraction"] is None


def test_corruption_timeline_falls_back_past_newest_invalid_generation():
    result = select_restore_checkpoint(
        [
            {"step": 100, "completed": True, "digest_valid": True},
            {"step": 200, "completed": True, "digest_valid": True},
            {"step": 300, "completed": True, "digest_valid": False},
            {"step": 400, "completed": False, "digest_valid": True},
        ],
        failure_step=450,
    )

    assert result["recoverable"]
    assert result["newest_checkpoint_step"] == 400
    assert result["restore_step"] == 200
    assert result["lost_steps"] == 250
    assert result["fallback_generations"] == 2


def test_corruption_timeline_reports_no_feasible_restore():
    result = select_restore_checkpoint(
        [{"step": 100, "completed": True, "digest_valid": False}], failure_step=120
    )
    assert not result["recoverable"]
    assert result["restore_step"] is None
    assert result["lost_steps"] is None


def test_checkpoint_fault_fixture_invariant_baseline_across_faults():
    corrupt = checkpoint_fault_fixture("cloud", fault="corrupt")
    incomplete = checkpoint_fault_fixture("cloud", fault="incomplete")

    assert corrupt["baseline_records"] == incomplete["baseline_records"]
    assert corrupt["baseline"] == incomplete["baseline"]
    assert corrupt["baseline"]["restore_step"] == 400
    assert corrupt["baseline"]["lost_steps"] == 50
    assert corrupt["baseline"]["fallback_generations"] == 0
    assert [r["step"] for r in corrupt["baseline_records"]] == [100, 200, 300, 400]
    assert all(r["completed"] and r["digest_valid"] for r in corrupt["baseline_records"])


def test_checkpoint_fault_fixture_only_newest_fault_and_restores_300():
    for fault, key in [("corrupt", "digest_valid"), ("incomplete", "completed")]:
        fixture = checkpoint_fault_fixture("mobile", fault=fault)
        res_records = fixture["result_records"]
        base_records = fixture["baseline_records"]

        for idx in range(3):
            assert res_records[idx] == base_records[idx]
            assert res_records[idx]["completed"] and res_records[idx]["digest_valid"]

        assert res_records[3]["step"] == 400
        assert not res_records[3][key]
        other_key = "completed" if key == "digest_valid" else "digest_valid"
        assert res_records[3][other_key]

        assert fixture["result"]["recoverable"]
        assert fixture["result"]["restore_step"] == 300
        assert fixture["result"]["lost_steps"] == 150
        assert fixture["result"]["fallback_generations"] == 1


def test_checkpoint_fault_fixture_nonmutation():
    fixture = checkpoint_fault_fixture("tinyml", fault="corrupt")
    base_copy = deepcopy(fixture["baseline_records"])
    res_copy = deepcopy(fixture["result_records"])

    fixture["baseline_records"][0]["completed"] = False
    fixture["result_records"][0]["completed"] = False

    fresh = checkpoint_fault_fixture("tinyml", fault="corrupt")
    assert fresh["baseline_records"] == base_copy
    assert fresh["result_records"] == res_copy
    assert fresh["baseline_records"] != fresh["result_records"]


def test_checkpoint_fault_fixture_all_tracks_have_context():
    for track_id in TRACKS:
        scenario = get_track_scenario(track_id)
        fixture = checkpoint_fault_fixture(track_id, fault="corrupt")

        assert scenario["training_backend_role"]
        assert scenario["manifest_role"]
        assert scenario["manifest_context"]

        assert fixture["training_backend_role"] == scenario["training_backend_role"]
        assert fixture["manifest_role"] == scenario["manifest_role"]
        assert fixture["manifest_context"] == scenario["manifest_context"]


def test_replica_placement_models_shared_failure_domain_correlation():
    colocated = replica_availability(
        replicas=3,
        placement_domains=1,
        component_availability=0.99,
        domain_mttf_h=1_000,
        domain_repair_min=60,
    )
    spread = replica_availability(
        replicas=3,
        placement_domains=3,
        component_availability=0.99,
        domain_mttf_h=1_000,
        domain_repair_min=60,
    )

    assert colocated["shared_domain_correlation_modeled"]
    assert spread["service_availability"] > colocated["service_availability"]


def test_serving_fleet_size_changes_counts_but_not_availability_fraction():
    small = serving_capacity("mobile", fleet_size=1_000)
    large = serving_capacity("mobile", fleet_size=10_000)

    assert small["available_fraction"] == large["available_fraction"]
    assert large["expected_unavailable_devices"] == pytest.approx(
        10 * small["expected_unavailable_devices"]
    )


def test_every_evaluator_result_preserves_exact_resolved_arguments():
    exposure = training_exposure("edge", workers=17, duration_h=9)
    checkpoint = checkpoint_tradeoff("edge", interval_min=11, requested_rpo_min=12)
    restore = select_restore_checkpoint(
        [{"step": 4, "completed": True, "digest_valid": True}], failure_step=9
    )
    capacity = serving_capacity("edge", fleet_size=321)
    replicas = replica_availability(
        replicas=3,
        placement_domains=2,
        component_availability=0.98,
        domain_mttf_h=2_000,
        domain_repair_min=7,
    )

    assert exposure["inputs"]["workers"] == 17
    assert exposure["inputs"]["duration_h"] == 9
    assert checkpoint["inputs"]["interval_min"] == 11
    assert checkpoint["inputs"]["requested_rpo_min"] == 12
    assert checkpoint["inputs"]["state_gb"] is None
    assert checkpoint["inputs"]["state_scale"] == 1
    assert restore["inputs"]["failure_step"] == 9
    assert restore["inputs"]["checkpoints"][0]["step"] == 4
    assert capacity["inputs"] == {"track_id": "edge", "fleet_size": 321}
    assert replicas["inputs"]["placement_domains"] == 2


def test_serialized_inputs_replay_results_without_unit_or_default_drift():
    exposure = training_exposure("tinyml", workers=19, duration_h=12)
    checkpoint = checkpoint_tradeoff(
        "mobile", state_scale=5, bandwidth_scale=3, bandwidth_allocation="restore"
    )
    restore = select_restore_checkpoint(
        [{"step": 7, "completed": True, "digest_valid": True}], failure_step=11
    )
    capacity = serving_capacity("cloud", fleet_size=777)
    replicas = replica_availability(
        replicas=4,
        placement_domains=2,
        component_availability=0.97,
        domain_mttf_h=5_000,
        domain_repair_min=9,
    )

    assert training_exposure(**exposure["inputs"])["system_mtbf_h"] == exposure["system_mtbf_h"]
    assert checkpoint_tradeoff(**checkpoint["inputs"])["recovery_s"] == checkpoint["recovery_s"]
    assert select_restore_checkpoint(**restore["inputs"])["restore_step"] == restore["restore_step"]
    assert serving_capacity(**capacity["inputs"])["expected_available_devices"] == capacity["expected_available_devices"]
    assert replica_availability(**replicas["inputs"])["service_availability"] == replicas["service_availability"]

    for result in (exposure, checkpoint, restore, capacity, replicas):
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "call",
    [
        lambda: training_exposure("unknown"),
        lambda: training_exposure("cloud", workers=0),
        lambda: training_exposure("cloud", duration_h=math.inf),
        lambda: checkpoint_tradeoff("edge", interval_min=0),
        lambda: select_restore_checkpoint([{}], failure_step=1),
        lambda: replica_availability(
            replicas=2,
            placement_domains=3,
            component_availability=0.99,
            domain_mttf_h=1_000,
            domain_repair_min=10,
        ),
        lambda: checkpoint_fault_fixture("cloud", fault="unknown_fault"),
    ],
)
def test_invalid_inputs_fail_explicitly(call):
    with pytest.raises(ValueError):
        call()
