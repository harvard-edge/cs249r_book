"""Capacity, exact-state recovery, and replay invariants for the Chapter 14 studies."""
from copy import deepcopy
import json

import pytest

from mlsysim.labs.experiments.v1_14_experiments import (
    TRACKS, canary_experiment, experiment_options, release_capacity_profile,
    release_recovery_experiment, replay_experiment, rollout_experiment,
)
from mlsysim.labs.experiments.v1_16_experiments import replay_capture


def response(track="tinyml", **changes):
    inputs = dict(track_id=track, proxy_threshold=0.15,
                  sampling_interval_hours=experiment_options(track)["step_hours"],
                  rollback_minutes=5, affected_fraction=0.25,
                  restore="complete_bundle", inflight="drain_pinned", logs="quarantine")
    return release_recovery_experiment(**{**inputs, **changes})


@pytest.mark.parametrize("track", TRACKS)
def test_nonzero_traffic_share_never_shrinks_candidate_copy(track):
    small = rollout_experiment(track, canary_fraction=0.10)
    large = rollout_experiment(track, canary_fraction=0.50)
    assert small["required_bytes"] == large["required_bytes"]
    assert small["resident_copies"] == large["resident_copies"] == 2
    assert not small["capacity_fits"] and not large["capacity_fits"]
    assert small["candidate_labeled"] == large["candidate_labeled"] == 0
    assert small["exposed_requests"] == large["exposed_requests"] == 0
    assert small["decision"] == "hold_capacity"
    assert not small["evidence_requirements_met"]


@pytest.mark.parametrize("track", TRACKS)
def test_wait_recovers_capacity_but_loses_mature_labels_by_same_deadline(track):
    immediate = rollout_experiment(track, canary_fraction=0.25)
    waited = rollout_experiment(track, canary_fraction=0.25, reservation_plan="wait")
    full_trace = canary_experiment(track, canary_fraction=0.25, stop_on_decision=False)
    assert waited["capacity_fits"]
    assert waited["launch_delay_hours"] > immediate["launch_delay_hours"]
    assert waited["observation_horizon_hours"] == immediate["observation_horizon_hours"]
    assert 0 < waited["candidate_labeled"] < full_trace["candidate_labeled"]
    assert waited["required_bytes"] == immediate["required_bytes"]
    assert waited["candidate_labeled"] + waited["baseline_labeled"] < full_trace["candidate_labeled"] + full_trace["baseline_labeled"]


@pytest.mark.parametrize("track", TRACKS)
def test_holding_baseline_installs_no_candidate(track):
    held = rollout_experiment(track, canary_fraction=0)
    assert held["resident_copies"] == 1
    assert held["capacity_fits"]
    assert held["candidate_labeled"] == 0
    waited_hold = rollout_experiment(track, canary_fraction=0, reservation_plan="wait")
    assert waited_hold["launch_delay_hours"] == 0
    assert waited_hold["available_bytes"] == held["available_bytes"]
    allocation = held["allocation"]
    assert allocation["pool_bytes"] + allocation["protected_bytes"] == pytest.approx(allocation["physical_bytes"])
    assert allocation["copy_bytes"] <= allocation["pool_bytes"] <= allocation["physical_bytes"]


def test_tiny_allocation_is_flash_and_declared_not_inference_sram():
    assert release_capacity_profile("tinyml")["resource"] == "flash banks"


@pytest.mark.parametrize("track", TRACKS)
def test_weights_only_and_transform_only_restoration_leave_distinct_breaks(track):
    weights = response(track, restore="weights_only")
    transform = response(track, restore="weights_transform")
    assert not weights["requests"][0]["component_checks"]["transform"]
    assert transform["requests"][0]["component_checks"]["transform"]
    assert not transform["requests"][0]["component_checks"]["cache_namespace"]
    assert not weights["verified_recovery"] and not transform["verified_recovery"]
    assert weights["mean_time_to_recovery_hours"] is None


@pytest.mark.parametrize("track", TRACKS)
def test_new_request_repair_does_not_relabel_an_admitted_candidate_request(track):
    mutated = response(track, inflight="reuse_mutated")
    assert mutated["requests"][0]["compatible"]
    assert not mutated["requests"][1]["compatible"]
    assert mutated["requests"][1]["expected_bundle"] == mutated["candidate_bundle"]
    assert not mutated["verified_recovery"]


@pytest.mark.parametrize("track", TRACKS)
def test_pinned_drain_and_confirmed_cancel_have_explicit_different_costs(track):
    drain = response(track)
    cancel = response(track, inflight="cancel_confirmed")
    assert drain["verified_recovery"] and cancel["verified_recovery"]
    assert drain["retained_candidate_copies"] > cancel["retained_candidate_copies"]
    assert drain["additional_drain_hours"] > cancel["additional_drain_hours"]
    assert drain["mean_time_to_recovery_hours"] > cancel["mean_time_to_recovery_hours"]
    assert cancel["confirmed_cancellations"] > drain["confirmed_cancellations"]
    assert cancel["requests"][1]["compatible"] is None
    assert not cancel["requests"][1]["completed"]


@pytest.mark.parametrize("track", TRACKS)
def test_compatibility_and_log_quarantine_and_cause_are_separate_gates(track):
    dirty = response(track, logs="retain")
    shifted = response(track, incident_cause="population_shift")
    assert dirty["serving_contract_restored"] and not dirty["verified_recovery"]
    assert shifted["serving_contract_restored"] and shifted["degraded_logs_quarantined"]
    assert not shifted["cause_repaired"] and not shifted["verified_recovery"]
    assert shifted["recovered_quality_pct"] is None
    assert shifted["total_incident_cost_usd"] is None


@pytest.mark.parametrize("track", TRACKS)
def test_faster_weights_only_operation_does_not_repair_state(track):
    fast = response(track, restore="weights_only", rollback_minutes=2)
    slow = response(track, restore="weights_only", rollback_minutes=30)
    assert fast["timing_reference"]["rollback_completed_hours"] < slow["timing_reference"]["rollback_completed_hours"]
    assert fast["requests"] == slow["requests"]
    assert not fast["verified_recovery"] and not slow["verified_recovery"]


@pytest.mark.parametrize("track", TRACKS)
def test_no_detection_does_not_become_verified_from_policy_selection(track):
    missing = response(track, proxy_threshold=1)
    assert missing["detected_hours"] is None
    assert not missing["serving_contract_restored"] and not missing["verified_recovery"]


@pytest.mark.parametrize("track", TRACKS)
def test_json_replay_and_capstone_dispatch_preserve_new_state(track):
    for key, baseline, result in (
        ("v1_14.rollout", rollout_experiment(track, canary_fraction=0.25),
         rollout_experiment(track, canary_fraction=0.25, reservation_plan="wait")),
        ("v1_14.release_recovery", response(track, restore="weights_only"), response(track)),
    ):
        restored = json.loads(json.dumps(result))
        assert replay_experiment(key, restored["inputs"]) == restored
        capture = {"track": track, "model_key": key, "baseline": baseline, "result": restored, "alternatives": ()}
        replayed = replay_capture("v1_14_experiments", capture)
        assert replayed["result"] == restored
        changed = deepcopy(restored)
        changed["inputs"]["track_id"] = "invalid"
        with pytest.raises(ValueError):
            replay_experiment(key, changed["inputs"])


def test_unknown_dispositions_are_not_silent_safe_defaults():
    with pytest.raises(ValueError):
        rollout_experiment("tinyml", canary_fraction=.25, reservation_plan="instant_free_space")
    for key in ("restore", "inflight", "logs", "incident_cause"):
        with pytest.raises(ValueError):
            response(**{key: "unknown"})
