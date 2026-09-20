"""Causal and boundary tests for the Volume I serving experiments."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v1_13_experiments import (
    ArrivalPhase,
    MODEL_ID,
    REPLAY_METHOD,
    ServingPolicy,
    arrival_trace_for_track,
    batching_policy_for_track,
    get_serving_scenario,
    make_arrival_trace,
    policy_for_track,
    replay_inputs_from_snapshot,
    replay_inputs_to_snapshot,
    replay_serving_snapshot,
    simulation_result_to_snapshot,
    simulate_serving,
    state_capacity,
    state_capacity_to_snapshot,
    state_pressure_scenario,
)


TRACKS = ("tinyml", "mobile", "edge", "cloud")


@pytest.mark.parametrize("track", TRACKS)
def test_all_track_endpoints_conserve_offered_requests(track):
    scenario = get_serving_scenario(track)
    arrivals = arrival_trace_for_track(track, "bursty")
    result = simulate_serving(scenario, arrivals, policy_for_track(track, "admit"))

    assert result.offered_count == len(arrivals)
    assert result.admitted_count + result.rejected_count == result.offered_count
    assert result.deadline_met_count + result.deadline_missed_count == result.completed_count
    assert result.completed_count == result.admitted_count
    assert result.p50_latency is not None
    assert result.p99_latency is not None
    assert result.p99_latency >= result.p50_latency
    assert 0 <= result.useful_completion_fraction <= 1
    assert result.inputs.scenario.track == track
    assert result.inputs.arrivals == arrivals


@pytest.mark.parametrize("track", TRACKS)
def test_kv_state_is_used_only_for_compatible_language_track(track):
    scenario = get_serving_scenario(track)
    capacity = state_capacity(scenario)

    assert capacity.feasible
    assert capacity.max_requests_per_replica >= 1
    if track == "cloud":
        assert capacity.kv_state_per_request.to("MiB").magnitude > 0
    else:
        assert capacity.kv_state_per_request.to("byte").magnitude == 0


def test_timeout_dispatches_sparse_requests_and_waits_are_decomposed():
    scenario = get_serving_scenario("mobile")
    arrivals = (Q_(0, "ms"), Q_(200, "ms"))
    policy = ServingPolicy(
        name="sparse timeout",
        batch_limit=4,
        batch_timeout=Q_(12, "ms"),
        reserved_replicas=1,
    )

    result = simulate_serving(scenario, arrivals, policy)

    assert len(result.batches) == 2
    for record in result.records:
        assert record.formation_wait.to("ms").magnitude == pytest.approx(12)
        assert record.resource_queue_wait.to("ms").magnitude == pytest.approx(0)
        assert record.dispatch - record.ready == record.formation_wait + record.resource_queue_wait


def test_empirical_percentile_uses_complete_request_duration():
    scenario = get_serving_scenario("tinyml")
    policy = ServingPolicy("single", batch_limit=1, batch_timeout=Q_(0, "ms"), reserved_replicas=1)

    result = simulate_serving(scenario, (Q_(0, "ms"),), policy)

    expected = (
        scenario.ingress_latency
        + scenario.preprocessing_paths[0].latency
        + scenario.execution_latency
        + scenario.postprocess_latency
    ).to("ms")
    assert result.records[0].total_latency == expected
    assert result.p50_latency == expected
    assert result.p99_latency == expected


def test_busy_server_produces_nonnegative_resource_queue_wait():
    scenario = get_serving_scenario("edge")
    arrivals = (Q_(0, "ms"), Q_(0, "ms"), Q_(0, "ms"))
    policy = ServingPolicy(
        name="one-at-a-time",
        batch_limit=1,
        batch_timeout=Q_(0, "ms"),
        reserved_replicas=1,
    )

    result = simulate_serving(scenario, arrivals, policy)

    waits = [record.resource_queue_wait.to("ms").magnitude for record in result.records]
    assert waits[0] == pytest.approx(0)
    assert waits[1] > 0
    assert waits[2] > waits[1]
    for record in result.records:
        assert record.formation_wait.to("ms").magnitude >= 0
        assert record.resource_queue_wait.to("ms").magnitude >= 0
        assert record.dispatch - record.ready == record.formation_wait + record.resource_queue_wait


def test_batching_trades_sparse_tail_delay_for_dense_execution_efficiency():
    scenario = get_serving_scenario("cloud")
    single = ServingPolicy("single", batch_limit=1, batch_timeout=Q_(0, "ms"), reserved_replicas=1)
    batched = ServingPolicy("batched", batch_limit=4, batch_timeout=Q_(20, "ms"), reserved_replicas=1)

    sparse = (Q_(0, "ms"), Q_(100, "ms"), Q_(200, "ms"), Q_(300, "ms"))
    sparse_single = simulate_serving(scenario, sparse, single)
    sparse_batched = simulate_serving(scenario, sparse, batched)
    assert sparse_batched.p99_latency > sparse_single.p99_latency

    dense = tuple(Q_(0, "ms") for _ in range(8))
    dense_single = simulate_serving(scenario, dense, single)
    dense_batched = simulate_serving(scenario, dense, batched)
    assert dense_batched.busy_replica_time < dense_single.busy_replica_time
    assert len(dense_batched.batches) < len(dense_single.batches)


def test_reservation_reduces_shift_tail_while_activation_saves_replica_time():
    scenario = get_serving_scenario("cloud")
    arrivals = arrival_trace_for_track("cloud", "shift")
    reserved = simulate_serving(scenario, arrivals, policy_for_track("cloud", "reserve"))
    activated = simulate_serving(scenario, arrivals, policy_for_track("cloud", "activate"))

    assert reserved.p99_latency < activated.p99_latency
    assert activated.provisioned_replica_time < reserved.provisioned_replica_time


def test_admission_loss_is_separate_and_offered_requests_stay_in_denominator():
    scenario = get_serving_scenario("cloud")
    arrivals = tuple(Q_(0, "ms") for _ in range(20))
    policy = ServingPolicy(
        name="bounded admission",
        batch_limit=1,
        batch_timeout=Q_(0, "ms"),
        reserved_replicas=1,
        queue_capacity=3,
    )

    result = simulate_serving(scenario, arrivals, policy)

    assert result.rejected_count == 17
    assert result.completed_count == 3
    assert result.p99_latency is not None
    assert result.useful_completion_fraction == pytest.approx(
        result.useful_completion_count / result.offered_count
    )
    assert all(record.rejection_reason == "admission queue full" for record in result.records if not record.admitted)


def test_preprocessing_parity_changes_usefulness_without_invented_quality_score():
    scenario = get_serving_scenario("tinyml")
    arrivals = arrival_trace_for_track("tinyml", "steady")[:8]
    policy = policy_for_track("tinyml", "baseline")
    matched = simulate_serving(scenario, arrivals, policy, scenario.preprocessing_paths[0])
    mismatched = simulate_serving(scenario, arrivals, policy, scenario.preprocessing_paths[1])

    assert matched.preprocessing_parity
    assert matched.useful_completion_count == matched.deadline_met_count
    assert not mismatched.preprocessing_parity
    assert mismatched.useful_completion_count == 0


def test_noncausal_scenario_note_cannot_change_any_outcome():
    scenario = get_serving_scenario("edge")
    renamed_note = replace(scenario, assumption_note="Same numerical scenario, different explanatory note.")
    arrivals = arrival_trace_for_track("edge", "steady")
    policy = policy_for_track("edge", "baseline")

    original = simulate_serving(scenario, arrivals, policy)
    changed = simulate_serving(renamed_note, arrivals, policy)

    assert original.records == changed.records
    assert original.batches == changed.batches
    assert original.p99_latency == changed.p99_latency
    assert original.useful_completion_fraction == changed.useful_completion_fraction


def test_replay_inputs_are_complete_and_reproduce_results():
    scenario = get_serving_scenario("mobile")
    result = simulate_serving(
        scenario,
        arrival_trace_for_track("mobile", "bursty"),
        policy_for_track("mobile", "activate"),
        scenario.preprocessing_paths[0],
    )

    replay = simulate_serving(
        result.inputs.scenario,
        result.inputs.arrivals,
        result.inputs.policy,
        result.inputs.preprocessing,
    )

    assert replay.records == result.records
    assert replay.batches == result.batches
    assert replay.p99_latency == result.p99_latency


def test_snapshot_inputs_survive_actual_json_roundtrip_and_replay():
    scenario = get_serving_scenario("cloud")
    result = simulate_serving(
        scenario,
        arrival_trace_for_track("cloud", "shift"),
        policy_for_track("cloud", "activate"),
        scenario.preprocessing_paths[1],
    )

    snapshot = replay_inputs_to_snapshot(result.inputs)
    decoded = json.loads(json.dumps(snapshot))
    restored = replay_inputs_from_snapshot(decoded)
    replay = replay_serving_snapshot(decoded)

    assert MODEL_ID == "v1_13_experiments"
    assert tuple(decoded) == (REPLAY_METHOD,)
    assert restored.scenario == result.inputs.scenario
    assert restored.arrivals == result.inputs.arrivals
    assert restored.policy == result.inputs.policy
    assert restored.preprocessing == result.inputs.preprocessing
    assert replay.records == result.records
    assert replay.batches == result.batches
    assert replay.preprocessing_parity == result.preprocessing_parity
    assert replay.useful_completion_fraction == result.useful_completion_fraction

    result_snapshot = json.loads(json.dumps(simulation_result_to_snapshot(result)))
    stages = result_snapshot["mean_stage_latency_ms"]
    assert sum(stages.values()) == pytest.approx(result_snapshot["mean_latency_ms"])


def test_snapshot_restore_rejects_missing_method_payload():
    with pytest.raises(ValueError, match=REPLAY_METHOD):
        replay_inputs_from_snapshot({})


def test_failed_state_requirement_rejects_every_offered_request_explicitly():
    scenario = replace(
        get_serving_scenario("mobile"),
        resident_model_memory=Q_(900, "MiB"),
    )
    arrivals = (Q_(0, "ms"), Q_(10, "ms"))

    result = simulate_serving(scenario, arrivals, policy_for_track("mobile", "baseline"))

    assert not state_capacity(scenario).feasible
    assert result.completed_count == 0
    assert result.rejected_count == result.offered_count
    assert result.p99_latency is None
    failed_snapshot = simulation_result_to_snapshot(result)
    assert failed_snapshot["p99_latency_ms"] is None
    assert failed_snapshot["latency_status"] == "unavailable_no_completed_requests"
    json.dumps(failed_snapshot, allow_nan=False)
    assert {record.rejection_reason for record in result.records} == {"state capacity"}


@pytest.mark.parametrize("track", TRACKS)
def test_state_pressure_reduces_capacity_and_overflow_fails(track):
    baseline = state_capacity(state_pressure_scenario(track, "baseline"))
    elevated = state_capacity(state_pressure_scenario(track, "elevated"))
    overflow = state_capacity(state_pressure_scenario(track, "overflow"))

    assert elevated.max_requests_per_replica < baseline.max_requests_per_replica
    assert not overflow.feasible
    assert overflow.max_requests_per_replica == 0
    json.loads(json.dumps(state_capacity_to_snapshot(overflow, pressure="overflow")))


@pytest.mark.parametrize("track", TRACKS)
def test_batching_policy_endpoints_change_actual_schedule(track):
    scenario = get_serving_scenario(track)
    arrivals = arrival_trace_for_track(track, "bursty")
    single = simulate_serving(scenario, arrivals, batching_policy_for_track(track, "single"))
    batched = simulate_serving(scenario, arrivals, batching_policy_for_track(track, "quad-patient"))

    assert len(batched.batches) <= len(single.batches)
    assert batched.busy_replica_time <= single.busy_replica_time
    assert batched.batches[0].dispatch > single.batches[0].dispatch


@pytest.mark.parametrize("track", TRACKS)
def test_arrival_rate_scale_changes_offered_load_for_every_track(track):
    baseline = arrival_trace_for_track(track, "shift", rate_scale=1.0)
    stressed = arrival_trace_for_track(track, "shift", rate_scale=2.0)

    assert len(stressed) > len(baseline)


@pytest.mark.parametrize(
    "phases",
    [
        (ArrivalPhase(Q_(0, "s"), 1),),
        (ArrivalPhase(Q_(1, "s"), -1),),
    ],
)
def test_invalid_arrival_phases_fail(phases):
    with pytest.raises(ValueError):
        make_arrival_trace(phases)


@pytest.mark.parametrize(
    "policy",
    [
        ServingPolicy("bad batch", 0, Q_(0, "ms"), 1),
        ServingPolicy("bad replicas", 1, Q_(0, "ms"), 0),
        ServingPolicy("bad queue", 1, Q_(0, "ms"), 1, queue_capacity=0),
    ],
)
def test_invalid_policy_requirements_fail(policy):
    with pytest.raises(ValueError):
        simulate_serving(get_serving_scenario("tinyml"), (Q_(0, "ms"),), policy)


def test_all_tracks_have_explicit_serving_roles():
    roles = {}
    for track in TRACKS:
        scenario = get_serving_scenario(track)
        assert bool(scenario.serving_role)
        roles[track] = scenario.serving_role

    assert "microcontroller" in roles["tinyml"].lower()
    assert "application" in roles["mobile"].lower() or "mobile" in roles["mobile"].lower()
    assert "gateway" in roles["edge"].lower()
    assert "cloud" in roles["cloud"].lower() or "service" in roles["cloud"].lower()


def test_state_capacity_snapshot_formats_units_without_truncation():
    tinyml_snap = state_capacity_to_snapshot(
        state_capacity(get_serving_scenario("tinyml")),
        pressure="baseline",
    )
    assert "KiB" in tinyml_snap["state_per_request_display"]
    assert tinyml_snap["state_per_request_display"] != "0.0 MiB"
    assert tinyml_snap["kv_state_per_request_display"] == "N/A"

    cloud_snap = state_capacity_to_snapshot(
        state_capacity(get_serving_scenario("cloud")),
        pressure="baseline",
    )
    assert "MiB" in cloud_snap["kv_state_per_request_display"]
    assert cloud_snap["kv_state_per_request_display"] != "N/A"
