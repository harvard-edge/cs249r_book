"""Causal tests for the Chapter 10 inference-fleet experiments."""

import math
import json

import pytest

from mlsysim.core.units import ureg
from mlsysim.engine.v2_10_experiments import (
    FormatEvidence,
    PlacementResult,
    Request,
    ServiceModel,
    assess_quality_qualified_format,
    compare_equal_budget_placements,
    cost_snapshot,
    deserialize_quantity,
    illustrative_track_scenario,
    lifetime_cost_crossover,
    qualified_cost,
    replay_requests,
    replay_snapshot,
    serialize_quantity,
    state_admission,
    unavailable_snapshot,
)


def _request(request_id, arrival_ms, work, state_mb=1):
    return Request(
        request_id,
        arrival_ms * ureg.millisecond,
        work,
        0,
        state_mb * ureg.megabyte,
    )


def _service(*, memory_mb=100, fragmentation=0.0):
    return ServiceModel(
        1 * ureg.millisecond,
        1 * ureg.millisecond,
        0 * ureg.millisecond,
        memory_mb * ureg.megabyte,
        20 * ureg.megabyte,
        0 * ureg.megabyte,
        fragmentation,
    )


def test_lifetime_crossover_uses_the_same_horizon_and_handles_zero_demand():
    result = lifetime_cost_crossover(
        training_cost=1_000 * ureg.dollar,
        request_rate=2 / ureg.second,
        cost_per_request=0.01 * ureg.dollar,
        horizon=1 * ureg.day,
    )
    assert result.serving_cost.m_as(ureg.dollar) == pytest.approx(1_728)
    assert result.serving_to_training_ratio == pytest.approx(1.728)
    assert result.crossover_time.m_as(ureg.day) == pytest.approx(1_000 / 1_728)

    no_demand = lifetime_cost_crossover(
        training_cost=1_000 * ureg.dollar,
        request_rate=0 / ureg.second,
        cost_per_request=0.01 * ureg.dollar,
        horizon=1 * ureg.year,
    )
    assert math.isinf(no_demand.crossover_time.magnitude)
    snapshot = cost_snapshot(no_demand, inputs={"request_rate": 0})
    assert snapshot["crossover_time"] is None
    assert "Infinity" not in json.dumps(snapshot, allow_nan=False)


def test_fragmentation_can_reject_state_even_when_weights_fit():
    requests = tuple(_request(str(index), 0, 1, state_mb=20) for index in range(4))
    compact = state_admission(
        requests,
        device_memory=100 * ureg.megabyte,
        weight_memory=20 * ureg.megabyte,
    )
    fragmented = state_admission(
        requests,
        device_memory=100 * ureg.megabyte,
        weight_memory=20 * ureg.megabyte,
        fragmentation_fraction=0.30,
    )
    assert compact.max_concurrent_requests == 4
    assert compact.rejected_request_ids == ()
    assert fragmented.max_concurrent_requests == 2
    assert fragmented.rejected_request_ids == ("2", "3")


def test_batch_replay_uses_identical_arrivals_and_exposes_deadline_tradeoff():
    trace = tuple(_request(str(index), index * 3, 10) for index in range(4))
    immediate = replay_requests(
        trace,
        service=_service(),
        replicas=1,
        policy="immediate",
        slo=20 * ureg.millisecond,
    )
    batched = replay_requests(
        trace,
        service=_service(),
        replicas=1,
        policy="windowed_batch",
        max_batch=4,
        batch_window=25 * ureg.millisecond,
        slo=20 * ureg.millisecond,
    )
    assert [o.arrival for o in immediate.outcomes] == [o.arrival for o in batched.outcomes]
    assert batched.makespan < immediate.makespan
    assert batched.outcomes[0].duration > immediate.outcomes[0].duration
    assert batched.on_time_count < immediate.on_time_count
    assert batched.request_count == batched.completed_count + batched.rejected_count


def test_p99_is_empirical_percentile_of_complete_request_durations():
    trace = (_request("short", 0, 1), _request("long", 100, 99))
    result = replay_requests(
        trace,
        service=_service(),
        replicas=1,
        policy="immediate",
        slo=1_000 * ureg.millisecond,
    )
    complete_ms = [duration.m_as(ureg.millisecond) for duration in result.completed_durations]
    assert complete_ms == pytest.approx([2, 100])
    assert result.p99_duration.m_as(ureg.millisecond) == pytest.approx(100)
    assert result.p99_method == "empirical nearest-rank over complete request durations"


def test_request_id_is_noncausal_for_timing():
    service = _service()
    first = replay_requests(
        (_request("a", 0, 5), _request("b", 20, 5)),
        service=service,
        replicas=1,
        policy="immediate",
        slo=100 * ureg.millisecond,
    )
    renamed = replay_requests(
        (_request("x", 0, 5), _request("y", 20, 5)),
        service=service,
        replicas=1,
        policy="immediate",
        slo=100 * ureg.millisecond,
    )
    assert first.completed_durations == renamed.completed_durations
    assert first.p99_duration == renamed.p99_duration


def test_equal_budget_placement_accounts_for_handoff_bytes_and_link_time():
    trace = tuple(_request(str(index), 0, 40) for index in range(4))
    fast = compare_equal_budget_placements(
        trace,
        service=_service(memory_mb=1_000),
        device_budget=4,
        slo=200 * ureg.millisecond,
        handoff_bytes=1 * ureg.megabyte,
        link_bandwidth=10 * ureg.gigabyte / ureg.second,
        link_latency=0.1 * ureg.millisecond,
    )
    slow = compare_equal_budget_placements(
        trace,
        service=_service(memory_mb=1_000),
        device_budget=4,
        slo=200 * ureg.millisecond,
        handoff_bytes=100 * ureg.megabyte,
        link_bandwidth=1 * ureg.gigabyte / ureg.second,
        link_latency=1 * ureg.millisecond,
    )
    by_name_fast = {result.placement: result for result in fast}
    by_name_slow = {result.placement: result for result in slow}
    assert by_name_fast["replicate"].replay.p99_duration == by_name_slow["replicate"].replay.p99_duration
    assert by_name_slow["shard"].replay.p99_duration > by_name_fast["shard"].replay.p99_duration
    assert by_name_slow["split_phases"].replay.p99_duration > by_name_fast["split_phases"].replay.p99_duration
    assert {result.device_budget for result in fast} == {4}


def test_sharding_can_admit_weights_that_replication_cannot_fit():
    trace = (_request("large-model-request", 0, 1),)
    service = ServiceModel(
        1 * ureg.millisecond,
        1 * ureg.millisecond,
        0 * ureg.millisecond,
        100 * ureg.megabyte,
        150 * ureg.megabyte,
        0 * ureg.megabyte,
    )
    placements = compare_equal_budget_placements(
        trace,
        service=service,
        device_budget=4,
        slo=100 * ureg.millisecond,
        handoff_bytes=1 * ureg.megabyte,
        link_bandwidth=10 * ureg.gigabyte / ureg.second,
        link_latency=0.1 * ureg.millisecond,
    )
    by_name = {result.placement: result for result in placements}
    assert by_name["replicate"].replay.rejected_count == 1
    assert by_name["shard"].replay.completed_count == 1


def test_warmup_and_replica_loss_reduce_real_capacity():
    trace = tuple(_request(str(index), 0, 20) for index in range(8))
    baseline = replay_requests(
        trace,
        service=_service(memory_mb=1_000),
        replicas=4,
        policy="immediate",
        slo=60 * ureg.millisecond,
    )
    stressed = replay_requests(
        trace,
        service=_service(memory_mb=1_000),
        replicas=4,
        lost_replicas=2,
        warmup=30 * ureg.millisecond,
        policy="immediate",
        slo=60 * ureg.millisecond,
    )
    assert stressed.makespan > baseline.makespan
    assert stressed.on_time_count < baseline.on_time_count


def test_cost_is_divided_by_deadline_compliant_work_without_hiding_rejections():
    trace = (_request("fits", 0, 1, state_mb=10), _request("rejected", 0, 1, state_mb=90))
    replay = replay_requests(
        trace,
        service=_service(memory_mb=100),
        replicas=1,
        policy="windowed_batch",
        max_batch=2,
        slo=10 * ureg.millisecond,
    )
    result = qualified_cost(total_cost=12 * ureg.dollar, replay=replay)
    assert result.request_count == 2
    assert result.completed_count == 1
    assert result.rejected_count == 1
    assert result.deadline_compliant_count == 1
    assert result.cost_per_deadline_compliant_request.m_as(ureg.dollar) == 12


@pytest.mark.parametrize("track_id", ["tinyml", "mobile", "edge", "cloud"])
def test_all_track_fixtures_are_compatible_and_replayable(track_id):
    scenario = illustrative_track_scenario(track_id)
    result = replay_requests(
        scenario.requests,
        service=scenario.service,
        replicas=2,
        policy="immediate",
        slo=scenario.slo,
    )
    assert result.completed_count > 0
    assert result.p99_duration.check("[time]")
    if track_id == "cloud":
        assert scenario.state_kind == "KV cache"
    else:
        assert "KV" not in scenario.state_kind


@pytest.mark.parametrize("track_id", ["tinyml", "mobile", "edge", "cloud"])
def test_each_track_has_valid_and_fragmentation_failed_admission(track_id):
    scenario = illustrative_track_scenario(track_id)
    compact = state_admission(
        scenario.requests,
        device_memory=scenario.service.device_memory,
        weight_memory=scenario.service.weight_memory,
        reserved_memory=scenario.service.reserved_memory,
    )
    fragmented = state_admission(
        scenario.requests,
        device_memory=scenario.service.device_memory,
        weight_memory=scenario.service.weight_memory,
        reserved_memory=scenario.service.reserved_memory,
        fragmentation_fraction=0.30,
    )
    assert len(compact.admitted_request_ids) == len(scenario.requests)
    assert fragmented.rejected_request_ids


def test_invalid_units_and_total_replica_loss_fail_loudly():
    with pytest.raises(TypeError, match="Pint Quantity"):
        Request("bad", 0, 1, 0, 1 * ureg.megabyte)

    with pytest.raises(ValueError, match="survive"):
        replay_requests(
            (_request("a", 0, 1),),
            service=_service(),
            replicas=2,
            lost_replicas=2,
            policy="immediate",
            slo=10 * ureg.millisecond,
        )


def test_evaluator_arguments_round_trip_quantities_without_unit_loss():
    scenario = illustrative_track_scenario("cloud")
    request_args = scenario.requests[0].to_evaluator_args()
    service_args = scenario.service.to_evaluator_args()
    restored_arrival = deserialize_quantity(request_args["arrival"])
    restored_memory = deserialize_quantity(service_args["device_memory"])
    assert restored_arrival == scenario.requests[0].arrival
    assert restored_memory == scenario.service.device_memory
    assert serialize_quantity(restored_memory)["unit"] == service_args["device_memory"]["unit"]


def test_quality_threshold_changes_acceptance_but_not_physical_replay():
    scenario = illustrative_track_scenario("mobile")
    replay_before = replay_requests(
        scenario.requests,
        service=scenario.service,
        replicas=2,
        policy="immediate",
        slo=scenario.slo,
    )
    evidence = FormatEvidence(
        "int8",
        "mobile-camera",
        "daylight-validation",
        0.91,
        "illustrative classroom fixture",
    )
    accepted = assess_quality_qualified_format(
        evidence,
        task_id="mobile-camera",
        population_id="daylight-validation",
        required_outcome_rate=0.90,
    )
    rejected = assess_quality_qualified_format(
        evidence,
        task_id="mobile-camera",
        population_id="daylight-validation",
        required_outcome_rate=0.95,
    )
    replay_after = replay_requests(
        scenario.requests,
        service=scenario.service,
        replicas=2,
        policy="immediate",
        slo=scenario.slo,
    )
    assert accepted.accepted is True
    assert rejected.accepted is False
    assert replay_before.completed_durations == replay_after.completed_durations


def test_quality_evidence_cannot_cross_tasks_or_populations():
    evidence = FormatEvidence("int8", "task-a", "population-a", 0.9, "fixture")
    with pytest.raises(ValueError, match="match both"):
        assess_quality_qualified_format(
            evidence,
            task_id="task-b",
            population_id="population-a",
            required_outcome_rate=0.8,
        )


def test_zero_completion_failure_snapshot_is_strict_finite_json():
    request = _request("oom", 0, 1, state_mb=90)
    result = replay_requests(
        (request,),
        service=_service(memory_mb=100),
        replicas=1,
        policy="immediate",
        slo=10 * ureg.millisecond,
    )
    snapshot = replay_snapshot(result, inputs={"case": "explicit OOM"})
    encoded = json.dumps(snapshot, allow_nan=False)
    assert result.completed_count == 0
    assert result.rejected_count == 1
    assert snapshot["p99_duration"] is None
    assert snapshot["p99_status"] == "unavailable: no completed requests"
    assert "Infinity" not in encoded


@pytest.mark.parametrize("track_id", ["tinyml", "mobile", "edge", "cloud"])
def test_all_tracks_have_rooted_backend_roles_and_applicability_distinctions(track_id):
    scenario = illustrative_track_scenario(track_id)
    assert scenario.backend_role
    assert scenario.startup_kind
    if track_id in ("tinyml", "mobile"):
        assert "wake" in scenario.startup_kind or "activation" in scenario.startup_kind
        assert scenario.supported_placements == ("replicate",)
    elif track_id == "edge":
        assert "provision" in scenario.startup_kind or "container" in scenario.startup_kind
        assert scenario.supported_placements == ("replicate", "shard")
    else:
        assert "provision" in scenario.startup_kind or "cold start" in scenario.startup_kind
        assert scenario.supported_placements == ("replicate", "shard", "split_phases")

    placements = compare_equal_budget_placements(
        scenario.requests,
        service=scenario.service,
        device_budget=scenario.device_budget,
        slo=scenario.slo,
        handoff_bytes=scenario.handoff_bytes,
        link_bandwidth=scenario.link_bandwidth,
        link_latency=scenario.link_latency,
        track_id=scenario.track_id,
    )
    by_name = {p.placement: p for p in placements}
    assert by_name["replicate"].available is True
    assert by_name["replicate"].replay is not None

    if track_id in ("tinyml", "mobile"):
        assert by_name["shard"].available is False
        assert by_name["shard"].replay is None
        assert "endpoint" in by_name["shard"].unsupported_reason
        assert by_name["split_phases"].available is False
        assert by_name["split_phases"].replay is None
        assert "endpoint" in by_name["split_phases"].unsupported_reason
    elif track_id == "edge":
        assert by_name["shard"].available is True
        assert by_name["shard"].replay is not None
        assert by_name["split_phases"].available is False
        assert by_name["split_phases"].replay is None
        assert "phase" in by_name["split_phases"].unsupported_reason
    else:
        assert by_name["shard"].available is True
        assert by_name["shard"].replay is not None
        assert by_name["split_phases"].available is True
        assert by_name["split_phases"].replay is not None


def test_unavailable_placement_snapshot_is_strict_finite_json_with_none_metrics():
    scenario = illustrative_track_scenario("tinyml")
    placements = compare_equal_budget_placements(
        scenario.requests,
        service=scenario.service,
        device_budget=scenario.device_budget,
        slo=scenario.slo,
        handoff_bytes=scenario.handoff_bytes,
        link_bandwidth=scenario.link_bandwidth,
        link_latency=scenario.link_latency,
        track_id="tinyml",
    )
    by_name = {p.placement: p for p in placements}
    unsupported = by_name["shard"]
    assert unsupported.available is False

    evaluator_inputs = {"placement": "shard", "device_budget": scenario.device_budget}
    snapshot = replay_snapshot(
        unsupported.replay,
        inputs=evaluator_inputs,
        available=unsupported.available,
        unsupported_reason=unsupported.unsupported_reason,
    )
    encoded = json.dumps(snapshot, allow_nan=False)

    assert snapshot["available"] is False
    assert snapshot["request_count"] is None
    assert snapshot["completed_count"] is None
    assert snapshot["rejected_count"] is None
    assert snapshot["deadline_compliant_count"] is None
    assert snapshot["p99_duration"] is None
    assert snapshot["makespan"] is None
    assert snapshot["completed_durations"] == []
    assert "unavailable" in snapshot["p99_status"]
    assert "Infinity" not in encoded
    assert "NaN" not in encoded
    assert snapshot["inputs"] == evaluator_inputs
