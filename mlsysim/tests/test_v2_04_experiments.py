import json
import math

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v2_04_experiments import (
    AccessEvent,
    TRACKS,
    TIER_PLANS,
    deserialize_evidence_inputs,
    evaluate_delivery,
    evaluate_layout,
    evaluate_storage_cost,
    evaluate_tier_plan,
    evaluate_track_cache,
    evaluate_track_checkpoint,
    evaluate_track_demand,
    evaluate_track_layout,
    make_access_trace,
    make_miss_burst_trace,
    serialize_for_evidence,
    simulate_cache_prefetch,
    simulate_checkpoint_publication,
    track_profile,
)


def test_all_tracks_represent_fleets_and_cross_a_starvation_boundary():
    for track_id, profile in TRACKS.items():
        baseline = evaluate_delivery(track_id)
        stressed = evaluate_delivery(track_id, consumers=profile["consumers"] * 3)

        assert "fleet" in baseline["fleet_shape"] or "pool" in baseline["fleet_shape"]
        assert baseline["consumer_unit"] in {"backend accelerator", "training accelerator"}
        assert not baseline["starved"], track_id
        assert baseline["accelerator_utilization"] == pytest.approx(1.0)
        assert stressed["starved"], track_id
        assert stressed["accelerator_utilization"] < 0.55


def test_delivery_equations_expose_each_service_rate_and_bottleneck():
    result = evaluate_delivery("edge", target_utilization=0.9, files_per_batch=4)
    batch_rate = result["required_batches_per_second"]

    assert result["required_bandwidth"].to("GB/s").magnitude == pytest.approx(
        (batch_rate * result["batch_bytes"] / Q_(1, "count")).to("GB/s").magnitude
    )
    assert set(result["service_rates"]) == {"storage", "metadata", "preprocess"}
    assert result["delivered_batches_per_second"] == result["service_rates"][result["bottleneck"]]


def test_track_demand_wrapper_scales_consumers_and_records_replay_arguments():
    result = evaluate_track_demand("edge", consumer_scale=2.0, target_utilization=0.9)

    assert result["consumers"] == 2 * TRACKS["edge"]["consumers"]
    assert result["inputs"] == {
        "track_id": "edge", "consumer_scale": 2.0, "target_utilization": 0.9
    }


def test_metadata_pressure_does_not_change_byte_demand_for_the_same_batches():
    one_file = evaluate_delivery("mobile", files_per_batch=1)
    many_files = evaluate_delivery("mobile", files_per_batch=40)

    assert many_files["required_bandwidth"] == one_file["required_bandwidth"]
    assert many_files["required_requests_per_second"] == 40 * one_file["required_requests_per_second"]
    assert many_files["accelerator_utilization"] < one_file["accelerator_utilization"]
    assert many_files["bottleneck"] == "metadata"


def test_cache_trace_can_have_high_hit_rate_and_nonzero_idle_time():
    trace = make_access_trace(accesses=24, working_set=3, spacing=Q_(20, "ms"))
    result = simulate_cache_prefetch(trace, cache_capacity=Q_(64, "MB"), prefetch_depth=1)

    assert result["hit_rate"] > 0.85
    assert result["cache_misses"] == 3
    assert result["total_stall"].to("ms").magnitude > 0
    assert {row["source"] for row in result["events"]} >= {"cache_hit", "prefetch"}


def test_correlated_miss_burst_defeats_finite_prefetch():
    def trace(spacing_ms):
        events = [AccessEvent(Q_(index * 20, "ms"), "hot", Q_(8, "MB")) for index in range(12)]
        start = Q_(240, "ms")
        events.extend(
            AccessEvent(start + Q_(offset * spacing_ms, "ms"), f"cold-{offset}", Q_(8, "MB"))
            for offset in range(5)
        )
        events.extend(
            AccessEvent(Q_(420 + index * 20, "ms"), "hot", Q_(8, "MB"))
            for index in range(20)
        )
        return tuple(events)

    spread = simulate_cache_prefetch(trace(25), cache_capacity=Q_(32, "MB"), prefetch_depth=1)
    burst = simulate_cache_prefetch(trace(1), cache_capacity=Q_(32, "MB"), prefetch_depth=1)
    deeper = simulate_cache_prefetch(trace(1), cache_capacity=Q_(32, "MB"), prefetch_depth=4)

    assert burst["hit_rate"] == spread["hit_rate"]
    assert burst["total_stall"] > spread["total_stall"]
    assert deeper["total_stall"] < burst["total_stall"]
    assert deeper["cache_misses"] == burst["cache_misses"]


def test_miss_burst_fixture_changes_timing_without_changing_accesses():
    spread = make_miss_burst_trace(cold_spacing=Q_(25, "ms"))
    burst = make_miss_burst_trace(cold_spacing=Q_(1, "ms"))

    assert [event.object_id for event in spread] == [event.object_id for event in burst]
    assert [event.payload for event in spread] == [event.payload for event in burst]
    assert burst[13].arrival < spread[13].arrival


def test_track_cache_adapts_to_all_tracks_and_preserves_causal_prefetch_principle():
    prev_batch_bytes = None
    for track_id, profile in TRACKS.items():
        spread = evaluate_track_cache(track_id, cold_spacing=Q_(25, "ms"), prefetch_depth=1)
        burst = evaluate_track_cache(track_id, cold_spacing=Q_(1, "ms"), prefetch_depth=1)
        deeper = evaluate_track_cache(track_id, cold_spacing=Q_(1, "ms"), prefetch_depth=4)

        assert burst["track_id"] == track_id
        assert burst["storage_bandwidth"] == profile["storage_bandwidth"]
        assert burst["consumer_unit"] == profile["consumer_unit"]
        expected_batch = (profile["sample_bytes"] * profile["samples_per_batch"]).to("byte")
        assert burst["batch_bytes"] == expected_batch
        if prev_batch_bytes is not None:
            assert burst["batch_bytes"] != prev_batch_bytes
        prev_batch_bytes = burst["batch_bytes"]

        assert burst["hit_rate"] == spread["hit_rate"]
        assert burst["cache_misses"] == spread["cache_misses"]
        assert burst["total_stall"] > spread["total_stall"]
        assert deeper["total_stall"] < burst["total_stall"]
        assert deeper["cache_misses"] == burst["cache_misses"]


def test_track_cache_records_exact_replayable_inputs():
    original = evaluate_track_cache("mobile", cold_spacing=Q_(2, "ms"), prefetch_depth=2)
    frozen = serialize_for_evidence(original)
    replayed_inputs = deserialize_evidence_inputs(frozen["inputs"])
    replayed = evaluate_track_cache(**replayed_inputs)

    assert frozen["inputs"]["track_id"] == "mobile"
    assert frozen["inputs"]["prefetch_depth"] == 2
    assert replayed["total_stall"] == original["total_stall"]
    assert replayed["max_stall"] == original["max_stall"]
    assert replayed["hit_rate"] == original["hit_rate"]


def test_shards_reduce_requests_and_bytes_but_oversized_shards_waste_tail_bytes():
    common = dict(
        samples=1_000,
        sample_bytes=Q_(8, "kB"),
        storage_bandwidth=Q_(500, "MB/s"),
        metadata_rate=Q_(500, "count/s"),
        preprocess_rate=Q_(50_000, "count/s"),
    )
    small = evaluate_layout(samples_per_object=1, **common)
    shard = evaluate_layout(samples_per_object=64, **common)
    oversized = evaluate_layout(samples_per_object=2_000, **common)

    assert shard["request_count"] < small["request_count"]
    assert shard["transferred_bytes"] < small["transferred_bytes"]
    assert small["bottleneck"] == "metadata"
    assert shard["effective_sample_rate"] > small["effective_sample_rate"]
    assert oversized["request_count"] == 1
    assert oversized["transferred_bytes"] > shard["transferred_bytes"]


def test_buying_bandwidth_cannot_fix_a_preprocessing_bottleneck():
    common = dict(
        samples=20_000,
        sample_bytes=Q_(32, "kB"),
        samples_per_object=256,
        metadata_rate=Q_(20_000, "count/s"),
        preprocess_rate=Q_(1_000, "count/s"),
    )
    baseline = evaluate_layout(storage_bandwidth=Q_(1, "GB/s"), **common)
    more_bandwidth = evaluate_layout(storage_bandwidth=Q_(10, "GB/s"), **common)

    assert baseline["bottleneck"] == more_bandwidth["bottleneck"] == "preprocess"
    assert more_bandwidth["effective_sample_rate"] == baseline["effective_sample_rate"]


def test_track_layout_wrapper_exposes_replayable_track_controls():
    result = evaluate_track_layout("mobile", samples_per_object=64, storage_scale=1.5)

    assert result["inputs"] == {
        "track_id": "mobile", "samples_per_object": 64, "storage_scale": 1.5
    }
    assert result["track_id"] == "mobile"


def test_checkpoint_is_restorable_only_after_durable_publication_completes():
    common = dict(
        checkpoint_bytes=Q_(10, "GB"),
        local_bandwidth=Q_(2, "GB/s"),
        durable_bandwidth=Q_(1, "GB/s"),
    )
    local_failure = simulate_checkpoint_publication(failure_time=Q_(3, "s"), **common)
    publish_failure = simulate_checkpoint_publication(failure_time=Q_(12, "s"), **common)
    boundary = simulate_checkpoint_publication(failure_time=Q_(15, "s"), **common)

    assert local_failure["failure_stage"] == "local_copy"
    assert publish_failure["failure_stage"] == "durable_publication"
    assert not local_failure["recoverable"]
    assert not publish_failure["recoverable"]
    assert publish_failure["restorable_bytes"] == Q_(0, "byte")
    assert boundary["recoverable"]
    assert boundary["restorable_bytes"] == Q_(10, "GB")


def test_track_checkpoint_injection_hits_each_named_stage():
    local = evaluate_track_checkpoint("cloud", failure_stage="local_copy")
    publishing = evaluate_track_checkpoint("cloud", failure_stage="durable_publication")
    after = evaluate_track_checkpoint("cloud", failure_stage="after_publication")

    assert local["failure_stage"] == "local_copy"
    assert publishing["failure_stage"] == "durable_publication"
    assert after["failure_stage"] == "after_publication"
    assert not local["recoverable"] and not publishing["recoverable"]
    assert after["recoverable"]


def test_cost_accounts_for_idle_accelerators_requests_movement_and_capacity():
    result = evaluate_storage_cost(
        duration=Q_(10, "hour"),
        accelerator_count=16,
        accelerator_rate=Q_(2, "dollar/hour"),
        accelerator_utilization=0.60,
        request_count=2_000_000,
        request_price_per_thousand=Q_(0.005, "dollar"),
        moved_bytes=Q_(4_000, "GB"),
        movement_price_per_gb=Q_(0.02, "dollar/GB"),
        stored_bytes=Q_(5_000, "GB"),
        capacity_price_per_gb_month=Q_(0.02, "dollar/GB/month"),
    )

    assert result["idle_accelerator_cost"] == Q_(128, "dollar")
    assert result["request_cost"] == Q_(10, "dollar")
    assert result["movement_cost"] == Q_(80, "dollar")
    assert result["capacity_cost"].magnitude > 0
    assert result["total_cost"] > result["accelerator_cost"]


def test_faster_tier_can_raise_movement_cost_but_lower_job_cost():
    common = dict(
        accelerator_count=32,
        accelerator_rate=Q_(3, "dollar/hour"),
        request_count=100_000,
        request_price_per_thousand=Q_(0.005, "dollar"),
        moved_bytes=Q_(2_000, "GB"),
    )
    slow = evaluate_storage_cost(
        duration=Q_(10, "hour"),
        accelerator_utilization=0.5,
        movement_price_per_gb=Q_(0.005, "dollar/GB"),
        **common,
    )
    fast = evaluate_storage_cost(
        duration=Q_(5.5, "hour"),
        accelerator_utilization=0.91,
        movement_price_per_gb=Q_(0.03, "dollar/GB"),
        **common,
    )

    assert fast["movement_cost"] > slow["movement_cost"]
    assert fast["idle_accelerator_cost"] < slow["idle_accelerator_cost"]
    assert fast["total_cost"] < slow["total_cost"]


def test_track_tier_plans_show_idle_cost_and_price_tradeoff_on_every_track():
    for track_id in TRACKS:
        plans = {tier: evaluate_tier_plan(track_id, tier_id=tier) for tier in TIER_PLANS}

        assert plans["economy"]["idle_accelerator_cost"].magnitude > 0
        assert plans["performance"]["movement_cost"] > plans["economy"]["movement_cost"]
        assert plans["performance"]["job_duration"] < plans["economy"]["job_duration"]
        assert all(plan["inputs"]["track_id"] == track_id for plan in plans.values())


def test_evidence_serializer_preserves_exact_replayable_evaluator_arguments():
    original = simulate_checkpoint_publication(
        checkpoint_bytes=Q_(10, "GiB"),
        local_bandwidth=Q_(2, "GiB/s"),
        durable_bandwidth=Q_(1, "GiB/s"),
        failure_time=Q_(12, "s"),
    )
    frozen = serialize_for_evidence(original)
    replay_args = deserialize_evidence_inputs(frozen["inputs"])
    replay = simulate_checkpoint_publication(**replay_args)

    assert frozen["inputs"]["checkpoint_bytes"] == {
        "__type__": "quantity", "magnitude": 10 * 1024**3, "unit": "B"
    }
    assert replay["durable_complete"] == original["durable_complete"]
    assert replay["recoverable"] == original["recoverable"]


def test_access_trace_arguments_survive_evidence_round_trip():
    original = simulate_cache_prefetch(
        make_access_trace(accesses=8, burst_start=4, burst_length=2), prefetch_depth=3
    )
    replay_args = deserialize_evidence_inputs(serialize_for_evidence(original["inputs"]))
    replay = simulate_cache_prefetch(**replay_args)

    assert all(isinstance(event, AccessEvent) for event in replay_args["events"])
    assert replay["total_stall"] == original["total_stall"]
    assert replay["hit_rate"] == original["hit_rate"]


def test_complete_tier_evidence_is_strict_finite_json():
    frozen = serialize_for_evidence(evaluate_tier_plan("cloud", tier_id="performance"))

    json.dumps(frozen, allow_nan=False)
    with pytest.raises(ValueError):
        serialize_for_evidence(math.inf)
    with pytest.raises(ValueError):
        serialize_for_evidence(Q_(math.nan, "second"))


@pytest.mark.parametrize(
    "call",
    [
        lambda: track_profile("unknown"),
        lambda: evaluate_delivery("edge", consumers=0),
        lambda: evaluate_delivery("edge", target_utilization=1.1),
        lambda: make_access_trace(accesses=4, burst_start=3, burst_length=2),
        lambda: simulate_cache_prefetch([], prefetch_depth=1),
        lambda: evaluate_layout(
            samples=10,
            sample_bytes=Q_(-1, "byte"),
            samples_per_object=1,
            storage_bandwidth=Q_(1, "GB/s"),
            metadata_rate=Q_(1, "count/s"),
            preprocess_rate=Q_(1, "count/s"),
        ),
        lambda: evaluate_storage_cost(
            duration=Q_(1, "hour"), accelerator_count=1,
            accelerator_rate=Q_(1, "dollar/hour"), accelerator_utilization=1.1,
            request_count=0, request_price_per_thousand=Q_(0, "dollar"),
            moved_bytes=Q_(0, "GB"), movement_price_per_gb=Q_(0, "dollar/GB"),
        ),
    ],
)
def test_invalid_inputs_fail_explicitly(call):
    with pytest.raises(ValueError):
        call()
