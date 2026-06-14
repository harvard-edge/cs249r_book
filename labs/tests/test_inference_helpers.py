from __future__ import annotations

from mlsysbook_labs import (
    batching_result,
    cost_crossover,
    get_lab_track_variant,
    get_track_profile,
    inference_economy_profile,
    resolve_mlsysim_ref,
    serving_plan,
    state_capacity,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v2_10_inference_economy", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return inference_economy_profile(track, variant, hardware, model), model


def test_cost_crossover_matches_cloud_reference_case():
    result = cost_crossover(setup_cost=2_000_000, demand_qps=100, cost_per_event=0.01)

    assert round(result.daily_cost) == 86_400
    assert 3.2 < result.crossover_weeks < 3.4


def test_cloud_kv_capacity_uses_model_and_hardware_refs():
    profile, model = _profile("cloud_fleet")
    result = state_capacity(
        profile,
        model,
        context_tokens=131_072,
        precision_bytes=2.0,
        devices_per_replica=8,
    )

    assert profile.hardware_ref == "Hardware.Cloud.H100"
    assert profile.model_ref == "Models.Language.Llama2_70B"
    assert result.weight_gb == 140
    assert 340 < result.state_per_request_gb < 345
    assert result.max_concurrent == 1


def test_device_tracks_have_positive_state_capacity():
    for track_id in ("iphone", "oura_ring", "robotaxi"):
        profile, model = _profile(track_id)
        result = state_capacity(
            profile,
            model,
            context_tokens=profile.context_tokens,
            precision_bytes=float(get_lab_track_variant("v2_10_inference_economy", track_id).defaults["precision_bytes"]),
            devices_per_replica=1,
        )

        assert result.total_memory_gb > 0
        assert result.state_per_request_gb > 0
        assert result.max_concurrent >= 1


def test_batching_result_reports_padding_speedup():
    result = batching_result(avg_len=4096, max_len=32768, batch_size=8)

    assert round(result.padding_waste_pct, 1) == 87.5
    assert result.speedup > 6


def test_serving_plan_sizes_daily_cost():
    profile, model = _profile("cloud_fleet")
    result = serving_plan(
        profile,
        model,
        target_qps=10_000,
        precision_bytes=0.5,
        batching_multiplier=3.0,
        devices_per_replica=4,
        context_tokens=32_768,
    )

    assert result.per_replica_qps > 0
    assert result.replicas_needed > 0
    assert result.daily_cost > 0
    assert result.baseline_daily_cost >= result.daily_cost
