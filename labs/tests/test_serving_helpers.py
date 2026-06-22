from __future__ import annotations

from mlsysbook_labs import (
    batching_tax,
    cache_capacity,
    cold_start_latency,
    get_lab_track_variant,
    get_track_profile,
    queueing_latency,
    resolve_mlsysim_ref,
    serving_track_profile,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_13_tail_latency_trap", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return serving_track_profile(track, variant, hardware, model), model


def test_queueing_reference_case_matches_tail_latency_trap():
    result = queueing_latency(
        arrival_qps=160,
        service_ms=5,
        replicas=1,
        service_cv=1.0,
        slo_ms=50,
    )

    assert round(result.utilization, 2) == 0.80
    assert 24 < result.mean_latency_ms < 26
    assert 113 < result.p99_latency_ms < 117
    assert not result.slo_ok


def test_batching_tax_reports_static_batch_delay():
    result = batching_tax(
        batch_size=32,
        arrival_qps=500,
        service_ms=5,
        slo_ms=50,
        efficiency_gain=6.4,
    )

    assert round(result.formation_delay_ms) == 31
    assert round(result.formation_slo_pct) == 62
    assert result.throughput_gain > 20


def test_cloud_cache_capacity_uses_llama_and_h100_refs():
    profile, model = _profile("cloud_fleet")
    result = cache_capacity(
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


def test_cold_start_exposes_llm_data_movement():
    profile, _model = _profile("cloud_fleet")
    result = cold_start_latency(
        profile,
        precision_bytes=2.0,
        warm_pool_replicas=0,
        scale_out_replicas=4,
    )

    assert result.model_weight_gb == 140
    assert result.cold_start_ms > 25_000
    assert result.exceeds_slo


def test_all_v1_13_profiles_resolve_refs():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile, _model = _profile(track_id)

        assert profile.hardware_ref
        assert profile.model_ref
        assert profile.arrival_qps > 0
        assert profile.service_ms > 0
        assert profile.slo_ms > 0
        assert profile.validation_tests
