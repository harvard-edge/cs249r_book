from __future__ import annotations

from mlsysbook_labs import (
    amdahl_speedup,
    benchmark_track_profile,
    get_lab_track_variant,
    get_track_profile,
    metric_gate,
    resolve_mlsysim_ref,
    sustained_benchmark,
    tail_latency,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_12_benchmarking_trap", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return benchmark_track_profile(track, variant, hardware, model)


def test_amdahl_reference_case_caps_system_speedup():
    result = amdahl_speedup(component_speedup=10, serial_pct=45)

    assert 1.9 < result.system_speedup < 2.1
    assert result.asymptote < 2.3
    assert result.new_serial_pct > 80


def test_sustained_benchmark_reports_fanless_throttle():
    result = sustained_benchmark(
        peak_value=30,
        tdp_w=60,
        duration_s=600,
        ambient_c=40,
        cooling="fanless",
    )

    assert result.throttled
    assert result.sustained_value < 30
    assert result.loss_pct > 0


def test_metric_gate_lists_violations():
    result = metric_gate(
        accuracy_pct=94,
        p99_latency_ms=150,
        power_w=6,
        throughput=400,
        thresholds={
            "accuracy_min_pct": 90,
            "p99_max_ms": 100,
            "power_max_w": 5,
            "throughput_min": 500,
        },
    )

    assert not result.all_pass
    assert len(result.violations) == 3


def test_tail_latency_heavy_tail_violates_slo():
    result = tail_latency(base_ms=50, sigma=0.8, slo_ms=200)

    assert result.p99_ms > 200
    assert not result.slo_ok
    assert result.violation_pct > 0


def test_all_v1_12_profiles_resolve_refs():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref
        assert profile.model_ref
        assert profile.benchmark_claim
        assert profile.hidden_failure_metric
        assert profile.tdp_w > 0
