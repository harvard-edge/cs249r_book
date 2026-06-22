from __future__ import annotations

from mlsysbook_labs import (
    fusion_traffic,
    gemm_workload,
    get_lab_track_variant,
    get_track_profile,
    hardware_roofline_profile,
    resolve_mlsysim_ref,
    roofline_point,
)


def _roofline_profile(track_id: str):
    profile = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_11_hardware_roofline", track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    return hardware_roofline_profile(profile, variant, hardware)


def test_roofline_profile_uses_mlsysim_hardware_refs():
    h100 = _roofline_profile("cloud_fleet")
    oura = _roofline_profile("oura_ring")

    assert h100.hardware_ref == "Hardware.Cloud.H100"
    assert h100.peak_tflops > oura.peak_tflops
    assert h100.bandwidth_gbs > oura.bandwidth_gbs
    assert h100.ridge_flop_per_byte > 0
    assert oura.memory_capacity_gb < 0.01


def test_gemm_workload_and_roofline_point_are_consistent():
    h100 = _roofline_profile("cloud_fleet")
    workload = gemm_workload(dimension=512, precision="fp16")
    point = roofline_point(h100, workload.arithmetic_intensity)

    assert workload.flops == 2 * 512**3
    assert workload.bytes_per_element == 2
    assert point.arithmetic_intensity == workload.arithmetic_intensity
    assert point.attainable_gflops <= point.peak_gflops
    assert point.regime in {"Memory-bound", "Compute-bound"}


def test_roofline_regime_changes_with_arithmetic_intensity():
    h100 = _roofline_profile("cloud_fleet")

    low = roofline_point(h100, h100.ridge_flop_per_byte / 2)
    high = roofline_point(h100, h100.ridge_flop_per_byte * 2)

    assert low.regime == "Memory-bound"
    assert high.regime == "Compute-bound"


def test_fusion_traffic_reduces_memory_time():
    result = fusion_traffic(elements=4096, bytes_per_element=2, bandwidth_gbs=1000)

    assert result.eager_bytes > result.fused_bytes
    assert result.eager_time_us > result.fused_time_us
    assert result.speedup == 3.0
