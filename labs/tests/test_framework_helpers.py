from __future__ import annotations

from mlsysbook_labs import (
    compile_break_even,
    dispatch_stack,
    framework_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    runtime_decision,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_07_framework_tax", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return framework_track_profile(track, variant, hardware, model)


def test_framework_profiles_resolve_refs_and_runtime_options():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref.startswith("Hardware.")
        assert profile.model_ref.startswith("Models.")
        assert profile.runtime_options
        assert profile.latency_budget_ms > 0
        assert profile.validation_tests
        assert profile.report_artifact == "runtime deployment recommendation"


def test_dispatch_stack_reports_track_specific_runtime_failures():
    iphone = _profile("iphone")
    oura = _profile("oura_ring")
    robotaxi = _profile("robotaxi")
    cloud = _profile("cloud_fleet")

    assert dispatch_stack(iphone, runtime_id="coreml_delegate").feasible is True
    assert dispatch_stack(iphone, runtime_id="pytorch_mobile").feasible is False
    assert dispatch_stack(oura, runtime_id="generated_c_kernels").feasible is True
    assert dispatch_stack(oura, runtime_id="dynamic_tflite").feasible is False
    assert dispatch_stack(robotaxi, runtime_id="tensorrt_static").feasible is True
    assert dispatch_stack(robotaxi, runtime_id="eager_pytorch").feasible is False
    assert dispatch_stack(cloud, runtime_id="cuda_graph_tensorrt").feasible is True
    assert dispatch_stack(cloud, runtime_id="eager_pytorch").dominant_overhead == "runtime dispatch"


def test_compile_break_even_uses_reuse_count():
    profile = _profile("cloud_fleet")
    result = compile_break_even(
        profile,
        runtime_id="cuda_graph_tensorrt",
        reuse_count=profile.default_reuse_count,
    )

    assert result.break_even_inferences is not None
    assert result.per_inference_savings_ms > 0
    assert result.pays_back is True


def test_runtime_decision_returns_unsupported_op_warning():
    profile = _profile("robotaxi")
    decision = runtime_decision(profile, runtime_id="tensorrt_static")

    assert decision.selected_label == "TensorRT static engine"
    assert decision.feasible is True
    assert decision.rejected_alternatives
    assert "RoboTaxi" in decision.memo_summary
    assert "unsupported" in decision.unsupported_op_warning
