from __future__ import annotations

from mlsysbook_labs import (
    architecture_decision,
    architecture_scaling_curve,
    architecture_signature,
    architecture_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_06_architecture_tax", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return architecture_track_profile(track, variant, hardware, model)


def _signature_map(track_id: str):
    profile = _profile(track_id)
    return {item.architecture_id: item for item in architecture_signature(profile)}


def test_architecture_profiles_resolve_refs_and_candidates():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref.startswith("Hardware.")
        assert profile.model_ref.startswith("Models.")
        assert profile.candidates
        assert profile.default_scale > 0
        assert profile.validation_tests
        assert profile.report_artifact == "architecture recommendation memo"


def test_architecture_signature_reports_track_specific_failures():
    iphone = _signature_map("iphone")
    oura = _signature_map("oura_ring")
    robotaxi = _signature_map("robotaxi")
    cloud = _signature_map("cloud_fleet")

    assert iphone["mobile_cnn"].feasible is True
    assert iphone["desktop_vit"].feasible is False
    assert oura["streaming_tcn"].feasible is True
    assert oura["micro_transformer"].dominant_constraint in {
        "activation memory",
        "latency",
        "power",
        "kernel support",
    }
    assert robotaxi["bounded_cnn_detector"].feasible is True
    assert robotaxi["large_multimodal_transformer"].feasible is False
    assert cloud["bert_encoder"].feasible is True
    assert cloud["wide_cnn_classifier"].dominant_constraint == "quality guardrail"


def test_architecture_scaling_curve_finds_first_failure():
    profile = _profile("iphone")
    curve = architecture_scaling_curve(profile, samples=12)

    assert len(curve.scale_values) == 12
    assert curve.first_failure_by_candidate["desktop_vit"] is not None
    assert curve.first_failure_by_candidate["mobile_cnn"] is not None
    assert "mobile_cnn" in curve.points_by_candidate


def test_architecture_decision_returns_rejections_and_memo():
    profile = _profile("robotaxi")
    decision = architecture_decision(profile, architecture_id="bounded_cnn_detector")

    assert decision.selected_label == "Bounded CNN detector"
    assert decision.feasible is True
    assert decision.rejected_alternatives
    assert "RoboTaxi" in decision.memo_summary
    assert decision.validation_requirement
