from __future__ import annotations

from mlsysbook_labs import (
    get_lab_track_variant,
    get_track_profile,
    memory_cliff,
    neural_compute_profile,
    operation_ledger,
    operator_design,
    resolve_mlsysim_ref,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_05_neural_computation", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return neural_compute_profile(track, variant, hardware, model)


def test_neural_compute_profiles_resolve_refs_and_designs():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref.startswith("Hardware.")
        assert profile.model_ref.startswith("Models.")
        assert profile.tensor_label
        assert profile.activation_budget_mb > 0
        assert profile.design_options


def test_operation_ledger_reports_track_specific_dominant_resources():
    results = {}
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)
        result = operation_ledger(profile, shape_multiplier=profile.default_shape_multiplier)
        results[track_id] = result

        assert result.activations_mb > 0
        assert result.bytes_moved_mb > result.weights_mb
        assert result.dominant_resource

    assert results["oura_ring"].dominant_resource == "activation memory"
    assert results["iphone"].dominant_resource in {"bandwidth", "power"}
    assert results["robotaxi"].dominant_resource in {"bandwidth", "activation memory"}


def test_memory_cliff_finds_tiny_threshold():
    profile = _profile("oura_ring")
    sweep = memory_cliff(profile, samples=20)

    assert sweep.threshold_multiplier is not None
    assert sweep.threshold_activation_mb is not None
    assert len(sweep.shape_values) == 20


def test_operator_design_reduces_activation_budget_pressure():
    profile = _profile("iphone")
    baseline = operator_design(profile, design_id="fp16_baseline", shape_multiplier=1.0)
    tiled = operator_design(profile, design_id="tiled_prefetch", shape_multiplier=1.0)

    assert tiled.activation_mb < baseline.activation_mb
    assert tiled.residual_risk
    assert tiled.memo_summary
