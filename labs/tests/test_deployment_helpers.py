from __future__ import annotations

from mlsysbook_labs import (
    deployment_mitigation,
    deployment_track_profile,
    evaluate_deployment_envelope,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    sweep_deployment_knob,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_02_physics_of_deployment", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return deployment_track_profile(track, variant, hardware, model)


def test_deployment_profiles_resolve_hardware_model_and_budgets():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref.startswith("Hardware.")
        assert profile.model_ref.startswith("Models.")
        assert profile.peak_tflops > 0
        assert profile.memory_budget_mb > 0
        assert profile.latency_budget_ms > 0
        assert profile.placement_options
        assert profile.mitigation_options


def test_default_envelopes_produce_track_specific_first_walls():
    walls = {}
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)
        result = evaluate_deployment_envelope(
            profile,
            workload_value=profile.default_knob,
            placement_id=profile.placement_options[0].placement_id,
        )
        walls[track_id] = result.first_wall
        assert result.checks
        assert result.violations

    assert walls["iphone"] == "power"
    assert walls["oura_ring"] in {"memory", "energy"}
    assert walls["robotaxi"] == "latency"
    assert walls["cloud_fleet"] in {"cost", "bandwidth", "latency"}
    assert len(set(walls.values())) >= 3


def test_sweep_names_first_threshold_crossing():
    profile = _profile("robotaxi")
    sweep = sweep_deployment_knob(profile, placement_id="vehicle_local", samples=16)

    assert sweep.threshold_crossing is not None
    assert sweep.threshold_wall == "latency"
    assert len(sweep.knob_values) == 16
    assert len(sweep.worst_headroom_pct) == 16


def test_mitigation_follows_binding_constraint_and_placement_risk():
    profile = _profile("cloud_fleet")
    result = evaluate_deployment_envelope(
        profile,
        workload_value=profile.default_knob,
        placement_id="central_gpu",
    )
    mitigation = deployment_mitigation(profile, result, placement_id="batch_queue")

    assert mitigation.binding_constraint == result.first_wall
    assert mitigation.placement_label == "Batch queue"
    assert mitigation.mitigation
    assert mitigation.new_risk
