from __future__ import annotations

from mlsysbook_labs import (
    diagnose_triad,
    get_lab_track_variant,
    get_track_profile,
    intervention_frontier,
    resolve_mlsysim_ref,
    triad_track_profile,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_01_ai_triad", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return triad_track_profile(track, variant, hardware, model)


def test_triad_diagnosis_uses_track_thresholds():
    profile = _profile("robotaxi")
    result = diagnose_triad(
        profile,
        data_score_pct=profile.default_data_pct,
        algorithm_score_pct=profile.default_algorithm_pct,
        machine_score_pct=profile.default_machine_pct,
    )

    assert result.binding_axis == "Data"
    assert not result.feasible
    assert "Data" in result.violations


def test_intervention_frontier_scores_wrong_axis_as_weaker():
    profile = _profile("oura_ring")
    result = intervention_frontier(
        profile,
        data_budget_pct=80,
        algorithm_budget_pct=10,
        machine_budget_pct=10,
        selected_intervention="Data",
    )

    assert result.selected_intervention == "Data"
    assert result.binding_axis in {"Algorithm", "Machine"}
    assert result.rejected_alternatives


def test_all_v1_01_profiles_resolve_refs_and_axis_text():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref
        assert profile.model_ref
        assert profile.failure_story
        assert profile.data_axis
        assert profile.algorithm_axis
        assert profile.machine_axis
        assert profile.validation_tests
