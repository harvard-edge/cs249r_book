from __future__ import annotations

from mlsysbook_labs import (
    coverage_profile,
    data_policy_decision,
    data_selection_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    selection_frontier,
    selection_utility,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_09_selection_paradox", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return data_selection_profile(track, variant, hardware, model)


def test_selection_profiles_resolve_refs_and_policies():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref.startswith("Hardware.")
        assert profile.model_ref.startswith("Models.")
        assert profile.policy_options
        assert profile.subgroups
        assert profile.validation_tests
        assert profile.report_artifact == "data selection policy memo"


def test_selection_utility_distinguishes_quantity_from_coverage():
    iphone = _profile("iphone")
    oura = _profile("oura_ring")
    robotaxi = _profile("robotaxi")
    cloud = _profile("cloud_fleet")

    assert selection_utility(iphone, policy_id="volume_logging").feasible is False
    assert selection_utility(iphone, policy_id="private_context_balanced").feasible is True
    assert selection_utility(oura, policy_id="continuous_raw_sampling").feasible is False
    assert selection_utility(oura, policy_id="signal_quality_balanced").feasible is True
    assert selection_utility(robotaxi, policy_id="random_miles").feasible is False
    assert selection_utility(robotaxi, policy_id="rare_event_mining").feasible is True
    assert selection_utility(cloud, policy_id="cheap_scale").feasible is False
    assert selection_utility(cloud, policy_id="subgroup_balanced").feasible is True


def test_selection_frontier_and_coverage_profile_surface_worst_group():
    profile = _profile("robotaxi")
    frontier = selection_frontier(profile)
    coverage = coverage_profile(profile, policy_id="rare_event_mining")

    assert len(frontier) == len(profile.policy_options)
    assert coverage.cells
    assert coverage.worst_subgroup
    assert coverage.worst_risk_score >= 0


def test_data_policy_decision_names_next_data_and_blind_spot():
    profile = _profile("cloud_fleet")
    decision = data_policy_decision(profile, policy_id="subgroup_balanced")

    assert decision.selected_label == "Subgroup-balanced curation"
    assert decision.feasible is True
    assert "language" in decision.next_data.lower()
    assert decision.accepted_blind_spot
    assert decision.rejected_alternatives
