from __future__ import annotations

from mlsysbook_labs import (
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    training_frontier,
    training_memory_stack,
    training_plan,
    training_track_profile,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_08_training_gauntlet", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return training_track_profile(track, variant, hardware, model)


def test_training_profiles_resolve_refs_and_strategies():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref.startswith("Hardware.")
        assert profile.model_ref.startswith("Models.")
        assert profile.model_params_m > 0
        assert profile.strategy_options
        assert profile.validation_tests
        assert profile.report_artifact == "training feasibility plan"


def test_training_memory_stack_distinguishes_full_training_from_adaptation():
    iphone = _profile("iphone")
    oura = _profile("oura_ring")
    robotaxi = _profile("robotaxi")
    cloud = _profile("cloud_fleet")

    assert training_memory_stack(iphone, strategy_id="full_on_device").feasible is False
    assert training_memory_stack(iphone, strategy_id="lora_personalization").feasible is True
    assert training_memory_stack(oura, strategy_id="full_ring_training").feasible is False
    assert training_memory_stack(oura, strategy_id="threshold_calibration").feasible is True
    assert training_memory_stack(robotaxi, strategy_id="local_full_retrain").feasible is False
    assert training_memory_stack(robotaxi, strategy_id="fleet_retrain_local_validation").dominant_component == "activations"
    assert training_memory_stack(cloud, strategy_id="fp32_adam_full").feasible is False
    assert training_memory_stack(cloud, strategy_id="mixed_checkpointed").feasible is True


def test_training_frontier_records_batch_boundary():
    profile = _profile("cloud_fleet")
    frontier = training_frontier(profile, strategy_id="mixed_checkpointed")

    assert frontier.points
    assert frontier.max_feasible_batch is not None
    assert frontier.first_infeasible_batch is not None
    assert frontier.max_feasible_batch < frontier.first_infeasible_batch


def test_training_plan_names_location_validation_and_handoff():
    profile = _profile("robotaxi")
    plan = training_plan(profile, strategy_id="fleet_retrain_local_validation")

    assert plan.selected_label == "Fleet retraining + local validation"
    assert plan.feasible is True
    assert "fleet" in plan.training_location.lower()
    assert "vehicle" in plan.validation_location.lower()
    assert plan.deployment_handoff
    assert plan.rejected_alternatives
