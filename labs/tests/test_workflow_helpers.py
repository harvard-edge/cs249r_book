from __future__ import annotations

from mlsysbook_labs import (
    constraint_tax,
    get_lab_track_variant,
    get_track_profile,
    iteration_frontier,
    resolve_mlsysim_ref,
    workflow_policy,
    workflow_track_profile,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_03_constraint_tax", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return workflow_track_profile(track, variant, hardware, model)


def test_workflow_profiles_resolve_refs_and_gates():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref.startswith("Hardware.")
        assert profile.model_ref.startswith("Models.")
        assert profile.constraint_name
        assert len(profile.stage_names) == 6
        assert profile.gate_options
        assert profile.release_policies
        assert profile.rollback_rules


def test_constraint_tax_rewards_early_gate():
    profile = _profile("robotaxi")
    late = constraint_tax(profile, discovery_stage=profile.default_discovery_stage)
    early = constraint_tax(profile, discovery_stage=profile.recommended_gate_stage)

    assert late.rework_days > early.rework_days
    assert late.avoidable_rework_days > 0
    assert late.late_discovery
    assert "Field rollout" in late.artifacts_to_rebuild


def test_iteration_frontier_trades_cycle_time_for_risk():
    profile = _profile("oura_ring")
    shallow = iteration_frontier(
        profile,
        validation_depth_pct=20,
        automation_pct=20,
        hardware_realism_pct=20,
        data_scale_pct=20,
    )
    deep = iteration_frontier(
        profile,
        validation_depth_pct=80,
        automation_pct=80,
        hardware_realism_pct=80,
        data_scale_pct=80,
    )

    assert deep.residual_risk_pct < shallow.residual_risk_pct
    assert deep.confidence_pct > shallow.confidence_pct
    assert shallow.bottleneck


def test_workflow_policy_packages_gate_and_blind_spot():
    profile = _profile("cloud_fleet")
    frontier = iteration_frontier(
        profile,
        validation_depth_pct=profile.default_validation_depth_pct,
        automation_pct=profile.default_automation_pct,
        hardware_realism_pct=profile.default_hardware_realism_pct,
        data_scale_pct=profile.default_data_scale_pct,
    )
    policy = workflow_policy(
        profile,
        frontier,
        gate_id="staging_load",
        release_policy=profile.release_policies[0],
        rollback_rule=profile.rollback_rules[0],
    )

    assert policy.gate_label == "Staging load gate"
    assert policy.rework_days_at_gate > 0
    assert policy.residual_risk_pct == frontier.residual_risk_pct
    assert policy.blind_spot
