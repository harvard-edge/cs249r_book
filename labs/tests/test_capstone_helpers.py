from __future__ import annotations

from mlsysbook_labs import (
    architecture_memo,
    capstone_track_profile,
    get_lab_track_variant,
    get_track_profile,
    replay_ledger,
    resolve_mlsysim_ref,
    sensitivity_audit,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_16_architects_audit", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return capstone_track_profile(track, variant, hardware, model)


def test_replay_ledger_uses_student_entries_and_presets():
    profile = _profile("cloud_fleet")
    result = replay_ledger(
        profile,
        {
            13: {"chapter": "v1_13", "decision": "increase warm pool"},
            "15": {"chapter": "v1_15", "decision": "add carbon audit"},
        },
    )

    assert result.entries_found == 2
    assert result.entries_expected == len(profile.prior_decisions)
    assert result.coverage_pct > 0
    assert any(decision.source == "student ledger" for decision in result.decisions)
    assert any(decision.source == "track preset" for decision in result.decisions)


def test_sensitivity_audit_flags_robotaxi_guardrail_failure():
    profile = _profile("robotaxi")
    result = sensitivity_audit(
        profile,
        workload_multiplier=1.2,
        model_growth_pct=10,
        guardrail_tightening_pct=25,
        evidence_confidence_pct=95,
    )

    assert not result.feasible
    assert "Guardrail tightening" in result.violations
    assert result.most_fragile


def test_architecture_memo_packages_track_decision():
    profile = _profile("oura_ring")
    result = architecture_memo(
        profile,
        revised_decision="reserve OTA flash headroom",
        top_risk="OTA payload exceeds safe update budget",
        mitigation="run OTA payload test before release",
    )

    assert result.track_id == "oura_ring"
    assert "reserve OTA flash headroom" in result.memo_summary
    assert result.validation_tests


def test_all_v1_16_profiles_resolve_refs_and_have_capstone_defaults():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref
        assert profile.model_ref
        assert profile.architecture_components
        assert profile.prior_decisions
        assert profile.sensitivity_defaults
        assert profile.revision_options
        assert profile.top_risks
        assert profile.validation_tests
