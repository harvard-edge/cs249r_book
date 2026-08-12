from __future__ import annotations

from mlsysbook_labs import (
    debt_cascade,
    drift_visibility,
    get_lab_track_variant,
    get_track_profile,
    ops_policy,
    ops_track_profile,
    resolve_mlsysim_ref,
    retraining_cadence,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_14_silent_degradation", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return ops_track_profile(track, variant, hardware, model)


def test_retraining_reference_case_matches_square_root_law():
    result = retraining_cadence(
        retrain_cost=10_000,
        drift_cost_per_day=500,
        current_days=30,
    )

    assert 6.2 < result.optimal_days < 6.4
    assert result.current_too_slow_factor > 4
    assert result.savings_vs_current > 0


def test_drift_visibility_accounts_for_label_delay():
    profile = _profile("cloud_fleet")
    result = drift_visibility(profile, days_since_deploy=30)

    assert result.true_psi > result.observed_psi
    assert result.alert_day > 0
    assert result.true_quality_pct < profile.baseline_quality_pct


def test_ops_policy_flags_loose_policy_violations():
    profile = _profile("iphone")
    result = ops_policy(
        profile,
        threshold_psi=0.5,
        cadence_days=120,
        canary_pct=0,
        rollback_hours=48,
    )

    assert not result.feasible
    assert len(result.violations) >= 3


def test_debt_cascade_compounds_beyond_linear():
    result = debt_cascade(missed_cycles=3, downstream_models=2, base_loss_pp=2.0)

    assert result.total_loss_pp > result.linear_loss_pp
    assert result.debt_multiplier > 3


def test_all_v1_14_profiles_resolve_refs():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref
        assert profile.model_ref
        assert profile.drift_source
        assert profile.monitoring_signal
        assert profile.rollback_policy
        assert profile.validation_tests
