from __future__ import annotations

from mlsysbook_labs import (
    carbon_budget,
    explanation_overhead,
    get_lab_track_variant,
    get_track_profile,
    metric_conflict,
    resolve_mlsysim_ref,
    responsibility_budget,
    responsibility_track_profile,
)


def _profile(track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant("v1_15_no_free_fairness", track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return responsibility_track_profile(track, variant, hardware, model)


def test_metric_conflict_exposes_subgroup_gap():
    profile = _profile("cloud_fleet")
    result = metric_conflict(
        profile,
        base_rate_a_pct=30,
        base_rate_b_pct=10,
        threshold=0.50,
    )

    assert result.fpr_gap_pp > profile.target_gap_pp
    assert result.harmed_party == profile.harmed_party
    assert result.conflict_summary


def test_explanation_overhead_flags_slo_risk_for_robotaxi_trace():
    profile = _profile("robotaxi")
    result = explanation_overhead(
        profile,
        method=profile.explanation_method,
        features=profile.explanation_features,
        coverage_pct=profile.explanation_coverage_pct,
    )

    assert result.multiplier >= 4
    assert result.total_latency_ms > profile.base_latency_ms
    assert result.slo_ok is False


def test_responsibility_budget_reports_violations_for_overbuilt_oura_policy():
    profile = _profile("oura_ring")
    result = responsibility_budget(
        profile,
        privacy_level=100,
        explanation_coverage_pct=100,
        robustness_level=100,
        monitoring_level=100,
    )

    assert result.energy_factor > profile.max_energy_factor
    assert not result.feasible
    assert result.violations


def test_carbon_budget_scales_with_retraining_and_explanations():
    profile = _profile("cloud_fleet")
    baseline = carbon_budget(
        profile,
        retrain_frequency_per_year=1,
        explanation_coverage_pct=0,
        grid_ci_g_per_kwh=profile.grid_ci_g_per_kwh,
    )
    responsible = carbon_budget(
        profile,
        retrain_frequency_per_year=profile.retrain_frequency_per_year,
        explanation_coverage_pct=profile.explanation_coverage_pct,
        grid_ci_g_per_kwh=profile.grid_ci_g_per_kwh,
    )

    assert responsible.total_kgco2_per_year > baseline.total_kgco2_per_year
    assert responsible.carbon_multiplier > 1


def test_all_v1_15_profiles_resolve_refs_and_policy_text():
    for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
        profile = _profile(track_id)

        assert profile.hardware_ref
        assert profile.model_ref
        assert profile.harmed_party
        assert profile.obligation
        assert profile.audit_signal
        assert profile.validation_tests
