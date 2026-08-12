from __future__ import annotations

from pathlib import Path

from mlsysbook_labs import (
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    system_curve,
    system_decision,
    system_design_ledger_summary,
    system_design_profile,
    system_frontier,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_DESIGN_LAB_IDS = (
    "v2_01_scale_illusion",
    "v2_02_compute_wall",
    "v2_03_network_fabric_design",
    "v2_04_data_pipeline_wall",
    "v2_05_parallelism_design",
    "v2_07_failure_budget_engineering",
    "v2_08_fleet_orchestration",
    "v2_09_optimization_trap",
    "v2_12_silent_fleet",
    "v2_13_price_of_privacy",
    "v2_14_robustness_budget",
    "v2_15_carbon_budget",
    "v2_16_fairness_budget",
    "v2_17_fleet_synthesis",
)


def _profile(lab_id: str, track_id: str):
    track = get_track_profile(track_id)
    variant = get_lab_track_variant(lab_id, track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    return system_design_profile(track, variant, hardware, model, title=lab_id), variant


def test_system_design_profiles_resolve_refs_and_report_artifacts():
    for lab_id in SYSTEM_DESIGN_LAB_IDS:
        for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
            profile, variant = _profile(lab_id, track_id)

            assert variant.assumptions["system_design_variant"] is True
            assert variant.assumptions.get("fallback_variant") is not True
            assert profile.hardware_ref.startswith("Hardware.")
            assert profile.model_ref.startswith("Models.")
            assert profile.hardware_name
            assert profile.model_name
            assert profile.report_artifact
            assert profile.validation_tests
            assert len(profile.decision_options) == 3
            assert {option.option_id for option in profile.decision_options} == {
                "local_baseline",
                "balanced_policy",
                "scale_first",
            }


def test_system_design_balanced_policy_is_default_feasible():
    for lab_id in SYSTEM_DESIGN_LAB_IDS:
        for track_id in ("iphone", "oura_ring", "robotaxi", "cloud_fleet"):
            profile, _variant = _profile(lab_id, track_id)
            frontier = system_frontier(profile)
            balanced = next(result for result in frontier if result.option_id == "balanced_policy")

            assert len(frontier) == 3
            assert balanced.feasible is True
            assert balanced.stress_ratio < 1.0
            assert balanced.quality_pct >= profile.quality_floor_pct
            assert balanced.guardrail_pct >= profile.guardrail_floor_pct
            assert any(not result.feasible for result in frontier)


def test_system_design_curve_and_decision_surface_residual_risk():
    for lab_id in SYSTEM_DESIGN_LAB_IDS:
        profile, _variant = _profile(lab_id, "robotaxi")
        curve = system_curve(profile, option_id="balanced_policy", samples=12)
        decision = system_decision(profile, option_id="balanced_policy")

        assert curve.points
        assert curve.first_failure is not None
        assert decision.feasible is True
        assert decision.selected_id == "balanced_policy"
        assert decision.mitigation
        assert decision.validation_requirement
        assert decision.residual_risk
        assert len(decision.rejected_alternatives) == 2


def test_system_design_ledger_summary_filters_prior_volume_decisions():
    class _State:
        history = {
            1: {
                "chapter": "v2_01",
                "track_id": "iphone",
                "scenario_id": "scale",
                "selected_option": "balanced_policy",
                "dominant_risk": "capacity stress",
            },
            2: {"chapter": "v1_02", "selected_option": "wrong volume"},
            17: {"chapter": "v2_17", "selected_option": "current lab"},
        }

    class _Ledger:
        _state = _State()

    summary = system_design_ledger_summary(_Ledger(), prefix="v2_", upto=17)

    assert len(summary) == 1
    assert summary[0].chapter == "v2_01"
    assert summary[0].selected_option == "balanced_policy"


def test_shared_system_design_renderer_uses_progressive_disclosure():
    source = (REPO_ROOT / "labs" / "mlsysbook_labs" / "system_design.py").read_text(encoding="utf-8")

    assert "prediction_complete = prediction.value is not None" in source
    assert "Select your prediction to unlock A2 and A3" in source
    assert "Complete the Part A prediction to unlock the scaling boundary evidence" in source
    assert "Complete the Part B boundary confirmation before writing the decision memo" in source
    assert "Choose a decision option to unlock C2, C3, and C4" in source
    assert 'report_heading = "## Report Status" if incomplete else "## Download Report"' in source
