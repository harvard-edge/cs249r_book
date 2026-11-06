"""Same-service comparison gates must remain independent of a tempting low total."""
import json

import pytest

from mlsysim.labs.experiments.v1_15_experiments import TRACKS, lifecycle_policy


@pytest.mark.parametrize("track", TRACKS)
def test_profile_changes_energy_work_but_keeps_finite_replay_and_service(track):
    baseline = lifecycle_policy(track)
    candidate = lifecycle_policy(track, execution="replayed_candidate")
    assert candidate["service_checks"] == baseline["service_checks"]
    assert all(candidate["service_checks"].values())
    assert candidate["replay_rows"] == baseline["replay_rows"]
    assert candidate["comparison_gate"] == "comparable_on_supplied_evidence"
    assert candidate["terms_kwh"]["inference"] == pytest.approx(baseline["terms_kwh"]["inference"] / 2)
    assert candidate["terms_kwh"]["preparation"] == pytest.approx(baseline["terms_kwh"]["initial_training"] / 2)
    for term in ("initial_training", "retraining", "explanation"):
        assert candidate["terms_kwh"][term] == baseline["terms_kwh"][term]
    assert candidate["total_energy_kwh"] == pytest.approx(sum(candidate["terms_kwh"].values()))
    assert candidate["operational_carbon_kg"] == pytest.approx(
        candidate["total_energy_kwh"] * candidate["carbon_intensity_g_per_kwh"] / 1000
    )


def test_preparation_can_reverse_per_decision_saving_without_an_accuracy_curve():
    tiny = lifecycle_policy("tinyml", execution="replayed_candidate")
    cloud = lifecycle_policy("cloud", execution="replayed_candidate")
    assert tiny["inference_energy_factor"] == cloud["inference_energy_factor"]
    assert tiny["energy_delta_kwh"] > 0
    assert not tiny["supported_energy_saving"]
    assert cloud["energy_delta_kwh"] < 0
    assert cloud["supported_energy_saving"]


@pytest.mark.parametrize("change,failed", [
    ({"demand_scale": 0.5}, "same_request_volume"),
    ({"demand_scale": 2}, "same_request_volume"),
    ({"retraining_runs": 0}, "same_retraining_obligation"),
    ({"explanation_share_pct": 0}, "same_explanation_obligation"),
])
def test_changed_obligation_is_ineligible_even_if_accounting_is_cheaper(change, failed):
    result = lifecycle_policy("cloud", execution="replayed_candidate", **change)
    assert result["comparison_gate"] == "ineligible"
    assert failed in result["comparison_failures"]
    assert not result["supported_energy_saving"]
    if change.get("demand_scale", 1) < 2:
        assert result["energy_delta_kwh"] < 0


def test_equal_energy_does_not_imply_equal_quality_evidence():
    supplied = lifecycle_policy("cloud", execution="replayed_candidate")
    absent = lifecycle_policy("cloud", execution="unreplayed_candidate")
    changed = lifecycle_policy("cloud", execution="changed_decision_candidate")
    assert supplied["terms_kwh"] == absent["terms_kwh"] == changed["terms_kwh"]
    assert absent["replay_mismatch_count"] is None
    assert absent["replay_rows"] == []
    assert absent["comparison_gate"] == "insufficient_quality_evidence"
    assert changed["replay_mismatch_count"] == 1
    assert changed["comparison_gate"] == "ineligible"
    assert not absent["supported_energy_saving"] and not changed["supported_energy_saving"]
    assert "does not establish deployment quality" in supplied["limitation"]


@pytest.mark.parametrize("track", TRACKS)
@pytest.mark.parametrize("execution", ("reference", "replayed_candidate", "unreplayed_candidate", "changed_decision_candidate"))
def test_finite_json_result_replays_exactly(track, execution):
    result = lifecycle_policy(track, execution=execution)
    restored = json.loads(json.dumps(result, allow_nan=False))
    assert lifecycle_policy(**restored["inputs"]) == restored


@pytest.mark.parametrize("kwargs", [
    {"execution": "fabricated"}, {"demand_scale": float("nan")},
    {"retraining_runs": True}, {"explanation_share_pct": -1},
])
def test_unsupported_controls_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        lifecycle_policy("tinyml", **kwargs)
