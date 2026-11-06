"""Regression checks for request parity and observable responsibility evidence."""

import json

import pytest

from mlsysim.labs.experiments.v1_14_experiments import parity_experiment, replay_experiment
from mlsysim.labs.experiments.v1_15_experiments import fairness_criteria, selective_feedback


@pytest.mark.parametrize("track", ("tinyml", "mobile", "edge", "cloud"))
def test_matching_marginals_do_not_establish_request_parity(track):
    swapped = parity_experiment(track, serving_transform="swapped")
    matched = parity_experiment(track, serving_transform="matched")
    assert swapped["aggregate_distribution_matches"]
    assert not swapped["request_parity_passes"]
    assert swapped["mismatch_count"] == swapped["request_count"]
    assert matched["request_parity_passes"]
    assert matched["mismatch_count"] == 0
    assert [r["raw_value"] for r in swapped["rows"]] == [r["raw_value"] for r in matched["rows"]]
    assert [r["training_value"] for r in swapped["rows"]] == [
        r["training_value"] for r in matched["rows"]
    ]


def test_fitted_state_is_required_for_compatible_retest():
    complete = parity_experiment("tinyml", release_bundle="compatible")
    weights = parity_experiment("tinyml", release_bundle="weights_only")
    missing = parity_experiment("tinyml", release_bundle="missing_state")
    assert complete["compatible_retest_passes"]
    assert weights["replay_available"] and not weights["compatible_retest_passes"]
    assert not missing["replay_available"] and missing["replayed_values"] is None
    assert not missing["compatible_retest_passes"]
    assert complete["release_bundle"]["weights"] == weights["release_bundle"]["weights"]
    assert len({r["release_identity"] for r in (complete, weights, missing)}) == 3
    assert replay_experiment("v1_14.parity", complete["inputs"]) == complete


@pytest.mark.parametrize("track", ("tinyml", "mobile", "edge", "cloud"))
def test_criterion_changes_gate_without_changing_audited_predictions(track):
    opportunity = fairness_criteria(track, criterion="equal_opportunity")
    parity = fairness_criteria(track, criterion="demographic_parity")
    assert opportunity["groups"] == parity["groups"]
    assert (
        opportunity["groups"]["reference"]["base_rate"]
        != opportunity["groups"]["affected"]["base_rate"]
    )
    assert opportunity["selected_gate"] == "pass"
    assert parity["selected_gate"] == "fail"
    assert opportunity["criterion_gates"]["equalized_odds"] == "pass"
    assert (
        fairness_criteria(track, minimum_group_count=21)["selected_gate"] == "insufficient_evidence"
    )
    assert "not a universal fairness verdict" in parity["limitation"]


def test_complete_independent_observation_recovers_population_denominators():
    routine = selective_feedback("edge")
    partial = selective_feedback("edge", audit_policy="independent_sample")
    complete = selective_feedback("edge", audit_policy="complete")
    assert not routine["population_outcomes_identified"]
    assert not partial["population_outcomes_identified"]
    assert complete["population_outcomes_identified"]
    assert complete["additional_audit_labels"] > partial["additional_audit_labels"] > 0
    for group in routine["groups"]:
        base = routine["groups"][group]
        final = complete["groups"][group]
        assert base["population_positive_rate"] is None
        assert base["unobserved_count"] > 0
        assert final["unobserved_count"] == 0
        assert final["observed_count"] == final["eligible_count"]
        assert final["routine_positive_rate"] == base["routine_positive_rate"]
        assert final["population_positive_rate"] != base["routine_positive_rate"]


def test_referral_selection_changes_observation_not_audited_population():
    refer_all = selective_feedback("cloud", referral_threshold=0, audit_policy="complete")
    refer_fewer = selective_feedback("cloud", referral_threshold=1.0, audit_policy="complete")
    for group in refer_all["groups"]:
        assert (
            refer_all["groups"][group]["population_positive_rate"]
            == refer_fewer["groups"][group]["population_positive_rate"]
        )
        assert (
            refer_all["groups"][group]["referred_count"]
            > refer_fewer["groups"][group]["referred_count"]
        )
        assert refer_fewer["groups"][group]["routine_positive_rate"] is None


@pytest.mark.parametrize(
    "runner,kwargs",
    [
        (parity_experiment, {"serving_transform": "unknown"}),
        (parity_experiment, {"release_bundle": "unknown"}),
        (fairness_criteria, {"criterion": "universal"}),
        (fairness_criteria, {"tolerance_pp": float("nan")}),
        (fairness_criteria, {"minimum_group_count": True}),
        (selective_feedback, {"audit_policy": "guessed"}),
        (selective_feedback, {"referral_threshold": -1}),
    ],
)
def test_bad_controls_fail_explicitly(runner, kwargs):
    with pytest.raises(ValueError):
        runner("tinyml", **kwargs)


def test_new_results_are_finite_json_and_replayable():
    for runner in (parity_experiment, fairness_criteria, selective_feedback):
        result = runner("mobile")
        restored = json.loads(json.dumps(result, allow_nan=False))
        assert restored == runner(**restored["inputs"])


def test_audited_threshold_frontier_preserves_outcomes_and_changes_errors():
    lower = fairness_criteria("tinyml", threshold=0.3)
    upper = fairness_criteria("tinyml", threshold=0.7)
    for group in lower["groups"]:
        low, high = lower["groups"][group], upper["groups"][group]
        assert low["base_rate"] == high["base_rate"]
        assert low["count"] == high["count"]
        assert low["tp"] > high["tp"]
        assert low["fp"] > high["fp"]
