"""Causal tests for the Volume II responsible-AI experiments."""

from __future__ import annotations

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v2_16_experiments import (
    PRIVACY_PROFILES,
    TRACK_PROFILES,
    assess_fairness,
    audit_evidence,
    check_privacy_profile,
    evaluate_capacity,
    mechanical_gate,
    record_release_decision,
    scored_fixture,
    track_profile,
)


TRACKS = tuple(TRACK_PROFILES)


@pytest.mark.parametrize("track_id", TRACKS)
def test_scored_fixtures_are_coherent_and_all_track_endpoints_run(track_id):
    cases = scored_fixture(track_id)
    assert len(cases) == 24
    assert len({case.subgroup for case in cases}) == 2
    assert all(case.true_label in {0, 1} for case in cases)
    assert all(0.0 <= case.score <= 1.0 for case in cases)
    assert all(1 <= case.sample_rank <= 12 for case in cases)

    assessment = assess_fairness(
        track_id,
        threshold=0.5,
        criterion="equal_opportunity",
        allowed_gap=0.35,
    )
    capacity = evaluate_capacity(track_id, explanation_share=0.01, review_share=0.001)
    audit = audit_evidence(
        track_id,
        threshold=0.5,
        criterion="equal_opportunity",
        allowed_gap=0.35,
        sampling_fraction=1.0,
        label_delay=Q_(1, "hour"),
    )
    privacy = check_privacy_profile("pseudonymous_events", require_case_review=True)

    assert assessment.criterion_passes
    assert assessment.ground_truth_positive_cases == 12
    assert capacity.feasible
    assert audit.observed_assessment is not None
    assert privacy.passes_requirements
    assert mechanical_gate(assessment, capacity, audit, privacy).passes


@pytest.mark.parametrize("track_id", TRACKS)
def test_each_track_has_a_failed_mechanical_case(track_id):
    assessment = assess_fairness(
        track_id,
        threshold=0.5,
        criterion="equal_opportunity",
        allowed_gap=0.01,
    )
    capacity = evaluate_capacity(
        track_id,
        explanation_share=1.0,
        review_share=1.0,
        explanation_workers=1,
        reviewers=1,
    )
    audit = audit_evidence(
        track_id,
        threshold=0.5,
        criterion="equal_opportunity",
        allowed_gap=0.01,
        sampling_fraction=0.05,
        label_delay=Q_(80, "hour"),
    )
    privacy = check_privacy_profile(
        "full_review_trace",
        require_case_review=True,
        max_retention=Q_(30, "day"),
        max_deletion_window=Q_(7, "day"),
    )

    gate = mechanical_gate(assessment, capacity, audit, privacy)
    assert not assessment.criterion_passes
    assert not capacity.feasible
    assert audit.observed_assessment is None
    assert not privacy.passes_requirements
    assert not gate.passes
    assert len(gate.violations) == 4


def test_allowed_gap_and_normative_criterion_do_not_rewrite_outcomes():
    strict = assess_fairness(
        "mobile", threshold=0.5, criterion="equal_opportunity", allowed_gap=0.01
    )
    permissive = assess_fairness(
        "mobile", threshold=0.5, criterion="demographic_parity", allowed_gap=0.50
    )

    assert strict.subgroup_metrics == permissive.subgroup_metrics
    assert strict.ground_truth_positive_cases == permissive.ground_truth_positive_cases
    assert strict.predicted_positive_cases == permissive.predicted_positive_cases
    assert strict.false_negative_cases == permissive.false_negative_cases
    assert not strict.criterion_passes
    assert permissive.criterion_passes


def test_scored_fixture_exposes_a_real_fairness_criterion_tradeoff():
    demographic = assess_fairness(
        "mobile", threshold=0.5, criterion="demographic_parity", allowed_gap=0.15
    )
    predictive = assess_fairness(
        "mobile", threshold=0.5, criterion="predictive_parity", allowed_gap=0.15
    )

    # Both policies inspect the exact same decisions.  Their different result
    # comes from the selected normative criterion, not a generated score bonus.
    assert demographic.subgroup_metrics == predictive.subgroup_metrics
    assert demographic.criterion_gap == pytest.approx(1 / 6)
    assert predictive.criterion_gap == pytest.approx(4 / 35)
    assert not demographic.criterion_passes
    assert predictive.criterion_passes


def test_threshold_replays_the_same_scores_and_labels():
    low = assess_fairness(
        "edge", threshold=0.35, criterion="equalized_odds", allowed_gap=1.0
    )
    high = assess_fairness(
        "edge", threshold=0.70, criterion="equalized_odds", allowed_gap=1.0
    )

    assert low.ground_truth_positive_cases == high.ground_truth_positive_cases == 12
    assert low.predicted_positive_cases > high.predicted_positive_cases
    assert low.false_negative_cases < high.false_negative_cases


def test_no_predicted_positives_makes_predictive_parity_undefined():
    assessment = assess_fairness(
        "edge", threshold=1.0, criterion="predictive_parity", allowed_gap=1.0
    )

    assert assessment.predicted_positive_cases == 0
    assert assessment.criterion_gap is None
    assert not assessment.criterion_passes
    assert all(
        metrics.positive_predictive_value is None
        for metrics in assessment.subgroup_metrics
    )
    audit = audit_evidence(
        "edge",
        threshold=1.0,
        criterion="predictive_parity",
        allowed_gap=1.0,
        sampling_fraction=1.0,
        label_delay=Q_(1, "hour"),
    )
    gate = mechanical_gate(
        assessment,
        evaluate_capacity("edge", explanation_share=0.01, review_share=0.001),
        audit,
        check_privacy_profile("pseudonymous_events", require_case_review=True),
    )
    assert not gate.passes
    assert any("undefined" in violation for violation in gate.violations)


def test_small_audit_sample_with_no_negatives_is_insufficient_evidence():
    audit = audit_evidence(
        "mobile",
        threshold=0.5,
        criterion="equalized_odds",
        allowed_gap=1.0,
        sampling_fraction=0.10,
        label_delay=Q_(1, "hour"),
    )

    assert audit.observed_assessment is not None
    assert audit.observed_assessment.criterion_gap is None
    assert not audit.observed_assessment.criterion_passes
    assert any("undefined" in limitation for limitation in audit.limitations)


def test_sampling_and_label_delay_change_evidence_not_population_outcomes():
    full = audit_evidence(
        "cloud",
        threshold=0.5,
        criterion="equal_opportunity",
        allowed_gap=0.35,
        sampling_fraction=1.0,
        label_delay=Q_(1, "hour"),
    )
    sparse_delayed = audit_evidence(
        "cloud",
        threshold=0.5,
        criterion="equal_opportunity",
        allowed_gap=0.35,
        sampling_fraction=0.25,
        label_delay=Q_(48, "hour"),
    )

    assert full.population_assessment == sparse_delayed.population_assessment
    assert sparse_delayed.eligible_cases < full.eligible_cases
    assert sparse_delayed.sampled_cases < full.sampled_cases
    assert sparse_delayed.limitations


def test_more_explanation_and_review_coverage_can_create_backlogs():
    light = evaluate_capacity("tinyml", explanation_share=0.01, review_share=0.001)
    heavy = evaluate_capacity("tinyml", explanation_share=1.0, review_share=0.20)

    assert light.explanation_capacity == heavy.explanation_capacity
    assert light.review_capacity == heavy.review_capacity
    assert heavy.explanation_arrivals > light.explanation_arrivals
    assert heavy.review_arrivals > light.review_arrivals
    assert heavy.review_backlog.m_as("count") > light.review_backlog.m_as("count")
    assert light.feasible
    assert not heavy.feasible


def test_privacy_uses_specified_profiles_instead_of_a_percentage():
    aggregate = check_privacy_profile("aggregate_counts", require_case_review=False)
    aggregate_case_review = check_privacy_profile("aggregate_counts", require_case_review=True)
    event_profile = check_privacy_profile("pseudonymous_events", require_case_review=True)
    full_trace = check_privacy_profile("full_review_trace", require_case_review=True)

    assert set(PRIVACY_PROFILES) == {
        "aggregate_counts",
        "pseudonymous_events",
        "full_review_trace",
    }
    assert aggregate.passes_requirements
    assert not aggregate.supports_case_review
    assert not aggregate_case_review.passes_requirements
    assert event_profile.passes_requirements
    assert not full_trace.passes_requirements
    assert full_trace.profile.raw_content_retained


def test_release_choice_records_normative_decision_without_changing_gate():
    assessment = assess_fairness(
        "tinyml", threshold=0.5, criterion="equal_opportunity", allowed_gap=0.35
    )
    capacity = evaluate_capacity("tinyml", explanation_share=0.01, review_share=0.001)
    audit = audit_evidence(
        "tinyml",
        threshold=0.5,
        criterion="equal_opportunity",
        allowed_gap=0.35,
        sampling_fraction=1.0,
        label_delay=Q_(1, "hour"),
    )
    privacy = check_privacy_profile("pseudonymous_events", require_case_review=True)
    gate = mechanical_gate(assessment, capacity, audit, privacy)

    release = record_release_decision(
        gate,
        choice="release",
        rejected_alternative="defer",
        owner="fleet responsibility lead",
        remedy="disable the affected path and provide appeal",
        reevaluation_trigger="subgroup gap exceeds 0.35",
    )
    restrict = record_release_decision(
        gate,
        choice="restrict",
        rejected_alternative="release",
        owner="fleet responsibility lead",
        remedy="disable the affected path and provide appeal",
        reevaluation_trigger="subgroup gap exceeds 0.35",
    )

    assert release.evidence_gate is gate
    assert restrict.evidence_gate is gate
    assert release.choice != restrict.choice
    assert release.decision_record_complete
    assert restrict.decision_record_complete


def test_release_record_reports_missing_qualitative_decisions():
    gate = mechanical_gate(
        assess_fairness("mobile", threshold=0.5, criterion="equal_opportunity", allowed_gap=0.35),
        evaluate_capacity("mobile", explanation_share=0.01, review_share=0.001),
        audit_evidence(
            "mobile",
            threshold=0.5,
            criterion="equal_opportunity",
            allowed_gap=0.35,
            sampling_fraction=1.0,
            label_delay=Q_(1, "hour"),
        ),
        check_privacy_profile("pseudonymous_events", require_case_review=True),
    )
    decision = record_release_decision(
        gate,
        choice="defer",
        rejected_alternative="defer",
        owner="",
        remedy="",
        reevaluation_trigger="",
    )

    assert not decision.decision_record_complete
    assert decision.missing_fields == (
        "distinct rejected alternative",
        "owner",
        "remedy",
        "reevaluation trigger",
    )


@pytest.mark.parametrize("track_id", TRACKS)
def test_track_profiles_define_fleet_roles_and_distinct_remedies(track_id):
    profile = track_profile(track_id)
    assert "fleet_role" in profile and profile["fleet_role"]
    assert "remedies" in profile and len(profile["remedies"]) >= 3
    assert "" in profile["remedies"].values()


def test_privacy_check_adapts_scope_to_fleet_role():
    tinyml_check = check_privacy_profile(
        "pseudonymous_events", track_id="tinyml", require_case_review=True
    )
    cloud_check = check_privacy_profile(
        "pseudonymous_events", track_id="cloud", require_case_review=True
    )
    assert tinyml_check.track_id == "tinyml"
    assert cloud_check.track_id == "cloud"
    assert tinyml_check.scope != cloud_check.scope
    assert "TinyML" in tinyml_check.scope or "endpoint" in tinyml_check.scope


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: track_profile("desktop"), "track_id"),
        (
            lambda: assess_fairness(
                "mobile", threshold=1.2, criterion="equal_opportunity", allowed_gap=0.1
            ),
            "threshold",
        ),
        (
            lambda: audit_evidence(
                "mobile",
                threshold=0.5,
                criterion="equal_opportunity",
                allowed_gap=0.1,
                sampling_fraction=-0.1,
            ),
            "sampling_fraction",
        ),
        (
            lambda: evaluate_capacity("mobile", explanation_share=0.1, review_share=0.1, reviewers=0),
            "reviewers",
        ),
        (
            lambda: check_privacy_profile(
                "pseudonymous_events", track_id="desktop", require_case_review=True
            ),
            "track_id",
        ),
    ],
)
def test_invalid_inputs_fail_loudly(call, match):
    with pytest.raises(ValueError, match=match):
        call()
