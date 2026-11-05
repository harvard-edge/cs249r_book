import math
import json

import pytest

from mlsysim.labs.experiments.v2_14_experiments import (
    SCENARIO_IDS,
    TRACK_IDS,
    conformal_quantile,
    conformal_experiment_payload,
    defense_experiment_payload,
    evaluate_conformal_fixture,
    evaluate_defense,
    evaluate_monitor,
    evaluate_selective_prediction,
    labeled_cohort_trace,
    monitor_experiment_payload,
    selective_experiment_payload,
    selective_fixture,
    summarize_trace,
)


@pytest.mark.parametrize("track_id", TRACK_IDS)
@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_all_track_and_scenario_endpoints_have_labeled_outcomes(track_id, scenario_id):
    trace = labeled_cohort_trace(track_id, scenario_id)
    assert len(trace) == 72
    assert {record.event_active for record in trace} == {False, True}
    assert all(record.prediction in (0, 1) and record.label in (0, 1) for record in trace)
    assert all(0.0 <= record.confidence <= 1.0 for record in trace)


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_concept_drift_can_change_errors_without_feature_drift(track_id):
    summaries = summarize_trace(labeled_cohort_trace(track_id, "concept_drift"))
    baseline = summaries[0]
    event = summaries[2]
    assert baseline.feature_total_variation == pytest.approx(0.0)
    assert event.feature_total_variation == pytest.approx(0.0)
    assert event.error_rate > baseline.error_rate


@pytest.mark.parametrize("scenario_id", ("covariate_shift", "corruption", "attack"))
def test_input_changes_are_computed_from_actual_bins(scenario_id):
    summaries = summarize_trace(labeled_cohort_trace("edge", scenario_id))
    assert summaries[2].feature_total_variation > 0.0
    assert summaries[2].error_rate > summaries[0].error_rate


def test_monitoring_changes_flags_not_underlying_outcomes():
    trace = labeled_cohort_trace("cloud", "concept_drift")
    strict = evaluate_monitor(
        trace, feature_threshold=0.10, error_threshold=0.90, label_delay_periods=2
    )
    sensitive = evaluate_monitor(
        trace, feature_threshold=0.10, error_threshold=0.20, label_delay_periods=2
    )
    assert strict.summaries == sensitive.summaries
    assert strict.first_detection_period is None
    assert sensitive.first_detection_period == 4
    assert sensitive.detection_delay_periods == 2


def test_delayed_correct_flag_is_not_counted_as_false_alarm():
    result = evaluate_monitor(
        labeled_cohort_trace("mobile", "concept_drift"),
        feature_threshold=0.20,
        error_threshold=0.25,
        label_delay_periods=4,
    )
    assert result.first_detection_period == 6
    assert result.false_alarms == 0


def test_monitor_counts_false_alarms_from_non_event_evidence():
    result = evaluate_monitor(
        labeled_cohort_trace("mobile", "covariate_shift"),
        feature_threshold=0.20,
        error_threshold=0.05,
        label_delay_periods=1,
    )
    # The fixed trace contains one ordinary error in each baseline/recovery
    # period, so an over-sensitive outcome threshold creates three false flags.
    assert result.false_alarms == 3


def test_feature_monitor_detects_covariate_shift_but_misses_concept_drift_without_labels():
    covariate = evaluate_monitor(
        labeled_cohort_trace("tinyml", "covariate_shift"),
        feature_threshold=0.20,
        error_threshold=1.0,
        label_delay_periods=3,
    )
    concept = evaluate_monitor(
        labeled_cohort_trace("tinyml", "concept_drift"),
        feature_threshold=0.20,
        error_threshold=1.0,
        label_delay_periods=3,
    )
    assert covariate.detection_delay_periods == 0
    assert concept.first_detection_period is None
    assert concept.missed_event_periods == 3


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_defense_tradeoff_uses_supplied_outcomes_and_costs(track_id):
    baseline = evaluate_defense(track_id, "baseline", "attack")
    robust = evaluate_defense(track_id, "robust_training", "attack")
    assert robust.stressed_accuracy > baseline.stressed_accuracy
    assert robust.clean_accuracy < baseline.clean_accuracy
    assert robust.mean_latency > baseline.mean_latency
    assert robust.mean_energy > baseline.mean_energy
    assert robust.mean_latency.check("[time]")
    assert robust.mean_energy.check("[energy]")
    assert robust.evidence_label == "illustrative supplied outcome fixture"


def test_defense_scope_is_reported_without_extrapolation():
    result = evaluate_defense("cloud", "input_filter", "concept_drift")
    assert not result.applicable
    assert "concept_drift" not in result.threat_scope
    assert result.stressed_accuracy is None


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_selective_prediction_uses_confidence_and_labels(track_id):
    examples = selective_fixture(track_id)
    low = evaluate_selective_prediction(examples, threshold=0.50, fallback_capacity=20)
    high = evaluate_selective_prediction(examples, threshold=0.90, fallback_capacity=20)
    assert high.coverage < low.coverage
    assert high.fallback_arrivals > low.fallback_arrivals
    assert high.selective_risk is not None
    assert low.selective_risk is not None
    assert high.selective_risk <= low.selective_risk
    assert sum(high.cohort_fallback_arrivals.values()) == high.fallback_arrivals


def test_fallback_capacity_cannot_change_risk_or_coverage():
    examples = selective_fixture("edge")
    scarce = evaluate_selective_prediction(examples, threshold=0.75, fallback_capacity=2)
    ample = evaluate_selective_prediction(examples, threshold=0.75, fallback_capacity=20)
    assert scarce.coverage == ample.coverage
    assert scarce.selective_risk == ample.selective_risk
    assert scarce.fallback_arrivals == ample.fallback_arrivals
    assert scarce.fallback_overflow > ample.fallback_overflow


def test_selective_prediction_reports_undefined_risk_when_nothing_is_accepted():
    result = evaluate_selective_prediction(
        selective_fixture("mobile"), threshold=1.0, fallback_capacity=20
    )
    assert result.accepted == 0
    assert result.selective_risk is None
    assert result.coverage == 0.0


def test_finite_conformal_quantile_uses_corrected_rank():
    rank, threshold = conformal_quantile([0.05, 0.10, 0.20, 0.30], alpha=0.20)
    assert rank == 4
    assert threshold == pytest.approx(0.30)


def test_conformal_small_sample_endpoint_requires_full_set():
    rank, threshold = conformal_quantile([0.1, 0.2, 0.3], alpha=0.10)
    assert rank == 4
    assert math.isinf(threshold)
    result = evaluate_conformal_fixture(
        [0.1, 0.2, 0.3],
        [0.05, 0.8],
        alpha=0.10,
        exchangeability_assumed=True,
    )
    assert result.full_set_required
    assert result.empirical_coverage == 1.0
    assert "not verified" in result.assumption_note


def test_all_evidence_payloads_are_strict_json_and_keep_exact_inputs():
    payloads = [
        monitor_experiment_payload(
            "edge", "concept_drift", feature_threshold=0.2,
            error_threshold=0.25, label_delay_periods=2,
        ),
        defense_experiment_payload("edge", "fallback_ensemble", "attack"),
        selective_experiment_payload("edge", threshold=0.8, fallback_capacity=4),
        conformal_experiment_payload(
            [0.1, 0.2, 0.3], [0.1, 0.8], alpha=0.1,
            exchangeability_assumed=True,
        ),
    ]
    for payload in payloads:
        json.dumps(payload, allow_nan=False)
        assert payload["inputs"]
    conformal = payloads[-1]
    assert conformal["inputs"]["alpha"] == 0.1
    assert conformal["result"]["full_set_required"]
    assert conformal["result"]["threshold"] is None


def test_conformal_empirical_coverage_is_separate_from_assumption():
    calibration = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    test_scores = [0.10, 0.20, 0.60, 0.70]
    assumed = evaluate_conformal_fixture(
        calibration, test_scores, alpha=0.20, exchangeability_assumed=True
    )
    shifted = evaluate_conformal_fixture(
        calibration, test_scores, alpha=0.20, exchangeability_assumed=False
    )
    assert assumed.empirical_coverage == shifted.empirical_coverage
    assert assumed.threshold == shifted.threshold
    assert assumed.exchangeability_assumed
    assert not shifted.exchangeability_assumed
    assert "empirical fixture coverage only" in shifted.assumption_note


@pytest.mark.parametrize("alpha", (0.0, 1.0, -0.1, 1.1))
def test_conformal_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        conformal_quantile([0.1, 0.2], alpha)


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_track_context_defines_volume2_fleet_roles_and_scenario(track_id):
    from mlsysim.labs.experiments.v2_14_experiments import track_context

    ctx = track_context(track_id)
    assert ctx["scenario_name"]
    assert ctx["device_role"]
    assert ctx["gateway_role"]
    assert ctx["backend_role"]
    assert ctx["fallback_destination"]
    assert ctx["unit"]


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_trace_event_interpretation_is_causal_to_scenario(scenario_id):
    from mlsysim.labs.experiments.v2_14_experiments import trace_event_interpretation

    text = trace_event_interpretation(scenario_id)
    assert isinstance(text, str) and len(text) > 20
    payload = monitor_experiment_payload(
        "edge", scenario_id, feature_threshold=0.2, error_threshold=0.25, label_delay_periods=2
    )
    assert payload["result"]["interpretation"] == text
    assert payload["result"]["unit"] == "scenes"


def test_defense_models_cost_boundary_without_inventing_scope_constraints():
    tinyml_ensemble = evaluate_defense("tinyml", "fallback_ensemble", "attack")
    assert tinyml_ensemble.applicable
    assert "Delegated to field gateway" in tinyml_ensemble.cost_boundary
    assert "unmodeled" in tinyml_ensemble.cost_boundary


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_selective_prediction_preserves_dynamic_units_and_cohort_breakdown(track_id):
    payload = selective_experiment_payload(track_id, threshold=0.80, fallback_capacity=5)
    res = payload["result"]
    assert res["unit"]
    assert res["cohort_fallback_arrivals"]
    assert sum(res["cohort_fallback_arrivals"].values()) == res["fallback_arrivals"]


def test_ordered_pipeline_blind_spots_and_legitimate_quarantine():
    from mlsysim.labs.experiments.v2_14_experiments import poisoning_experiment_payload
    runs = [poisoning_experiment_payload(layer_count=n, lineage=False)["result"] for n in range(1,5)]
    assert [run["retained_poison_records"] for run in runs] == [3, 2, 1, 1]
    assert [run["quarantined_legitimate_records"] for run in runs] == [0, 1, 1, 1]
    assert [run["promotion_blocked"] for run in runs] == [False, False, False, True]
    assert {run["clean_probe_correct"] for run in runs} == {11}
    assert {run["trigger_probe_correct"] for run in runs} == {0}
    # A post-training check blocks the candidate, rather than rewriting rows or outcomes.
    assert runs[2]["record_trace"] == runs[3]["record_trace"]


def test_lineage_bounds_historical_reversal_without_repairing_weights():
    from mlsysim.labs.experiments.v2_14_experiments import poisoning_experiment_payload
    no = poisoning_experiment_payload(layer_count=4, lineage=False)["result"]
    yes = poisoning_experiment_payload(layer_count=4, lineage=True)["result"]
    assert no["identified_descendants"] == [] and not no["descendant_scope_known"]
    assert yes["identified_descendants"] == ["checkpoint-from-batch", "deployed-descendant"]
    assert not yes["descendant_weights_repaired"]
    for key in ("record_trace", "retained_poison_records", "promotion_blocked"):
        assert no[key] == yes[key]


@pytest.mark.parametrize("value", [True, 1.0, 0, 5, "4"])
def test_poisoning_layers_reject_invalid_controls(value):
    from mlsysim.labs.experiments.v2_14_experiments import poisoning_experiment_payload
    with pytest.raises(ValueError):
        poisoning_experiment_payload(layer_count=value, lineage=True)


def test_generation_agreement_and_entropy_can_accept_same_false_answers():
    from mlsysim.labs.experiments.v2_14_experiments import generative_reliability_payload
    entropy = generative_reliability_payload(routing="entropy", evidence_source="current")["result"]
    agreement = generative_reliability_payload(routing="consistency", evidence_source="current")["result"]
    evidence = generative_reliability_payload(routing="evidence", evidence_source="current")["result"]
    assert entropy["current_reference_errors_accepted"] == agreement["current_reference_errors_accepted"] == 2
    assert evidence["current_reference_errors_accepted"] == 0
    assert agreement["decode_sequences"] == 3 * entropy["decode_sequences"]
    assert evidence["review_arrivals"] > entropy["review_arrivals"]
    assert evidence["evidence_checks"] == evidence["requests"]
    assert evidence["retrieval_gap_cases"] == evidence["generation_gap_cases"] == 1


def test_archived_evidence_can_support_obsolete_advice_without_changing_answers():
    from mlsysim.labs.experiments.v2_14_experiments import generative_reliability_payload
    current = generative_reliability_payload(routing="evidence", evidence_source="current")["result"]
    old = generative_reliability_payload(routing="evidence", evidence_source="archived")["result"]
    assert current["unsupported_accepted"] == old["unsupported_accepted"] == 0
    assert current["current_reference_errors_accepted"] == 0
    assert old["current_reference_errors_accepted"] == 1
    assert [row["answer"] for row in current["cases"]] == [row["answer"] for row in old["cases"]]


def test_combined_reviews_round_trip_replay_and_do_not_conflate_guarantees():
    from mlsysim.labs.experiments.v2_14_experiments import robustness_review_payload, competence_review_payload
    pipeline = robustness_review_payload("edge", "input_filter", "concept_drift", layer_count=4, lineage=True)
    assert pipeline["result"]["stressed_accuracy"] is None
    assert pipeline["result"]["poisoning"]["promotion_blocked"]
    competence = competence_review_payload([0.1,0.2,0.3], [0.2,0.8], alpha=0.1,
                                         exchangeability_assumed=False, routing="consistency", evidence_source="current")
    assert competence["result"]["full_set_required"]
    assert not competence["result"]["exchangeability_assumed"]
    assert competence["result"]["generation"]["current_reference_errors_accepted"] == 2
    for function, payload in ((robustness_review_payload,pipeline),(competence_review_payload,competence)):
        restored = json.loads(json.dumps(payload,allow_nan=False))
        assert function(**restored["inputs"]) == restored
