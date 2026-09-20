from dataclasses import replace

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v1_14_experiments import (
    TRACKS,
    CanaryEvidencePolicy,
    approximate_retraining_interval,
    evaluate_canary,
    evaluate_monitoring,
    evaluate_retraining,
    get_track_scenario,
    incident_experiment,
    monitoring_experiment,
    replay_capture,
    replay_experiment,
    replay_incident,
    restore_experiment,
    retraining_experiment,
    canary_experiment,
    serialize_experiment,
)


def _step_hours(track_id: str) -> float:
    scenario = get_track_scenario(track_id)
    return (
        scenario.monitoring.observations[1].event_time
        - scenario.monitoring.observations[0].event_time
    ).to("hour").magnitude


@pytest.mark.parametrize("track_id", TRACKS)
def test_every_track_is_an_explicit_outcome_labeled_scenario(track_id):
    scenario = get_track_scenario(track_id)

    assert scenario.track_id == track_id
    assert "Illustrative" in scenario.scenario_note
    assert len(scenario.monitoring.observations) == 12
    assert len(scenario.retraining.baseline_outcomes) == 12
    assert all(
        len(trace) == 12 for trace in scenario.retraining.candidate_outcome_traces
    )
    assert {row.cohort for row in scenario.canary_observations} == {
        "majority",
        "minority",
    }


@pytest.mark.parametrize("track_id", TRACKS)
def test_proxy_alerts_are_compared_with_delayed_outcomes(track_id):
    scenario = get_track_scenario(track_id)
    result = evaluate_monitoring(
        scenario.monitoring,
        proxy_threshold=0.30,
        sampling_interval=Q_(_step_hours(track_id), "hour"),
    )

    assert result.alert_event_times
    assert result.failure_episode_starts
    assert result.false_investigations == 0
    assert result.missed_failures == 0
    assert result.detection_delays[0].to("hour").magnitude == pytest.approx(0)
    expected_confirmation = (
        result.failure_episode_starts[0] + scenario.monitoring.label_delay
    )
    assert result.first_outcome_confirmation == expected_confirmation


def test_monitor_sensitivity_trades_false_investigations_against_misses_and_delay():
    monitoring = get_track_scenario("mobile").monitoring
    step = Q_(_step_hours("mobile"), "hour")
    sensitive = evaluate_monitoring(
        monitoring, proxy_threshold=0.15, sampling_interval=step
    )
    balanced = evaluate_monitoring(
        monitoring, proxy_threshold=0.30, sampling_interval=step
    )
    insensitive = evaluate_monitoring(
        monitoring, proxy_threshold=0.40, sampling_interval=step
    )

    assert sensitive.false_investigations > balanced.false_investigations
    assert sensitive.telemetry_cost > balanced.telemetry_cost
    assert insensitive.missed_failures > balanced.missed_failures
    assert insensitive.detection_delays[0] > balanced.detection_delays[0]


def test_monitoring_cadence_changes_observation_cost_without_changing_outcomes():
    monitoring = get_track_scenario("edge").monitoring
    original_outcomes = tuple(row.outcome_quality for row in monitoring.observations)
    step = _step_hours("edge")
    frequent = evaluate_monitoring(
        monitoring, proxy_threshold=0.30, sampling_interval=Q_(step, "hour")
    )
    sparse = evaluate_monitoring(
        monitoring, proxy_threshold=0.30, sampling_interval=Q_(2 * step, "hour")
    )

    assert frequent.observations_collected > sparse.observations_collected
    assert frequent.telemetry_cost > sparse.telemetry_cost
    assert tuple(row.outcome_quality for row in monitoring.observations) == original_outcomes


def test_monitoring_quality_floor_classifies_but_does_not_modify_outcomes():
    monitoring = get_track_scenario("tinyml").monitoring
    outcomes = tuple(row.outcome_quality for row in monitoring.observations)
    strict = replace(monitoring, quality_floor=0.91)
    relaxed = replace(monitoring, quality_floor=0.85)

    strict_result = evaluate_monitoring(
        strict, proxy_threshold=0.30, sampling_interval=Q_(12, "hour")
    )
    relaxed_result = evaluate_monitoring(
        relaxed, proxy_threshold=0.30, sampling_interval=Q_(12, "hour")
    )

    assert strict_result.missed_failures != relaxed_result.missed_failures
    assert tuple(row.outcome_quality for row in strict.observations) == outcomes
    assert tuple(row.outcome_quality for row in relaxed.observations) == outcomes


@pytest.mark.parametrize("track_id", TRACKS)
def test_retraining_jobs_consume_resources_and_change_version_only_after_completion(track_id):
    scenario = get_track_scenario(track_id)
    step = _step_hours(track_id)
    result = evaluate_retraining(
        scenario.retraining,
        policy="scheduled",
        scheduled_interval=Q_(5 * step, "hour"),
    )

    assert result.jobs
    first_job = result.jobs[0]
    assert first_job.completed_at > first_job.started_at
    assert result.compute_resource_hours.to("hour").magnitude > 0
    assert result.retraining_cost.to("USD").magnitude > 0
    completion_index = next(
        index
        for index, event_time in enumerate(scenario.retraining.event_times)
        if event_time >= first_job.completed_at
    )
    assert result.deployed_outcomes[:completion_index] == (
        scenario.retraining.baseline_outcomes[:completion_index]
    )
    assert first_job.promoted_at == first_job.completed_at


@pytest.mark.parametrize("track_id", TRACKS)
def test_delayed_evidence_trigger_uses_fewer_jobs_but_accepts_more_staleness(track_id):
    scenario = get_track_scenario(track_id)
    step = _step_hours(track_id)
    scheduled = evaluate_retraining(
        scenario.retraining,
        policy="scheduled",
        scheduled_interval=Q_(5 * step, "hour"),
    )
    triggered = evaluate_retraining(
        scenario.retraining,
        policy="evidence_triggered",
    )

    assert len(triggered.jobs) <= len(scheduled.jobs)
    assert triggered.retraining_cost <= scheduled.retraining_cost
    assert triggered.stale_outcome_loss >= scheduled.stale_outcome_loss


def test_promotion_floor_changes_acceptance_without_changing_candidate_evidence():
    retraining = get_track_scenario("cloud").retraining
    original_candidate = retraining.candidate_outcome_traces[0]
    accepted = evaluate_retraining(
        retraining,
        policy="scheduled",
        scheduled_interval=Q_(10, "hour"),
    )
    rejected = evaluate_retraining(
        replace(retraining, candidate_promotion_floor=0.99),
        policy="scheduled",
        scheduled_interval=Q_(10, "hour"),
    )

    assert accepted.promotions > rejected.promotions
    assert retraining.candidate_outcome_traces[0] == original_candidate
    assert rejected.jobs[0].candidate_quality_at_completion == accepted.jobs[0].candidate_quality_at_completion


def test_square_root_interval_has_time_units_and_expected_scaling():
    baseline = approximate_retraining_interval(
        Q_(100, "USD"), Q_(2, "USD/hour**2")
    )
    expensive = approximate_retraining_interval(
        Q_(400, "USD"), Q_(2, "USD/hour**2")
    )

    assert baseline.to("hour").magnitude == pytest.approx(10)
    assert expensive.to("hour").magnitude == pytest.approx(20)


@pytest.mark.parametrize("track_id", TRACKS)
def test_zero_canary_traffic_has_zero_exposure_and_no_candidate_evidence(track_id):
    scenario = get_track_scenario(track_id)
    result = evaluate_canary(
        scenario.canary_observations,
        canary_fraction=0,
        label_delay=scenario.monitoring.label_delay,
        evidence_policy=scenario.canary_policy,
    )

    assert result.exposed_requests == 0
    assert result.exposed_failures == 0
    assert result.candidate_labeled == 0
    assert result.candidate_observed_rate is None
    assert not result.evidence_requirements_met
    assert result.evidence_ready_at is None
    assert result.decision == "continue"
    assert result.decision_time is None


def test_more_canary_traffic_accumulates_evidence_faster_and_exposes_more_failures():
    scenario = get_track_scenario("mobile")
    small = evaluate_canary(
        scenario.canary_observations,
        canary_fraction=0.25,
        label_delay=scenario.monitoring.label_delay,
        evidence_policy=scenario.canary_policy,
        stop_on_decision=False,
    )
    large = evaluate_canary(
        scenario.canary_observations,
        canary_fraction=0.50,
        label_delay=scenario.monitoring.label_delay,
        evidence_policy=scenario.canary_policy,
        stop_on_decision=False,
    )

    assert large.candidate_labeled > small.candidate_labeled
    assert large.exposed_requests > small.exposed_requests
    assert large.exposed_failures > small.exposed_failures
    assert large.evidence_ready_at < small.evidence_ready_at


def test_canary_margins_gate_decision_without_changing_observed_outcomes():
    scenario = get_track_scenario("mobile")
    ordinary = evaluate_canary(
        scenario.canary_observations,
        canary_fraction=0.25,
        label_delay=scenario.monitoring.label_delay,
        evidence_policy=scenario.canary_policy,
        stop_on_decision=False,
    )
    stringent_policy = replace(scenario.canary_policy, promotion_margin=0.50)
    stringent = evaluate_canary(
        scenario.canary_observations,
        canary_fraction=0.25,
        label_delay=scenario.monitoring.label_delay,
        evidence_policy=stringent_policy,
        stop_on_decision=False,
    )

    assert ordinary.candidate_labeled == stringent.candidate_labeled
    assert ordinary.candidate_successes == stringent.candidate_successes
    assert ordinary.candidate_observed_rate == stringent.candidate_observed_rate
    assert ordinary.decision == "promote"
    assert stringent.decision == "continue"


def test_cohort_coverage_is_a_real_gate():
    scenario = get_track_scenario("edge")
    impossible = CanaryEvidencePolicy(
        minimum_candidate_labels=1,
        minimum_baseline_labels=1,
        minimum_labels_per_cohort=10_000,
        required_cohorts=scenario.canary_policy.required_cohorts,
        promotion_margin=0,
        rollback_margin=0,
    )
    result = evaluate_canary(
        scenario.canary_observations,
        canary_fraction=0.5,
        label_delay=scenario.monitoring.label_delay,
        evidence_policy=impossible,
    )

    assert result.candidate_labeled > 0
    assert not result.evidence_requirements_met
    assert result.decision == "continue"


@pytest.mark.parametrize("track_id", TRACKS)
def test_incident_replay_uses_sequential_stage_times_and_direct_costs(track_id):
    scenario = get_track_scenario(track_id)
    result = replay_incident(
        scenario.monitoring,
        proxy_threshold=0.30,
        sampling_interval=Q_(_step_hours(track_id), "hour"),
        policy=scenario.incident_policy,
    )

    assert result.detected_at >= result.incident_started_at
    assert result.triage_completed_at == result.detected_at + scenario.incident_policy.triage_duration
    assert result.rollback_completed_at == result.triage_completed_at + scenario.incident_policy.rollback_duration
    assert result.recovery_validated_at == result.rollback_completed_at + scenario.incident_policy.validation_duration
    assert result.total_incident_cost == result.exposure_cost + result.response_labor_cost
    assert result.exposed_requests > 0
    assert result.verified_recovery
    assert result.met_recovery_objective


def test_faster_rollback_reduces_exposure_without_changing_detection_or_recovered_quality():
    scenario = get_track_scenario("cloud")
    slow = replay_incident(
        scenario.monitoring,
        proxy_threshold=0.30,
        sampling_interval=Q_(2, "hour"),
        policy=replace(scenario.incident_policy, rollback_duration=Q_(30, "minute")),
    )
    fast = replay_incident(
        scenario.monitoring,
        proxy_threshold=0.30,
        sampling_interval=Q_(2, "hour"),
        policy=replace(scenario.incident_policy, rollback_duration=Q_(2, "minute")),
    )

    assert fast.detected_at == slow.detected_at
    assert fast.exposed_requests < slow.exposed_requests
    assert fast.exposure_cost < slow.exposure_cost
    assert fast.recovered_quality == slow.recovered_quality


def test_an_undetected_incident_accumulates_exposure_without_fake_recovery():
    scenario = get_track_scenario("tinyml")
    result = replay_incident(
        scenario.monitoring,
        proxy_threshold=1.0,
        sampling_interval=Q_(12, "hour"),
        policy=scenario.incident_policy,
    )

    assert result.detected_at is None
    assert result.exposed_requests > 0
    assert result.exposure_cost.to("USD").magnitude > 0
    assert result.response_labor_cost.to("USD").magnitude == 0
    assert not result.verified_recovery
    assert not result.met_recovery_objective


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (lambda: get_track_scenario("watch"), "unknown track_id"),
        (
            lambda: evaluate_monitoring(
                get_track_scenario("mobile").monitoring,
                proxy_threshold=-0.1,
                sampling_interval=Q_(1, "hour"),
            ),
            "proxy_threshold",
        ),
        (
            lambda: evaluate_canary(
                get_track_scenario("mobile").canary_observations,
                canary_fraction=1.1,
                label_delay=Q_(1, "hour"),
                evidence_policy=get_track_scenario("mobile").canary_policy,
            ),
            "canary_fraction",
        ),
        (
            lambda: evaluate_retraining(
                get_track_scenario("mobile").retraining,
                policy="scheduled",
            ),
            "scheduled_interval",
        ),
    ],
)
def test_invalid_inputs_fail_clearly(action, message):
    with pytest.raises(ValueError, match=message):
        action()


@pytest.mark.parametrize("track_id", TRACKS)
def test_json_results_restore_and_replay_from_exact_inputs(track_id):
    step = _step_hours(track_id)
    results = (
        ("v1_14.monitoring", monitoring_experiment(
            track_id, proxy_threshold=0.30, sampling_interval_hours=step,
        )),
        ("v1_14.retraining", retraining_experiment(
            track_id, policy="scheduled", scheduled_interval_hours=5 * step,
        )),
        ("v1_14.canary", canary_experiment(
            track_id, canary_fraction=0.25, stop_on_decision=False,
        )),
        ("v1_14.incident", incident_experiment(
            track_id, proxy_threshold=0.30, sampling_interval_hours=step,
            rollback_minutes=5, affected_fraction=0.25,
        )),
    )

    for model_key, result in results:
        restored = restore_experiment(serialize_experiment(result))
        assert restored == result
        assert replay_experiment(model_key, restored["inputs"]) == result


def test_typed_dataclass_serializer_preserves_quantity_units():
    scenario = get_track_scenario("tinyml")
    restored = restore_experiment(serialize_experiment(scenario.monitoring))

    assert restored["label_delay"] == {"unit": "h", "value": 36}
    assert restored["cost_per_observation"]["unit"] == "USD"


def test_unknown_replay_model_fails_clearly():
    with pytest.raises(ValueError, match="unknown model_key"):
        replay_experiment("v1_14.score", {})


def test_unavailable_failure_results_are_finite_json_with_explicit_status():
    result = incident_experiment(
        "tinyml",
        proxy_threshold=1.0,
        sampling_interval_hours=12,
        rollback_minutes=5,
        affected_fraction=0.25,
    )
    restored = restore_experiment(serialize_experiment(result))

    assert restored["detection_status"] == "unavailable"
    assert restored["recovery_status"] == "unavailable"
    assert restored["detected_hours"] is None
    assert restored["mean_time_to_recovery_hours"] is None


def test_capture_replay_restores_baseline_result_chosen_and_alternatives():
    baseline = canary_experiment("cloud", canary_fraction=0, stop_on_decision=False)
    rejected = canary_experiment("cloud", canary_fraction=0.25, stop_on_decision=False)
    other = canary_experiment("cloud", canary_fraction=0.50, stop_on_decision=False)
    snapshot = {
        "model_key": "v1_14.canary",
        "baseline": baseline,
        "result": rejected,
        "chosen_result": baseline,
        "alternatives": [baseline, rejected, other],
        "decision": "none",
        "result_role": "rejected alternative",
    }

    replayed = replay_capture(restore_experiment(serialize_experiment(snapshot)))

    assert replayed["baseline"] == baseline
    assert replayed["result"] == rejected
    assert replayed["chosen_result"] == baseline
    assert replayed["alternatives"] == [baseline, rejected, other]
    assert replayed["decision"] == "none"
    assert replayed["result_role"] == "rejected alternative"
