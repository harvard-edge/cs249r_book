import math

import pytest

from mlsysim.labs.experiments.v2_13_experiments import (
    ArtifactCopy,
    IncidentEvidence,
    PrivacyEvent,
    SuppliedQualityObservation,
    ThreatControl,
    TrustMechanism,
    compose_privacy_events,
    evaluate_threat_coverage,
    evaluate_retention,
    evaluate_supplied_quality,
    evaluate_trust_latency,
    get_track_scenario,
    get_deletion_artifacts,
    get_privacy_epsilon_options,
    get_quality_observation,
    get_threat_control,
    get_trust_mechanism,
    propagate_deletion,
    release_bounded_mean,
)


def test_bounded_mean_uses_replace_one_sensitivity_and_seeded_laplace_noise():
    release = release_bounded_mean(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        lower_bound=0.0,
        upper_bound=1.0,
        epsilon=0.5,
        release_id="release-a",
        seed=7,
    )

    assert release.adjacency == "fixed-size replace-one over one bounded record"
    assert release.release_mode == "seeded classroom demonstration"
    assert not release.production_safe_randomness
    assert release.seed == 7
    assert release.true_mean == pytest.approx(0.5)
    assert release.sensitivity == pytest.approx(0.2)
    assert release.laplace_scale == pytest.approx(0.4)
    assert 0.0 <= release.published_release <= 1.0


def test_stronger_privacy_increases_noise_scale_for_same_bounded_query():
    common = dict(
        values=[2.0, 4.0, 6.0, 8.0],
        lower_bound=0.0,
        upper_bound=10.0,
        release_id="mean",
        seed=11,
        clip_output=False,
    )
    stronger = release_bounded_mean(epsilon=0.25, **common)
    weaker = release_bounded_mean(epsilon=1.0, **common)

    assert stronger.sensitivity == weaker.sensitivity
    assert stronger.laplace_scale == pytest.approx(4 * weaker.laplace_scale)
    assert abs(stronger.noise) == pytest.approx(4 * abs(weaker.noise))


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"values": []},
        {"epsilon": 0.0},
        {"epsilon": math.inf},
        {"epsilon": math.nan},
        {"lower_bound": -math.inf},
        {"upper_bound": math.inf},
        {"values": [0.0, math.nan]},
        {"values": [0.0, math.inf], "upper_bound": math.inf},
        {"lower_bound": 1.0, "upper_bound": 1.0},
        {"values": [0.0, 2.0], "lower_bound": 0.0, "upper_bound": 1.0},
    ],
)
def test_bounded_mean_rejects_invalid_guarantee_inputs(bad_kwargs):
    kwargs = dict(
        values=[0.0, 1.0],
        lower_bound=0.0,
        upper_bound=1.0,
        epsilon=1.0,
        release_id="release-a",
        seed=1,
    )
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        release_bounded_mean(**kwargs)


def test_basic_composition_counts_releases_but_not_fixed_output_operations():
    result = compose_privacy_events(
        [
            PrivacyEvent("training-release", "dp_release", epsilon=0.8),
            PrivacyEvent(
                "dashboard",
                "fixed_release_postprocessing",
                source_release_id="training-release",
            ),
            PrivacyEvent(
                "query-1",
                "fixed_model_query",
                source_release_id="training-release",
            ),
            PrivacyEvent("second-release", "dp_release", epsilon=0.35),
        ]
    )

    assert result.total_epsilon == pytest.approx(1.15)
    assert result.accounted_release_ids == ("training-release", "second-release")
    assert result.accounted_release_count == 2
    assert result.zero_additional_training_loss_ids == ("dashboard", "query-1")
    assert result.zero_additional_training_loss_count == 2
    assert result.guarantee_complete


def test_new_unaccounted_private_access_invalidates_complete_dp_claim():
    result = compose_privacy_events(
        [
            PrivacyEvent("release", "dp_release", epsilon=0.5),
            PrivacyEvent("fine-tune", "new_private_data_access"),
        ]
    )

    assert result.total_epsilon == pytest.approx(0.5)
    assert result.unguaranteed_private_access_ids == ("fine-tune",)
    assert not result.guarantee_complete


def test_fixed_model_query_must_reference_an_existing_release():
    with pytest.raises(ValueError, match="earlier DP release"):
        compose_privacy_events(
            [PrivacyEvent("query", "fixed_model_query", source_release_id="missing")]
        )


@pytest.mark.parametrize("epsilon", [math.nan, math.inf, -math.inf])
def test_composition_rejects_nonfinite_epsilon(epsilon):
    with pytest.raises(ValueError, match="finite"):
        compose_privacy_events(
            [PrivacyEvent("release", "dp_release", epsilon=epsilon)]
        )


def test_threat_coverage_is_categorical_and_independent_of_latency():
    transport = ThreatControl(
        name="authenticated encryption",
        covered_threats=frozenset({"network_observer"}),
        protected_boundary="data in transit",
    )
    covered = evaluate_threat_coverage("network_observer", transport)
    uncovered = evaluate_threat_coverage("host_operator", transport)

    assert covered.covered
    assert not uncovered.covered
    assert uncovered.uncovered_capabilities == ("host_operator",)
    assert not hasattr(covered, "latency_ms")


def test_combined_adversary_requires_every_named_capability():
    enclave = ThreatControl(
        name="confidential execution",
        covered_threats=frozenset({"host_operator"}),
        protected_boundary="runtime memory",
    )
    result = evaluate_threat_coverage(
        "host_operator", enclave, required_capabilities=["physical_attacker"]
    )
    assert not result.covered
    assert result.uncovered_capabilities == ("physical_attacker",)


def test_trust_deadline_and_threat_coverage_are_separate_outcomes():
    mechanism = TrustMechanism(
        name="session enclave",
        protected_threats=frozenset({"host_operator"}),
        fixed_overhead_ms=20.0,
        per_request_overhead_ms=2.0,
    )
    amortized = evaluate_trust_latency(
        mechanism,
        adversary="host_operator",
        base_latency_ms=10.0,
        requests_per_session=10,
        deadline_ms=15.0,
    )
    cold = evaluate_trust_latency(
        mechanism,
        adversary="host_operator",
        base_latency_ms=10.0,
        requests_per_session=1,
        deadline_ms=15.0,
    )

    assert amortized.protected and cold.protected
    assert not amortized.meets_deadline
    assert not cold.meets_deadline
    assert amortized.total_latency_ms == pytest.approx(14.0)
    assert amortized.average_latency_ms == amortized.total_latency_ms
    assert amortized.cold_latency_ms == cold.cold_latency_ms
    assert amortized.cold_latency_ms == pytest.approx(
        10.0 + mechanism.per_request_overhead_ms + mechanism.fixed_overhead_ms
    )
    assert amortized.steady_latency_ms == pytest.approx(
        10.0 + mechanism.per_request_overhead_ms
    )
    assert amortized.steady_latency_ms < 15.0 < amortized.cold_latency_ms
    assert amortized.average_latency_ms < cold.average_latency_ms


def test_noncausal_deadline_does_not_change_threat_coverage_or_latency():
    mechanism = TrustMechanism(
        name="isolated runtime",
        protected_threats=frozenset({"host_operator"}),
        fixed_overhead_ms=4.0,
        per_request_overhead_ms=1.0,
    )
    tight = evaluate_trust_latency(
        mechanism,
        adversary="host_operator",
        base_latency_ms=8.0,
        requests_per_session=2,
        deadline_ms=10.0,
    )
    loose = evaluate_trust_latency(
        mechanism,
        adversary="host_operator",
        base_latency_ms=8.0,
        requests_per_session=2,
        deadline_ms=20.0,
    )

    assert tight.total_latency_ms == loose.total_latency_ms
    assert tight.protected == loose.protected
    assert not tight.meets_deadline and loose.meets_deadline


def test_deletion_traverses_branches_and_accumulates_dependency_time():
    result = propagate_deletion(
        [
            ArtifactCopy("source", None, 1.0, incident_evidence_events=2),
            ArtifactCopy("cache", "source", 3.0, incident_evidence_events=4),
            ArtifactCopy("backup", "source", 8.0, incident_evidence_events=5),
            ArtifactCopy("export", "cache", 2.0, incident_evidence_events=1),
        ],
        source_artifact_id="source",
        deletion_sla_hours=10.0,
    )

    completion = {
        status.artifact_id: status.deletion_completed_at_hours
        for status in result.statuses
    }
    assert completion == {"source": 1.0, "backup": 9.0, "cache": 4.0, "export": 6.0}
    assert result.complete
    assert result.completion_time_hours == pytest.approx(9.0)
    assert result.incident_evidence_events_lost == 12


def test_untracked_copy_prevents_deletion_proof_without_inventing_completion():
    result = propagate_deletion(
        [
            ArtifactCopy("source", None, 1.0),
            ArtifactCopy("shadow", "source", 2.0, tracked=False),
        ],
        source_artifact_id="source",
        deletion_sla_hours=24.0,
    )

    assert not result.complete
    assert result.completion_time_hours is None
    assert result.untracked_artifact_ids == ("shadow",)


def test_shorter_retention_expires_more_incident_evidence():
    evidence = [
        IncidentEvidence("recent-a", "gateway", 2.0),
        IncidentEvidence("recent-b", "gateway", 12.0),
        IncidentEvidence("old", "archive", 48.0),
    ]
    short = evaluate_retention(evidence, retention_hours=8.0)
    long = evaluate_retention(evidence, retention_hours=72.0)

    assert short.retained_event_ids == ("recent-a",)
    assert short.retained_event_count == 1
    assert short.expired_event_ids == ("recent-b", "old")
    assert short.expired_event_count == 2
    assert len(short.retained_event_ids) < len(long.retained_event_ids)
    assert long.retained_events_by_artifact == (("archive", 1), ("gateway", 2))


def test_known_late_deletion_reports_actual_completion_and_sla_failure():
    result = propagate_deletion(
        [ArtifactCopy("source", None, 5.0), ArtifactCopy("backup", "source", 8.0)],
        source_artifact_id="source",
        deletion_sla_hours=10.0,
    )

    assert not result.complete
    assert result.completion_time_hours == pytest.approx(13.0)


def test_supplied_quality_is_a_counted_observation_not_epsilon_formula():
    observation = SuppliedQualityObservation(
        candidate_id="private-candidate-a",
        task="illustrative matched evaluation set",
        correct=81,
        evaluated=100,
        evidence_label="illustrative scenario outcome",
    )
    assert observation.accuracy == pytest.approx(0.81)
    assert not hasattr(observation, "epsilon")


@pytest.mark.parametrize("track", ["tinyml", "mobile", "edge", "cloud"])
def test_track_catalogs_supply_separate_controls_quality_and_copy_variants(track):
    scenario = get_track_scenario(track)
    epsilon = get_privacy_epsilon_options(track)[0]
    quality = evaluate_supplied_quality(get_quality_observation(track, epsilon))
    control = get_threat_control("confidential_runtime")
    mechanism = get_trust_mechanism("confidential_runtime")
    slow = get_deletion_artifacts(track, "slow")
    untracked = get_deletion_artifacts(track, "untracked")

    assert quality.evaluated == 100
    assert "illustrative" in quality.evidence_label
    assert control.covered_threats == mechanism.protected_threats
    assert slow[-1].deletion_time_hours > scenario.deletion_sla_hours
    assert not untracked[-1].tracked


@pytest.mark.parametrize("track", ["tinyml", "mobile", "edge", "cloud"])
def test_every_track_scenario_runs_privacy_and_deletion_endpoints(track):
    scenario = get_track_scenario(track)
    release = release_bounded_mean(
        scenario.bounded_values,
        lower_bound=scenario.lower_bound,
        upper_bound=scenario.upper_bound,
        epsilon=1.0,
        release_id=f"{track}-release",
        seed=3,
    )
    deletion = propagate_deletion(
        scenario.artifact_copies,
        source_artifact_id=scenario.artifact_copies[0].artifact_id,
        deletion_sla_hours=scenario.deletion_sla_hours,
    )

    assert math.isfinite(release.published_release)
    assert release.lower_bound <= release.published_release <= release.upper_bound
    assert deletion.statuses


def test_unknown_track_fails_explicitly():
    with pytest.raises(ValueError, match="unsupported track"):
        get_track_scenario("satellite")  # type: ignore[arg-type]


@pytest.mark.parametrize("track", ["tinyml", "mobile", "edge", "cloud"])
def test_scenarios_expose_fleet_roles_and_attack_surface(track):
    scenario = get_track_scenario(track)
    assert len(scenario.fleet_roles) >= 3
    assert scenario.adversary_locus
    assert scenario.metric_units
    assert scenario.path_description


def test_mechanisms_and_controls_expose_execution_tiers():
    for control_id in ("transport", "secure_aggregation", "confidential_runtime", "locked_device"):
        control = get_threat_control(control_id)
        mechanism = get_trust_mechanism(control_id)
        assert control.execution_tier
        assert mechanism.execution_tier

    coverage = evaluate_threat_coverage("physical_attacker", get_threat_control("locked_device"))
    assert coverage.execution_tier == get_threat_control("locked_device").execution_tier

    latency = evaluate_trust_latency(
        get_trust_mechanism("transport"),
        adversary="network_observer",
        base_latency_ms=10.0,
        requests_per_session=1,
        deadline_ms=20.0,
    )
    assert latency.execution_tier == get_trust_mechanism("transport").execution_tier


def test_baseline_deletion_sla_outcome_reflects_scenario_bottlenecks():
    for track, expected_complete in [("tinyml", True), ("mobile", False), ("edge", True), ("cloud", False)]:
        scenario = get_track_scenario(track)
        result = propagate_deletion(scenario.artifact_copies, source_artifact_id=scenario.artifact_copies[0].artifact_id, deletion_sla_hours=scenario.deletion_sla_hours)
        assert result.complete == expected_complete


def test_daily_limits_delay_collection_without_a_lifetime_bound():
    from mlsysim.labs.experiments.v2_13_experiments import evaluate_collection_budget
    args = dict(queries_per_day_per_principal=50, daily_limit_per_principal=50,
                lifetime_limit_per_principal=None, principal_count=1, target_queries=1000)
    early = evaluate_collection_budget(**args, days=10)
    late = evaluate_collection_budget(**args, days=30)
    assert not early.target_collected
    assert late.target_collected
    assert late.collected_queries == 1500
    assert late.history[0].cumulative_queries == 50


def test_persistent_counter_survives_days_but_does_not_link_independent_accounts():
    from mlsysim.labs.experiments.v2_13_experiments import evaluate_collection_budget
    args = dict(days=30, queries_per_day_per_principal=50, daily_limit_per_principal=50,
                lifetime_limit_per_principal=300, target_queries=1000)
    single = evaluate_collection_budget(**args, principal_count=1)
    many = evaluate_collection_budget(**args, principal_count=4)
    assert single.collected_queries == 300
    assert single.denied_queries == 1200
    assert single.history[-1].admitted_queries == 0
    assert not single.target_collected
    assert many.collected_queries == 1200
    assert many.target_collected
    assert all(day.cumulative_queries <= 300 for day in single.history)


def test_mean_attack_influence_grows_and_trimming_one_extreme_is_finite_not_certified():
    from mlsysim.labs.experiments.v2_13_experiments import evaluate_gradient_integrity, get_integrity_round
    args = dict(honest_updates=get_integrity_round(1), trim_each_tail=1)
    small = evaluate_gradient_integrity(**args, malicious_updates=[-5.0], rule="mean")
    large = evaluate_gradient_integrity(**args, malicious_updates=[-500.0], rule="mean")
    trimmed = evaluate_gradient_integrity(**args, malicious_updates=[-500.0], rule="trimmed_mean")
    assert large.aggregate_error > small.aggregate_error
    assert large.loss_after > small.loss_after > small.loss_before
    assert trimmed.aggregate == pytest.approx(0.9)
    assert trimmed.improves_objective
    assert trimmed.gather_to_ring_ratio == pytest.approx(2.5)
    assert trimmed.within_fault_budget


def test_two_compromised_workers_break_the_supplied_one_fault_trim_round():
    from mlsysim.labs.experiments.v2_13_experiments import evaluate_gradient_integrity, get_integrity_round
    result = evaluate_gradient_integrity(honest_updates=get_integrity_round(2),
        malicious_updates=[-50.0, -50.0], rule="trimmed_mean", trim_each_tail=1)
    assert not result.within_fault_budget
    assert not result.improves_objective
    assert result.loss_after > result.loss_before


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_collection_rejects_invalid_count_inputs(bad):
    from mlsysim.labs.experiments.v2_13_experiments import evaluate_collection_budget
    with pytest.raises(ValueError):
        evaluate_collection_budget(days=bad, queries_per_day_per_principal=50,
            daily_limit_per_principal=50, lifetime_limit_per_principal=300,
            principal_count=1, target_queries=1000)


def test_collection_and_gradient_payloads_replay_after_json_roundtrip():
    import json
    from dataclasses import asdict
    from mlsysim.labs.experiments.v2_13_experiments import (
        evaluate_collection_budget, evaluate_gradient_integrity, get_collection_assumptions,
        get_integrity_round,
    )
    collection_inputs = {**get_collection_assumptions(), "days": 30, "principal_count": 2}
    gradient_inputs = dict(honest_updates=get_integrity_round(1), malicious_updates=[-50.0],
                           rule="trimmed_mean", trim_each_tail=1)
    for method, inputs in ((evaluate_collection_budget, collection_inputs),
                           (evaluate_gradient_integrity, gradient_inputs)):
        original = json.loads(json.dumps(asdict(method(**inputs))))
        replay = json.loads(json.dumps(asdict(method(**json.loads(json.dumps(inputs))))))
        assert original == replay


def test_exact_clean_label_poison_search_retrains_and_hides_target_failure_in_aggregate():
    from mlsysim.labs.experiments.v2_13_experiments import evaluate_clean_label_poisoning
    result = evaluate_clean_label_poisoning(poison_budget=1, attack_objective="targeted")
    assert all(candidate.label_matches_truth for candidate in result.candidates)
    assert result.selected.candidate_feature == 0.01
    assert not result.baseline.target_misclassified
    assert result.selected.target_misclassified
    assert result.selected.broad_correct == result.baseline.broad_correct == result.selected.broad_evaluated
    assert result.selected.target_squared_error > result.baseline.target_squared_error
    assert result.selected.outer_objective == max(candidate.outer_objective for candidate in result.candidates)
    # Refit independently from the selected records to verify the inner solution.
    xs = result.training_features + (result.selected.candidate_feature,)
    ys = result.training_labels + (result.selected.candidate_label,)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / sum((x-mx)**2 for x in xs)
    assert result.selected.slope == pytest.approx(slope)
    assert result.selected.intercept == pytest.approx(my-slope*mx)


def test_clean_label_attack_objective_changes_which_poison_is_selected():
    from mlsysim.labs.experiments.v2_13_experiments import evaluate_clean_label_poisoning
    targeted = evaluate_clean_label_poisoning(poison_budget=1, attack_objective="targeted")
    broad = evaluate_clean_label_poisoning(poison_budget=1, attack_objective="broad")
    clean = evaluate_clean_label_poisoning(poison_budget=0, attack_objective="targeted")
    assert targeted.selected.candidate_feature != broad.selected.candidate_feature
    assert broad.selected.broad_squared_error >= targeted.selected.broad_squared_error
    assert clean.selected == clean.baseline


@pytest.mark.parametrize("budget", [-1, 3, 1.5, True])
def test_clean_label_poisoning_rejects_invalid_dose(budget):
    from mlsysim.labs.experiments.v2_13_experiments import evaluate_clean_label_poisoning
    with pytest.raises(ValueError):
        evaluate_clean_label_poisoning(poison_budget=budget, attack_objective="targeted")
