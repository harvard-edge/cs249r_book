import pytest

from mlsysim.core.units import Q_, USD, count, hour, minute
from mlsysim.engine.v2_12_experiments import (
    ArtifactRequirement,
    CanaryCohort,
    DependencyContract,
    EvidenceRecord,
    EvidenceRequirement,
    IncidentStage,
    canary_evidence,
    incident_harm,
    evaluate_canary,
    evaluate_dependency,
    evaluate_incident,
    evaluate_portfolio,
    evaluate_release_change,
    get_operations_track,
    portfolio_costs,
    trace_dependency_changes,
    validate_release_contract,
)


TRACKS = {
    "tinyml": {"models": 12, "rate": 900 * count / hour, "delay": 24 * hour},
    "mobile": {"models": 40, "rate": 15_000 * count / hour, "delay": 8 * hour},
    "edge": {"models": 80, "rate": 4_000 * count / hour, "delay": 2 * hour},
    "cloud": {"models": 200, "rate": 120_000 * count / hour, "delay": 0.25 * hour},
}


@pytest.mark.parametrize("track", TRACKS)
def test_all_fleet_track_endpoints_produce_bounded_results(track):
    profile = TRACKS[track]
    costs = portfolio_costs(
        model_count=profile["models"],
        periods=12,
        adoption_fraction=0.75,
        independent_setup_cost_per_model=2_000 * USD,
        independent_operation_cost_per_model_period=900 * USD,
        shared_platform_fixed_cost=35_000 * USD,
        shared_operation_cost_per_adopted_model_period=250 * USD,
        migration_cost_per_adopted_model=500 * USD,
        unadopted_toil_cost_per_model_period=100 * USD,
    )
    evidence = canary_evidence(
        duration=30 * hour,
        canary_fraction=0.05,
        cohorts=[CanaryCohort(track, profile["rate"], profile["delay"], 10)],
    )

    assert costs.adopted_models + costs.unadopted_models == profile["models"]
    assert 0 <= evidence.total_labeled_samples <= evidence.total_exposed_requests


@pytest.mark.parametrize("track", TRACKS)
def test_lab_evaluators_exercise_every_track(track):
    profile = get_operations_track(track)
    portfolio = evaluate_portfolio(track, 0.75)
    dependency = evaluate_dependency(track, True)
    canary = evaluate_canary(track, duration_hours=36, canary_fraction=0.05)
    incident = evaluate_incident(track, "detection")
    release_baseline = evaluate_release_change(track, False)
    release = evaluate_release_change(track, True)

    assert profile.scenario
    assert portfolio.model_count > 0
    assert dependency.reached_consumers
    assert canary.total_exposed_requests.magnitude > 0
    assert incident.total_affected_requests.magnitude > 0
    assert release_baseline.releasable
    assert not release.releasable


def test_lab_evaluator_interventions_have_opposite_side_effects_and_causal_changes():
    low_adoption = evaluate_portfolio("cloud", 0.25)
    high_adoption = evaluate_portfolio("cloud", 1.0)
    short_canary = evaluate_canary("mobile", duration_hours=4, canary_fraction=0.01)
    wide_canary = evaluate_canary("mobile", duration_hours=4, canary_fraction=0.10)
    baseline_incident = evaluate_incident("edge", "baseline")
    detection_incident = evaluate_incident("edge", "detection")

    assert high_adoption.platform_total < low_adoption.platform_total
    assert high_adoption.unadopted_toil_cost < low_adoption.unadopted_toil_cost
    assert wide_canary.total_labeled_samples > short_canary.total_labeled_samples
    assert wide_canary.total_exposed_requests > short_canary.total_exposed_requests
    assert detection_incident.total_affected_requests < baseline_incident.total_affected_requests


def test_lab_evaluators_reject_unknown_track_and_strategy():
    with pytest.raises(ValueError, match="unknown operations track"):
        get_operations_track("datacenter")
    with pytest.raises(ValueError, match="unknown incident strategy"):
        evaluate_incident("cloud", "magic-score")


def test_portfolio_cost_identity_and_break_even_are_explicit():
    result = portfolio_costs(
        model_count=20,
        periods=1,
        adoption_fraction=1.0,
        independent_setup_cost_per_model=1_000 * USD,
        independent_operation_cost_per_model_period=500 * USD,
        shared_platform_fixed_cost=10_000 * USD,
        shared_operation_cost_per_adopted_model_period=250 * USD,
    )

    assert result.independent_total.m_as("USD") == pytest.approx(30_000)
    assert result.platform_total.m_as("USD") == pytest.approx(15_000)
    assert result.savings.m_as("USD") == pytest.approx(15_000)
    assert result.break_even_model_count == 8


def test_higher_adoption_reduces_duplication_when_shared_operation_is_cheaper():
    common = dict(
        model_count=100,
        periods=4,
        independent_setup_cost_per_model=1_000 * USD,
        independent_operation_cost_per_model_period=800 * USD,
        shared_platform_fixed_cost=50_000 * USD,
        shared_operation_cost_per_adopted_model_period=250 * USD,
        migration_cost_per_adopted_model=200 * USD,
        unadopted_toil_cost_per_model_period=150 * USD,
    )
    low = portfolio_costs(adoption_fraction=0.25, **common)
    high = portfolio_costs(adoption_fraction=0.9, **common)

    assert high.platform_total < low.platform_total
    assert high.unadopted_duplication_cost < low.unadopted_duplication_cost
    assert high.unadopted_toil_cost < low.unadopted_toil_cost


def test_portfolio_rejects_invalid_adoption_and_has_no_break_even_when_margin_negative():
    kwargs = dict(
        model_count=10,
        periods=1,
        independent_setup_cost_per_model=100 * USD,
        independent_operation_cost_per_model_period=100 * USD,
        shared_platform_fixed_cost=1_000 * USD,
        shared_operation_cost_per_adopted_model_period=300 * USD,
    )
    with pytest.raises(ValueError, match="adoption_fraction"):
        portfolio_costs(adoption_fraction=1.1, **kwargs)
    assert portfolio_costs(adoption_fraction=1.0, **kwargs).break_even_model_count is None


def test_dependency_trace_reports_compatible_and_blocked_consumer_versions():
    contracts = [
        DependencyContract("features", "ranker", "3", ("features-v1",)),
        DependencyContract("features", "fraud", "8", ("features-v1", "features-v2")),
        DependencyContract("ranker", "homepage", "5", ("ranker-v3",)),
    ]
    result = trace_dependency_changes(
        {"features": "features-v2", "ranker": "ranker-v4"}, contracts
    )

    assert result.reached_consumers == ("fraud", "homepage", "ranker")
    assert result.blocked_consumers == ("homepage", "ranker")
    assert {check.consumer: check.consumer_version for check in result.checks} == {
        "ranker": "3",
        "fraud": "8",
        "homepage": "5",
    }


def test_unrelated_version_change_cannot_alter_dependency_outcome():
    contracts = [DependencyContract("schema", "model-a", "2", ("v1",))]
    baseline = trace_dependency_changes({"schema": "v2"}, contracts)
    with_unrelated = trace_dependency_changes(
        {"schema": "v2", "unrelated-dashboard": "v99"}, contracts
    )

    assert with_unrelated.checks == baseline.checks
    assert with_unrelated.blocked_consumers == baseline.blocked_consumers


def test_canary_tracks_cohort_distribution_and_label_delay_without_significance_claim():
    result = canary_evidence(
        duration=4 * hour,
        canary_fraction=0.10,
        cohorts=[
            CanaryCohort("fast labels", 1_000 * count / hour, 1 * hour, 250),
            CanaryCohort("slow labels", 100 * count / hour, 5 * hour, 1),
        ],
    )

    fast, slow = result.cohorts
    assert fast.exposed_requests.m_as("count") == pytest.approx(400)
    assert fast.labeled_samples.m_as("count") == pytest.approx(300)
    assert slow.exposed_requests.m_as("count") == pytest.approx(40)
    assert slow.labeled_samples.m_as("count") == pytest.approx(0)
    assert not result.evidence_requirements_met
    assert result.establishes_statistical_significance is False


def test_more_canary_traffic_increases_both_evidence_and_exposure():
    cohort = CanaryCohort("site fleet", 2_000 * count / hour, 1 * hour, 100)
    small = canary_evidence(duration=3 * hour, canary_fraction=0.01, cohorts=[cohort])
    large = canary_evidence(duration=3 * hour, canary_fraction=0.10, cohorts=[cohort])

    assert large.total_labeled_samples > small.total_labeled_samples
    assert large.total_exposed_requests > small.total_exposed_requests


def test_incident_harm_sums_actual_affected_requests_by_stage():
    result = incident_harm(
        request_rate=1_000 * count / minute,
        stages=[
            IncidentStage("detection", 5 * minute, 0.20),
            IncidentStage("diagnosis", 10 * minute, 0.20),
            IncidentStage("mitigation", 4 * minute, 0.08),
            IncidentStage("recovery", 20 * minute, 0.02),
        ],
    )

    assert [stage.affected_requests.m_as("count") for stage in result.stages] == pytest.approx(
        [1_000, 2_000, 320, 400]
    )
    assert result.total_affected_requests.m_as("count") == pytest.approx(3_720)


def test_detection_investment_reduces_harm_but_stage_label_is_noncausal():
    common = [
        IncidentStage("diagnosis", 8 * minute, 0.1),
        IncidentStage("recovery", 10 * minute, 0.02),
    ]
    slow = incident_harm(
        request_rate=500 * count / minute,
        stages=[IncidentStage("detection", 10 * minute, 0.1), *common],
    )
    fast = incident_harm(
        request_rate=500 * count / minute,
        stages=[IncidentStage("detection", 2 * minute, 0.1), *common],
    )
    renamed = incident_harm(
        request_rate=500 * count / minute,
        stages=[IncidentStage("alerting", 2 * minute, 0.1), *common],
    )

    assert fast.total_affected_requests < slow.total_affected_requests
    assert renamed.total_affected_requests == fast.total_affected_requests


def test_release_contract_detects_missing_and_stale_versioned_evidence():
    artifacts = {"model": "m2", "features": "f3"}
    requirements = [
        ArtifactRequirement("model", "m2"),
        ArtifactRequirement("features", "f3"),
    ]
    evidence_requirements = [
        EvidenceRequirement("offline-eval", {"model": "m2", "features": "f3"}),
        EvidenceRequirement("canary", {"model": "m2"}),
    ]
    result = validate_release_contract(
        current_artifact_versions=artifacts,
        artifact_requirements=requirements,
        evidence_requirements=evidence_requirements,
        evidence_records=[
            EvidenceRecord("offline-eval", {"model": "m2", "features": "f2"})
        ],
    )

    assert not result.releasable
    assert result.missing_evidence == ("canary",)
    assert result.stale_evidence == ("offline-eval",)


def test_release_contract_passes_only_when_artifacts_and_evidence_match():
    result = validate_release_contract(
        current_artifact_versions={"model": "m2"},
        artifact_requirements=[ArtifactRequirement("model", "m2")],
        evidence_requirements=[EvidenceRequirement("eval", {"model": "m2"})],
        evidence_records=[EvidenceRecord("eval", {"model": "m2"})],
    )
    changed = validate_release_contract(
        current_artifact_versions={"model": "m3"},
        artifact_requirements=[ArtifactRequirement("model", "m2")],
        evidence_requirements=[EvidenceRequirement("eval", {"model": "m2"})],
        evidence_records=[EvidenceRecord("eval", {"model": "m2"})],
    )

    assert result.releasable
    assert not changed.releasable
    assert changed.artifact_mismatches == ("model",)
    assert changed.stale_evidence == ("eval",)


def test_track_profiles_define_distinct_deployment_artifacts_and_rollback_scopes():
    expected = {
        "tinyml": ("firmware", "inferences"),
        "mobile": ("mobile bundle", "queries"),
        "edge": ("gateway package", "frames"),
        "cloud": ("cloud service", "requests"),
    }
    for track_id, (expected_artifact, expected_unit) in expected.items():
        profile = get_operations_track(track_id)
        assert profile.deployment_artifact == expected_artifact
        assert profile.workload_unit == expected_unit
        assert profile.rollback_scope
        assert profile.update_scope
        assert profile.incident_context
        compat_name = f"{profile.deployment_artifact.replace(' ', '-')}-compatibility"
        baseline = evaluate_release_change(track_id, False)
        changed = evaluate_release_change(track_id, True)
        assert baseline.releasable
        assert not changed.releasable
        assert compat_name not in changed.stale_evidence
        assert "offline-evaluation" in changed.stale_evidence
        assert "canary-observation" in changed.stale_evidence


def test_deployment_artifact_version_change_causally_invalidates_compatibility_evidence():
    profile = get_operations_track("tinyml")
    versions = dict(profile.release_artifact_versions)
    versions[profile.deployment_artifact] = "firmware-v999"
    compat_name = f"{profile.deployment_artifact.replace(' ', '-')}-compatibility"
    contract = validate_release_contract(
        current_artifact_versions=versions,
        artifact_requirements=[ArtifactRequirement(k, v) for k, v in versions.items()],
        evidence_requirements=[
            EvidenceRequirement(
                compat_name,
                {"model": versions["model"], profile.deployment_artifact: versions[profile.deployment_artifact]},
            )
        ],
        evidence_records=profile.release_evidence_records,
    )
    assert not contract.releasable
    assert compat_name in contract.stale_evidence
