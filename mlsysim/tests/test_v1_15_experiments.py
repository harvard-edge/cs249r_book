import json

import pytest

from mlsysim.engine.v1_15_experiments import (
    LINEAGE_POLICIES,
    TRACKS,
    bounded_mean_release,
    investigate_incident,
    lifecycle_footprint,
    population_mix,
    threshold_consequences,
)


@pytest.mark.parametrize("track_id", TRACKS)
def test_every_track_exposes_all_five_experiments(track_id):
    mix = population_mix(track_id)
    threshold = threshold_consequences(track_id)
    privacy = bounded_mean_release(track_id)
    lifecycle = lifecycle_footprint(track_id)
    incident = investigate_incident(track_id)

    assert mix["track_id"] == track_id
    assert set(threshold["groups"]) == {"reference", "affected"}
    assert privacy["mechanism"] == "bounded-mean Laplace mechanism"
    assert lifecycle["total_energy_kwh"] > 0
    assert incident["fully_reconstructable_decisions"] == incident["incident_decisions"]
    for result in (mix, threshold, privacy, lifecycle, incident):
        assert "Illustrative" in result["assumption"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_population_weights_change_aggregates_but_not_subgroup_rules(track_id):
    low_share = population_mix(track_id, affected_share_pct=10)
    high_share = population_mix(track_id, affected_share_pct=90)

    assert low_share["groups"] == high_share["groups"]
    assert low_share["threshold"] == high_share["threshold"]
    assert low_share["aggregate"]["accuracy"] != high_share["aggregate"]["accuracy"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_higher_positive_score_threshold_never_increases_tpr_or_fpr(track_id):
    low = threshold_consequences(track_id, threshold=0.3)
    middle = threshold_consequences(track_id, threshold=0.5)
    high = threshold_consequences(track_id, threshold=0.7)

    for group in ("reference", "affected"):
        assert low["groups"][group]["tpr"] >= middle["groups"][group]["tpr"]
        assert middle["groups"][group]["tpr"] >= high["groups"][group]["tpr"]
        assert low["groups"][group]["fpr"] >= middle["groups"][group]["fpr"]
        assert middle["groups"][group]["fpr"] >= high["groups"][group]["fpr"]
    assert low["expected_false_positives"] >= high["expected_false_positives"]
    assert low["expected_false_negatives"] <= high["expected_false_negatives"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_population_size_changes_counts_not_rates(track_id):
    small = threshold_consequences(track_id, population_size=1_000)
    large = threshold_consequences(track_id, population_size=10_000)

    assert small["groups"] == large["groups"]
    assert small["aggregate"] == large["aggregate"]
    assert large["expected_false_positives"] == pytest.approx(10 * small["expected_false_positives"])
    assert large["expected_false_negatives"] == pytest.approx(10 * small["expected_false_negatives"])


@pytest.mark.parametrize("track_id", TRACKS)
def test_bounded_mean_laplace_equations_and_basic_composition(track_id):
    result = bounded_mean_release(track_id, epsilon_per_release=0.8, releases=3, seed=41)
    lower, upper = result["public_bounds"]

    assert result["l1_sensitivity"] == pytest.approx((upper - lower) / result["sample_size"])
    assert result["laplace_scale"] == pytest.approx(result["l1_sensitivity"] / 0.8)
    assert result["basic_composed_epsilon"] == pytest.approx(2.4)
    assert len(result["released_values"]) == 3
    assert all(lower <= value <= upper for value in result["released_values"])
    assert result["released_summary_bytes"] == 24
    assert "fixed public seed" in result["randomness_caveat"]
    assert "Retention alone" in result["accounting_caveat"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_privacy_strength_changes_noise_not_data_or_storage(track_id):
    stronger = bounded_mean_release(track_id, epsilon_per_release=0.25, seed=8)
    weaker = bounded_mean_release(track_id, epsilon_per_release=2.0, seed=8)

    assert stronger["laplace_scale"] > weaker["laplace_scale"]
    assert stronger["true_bounded_mean"] == weaker["true_bounded_mean"]
    assert stronger["raw_storage_bytes"] == weaker["raw_storage_bytes"]
    assert stronger["sample_size"] == weaker["sample_size"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_lifecycle_energy_is_the_sum_of_terms_and_carbon_uses_same_boundary(track_id):
    result = lifecycle_footprint(track_id)
    summed = sum(result["terms_kwh"].values())

    assert result["total_energy_kwh"] == pytest.approx(summed)
    expected_carbon_kg = result["total_energy_kwh"] * result["carbon_intensity_g_per_kwh"] / 1_000
    assert result["operational_carbon_kg"] == pytest.approx(expected_carbon_kg)
    assert "Embodied energy" in result["boundary"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_inference_demand_changes_serving_work_not_training_work(track_id):
    baseline = lifecycle_footprint(track_id, demand_scale=1)
    doubled = lifecycle_footprint(track_id, demand_scale=2)

    assert doubled["terms_kwh"]["initial_training"] == baseline["terms_kwh"]["initial_training"]
    assert doubled["terms_kwh"]["retraining"] == baseline["terms_kwh"]["retraining"]
    assert doubled["terms_kwh"]["inference"] == pytest.approx(2 * baseline["terms_kwh"]["inference"])
    assert doubled["terms_kwh"]["explanation"] == pytest.approx(2 * baseline["terms_kwh"]["explanation"])


@pytest.mark.parametrize("track_id", TRACKS)
def test_lineage_missingness_changes_reconstructability_not_fairness(track_id):
    complete = investigate_incident(track_id)
    missing = investigate_incident(
        track_id,
        retained_fields=("model_version", "decision_threshold", "deployment_id"),
    )

    assert complete["answerable_question_count"] == 3
    assert missing["answerable_question_count"] == 1
    assert missing["fully_reconstructable_decisions"] == 0
    assert missing["evidence_storage_bytes"] < complete["evidence_storage_bytes"]
    assert missing["fairness_snapshot"] == complete["fairness_snapshot"]
    assert missing["decisions_before_containment"] == complete["decisions_before_containment"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_containment_delay_changes_exposure_not_evidence_or_outcomes(track_id):
    fast = investigate_incident(track_id, containment_delay_hours=0.5)
    slow = investigate_incident(track_id, containment_delay_hours=8.0)

    assert slow["decisions_before_containment"] == pytest.approx(16 * fast["decisions_before_containment"])
    assert slow["fully_reconstructable_decisions"] == fast["fully_reconstructable_decisions"]
    assert slow["evidence_storage_bytes"] == fast["evidence_storage_bytes"]
    assert slow["fairness_snapshot"] == fast["fairness_snapshot"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_each_track_rejects_invalid_experiment_inputs(track_id):
    with pytest.raises(ValueError, match="threshold"):
        threshold_consequences(track_id, threshold=1.01)
    with pytest.raises(ValueError, match="epsilon_per_release"):
        bounded_mean_release(track_id, epsilon_per_release=0)
    with pytest.raises(ValueError, match="years"):
        lifecycle_footprint(track_id, years=0)
    with pytest.raises(ValueError, match="not both"):
        lifecycle_footprint(track_id, daily_inferences=100, demand_scale=2)
    with pytest.raises(ValueError, match="unknown retained_fields"):
        investigate_incident(track_id, retained_fields=("moral_score",))


def test_unknown_track_fails_explicitly():
    with pytest.raises(ValueError, match="track_id"):
        population_mix("satellite")


def test_lineage_policies_are_valid_backend_scenarios():
    for fields in LINEAGE_POLICIES.values():
        assert investigate_incident("edge", retained_fields=fields)["inputs"]["retained_fields"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_all_experiment_results_are_directly_evidence_serializable(track_id):
    results = (
        population_mix(track_id),
        threshold_consequences(track_id),
        bounded_mean_release(track_id),
        lifecycle_footprint(track_id),
        investigate_incident(track_id),
    )

    for result in results:
        assert json.loads(json.dumps(result, allow_nan=False))["inputs"]
