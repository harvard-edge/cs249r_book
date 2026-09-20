"""Causal checks for the Volume I Chapter 4 teaching experiments."""

from dataclasses import replace
import json

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v1_04_experiments import (
    ProducerRecord,
    capture_experiment,
    contract_inputs,
    evaluate_contract,
    evaluate_freshness,
    evaluate_pipeline,
    evaluate_split,
    freshness_inputs,
    get_track_scenario,
    pipeline_inputs,
    replay_experiment,
    retention_inputs,
    retain_for_annotation,
    split_inputs,
)


TRACKS = ("tinyml", "mobile", "edge", "cloud")


@pytest.mark.parametrize("track", TRACKS)
def test_all_tracks_expose_complete_representative_fixtures(track):
    scenario = get_track_scenario(track)

    assert scenario.track == track
    assert any(record.duplicate_of for record in scenario.candidates)
    assert len({record.cohort for record in scenario.candidates}) >= 3
    assert len(scenario.freshness_policies) == 3
    assert any(record.semantic_unit != scenario.expected_semantic_unit for record in scenario.producer_records)


@pytest.mark.parametrize("track", TRACKS)
def test_retention_policies_respect_same_annotation_budget(track):
    scenario = get_track_scenario(track)
    newest = retain_for_annotation(scenario.candidates, scenario.annotation_budget, "newest")
    coverage = retain_for_annotation(scenario.candidates, scenario.annotation_budget, "coverage")
    deduplicated = retain_for_annotation(scenario.candidates, scenario.annotation_budget, "deduplicate")

    for result in (newest, coverage, deduplicated):
        assert result.annotation_time <= scenario.annotation_budget
        assert result.budget_remaining >= Q_(0, "minute")
    assert deduplicated.duplicate_records == 0
    assert newest.duplicate_records == 1
    assert len(coverage.represented_cohorts) > len(newest.represented_cohorts)


def test_retention_reports_supplied_evidence_without_changing_selection():
    scenario = get_track_scenario("mobile")
    baseline = retain_for_annotation(scenario.candidates, scenario.annotation_budget, "deduplicate")
    changed = tuple(replace(record, has_task_evidence=not record.has_task_evidence) for record in scenario.candidates)
    observed = retain_for_annotation(changed, scenario.annotation_budget, "deduplicate")

    assert observed.retained_ids == baseline.retained_ids
    assert observed.records_with_task_evidence != baseline.records_with_task_evidence


@pytest.mark.parametrize("track", TRACKS)
def test_entity_disjoint_split_removes_key_overlap_and_can_lower_observed_accuracy(track):
    scenario = get_track_scenario(track)
    leaky = scenario.leaky_split
    clean = scenario.entity_disjoint_split

    leaky_result = evaluate_split(
        scenario.labeled_records,
        train_ids=leaky.train_ids,
        test_ids=leaky.test_ids,
        predicted_labels=leaky.predictions,
        preprocessing_fit_ids=leaky.train_ids + leaky.test_ids,
    )
    clean_result = evaluate_split(
        scenario.labeled_records,
        train_ids=clean.train_ids,
        test_ids=clean.test_ids,
        predicted_labels=clean.predictions,
        preprocessing_fit_ids=clean.train_ids,
    )

    assert leaky_result.key_leakage_fraction > 0
    assert leaky_result.preprocessing_test_records_seen == len(leaky.test_ids)
    assert clean_result.key_leakage_fraction == 0
    assert clean_result.preprocessing_test_records_seen == 0
    assert leaky_result.observed_accuracy > clean_result.observed_accuracy


def test_split_accuracy_comes_only_from_supplied_labels_and_predictions():
    scenario = get_track_scenario("edge")
    split = scenario.entity_disjoint_split
    baseline = evaluate_split(
        scenario.labeled_records,
        train_ids=split.train_ids,
        test_ids=split.test_ids,
        predicted_labels=split.predictions,
        preprocessing_fit_ids=split.train_ids,
    )
    extra_fit = evaluate_split(
        scenario.labeled_records,
        train_ids=split.train_ids,
        test_ids=split.test_ids,
        predicted_labels=split.predictions,
        preprocessing_fit_ids=split.train_ids + split.test_ids,
    )

    assert extra_fit.preprocessing_test_records_seen > baseline.preprocessing_test_records_seen
    assert extra_fit.observed_accuracy == baseline.observed_accuracy


@pytest.mark.parametrize("track", TRACKS)
def test_pipeline_has_valid_and_failed_resource_cases_for_every_track(track):
    scenario = get_track_scenario(track)
    baseline = evaluate_pipeline(
        sample_size=scenario.sample_size,
        compression_ratio=1,
        read_bandwidth=scenario.read_bandwidth,
        base_decode_rate_per_second=scenario.base_decode_rate_per_second,
        compression_decode_work=1,
        transform_rate_per_second=scenario.transform_rate_per_second,
        required_rate_per_second=scenario.required_rate_per_second,
    )
    starved = evaluate_pipeline(
        sample_size=scenario.sample_size,
        compression_ratio=1,
        read_bandwidth=scenario.read_bandwidth / 100,
        base_decode_rate_per_second=scenario.base_decode_rate_per_second,
        compression_decode_work=1,
        transform_rate_per_second=scenario.transform_rate_per_second,
        required_rate_per_second=scenario.required_rate_per_second,
    )

    assert baseline.meets_required_rate
    assert baseline.accelerator_wait_fraction == 0
    assert not starved.meets_required_rate
    assert starved.bottleneck_stage == "read"
    assert starved.accelerator_wait_fraction > 0


def test_compression_trades_read_supply_against_decode_work():
    inputs = dict(
        sample_size=Q_(4, "megabyte"),
        read_bandwidth=Q_(20, "megabyte/second"),
        base_decode_rate_per_second=30,
        transform_rate_per_second=100,
        required_rate_per_second=20,
    )
    raw = evaluate_pipeline(**inputs, compression_ratio=1, compression_decode_work=1)
    compressed = evaluate_pipeline(**inputs, compression_ratio=8, compression_decode_work=2)

    assert compressed.read_rate_per_second > raw.read_rate_per_second
    assert compressed.decode_rate_per_second < raw.decode_rate_per_second
    assert raw.bottleneck_stage == "read"
    assert compressed.bottleneck_stage == "decode"
    assert compressed.service_rate_per_second > raw.service_rate_per_second


def test_required_demand_changes_waiting_but_not_pipeline_supply():
    scenario = get_track_scenario("cloud")
    common = dict(
        sample_size=scenario.sample_size,
        compression_ratio=2,
        read_bandwidth=scenario.read_bandwidth,
        base_decode_rate_per_second=scenario.base_decode_rate_per_second,
        compression_decode_work=1.5,
        transform_rate_per_second=scenario.transform_rate_per_second,
    )
    low_demand = evaluate_pipeline(**common, required_rate_per_second=100)
    high_demand = evaluate_pipeline(**common, required_rate_per_second=2000)

    assert high_demand.service_rate_per_second == low_demand.service_rate_per_second
    assert high_demand.accelerator_wait_fraction > low_demand.accelerator_wait_fraction


@pytest.mark.parametrize("track", TRACKS)
def test_freshness_alternatives_have_time_traffic_and_annotation_tradeoffs(track):
    scenario = get_track_scenario(track)
    results = {
        policy.name: evaluate_freshness(
            policy,
            event_rate=scenario.event_rate,
            horizon=scenario.freshness_horizon,
            freshness_sla=scenario.freshness_sla,
        )
        for policy in scenario.freshness_policies
    }

    assert results["stream"].worst_case_age < results["batch"].worst_case_age
    assert results["local feature"].traffic < results["stream"].traffic
    assert results["local feature"].annotation_time > results["stream"].annotation_time
    assert results["stream"].meets_freshness_sla


def test_freshness_sla_changes_acceptance_only():
    scenario = get_track_scenario("mobile")
    policy = scenario.freshness_policies[0]
    strict = evaluate_freshness(
        policy,
        event_rate=scenario.event_rate,
        horizon=scenario.freshness_horizon,
        freshness_sla=Q_(5, "second"),
    )
    relaxed = evaluate_freshness(
        policy,
        event_rate=scenario.event_rate,
        horizon=scenario.freshness_horizon,
        freshness_sla=Q_(60, "second"),
    )

    assert strict.worst_case_age == relaxed.worst_case_age
    assert strict.traffic == relaxed.traffic
    assert not strict.meets_freshness_sla
    assert relaxed.meets_freshness_sla


@pytest.mark.parametrize("track", TRACKS)
def test_semantic_contract_blocks_unit_change_at_validation_cost(track):
    scenario = get_track_scenario(track)
    common = dict(
        records=scenario.producer_records,
        expected_schema_version=1,
        expected_semantic_unit=scenario.expected_semantic_unit,
        schema_check_time_per_record=Q_(2, "millisecond"),
        semantic_check_time_per_record=Q_(3, "millisecond"),
        recovery_time_per_escaped_record=Q_(20, "minute"),
    )
    schema = evaluate_contract(**common, level="schema")
    semantic = evaluate_contract(**common, level="semantic")

    assert schema.escaped_semantic_errors == 1
    assert schema.rejected_records == 1
    assert semantic.escaped_semantic_errors == 0
    assert semantic.rejected_records == 2
    assert semantic.validation_time > schema.validation_time
    assert semantic.recovery_time < schema.recovery_time


def test_contract_does_not_treat_feature_magnitude_as_validity_score():
    scenario = get_track_scenario("tinyml")
    common = dict(
        expected_schema_version=1,
        expected_semantic_unit=scenario.expected_semantic_unit,
        level="semantic",
        schema_check_time_per_record=Q_(1, "millisecond"),
        semantic_check_time_per_record=Q_(1, "millisecond"),
        recovery_time_per_escaped_record=Q_(10, "minute"),
    )
    baseline = evaluate_contract(scenario.producer_records, **common)
    changed_values = tuple(
        ProducerRecord(
            record.record_id,
            record.schema_version,
            record.feature_value * 1000,
            record.semantic_unit,
        )
        for record in scenario.producer_records
    )
    observed = evaluate_contract(changed_values, **common)

    assert observed == baseline


@pytest.mark.parametrize(
    "call",
    [
        lambda: get_track_scenario("satellite"),
        lambda: evaluate_pipeline(
            sample_size=Q_(1, "byte"),
            compression_ratio=0.5,
            read_bandwidth=Q_(1, "byte/second"),
            base_decode_rate_per_second=1,
            compression_decode_work=1,
            transform_rate_per_second=1,
            required_rate_per_second=1,
        ),
    ],
)
def test_invalid_experiment_inputs_fail_with_words(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    "bad_inputs",
    [
        {"read_bandwidth": Q_(0, "byte/second")},
        {"read_bandwidth": Q_(float("nan"), "byte/second")},
        {"base_decode_rate_per_second": float("nan")},
        {"compression_decode_work": float("inf")},
        {"transform_rate_per_second": -1},
        {"required_rate_per_second": float("nan")},
    ],
)
def test_pipeline_rejects_nonpositive_or_nonfinite_supply_and_demand(bad_inputs):
    inputs = {
        "sample_size": Q_(1, "megabyte"),
        "compression_ratio": 1,
        "read_bandwidth": Q_(10, "megabyte/second"),
        "base_decode_rate_per_second": 20,
        "compression_decode_work": 1,
        "transform_rate_per_second": 20,
        "required_rate_per_second": 10,
    }
    inputs.update(bad_inputs)

    with pytest.raises(ValueError):
        evaluate_pipeline(**inputs)


def _all_method_snapshots():
    scenario = get_track_scenario("mobile")
    split = scenario.leaky_split
    policy = scenario.freshness_policies[0]
    return (
        capture_experiment(
            "retention",
            records=scenario.candidates,
            annotation_budget=scenario.annotation_budget,
            policy="coverage",
        ),
        capture_experiment(
            "split",
            records=scenario.labeled_records,
            train_ids=split.train_ids,
            test_ids=split.test_ids,
            predicted_labels=split.predictions,
            preprocessing_fit_ids=split.train_ids + split.test_ids,
        ),
        capture_experiment(
            "pipeline",
            sample_size=scenario.sample_size,
            compression_ratio=4,
            read_bandwidth=scenario.read_bandwidth,
            base_decode_rate_per_second=scenario.base_decode_rate_per_second,
            compression_decode_work=1.75,
            transform_rate_per_second=scenario.transform_rate_per_second,
            required_rate_per_second=scenario.required_rate_per_second,
        ),
        capture_experiment(
            "freshness",
            policy=policy,
            event_rate=scenario.event_rate,
            horizon=scenario.freshness_horizon,
            freshness_sla=scenario.freshness_sla,
        ),
        capture_experiment(
            "contract",
            records=scenario.producer_records,
            expected_schema_version=1,
            expected_semantic_unit=scenario.expected_semantic_unit,
            level="semantic",
            schema_check_time_per_record=Q_(2, "millisecond"),
            semantic_check_time_per_record=Q_(3, "millisecond"),
            recovery_time_per_escaped_record=Q_(20, "minute"),
        ),
    )


@pytest.mark.parametrize("snapshot", _all_method_snapshots())
def test_all_experiment_snapshots_round_trip_through_json(snapshot):
    serialized = json.loads(json.dumps(snapshot.to_dict(), allow_nan=False))
    replayed = replay_experiment(serialized)

    assert replayed.method == snapshot.method
    assert replayed.inputs == snapshot.inputs
    assert replayed.result == snapshot.result


@pytest.mark.parametrize("track", TRACKS)
def test_notebook_input_builders_replay_all_methods_for_every_track(track):
    scenario = get_track_scenario(track)
    snapshots = (
        capture_experiment("retention", **retention_inputs(scenario, "coverage")),
        capture_experiment("split", **split_inputs(scenario, "entity", "train")),
        capture_experiment(
            "pipeline",
            **pipeline_inputs(
                scenario,
                storage_path="native",
                compression="balanced",
                transform_lanes=2,
            ),
        ),
        capture_experiment("freshness", **freshness_inputs(scenario, "stream")),
        capture_experiment("contract", **contract_inputs(scenario, "semantic")),
    )

    for snapshot in snapshots:
        serialized = json.loads(json.dumps(snapshot.to_dict(), allow_nan=False))
        assert replay_experiment(serialized).to_dict() == serialized
    assert snapshots[2].result["meets_required_rate"]
