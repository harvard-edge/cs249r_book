"""Causal checks for the Volume I Chapter 4 teaching experiments."""

from dataclasses import replace
import json

import pytest

from mlsysim.core.units import Q_
from mlsysim.labs.experiments.v1_04_experiments import (
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


@pytest.mark.parametrize("track", TRACKS)
def test_label_detail_resolves_action_collision_but_costs_annotation_breadth(track):
    scenario = get_track_scenario(track)
    coarse = retain_for_annotation(**retention_inputs(scenario, "coverage", "coarse", "route"))
    specific = retain_for_annotation(**retention_inputs(scenario, "coverage", "specific", "route"))
    assert coarse.ambiguous_label_count > 0
    assert coarse.action_resolvable_records == 0
    assert specific.ambiguous_label_count == 0
    assert specific.action_resolvable_records > 0
    assert len(specific.retained_ids) < len(coarse.retained_ids)
    assert specific.annotation_time <= scenario.annotation_budget
    assert specific.represented_cohort_count < coarse.represented_cohort_count
    # Changing annotation representation must not change the required actions.
    coarse_actions = retention_inputs(scenario, "coverage", "coarse", "route")["rubric"].actions_by_cohort
    specific_actions = retention_inputs(scenario, "coverage", "specific", "route")["rubric"].actions_by_cohort
    assert coarse_actions == specific_actions
    assert all(row["required_action"] == coarse_actions[row["condition"]] for row in specific.label_assignments)


@pytest.mark.parametrize("track", TRACKS)
def test_presence_task_reverses_need_for_finer_labels(track):
    scenario = get_track_scenario(track)
    coarse = retain_for_annotation(**retention_inputs(scenario, "coverage", "coarse", "presence"))
    specific = retain_for_annotation(**retention_inputs(scenario, "coverage", "specific", "presence"))
    assert coarse.ambiguous_label_count == specific.ambiguous_label_count == 0
    assert coarse.action_resolvable_records > specific.action_resolvable_records
    assert len(coarse.retained_ids) > len(specific.retained_ids)


def test_specific_labels_do_not_create_missing_raw_evidence_or_hide_vocabulary_collisions():
    scenario = get_track_scenario("edge")
    inputs = retention_inputs(scenario, "newest", "specific", "route")
    # A generous budget admits all records so missing evidence remains visible.
    inputs["annotation_budget"] = Q_(100, "minute")
    result = retain_for_annotation(**inputs)
    missing = [row for row in result.label_assignments if not row["has_task_evidence"]]
    assert missing and all(not row["action_resolvable"] for row in missing)
    # Retaining only a single condition does not silently repair the coarse rubric.
    coarse = retention_inputs(scenario, "newest", "coarse", "route")
    coarse["annotation_budget"] = Q_(4, "minute")
    restricted = retain_for_annotation(**coarse)
    assert restricted.represented_cohort_count == 1
    assert restricted.ambiguous_label_count > 0
    assert restricted.action_resolvable_records == 0


@pytest.mark.parametrize("track", TRACKS)
@pytest.mark.parametrize("task", ("presence", "route"))
@pytest.mark.parametrize("rubric", ("coarse", "specific"))
def test_label_rubric_full_snapshot_replay_preserves_task_cost_and_truth(track, task, rubric):
    scenario = get_track_scenario(track)
    original = capture_experiment("retention", **retention_inputs(scenario, "coverage", rubric, task))
    payload = json.loads(json.dumps(original.to_dict(), allow_nan=False))
    assert replay_experiment(payload).to_dict() == payload
    # Evidence flags are reported but are never used to prioritize annotation.
    inputs = retention_inputs(scenario, "coverage", rubric, task)
    inputs["records"] = tuple(replace(record, has_task_evidence=False) for record in scenario.candidates)
    no_evidence = retain_for_annotation(**inputs)
    assert list(no_evidence.retained_ids) == payload["result"]["retained_ids"]
    assert no_evidence.action_resolvable_records == 0


def test_invalid_label_rubrics_fail_before_selection():
    from mlsysim.labs.experiments.v1_04_experiments import LabelRubric
    scenario = get_track_scenario("tinyml")
    for bad in (
        LabelRubric("bad", {}, {}, 1.0, "route"),
        replace(retention_inputs(scenario, "coverage")["rubric"], annotation_multiplier=float("nan")),
    ):
        with pytest.raises(ValueError):
            retain_for_annotation(scenario.candidates, scenario.annotation_budget, "coverage", rubric=bad)


@pytest.mark.parametrize("track", TRACKS)
def test_reordering_raw_candidates_cannot_rewrite_label_task_requirements(track):
    scenario = get_track_scenario(track)
    reordered = replace(scenario, candidates=tuple(reversed(scenario.candidates)))
    for rubric in ("coarse", "specific"):
        baseline = retention_inputs(scenario, "coverage", rubric, "route")
        shuffled = retention_inputs(reordered, "coverage", rubric, "route")
        assert baseline["rubric"] == shuffled["rubric"]
        assert retain_for_annotation(**baseline) == retain_for_annotation(**shuffled)
    unsupported = replace(scenario, candidates=(replace(scenario.candidates[0], cohort="unknown-condition"),))
    with pytest.raises(ValueError, match="every record cohort"):
        retain_for_annotation(**retention_inputs(unsupported, "coverage"))


@pytest.mark.parametrize("track", TRACKS)
def test_freshness_decision_replay_separates_age_from_behavior(track):
    scenario = get_track_scenario(track)
    changing = {p: capture_experiment("freshness", **freshness_inputs(scenario, p, "changing"))
                for p in ("batch", "stream", "local feature")}
    stable = {p: capture_experiment("freshness", **freshness_inputs(scenario, p, "stable"))
              for p in changing}
    assert changing["batch"].result["stale_decisions"] > changing["stream"].result["stale_decisions"]
    for policy in changing:
        assert stable[policy].result["stale_decisions"] == 0
        assert stable[policy].result["worst_case_age"] == changing[policy].result["worst_case_age"]
        assert stable[policy].result["traffic"] == changing[policy].result["traffic"]
        assert stable[policy].result["meets_freshness_sla"] == changing[policy].result["meets_freshness_sla"]
        for run in (changing[policy], stable[policy]):
            replayed = replay_experiment(json.loads(json.dumps(run.to_dict())))
            assert replayed.result == run.result
            for row in run.result["decision_trace"]:
                assert row["mismatch"] == (row["available_action"] != row["current_required_action"])


def test_annotation_review_finds_disagreement_but_shared_error_needs_gold():
    scenario = get_track_scenario("tinyml")
    runs = {policy: capture_experiment("retention", **retention_inputs(scenario, "coverage", "specific", "route", policy))
            for policy in ("single", "disagreements", "gold")}
    audits = {policy: run.result["annotation_audit"] for policy, run in runs.items()}
    assert audits["single"]["reference_mismatches"] > audits["disagreements"]["reference_mismatches"] > audits["gold"]["reference_mismatches"]
    assert audits["single"]["labor_minutes"] < audits["disagreements"]["labor_minutes"] < audits["gold"]["labor_minutes"]
    for policy, run in runs.items():
        assert run.result["retained_ids"] == runs["single"].result["retained_ids"]
        assert audits[policy]["unresolved_records"] == 1
        assert replay_experiment(json.loads(json.dumps(run.to_dict()))).result == run.result
    agreed_error = next(row for row in audits["disagreements"]["records"] if row["record_id"] == "audit-shared-error")
    assert agreed_error["agreement"] and agreed_error["reference_mismatch"] and not agreed_error["reviewed"]
    unresolved = next(row for row in audits["gold"]["records"] if row["record_id"] == "audit-missing-evidence")
    assert unresolved["final"] is None and unresolved["unresolved"]


def test_review_reacts_to_supplied_annotations_not_hardcoded_policy_score():
    scenario = get_track_scenario("cloud")
    arguments = retention_inputs(scenario, "newest", review_policy="disagreements")
    original = capture_experiment("retention", **arguments)
    audit = arguments["annotation_audit_inputs"]
    audit["records"] = tuple({**row, "first": row["reference"], "second": row["reference"]} for row in audit["records"])
    changed = capture_experiment("retention", **arguments)
    assert changed.result["annotation_audit"]["reference_mismatches"] == 0
    assert changed.result["annotation_audit"]["labor_minutes"] < original.result["annotation_audit"]["labor_minutes"]
    assert changed.result["annotation_audit"]["unresolved_records"] == 1


def test_annotation_agreement_is_chance_corrected_and_not_rewritten_by_gold():
    scenario = get_track_scenario("mobile")
    reviews = [capture_experiment("retention", **retention_inputs(scenario, "newest", review_policy=p)).result["annotation_audit"]
               for p in ("single", "disagreements", "gold")]
    single, double, gold = reviews
    assert single["observed_agreement"] is None and single["cohens_kappa"] is None
    assert double["cohens_kappa"] == pytest.approx((double["observed_agreement"] - double["chance_agreement"]) / (1 - double["chance_agreement"]))
    assert double["cohens_kappa"] == gold["cohens_kappa"]
    assert double["reference_mismatches"] > gold["reference_mismatches"]


def test_annotation_perfect_shared_error_has_agreement_but_not_correctness():
    from mlsysim.labs.experiments.v1_04_experiments import evaluate_annotation_audit
    audit = evaluate_annotation_audit(
        records=({"record_id": "a", "first": "event", "second": "event", "reference": "background"},),
        policy="disagreements", minutes_per_annotation=1, minutes_per_adjudication=2, minutes_per_gold_check=1)
    assert audit["observed_agreement"] == 1
    assert audit["cohens_kappa"] is None
    assert audit["reference_mismatches"] == 1
