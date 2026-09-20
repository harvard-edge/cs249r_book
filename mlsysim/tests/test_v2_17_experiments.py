"""Causal tests for the Volume II fleet-synthesis experiments."""

from dataclasses import replace

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v2_01_experiments import evaluate_c3_scaling, serialize_evaluation
from mlsysim.engine.v2_07_experiments import checkpoint_tradeoff
from mlsysim.engine.v2_10_experiments import (
    illustrative_track_scenario,
    serialize_quantity,
)
from mlsysim.engine.v2_15_experiments import serialize_evaluator_args
from mlsysim.engine.v2_17_experiments import (
    REPLAY_TARGETS,
    REQUIRED_MODELS,
    ControlPatch,
    RepairPlan,
    apply_patch,
    audit_ledger_evidence,
    compare_equal_budget_repairs,
    evidence_inputs,
    evaluate_configuration,
    instructor_scenario,
    record_deployment_decision,
    replay_evidence,
    serialize_configuration,
    deserialize_configuration,
)


TRACKS = ("tinyml", "mobile", "edge", "cloud")


@pytest.mark.parametrize("track_id", TRACKS)
def test_all_track_scenarios_compose_native_models_and_pass_baseline(track_id):
    scenario = instructor_scenario(track_id)
    result = evaluate_configuration(scenario)

    assert scenario.scenario_origin == "instructor_scenario"
    assert "illustrative instructor scenario" in scenario.scenario_label
    assert result.physical_feasible
    assert result.c3_result.step_time.check("[time]")
    assert Q_(result.recovery_result["recovery_s"], "second").check("[time]")
    assert result.serving_result.p99_duration.check("[time]")
    assert result.energy_result.facility_energy.check("[energy]")
    assert {check.constraint for check in result.checks} == {
        "C3 step time",
        "recovery time",
        "recovery waste",
        "serving p99",
        "serving completion",
        "absolute facility energy",
        "electrical capacity",
        "cooling capacity",
    }


@pytest.mark.parametrize("track_id", TRACKS)
def test_each_track_can_fail_each_applicable_physical_requirement(track_id):
    baseline = instructor_scenario(track_id)

    c3_failure = evaluate_configuration(
        apply_patch(baseline, ControlPatch(bandwidth_multiplier=0.03, coordination_multiplier=10.0))
    )
    recovery_failure = evaluate_configuration(apply_patch(baseline, ControlPatch(recovery_time_multiplier=20.0)))
    serving_failure = evaluate_configuration(apply_patch(baseline, ControlPatch(serving_time_multiplier=200.0)))
    energy_failure = evaluate_configuration(apply_patch(baseline, ControlPatch(energy_work_multiplier=20.0)))
    facility_failure = evaluate_configuration(apply_patch(baseline, ControlPatch(it_power_multiplier=2.0)))

    assert "C3 step time" in c3_failure.failed_constraints
    recovery_not_passing = set(recovery_failure.failed_constraints) | set(recovery_failure.unevaluated_constraints)
    assert "recovery time" in recovery_not_passing
    assert "serving p99" in serving_failure.failed_constraints
    assert "absolute facility energy" in energy_failure.failed_constraints
    assert {"electrical capacity", "cooling capacity"} & set(facility_failure.failed_constraints)


def test_rejected_requests_fail_completion_without_being_hidden_in_p99():
    scenario = instructor_scenario("cloud")
    service = scenario.serving.service
    no_state_capacity = replace(
        service,
        device_memory=service.weight_memory + service.reserved_memory,
    )
    constrained = replace(
        scenario,
        serving=replace(scenario.serving, service=no_state_capacity),
    )
    result = evaluate_configuration(constrained)

    assert result.serving_result.rejected_count == result.serving_result.request_count
    assert "serving completion" in result.failed_constraints
    assert "serving p99" not in result.failed_constraints
    assert "serving p99" in result.unevaluated_constraints


def test_bandwidth_is_causal_only_for_c3_and_metadata_is_noncausal():
    baseline = instructor_scenario("cloud")
    fast = evaluate_configuration(baseline)
    slow = evaluate_configuration(apply_patch(baseline, ControlPatch(bandwidth_multiplier=0.25)))
    relabeled = evaluate_configuration(
        replace(
            baseline,
            scenario_label="same causal inputs with a different label",
            scenario_origin="ledger_replay",
        )
    )

    assert slow.c3_result.communication_time > fast.c3_result.communication_time
    assert slow.c3_result.compute_time == fast.c3_result.compute_time
    assert slow.recovery_result["recovery_s"] == fast.recovery_result["recovery_s"]
    assert slow.serving_result.p99_duration == fast.serving_result.p99_duration
    assert slow.energy_result.facility_energy == fast.energy_result.facility_energy
    assert relabeled.checks == fast.checks


def test_equal_budget_repairs_start_from_one_failure_and_expose_tradeoffs():
    scenario = instructor_scenario("cloud")
    stress = ControlPatch(
        bandwidth_multiplier=0.20,
        coordination_multiplier=4.0,
        energy_work_multiplier=2.0,
    )
    fabric_repair = RepairPlan(
        "fabric",
        "Restore fabric margin",
        Q_(100_000, "dollar"),
        ControlPatch(bandwidth_multiplier=5.0, coordination_multiplier=0.25),
        "Communication improves while the stressed absolute energy remains.",
    )
    energy_repair = RepairPlan(
        "efficiency",
        "Reduce work energy",
        Q_(100_000, "dollar"),
        ControlPatch(energy_efficiency_multiplier=0.40),
        "Energy improves while the stressed C3 path remains.",
    )

    comparison = compare_equal_budget_repairs(scenario, stress=stress, repairs=(fabric_repair, energy_repair))
    results = {plan.repair_id: result for plan, result in comparison.repairs}

    assert comparison.failing_evaluation.failed_constraints
    assert comparison.common_budget == Q_(100_000, "dollar")
    assert "C3 step time" not in results["fabric"].failed_constraints
    assert "absolute facility energy" in results["fabric"].failed_constraints
    assert "absolute facility energy" not in results["efficiency"].failed_constraints
    assert "C3 step time" in results["efficiency"].failed_constraints
    assert comparison.failing_evaluation.configuration != scenario


def test_unequal_budget_repairs_are_rejected_before_comparison():
    repairs = (
        RepairPlan("a", "A", Q_(10, "dollar"), ControlPatch(), "first"),
        RepairPlan("b", "B", Q_(11, "dollar"), ControlPatch(), "second"),
    )
    with pytest.raises(ValueError, match="same resource budget"):
        compare_equal_budget_repairs(
            instructor_scenario("edge"),
            stress=ControlPatch(energy_work_multiplier=20.0),
            repairs=repairs,
        )


def _exact_inputs(chapter):
    if chapter == 1:
        inputs = {
            "workers": 8,
            "single_worker_compute_time": Q_(8, "second"),
            "communication_payload": Q_(1, "gigabyte"),
            "effective_bandwidth": Q_(8, "gigabyte/second"),
            "communication_startup": Q_(1, "millisecond"),
            "coordination_base": Q_(1, "millisecond"),
            "coordination_per_added_worker": Q_(0.2, "millisecond"),
            "useful_work_per_step": Q_(1024, "count"),
            "overlap_fraction": 0.25,
        }
        result = evaluate_c3_scaling(**inputs)
        return serialize_evaluation(inputs=inputs, result=result)["inputs"]
    if chapter == 7:
        return checkpoint_tradeoff("cloud")["inputs"]
    if chapter == 10:
        scenario = illustrative_track_scenario("cloud")
        return {
            "requests": [request.to_evaluator_args() for request in scenario.requests],
            "service": scenario.service.to_evaluator_args(),
            "replicas": 4,
            "policy": "immediate",
            "slo": serialize_quantity(scenario.slo),
            "max_batch": 1,
            "batch_window": serialize_quantity(Q_(0, "millisecond")),
            "warmup": serialize_quantity(Q_(0, "millisecond")),
            "lost_replicas": 0,
            "service_scale": 1.0,
            "handoff_time": serialize_quantity(Q_(0, "millisecond")),
        }
    energy_inputs = {
        "operations": Q_(1e16, "flop"),
        "moved_bytes": Q_(1e12, "byte"),
        "energy_per_operation": Q_(6, "picojoule/flop"),
        "energy_per_byte": Q_(45, "picojoule/byte"),
        "pue": 1.15,
        "other_it_energy": Q_(1, "kilojoule"),
        "latency": Q_(1, "second"),
        "latency_deadline": Q_(2, "second"),
        "quality": 0.9,
        "minimum_quality": 0.8,
    }
    return serialize_evaluator_args(energy_inputs)


def _ledger_entry(chapter, track="cloud", model_id=None):
    model = model_id or REQUIRED_MODELS[chapter]
    exact_inputs = _exact_inputs(chapter)
    target_part, target_key = REPLAY_TARGETS[REQUIRED_MODELS[chapter]]
    return {
        "schema_version": 1,
        "lab_id": f"v2_{chapter:02d}",
        "track_id": track,
        "model_id": model,
        "evidence": {
            target_part: {
                "track": track,
                "part": target_part,
                "prediction": "higher",
                "inputs": {"selected": "intervention"},
                "baseline": {"inputs": exact_inputs, "outputs": {}},
                "result": {"inputs": exact_inputs, "outputs": {}},
                "alternatives": [],
                "decision": "keep margin",
                "model_key": target_key,
                "upstream_fingerprint": "fixture",
            }
        },
        "recommendation": "keep margin",
        "rejected_alternative": "maximize scale",
        "reevaluation_trigger": "traffic doubles",
        "residual_risk": "unseen burst",
        "rationale": "the intervention preserved the tested limit",
    }


def test_missing_evidence_is_distinct_from_failed_physics_and_fallback_is_labeled():
    physical = evaluate_configuration(instructor_scenario("cloud"))
    audit = audit_ledger_evidence({}, track_id="cloud")
    fallback_audit = audit_ledger_evidence(
        {},
        track_id="cloud",
        instructor_fallbacks={chapter: {"scenario": "illustrative"} for chapter in REQUIRED_MODELS},
    )

    assert physical.physical_feasible
    assert not audit.ready
    assert audit.missing_chapters == tuple(REQUIRED_MODELS)
    assert fallback_audit.ready
    assert all(item.status == "instructor_scenario" for item in fallback_audit.items)
    assert all("instructor scenario" in item.source_label for item in fallback_audit.items)


def test_only_matching_stable_ledger_contract_is_replayable():
    history = {chapter: _ledger_entry(chapter) for chapter in REQUIRED_MODELS}
    history[7] = _ledger_entry(7, model_id="unknown_reliability_model")
    audit = audit_ledger_evidence(history, track_id="cloud")

    assert audit.incompatible_chapters == (7,)
    compatible = next(item for item in audit.items if item.chapter == 1)
    assert evidence_inputs(compatible, part="B", condition="result") == _exact_inputs(1)
    incompatible = next(item for item in audit.items if item.chapter == 7)
    with pytest.raises(ValueError, match="compatible student ledger"):
        evidence_inputs(incompatible, part="E")


@pytest.mark.parametrize("chapter", tuple(REQUIRED_MODELS))
def test_each_recognized_ledger_backend_round_trips_through_real_dispatch(chapter):
    audit = audit_ledger_evidence({chapter: _ledger_entry(chapter)}, track_id="cloud")
    item = next(item for item in audit.items if item.chapter == chapter)
    target_part, _model_key = REPLAY_TARGETS[REQUIRED_MODELS[chapter]]

    assert item.status == "ledger"
    replayed = replay_evidence(item, part=target_part)
    if chapter == 1:
        assert replayed.workers == 8
    elif chapter == 7:
        assert replayed["track_id"] == "cloud"
    elif chapter == 10:
        assert replayed.request_count == 5
    else:
        assert replayed.energy.facility_energy.check("[energy]")


@pytest.mark.parametrize("track_id", TRACKS)
def test_configuration_serialization_round_trip_preserves_physics(track_id):
    scenario = instructor_scenario(track_id)
    restored = deserialize_configuration(serialize_configuration(scenario))

    before = evaluate_configuration(scenario)
    after = evaluate_configuration(restored)
    assert after.checks == before.checks


@pytest.mark.parametrize("decision", ("release", "restrict", "defer"))
def test_deployment_record_preserves_student_choice(decision):
    record = record_deployment_decision(
        decision=decision,
        selected_repair="fabric repair",
        rejected_alternative="efficiency repair",
        condition="rollback if p99 exceeds 200 ms",
        residual_uncertainty="the held-out regional burst has not been replayed",
    )
    assert record.decision == decision
    assert record.condition == "rollback if p99 exceeds 200 ms"


@pytest.mark.parametrize("track_id", ("tinyml", "mobile"))
def test_device_tracks_explicitly_name_fleet_roles_and_boundaries(track_id):
    scenario = instructor_scenario(track_id)
    label = scenario.scenario_label.lower()
    assert "backend training/recovery" in label
    assert "serving pool" in label
    assert "energy boundary" in label
