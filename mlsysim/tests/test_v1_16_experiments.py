from dataclasses import replace

import pytest

from mlsysim.engine.v1_16_experiments import (
    DeploymentEnvelope,
    EvidenceStatus,
    TRACK_ALIASES,
    apply_prepared_intervention,
    audit_ledger,
    compare_prepared_intervention,
    evaluate_envelope_change,
    evaluate_envelope_fraction,
    evaluate_joint_design,
    prepared_instructor_scenario,
    replay_capture,
    transfer_design,
)


TRACKS = ("tinyml", "mobile", "edge", "cloud")


def capture(track="mobile", part="A", requirement_results=None):
    result = {
        "inputs": {"track_id": track, "candidate": "balanced"},
        "requirement_results": requirement_results or {"latency": True},
    }
    return {
        "track": track,
        "part": part,
        "prediction": "latency binds",
        "inputs": {"candidate": "balanced"},
        "baseline": {"inputs": {"track_id": track, "candidate": "compact"}},
        "result": result,
        "alternatives": [],
        "decision": "keep",
        "model_key": "evaluate",
        "upstream_fingerprint": "a" * 64,
    }


def payload(chapter, *, track="mobile", evidence=None, **updates):
    value = {
        "schema_version": 1,
        "lab_id": f"v1_{chapter:02d}",
        "track_id": track,
        "model_id": f"v1_{chapter:02d}_experiments",
        "evidence": {"A": capture(track=track)} if evidence is None else evidence,
        "recommendation": "keep",
        "rejected_alternative": "larger",
        "reevaluation_trigger": "latency exceeds limit",
        "residual_risk": "unobserved inputs",
        "rationale": "The saved comparison supports the choice.",
    }
    value.update(updates)
    return value


def test_audit_keeps_missing_incomplete_failed_and_incompatible_distinct():
    history = {
        2: payload(2),
        "3": payload(
            3,
            evidence={"A": capture(requirement_results={"latency": False, "memory": True})},
        ),
        4: payload(4, evidence={}),
        5: payload(5, track="edge"),
        6: payload(6, schema_version=2),
    }

    audit = audit_ledger(history, "iphone", required_chapters=(2, 3, 4, 5, 6, 7))

    assert [record.status for record in audit.records] == [
        EvidenceStatus.INCOMPATIBLE,
        EvidenceStatus.INCOMPATIBLE,
        EvidenceStatus.INCOMPLETE,
        EvidenceStatus.INCOMPATIBLE,
        EvidenceStatus.INCOMPATIBLE,
        EvidenceStatus.MISSING,
    ]
    assert audit.records[1].failed_requirements == ()
    assert audit.counts == {
        "missing": 1,
        "incomplete": 1,
        "incompatible": 4,
        "schema_valid_unreplayed": 0,
        "demonstrated": 0,
        "failed": 0,
    }
    assert not audit.release_evidence_complete


def test_audit_rejects_legacy_records_instead_of_guessing_their_schema():
    history = {
        2: {"track": "cloud_fleet", "latency_ms": 12.0},
        3: {"track_id": "cloud", "violations": []},
    }
    audit = audit_ledger(history, "cloud", required_chapters=(2, 3))

    assert all(record.status == EvidenceStatus.INCOMPATIBLE for record in audit.records)
    assert not audit.release_evidence_complete


def test_incomplete_synthesis_is_not_reported_as_missing_or_failed_evidence():
    audit = audit_ledger(
        {2: payload(2, rationale="")},
        "mobile",
        required_chapters=(2,),
    )
    assert audit.records[0].status == EvidenceStatus.INCOMPLETE
    assert audit.records[0].failed_requirements == ()


@pytest.mark.parametrize(
    "payload",
    [
        payload(2, model_id="unknown"),
        payload(2, lab_id="v1_03"),
        payload(2, evidence={"A": {"track": "mobile"}}),
        payload(2, evidence={"A": capture(track="unknown")}),
        payload(2, evidence={"A": {**capture(), "upstream_fingerprint": "short"}}),
        ["not", "a", "mapping"],
    ],
)
def test_malformed_ledger_records_are_incompatible(payload):
    audit = audit_ledger({2: payload}, "mobile", required_chapters=(2,))
    assert audit.records[0].status == EvidenceStatus.INCOMPATIBLE


def test_chapter_09_capture_replays_both_exact_input_arms():
    from mlsysim.engine.v1_09_experiments import evaluate_policy

    baseline = evaluate_policy("mobile", "uniform", 8)
    result = evaluate_policy("mobile", "coverage", 8)
    evidence = capture(track="mobile")
    evidence.update(
        model_key="evaluate_policy",
        baseline=baseline,
        result=result,
    )
    ledger_payload = payload(9, evidence={"A": evidence})

    audit = audit_ledger({9: ledger_payload}, "mobile", required_chapters=(9,))
    replayed = replay_capture("v1_09_experiments", evidence)

    assert audit.records[0].status == EvidenceStatus.DEMONSTRATED
    assert replayed["baseline"] == baseline
    assert replayed["result"] == result


def test_chosen_result_controls_verdict_while_tested_failure_is_preserved():
    from mlsysim.engine.v1_09_experiments import evaluate_policy

    passing = evaluate_policy("mobile", "coverage", 8)
    failing = evaluate_policy("mobile", "uniform", 8)
    evidence = capture(track="mobile")
    evidence.update(
        model_key="evaluate_policy",
        baseline=passing,
        result=failing,
        chosen_result=passing,
        result_role="rejected alternative",
    )
    ledger_payload = payload(9, evidence={"A": evidence})

    audit = audit_ledger({9: ledger_payload}, "mobile", required_chapters=(9,))

    assert audit.records[0].status == EvidenceStatus.DEMONSTRATED
    assert audit.records[0].failed_requirements == ()
    assert audit.records[0].experiment_failures


def test_chapter_06_json_snapshot_replays_through_its_versioned_method():
    from mlsysim.engine.v1_06_experiments import (
        evaluate_candidate,
        evaluation_replay_snapshot,
        get_scenario,
    )

    scenario = get_scenario("tinyml")
    baseline = evaluation_replay_snapshot(evaluate_candidate(scenario, "tiny_conv"))
    result = evaluation_replay_snapshot(evaluate_candidate(scenario, "tiny_recurrent"))
    evidence = capture(track="tinyml")
    evidence.update(
        model_key=baseline["model_key"],
        baseline=baseline,
        result=result,
    )
    ledger_payload = payload(6, track="tinyml", evidence={"A": evidence})

    audit = audit_ledger({6: ledger_payload}, "tinyml", required_chapters=(6,))
    replayed = replay_capture("v1_06_experiments", evidence)

    assert audit.records[0].status == EvidenceStatus.DEMONSTRATED
    assert replayed["baseline"].candidate_id == "tiny_conv"
    assert replayed["result"].candidate_id == "tiny_recurrent"


def test_chapter_05_evaluate_track_replays_with_optional_chosen_result():
    from mlsysim.engine.v1_05_experiments import (
        EVALUATE_MODEL_KEY,
        evaluate_track,
        snapshot,
    )

    baseline = snapshot(evaluate_track("mobile", phase="inference"))
    failing = snapshot(evaluate_track("mobile", phase="training", optimizer="adam"))
    evidence = capture(track="mobile")
    evidence.update(
        model_key=EVALUATE_MODEL_KEY,
        baseline=baseline,
        result=failing,
        chosen_result=baseline,
        result_role="rejected alternative",
    )
    ledger_payload = payload(5, track="mobile", evidence={"A": evidence})

    audit = audit_ledger({5: ledger_payload}, "mobile", required_chapters=(5,))
    replayed = replay_capture("v1_05_experiments", evidence)

    assert audit.records[0].status == EvidenceStatus.DEMONSTRATED
    assert audit.records[0].failed_requirements == ()
    assert audit.records[0].experiment_failures == ("A:result:memory",)
    assert snapshot(replayed["baseline"]) == baseline
    assert snapshot(replayed["result"]) == failing
    assert snapshot(replayed["chosen_result"]) == baseline
    assert replayed["chosen_result"]["violations"] == []
    assert replayed["result"]["violations"] == ["memory"]


def test_chapter_05_numerical_case_replays_with_optional_chosen_result():
    from mlsysim.engine.v1_05_experiments import (
        NUMERICAL_MODEL_KEY,
        run_numerical_case,
        snapshot,
    )

    baseline = snapshot(run_numerical_case("integer_clipping", "fp32"))
    result = snapshot(run_numerical_case("integer_clipping", "int8"))
    evidence = capture(track="mobile")
    evidence.update(
        model_key=NUMERICAL_MODEL_KEY,
        baseline=baseline,
        result=result,
        chosen_result=baseline,
    )
    ledger_payload = payload(5, track="mobile", evidence={"A": evidence})

    audit = audit_ledger({5: ledger_payload}, "mobile", required_chapters=(5,))
    replayed = replay_capture("v1_05_experiments", evidence)

    assert audit.records[0].status == EvidenceStatus.DEMONSTRATED
    assert audit.records[0].failed_requirements == ()
    assert replayed["baseline"] == baseline
    assert replayed["result"] == result
    assert replayed["chosen_result"] == baseline
    assert replayed["baseline"]["input_clipped"] is False
    assert replayed["result"]["input_clipped"] is True
    assert replayed["result"]["output"] == 254


def test_chapter_07_capture_replays_with_optional_chosen_result():
    from mlsysim.engine.v1_07_experiments import (
        MODEL_KEY,
        evaluate_dispatch,
    )

    baseline = evaluate_dispatch("mobile", 6)
    result = evaluate_dispatch("mobile", 1)
    evidence = capture(track="mobile")
    evidence.update(
        model_key=MODEL_KEY,
        baseline=baseline,
        result=result,
        chosen_result=baseline,
    )
    ledger_payload = payload(7, track="mobile", evidence={"A": evidence})

    audit = audit_ledger({7: ledger_payload}, "mobile", required_chapters=(7,))
    replayed = replay_capture("v1_07_experiments", evidence)

    assert audit.records[0].status == EvidenceStatus.DEMONSTRATED
    assert audit.records[0].failed_requirements == ()
    assert replayed["baseline"] == baseline
    assert replayed["result"] == result
    assert replayed["chosen_result"] == baseline
    assert replayed["baseline"]["dispatches"] == 6
    assert replayed["result"]["dispatches"] == 1


def test_unknown_model_keys_are_rejected_for_chapter_05_and_07():
    with pytest.raises(ValueError, match="model_key"):
        replay_capture(
            "v1_05_experiments",
            {
                "model_key": "v1_05.unknown_method",
                "baseline": {"inputs": {"case_id": "rounding", "numeric_format": "fp32"}},
                "result": {"inputs": {"case_id": "rounding", "numeric_format": "int8"}},
            },
        )

    with pytest.raises(ValueError, match="unknown Chapter 7 model_key"):
        replay_capture(
            "v1_07_experiments",
            {
                "model_key": "unknown_key",
                "baseline": {"inputs": {"track_id": "mobile", "dispatches": 6, "tensor_elements": 524288}},
                "result": {"inputs": {"track_id": "mobile", "dispatches": 1, "tensor_elements": 524288}},
            },
        )


def test_instructor_scenario_is_labeled_and_separate_from_student_history():
    scenario = prepared_instructor_scenario("oura_ring")

    assert scenario["kind"] == "prepared_instructor_scenario"
    assert scenario["is_student_history"] is False
    assert scenario["snapshot"].track_id == "tinyml"
    assert "not measurements" in scenario["scenario_assumption"]
    assert "not inferred" in scenario["quality_assumption"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_every_prepared_track_composes_and_has_reachable_constraint_failures(track_id):
    scenario = prepared_instructor_scenario(track_id)
    baseline = evaluate_joint_design(scenario["snapshot"], scenario["envelope"])
    assert baseline["feasible"], (track_id, baseline)

    for limit_name, result_name in (
        ("memory_limit_mb", "memory"),
        ("latency_limit_ms", "latency"),
        ("energy_limit_mj", "energy"),
    ):
        failed_envelope = replace(scenario["envelope"], **{limit_name: 0.01})
        result = evaluate_joint_design(scenario["snapshot"], failed_envelope)
        assert result["requirement_results"][result_name] is False, track_id

    failed_quality = replace(
        scenario["envelope"],
        quality_floor_pct=min(100.0, scenario["snapshot"].quality_pct + 0.1),
    )
    assert evaluate_joint_design(scenario["snapshot"], failed_quality)[
        "requirement_results"
    ]["quality"] is False


def test_joint_latency_and_energy_are_sums_of_explicit_physical_terms():
    scenario = prepared_instructor_scenario("edge")
    result = evaluate_joint_design(scenario["snapshot"], scenario["envelope"])
    snapshot = scenario["snapshot"]

    assert result["latency_ms"] == pytest.approx(
        result["movement_ms"] + result["compute_ms"] + result["overhead_ms"]
    )
    expected_energy_mj = (
        result["movement_ms"] * snapshot.movement_power_w
        + result["compute_ms"] * snapshot.compute_power_w
        + result["overhead_ms"] * snapshot.overhead_power_w
    )
    assert result["energy_mj"] == pytest.approx(expected_energy_mj)


@pytest.mark.parametrize("track_id", TRACKS)
def test_interventions_have_physical_and_matched_quality_tradeoffs(track_id):
    quantized = compare_prepared_intervention(track_id, "quantize")
    larger = compare_prepared_intervention(track_id, "larger_model")

    assert quantized["delta"]["memory_mb"] < 0
    assert quantized["delta"]["movement_ms"] < 0
    assert quantized["delta"]["quality_pct"] < 0
    assert quantized["delta"]["overhead_ms"] > 0

    assert larger["delta"]["memory_mb"] > 0
    assert larger["delta"]["latency_ms"] > 0
    assert larger["delta"]["energy_mj"] > 0
    assert larger["delta"]["quality_pct"] > 0


def test_compute_capability_changes_only_causal_execution_outcomes():
    scenario = prepared_instructor_scenario("mobile")
    snapshot = scenario["snapshot"]
    baseline = evaluate_joint_design(snapshot, scenario["envelope"])
    faster = evaluate_joint_design(
        replace(snapshot, compute_rate_mflop_per_ms=2 * snapshot.compute_rate_mflop_per_ms),
        scenario["envelope"],
    )

    assert faster["compute_ms"] == pytest.approx(baseline["compute_ms"] / 2)
    assert faster["latency_ms"] < baseline["latency_ms"]
    assert faster["energy_mj"] < baseline["energy_mj"]
    assert faster["memory_mb"] == baseline["memory_mb"]
    assert faster["movement_ms"] == baseline["movement_ms"]
    assert faster["quality_pct"] == baseline["quality_pct"]


def test_quality_evidence_does_not_change_physical_outcomes():
    scenario = prepared_instructor_scenario("edge")
    snapshot = scenario["snapshot"]
    baseline = evaluate_joint_design(snapshot, scenario["envelope"])
    new_observation = evaluate_joint_design(
        replace(snapshot, quality_pct=snapshot.quality_pct - 8.0, quality_evidence_id="new_matched_observation"),
        scenario["envelope"],
    )

    for key in ("memory_mb", "movement_ms", "compute_ms", "overhead_ms", "latency_ms", "energy_mj"):
        assert new_observation[key] == baseline[key]
    assert new_observation["quality_pct"] != baseline["quality_pct"]


def test_envelope_changes_acceptance_without_changing_physical_outcomes():
    scenario = prepared_instructor_scenario("cloud")
    comparison = evaluate_envelope_change(
        scenario["snapshot"],
        scenario["envelope"],
        latency_limit_ms=10.0,
    )

    for key in ("memory_mb", "movement_ms", "compute_ms", "overhead_ms", "latency_ms", "energy_mj", "quality_pct"):
        assert comparison["changed"][key] == comparison["baseline"][key]
    assert comparison["baseline"]["requirement_results"]["latency"] is True
    assert comparison["changed"]["requirement_results"]["latency"] is False


def test_envelope_fraction_changes_only_the_selected_limit():
    scenario = prepared_instructor_scenario("mobile")
    comparison = evaluate_envelope_fraction(
        scenario["snapshot"], scenario["envelope"],
        requirement="latency", fraction=0.5,
    )

    assert comparison["changed"]["limits"]["latency_ms"] == pytest.approx(
        comparison["baseline"]["limits"]["latency_ms"] / 2
    )
    assert comparison["changed"]["limits"]["memory_mb"] == comparison["baseline"]["limits"]["memory_mb"]
    assert comparison["changed"]["limits"]["energy_mj"] == comparison["baseline"]["limits"]["energy_mj"]


def test_cross_task_transfer_marks_quality_unavailable_until_matched_evidence_exists():
    source = prepared_instructor_scenario("mobile")["snapshot"]
    target = prepared_instructor_scenario("edge")["envelope"]

    unknown_quality = transfer_design(source, target)
    observed_quality = transfer_design(
        source,
        target,
        matched_quality_pct=92.0,
        matched_quality_evidence_id="edge_population_observation_v1",
    )

    assert unknown_quality["quality_pct"] is None
    assert unknown_quality["requirement_results"]["quality"] is None
    assert "quality" in unknown_quality["unavailable_requirements"]
    assert not unknown_quality["feasible"]
    assert observed_quality["quality_pct"] == 92.0
    assert observed_quality["requirement_results"]["quality"] is True


def test_aliases_cover_current_and_canonical_track_ids():
    assert set(TRACKS).issubset(TRACK_ALIASES)
    assert TRACK_ALIASES["oura_ring"] == "tinyml"
    assert TRACK_ALIASES["iphone"] == "mobile"
    assert TRACK_ALIASES["robotaxi"] == "edge"
    assert TRACK_ALIASES["cloud_fleet"] == "cloud"


def test_prepared_intervention_rejects_unlabeled_student_snapshot():
    scenario = prepared_instructor_scenario("tinyml")
    student = replace(scenario["snapshot"], source="student_ledger")
    with pytest.raises(ValueError, match="prepared instructor"):
        apply_prepared_intervention(student, "quantize")


def test_invalid_inputs_fail_clearly():
    scenario = prepared_instructor_scenario("tinyml")
    with pytest.raises(ValueError, match="track_id"):
        prepared_instructor_scenario("serverless")
    with pytest.raises(ValueError, match="quality_evidence_id"):
        evaluate_joint_design(
            replace(scenario["snapshot"], quality_evidence_id=None),
            scenario["envelope"],
        )
    with pytest.raises(ValueError, match="supplied together"):
        transfer_design(
            scenario["snapshot"],
            scenario["envelope"],
            matched_quality_pct=90.0,
        )
    with pytest.raises(ValueError, match="positive"):
        evaluate_joint_design(
            scenario["snapshot"],
            replace(scenario["envelope"], latency_limit_ms=0),
        )
