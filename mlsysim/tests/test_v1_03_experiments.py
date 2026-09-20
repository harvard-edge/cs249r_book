import math
import json

import pytest

from mlsysim.engine.v1_03_experiments import (
    STAGES,
    TRACKS,
    compare_iteration_plans,
    compare_validation_stages,
    evaluate_release_checks,
    feedback_timeline,
    hold_workflow_decision,
    replay,
    simulate_iterations,
    trace_requirement,
    track_summary,
    validation_timing,
)


def test_all_tracks_expose_distinct_workflow_scenarios_and_two_requirements():
    scenarios = set()
    seeded_defects = set()
    for track_id, profile in TRACKS.items():
        scenarios.add(profile["scenario"])
        assert len(profile["requirements"]) == 2
        for requirement_id in profile["requirements"]:
            trace = trace_requirement(track_id, requirement_id)
            assert trace["affected_artifacts"]
            assert sum(trace["affected_stage_counts"].values()) == trace["affected_artifact_count"]
            assert trace["seeded_defect"] in profile["defects"]
            assert trace["seeded_defect"] not in seeded_defects
            seeded_defects.add(trace["seeded_defect"])

    assert len(scenarios) == len(TRACKS)


def test_requirement_graph_changes_only_descendants_of_selected_root():
    latency = trace_requirement("mobile", "latency")
    privacy = trace_requirement("mobile", "privacy")

    assert "latency_requirement" in latency["affected_artifacts"]
    assert "privacy_requirement" in latency["unaffected_artifacts"]
    assert "privacy_requirement" in privacy["affected_artifacts"]
    assert "latency_requirement" in privacy["unaffected_artifacts"]
    assert latency["affected_artifacts"] != privacy["affected_artifacts"]


def test_every_dependency_moves_forward_or_stays_within_a_stage():
    for profile in TRACKS.values():
        artifacts = profile["artifacts"]
        for artifact in artifacts.values():
            for dependency in artifact["depends_on"]:
                assert STAGES.index(artifacts[dependency]["stage"]) <= STAGES.index(artifact["stage"])


@pytest.mark.parametrize("track_id", TRACKS)
def test_late_discovery_sums_more_incurred_rework_for_every_track(track_id):
    early = validation_timing(track_id, "data")
    late = validation_timing(track_id, "deployment")

    assert late["base_rework_days"] > early["base_rework_days"]
    assert late["incurred_artifact_count"] > early["incurred_artifact_count"]
    assert set(early["incurred_artifacts"]) < set(late["incurred_artifacts"])
    assert early["rework_days"] == early["base_rework_days"]
    assert late["rework_days"] == late["base_rework_days"]


def test_escalation_is_an_explicit_linear_sensitivity_not_a_stage_law():
    base = validation_timing("edge", "deployment", escalation_factor=1)
    stressed = validation_timing("edge", "deployment", escalation_factor=1.5)

    assert stressed["base_rework_days"] == base["base_rework_days"]
    assert stressed["rework_days"] == pytest.approx(1.5 * base["rework_days"])
    assert stressed["inspection_days"] == base["inspection_days"]
    assert "Sensitivity" in stressed["escalation_assumption"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_comparison_uses_same_defect_and_finds_avoidable_time(track_id):
    comparison = compare_validation_stages(track_id, "data", "deployment")

    assert comparison["same_seeded_defect"]
    assert comparison["avoidable_days"] > 0
    assert comparison["later"]["total_response_days"] > comparison["earlier"]["total_response_days"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_faster_iteration_completes_more_attempts_but_not_a_releasable_candidate(track_id):
    comparison = compare_iteration_plans(track_id)
    rapid = comparison["rapid_offline"]
    target = comparison["target_in_loop"]

    assert comparison["same_budget"]
    assert rapid["completed_iterations"] > target["completed_iterations"]
    assert not rapid["has_releasable_candidate"]
    assert target["has_releasable_candidate"]
    assert rapid["failed_iterations"] > 0
    assert target["failed_iterations"] > 0


def test_budget_changes_iteration_count_without_changing_supplied_outcomes():
    short = simulate_iterations("cloud", "target_in_loop", development_budget_days=6)
    long = simulate_iterations("cloud", "target_in_loop", development_budget_days=10)

    assert short["completed_iterations"] == 2
    assert long["completed_iterations"] == 3
    assert short["trajectory"] == long["trajectory"][:2]
    assert short["trajectory"][1]["target_task_success_pct"] == long["trajectory"][1]["target_task_success_pct"]


@pytest.mark.parametrize("decision_point", ("iteration", "release"))
def test_hold_is_a_replayable_baseline_without_an_invented_outcome(decision_point):
    held = hold_workflow_decision("edge", decision_point)

    assert held["inputs"] == {
        "track_id": "edge",
        "decision_point": decision_point,
        "action": "hold",
    }
    assert held["status"] == "held"
    assert held["outcome_available"] is False


@pytest.mark.parametrize("track_id", TRACKS)
def test_each_single_release_check_has_a_failure_and_combined_check_covers_seeded_defects(track_id):
    offline = evaluate_release_checks(track_id, "offline")
    target = evaluate_release_checks(track_id, "target")
    combined = evaluate_release_checks(track_id, "combined")

    assert not offline["release_defensible"]
    assert not target["release_defensible"]
    assert offline["escaped_defects"]
    assert target["escaped_defects"]
    assert combined["release_defensible"]
    assert not combined["escaped_defects"]
    assert combined["inspection_days"] == offline["inspection_days"] + target["inspection_days"]


def test_target_check_has_a_distinct_causal_detection_scope():
    offline = evaluate_release_checks("tinyml", "offline")
    target = evaluate_release_checks("tinyml", "target")

    assert "sensor_duty_cycle_mismatch" in offline["escaped_defects"]
    assert "sensor_duty_cycle_mismatch" in target["detected_defects"]
    assert "rare_event_gap" in target["escaped_defects"]
    assert "rare_event_gap" in offline["detected_defects"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_more_frequent_monitoring_reduces_alert_delay_but_does_not_change_outcomes(track_id):
    frequent = feedback_timeline(track_id, signal_check_interval_hours=2)
    sparse = feedback_timeline(track_id, signal_check_interval_hours=7)

    assert frequent["signal_detection_delay_hours"] <= sparse["signal_detection_delay_hours"]
    assert frequent["inspection_count_through_evidence"] > sparse["inspection_count_through_evidence"]
    assert frequent["requests_after_change"] == sparse["requests_after_change"]
    assert frequent["adverse_outcomes_after_change"] == sparse["adverse_outcomes_after_change"]


def test_outcome_delay_changes_evidence_time_not_signal_or_underlying_outcome():
    fast_labels = feedback_timeline("mobile", 5, outcome_delay_hours=10)
    slow_labels = feedback_timeline("mobile", 5, outcome_delay_hours=50)

    fast_alert = next(event for event in fast_labels["timeline"] if event["kind"] == "alert_signal")
    slow_alert = next(event for event in slow_labels["timeline"] if event["kind"] == "alert_signal")
    fast_outcome = next(event for event in fast_labels["timeline"] if event["kind"] == "outcome_evidence")
    slow_outcome = next(event for event in slow_labels["timeline"] if event["kind"] == "outcome_evidence")

    assert fast_alert["hour"] == slow_alert["hour"]
    assert fast_outcome["hour"] < slow_outcome["hour"]
    assert fast_labels["adverse_outcomes_after_change"] == slow_labels["adverse_outcomes_after_change"]
    assert fast_labels["alert_revisit_stage"] == "validation"
    assert fast_labels["outcome_revisit_stage"] == "data"


def test_each_experiment_result_records_canonical_replay_inputs():
    trace = trace_requirement("tinyml")
    timing = validation_timing("tinyml", 5)
    iterations = simulate_iterations("tinyml", "rapid_offline")
    checks = evaluate_release_checks("tinyml", "combined")
    feedback = feedback_timeline("tinyml", 3)

    assert trace["inputs"] == {"track_id": "tinyml", "requirement_id": "energy"}
    assert timing["inputs"] == {
        "track_id": "tinyml",
        "discovery_stage": "deployment",
        "requirement_id": "energy",
        "escalation_factor": 1.0,
    }
    assert iterations["inputs"] == {
        "track_id": "tinyml",
        "plan_id": "rapid_offline",
        "development_budget_days": 8.0,
    }
    assert checks["inputs"] == {"track_id": "tinyml", "check_plan": "combined"}
    assert feedback["inputs"] == {
        "track_id": "tinyml",
        "signal_check_interval_hours": 3.0,
        "outcome_delay_hours": 24.0,
    }


@pytest.mark.parametrize("track_id", TRACKS)
def test_success_and_failure_results_are_strict_json_for_evidence_capture(track_id):
    results = (
        trace_requirement(track_id),
        validation_timing(track_id, "monitoring", escalation_factor=2),
        simulate_iterations(track_id, "target_in_loop", development_budget_days=4),
        evaluate_release_checks(track_id, "offline"),
        evaluate_release_checks(track_id, "combined"),
        hold_workflow_decision(track_id, "release"),
        feedback_timeline(track_id, 7, outcome_delay_hours=48),
    )

    for result in results:
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("track_id", TRACKS)
def test_actual_capture_results_replay_after_json_roundtrip(track_id):
    requirement_ids = tuple(TRACKS[track_id]["requirements"])
    selected = TRACKS[track_id]["default_requirement"]
    other = next(key for key in requirement_ids if key != selected)
    timing = compare_validation_stages(track_id, "data", "deployment", selected)
    actual_results = (
        ("v1_03_experiments.trace_requirement", trace_requirement(track_id, selected)),
        ("v1_03_experiments.trace_requirement", trace_requirement(track_id, other)),
        ("v1_03_experiments.validation_timing", timing["earlier"]),
        ("v1_03_experiments.validation_timing", timing["later"]),
        ("v1_03_experiments.simulate_iterations", simulate_iterations(track_id, "rapid_offline")),
        ("v1_03_experiments.simulate_iterations", simulate_iterations(track_id, "target_in_loop")),
        ("v1_03_experiments.simulate_iterations", hold_workflow_decision(track_id, "iteration")),
        ("v1_03_experiments.evaluate_release_checks", evaluate_release_checks(track_id, "offline")),
        ("v1_03_experiments.evaluate_release_checks", evaluate_release_checks(track_id, "target")),
        ("v1_03_experiments.evaluate_release_checks", evaluate_release_checks(track_id, "combined")),
        ("v1_03_experiments.evaluate_release_checks", hold_workflow_decision(track_id, "release")),
        ("v1_03_experiments.feedback_timeline", feedback_timeline(track_id, 11)),
        ("v1_03_experiments.feedback_timeline", feedback_timeline(track_id, 4, 24)),
    )

    for model_key, original in actual_results:
        snapshot = json.loads(json.dumps(original, allow_nan=False))
        replayed = replay(model_key, snapshot["inputs"])
        assert json.loads(json.dumps(replayed, allow_nan=False)) == snapshot


@pytest.mark.parametrize(
    ("model_key", "inputs"),
    [
        ("unknown", {"track_id": "edge"}),
        ("v1_03_experiments.trace_requirement", {"track_id": "edge"}),
        (
            "v1_03_experiments.trace_requirement",
            {"track_id": "edge", "requirement_id": "latency", "extra": True},
        ),
        (
            "v1_03_experiments.simulate_iterations",
            {"track_id": "edge", "decision_point": "release", "action": "hold"},
        ),
        (
            "v1_03_experiments.evaluate_release_checks",
            {"track_id": "edge", "decision_point": "release", "action": "run"},
        ),
    ],
)
def test_replay_rejects_unknown_dispatch_and_nonexact_inputs(model_key, inputs):
    with pytest.raises(ValueError):
        replay(model_key, inputs)


@pytest.mark.parametrize(
    "call",
    [
        lambda: trace_requirement("unknown"),
        lambda: trace_requirement("edge", "unknown"),
        lambda: validation_timing("edge", "unknown"),
        lambda: validation_timing("edge", 0),
        lambda: validation_timing("edge", "data", escalation_factor=0.5),
        lambda: compare_validation_stages("edge", "deployment", "data"),
        lambda: simulate_iterations("edge", "unknown"),
        lambda: simulate_iterations("edge", "rapid_offline", development_budget_days=math.inf),
        lambda: evaluate_release_checks("edge", "unknown"),
        lambda: hold_workflow_decision("edge", "unknown"),
        lambda: feedback_timeline("edge", 0),
        lambda: feedback_timeline("edge", 1, outcome_delay_hours=-1),
    ],
)
def test_invalid_inputs_fail_explicitly(call):
    with pytest.raises(ValueError):
        call()


def test_track_summary_exposes_track_defaults_and_feedback():
    for track_id, track in TRACKS.items():
        summary = track_summary(track_id)
        assert summary["track_id"] == track_id
        assert summary["default_outcome_delay_hours"] == int(track["feedback"]["outcome_delay_hours"])
        assert summary["requests"] == track["feedback"]["requests"]
        assert summary["development_budget_days"] == track["development_budget_days"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_feedback_timeline_defaults_to_track_outcome_delay(track_id):
    expected_delay = TRACKS[track_id]["feedback"]["outcome_delay_hours"]
    timeline = feedback_timeline(track_id, signal_check_interval_hours=4)
    assert timeline["outcome_delay_hours"] == expected_delay
