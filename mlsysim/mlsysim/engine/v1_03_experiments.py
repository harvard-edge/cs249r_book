"""Workflow experiments for the Volume I lifecycle lab.

All profiles in this module are explicit, illustrative scenario assumptions.
They are not measurements of a production team, model, or device.  The model
keeps four things separate:

* requirement propagation is a dependency-graph traversal;
* rework is the sum of affected artifact costs already incurred;
* iteration outcomes come from supplied candidate trajectories; and
* monitoring reveals signals and delayed outcomes without changing them.

Time arithmetic uses Pint internally and public results include the unit in
their field name.  Percentages are task-specific observed rates in the fixed
scenario fixtures, never generic readiness or confidence scores.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from typing import Any

from mlsysim.core.units import Q_


STAGES = (
    "definition",
    "data",
    "modeling",
    "validation",
    "deployment",
    "monitoring",
)

SCENARIO_NOTICE = (
    "Illustrative workflow fixture for causal comparison; not an empirical "
    "measurement of a team, application, or branded system."
)


def _artifact(stage: str, days: float, depends_on: tuple[str, ...]) -> dict[str, Any]:
    return {"stage": stage, "rework_days": days, "depends_on": depends_on}


def _attempt(candidate: str, days: float, offline: float, target: float | None, outcome: str) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "duration_days": days,
        "offline_task_success_pct": offline,
        "target_task_success_pct": target,
        "outcome": outcome,
    }


# Each track poses the same five questions with a different binding requirement,
# target-only defect, iteration cadence, and feedback delay.
TRACKS: dict[str, dict[str, Any]] = {
    "tinyml": {
        "display": "TinyML",
        "scenario": "A battery-powered acoustic event classifier",
        "requirements": {
            "energy": {
                "label": "Energy per decision",
                "limit": "1.5 mJ per decision",
                "root": "energy_requirement",
                "seeded_defect": "continuous_sampling_assumption",
            },
            "memory": {
                "label": "Peak working memory",
                "limit": "256 KiB",
                "root": "memory_requirement",
                "seeded_defect": "activation_buffer_overflow",
            },
        },
        "default_requirement": "energy",
        "artifacts": {
            "energy_requirement": _artifact("definition", 0.5, ()),
            "memory_requirement": _artifact("definition", 0.5, ()),
            "sampling_plan": _artifact("data", 1.0, ("energy_requirement",)),
            "feature_window": _artifact("data", 1.5, ("sampling_plan", "memory_requirement")),
            "model_candidate": _artifact("modeling", 2.0, ("feature_window",)),
            "device_runtime": _artifact("validation", 2.5, ("model_candidate", "energy_requirement")),
            "device_test": _artifact("validation", 1.0, ("model_candidate", "device_runtime")),
            "release_package": _artifact("deployment", 1.0, ("device_test",)),
            "monitoring_policy": _artifact("monitoring", 0.5, ("release_package", "energy_requirement")),
        },
        "inspection_days": (0.5, 0.8, 1.0, 1.2, 1.5, 1.5),
        "development_budget_days": 8.0,
        "release_floor_pct": 82.0,
        "iteration_plans": {
            "rapid_offline": (
                _attempt("A", 1.5, 86.0, None, "offline_pass"),
                _attempt("B", 1.5, 88.0, None, "offline_pass"),
                _attempt("C", 1.5, 90.0, 76.0, "target_failure"),
                _attempt("D", 1.5, 91.0, 79.0, "target_failure"),
                _attempt("E", 1.5, 91.5, 80.0, "target_failure"),
            ),
            "target_in_loop": (
                _attempt("A", 2.5, 86.0, 78.0, "target_failure"),
                _attempt("B", 2.5, 87.0, 83.0, "released"),
                _attempt("C", 2.5, 88.0, 85.0, "released"),
            ),
        },
        "defects": (
            "rare_event_gap",
            "activation_buffer_overflow",
            "sensor_duty_cycle_mismatch",
            "continuous_sampling_assumption",
        ),
        "offline_detects": ("rare_event_gap", "activation_buffer_overflow"),
        "target_detects": (
            "activation_buffer_overflow",
            "sensor_duty_cycle_mismatch",
            "continuous_sampling_assumption",
        ),
        "offline_check_days": 0.8,
        "target_check_days": 1.2,
        "feedback": {"shift_hour": 12.0, "requests": 400, "adverse_outcomes": 32, "outcome_delay_hours": 24.0},
    },
    "mobile": {
        "display": "Mobile",
        "scenario": "An on-device image assistance feature",
        "requirements": {
            "latency": {
                "label": "Interactive latency",
                "limit": "40 ms per image",
                "root": "latency_requirement",
                "seeded_defect": "camera_preprocessing_mismatch",
            },
            "privacy": {
                "label": "Raw-image retention",
                "limit": "no raw images leave the device",
                "root": "privacy_requirement",
                "seeded_defect": "debug_upload_enabled",
            },
        },
        "default_requirement": "latency",
        "artifacts": {
            "latency_requirement": _artifact("definition", 0.5, ()),
            "privacy_requirement": _artifact("definition", 0.5, ()),
            "capture_contract": _artifact("data", 1.0, ("privacy_requirement", "latency_requirement")),
            "preprocessing_graph": _artifact("data", 1.5, ("capture_contract",)),
            "model_candidate": _artifact("modeling", 2.0, ("preprocessing_graph", "latency_requirement")),
            "app_integration": _artifact("validation", 3.0, ("model_candidate", "privacy_requirement")),
            "phone_test": _artifact("validation", 1.5, ("app_integration",)),
            "app_release": _artifact("deployment", 1.0, ("phone_test",)),
            "telemetry_policy": _artifact("monitoring", 1.0, ("app_release", "privacy_requirement")),
        },
        "inspection_days": (0.5, 0.8, 1.0, 1.5, 2.0, 2.0),
        "development_budget_days": 10.0,
        "release_floor_pct": 86.0,
        "iteration_plans": {
            "rapid_offline": (
                _attempt("A", 1.5, 88.0, None, "offline_pass"),
                _attempt("B", 1.5, 90.0, None, "offline_pass"),
                _attempt("C", 1.5, 91.0, 81.0, "target_failure"),
                _attempt("D", 1.5, 92.0, 83.0, "target_failure"),
                _attempt("E", 1.5, 92.5, 84.0, "target_failure"),
                _attempt("F", 1.5, 93.0, 84.5, "target_failure"),
            ),
            "target_in_loop": (
                _attempt("A", 3.0, 88.0, 83.0, "target_failure"),
                _attempt("B", 3.0, 89.0, 87.0, "released"),
                _attempt("C", 3.0, 90.0, 88.0, "released"),
            ),
        },
        "defects": (
            "low_light_gap",
            "os_memory_pressure",
            "camera_preprocessing_mismatch",
            "debug_upload_enabled",
        ),
        "offline_detects": ("low_light_gap", "os_memory_pressure", "debug_upload_enabled"),
        "target_detects": ("os_memory_pressure", "camera_preprocessing_mismatch"),
        "offline_check_days": 1.0,
        "target_check_days": 1.5,
        "feedback": {"shift_hour": 24.0, "requests": 1000, "adverse_outcomes": 55, "outcome_delay_hours": 48.0},
    },
    "edge": {
        "display": "Edge",
        "scenario": "A local visual inspection station",
        "requirements": {
            "latency": {
                "label": "Inspection deadline",
                "limit": "25 ms per item",
                "root": "latency_requirement",
                "seeded_defect": "sustained_thermal_slowdown",
            },
            "availability": {
                "label": "Disconnected operation",
                "limit": "8 hours without network service",
                "root": "availability_requirement",
                "seeded_defect": "remote_label_dependency",
            },
        },
        "default_requirement": "latency",
        "artifacts": {
            "latency_requirement": _artifact("definition", 0.5, ()),
            "availability_requirement": _artifact("definition", 0.5, ()),
            "sensor_contract": _artifact("data", 1.0, ("latency_requirement", "availability_requirement")),
            "preprocessing_pipeline": _artifact("data", 2.0, ("sensor_contract",)),
            "model_candidate": _artifact("modeling", 2.5, ("preprocessing_pipeline", "latency_requirement")),
            "station_integration": _artifact("validation", 3.0, ("model_candidate", "availability_requirement")),
            "sustained_test": _artifact("validation", 2.0, ("station_integration",)),
            "station_release": _artifact("deployment", 1.0, ("sustained_test",)),
            "alert_route": _artifact("monitoring", 1.0, ("station_release", "availability_requirement")),
        },
        "inspection_days": (0.5, 0.8, 1.2, 1.8, 2.5, 2.5),
        "development_budget_days": 12.0,
        "release_floor_pct": 90.0,
        "iteration_plans": {
            "rapid_offline": (
                _attempt("A", 2.0, 91.0, None, "offline_pass"),
                _attempt("B", 2.0, 93.0, None, "offline_pass"),
                _attempt("C", 2.0, 94.0, 86.0, "target_failure"),
                _attempt("D", 2.0, 95.0, 88.0, "target_failure"),
                _attempt("E", 2.0, 95.5, 89.0, "target_failure"),
                _attempt("F", 2.0, 96.0, 89.5, "target_failure"),
            ),
            "target_in_loop": (
                _attempt("A", 3.5, 91.0, 87.0, "target_failure"),
                _attempt("B", 3.5, 92.0, 91.0, "released"),
                _attempt("C", 3.5, 93.0, 92.0, "released"),
            ),
        },
        "defects": (
            "rare_surface_gap",
            "sensor_format_mismatch",
            "sustained_thermal_slowdown",
            "remote_label_dependency",
        ),
        "offline_detects": ("rare_surface_gap", "sensor_format_mismatch"),
        "target_detects": ("sensor_format_mismatch", "sustained_thermal_slowdown", "remote_label_dependency"),
        "offline_check_days": 1.2,
        "target_check_days": 2.0,
        "feedback": {"shift_hour": 8.0, "requests": 800, "adverse_outcomes": 28, "outcome_delay_hours": 12.0},
    },
    "cloud": {
        "display": "Cloud",
        "scenario": "A shared document classification endpoint",
        "requirements": {
            "latency": {
                "label": "Service response deadline",
                "limit": "150 ms per request",
                "root": "latency_requirement",
                "seeded_defect": "production_payload_mismatch",
            },
            "cost": {
                "label": "Serving cost",
                "limit": "$0.30 per 1,000 requests",
                "root": "cost_requirement",
                "seeded_defect": "retry_amplification",
            },
        },
        "default_requirement": "latency",
        "artifacts": {
            "latency_requirement": _artifact("definition", 0.5, ()),
            "cost_requirement": _artifact("definition", 0.5, ()),
            "request_contract": _artifact("data", 1.0, ("latency_requirement", "cost_requirement")),
            "batching_plan": _artifact("modeling", 1.5, ("request_contract", "cost_requirement")),
            "model_candidate": _artifact("modeling", 2.0, ("batching_plan", "latency_requirement")),
            "service_integration": _artifact("validation", 3.0, ("model_candidate", "request_contract")),
            "staging_test": _artifact("validation", 1.5, ("service_integration",)),
            "service_release": _artifact("deployment", 1.0, ("staging_test",)),
            "service_alert": _artifact("monitoring", 1.0, ("service_release", "cost_requirement")),
        },
        "inspection_days": (0.5, 0.8, 1.0, 1.5, 2.0, 2.0),
        "development_budget_days": 10.0,
        "release_floor_pct": 88.0,
        "iteration_plans": {
            "rapid_offline": (
                _attempt("A", 1.5, 90.0, None, "offline_pass"),
                _attempt("B", 1.5, 92.0, None, "offline_pass"),
                _attempt("C", 1.5, 93.0, 82.0, "target_failure"),
                _attempt("D", 1.5, 94.0, 85.0, "target_failure"),
                _attempt("E", 1.5, 94.5, 86.0, "target_failure"),
                _attempt("F", 1.5, 95.0, 87.0, "target_failure"),
            ),
            "target_in_loop": (
                _attempt("A", 3.0, 90.0, 84.0, "target_failure"),
                _attempt("B", 3.0, 91.0, 89.0, "released"),
                _attempt("C", 3.0, 92.0, 90.0, "released"),
            ),
        },
        "defects": (
            "long_document_gap",
            "production_payload_mismatch",
            "identity_config_mismatch",
            "retry_amplification",
        ),
        "offline_detects": ("long_document_gap", "production_payload_mismatch"),
        "target_detects": ("production_payload_mismatch", "identity_config_mismatch", "retry_amplification"),
        "offline_check_days": 1.0,
        "target_check_days": 1.5,
        "feedback": {"shift_hour": 18.0, "requests": 2000, "adverse_outcomes": 70, "outcome_delay_hours": 36.0},
    },
}


def _track(track_id: str) -> dict[str, Any]:
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}")
    return TRACKS[track_id]


def _finite(value: float, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _stage_index(stage: str | int) -> int:
    if isinstance(stage, bool):
        raise ValueError("stage must be a lifecycle stage name or number")
    if isinstance(stage, int):
        if 1 <= stage <= len(STAGES):
            return stage - 1
        raise ValueError(f"stage number must be between 1 and {len(STAGES)}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}")
    return STAGES.index(stage)


def _days(value: float) -> float:
    return round(Q_(value, "day").to("day").magnitude, 4)


def trace_requirement(track_id: str, requirement_id: str | None = None) -> dict[str, Any]:
    """Traverse the lifecycle graph from a selected requirement.

    Only descendants of the requirement root are returned.  Unrelated roots
    provide an explicit noncausal control for the lab and tests.
    """
    track = _track(track_id)
    requirement_id = requirement_id or track["default_requirement"]
    requirements = track["requirements"]
    if requirement_id not in requirements:
        raise ValueError(f"requirement_id must be one of {', '.join(requirements)}")
    root = requirements[requirement_id]["root"]
    artifacts = track["artifacts"]
    children: dict[str, list[str]] = {name: [] for name in artifacts}
    for name, artifact in artifacts.items():
        for dependency in artifact["depends_on"]:
            children[dependency].append(name)

    affected: list[str] = []
    queue = deque([root])
    seen = {root}
    while queue:
        current = queue.popleft()
        affected.append(current)
        for child in children[current]:
            if child not in seen:
                seen.add(child)
                queue.append(child)
    affected.sort(key=lambda name: (STAGES.index(artifacts[name]["stage"]), name))
    unaffected = [name for name in artifacts if name not in seen]
    paths = [
        {"artifact": name, "stage": artifacts[name]["stage"], "depends_on": artifacts[name]["depends_on"]}
        for name in affected
    ]
    affected_stage_counts = {
        stage: sum(artifacts[name]["stage"] == stage for name in affected)
        for stage in STAGES
    }
    return {
        "track_id": track_id,
        "track": track["display"],
        "scenario": track["scenario"],
        "requirement_id": requirement_id,
        "requirement_label": requirements[requirement_id]["label"],
        "requirement_limit": requirements[requirement_id]["limit"],
        "seeded_defect": requirements[requirement_id]["seeded_defect"],
        "inputs": {"track_id": track_id, "requirement_id": requirement_id},
        "affected_artifact_count": len(affected),
        "affected_stage_counts": affected_stage_counts,
        "affected_artifacts": tuple(affected),
        "unaffected_artifacts": tuple(unaffected),
        "propagation": tuple(paths),
        "scenario_notice": SCENARIO_NOTICE,
    }


def validation_timing(
    track_id: str,
    discovery_stage: str | int,
    requirement_id: str | None = None,
    escalation_factor: float = 1.0,
) -> dict[str, Any]:
    """Compute inspection plus rework cost for discovering one seeded defect.

    ``escalation_factor`` is an explicit sensitivity assumption applied to
    incurred artifact rework, not an empirical law and not an exponential
    function of stage.  The default is one, so base results are direct sums.
    """
    stage_index = _stage_index(discovery_stage)
    escalation_factor = _finite(escalation_factor, "escalation_factor", minimum=1.0)
    track = _track(track_id)
    trace = trace_requirement(track_id, requirement_id)
    artifacts = track["artifacts"]
    incurred = [
        name
        for name in trace["affected_artifacts"]
        if _stage_index(artifacts[name]["stage"]) <= stage_index
    ]
    base_rework = sum(artifacts[name]["rework_days"] for name in incurred)
    inspection = track["inspection_days"][stage_index]
    adjusted_rework = base_rework * escalation_factor
    return {
        **trace,
        "inputs": {
            "track_id": track_id,
            "discovery_stage": STAGES[stage_index],
            "requirement_id": trace["requirement_id"],
            "escalation_factor": escalation_factor,
        },
        "discovery_stage": STAGES[stage_index],
        "discovery_stage_number": stage_index + 1,
        "inspection_days": _days(inspection),
        "incurred_artifacts": tuple(incurred),
        "incurred_artifact_count": len(incurred),
        "base_rework_days": _days(base_rework),
        "escalation_factor": escalation_factor,
        "rework_days": _days(adjusted_rework),
        "total_response_days": _days(inspection + adjusted_rework),
        "escalation_assumption": (
            "Sensitivity multiplier chosen by the learner; 1.0 is the direct artifact-cost sum."
        ),
    }


def compare_validation_stages(
    track_id: str,
    earlier_stage: str | int,
    later_stage: str | int,
    requirement_id: str | None = None,
    escalation_factor: float = 1.0,
) -> dict[str, Any]:
    """Compare two discovery stages against the same requirement defect."""
    earlier = validation_timing(track_id, earlier_stage, requirement_id, escalation_factor)
    later = validation_timing(track_id, later_stage, requirement_id, escalation_factor)
    if earlier["discovery_stage_number"] >= later["discovery_stage_number"]:
        raise ValueError("earlier_stage must precede later_stage")
    return {
        "inputs": {
            "track_id": track_id,
            "earlier_stage": earlier["discovery_stage"],
            "later_stage": later["discovery_stage"],
            "requirement_id": earlier["requirement_id"],
            "escalation_factor": escalation_factor,
        },
        "earlier": earlier,
        "later": later,
        "avoidable_days": _days(later["total_response_days"] - earlier["total_response_days"]),
        "same_seeded_defect": earlier["seeded_defect"] == later["seeded_defect"],
        "scenario_notice": SCENARIO_NOTICE,
    }


def simulate_iterations(
    track_id: str,
    plan_id: str,
    development_budget_days: float | None = None,
) -> dict[str, Any]:
    """Run a supplied candidate trajectory inside a finite time budget."""
    track = _track(track_id)
    if plan_id not in track["iteration_plans"]:
        raise ValueError(f"plan_id must be one of {', '.join(track['iteration_plans'])}")
    budget = track["development_budget_days"] if development_budget_days is None else _finite(
        development_budget_days, "development_budget_days", minimum=0.01
    )
    elapsed = Q_(0, "day")
    budget_q = Q_(budget, "day")
    completed: list[dict[str, Any]] = []
    for ordinal, attempt in enumerate(track["iteration_plans"][plan_id], start=1):
        duration = Q_(attempt["duration_days"], "day")
        if elapsed + duration > budget_q:
            break
        elapsed += duration
        target = attempt["target_task_success_pct"]
        record = dict(attempt)
        record.update(
            {
                "iteration": ordinal,
                "completed_at_day": _days(elapsed.magnitude),
                "meets_release_floor": target is not None and target >= track["release_floor_pct"],
            }
        )
        completed.append(record)
    qualifying = [attempt for attempt in completed if attempt["meets_release_floor"]]
    observed_target = [attempt for attempt in completed if attempt["target_task_success_pct"] is not None]
    best = max(observed_target, key=lambda item: item["target_task_success_pct"], default=None)
    return {
        "track_id": track_id,
        "plan_id": plan_id,
        "inputs": {
            "track_id": track_id,
            "plan_id": plan_id,
            "development_budget_days": _days(budget),
        },
        "development_budget_days": _days(budget),
        "elapsed_days": _days(elapsed.magnitude),
        "unused_days": _days((budget_q - elapsed).magnitude),
        "completed_iterations": len(completed),
        "failed_iterations": sum(item["outcome"] == "target_failure" for item in completed),
        "trajectory": tuple(completed),
        "release_floor_pct": track["release_floor_pct"],
        "best_candidate": None if best is None else best["candidate"],
        "best_target_task_success_pct": None if best is None else best["target_task_success_pct"],
        "has_releasable_candidate": bool(qualifying),
        "quality_fixture_notice": (
            "Illustrative task-success observations supplied by the scenario; not a universal model-quality curve."
        ),
        "scenario_notice": SCENARIO_NOTICE,
    }


def compare_iteration_plans(track_id: str, development_budget_days: float | None = None) -> dict[str, Any]:
    """Compare the two supplied iteration plans under one shared budget."""
    rapid = simulate_iterations(track_id, "rapid_offline", development_budget_days)
    target = simulate_iterations(track_id, "target_in_loop", development_budget_days)
    return {
        "inputs": {
            "track_id": track_id,
            "development_budget_days": rapid["development_budget_days"],
        },
        "rapid_offline": rapid,
        "target_in_loop": target,
        "iteration_count_delta": rapid["completed_iterations"] - target["completed_iterations"],
        "same_budget": rapid["development_budget_days"] == target["development_budget_days"],
        "scenario_notice": SCENARIO_NOTICE,
    }


def hold_workflow_decision(track_id: str, decision_point: str) -> dict[str, Any]:
    """Represent an explicit decision to stop before iteration or release.

    This has no simulated performance outcome.  It exists so a defensible
    no-feasible decision can be captured as the chosen baseline while a tested
    candidate remains the comparison result.
    """
    track = _track(track_id)
    if decision_point not in ("iteration", "release"):
        raise ValueError("decision_point must be one of iteration, release")
    return {
        "track_id": track_id,
        "inputs": {"track_id": track_id, "decision_point": decision_point, "action": "hold"},
        "decision_point": decision_point,
        "action": "hold",
        "status": "held",
        "outcome_available": False,
        "reason": "No tested candidate was selected; the workflow stops at this decision point.",
        "scenario": track["scenario"],
        "scenario_notice": SCENARIO_NOTICE,
    }


def evaluate_release_checks(track_id: str, check_plan: str) -> dict[str, Any]:
    """Apply concrete checks to seeded defects and report escaped defects."""
    if check_plan not in ("offline", "target", "combined"):
        raise ValueError("check_plan must be one of offline, target, combined")
    track = _track(track_id)
    defects = set(track["defects"])
    offline = set(track["offline_detects"])
    target = set(track["target_detects"])
    if check_plan == "offline":
        detected = offline
        cost = track["offline_check_days"]
    elif check_plan == "target":
        detected = target
        cost = track["target_check_days"]
    else:
        detected = offline | target
        cost = track["offline_check_days"] + track["target_check_days"]
    escaped = defects - detected
    return {
        "track_id": track_id,
        "check_plan": check_plan,
        "inputs": {"track_id": track_id, "check_plan": check_plan},
        "seeded_defects": tuple(sorted(defects)),
        "detected_defects": tuple(sorted(detected & defects)),
        "escaped_defects": tuple(sorted(escaped)),
        "seeded_defect_count": len(defects),
        "detected_defect_count": len(detected & defects),
        "escaped_defect_count": len(escaped),
        "inspection_days": _days(cost),
        "release_defensible": not escaped,
        "decision": "hold" if escaped else "release",
        "evidence_scope": (
            "Detection results apply only to the named seeded defects; unseeded failure modes remain a limitation."
        ),
        "scenario_notice": SCENARIO_NOTICE,
    }


def feedback_timeline(
    track_id: str,
    signal_check_interval_hours: float,
    outcome_delay_hours: float | None = None,
) -> dict[str, Any]:
    """Build a production timeline that separates alerts from outcomes.

    A signal check is aligned to fixed intervals starting at release hour zero.
    Changing the interval changes alert delay and inspection count only.  It
    cannot change the fixture's underlying requests or adverse outcomes.
    """
    track = _track(track_id)
    feedback = track["feedback"]
    interval = _finite(signal_check_interval_hours, "signal_check_interval_hours", minimum=0.01)
    outcome_delay = feedback["outcome_delay_hours"] if outcome_delay_hours is None else _finite(
        outcome_delay_hours, "outcome_delay_hours", minimum=0.0
    )
    shift = Q_(feedback["shift_hour"], "hour")
    interval_q = Q_(interval, "hour")
    check_number = math.ceil((shift / interval_q).to("").magnitude)
    alert = check_number * interval_q
    outcome_available = shift + Q_(outcome_delay, "hour")
    events = (
        {"hour": 0.0, "kind": "release", "revisit_stage": None, "meaning": "baseline enters production"},
        {
            "hour": round(shift.magnitude, 4),
            "kind": "world_change",
            "revisit_stage": None,
            "meaning": "the production population changes",
        },
        {
            "hour": round(alert.magnitude, 4),
            "kind": "alert_signal",
            "revisit_stage": "validation",
            "meaning": "input signal warrants investigation; it does not establish task harm",
        },
        {
            "hour": round(outcome_available.magnitude, 4),
            "kind": "outcome_evidence",
            "revisit_stage": "data",
            "meaning": "labeled outcomes can test whether the population and model evidence remain valid",
        },
    )
    events = tuple(sorted(events, key=lambda event: (event["hour"], event["kind"])))
    horizon = max(alert, outcome_available)
    inspection_count = math.floor((horizon / interval_q).to("").magnitude)
    return {
        "track_id": track_id,
        "inputs": {
            "track_id": track_id,
            "signal_check_interval_hours": round(interval, 4),
            "outcome_delay_hours": round(outcome_delay, 4),
        },
        "signal_check_interval_hours": round(interval, 4),
        "signal_detection_delay_hours": round((alert - shift).to("hour").magnitude, 4),
        "outcome_delay_hours": round(outcome_delay, 4),
        "inspection_count_through_evidence": inspection_count,
        "requests_after_change": feedback["requests"],
        "adverse_outcomes_after_change": feedback["adverse_outcomes"],
        "timeline": events,
        "alert_revisit_stage": "validation",
        "outcome_revisit_stage": "data",
        "causal_note": (
            "Monitoring cadence changes observation time and inspection effort, not the fixed underlying outcomes."
        ),
        "scenario_notice": SCENARIO_NOTICE,
    }


_REPLAY_MODELS = {
    "v1_03_experiments.trace_requirement",
    "v1_03_experiments.validation_timing",
    "v1_03_experiments.simulate_iterations",
    "v1_03_experiments.evaluate_release_checks",
    "v1_03_experiments.feedback_timeline",
}


def _replay_payload(inputs: Mapping[str, Any], expected: set[str]) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a mapping")
    payload = dict(inputs)
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"replay inputs mismatch: missing={missing}, extra={extra}")
    return payload


def replay(model_key: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one captured Chapter 3 result through an allowlisted API.

    ``inputs`` must be the exact mapping stored inside a baseline, result, or
    chosen result.  No dynamic imports, attribute lookup, or inferred defaults
    are used at this boundary.
    """
    if model_key not in _REPLAY_MODELS:
        raise ValueError(f"unknown V1-03 replay model: {model_key}")

    if model_key == "v1_03_experiments.trace_requirement":
        payload = _replay_payload(inputs, {"track_id", "requirement_id"})
        return trace_requirement(**payload)

    if model_key == "v1_03_experiments.validation_timing":
        payload = _replay_payload(
            inputs,
            {"track_id", "discovery_stage", "requirement_id", "escalation_factor"},
        )
        return validation_timing(**payload)

    if model_key == "v1_03_experiments.simulate_iterations":
        if isinstance(inputs, Mapping) and "action" in inputs:
            payload = _replay_payload(inputs, {"track_id", "decision_point", "action"})
            if payload.pop("action") != "hold" or payload["decision_point"] != "iteration":
                raise ValueError("simulate_iterations hold replay requires action='hold' at iteration")
            return hold_workflow_decision(**payload)
        payload = _replay_payload(inputs, {"track_id", "plan_id", "development_budget_days"})
        return simulate_iterations(**payload)

    if model_key == "v1_03_experiments.evaluate_release_checks":
        if isinstance(inputs, Mapping) and "action" in inputs:
            payload = _replay_payload(inputs, {"track_id", "decision_point", "action"})
            if payload.pop("action") != "hold" or payload["decision_point"] != "release":
                raise ValueError("evaluate_release_checks hold replay requires action='hold' at release")
            return hold_workflow_decision(**payload)
        payload = _replay_payload(inputs, {"track_id", "check_plan"})
        return evaluate_release_checks(**payload)

    payload = _replay_payload(
        inputs,
        {"track_id", "signal_check_interval_hours", "outcome_delay_hours"},
    )
    return feedback_timeline(**payload)


def track_summary(track_id: str) -> Mapping[str, Any]:
    """Return the learner-facing choices for one track without live results."""
    track = _track(track_id)
    return {
        "track_id": track_id,
        "display": track["display"],
        "scenario": track["scenario"],
        "requirements": {
            key: {"label": value["label"], "limit": value["limit"]}
            for key, value in track["requirements"].items()
        },
        "default_requirement": track["default_requirement"],
        "development_budget_days": track["development_budget_days"],
        "release_floor_pct": track["release_floor_pct"],
        "default_outcome_delay_hours": int(track["feedback"]["outcome_delay_hours"]),
        "shift_hour": track["feedback"]["shift_hour"],
        "requests": track["feedback"]["requests"],
        "adverse_outcomes": track["feedback"]["adverse_outcomes"],
        "scenario_notice": SCENARIO_NOTICE,
    }
