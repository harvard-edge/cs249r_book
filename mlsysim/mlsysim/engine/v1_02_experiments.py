"""Bounded placement experiments for the deployment-environments lab.

The scenarios in this module are illustrative teaching fixtures.  They are not
measurements of branded devices or production applications.  The model keeps
four mechanisms separate:

* local execution;
* remote request transfer + fiber propagation + fixed remote execution +
  response transfer;
* memory fit and local execution-time feasibility; and
* average duty-cycle power and energy.

Filtering choices are a finite supplied set.  Their retained-information
percentages are illustrative observations for the matched task in each track,
not a formula connecting compression to model quality.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from mlsysim.core.units import Q_
from mlsysim.physics.networking import calc_network_latency_ms
from mlsysim.physics.quantities import energy_from_power, transfer_time


TRACKS: dict[str, dict[str, Any]] = {
    "tinyml": {
        "display": "TinyML",
        "scenario": "A remote agricultural anomaly sensor",
        "payload_mb": 0.8,
        "payload_options_mb": (0.01, 0.8, 1.6),
        "connection_mbps": 2.0,
        "connection_options_mbps": (1.0, 2.0, 5.0),
        "distance_km": 1_000.0,
        "response_kb": 2.0,
        "local_execution_ms": 8.0,
        "remote_execution_ms": 8.0,
        "deadline_ms": 100.0,
        "workload_memory_mb": 0.36,
        "memory_capacity_mb": 0.512,
        "active_power_w": 0.40,
        "idle_power_w": 0.00001,
        "average_power_budget_w": 0.05,
        "baseline_duty_cycle": 0.02,
        "minimum_observation_fraction": 0.01,
        "duty_cycle_step": 0.01,
        "upload_limit_mb": 0.10,
        "minimum_retained_information_pct": 90.0,
    },
    "mobile": {
        "display": "Mobile",
        "scenario": "An interactive personal media assistant",
        "payload_mb": 8.0,
        "payload_options_mb": (0.25, 8.0, 16.0),
        "connection_mbps": 40.0,
        "connection_options_mbps": (20.0, 40.0, 100.0),
        "distance_km": 2_000.0,
        "response_kb": 20.0,
        "local_execution_ms": 24.0,
        "remote_execution_ms": 14.0,
        "deadline_ms": 100.0,
        "workload_memory_mb": 520.0,
        "memory_capacity_mb": 768.0,
        "active_power_w": 5.0,
        "idle_power_w": 0.40,
        "average_power_budget_w": 2.0,
        "baseline_duty_cycle": 0.15,
        "minimum_observation_fraction": 0.10,
        "duty_cycle_step": 0.05,
        "upload_limit_mb": 1.0,
        "minimum_retained_information_pct": 92.0,
    },
    "edge": {
        "display": "Edge",
        "scenario": "A factory visual-inspection station",
        "payload_mb": 12.0,
        "payload_options_mb": (0.25, 12.0, 24.0),
        "connection_mbps": 100.0,
        "connection_options_mbps": (50.0, 100.0, 250.0),
        "distance_km": 300.0,
        "response_kb": 8.0,
        "local_execution_ms": 16.0,
        "remote_execution_ms": 11.0,
        "deadline_ms": 50.0,
        "workload_memory_mb": 4_500.0,
        "memory_capacity_mb": 8_000.0,
        "active_power_w": 25.0,
        "idle_power_w": 4.0,
        "average_power_budget_w": 15.0,
        "baseline_duty_cycle": 0.35,
        "minimum_observation_fraction": 0.25,
        "duty_cycle_step": 0.05,
        "upload_limit_mb": 2.0,
        "minimum_retained_information_pct": 92.0,
    },
    "cloud": {
        "display": "Cloud",
        "scenario": (
            "A regional document-understanding service (regional ingress host "
            "vs. centralized remote accelerator)"
        ),
        "payload_mb": 2.0,
        "payload_options_mb": (0.5, 2.0, 4.0),
        "connection_mbps": 200.0,
        "connection_options_mbps": (50.0, 200.0, 1_000.0),
        "distance_km": 800.0,
        "response_kb": 20.0,
        "local_execution_ms": 80.0,
        "remote_execution_ms": 8.0,
        "deadline_ms": 120.0,
        "workload_memory_mb": 18_000.0,
        "memory_capacity_mb": 24_000.0,
        "active_power_w": 350.0,
        "idle_power_w": 80.0,
        "average_power_budget_w": 250.0,
        "baseline_duty_cycle": 0.50,
        "minimum_observation_fraction": 0.30,
        "duty_cycle_step": 0.05,
        "upload_limit_mb": 1.0,
        "minimum_retained_information_pct": 94.0,
    },
}


FILTERING_ALTERNATIVES: dict[str, dict[str, Any]] = {
    "raw": {
        "display": "Send the raw input",
        "upload_fraction": 1.0,
        "preprocess_ms": 0.0,
        "preprocess_power_w": {
            "tinyml": 0.0,
            "mobile": 0.0,
            "edge": 0.0,
            "cloud": 0.0,
        },
        "retained_information_pct": {
            "tinyml": 100.0,
            "mobile": 100.0,
            "edge": 100.0,
            "cloud": 100.0,
        },
    },
    "event_filter": {
        "display": "Keep detected events",
        "upload_fraction": 0.02,
        "preprocess_ms": 3.0,
        "preprocess_power_w": {
            "tinyml": 0.20,
            "mobile": 2.0,
            "edge": 8.0,
            "cloud": 100.0,
        },
        "retained_information_pct": {
            "tinyml": 94.0,
            "mobile": 95.0,
            "edge": 96.0,
            "cloud": 97.0,
        },
    },
    "feature_summary": {
        "display": "Send a compact feature summary",
        "upload_fraction": 0.005,
        "preprocess_ms": 8.0,
        "preprocess_power_w": {
            "tinyml": 0.35,
            "mobile": 3.5,
            "edge": 15.0,
            "cloud": 180.0,
        },
        "retained_information_pct": {
            "tinyml": 86.0,
            "mobile": 88.0,
            "edge": 89.0,
            "cloud": 91.0,
        },
    },
}


SCENARIO_ASSUMPTION = (
    "Illustrative teaching fixture; physical quantities are scenario inputs, "
    "not measurements of a named product or deployment."
)
FILTER_EVIDENCE_ASSUMPTION = (
    "Supplied illustrative matched-task observations; retained information is "
    "not inferred from payload reduction."
)
MODEL_KEY = "v1_02_experiments"


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be a finite number at least {minimum}")
    return result


def _positive(value: Any, name: str) -> float:
    result = _number(value, name)
    if result == 0:
        raise ValueError(f"{name} must be greater than 0")
    return result


def _track(track_id: str) -> Mapping[str, Any]:
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}")
    return TRACKS[track_id]


def _filter(filter_id: str) -> Mapping[str, Any]:
    if filter_id not in FILTERING_ALTERNATIVES:
        choices = ", ".join(FILTERING_ALTERNATIVES)
        raise ValueError(f"filter_id must be one of {choices}")
    return FILTERING_ALTERNATIVES[filter_id]


def _rounded(value: float) -> float:
    return round(value, 6)


def placement_accounting(
    track_id: str,
    *,
    payload_mb: float | None = None,
    connection_mbps: float | None = None,
    distance_km: float | None = None,
    connectivity_available: bool = True,
    filter_id: str = "raw",
) -> dict[str, Any]:
    """Evaluate local and remote latency for one matched request.

    Remote latency is request upload + fiber round-trip propagation + fixed
    remote execution + response download.  The selected connection rate is
    assumed symmetric.  Filtering changes only request bytes and adds explicit
    local preprocessing; remote compute and response size remain fixed.
    """
    track = _track(track_id)
    filter_profile = _filter(filter_id)
    if not isinstance(connectivity_available, bool):
        raise ValueError("connectivity_available must be a Boolean")

    raw_payload_mb = _positive(
        track["payload_mb"] if payload_mb is None else payload_mb,
        "payload_mb",
    )
    link_mbps = _positive(
        track["connection_mbps"] if connection_mbps is None else connection_mbps,
        "connection_mbps",
    )
    link_distance_km = _number(
        track["distance_km"] if distance_km is None else distance_km,
        "distance_km",
    )

    upload_fraction = filter_profile["upload_fraction"]
    uploaded_mb = raw_payload_mb * upload_fraction
    bandwidth = Q_(link_mbps, "megabit / second")
    upload_ms = transfer_time(Q_(uploaded_mb, "megabyte"), bandwidth).m_as("millisecond")
    response_ms = transfer_time(
        Q_(track["response_kb"], "kilobyte"), bandwidth
    ).m_as("millisecond")
    propagation_ms = calc_network_latency_ms(Q_(link_distance_km, "kilometer"))
    preprocess_ms = filter_profile["preprocess_ms"]
    remote_execution_ms = track["remote_execution_ms"]
    calculated_remote_ms = (
        preprocess_ms
        + upload_ms
        + propagation_ms
        + remote_execution_ms
        + response_ms
    )
    remote_latency_ms = calculated_remote_ms if connectivity_available else None

    deadline_ms = track["deadline_ms"]
    response_and_fixed_ms = (
        preprocess_ms + propagation_ms + remote_execution_ms + response_ms
    )
    available_upload_ms = deadline_ms - response_and_fixed_ms
    if connectivity_available and available_upload_ms >= 0:
        boundary_payload_mb = (
            bandwidth * Q_(available_upload_ms, "millisecond") / upload_fraction
        ).to("megabyte").magnitude
    else:
        boundary_payload_mb = None

    retained_information_pct = filter_profile["retained_information_pct"][track_id]
    upload_limit_mb = track["upload_limit_mb"]
    remote_violations: list[str] = []
    if not connectivity_available:
        remote_violations.append("connectivity")
    elif calculated_remote_ms > deadline_ms:
        remote_violations.append("deadline")
    if uploaded_mb > upload_limit_mb:
        remote_violations.append("upload")
    if retained_information_pct < track["minimum_retained_information_pct"]:
        remote_violations.append("retained_information")

    preprocess_power_w = filter_profile["preprocess_power_w"][track_id]
    preprocess_energy_mj = energy_from_power(
        Q_(preprocess_power_w, "watt"),
        Q_(preprocess_ms, "millisecond"),
    ).m_as("millijoule")

    return {
        "track_id": track_id,
        "filter_id": filter_id,
        "connectivity_available": connectivity_available,
        "local_latency_ms": track["local_execution_ms"],
        "remote_latency_ms": (
            None if remote_latency_ms is None else _rounded(remote_latency_ms)
        ),
        "deadline_ms": deadline_ms,
        "local_deadline_met": track["local_execution_ms"] <= deadline_ms,
        "remote_feasible": not remote_violations,
        "remote_violations": remote_violations,
        "feasible": not remote_violations,
        "violations": remote_violations,
        "raw_payload_mb": raw_payload_mb,
        "uploaded_payload_mb": _rounded(uploaded_mb),
        "connection_mbps": link_mbps,
        "distance_km": link_distance_km,
        "request_upload_ms": _rounded(upload_ms),
        "propagation_rtt_ms": _rounded(propagation_ms),
        "remote_execution_ms": remote_execution_ms,
        "response_download_ms": _rounded(response_ms),
        "filter_preprocess_ms": preprocess_ms,
        "filter_preprocess_power_w": preprocess_power_w,
        "filter_preprocess_energy_mj": _rounded(preprocess_energy_mj),
        "retained_information_pct": retained_information_pct,
        "minimum_retained_information_pct": track[
            "minimum_retained_information_pct"
        ],
        "upload_limit_mb": upload_limit_mb,
        "deadline_boundary_payload_mb": (
            None if boundary_payload_mb is None else _rounded(boundary_payload_mb)
        ),
        "link_assumption": "Selected connection rate applies symmetrically to upload and response download.",
        "scenario_assumption": SCENARIO_ASSUMPTION,
        "filter_evidence_assumption": FILTER_EVIDENCE_ASSUMPTION,
        "inputs": {
            "track_id": track_id,
            "payload_mb": raw_payload_mb,
            "connection_mbps": link_mbps,
            "distance_km": link_distance_km,
            "connectivity_available": connectivity_available,
            "filter_id": filter_id,
        },
    }


def workload_feasibility(
    track_id: str,
    *,
    memory_scale: float = 1.0,
    execution_scale: float = 1.0,
) -> dict[str, Any]:
    """Check memory fit and local execution time as separate hard constraints."""
    track = _track(track_id)
    memory_scale = _positive(memory_scale, "memory_scale")
    execution_scale = _positive(execution_scale, "execution_scale")
    required_memory_mb = track["workload_memory_mb"] * memory_scale
    execution_ms = track["local_execution_ms"] * execution_scale
    memory_fits = required_memory_mb <= track["memory_capacity_mb"]
    execution_fits = execution_ms <= track["deadline_ms"]
    violations: list[str] = []
    if not memory_fits:
        violations.append("memory")
    if not execution_fits:
        violations.append("execution_time")
    return {
        "track_id": track_id,
        "required_memory_mb": _rounded(required_memory_mb),
        "memory_capacity_mb": track["memory_capacity_mb"],
        "memory_fits": memory_fits,
        "memory_utilization_pct": _rounded(
            100.0 * required_memory_mb / track["memory_capacity_mb"]
        ),
        "execution_ms": _rounded(execution_ms),
        "deadline_ms": track["deadline_ms"],
        "execution_fits": execution_fits,
        "execution_budget_utilization_pct": _rounded(
            100.0 * execution_ms / track["deadline_ms"]
        ),
        "feasible": not violations,
        "violations": violations,
        "scenario_assumption": SCENARIO_ASSUMPTION,
        "inputs": {
            "track_id": track_id,
            "memory_scale": memory_scale,
            "execution_scale": execution_scale,
        },
    }


def sustained_operation(
    track_id: str,
    *,
    duty_cycle: float | None = None,
    horizon_hours: float = 24.0,
) -> dict[str, Any]:
    """Compute average power and energy for an explicit active duty cycle."""
    track = _track(track_id)
    selected_duty_cycle = _number(
        track["baseline_duty_cycle"] if duty_cycle is None else duty_cycle,
        "duty_cycle",
    )
    if selected_duty_cycle > 1.0:
        raise ValueError("duty_cycle must be at most 1")
    horizon_hours = _positive(horizon_hours, "horizon_hours")
    average_power_w = (
        track["active_power_w"] * selected_duty_cycle
        + track["idle_power_w"] * (1.0 - selected_duty_cycle)
    )
    energy_wh = energy_from_power(
        Q_(average_power_w, "watt"), Q_(horizon_hours, "hour")
    ).m_as("watt_hour")
    budget_w = track["average_power_budget_w"]
    minimum_observation_fraction = track["minimum_observation_fraction"]
    active_observation_hours = horizon_hours * selected_duty_cycle
    minimum_observation_hours = horizon_hours * minimum_observation_fraction
    power_requirement_met = average_power_w <= budget_w
    service_requirement_met = (
        selected_duty_cycle >= minimum_observation_fraction
    )
    violations: list[str] = []
    if not power_requirement_met:
        violations.append("average_power")
    if not service_requirement_met:
        violations.append("observation_time")
    return {
        "track_id": track_id,
        "duty_cycle": selected_duty_cycle,
        "active_power_w": track["active_power_w"],
        "idle_power_w": track["idle_power_w"],
        "average_power_w": _rounded(average_power_w),
        "average_power_budget_w": budget_w,
        "horizon_hours": horizon_hours,
        "energy_wh": _rounded(energy_wh),
        "active_observation_hours": _rounded(active_observation_hours),
        "minimum_observation_fraction": minimum_observation_fraction,
        "minimum_observation_hours": _rounded(minimum_observation_hours),
        "power_requirement_met": power_requirement_met,
        "service_requirement_met": service_requirement_met,
        "sustained_feasible": not violations,
        "violations": violations,
        "service_requirement_assumption": (
            "Minimum active-observation fraction is a supplied mission "
            "requirement, not a physical law."
        ),
        "scenario_assumption": SCENARIO_ASSUMPTION,
        "inputs": {
            "track_id": track_id,
            "duty_cycle": selected_duty_cycle,
            "horizon_hours": horizon_hours,
        },
    }


def _local_placement_summary(track_id: str) -> dict[str, Any]:
    track = _track(track_id)
    fit = workload_feasibility(track_id)
    violations = list(fit["violations"])
    return {
        "design_id": "local",
        "latency_ms": track["local_execution_ms"],
        "uploaded_payload_mb": 0.0,
        "retained_information_pct": 100.0,
        "feasible": not violations,
        "violations": violations,
        "inputs": {"track_id": track_id, "placement": "local"},
    }


def _remote_placement_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "design_id": f"remote_{result['filter_id']}",
        "latency_ms": result["remote_latency_ms"],
        "uploaded_payload_mb": result["uploaded_payload_mb"],
        "retained_information_pct": result["retained_information_pct"],
        "feasible": result["remote_feasible"],
        "violations": result["remote_violations"],
        "inputs": {**result["inputs"], "placement": "remote"},
        "breakdown": dict(result),
    }


def compare_placements(
    track_id: str,
    *,
    payload_mb: float | None = None,
    connection_mbps: float | None = None,
    distance_km: float | None = None,
    connectivity_available: bool = True,
) -> dict[str, Any]:
    """Return the local design and every supplied remote filtering design."""
    local = _local_placement_summary(track_id)
    remote: dict[str, dict[str, Any]] = {}
    for filter_id in FILTERING_ALTERNATIVES:
        result = placement_accounting(
            track_id,
            payload_mb=payload_mb,
            connection_mbps=connection_mbps,
            distance_km=distance_km,
            connectivity_available=connectivity_available,
            filter_id=filter_id,
        )
        remote[filter_id] = _remote_placement_summary(result)
    return {
        "track_id": track_id,
        "local": local,
        "remote": remote,
        "scenario_assumption": SCENARIO_ASSUMPTION,
        "filter_evidence_assumption": FILTER_EVIDENCE_ASSUMPTION,
    }


def replay(model_key: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one saved Chapter 2 result from its JSON input mapping.

    Dispatch is deliberately closed: only the model key and exact input shapes
    emitted by this module are accepted.  This avoids evaluating arbitrary
    functions or treating presentation metadata as simulator arguments.
    """
    if model_key != MODEL_KEY:
        raise ValueError(f"model_key must be {MODEL_KEY!r}")
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a mapping")
    arguments = dict(inputs)
    keys = set(arguments)

    placement_keys = {
        "track_id",
        "payload_mb",
        "connection_mbps",
        "distance_km",
        "connectivity_available",
        "filter_id",
    }
    if keys == placement_keys:
        return placement_accounting(**arguments)
    if keys == placement_keys | {"placement"}:
        placement = arguments.pop("placement")
        if placement != "remote":
            raise ValueError("placement must be 'remote' for remote placement inputs")
        return _remote_placement_summary(placement_accounting(**arguments))
    if keys == {"track_id", "placement"}:
        if arguments["placement"] != "local":
            raise ValueError("placement must be 'local' for local placement inputs")
        return _local_placement_summary(arguments["track_id"])
    if keys == {"track_id", "memory_scale", "execution_scale"}:
        return workload_feasibility(**arguments)
    if keys == {"track_id", "duty_cycle", "horizon_hours"}:
        return sustained_operation(**arguments)
    raise ValueError("inputs do not match a saved Chapter 2 evaluation shape")
