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


from mlsysim.engine.scenarios import Scenarios

_STUDY = Scenarios.PlacementStudy
_DATA = _STUDY.fixtures
STUDY_VERSION = _STUDY.version
TRACKS = _DATA["tracks"]
FILTERING_ALTERNATIVES = _DATA["filters"]


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



def study_profile(track_id: str) -> dict[str, Any]:
    """Detached UI options from the semantic scenario registry."""
    from copy import deepcopy
    return deepcopy({**dict(_track(track_id)), "version": STUDY_VERSION,
                     **{key: _DATA[key] for key in ("resource_cases", "interventions", "duty_options", "battery_multipliers", "horizon_hours")}})


def _study_inputs(kind: str, track_id: str, scenario_version: str, **kwargs):
    _track(track_id)
    if scenario_version != STUDY_VERSION:
        raise ValueError("Unsupported placement scenario_version")
    return dict(study=kind, scenario_version=scenario_version, track_id=track_id, **kwargs)


def locality_study(track_id: str, *, payload_mb: float, connection_mbps: float,
                   ideal_compute: bool = False, scenario_version: str = STUDY_VERSION):
    """Optimistic remote path bound; passing only avoids an early rejection."""
    if not isinstance(ideal_compute, bool):
        raise ValueError("ideal_compute must be Boolean")
    inputs = _study_inputs("locality", track_id, scenario_version, payload_mb=payload_mb,
                           connection_mbps=connection_mbps, ideal_compute=ideal_compute)
    result = placement_accounting(track_id, payload_mb=payload_mb, connection_mbps=connection_mbps)
    execution = 0.0 if ideal_compute else result["remote_execution_ms"]
    lower = (result["request_upload_ms"] + result["propagation_rtt_ms"] +
             execution + result["response_download_ms"])
    focused = {key: result[key] for key in (
        "track_id", "request_upload_ms", "propagation_rtt_ms", "response_download_ms",
        "deadline_ms", "raw_payload_mb", "connection_mbps", "distance_km")}
    return {**focused, "inputs": inputs, "remote_execution_ms": execution,
            "lower_bound_ms": _rounded(lower), "violations": ["deadline_lower_bound"] if lower > result["deadline_ms"] else [],
            "feasible": None if lower <= result["deadline_ms"] else False,
            "readiness": "Unresolved: complete path not validated" if lower <= result["deadline_ms"] else "Rejected by lower bound"}


def resource_study(track_id: str, *, workload_case: str = "ordinary", intervention: str = "baseline",
                   scenario_version: str = STUDY_VERSION):
    """Resource-specific controlled hypothetical; no quality or bandwidth inference."""
    inputs = _study_inputs("resource", track_id, scenario_version, workload_case=workload_case, intervention=intervention)
    if workload_case not in _DATA["resource_cases"] or intervention not in _DATA["interventions"]:
        raise ValueError("Unknown resource case or intervention")
    if track_id == "tinyml" and intervention == "bandwidth":
        raise ValueError("TinyML fixture has no defensible bandwidth intervention")
    if track_id != "tinyml" and intervention == "flash":
        raise ValueError("Flash intervention is only supplied for TinyML")
    track = _track(track_id)
    case = _DATA["resource_cases"][workload_case]
    change = _DATA["interventions"][intervention]
    resident = track["workload_memory_mb"] * case["resident_scale"]
    capacity = track["memory_capacity_mb"] * change["capacity"]
    shares = _DATA["timing_shares"]
    compute = track["local_execution_ms"] * shares["compute"] * case["compute_scale"] / change["compute"]
    movement = track["local_execution_ms"] * shares["movement"] * case["traffic_scale"] / change["bandwidth"]
    overhead = track["local_execution_ms"] * shares["overhead"]
    pools = [{"pool": "SRAM" if track_id == "tinyml" else "Working memory", "required_mb": resident, "capacity_mb": capacity}]
    if track_id == "tinyml":
        pools.append({"pool": "Flash", "required_mb": _DATA["tiny_weights_mb"], "capacity_mb": _DATA["tiny_flash_mb"] * change["flash"]})
    violations = [row["pool"] for row in pools if row["required_mb"] > row["capacity_mb"]]
    latency = compute + movement + overhead
    if latency > track["deadline_ms"]: violations.append("deadline")
    return dict(track_id=track_id, inputs=inputs, pools=pools, compute_ms=_rounded(compute), movement_ms=_rounded(movement),
                overhead_ms=_rounded(overhead), latency_ms=_rounded(latency), deadline_ms=track["deadline_ms"],
                binding_resource=violations[0] if violations else ("movement" if movement > compute else "arithmetic"),
                feasible=not violations, violations=violations, provenance=_DATA["provenance"],
                assumption="Hypothetical one-resource intervention, fixed format and task; no hardware price/performance prediction.")


def operation_study(track_id: str, *, duty_cycle: float, battery_multiplier: float = 1.0,
                    scenario_version: str = STUDY_VERSION):
    """Independent service, mission energy and sustained-power screens."""
    inputs = _study_inputs("operation", track_id, scenario_version, duty_cycle=duty_cycle, battery_multiplier=battery_multiplier)
    _positive(battery_multiplier, "battery_multiplier")
    if battery_multiplier not in _DATA["battery_multipliers"]:
        raise ValueError("Unsupported mission energy allowance")
    result = sustained_operation(track_id, duty_cycle=duty_cycle, horizon_hours=_DATA["horizon_hours"])
    budget = _DATA["mission_energy_allowance_wh"][track_id] * battery_multiplier
    energy_met = result["energy_wh"] <= budget
    violations = list(result["violations"])
    if not energy_met: violations.append("mission_energy")
    return {**result, "inputs": inputs, "mission_energy_budget_wh": _rounded(budget), "mission_energy_met": energy_met,
            "sustained_power_met": result["power_requirement_met"], "feasible": not violations, "violations": violations,
            "assumption": "Endpoint active/idle accounting only; excludes sensing/radio/conversion losses. Power is a supplied steady-operation screen, not a temperature or throttling simulation."}


def placement_study(track_id: str, *, payload_mb: float, connection_mbps: float,
                    workload_case: str, intervention: str, duty_cycle: float,
                    battery_multiplier: float, connectivity_available: bool = True,
                    raw_data_allowed: bool = True, scenario_version: str = STUDY_VERSION):
    """Compose current saved conditions and preserve every gate before partial cost."""
    if not isinstance(raw_data_allowed, bool) or not isinstance(connectivity_available, bool):
        raise ValueError("Policy and connectivity must be Boolean")
    inputs = _study_inputs("placement", track_id, scenario_version, payload_mb=payload_mb,
        connection_mbps=connection_mbps, workload_case=workload_case, intervention=intervention,
        duty_cycle=duty_cycle, battery_multiplier=battery_multiplier,
        connectivity_available=connectivity_available, raw_data_allowed=raw_data_allowed)
    resource = resource_study(track_id, workload_case=workload_case, intervention=intervention)
    operation = operation_study(track_id, duty_cycle=duty_cycle, battery_multiplier=battery_multiplier)
    track = _track(track_id)
    candidates = {}
    ordinary_resources = resource_study(track_id, workload_case=workload_case, intervention="baseline")
    case = _DATA["resource_cases"][workload_case]
    remote_shares = _DATA["remote_timing_shares"]
    remote_compute_ms = track["remote_execution_ms"] * (remote_shares["compute"] * case["compute_scale"] + remote_shares["movement"] * case["traffic_scale"] + remote_shares["overhead"])
    for candidate in ("local", *FILTERING_ALTERNATIVES):
        remote = candidate != "local"
        path = placement_accounting(track_id, payload_mb=payload_mb, connection_mbps=connection_mbps,
                    connectivity_available=connectivity_available, filter_id=candidate if remote else "raw")
        quality = path["retained_information_pct"] if remote else 100.0
        fraction = _DATA["filter_memory_fraction"] if candidate not in ("local", "raw") else (0.0 if remote else 1.0)
        memory_met = all(pool["required_mb"] * fraction <= pool["capacity_mb"] for pool in resource["pools"])
        if remote:
            # The remote worker is a separately supplied larger memory envelope.
            memory_met = memory_met and all(pool["required_mb"] <= pool["capacity_mb"] * _DATA["remote_capacity_scale"] for pool in ordinary_resources["pools"])
        latency = (path["remote_latency_ms"] if remote else resource["latency_ms"])
        if latency is not None:
            latency += _DATA["complete_path_overhead_ms"]
            if remote: latency += remote_compute_ms - track["remote_execution_ms"]
        endpoint_active = track["active_power_w"] * (_DATA["filter_active_fraction"] if candidate not in ("local", "raw") else (_DATA["remote_endpoint_active_fraction"] if remote else 1.0))
        average = endpoint_active * duty_cycle + track["idle_power_w"] * (1-duty_cycle)
        energy = energy_from_power(Q_(average,"watt"), Q_(_DATA["horizon_hours"],"hour")).m_as("watt_hour")
        gates = {
            "task": quality >= track["minimum_retained_information_pct"],
            "privacy": not (candidate == "raw" and not raw_data_allowed),
            "connectivity": not remote or connectivity_available,
            "deadline": latency is not None and latency <= track["deadline_ms"],
            "resource": memory_met,
            "service": operation["service_requirement_met"],
            "sustained_power": average <= track["average_power_budget_w"],
            "mission_energy": energy <= operation["mission_energy_budget_wh"],
            "upload": not remote or path["uploaded_payload_mb"] <= track["upload_limit_mb"],
        }
        violations = [name for name, met in gates.items() if not met]
        costs = _DATA["costs"][candidate]
        candidates[candidate] = dict(design_id=candidate, latency_ms=latency, gates=gates,
            feasible=not violations, violations=violations, retained_information_pct=quality,
            endpoint_average_power_w=_rounded(average), endpoint_energy_wh=_rounded(energy),
            total_energy_evidence="Unknown: sensing, radio, conversion and remote host terms are not supplied",
            readiness="Conditional supplied-screen result; target validation required",
            cost_components_usd=costs, partial_cost_usd=_rounded(sum(costs.values())) if not violations else None,
            cost_status="Partial estimate; lifecycle ranking unresolved" if not violations else "Excluded before cost",
            interface_obligation="Keep feature definitions, preprocessing and model versions synchronized" if candidate not in ("local","raw") else "No filtering boundary",
            inputs={**inputs, "candidate": candidate})
    return dict(track_id=track_id, inputs=inputs, candidates=candidates,
                feasible=any(row["feasible"] for row in candidates.values()), violations=[] if any(row["feasible"] for row in candidates.values()) else ["no_candidate_survives"],
                cost_boundary=_DATA["cost_boundary"], scenario_version=STUDY_VERSION,
                timing_assumption="Synthetic complete-path allowance added to modeled path; supplied study only, no measured tail-latency or deployment readiness claim.",
                quality_assumption=FILTER_EVIDENCE_ASSUMPTION,
                service_assumption="All candidates retain the same observation cadence. Remote options include a smaller local endpoint load and require a separate remote worker; remote host energy is outside endpoint budget.")


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
    if "study" in arguments:
        kind = arguments.pop("study")
        if "scenario_version" not in arguments:
            raise ValueError("New placement studies require explicit scenario_version")
        functions = {"locality": locality_study, "resource": resource_study,
                     "operation": operation_study, "placement": placement_study}
        if not isinstance(kind, str) or kind not in functions:
            raise ValueError("Unknown placement study")
        common = {"track_id", "scenario_version"}
        shapes = {
            "locality": common | {"payload_mb", "connection_mbps", "ideal_compute"},
            "resource": common | {"workload_case", "intervention"},
            "operation": common | {"duty_cycle", "battery_multiplier"},
            "placement": common | {"payload_mb", "connection_mbps", "workload_case", "intervention", "duty_cycle", "battery_multiplier", "connectivity_available", "raw_data_allowed"},
        }
        expected = shapes[kind]
        if set(arguments) != expected and not (kind == "placement" and set(arguments) == expected | {"candidate"}):
            raise ValueError("inputs do not match a saved study shape")
        candidate = arguments.pop("candidate", None)
        if candidate is not None and kind != "placement": raise ValueError("candidate only valid for placement")
        try:
            result = functions[kind](**arguments)
        except TypeError as exc:
            raise ValueError("inputs do not match a saved study shape") from exc
        if candidate is None: return result
        if candidate not in result["candidates"]: raise ValueError("Unknown placement candidate")
        return result["candidates"][candidate]
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
