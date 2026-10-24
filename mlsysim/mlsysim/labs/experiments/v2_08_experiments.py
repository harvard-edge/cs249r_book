"""Deterministic fleet-orchestration experiments for Volume II, Chapter 8.

The scenarios in this module are small illustrative fixtures, not traces from a
production scheduler.  Unlike a queueing approximation, every schedule result
is derived from explicit job arrivals, resource bundles, placements, and
completion or preemption events.  Quantities with physical meaning are
calculated with the MLSysIM Pint registry and returned with explicit unit names.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from mlsysim.core.units import Q_, ureg


POLICIES = ("fifo_gang", "backfill", "topology_aware", "priority_preempt")
RESOURCE_KEYS = ("accelerators", "cpu_cores", "memory_gb", "network_gbps")

# These are deliberately generic fleet scenarios. Device-oriented tracks use a
# real fleet-level unit (a shared build/test or gateway pool), never an MCU GPU
# collective analogy.
TRACKS: dict[str, dict[str, Any]] = {
    "tinyml": {
        "display": "TinyML",
        "fleet": "regional firmware build and device-test gateway pool",
        "resource_label": "gateway slots",
        "topology_label": "test rack",
        "provenance": "Illustrative teaching fixture; not a measured production trace.",
        "nodes": 4,
        "accelerators_per_node": 2,
        "link_local_gbps": 20.0,
        "link_remote_gbps": 2.0,
        "warm_capacity_per_replica": 18.0,
        "long_duration_min": 18,
        "checkpoint_interval_min": 4,
        "loading_delay_min": 3,
        "warmup_delay_min": 1,
        "urgent_work": "firmware qualification for a failing device cohort",
        "background_work": "routine model build and device-test sweep",
        "wait_slider": {"min": 5.0, "max": 80.0, "value": 30.0, "step": 5.0},
        "arrival_slider": {"min": 2, "max": 14, "value": 6, "step": 1},
        "checkpoint_slider": {"min": 2, "max": 14, "value": 4, "step": 2},
    },
    "mobile": {
        "display": "Mobile",
        "fleet": "regional mobile build and device-test pool",
        "resource_label": "device-test lanes",
        "topology_label": "lab rack",
        "provenance": "Illustrative teaching fixture; not a measured production trace.",
        "nodes": 4,
        "accelerators_per_node": 2,
        "link_local_gbps": 40.0,
        "link_remote_gbps": 5.0,
        "warm_capacity_per_replica": 30.0,
        "long_duration_min": 28,
        "checkpoint_interval_min": 6,
        "loading_delay_min": 4,
        "warmup_delay_min": 2,
        "urgent_work": "release-candidate device compatibility test",
        "background_work": "nightly application and model test matrix",
        "wait_slider": {"min": 2.0, "max": 50.0, "value": 15.0, "step": 1.0},
        "arrival_slider": {"min": 2, "max": 22, "value": 8, "step": 1},
        "checkpoint_slider": {"min": 2, "max": 16, "value": 6, "step": 2},
    },
    "edge": {
        "display": "Edge",
        "fleet": "site gateway accelerator pool",
        "resource_label": "accelerator slots",
        "topology_label": "site",
        "provenance": "Illustrative teaching fixture; not a measured production trace.",
        "nodes": 4,
        "accelerators_per_node": 4,
        "link_local_gbps": 100.0,
        "link_remote_gbps": 10.0,
        "warm_capacity_per_replica": 45.0,
        "long_duration_min": 38,
        "checkpoint_interval_min": 8,
        "loading_delay_min": 2,
        "warmup_delay_min": 2,
        "urgent_work": "site inference recovery workload",
        "background_work": "regional retraining and validation run",
        "wait_slider": {"min": 1.0, "max": 25.0, "value": 8.0, "step": 1.0},
        "arrival_slider": {"min": 2, "max": 30, "value": 10, "step": 1},
        "checkpoint_slider": {"min": 2, "max": 18, "value": 8, "step": 2},
    },
    "cloud": {
        "display": "Cloud",
        "fleet": "regional accelerator pool",
        "resource_label": "GPU slots",
        "topology_label": "high-bandwidth domain",
        "provenance": "Illustrative teaching fixture; not a measured production trace.",
        "nodes": 4,
        "accelerators_per_node": 8,
        "link_local_gbps": 900.0,
        "link_remote_gbps": 100.0,
        "warm_capacity_per_replica": 60.0,
        "long_duration_min": 50,
        "checkpoint_interval_min": 12,
        "loading_delay_min": 5,
        "warmup_delay_min": 3,
        "urgent_work": "serving repair or interactive debugging job",
        "background_work": "distributed training run",
        "wait_slider": {"min": 0.2, "max": 3.0, "value": 0.8, "step": 0.2},
        "arrival_slider": {"min": 2, "max": 40, "value": 14, "step": 1},
        "checkpoint_slider": {"min": 2, "max": 24, "value": 12, "step": 2},
    },
}


def _finite(value: Any, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _track(track_id: str) -> dict[str, Any]:
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}")
    return TRACKS[track_id]


def _bundle(raw: Mapping[str, Any], name: str) -> dict[str, float]:
    unknown = set(raw) - set(RESOURCE_KEYS)
    if unknown:
        raise ValueError(f"{name} has unknown resources: {', '.join(sorted(unknown))}")
    return {key: _finite(raw.get(key, 0), f"{name}.{key}") for key in RESOURCE_KEYS}


def track_fixture(track_id: str) -> dict[str, Any]:
    """Return a bounded explicit topology and job trace for a teaching track."""
    profile = _track(track_id)
    per_node = profile["accelerators_per_node"]
    nodes = []
    for index in range(profile["nodes"]):
        nodes.append(
            {
                "id": f"n{index}",
                "domain": f"d{index // 2}",
                "capacity": {
                    "accelerators": per_node,
                    "cpu_cores": per_node * 8,
                    "memory_gb": per_node * 64,
                    "network_gbps": profile["link_local_gbps"],
                },
            }
        )
    gang = per_node * 2
    jobs = [
        {
            "id": "long-a",
            "tenant": "research",
            "arrival_min": 0,
            "duration_min": profile["long_duration_min"],
            "priority": 1,
            "same_domain": True,
            "resources": {"accelerators": gang, "cpu_cores": gang * 4, "memory_gb": gang * 24},
            "checkpoint_interval_min": profile["checkpoint_interval_min"],
        },
        {
            "id": "long-b",
            "tenant": "research",
            "arrival_min": 1,
            "duration_min": round(profile["long_duration_min"] * 0.84, 2),
            "priority": 1,
            "same_domain": True,
            "resources": {"accelerators": gang, "cpu_cores": gang * 4, "memory_gb": gang * 24},
            "checkpoint_interval_min": max(2, profile["checkpoint_interval_min"] - 2),
        },
        {
            "id": "short",
            "tenant": "interactive",
            "arrival_min": 2,
            "duration_min": 5,
            "priority": 2,
            "same_domain": False,
            "resources": {"accelerators": gang, "cpu_cores": gang * 4, "memory_gb": gang * 24},
            "checkpoint_interval_min": 2,
        },
        {
            "id": "urgent",
            "tenant": "operations",
            "arrival_min": 8,
            "duration_min": 4,
            "priority": 10,
            "same_domain": False,
            "resources": {"accelerators": per_node, "cpu_cores": per_node * 2, "memory_gb": per_node * 8},
            "checkpoint_interval_min": 2,
        },
    ]
    return {
        **deepcopy(profile),
        "track_id": track_id,
        "nodes": nodes,
        "jobs": jobs,
        "inputs": {"track_id": track_id},
    }


def _validate_nodes(nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not nodes:
        raise ValueError("nodes must not be empty")
    result = []
    seen = set()
    for index, raw in enumerate(nodes):
        node_id = str(raw.get("id", ""))
        if not node_id or node_id in seen:
            raise ValueError("node ids must be nonempty and unique")
        seen.add(node_id)
        result.append(
            {
                "id": node_id,
                "domain": str(raw.get("domain", node_id)),
                "capacity": _bundle(raw.get("capacity", {}), f"nodes[{index}].capacity"),
            }
        )
    return result


def _validate_jobs(jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not jobs:
        raise ValueError("jobs must not be empty")
    result = []
    seen = set()
    for index, raw in enumerate(jobs):
        job_id = str(raw.get("id", ""))
        if not job_id or job_id in seen:
            raise ValueError("job ids must be nonempty and unique")
        seen.add(job_id)
        resources = _bundle(raw.get("resources", {}), f"jobs[{index}].resources")
        if resources["accelerators"] <= 0:
            raise ValueError(f"job {job_id} must request accelerators")
        result.append(
            {
                "id": job_id,
                "tenant": str(raw.get("tenant", "default")),
                "arrival_min": _finite(raw.get("arrival_min"), f"job {job_id} arrival_min"),
                "duration_min": _finite(raw.get("duration_min"), f"job {job_id} duration_min", 1e-12),
                "priority": int(raw.get("priority", 0)),
                "same_domain": bool(raw.get("same_domain", False)),
                "resources": resources,
                "checkpoint_interval_min": _finite(
                    raw.get("checkpoint_interval_min", 10), f"job {job_id} checkpoint_interval_min", 1e-12
                ),
            }
        )
    return sorted(result, key=lambda item: (item["arrival_min"], item["id"]))


def _free_by_node(nodes: Sequence[Mapping[str, Any]], running: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    free = {node["id"]: dict(node["capacity"]) for node in nodes}
    for state in running.values():
        for node_id, bundle in state["placement"].items():
            for resource, amount in bundle.items():
                free[node_id][resource] -= amount
    return free


def _find_placement(
    job: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    free: Mapping[str, Mapping[str, float]],
    topology_aware: bool,
) -> dict[str, dict[str, float]] | None:
    domain_options: list[set[str]]
    domains = {str(node["domain"]) for node in nodes}
    if job["same_domain"] or topology_aware:
        domain_options = [{domain} for domain in sorted(domains)]
    else:
        domain_options = [domains]
    for allowed_domains in domain_options:
        candidates = [node for node in nodes if str(node["domain"]) in allowed_domains]
        candidates.sort(key=lambda node: (free[node["id"]]["accelerators"], node["id"]), reverse=True)
        remaining = dict(job["resources"])
        placement: dict[str, dict[str, float]] = {}
        for node in candidates:
            node_id = node["id"]
            accelerator_take = min(remaining["accelerators"], free[node_id]["accelerators"])
            if accelerator_take <= 0:
                continue
            fraction = accelerator_take / job["resources"]["accelerators"]
            proposal = {"accelerators": accelerator_take}
            feasible = True
            for resource in RESOURCE_KEYS[1:]:
                amount = job["resources"][resource] * fraction
                if amount > free[node_id][resource] + 1e-9:
                    feasible = False
                    break
                proposal[resource] = amount
            if feasible:
                placement[node_id] = proposal
                for resource, amount in proposal.items():
                    remaining[resource] -= amount
            if remaining["accelerators"] <= 1e-9:
                return placement
    return None


def simulate_schedule(
    nodes: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    policy: str = "fifo_gang",
    horizon_min: float = 240,
) -> dict[str, Any]:
    """Run a bounded deterministic gang scheduler over explicit job records.

    ``fifo_gang`` blocks behind an infeasible queue head; ``backfill`` starts a
    later feasible job; ``topology_aware`` additionally restricts every job to
    one named domain; and ``priority_preempt`` can evict lower-priority jobs.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {', '.join(POLICIES)}")
    horizon = _finite(horizon_min, "horizon_min", 1e-12)
    checked_nodes = _validate_nodes(nodes)
    checked_jobs = _validate_jobs(jobs)
    total_capacity = {resource: sum(node["capacity"][resource] for node in checked_nodes) for resource in RESOURCE_KEYS}
    total_accelerators = total_capacity["accelerators"]
    if any(job["resources"][resource] > total_capacity[resource] for job in checked_jobs for resource in RESOURCE_KEYS):
        raise ValueError("a job requests more of a resource than the fleet owns")

    states = {
        job["id"]: {
            **job,
            "status": "future",
            "remaining_min": job["duration_min"],
            "placement": {},
            "start_min": None,
            "finish_min": None,
            "segment_progress_min": 0.0,
            "lost_work_min": 0.0,
            "preemptions": 0,
        }
        for job in checked_jobs
    }
    events: list[dict[str, Any]] = []
    running: dict[str, Any] = {}
    pending: list[str] = []
    time = 0.0
    last_time = 0.0
    allocated_accelerator_minutes = 0.0
    lost_accelerator_minutes = 0.0

    def emit(kind: str, job_id: str, **details: Any) -> None:
        events.append({"time_min": round(time, 6), "kind": kind, "job_id": job_id, **details})

    def advance(to_time: float) -> None:
        nonlocal allocated_accelerator_minutes, last_time
        elapsed = to_time - last_time
        for state in running.values():
            state["remaining_min"] = max(0.0, state["remaining_min"] - elapsed)
            state["segment_progress_min"] += elapsed
            allocated_accelerator_minutes += elapsed * state["resources"]["accelerators"]
        last_time = to_time

    def preempt_for(candidate: Mapping[str, Any]) -> bool:
        nonlocal lost_accelerator_minutes
        victims = sorted(
            (state for state in running.values() if state["priority"] < candidate["priority"]),
            key=lambda state: (state["priority"], -state["resources"]["accelerators"], state["id"]),
        )
        changed = False
        for victim in victims:
            interval = victim["checkpoint_interval_min"]
            lost = victim["segment_progress_min"] % interval
            victim["remaining_min"] += lost
            victim["lost_work_min"] += lost
            lost_accelerator_minutes += lost * victim["resources"]["accelerators"]
            victim["segment_progress_min"] = 0.0
            victim["preemptions"] += 1
            victim["status"] = "pending"
            victim["placement"] = {}
            del running[victim["id"]]
            pending.append(victim["id"])
            emit("preempt", victim["id"], by_job_id=candidate["id"], lost_work_min=round(lost, 6))
            changed = True
            free = _free_by_node(checked_nodes, running)
            if _find_placement(candidate, checked_nodes, free, topology_aware=False):
                return True
        return changed

    while time <= horizon:
        future_times = [state["arrival_min"] for state in states.values() if state["status"] == "future"]
        completion_times = [time + state["remaining_min"] for state in running.values()]
        next_time = min(future_times + completion_times, default=None)
        if next_time is None:
            break
        time = min(next_time, horizon)
        advance(time)

        for job_id in sorted(list(running)):
            state = running[job_id]
            if state["remaining_min"] <= 1e-9:
                state["status"] = "complete"
                state["finish_min"] = time
                state["placement"] = {}
                del running[job_id]
                emit("complete", job_id)
        arrivals = [
            state for state in states.values() if state["status"] == "future" and state["arrival_min"] <= time + 1e-9
        ]
        for state in sorted(arrivals, key=lambda item: item["id"]):
            state["status"] = "pending"
            pending.append(state["id"])
            emit("arrive", state["id"])

        made_progress = True
        while made_progress and pending:
            made_progress = False
            order = (
                sorted(
                    set(pending),
                    key=lambda job_id: (-states[job_id]["priority"], states[job_id]["arrival_min"], job_id),
                )
                if policy == "priority_preempt"
                else list(dict.fromkeys(pending))
            )
            candidates = order if policy in ("backfill", "topology_aware", "priority_preempt") else order[:1]
            for job_id in candidates:
                state = states[job_id]
                free = _free_by_node(checked_nodes, running)
                placement = _find_placement(state, checked_nodes, free, topology_aware=policy == "topology_aware")
                if placement is None and policy == "priority_preempt" and preempt_for(state):
                    free = _free_by_node(checked_nodes, running)
                    placement = _find_placement(state, checked_nodes, free, topology_aware=False)
                if placement is None:
                    aggregate_free = sum(bundle["accelerators"] for bundle in free.values())
                    if aggregate_free >= state["resources"]["accelerators"] and state["same_domain"]:
                        if not any(
                            event["kind"] == "topology_wait"
                            and event["job_id"] == job_id
                            and event["time_min"] == round(time, 6)
                            for event in events
                        ):
                            emit("topology_wait", job_id, aggregate_free=aggregate_free)
                    continue
                state["placement"] = placement
                state["status"] = "running"
                state["start_min"] = time if state["start_min"] is None else state["start_min"]
                running[job_id] = state
                pending.remove(job_id)
                emit("start" if state["preemptions"] == 0 else "resume", job_id, nodes=sorted(placement))
                if job_id != order[0] and policy in ("backfill", "topology_aware"):
                    emit("backfill", job_id, blocked_job_id=order[0])
                made_progress = True
                if policy == "fifo_gang":
                    break
        if time >= horizon:
            break

    advance(horizon)
    useful_accelerator_minutes = allocated_accelerator_minutes - lost_accelerator_minutes
    result_jobs = {}
    for job_id, state in states.items():
        wait = None if state["start_min"] is None else state["start_min"] - state["arrival_min"]
        result_jobs[job_id] = {
            "status": state["status"],
            "start_min": state["start_min"],
            "finish_min": state["finish_min"],
            "wait_min": None if wait is None else round(wait, 6),
            "lost_work_min": round(state["lost_work_min"], 6),
            "preemptions": state["preemptions"],
            "tenant": state["tenant"],
        }
    return {
        "model_id": "v2_08_experiments",
        "inputs": {
            "nodes": deepcopy(list(nodes)),
            "jobs": deepcopy(list(jobs)),
            "policy": policy,
            "horizon_min": horizon,
        },
        "policy": policy,
        "horizon_min": horizon,
        "jobs": result_jobs,
        "events": events,
        "completed_jobs": sum(state["status"] == "complete" for state in states.values()),
        "allocated_accelerator_minutes": round(allocated_accelerator_minutes, 6),
        "lost_accelerator_minutes": round(lost_accelerator_minutes, 6),
        "useful_accelerator_minutes": round(useful_accelerator_minutes, 6),
        "utilization_pct": round(100 * allocated_accelerator_minutes / (total_accelerators * horizon), 6),
        "useful_utilization_pct": round(100 * useful_accelerator_minutes / (total_accelerators * horizon), 6),
        "provenance": "Deterministic simulation of the supplied job and topology records.",
    }


def partial_allocation_deadlock(total_accelerators: int, job_requests: Mapping[str, int]) -> dict[str, Any]:
    """Round-robin partial allocation showing rigid-job hold-and-wait."""
    total = _positive_int(total_accelerators, "total_accelerators")
    if len(job_requests) < 2:
        raise ValueError("at least two jobs are required")
    requests = {
        str(job_id): _positive_int(request, f"request for {job_id}") for job_id, request in job_requests.items()
    }
    held = {job_id: 0 for job_id in sorted(requests)}
    remaining = total
    while remaining:
        changed = False
        for job_id in held:
            if held[job_id] < requests[job_id] and remaining:
                held[job_id] += 1
                remaining -= 1
                changed = True
        if not changed:
            break
    runnable = [job_id for job_id in held if held[job_id] == requests[job_id]]
    blocked = [job_id for job_id in held if held[job_id] < requests[job_id]]
    return {
        "model_id": "v2_08_experiments",
        "inputs": {
            "total_accelerators": total_accelerators,
            "job_requests": deepcopy(dict(job_requests)),
        },
        "held_accelerators": held,
        "runnable_jobs": runnable,
        "blocked_jobs": blocked,
        "deadlocked": bool(blocked) and not runnable and remaining == 0,
        "useful_accelerators": sum(held[job_id] for job_id in runnable),
        "idle_held_accelerators": sum(held[job_id] for job_id in blocked),
        "events": [{"kind": "partial_hold", "job_id": job_id, "accelerators": held[job_id]} for job_id in held],
    }


def topology_crossover(
    wait_min: float, payload_gb: float, iterations: int, local_gbps: float, remote_gbps: float, compute_min: float = 0
) -> dict[str, Any]:
    """Compare waiting for one domain with starting across a slower link now."""
    wait = Q_(_finite(wait_min, "wait_min"), ureg.minute)
    payload = Q_(_finite(payload_gb, "payload_gb"), ureg.GB)
    count = _positive_int(iterations, "iterations")
    local_bw = Q_(_finite(local_gbps, "local_gbps", 1e-12), ureg.Gbps)
    remote_bw = Q_(_finite(remote_gbps, "remote_gbps", 1e-12), ureg.Gbps)
    compute = Q_(_finite(compute_min, "compute_min"), ureg.minute)
    local_comm = (payload / local_bw * count).to(ureg.minute)
    remote_comm = (payload / remote_bw * count).to(ureg.minute)
    wait_local_total = wait + compute + local_comm
    start_remote_total = compute + remote_comm
    crossover_wait = remote_comm - local_comm
    return {
        "model_id": "v2_08_experiments",
        "inputs": {
            "wait_min": wait_min,
            "payload_gb": payload_gb,
            "iterations": iterations,
            "local_gbps": local_gbps,
            "remote_gbps": remote_gbps,
            "compute_min": compute_min,
        },
        "wait_local_total_min": round(wait_local_total.to(ureg.minute).magnitude, 6),
        "start_remote_total_min": round(start_remote_total.to(ureg.minute).magnitude, 6),
        "local_communication_min": round(local_comm.to(ureg.minute).magnitude, 6),
        "remote_communication_min": round(remote_comm.to(ureg.minute).magnitude, 6),
        "crossover_wait_min": round(crossover_wait.to(ureg.minute).magnitude, 6),
        "decision": "wait_for_local" if wait_local_total < start_remote_total else "start_remote",
    }


def dominant_shares(capacity: Mapping[str, Any], allocations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compute per-tenant resource and dominant shares from allocations."""
    fleet = _bundle(capacity, "capacity")
    if any(fleet[key] <= 0 for key in RESOURCE_KEYS):
        raise ValueError("every fleet capacity must be greater than zero")
    result = {}
    for tenant, raw in allocations.items():
        allocation = _bundle(raw, f"allocation for {tenant}")
        shares = {key: allocation[key] / fleet[key] for key in RESOURCE_KEYS}
        result[str(tenant)] = {
            "allocation": allocation,
            "shares": shares,
            "share_pct": {key: value * 100 for key, value in shares.items()},
            "dominant_share": max(shares.values()),
            "dominant_share_pct": max(shares.values()) * 100,
            "dominant_resource": max(shares, key=shares.get),
        }
    values = [item["dominant_share"] for item in result.values()]
    return {
        "model_id": "v2_08_experiments",
        "inputs": {
            "capacity": deepcopy(dict(capacity)),
            "allocations": deepcopy({key: dict(value) for key, value in allocations.items()}),
        },
        "tenants": result,
        "dominant_share_gap": max(values) - min(values) if values else 0.0,
        "dominant_share_gap_pct": (max(values) - min(values)) * 100 if values else 0.0,
    }


def dominant_resource_fairness(
    capacity: Mapping[str, Any], tenant_bundles: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Progressively allocate indivisible bundles by lowest dominant share."""
    fleet = _bundle(capacity, "capacity")
    if any(fleet[key] <= 0 for key in RESOURCE_KEYS):
        raise ValueError("every fleet capacity must be greater than zero")
    bundles = {str(tenant): _bundle(raw, f"bundle for {tenant}") for tenant, raw in tenant_bundles.items()}
    if not bundles or any(all(amount == 0 for amount in bundle.values()) for bundle in bundles.values()):
        raise ValueError("each tenant must request a nonempty bundle")
    allocations = {tenant: {key: 0.0 for key in RESOURCE_KEYS} for tenant in bundles}
    counts = {tenant: 0 for tenant in bundles}
    used = {key: 0.0 for key in RESOURCE_KEYS}
    events = []
    while True:
        shares = dominant_shares(fleet, allocations)["tenants"]
        order = sorted(bundles, key=lambda tenant: (shares[tenant]["dominant_share"], tenant))
        granted = False
        for tenant in order:
            bundle = bundles[tenant]
            if all(used[key] + bundle[key] <= fleet[key] + 1e-9 for key in RESOURCE_KEYS):
                counts[tenant] += 1
                for key in RESOURCE_KEYS:
                    used[key] += bundle[key]
                    allocations[tenant][key] += bundle[key]
                events.append({"kind": "grant", "tenant": tenant, "bundle_number": counts[tenant]})
                granted = True
                break
        if not granted:
            break
    share_result = dominant_shares(fleet, allocations)
    return {
        **share_result,
        "inputs": {
            "capacity": deepcopy(dict(capacity)),
            "tenant_bundles": deepcopy({key: dict(value) for key, value in tenant_bundles.items()}),
        },
        "bundle_counts": counts,
        "used": used,
        "events": events,
    }


def simulate_warm_capacity(
    demand_trace: Sequence[Mapping[str, Any]],
    capacity_per_replica: float,
    warm_reserve: int,
    loading_delay_min: int,
    warmup_delay_min: int,
    replica_cost_per_hour: float,
) -> dict[str, Any]:
    """Replay per-minute demand with explicit loading and warm capacity states."""
    per_replica = _finite(capacity_per_replica, "capacity_per_replica", 1e-12)
    reserve = _positive_int(warm_reserve, "warm_reserve") if warm_reserve else 0
    loading = int(_finite(loading_delay_min, "loading_delay_min"))
    warmup = int(_finite(warmup_delay_min, "warmup_delay_min"))
    cost_rate = Q_(_finite(replica_cost_per_hour, "replica_cost_per_hour"), ureg.USD / ureg.hour)
    if not demand_trace:
        raise ValueError("demand_trace must not be empty")
    points = sorted(
        (int(_finite(point["time_min"], "time_min")), _finite(point["demand_per_min"], "demand_per_min"))
        for point in demand_trace
    )
    if len({time for time, _ in points}) != len(points):
        raise ValueError("demand trace times must be unique")
    end = points[-1][0]
    ready = reserve
    pending_ready: list[int] = []
    timeline = []
    missed = 0.0
    idle_replica_minutes = 0.0
    active_replica_minutes = 0.0
    first_shortfall = None
    cursor = 0
    demand = points[0][1] if points[0][0] == 0 else 0.0
    for minute in range(end + 1):
        while cursor < len(points) and points[cursor][0] <= minute:
            demand = points[cursor][1]
            cursor += 1
        newly_ready = sum(ready_min <= minute for ready_min in pending_ready)
        if newly_ready:
            ready += newly_ready
            pending_ready = [ready_min for ready_min in pending_ready if ready_min > minute]
        needed = math.ceil(demand / per_replica)
        owned = ready + len(pending_ready)
        if needed > owned:
            pending_ready.extend([minute + loading + warmup] * (needed - owned))
        served = min(demand, ready * per_replica)
        shortfall = demand - served
        missed += shortfall
        if shortfall > 0 and first_shortfall is None:
            first_shortfall = minute
        busy = min(ready, math.ceil(served / per_replica) if served else 0)
        idle_replica_minutes += ready - busy
        active_replica_minutes += ready + len(pending_ready)
        timeline.append(
            {
                "time_min": minute,
                "demand_per_min": demand,
                "ready_replicas": ready,
                "loading_replicas": len(pending_ready),
                "served_per_min": served,
                "shortfall_per_min": shortfall,
            }
        )
    cost = (active_replica_minutes * ureg.minute * cost_rate).to(ureg.USD)
    return {
        "model_id": "v2_08_experiments",
        "inputs": {
            "demand_trace": deepcopy(list(demand_trace)),
            "capacity_per_replica": capacity_per_replica,
            "warm_reserve": warm_reserve,
            "loading_delay_min": loading_delay_min,
            "warmup_delay_min": warmup_delay_min,
            "replica_cost_per_hour": replica_cost_per_hour,
        },
        "timeline": timeline,
        "missed_requests": round(missed, 6),
        "idle_replica_minutes": round(idle_replica_minutes, 6),
        "replica_minutes_charged": round(active_replica_minutes, 6),
        "cost_usd": round(cost.magnitude, 6),
        "first_shortfall_min": first_shortfall,
        "readiness_delay_min": loading + warmup,
        "provenance": "Deterministic replay of the supplied demand and capacity-transition records.",
    }


def gang_backfill_comparison(track_id: str) -> dict[str, Any]:
    """Compare partial hold, FIFO gang scheduling, and backfill on one trace."""
    fixture = track_fixture(track_id)
    per_node = fixture["accelerators_per_node"]
    total = len(fixture["nodes"]) * per_node
    head_size = total - per_node
    partial = partial_allocation_deadlock(total, {"rigid-a": head_size, "rigid-b": head_size})
    jobs = [
        {
            "id": "owner",
            "tenant": "incumbent",
            "arrival_min": 0,
            "duration_min": fixture["long_duration_min"],
            "priority": 1,
            "same_domain": False,
            "resources": {
                "accelerators": per_node * 2,
                "cpu_cores": per_node * 4,
                "memory_gb": per_node * 16,
            },
            "checkpoint_interval_min": fixture["checkpoint_interval_min"],
        },
        {
            "id": "large-head",
            "tenant": "training",
            "arrival_min": 1,
            "duration_min": 6,
            "priority": 1,
            "same_domain": False,
            "resources": {
                "accelerators": head_size,
                "cpu_cores": head_size * 2,
                "memory_gb": head_size * 8,
            },
            "checkpoint_interval_min": 3,
        },
        {
            "id": "short",
            "tenant": "interactive",
            "arrival_min": 2,
            "duration_min": 3,
            "priority": 1,
            "same_domain": False,
            "resources": {"accelerators": per_node, "cpu_cores": per_node * 2, "memory_gb": per_node * 8},
            "checkpoint_interval_min": 2,
        },
    ]
    horizon = fixture["long_duration_min"] + 12
    fifo = simulate_schedule(fixture["nodes"], jobs, "fifo_gang", horizon)
    backfill = simulate_schedule(fixture["nodes"], jobs, "backfill", horizon)
    return {
        "model_id": "v2_08_experiments",
        "inputs": {"track_id": track_id},
        "partial": partial,
        "fifo": fifo,
        "backfill": backfill,
        "resource_label": fixture["resource_label"],
    }


def topology_wait_comparison(track_id: str, wait_min: float, payload_gb: float, iterations: int) -> dict[str, Any]:
    """Evaluate a live topology wait against the zero-wait boundary case."""
    fixture = track_fixture(track_id)
    baseline = topology_crossover(
        wait_min=0,
        payload_gb=payload_gb,
        iterations=iterations,
        local_gbps=fixture["link_local_gbps"],
        remote_gbps=fixture["link_remote_gbps"],
        compute_min=fixture["long_duration_min"],
    )
    result = topology_crossover(
        wait_min=wait_min,
        payload_gb=payload_gb,
        iterations=iterations,
        local_gbps=fixture["link_local_gbps"],
        remote_gbps=fixture["link_remote_gbps"],
        compute_min=fixture["long_duration_min"],
    )
    return {
        "model_id": "v2_08_experiments",
        "inputs": {"track_id": track_id, "wait_min": wait_min, "payload_gb": payload_gb, "iterations": iterations},
        "baseline": baseline,
        "result": result,
        "topology_label": fixture["topology_label"],
    }


def topology_wait_curve(
    track_id: str, payload_gb: float, iterations: int, num_points: int = 15
) -> list[dict[str, Any]]:
    """Sample topology wait points covering the track's calibrated placement timescale."""
    fixture = track_fixture(track_id)
    slider = fixture.get("wait_slider", {"min": 1.0, "max": 70.0, "step": 5.0})
    max_val = float(slider["max"])
    step = max_val / (num_points - 1)
    waits = [round(i * step, 4) for i in range(num_points)]
    return [topology_wait_comparison(track_id, w, payload_gb, iterations) for w in waits]


def preemption_comparison(track_id: str, urgent_arrival_min: float, checkpoint_interval_min: float) -> dict[str, Any]:
    """Compare FIFO with priority preemption on one identical fleet trace."""
    fixture = track_fixture(track_id)
    total = len(fixture["nodes"]) * fixture["accelerators_per_node"]
    urgent_slots = fixture["accelerators_per_node"]
    jobs = [
        {
            "id": "background",
            "tenant": "background",
            "arrival_min": 0,
            "duration_min": fixture["long_duration_min"],
            "priority": 1,
            "same_domain": False,
            "resources": {"accelerators": total, "cpu_cores": total * 2, "memory_gb": total * 8},
            "checkpoint_interval_min": checkpoint_interval_min,
        },
        {
            "id": "urgent",
            "tenant": "urgent",
            "arrival_min": urgent_arrival_min,
            "duration_min": 3,
            "priority": 10,
            "same_domain": False,
            "resources": {
                "accelerators": urgent_slots,
                "cpu_cores": urgent_slots * 2,
                "memory_gb": urgent_slots * 8,
            },
            "checkpoint_interval_min": 1,
        },
    ]
    horizon = fixture["long_duration_min"] + checkpoint_interval_min + 10
    return {
        "model_id": "v2_08_experiments",
        "inputs": {
            "track_id": track_id,
            "urgent_arrival_min": urgent_arrival_min,
            "checkpoint_interval_min": checkpoint_interval_min,
        },
        "baseline": simulate_schedule(fixture["nodes"], jobs, "fifo_gang", horizon),
        "result": simulate_schedule(fixture["nodes"], jobs, "priority_preempt", horizon),
        "urgent_work": fixture["urgent_work"],
        "background_work": fixture["background_work"],
    }


def fairness_comparison(track_id: str) -> dict[str, Any]:
    """Compare equal accelerator counts with allocation by dominant share."""
    fixture = track_fixture(track_id)
    total_accelerators = len(fixture["nodes"]) * fixture["accelerators_per_node"]
    capacity = {
        "accelerators": total_accelerators,
        "cpu_cores": total_accelerators * 8,
        "memory_gb": total_accelerators * 64,
        "network_gbps": fixture["link_local_gbps"] * len(fixture["nodes"]),
    }
    half = total_accelerators / 2
    equal_allocations = {
        "compute-heavy": {
            "accelerators": half,
            "cpu_cores": capacity["cpu_cores"] * 0.75,
            "memory_gb": capacity["memory_gb"] * 0.25,
            "network_gbps": capacity["network_gbps"] * 0.25,
        },
        "network-heavy": {
            "accelerators": half,
            "cpu_cores": capacity["cpu_cores"] * 0.125,
            "memory_gb": capacity["memory_gb"] * 0.125,
            "network_gbps": capacity["network_gbps"] * 0.5,
        },
    }
    bundles = {
        "compute-heavy": {
            "accelerators": capacity["accelerators"] / 8,
            "cpu_cores": capacity["cpu_cores"] / 8,
            "memory_gb": capacity["memory_gb"] / 8,
            "network_gbps": capacity["network_gbps"] / 8,
        },
        "network-heavy": {
            "accelerators": capacity["accelerators"] / 4,
            "cpu_cores": capacity["cpu_cores"] / 16,
            "memory_gb": capacity["memory_gb"] / 16,
            "network_gbps": capacity["network_gbps"] / 4,
        },
    }
    return {
        "model_id": "v2_08_experiments",
        "inputs": {"track_id": track_id},
        "baseline": dominant_shares(capacity, equal_allocations),
        "result": dominant_resource_fairness(capacity, bundles),
        "resource_label": fixture["resource_label"],
    }


def warm_capacity_comparison(track_id: str, warm_reserve: int, demand_scale: float) -> dict[str, Any]:
    """Compare cold start with a warm reserve on one demand-ramp trace."""
    fixture = track_fixture(track_id)
    scale = _finite(demand_scale, "demand_scale", 1e-12)
    capacity = fixture["warm_capacity_per_replica"]
    trace = [
        {"time_min": 0, "demand_per_min": capacity * 0.4 * scale},
        {"time_min": 3, "demand_per_min": capacity * 2.8 * scale},
        {"time_min": 10, "demand_per_min": capacity * 0.6 * scale},
        {"time_min": 14, "demand_per_min": capacity * 0.6 * scale},
    ]
    kwargs = {
        "demand_trace": trace,
        "capacity_per_replica": capacity,
        "loading_delay_min": fixture["loading_delay_min"],
        "warmup_delay_min": fixture["warmup_delay_min"],
        "replica_cost_per_hour": 6.0,
    }
    return {
        "model_id": "v2_08_experiments",
        "inputs": {"track_id": track_id, "warm_reserve": warm_reserve, "demand_scale": demand_scale},
        "baseline": simulate_warm_capacity(warm_reserve=0, **kwargs),
        "result": simulate_warm_capacity(warm_reserve=warm_reserve, **kwargs),
        "capacity_unit": "requests/min",
    }


__all__ = [
    "POLICIES",
    "RESOURCE_KEYS",
    "TRACKS",
    "dominant_resource_fairness",
    "dominant_shares",
    "fairness_comparison",
    "gang_backfill_comparison",
    "partial_allocation_deadlock",
    "simulate_schedule",
    "simulate_warm_capacity",
    "topology_crossover",
    "topology_wait_comparison",
    "topology_wait_curve",
    "track_fixture",
    "preemption_comparison",
    "warm_capacity_comparison",
]
