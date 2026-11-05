"""Fault-tolerance experiments for the Volume II fleet lab.

The analytical experiments keep three reliability questions distinct:

* a coupled training job is interrupted by any worker or failure-domain event;
* a serving fleet loses capacity when independent devices fail; and
* replicated service availability depends on where replicas share failure domains.

Exact synthetic fault, progress, checkpoint-shard and resize traces additionally
test necessary correctness conditions. Their injected events and time units are
scenario assumptions, not measured failure or detection rates.

Track values are explicitly illustrative scenario assumptions.  Reusable component
and recovery values come from :mod:`mlsysim.systems.reliability`.
"""

from __future__ import annotations

import math
import json
import zlib
from collections.abc import Mapping, Sequence
from copy import deepcopy

from mlsysim.core.units import Q_, ureg
from mlsysim.physics.reliability import calc_failure_probability, calc_young_daly_interval
from mlsysim.systems.reliability import Reliability


# These packets describe four teaching contexts, rather than branded hardware.
# Each device-oriented track uses an upstream training pool for checkpoint work and
# an independent deployed fleet for the separate capacity-loss experiment.
TRACKS = {
    "tinyml": {
        "label": "TinyML sensing fleet",
        "training_unit": "backend training accelerators",
        "training_backend_role": "Central training cluster checkpointing sensor-model generations; MCUs run inference only and do not write training checkpoints",
        "manifest_role": "Central training cluster publishing sensor-model training checkpoint generations",
        "manifest_context": "Upstream backend servers checkpoint training state to restore interrupted runs; low-power microcontrollers run inference only and do not consume training checkpoint manifests.",
        "serving_unit": "sensors",
        "replica_role": "ingestion gateway",
        "recovery_assumption": "automated sensor restart or reconnection",
        "training_workers": 32,
        "job_duration_h": 10.0,
        "checkpoint_state_gb": 12.0,
        "write_bandwidth_gbs": 0.20,
        "checkpoint_interval_min": 60.0,
        "rpo_target_min": 90.0,
        "rto_target_s": 900.0,
        "failure_domains": 4,
        "domain_mttf_h": 80_000.0,
        "serving_devices": 100_000,
        "serving_component_mttf_h": 30_000.0,
        "serving_repair_min": 30.0,
        "serving_domains": 100,
        "serving_domain_mttf_h": 120_000.0,
        "serving_domain_repair_min": 45.0,
    },
    "mobile": {
        "label": "Mobile rollout fleet",
        "training_unit": "backend training accelerators",
        "training_backend_role": "Backend cluster checkpointing mobile model generations; phone endpoints run on-device inference without local training checkpoints",
        "manifest_role": "Backend cluster publishing mobile model training checkpoint generations",
        "manifest_context": "Dedicated backend infrastructure checkpoints training state to restore interrupted runs; mobile devices run on-device inference and do not consume training checkpoint manifests.",
        "serving_unit": "phone clients",
        "replica_role": "ingestion service",
        "recovery_assumption": "automated phone client restart or reconnection",
        "training_workers": 128,
        "job_duration_h": 24.0,
        "checkpoint_state_gb": 80.0,
        "write_bandwidth_gbs": 1.0,
        "checkpoint_interval_min": 45.0,
        "rpo_target_min": 60.0,
        "rto_target_s": 600.0,
        "failure_domains": 8,
        "domain_mttf_h": 60_000.0,
        "serving_devices": 250_000,
        "serving_component_mttf_h": 20_000.0,
        "serving_repair_min": 12.0,
        "serving_domains": 50,
        "serving_domain_mttf_h": 90_000.0,
        "serving_domain_repair_min": 30.0,
    },
    "edge": {
        "label": "Regional edge learning pool",
        "training_unit": "regional training accelerators",
        "training_backend_role": "Regional edge cluster checkpointing localized model generations across edge aggregation nodes",
        "manifest_role": "Regional edge cluster publishing localized training checkpoint generations",
        "manifest_context": "Regional servers coordinate and checkpoint partition training state to restore interrupted runs across regional nodes.",
        "serving_unit": "regional serving nodes",
        "replica_role": "gateway service",
        "recovery_assumption": "automated regional node restart or reconnection",
        "training_workers": 256,
        "job_duration_h": 36.0,
        "checkpoint_state_gb": 480.0,
        "write_bandwidth_gbs": 8.0,
        "checkpoint_interval_min": 30.0,
        "rpo_target_min": 45.0,
        "rto_target_s": 480.0,
        "failure_domains": 16,
        "domain_mttf_h": 40_000.0,
        "serving_devices": 5_000,
        "serving_component_mttf_h": 8_000.0,
        "serving_repair_min": 3.0,
        "serving_domains": 25,
        "serving_domain_mttf_h": 50_000.0,
        "serving_domain_repair_min": 8.0,
    },
    "cloud": {
        "label": "Cloud training fleet",
        "training_unit": "accelerators",
        "training_backend_role": "Datacenter accelerator cluster checkpointing synchronized distributed training generations",
        "manifest_role": "Datacenter training cluster publishing distributed checkpoint generations",
        "manifest_context": "Coupled cloud accelerator ranks checkpoint synchronized training state to shared storage.",
        "serving_unit": "independent inference workers",
        "replica_role": "front-end service",
        "recovery_assumption": "automated inference worker restart or reconnection",
        "training_workers": 2_048,
        "job_duration_h": 72.0,
        "checkpoint_state_gb": 1_400.0,
        "write_bandwidth_gbs": float(Reliability.Recovery.checkpoint_write_bw_gbs),
        "checkpoint_interval_min": 20.0,
        "rpo_target_min": 30.0,
        "rto_target_s": 600.0,
        "failure_domains": 256,
        "domain_mttf_h": 30_000.0,
        "serving_devices": 4_096,
        "serving_component_mttf_h": float(Reliability.Gpu.mttf_hours),
        "serving_repair_min": 10.0,
        "serving_domains": 32,
        "serving_domain_mttf_h": 50_000.0,
        "serving_domain_repair_min": 15.0,
    },
}


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def get_track_scenario(track_id: str) -> dict:
    """Return a copy of one illustrative track packet with physical scenario policy values."""
    if track_id not in TRACKS:
        raise ValueError(f"unknown track_id {track_id!r}; choose one of {tuple(TRACKS)}")
    scenario = deepcopy(TRACKS[track_id])
    workers = scenario["training_workers"]
    interval = scenario["checkpoint_interval_min"]
    scenario["worker_min"] = max(2, workers // 8)
    scenario["worker_max"] = workers * 2
    scenario["worker_step"] = max(1, workers // 16)
    scenario["worker_default"] = workers
    scenario["worker_sweep_baseline"] = max(1, workers // 8)
    scenario["checkpoint_interval_default"] = min(180, interval * 2)
    return scenario


def track_scenario(track_id: str) -> dict:
    """Return a copy of one illustrative track packet."""
    return get_track_scenario(track_id)


def training_exposure(
    track_id: str,
    *,
    workers: int | None = None,
    duration_h: float | None = None,
    component_mttf_h: float | None = None,
    failure_domains: int | None = None,
    domain_mttf_h: float | None = None,
) -> dict:
    """Calculate interruption exposure for one coupled training job.

    Independent worker hazards and correlated failure-domain events are explicit
    additive event streams.  One event from either stream interrupts the job.
    """
    scenario = track_scenario(track_id)
    n_workers = _integer(workers if workers is not None else scenario["training_workers"], "workers")
    duration = _positive(duration_h if duration_h is not None else scenario["job_duration_h"], "duration_h")
    component_mttf = _positive(
        component_mttf_h if component_mttf_h is not None else float(Reliability.Gpu.mttf_hours),
        "component_mttf_h",
    )
    n_domains = _integer(
        failure_domains if failure_domains is not None else scenario["failure_domains"],
        "failure_domains",
    )
    domain_mttf = _positive(
        domain_mttf_h if domain_mttf_h is not None else scenario["domain_mttf_h"],
        "domain_mttf_h",
    )
    worker_rate_per_h = n_workers / component_mttf
    domain_rate_per_h = n_domains / domain_mttf
    interruption_rate_per_h = worker_rate_per_h + domain_rate_per_h
    system_mtbf_h = 1.0 / interruption_rate_per_h
    expected_interruptions = interruption_rate_per_h * duration

    return {
        "track_id": track_id,
        "inputs": {
            "track_id": track_id,
            "workers": n_workers,
            "duration_h": duration,
            "component_mttf_h": component_mttf,
            "failure_domains": n_domains,
            "domain_mttf_h": domain_mttf,
        },
        "semantics": "coupled_training_job",
        "workers": n_workers,
        "failure_domains": n_domains,
        "duration_h": duration,
        "worker_event_rate_per_h": worker_rate_per_h,
        "domain_event_rate_per_h": domain_rate_per_h,
        "interruption_rate_per_h": interruption_rate_per_h,
        "system_mtbf_h": system_mtbf_h,
        "expected_interruptions": expected_interruptions,
        "interruption_probability": calc_failure_probability(
            Q_(system_mtbf_h, "hour"), Q_(duration, "hour")
        ),
    }


def checkpoint_tradeoff(
    track_id: str,
    *,
    interval_min: float | None = None,
    state_gb: float | None = None,
    state_scale: float = 1.0,
    write_bandwidth_gbs: float | None = None,
    bandwidth_scale: float = 1.0,
    bandwidth_allocation: str = "both",
    checkpoint_pause_s: float | None = None,
    detection_s: float | None = None,
    restart_s: float | None = None,
    load_s: float | None = None,
    warmup_s: float | None = None,
    requested_rpo_min: float | None = None,
    requested_rto_s: float | None = None,
) -> dict:
    """Evaluate checkpoint write, rework, and recovery losses separately.

    Fractions use a steady-state wall-clock basis.  The checkpoint pause is
    charged once per interval, expected rework is half an interval per
    interruption, and the four recovery phases are charged once per
    interruption.  ``requested_rpo_min`` is the maximum acceptable rollback
    window; it is compared with the full checkpoint interval, not the expected
    half-interval loss.  The Young optimum is the first-order approximation and
    assumes a checkpoint pause much shorter than system MTBF.
    """
    scenario = track_scenario(track_id)
    interval = _positive(
        interval_min if interval_min is not None else scenario["checkpoint_interval_min"],
        "interval_min",
    )
    scale = _positive(state_scale, "state_scale")
    base_state = _positive(state_gb if state_gb is not None else scenario["checkpoint_state_gb"], "state_gb")
    state = base_state * scale
    base_bandwidth = _positive(
        write_bandwidth_gbs if write_bandwidth_gbs is not None else scenario["write_bandwidth_gbs"],
        "write_bandwidth_gbs",
    )
    bandwidth_multiplier = _positive(bandwidth_scale, "bandwidth_scale")
    if bandwidth_allocation not in {"write", "restore", "both"}:
        raise ValueError("bandwidth_allocation must be 'write', 'restore', or 'both'")
    write_bandwidth = base_bandwidth * (
        bandwidth_multiplier if bandwidth_allocation in {"write", "both"} else 1.0
    )
    load_bandwidth = base_bandwidth * (
        bandwidth_multiplier if bandwidth_allocation in {"restore", "both"} else 1.0
    )
    detect = _positive(
        detection_s if detection_s is not None else float(Reliability.Recovery.detection_time_s),
        "detection_s",
    )
    restart = _positive(
        restart_s if restart_s is not None else float(Reliability.Recovery.restart_time_s),
        "restart_s",
    )
    load = _positive(load_s if load_s is not None else state / load_bandwidth, "load_s")
    warmup = _positive(
        warmup_s if warmup_s is not None else float(Reliability.Recovery.warmup_time_s),
        "warmup_s",
    )

    exposure = training_exposure(track_id)
    mtbf_s = Q_(exposure["system_mtbf_h"], "hour").to("second")
    write_s = Q_(state / write_bandwidth, "second")
    pause = _positive(
        checkpoint_pause_s if checkpoint_pause_s is not None else write_s.m_as("second"),
        "checkpoint_pause_s",
    )
    pause_s = Q_(pause, "second")
    interval_s = Q_(interval, "minute").to("second")
    rpo_target = _positive(
        requested_rpo_min if requested_rpo_min is not None else scenario["rpo_target_min"],
        "requested_rpo_min",
    )
    rto_target = _positive(
        requested_rto_s if requested_rto_s is not None else scenario["rto_target_s"],
        "requested_rto_s",
    )

    write_tax = (pause_s / interval_s).to(ureg.dimensionless).magnitude
    rework_tax = (interval_s / (2 * mtbf_s)).to(ureg.dimensionless).magnitude
    recovery_s = Q_(detect + restart + load + warmup, "second")
    recovery_tax = (recovery_s / mtbf_s).to(ureg.dimensionless).magnitude
    total_tax = write_tax + rework_tax + recovery_tax
    pause_to_mtbf = (pause_s / mtbf_s).to(ureg.dimensionless).magnitude
    interval_to_mtbf = (interval_s / mtbf_s).to(ureg.dimensionless).magnitude
    recovery_to_mtbf = (recovery_s / mtbf_s).to(ureg.dimensionless).magnitude
    young_approximation_valid = pause_to_mtbf <= 0.10
    optimal = calc_young_daly_interval(pause_s, mtbf_s) if young_approximation_valid else None
    validity_issues = []
    if pause >= interval_s.m_as("second"):
        validity_issues.append("checkpoint pause is at least the checkpoint interval")
    if interval_to_mtbf > 0.10:
        validity_issues.append("checkpoint interval exceeds 10% of system MTBF")
    if recovery_to_mtbf > 0.10:
        validity_issues.append("recovery time exceeds 10% of system MTBF")
    if not young_approximation_valid:
        validity_issues.append("checkpoint pause exceeds 10% of system MTBF")
    if total_tax >= 1.0:
        validity_issues.append("first-order loss terms sum to at least one")
    loss_approximation_valid = not validity_issues

    return {
        "track_id": track_id,
        "inputs": {
            "track_id": track_id,
            "interval_min": interval,
            "state_gb": state_gb,
            "state_scale": scale,
            "write_bandwidth_gbs": base_bandwidth,
            "bandwidth_scale": bandwidth_multiplier,
            "bandwidth_allocation": bandwidth_allocation,
            "checkpoint_pause_s": pause,
            "detection_s": detect,
            "restart_s": restart,
            "load_s": load,
            "warmup_s": warmup,
            "requested_rpo_min": rpo_target,
            "requested_rto_s": rto_target,
        },
        "interval_min": interval,
        "effective_state_gb": state,
        "effective_write_bandwidth_gbs": write_bandwidth,
        "effective_load_bandwidth_gbs": load_bandwidth,
        "additional_bandwidth_gbs": base_bandwidth * (bandwidth_multiplier - 1.0),
        "checkpoint_write_s": write_s.m_as("second"),
        "checkpoint_pause_s": pause,
        "optimal_interval_min": optimal.m_as("minute") if optimal is not None else None,
        "expected_lost_work_min": interval / 2.0,
        "detection_s": detect,
        "restart_s": restart,
        "load_s": load,
        "warmup_s": warmup,
        "recovery_s": recovery_s.m_as("second"),
        "requested_rpo_min": rpo_target,
        "requested_rto_s": rto_target,
        "rpo_ok": interval <= rpo_target,
        "rto_ok": recovery_s.m_as("second") <= rto_target,
        "write_tax": write_tax,
        "rework_tax": rework_tax,
        "recovery_tax": recovery_tax,
        "total_waste_fraction": total_tax,
        "goodput_fraction": 1.0 - total_tax if loss_approximation_valid else None,
        "dominant_interval_cost": "checkpoint_writes" if write_tax > rework_tax else "lost_work",
        "time_basis": "steady-state wall-clock fraction",
        "rpo_definition": "maximum rollback window equals checkpoint interval",
        "young_scope": "Young first-order approximation; checkpoint pause must be much shorter than system MTBF",
        "young_approximation_valid": young_approximation_valid,
        "loss_approximation_valid": loss_approximation_valid,
        "validity_issues": tuple(validity_issues),
    }


def select_restore_checkpoint(
    checkpoints: Sequence[Mapping[str, object]], *, failure_step: int
) -> dict:
    """Select a committed, digest-valid consistent generation before a failure.

    Explicit shard fixtures must contain every expected shard at the manifest's
    step and a published manifest. Legacy scalar records retain their declared
    completed/digest semantics; they do not establish a distributed cut.
    """
    failed_at = _integer(failure_step, "failure_step", minimum=0)
    normalized = []
    for record in checkpoints:
        if "step" not in record:
            raise ValueError("every checkpoint record must include step")
        step = _integer(record["step"], "checkpoint step", minimum=0)
        item = {
            "step": step,
            "completed": bool(record.get("completed", False)),
            "digest_valid": bool(record.get("digest_valid", False)),
        }
        if "shard_steps" in record:
            item["shard_steps"] = [_integer(value, "shard step", minimum=0) for value in record["shard_steps"]]
            item["expected_shards"] = _integer(record.get("expected_shards", len(item["shard_steps"])), "expected_shards")
            item["manifest_published"] = bool(record.get("manifest_published", False))
        normalized.append(item)
    generation_checks = []
    for item in normalized:
        cut_valid = (len(item["shard_steps"]) == item["expected_shards"] and all(value == item["step"] for value in item["shard_steps"])) if "shard_steps" in item else None
        committed = item.get("manifest_published", item["completed"])
        eligible = item["step"] <= failed_at and item["completed"] and item["digest_valid"] and committed and cut_valid is not False
        generation_checks.append({"step": item["step"], "completed": item["completed"], "digest_valid": item["digest_valid"], "consistent_cut": cut_valid, "manifest_published": committed, "eligible": eligible})
    candidates = [record for record, check in zip(normalized, generation_checks) if check["eligible"]]
    if not candidates:
        return {
            "inputs": {"checkpoints": deepcopy(normalized), "failure_step": failed_at},
            "recoverable": False,
            "generation_checks": generation_checks,
            "restore_step": None,
            "newest_checkpoint_step": max((item["step"] for item in normalized), default=None),
            "lost_steps": None,
            "fallback_generations": None,
        }
    selected = max(candidates, key=lambda item: item["step"])
    newer = [item for item in normalized if selected["step"] < item["step"] <= failed_at]
    return {
        "inputs": {"checkpoints": deepcopy(normalized), "failure_step": failed_at},
        "recoverable": True,
        "generation_checks": generation_checks,
        "restore_step": selected["step"],
        "newest_checkpoint_step": max((item["step"] for item in normalized), default=None),
        "lost_steps": failed_at - selected["step"],
        "fallback_generations": len(newer),
    }


def checkpoint_fault_fixture(track_id: str, fault: str = "corrupt") -> dict:
    """Construct an immutable checkpoint-fault experiment fixture.

    Baseline records are identical across all fault choices (steps 100, 200, 300, 400
    all completed and digest-valid, failure at step 450). The fault copy modifies only
    the newest generation (step 400), causing recovery to safely fall back to step 300.
    """
    scenario = track_scenario(track_id)
    failure_step = 450
    baseline_records = [
        {"step": 100, "completed": True, "digest_valid": True},
        {"step": 200, "completed": True, "digest_valid": True},
        {"step": 300, "completed": True, "digest_valid": True},
        {"step": 400, "completed": True, "digest_valid": True},
    ]
    for record in baseline_records:
        record.update(shard_steps=[record["step"]] * 4, expected_shards=4, manifest_published=True)
    result_records = deepcopy(baseline_records)
    if fault == "corrupt":
        result_records[3]["digest_valid"] = False
    elif fault == "incomplete":
        result_records[3]["completed"] = False
    elif fault == "mixed_step":
        result_records[3]["shard_steps"][3] = 300
    elif fault == "unpublished":
        result_records[3]["manifest_published"] = False
    elif fault == "missing_shard":
        result_records[3]["shard_steps"].pop()
    else:
        raise ValueError(f"unknown fault {fault!r}; choose corrupt, incomplete, mixed_step, unpublished or missing_shard")

    baseline_selection = select_restore_checkpoint(baseline_records, failure_step=failure_step)
    result_selection = select_restore_checkpoint(result_records, failure_step=failure_step)

    return {
        "track_id": track_id,
        "inputs": {"track_id": track_id, "fault": fault},
        "fault": fault,
        "failure_step": failure_step,
        "baseline_records": deepcopy(baseline_records),
        "result_records": deepcopy(result_records),
        "baseline": baseline_selection,
        "result": result_selection,
        "training_backend_role": scenario.get("training_backend_role", ""),
        "manifest_role": scenario.get("manifest_role", ""),
        "manifest_context": scenario.get("manifest_context", ""),
    }


def serving_capacity(track_id: str, *, fleet_size: int | None = None) -> dict:
    """Calculate expected available capacity for independently serving devices."""
    scenario = track_scenario(track_id)
    devices = _integer(fleet_size if fleet_size is not None else scenario["serving_devices"], "fleet_size")
    component_mttf_min = _positive(scenario["serving_component_mttf_h"], "serving_component_mttf_h") * 60
    component_repair_min = _positive(scenario["serving_repair_min"], "serving_repair_min")
    domain_mttf_min = _positive(scenario["serving_domain_mttf_h"], "serving_domain_mttf_h") * 60
    domain_repair_min = _positive(scenario["serving_domain_repair_min"], "serving_domain_repair_min")
    component_availability = component_mttf_min / (component_mttf_min + component_repair_min)
    domain_availability = domain_mttf_min / (domain_mttf_min + domain_repair_min)
    available_fraction = component_availability * domain_availability
    return {
        "track_id": track_id,
        "inputs": {"track_id": track_id, "fleet_size": devices},
        "semantics": "independent_serving_capacity",
        "fleet_size": devices,
        "component_availability": component_availability,
        "domain_availability": domain_availability,
        "available_fraction": available_fraction,
        "expected_available_devices": devices * available_fraction,
        "expected_unavailable_devices": devices * (1.0 - available_fraction),
        "fleet_terminated": False,
    }


def replica_availability(
    *,
    replicas: int,
    placement_domains: int,
    component_availability: float,
    domain_mttf_h: float,
    domain_repair_min: float,
) -> dict:
    """Calculate service availability with replicas placed across failure domains."""
    replica_count = _integer(replicas, "replicas")
    domains = _integer(placement_domains, "placement_domains")
    if domains > replica_count:
        raise ValueError("placement_domains cannot exceed replicas")
    if not 0 < component_availability < 1:
        raise ValueError("component_availability must be between zero and one")
    domain_mttf_min = _positive(domain_mttf_h, "domain_mttf_h") * 60
    repair_min = _positive(domain_repair_min, "domain_repair_min")
    domain_down = repair_min / (domain_mttf_min + repair_min)

    group_sizes = [replica_count // domains] * domains
    for index in range(replica_count % domains):
        group_sizes[index] += 1
    component_down = 1.0 - component_availability
    group_down = [domain_down + (1.0 - domain_down) * component_down**size for size in group_sizes]
    service_down = math.prod(group_down)
    return {
        "inputs": {
            "replicas": replica_count,
            "placement_domains": domains,
            "component_availability": component_availability,
            "domain_mttf_h": float(domain_mttf_h),
            "domain_repair_min": repair_min,
        },
        "replicas": replica_count,
        "placement_domains": domains,
        "replicas_per_domain": tuple(group_sizes),
        "domain_down_probability": domain_down,
        "service_availability": 1.0 - service_down,
        "shared_domain_correlation_modeled": domains < replica_count,
    }


__all__ = [
    "TRACKS",
    "checkpoint_fault_fixture",
    "checkpoint_tradeoff",
    "get_track_scenario",
    "replica_availability",
    "select_restore_checkpoint",
    "serving_capacity",
    "track_scenario",
    "training_exposure",
]


def fault_verification_trace(*, persistence: str = "transient", manifestation: str = "value_corrupting", scope: str = "rank", verification: str = "all") -> dict:
    """Replay an explicitly injected fault, not a field failure-rate estimate.

    Arithmetic corruption precedes a correctly checksummed transfer. Verification
    recomputes selected rank contributions; CRC alone therefore cannot detect it.
    Three attempts distinguish retry success from recurrence and permanence.
    """
    if persistence not in {"transient", "intermittent", "permanent"}:
        raise ValueError("unknown persistence")
    if manifestation not in {"fail_stop", "degraded", "value_corrupting"}:
        raise ValueError("unknown manifestation")
    if scope not in {"rank", "rack"} or verification not in {"none", "rank0", "all"}:
        raise ValueError("unknown scope or verification")
    affected = [3] if scope == "rank" else [2, 3]
    checked = {"none": [], "rank0": [0], "all": list(range(4))}[verification]
    expected = [2, 4, 6, 8]  # synthetic scalar contributions, not a measured gradient distribution
    attempts = []
    for attempt in range(3):
        active = persistence == "permanent" or attempt == 0 or (persistence == "intermittent" and attempt == 2)
        values = [value + (10 if active and rank in affected and manifestation == "value_corrupting" else 0) for rank, value in enumerate(expected)]
        alive = [not(active and rank in affected and manifestation == "fail_stop") for rank in range(4)]
        durations = [8 if active and rank in affected and manifestation == "degraded" else 1 for rank in range(4)]
        payload = json.dumps(values).encode("utf-8")
        sent_crc = zlib.crc32(payload)
        received_crc = zlib.crc32(bytes(payload))
        mismatches = [rank for rank in checked if values[rank] != expected[rank]]
        collective_completes = all(alive)
        observed = sum(values) / len(values) if collective_completes else None
        attempts.append({"attempt": attempt, "fault_active": active, "heartbeat_alive": alive,
                         "step_duration_units": max(durations) if collective_completes else None,
                         "contributions": values, "sent_crc": sent_crc, "received_crc": received_crc,
                         "transport_crc_valid": sent_crc == received_crc,
                         "checked_ranks": checked, "arithmetic_mismatches": mismatches,
                         "collective_completes": collective_completes,
                         "reduced_mean": observed,
                         "accepted": collective_completes and not mismatches,
                         "accepted_wrong_value": collective_completes and not mismatches and observed != sum(expected) / len(expected)})
    return {"inputs": {"persistence": persistence, "manifestation": manifestation, "scope": scope, "verification": verification},
            "axes": {"persistence": persistence, "manifestation": manifestation, "scope": scope},
            "affected_ranks": affected, "expected_contributions": expected,
            "reference_mean": sum(expected) / len(expected), "attempts": attempts,
            "verification_recomputations": len(checked) * len(attempts),
            "retry_response": {"transient": "retry then verify", "intermittent": "quarantine recurring source and verify", "permanent": "replace and restore"}[persistence],
            "scope_note": "Rack injection shares one fault across ranks 2 and 3; this is a supplied trace, not a correlation estimate."}


def gray_failure_trace(*, detector: str = "progress", threshold: float = 2.0, incident: str = "gray") -> dict:
    """Separate heartbeat liveness from slowest-rank progress on exact traces."""
    if detector not in {"heartbeat", "progress"} or incident not in {"healthy", "gray", "crash"}:
        raise ValueError("unknown detector or incident")
    limit = _positive(threshold, "threshold")
    times = [1.0, 1.0, 1.0, 8.0 if incident == "gray" else 1.0]
    alive = [True, True, True, incident != "crash"]
    suspected = [rank for rank in range(4) if not alive[rank] or (detector == "progress" and times[rank] > limit)]
    return {"inputs": {"detector": detector, "threshold": limit, "incident": incident},
            "heartbeat_alive": alive, "rank_step_units": times,
            "barrier_step_units": max(times) if all(alive) else None,
            "suspected_ranks": suspected, "detected": bool(suspected),
            "false_positive_ranks": [rank for rank in suspected if alive[rank] and times[rank] == 1.0],
            "response": "recover stopped rank" if not all(alive) else ("mitigate straggler; checkpoint restore does not remove slow hardware" if suspected and incident == "gray" else "inspect false alarm" if suspected else "continue without alarm"),
            "scope": "One deterministic observation window; no measured latency distribution or false-alarm probability."}


def elastic_recovery_trace(*, preserve_batch: bool = True, data_cursor: str = "exact", reconstruct_state: bool = True, fence_epoch: bool = True) -> dict:
    """Resize a supplied data-parallel job while checking independent invariants.

    State IDs, sample IDs and writer epochs are explicit synthetic records. No
    statistical convergence claim follows from passing these necessary gates.
    """
    if data_cursor not in {"exact", "rewind", "skip"}:
        raise ValueError("unknown data_cursor")
    for name, value in (("preserve_batch", preserve_batch), ("reconstruct_state", reconstruct_state), ("fence_epoch", fence_epoch)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
    old_workers, survivors, microbatch, old_accumulation = 4, 2, 2, 1
    accumulation = old_workers // survivors if preserve_batch else old_accumulation
    consumed = list(range(8))
    start = {"exact": 8, "rewind": 4, "skip": 12}[data_cursor]
    resumed = list(range(start, 16))
    observed = consumed + resumed
    duplicates = sorted({item for item in observed if observed.count(item) > 1})
    missing = sorted(set(range(16)) - set(observed))
    required_state = [f"optimizer_shard_{rank}" for rank in range(old_workers)]
    available_state = required_state if reconstruct_state else required_state[:survivors]
    writer_attempts = [{"epoch": 1, "writer": "lost-rank-late-write"}, {"epoch": 2, "writer": "resumed-job"}]
    writes = [{**entry, "accepted": not fence_epoch or entry["epoch"] == 2} for entry in writer_attempts]
    gates = {"effective_batch_preserved": survivors * microbatch * accumulation == old_workers * microbatch * old_accumulation,
             "exactly_once_data": not duplicates and not missing,
             "all_state_reconstructable": set(available_state) == set(required_state),
             "stale_writer_fenced": not writes[0]["accepted"]}
    return {"inputs": {"preserve_batch": preserve_batch, "data_cursor": data_cursor, "reconstruct_state": reconstruct_state, "fence_epoch": fence_epoch},
            "before": {"workers": old_workers, "microbatch": microbatch, "accumulation": old_accumulation, "effective_batch": old_workers * microbatch * old_accumulation},
            "after": {"workers": survivors, "microbatch": microbatch, "accumulation": accumulation, "effective_batch": survivors * microbatch * accumulation},
            "consumed_samples": consumed, "resumed_samples": resumed, "duplicate_samples": duplicates, "missing_samples": missing,
            "required_state": required_state, "available_state": available_state, "writer_attempts": writes,
            "gates": gates, "safe_to_resume": all(gates.values()),
            "scope": "Necessary consistency checks on supplied data-parallel records; no reshard implementation, learning-rate validation or measured recovery throughput."}
