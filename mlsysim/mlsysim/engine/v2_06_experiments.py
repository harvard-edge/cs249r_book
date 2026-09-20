"""Chapter 6 collective-communication experiments.

The functions in this module compose MLSysIM's unit-aware communication
physics into small, deterministic experiments.  Track constants are explicitly
illustrative workload fixtures.  They are not measurements of named products.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from mlsysim.core.units import GB, MB, Q_, byte, millisecond, second
from mlsysim.hardware.registry import Hardware
from mlsysim.physics import (
    calc_all_to_all_time,
    calc_bisection_bandwidth,
    calc_hierarchical_allreduce_time,
    calc_ring_allreduce_time,
    calc_tree_allreduce_time,
)
from mlsysim.systems.registry import Systems


TRACK_IDS = ("tinyml", "mobile", "edge", "cloud")


def _gbps(value: float):
    return Q_(value, "gigabit / second").to(byte / second)


_TRACKS: dict[str, dict[str, Any]] = {
    "tinyml": {
        "display": "TinyML",
        "fleet_semantics": "Device updates terminate at gateways; the participants are gateway aggregation workers.",
        "participant_kind": "gateway aggregation workers",
        "compute_phase": "gateway aggregation",
        "participants": 16,
        "local_group": 4,
        "payload": 1 * MB,
        "inter_bandwidth": Systems.Fabrics.Ethernet_10G.bandwidth,
        "inter_latency": Q_(0.35, "millisecond"),
        "intra_bandwidth": Systems.Fabrics.Ethernet_100G.bandwidth,
        "intra_latency": Q_(40, "microsecond"),
        "endpoint_count": 512,
        "endpoint_payload": Q_(64, "kilobyte"),
        "endpoint_bandwidth": _gbps(0.02),
        "endpoint_latency": Q_(30, "millisecond"),
        "backward_ms": 80.0,
        "calibration_flat_ms": 18.5,
        "calibration_hier_ms": 8.8,
    },
    "mobile": {
        "display": "Mobile",
        "fleet_semantics": "Phones upload updates to regional services; the participants are backend aggregation workers.",
        "participant_kind": "regional backend workers",
        "compute_phase": "backend aggregation",
        "participants": 32,
        "local_group": 4,
        "payload": 8 * MB,
        "inter_bandwidth": Systems.Fabrics.Ethernet_10G.bandwidth,
        "inter_latency": Q_(0.8, "millisecond"),
        "intra_bandwidth": Systems.Fabrics.Ethernet_100G.bandwidth,
        "intra_latency": Q_(25, "microsecond"),
        "endpoint_count": 2_048,
        "endpoint_payload": Q_(256, "kilobyte"),
        "endpoint_bandwidth": _gbps(0.05),
        "endpoint_latency": Q_(45, "millisecond"),
        "backward_ms": 120.0,
        "calibration_flat_ms": 39.0,
        "calibration_hier_ms": 15.2,
    },
    "edge": {
        "display": "Edge",
        "fleet_semantics": "Regional accelerator pools exchange training updates; sensors are not collective participants.",
        "participant_kind": "regional accelerator workers",
        "compute_phase": "backward pass",
        "participants": 64,
        "local_group": 8,
        "payload": 64 * MB,
        "inter_bandwidth": Systems.Fabrics.Ethernet_100G.bandwidth,
        "inter_latency": Systems.SwitchFabric.AlphaRoce,
        "intra_bandwidth": Systems.Fabrics.Ethernet_400G.bandwidth,
        "intra_latency": Q_(2, "microsecond"),
        "endpoint_count": 256,
        "endpoint_payload": Q_(1, "megabyte"),
        "endpoint_bandwidth": Systems.Fabrics.Ethernet_10G.bandwidth,
        "endpoint_latency": Q_(0.5, "millisecond"),
        "backward_ms": 180.0,
        "calibration_flat_ms": 17.4,
        "calibration_hier_ms": 7.1,
    },
    "cloud": {
        "display": "Cloud",
        "fleet_semantics": "Multi-node accelerator workers execute one synchronous training collective.",
        "participant_kind": "multi-node accelerator workers",
        "compute_phase": "backward pass",
        "participants": 128,
        "local_group": 8,
        "payload": 256 * MB,
        "inter_bandwidth": Systems.Fabrics.InfiniBand_NDR.bandwidth,
        "inter_latency": Systems.SwitchFabric.AlphaNdr,
        "intra_bandwidth": Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        "intra_latency": Q_(0.5, "microsecond"),
        "endpoint_count": 0,
        "endpoint_payload": 0 * MB,
        "endpoint_bandwidth": Systems.Fabrics.InfiniBand_NDR.bandwidth,
        "endpoint_latency": 0 * second,
        "backward_ms": 260.0,
        "calibration_flat_ms": 14.0,
        "calibration_hier_ms": 4.2,
    },
}


_COMPRESSION_EVIDENCE: dict[str, dict[str, Any]] = {
    "none": {
        "ratio": 1.0,
        "encode_gb_s": None,
        "decode_gb_s": None,
        "target_step_multiplier": 1.0,
        "quality_target_reached": True,
        "quality_gap_pp": 0.0,
        "error_feedback": False,
    },
    "fp8": {
        "ratio": 2.0,
        "encode_gb_s": 80.0,
        "decode_gb_s": 120.0,
        "target_step_multiplier": 1.03,
        "quality_target_reached": True,
        "quality_gap_pp": 0.0,
        "error_feedback": False,
    },
    "topk_error_feedback": {
        "ratio": 20.0,
        "encode_gb_s": 12.0,
        "decode_gb_s": 20.0,
        "target_step_multiplier": 1.12,
        "quality_target_reached": True,
        "quality_gap_pp": 0.0,
        "error_feedback": True,
    },
    "topk_naive": {
        "ratio": 20.0,
        "encode_gb_s": 14.0,
        "decode_gb_s": 22.0,
        "target_step_multiplier": 1.0,
        "quality_target_reached": False,
        "quality_gap_pp": 2.5,
        "error_feedback": False,
    },
}


def _track(track_id: str) -> dict[str, Any]:
    if track_id not in _TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACK_IDS)}")
    return _TRACKS[track_id]


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _ms(value) -> float:
    return round(float(value.to(millisecond).magnitude), 6)


def _mb(value) -> float:
    return round(float(value.to(MB).magnitude), 6)


def _gb_s(value) -> float:
    return round(float(value.to(GB / second).magnitude), 6)


def _allreduce_time(algorithm: str, payload, participants: int, bandwidth, latency):
    if algorithm == "ring":
        return calc_ring_allreduce_time(payload, participants, bandwidth, latency)
    if algorithm == "tree":
        return calc_tree_allreduce_time(payload, participants, bandwidth, latency)
    raise ValueError("algorithm must be ring or tree")


def track_profile(track_id: str) -> dict[str, Any]:
    """Return the learner-facing fleet semantics and default physical envelope."""
    track = _track(track_id)
    endpoint_upload_ms = 0.0
    if track["endpoint_count"]:
        endpoint_upload_ms = _ms(
            track["endpoint_latency"]
            + track["endpoint_payload"] / track["endpoint_bandwidth"]
        )
    return {
        "inputs": {"track_id": track_id},
        "track_id": track_id,
        "display": track["display"],
        "fleet_semantics": track["fleet_semantics"],
        "participant_kind": track["participant_kind"],
        "participants": track["participants"],
        "local_group": track["local_group"],
        "compute_phase": track.get("compute_phase", "backward pass"),
        "payload_mb": _mb(track["payload"]),
        "inter_bandwidth_gb_s": _gb_s(track["inter_bandwidth"]),
        "inter_latency_ms": _ms(track["inter_latency"]),
        "endpoint_count": track["endpoint_count"],
        "endpoint_payload_mb": _mb(track["endpoint_payload"]),
        "endpoint_upload_ms_per_device": endpoint_upload_ms,
        "endpoint_traffic_is_collective": False,
        "provenance": "Illustrative fleet workload envelope; physical fabric rates use MLSysIM registry quantities.",
    }


def algorithm_crossover(
    track_id: str,
    *,
    payload_mb: float | None = None,
    participants: int | None = None,
) -> dict[str, Any]:
    """Compare ring and simple-tree AllReduce with correct alpha-beta counts."""
    track = _track(track_id)
    n = track["participants"] if participants is None else _integer(participants, "participants", minimum=3)
    payload_value = _mb(track["payload"]) if payload_mb is None else _finite(payload_mb, "payload_mb", minimum=0.0)
    payload = payload_value * MB
    bandwidth = track["inter_bandwidth"]
    latency = track["inter_latency"]
    levels = math.ceil(math.log2(n))

    ring_latency = 2 * (n - 1) * latency
    tree_latency = 2 * levels * latency
    ring_transfer = 2 * (n - 1) / n * payload / bandwidth
    tree_transfer = 2 * levels * payload / bandwidth
    ring = calc_ring_allreduce_time(payload, n, bandwidth, latency)
    tree = calc_tree_allreduce_time(payload, n, bandwidth, latency)
    transfer_coefficient_gap = 2 * levels - 2 * (n - 1) / n
    exact_crossover = ((ring_latency - tree_latency) * bandwidth / transfer_coefficient_gap).to(MB)
    inputs = {"track_id": track_id, "payload_mb": payload_value, "participants": n}

    return {
        "inputs": inputs,
        "track_id": track_id,
        "participant_kind": track["participant_kind"],
        "participants": n,
        "payload_mb": payload_value,
        "bandwidth_gb_s": _gb_s(bandwidth),
        "latency_ms": _ms(latency),
        "crossover_mb": _mb(exact_crossover),
        "winner": "ring" if ring < tree else "tree",
        "ring": {
            "inputs": {**inputs, "algorithm": "ring"},
            "startup_steps": 2 * (n - 1),
            "startup_ms": _ms(ring_latency),
            "transfer_mb": _mb(2 * (n - 1) / n * payload),
            "transfer_ms": _ms(ring_transfer),
            "total_ms": _ms(ring),
        },
        "tree": {
            "inputs": {**inputs, "algorithm": "tree"},
            "startup_steps": 2 * levels,
            "startup_ms": _ms(tree_latency),
            "transfer_mb": _mb(2 * levels * payload),
            "transfer_ms": _ms(tree_transfer),
            "total_ms": _ms(tree),
        },
        "model": "Analytical alpha-beta comparison; the tree is the simple full-message tree model.",
    }


def algorithm_cases(track_id: str) -> dict[str, Any]:
    """Return three MLSysIM-computed payload choices around the exact crossover."""
    boundary = algorithm_crossover(track_id, payload_mb=0)
    crossover_mb = boundary["crossover_mb"]
    track_payload_mb = track_profile(track_id)["payload_mb"]
    cases = (
        {"case_id": "below", "label": "Below crossover", "payload_mb": round(crossover_mb / 10, 6)},
        {"case_id": "track", "label": "Track payload", "payload_mb": track_payload_mb},
        {"case_id": "above", "label": "Above crossover", "payload_mb": round(crossover_mb * 10, 6)},
    )
    return {
        "inputs": {"track_id": track_id},
        "track_id": track_id,
        "crossover_mb": crossover_mb,
        "cases": cases,
    }


def semantic_exchange(
    track_id: str,
    *,
    payload_mb: float | None = None,
    hotspot_fraction: float | None = None,
    oversubscription: float = 1.0,
) -> dict[str, Any]:
    """Compare identical-result reduction with destination-specific routed traffic.

    ``payload_mb`` is each participant's logical input buffer for both cases.
    The routed case sends distinct records and is bounded by the hottest
    receiver and fabric bisection capacity; it does not pretend to be a
    reduction with a different label.
    """
    track = _track(track_id)
    n = track["participants"]
    payload_value = _mb(track["payload"]) if payload_mb is None else _finite(payload_mb, "payload_mb", minimum=0.0)
    payload = payload_value * MB
    hot = 1 / n if hotspot_fraction is None else _finite(hotspot_fraction, "hotspot_fraction", minimum=0.0)
    if hot < 1 / n or hot > 1:
        raise ValueError("hotspot_fraction must be between the balanced share 1/N and 1")
    oversub = _finite(oversubscription, "oversubscription", minimum=1.0)
    bandwidth = track["inter_bandwidth"]
    latency = track["inter_latency"]

    reduction = calc_ring_allreduce_time(payload, n, bandwidth, latency)
    balanced_routed = calc_all_to_all_time(payload, n, bandwidth, latency)
    startup = (n - 1) * latency
    outbound_bytes = (n - 1) / n * payload
    hottest_receiver_bytes = (n - 1) * payload * hot
    injection_time = outbound_bytes / bandwidth
    receiver_time = hottest_receiver_bytes / bandwidth
    crossing_bytes = n * payload / 2
    bisection_bandwidth = calc_bisection_bandwidth(n // 2, bandwidth, oversub)
    bisection_time = crossing_bytes / bisection_bandwidth
    transfer_bounds = {
        "injection": injection_time,
        "hottest_receiver": receiver_time,
        "bisection": bisection_time,
    }
    limiting = max(transfer_bounds, key=transfer_bounds.get)
    total_bounds = {
        "startup": startup,
        "injection": injection_time,
        "hottest_receiver": receiver_time,
        "bisection": bisection_time,
    }
    limiting_bound = max(total_bounds, key=total_bounds.get)
    routed = startup + max(transfer_bounds.values())
    inputs = {
        "track_id": track_id,
        "payload_mb": payload_value,
        "hotspot_fraction": hot,
        "oversubscription": oversub,
    }

    return {
        "inputs": inputs,
        "track_id": track_id,
        "participants": n,
        "payload_mb_per_participant": payload_value,
        "reduction": {
            "inputs": {
                "track_id": track_id,
                "payload_mb": payload_value,
                "participants": n,
                "semantic": "reduction",
            },
            "semantic": "Every participant receives the same elementwise aggregate.",
            "output_is_identical_at_each_participant": True,
            "time_ms": _ms(reduction),
            "per_participant_wire_mb": _mb(2 * (n - 1) / n * payload),
        },
        "routed": {
            "inputs": {**inputs, "semantic": "routed"},
            "semantic": "Each record has a destination; receivers obtain different records.",
            "output_is_identical_at_each_participant": False,
            "balanced_schedule_ms": _ms(balanced_routed),
            "time_ms": _ms(routed),
            "hotspot_fraction": hot,
            "hottest_receiver_mb": _mb(hottest_receiver_bytes),
            "bisection_bandwidth_gb_s": _gb_s(bisection_bandwidth),
            "limiting_transfer_bound": limiting,
            "limiting_bound": limiting_bound,
            "oversubscription": oversub,
        },
        "endpoint_note": track["fleet_semantics"],
    }


def routing_cases(track_id: str) -> dict[str, Any]:
    """Return balanced and hotspot destination shares for the routed experiment."""
    participants = _track(track_id)["participants"]
    cases = (
        {"case_id": "balanced", "label": "Balanced destinations", "hotspot_fraction": 1 / participants},
        {"case_id": "moderate", "label": "Moderate hotspot", "hotspot_fraction": 0.15},
        {"case_id": "severe", "label": "Severe hotspot", "hotspot_fraction": 0.35},
    )
    return {
        "inputs": {"track_id": track_id},
        "track_id": track_id,
        "participants": participants,
        "cases": cases,
    }


def _calibration_fixture(track_id: str) -> dict[str, Any]:
    track = _track(track_id)
    return {
        "track_id": track_id,
        "participants": track["participants"],
        "local_group": track["local_group"],
        "payload_mb": _mb(track["payload"]),
        "flat_observed_ms": track["calibration_flat_ms"],
        "hierarchical_observed_ms": track["calibration_hier_ms"],
        "flat_algorithm": "ring",
        "hierarchical_algorithm": "local reduce-scatter + inter-group ring + local all-gather",
        "placement": "contiguous local groups with one independent inter-group path per local rank",
        "provenance": "Supplied illustrative calibration fixture; not a live or product benchmark.",
    }


def topology_comparison(
    track_id: str,
    *,
    payload_mb: float | None = None,
    calibration_fixture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare flat and hierarchical schedules, then apply matched calibration."""
    track = _track(track_id)
    n = track["participants"]
    group = track["local_group"]
    nodes = n // group
    payload_value = _mb(track["payload"]) if payload_mb is None else _finite(payload_mb, "payload_mb", minimum=0.0)
    payload = payload_value * MB
    flat = calc_ring_allreduce_time(payload, n, track["inter_bandwidth"], track["inter_latency"])
    hierarchical = calc_hierarchical_allreduce_time(
        payload,
        nodes,
        group,
        track["intra_bandwidth"],
        track["inter_bandwidth"],
        track["intra_latency"],
        track["inter_latency"],
    )

    fixture = dict(_calibration_fixture(track_id) if calibration_fixture is None else calibration_fixture)
    required = {
        "track_id", "participants", "local_group", "payload_mb", "flat_observed_ms",
        "hierarchical_observed_ms", "flat_algorithm", "hierarchical_algorithm", "placement", "provenance",
    }
    missing = required - fixture.keys()
    if missing:
        raise ValueError(f"calibration fixture is missing: {', '.join(sorted(missing))}")
    if fixture["track_id"] != track_id or fixture["participants"] != n or fixture["local_group"] != group:
        raise ValueError("calibration fixture must match track, process-group size, and local grouping")
    fixture_payload = _finite(fixture["payload_mb"], "fixture payload_mb", minimum=0.0) * MB
    flat_reference = calc_ring_allreduce_time(
        fixture_payload, n, track["inter_bandwidth"], track["inter_latency"]
    )
    hierarchy_reference = calc_hierarchical_allreduce_time(
        fixture_payload,
        nodes,
        group,
        track["intra_bandwidth"],
        track["inter_bandwidth"],
        track["intra_latency"],
        track["inter_latency"],
    )
    flat_factor = _finite(fixture["flat_observed_ms"], "flat_observed_ms", minimum=0.0) / _ms(flat_reference)
    hierarchy_factor = _finite(fixture["hierarchical_observed_ms"], "hierarchical_observed_ms", minimum=0.0) / _ms(hierarchy_reference)
    calibrated_flat = flat * flat_factor
    calibrated_hierarchy = hierarchical * hierarchy_factor
    inputs = {
        "track_id": track_id,
        "payload_mb": payload_value,
        "calibration_fixture": fixture,
    }

    return {
        "inputs": inputs,
        "track_id": track_id,
        "participants": n,
        "local_group": group,
        "payload_mb": payload_value,
        "flat": {
            "inputs": {**inputs, "schedule": "flat_ring"},
            "analytical_ms": _ms(flat),
            "calibration_factor": round(flat_factor, 6),
            "calibrated_ms": _ms(calibrated_flat),
        },
        "hierarchical": {
            "inputs": {**inputs, "schedule": "hierarchical"},
            "analytical_ms": _ms(hierarchical),
            "calibration_factor": round(hierarchy_factor, 6),
            "calibrated_ms": _ms(calibrated_hierarchy),
        },
        "analytical_winner": "flat" if flat < hierarchical else "hierarchical",
        "calibrated_winner": "flat" if calibrated_flat < calibrated_hierarchy else "hierarchical",
        "fixture": fixture,
        "assumption": "The hierarchy model assumes one independent inter-group path per local rank.",
    }


def _default_layers(track_id: str) -> list[dict[str, float | str]]:
    backward_ms = _TRACKS[track_id]["backward_ms"]
    payload_mb = _mb(_TRACKS[track_id]["payload"])
    shares = (0.10, 0.16, 0.22, 0.24, 0.18, 0.10)
    return [
        {
            "layer": f"L{index + 1}",
            "ready_ms": round(backward_ms * (index + 1) / len(shares), 6),
            "gradient_mb": round(payload_mb * share, 6),
        }
        for index, share in enumerate(shares)
    ]


def overlap_bucket_options(track_id: str) -> dict[str, Any]:
    """Return bucket controls derived from the track's full gradient payload."""
    payload_mb = _mb(_track(track_id)["payload"])
    options = (
        {"option_id": "fine", "label": "Fine buckets", "bucket_mb": round(payload_mb * 0.10, 6)},
        {"option_id": "medium", "label": "Medium buckets", "bucket_mb": round(payload_mb * 0.25, 6)},
        {"option_id": "fused", "label": "One fused bucket", "bucket_mb": payload_mb},
    )
    return {
        "inputs": {"track_id": track_id},
        "track_id": track_id,
        "payload_mb": payload_mb,
        "options": options,
    }


def overlap_timeline(
    track_id: str,
    *,
    bucket_mb: float,
    algorithm: str = "ring",
    layers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Schedule ready gradient buckets on one serialized network resource."""
    track = _track(track_id)
    bucket_limit = _finite(bucket_mb, "bucket_mb", minimum=0.0)
    if bucket_limit == 0:
        raise ValueError("bucket_mb must be greater than 0")
    source = _default_layers(track_id) if layers is None else list(layers)
    if not source:
        raise ValueError("layers must contain at least one readiness record")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(source):
        if not isinstance(item, Mapping):
            raise ValueError("each layer must be a mapping")
        ready = _finite(item.get("ready_ms"), f"layers[{index}].ready_ms", minimum=0.0)
        size = _finite(item.get("gradient_mb"), f"layers[{index}].gradient_mb", minimum=0.0)
        if size == 0:
            raise ValueError("gradient_mb must be greater than 0")
        records.append({"layer": str(item.get("layer", f"L{index + 1}")), "ready_ms": ready, "gradient_mb": size})
    records.sort(key=lambda item: item["ready_ms"])

    buckets: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    pending_mb = 0.0

    def flush() -> None:
        nonlocal pending, pending_mb
        if pending:
            buckets.append({
                "layers": [item["layer"] for item in pending],
                "ready_ms": max(item["ready_ms"] for item in pending),
                "payload_mb": pending_mb,
            })
            pending = []
            pending_mb = 0.0

    for record in records:
        if pending and pending_mb + record["gradient_mb"] > bucket_limit:
            flush()
        pending.append(record)
        pending_mb += record["gradient_mb"]
        if pending_mb >= bucket_limit:
            flush()
    flush()

    network_free_ms = 0.0
    total_communication_ms = 0.0
    events: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        duration_ms = _ms(
            _allreduce_time(
                algorithm,
                bucket["payload_mb"] * MB,
                track["participants"],
                track["inter_bandwidth"],
                track["inter_latency"],
            )
        )
        start_ms = max(bucket["ready_ms"], network_free_ms)
        end_ms = start_ms + duration_ms
        events.append({
            "bucket": index + 1,
            **bucket,
            "start_ms": round(start_ms, 6),
            "duration_ms": duration_ms,
            "end_ms": round(end_ms, 6),
        })
        network_free_ms = end_ms
        total_communication_ms += duration_ms

    backward_end_ms = max(record["ready_ms"] for record in records)
    final_end_ms = events[-1]["end_ms"]
    exposed_ms = max(0.0, final_end_ms - backward_end_ms)
    hidden_ms = max(0.0, total_communication_ms - exposed_ms)
    overlap_fraction = round(hidden_ms / total_communication_ms, 6) if total_communication_ms else 0.0
    outcome = "mostly_hidden" if overlap_fraction >= 0.5 else "mostly_exposed"
    inputs = {
        "track_id": track_id,
        "bucket_mb": bucket_limit,
        "algorithm": algorithm,
        "layers": records,
    }
    return {
        "inputs": inputs,
        "track_id": track_id,
        "algorithm": algorithm,
        "compute_phase": track.get("compute_phase", "backward pass"),
        "bucket_limit_mb": bucket_limit,
        "backward_end_ms": round(backward_end_ms, 6),
        "total_gradient_mb": round(sum(item["gradient_mb"] for item in records), 6),
        "total_communication_ms": round(total_communication_ms, 6),
        "hidden_communication_ms": round(hidden_ms, 6),
        "exposed_communication_ms": round(exposed_ms, 6),
        "overlap_fraction": overlap_fraction,
        "outcome": outcome,
        "step_end_ms": round(max(backward_end_ms, final_end_ms), 6),
        "bucket_count": len(events),
        "events": events,
        "resource_model": "Buckets launch after all member gradients are ready and share one serialized network resource.",
    }


def compression_comparison(
    track_id: str,
    *,
    method: str,
    payload_mb: float | None = None,
    baseline_compute_ms: float | None = None,
    baseline_target_steps: int = 1_000,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate byte savings, codec work, and supplied target-quality evidence."""
    track = _track(track_id)
    if method not in _COMPRESSION_EVIDENCE:
        raise ValueError(f"method must be one of {', '.join(_COMPRESSION_EVIDENCE)}")
    fixture = dict(_COMPRESSION_EVIDENCE[method] if evidence is None else evidence)
    required = {
        "ratio", "encode_gb_s", "decode_gb_s", "target_step_multiplier",
        "quality_target_reached", "quality_gap_pp", "error_feedback",
    }
    missing = required - fixture.keys()
    if missing:
        raise ValueError(f"compression evidence is missing: {', '.join(sorted(missing))}")
    ratio = _finite(fixture["ratio"], "ratio", minimum=1.0)
    step_multiplier = _finite(fixture["target_step_multiplier"], "target_step_multiplier", minimum=1.0)
    quality_gap = _finite(fixture["quality_gap_pp"], "quality_gap_pp", minimum=0.0)
    target_steps = _integer(baseline_target_steps, "baseline_target_steps")
    compute_ms = track["backward_ms"] if baseline_compute_ms is None else _finite(
        baseline_compute_ms, "baseline_compute_ms", minimum=0.0
    )
    payload_value = _mb(track["payload"]) if payload_mb is None else _finite(payload_mb, "payload_mb", minimum=0.0)
    payload = payload_value * MB
    compressed = payload / ratio
    uncompressed_comm = calc_ring_allreduce_time(
        payload, track["participants"], track["inter_bandwidth"], track["inter_latency"]
    )
    compressed_comm = calc_ring_allreduce_time(
        compressed, track["participants"], track["inter_bandwidth"], track["inter_latency"]
    )
    encode_ms = 0.0
    decode_ms = 0.0
    if ratio > 1:
        encode_rate = _finite(fixture["encode_gb_s"], "encode_gb_s", minimum=0.0)
        decode_rate = _finite(fixture["decode_gb_s"], "decode_gb_s", minimum=0.0)
        if encode_rate == 0 or decode_rate == 0:
            raise ValueError("codec rates must be greater than 0 for compressed methods")
        encode_ms = _ms(payload / (encode_rate * GB / second))
        decode_ms = _ms(compressed / (decode_rate * GB / second))
    baseline_step_ms = compute_ms + _ms(uncompressed_comm)
    method_step_ms = compute_ms + _ms(compressed_comm) + encode_ms + decode_ms
    reached = fixture["quality_target_reached"]
    if not isinstance(reached, bool):
        raise ValueError("quality_target_reached must be boolean")
    steps_to_target = math.ceil(target_steps * step_multiplier) if reached else None
    time_to_target_ms = method_step_ms * steps_to_target if steps_to_target is not None else None
    baseline_time_to_target_ms = baseline_step_ms * target_steps
    if not reached:
        outcome = "misses"
    elif time_to_target_ms < baseline_time_to_target_ms:
        outcome = "faster"
    else:
        outcome = "slower"
    baseline_inputs = {
        "track_id": track_id,
        "method": "none",
        "payload_mb": payload_value,
        "baseline_compute_ms": compute_ms,
        "baseline_target_steps": target_steps,
        "evidence": dict(_COMPRESSION_EVIDENCE["none"]),
    }
    inputs = {
        "track_id": track_id,
        "method": method,
        "payload_mb": payload_value,
        "baseline_compute_ms": compute_ms,
        "baseline_target_steps": target_steps,
        "evidence": fixture,
    }

    return {
        "inputs": inputs,
        "track_id": track_id,
        "method": method,
        "outcome": outcome,
        "payload_mb": payload_value,
        "compressed_payload_mb": _mb(compressed),
        "compression_ratio": ratio,
        "baseline_communication_ms": _ms(uncompressed_comm),
        "communication_ms": _ms(compressed_comm),
        "encode_ms": round(encode_ms, 6),
        "decode_ms": round(decode_ms, 6),
        "codec_ms": round(encode_ms + decode_ms, 6),
        "baseline_step_ms": round(baseline_step_ms, 6),
        "step_ms": round(method_step_ms, 6),
        "baseline_target_steps": target_steps,
        "steps_to_target": steps_to_target,
        "time_to_target_ms": round(time_to_target_ms, 6) if time_to_target_ms is not None else None,
        "quality_target_reached": reached,
        "quality_gap_pp": quality_gap,
        "error_feedback": bool(fixture["error_feedback"]),
        "evidence": fixture,
        "evidence_provenance": "Supplied illustrative convergence fixture; not a universal compression-quality law or live measurement.",
        "baseline": {
            "inputs": baseline_inputs,
            "communication_ms": _ms(uncompressed_comm),
            "step_ms": round(baseline_step_ms, 6),
            "steps_to_target": target_steps,
            "time_to_target_ms": round(baseline_time_to_target_ms, 6),
            "quality_target_reached": True,
        },
        "result": {
            "inputs": inputs,
            "method": method,
            "outcome": outcome,
            "communication_ms": _ms(compressed_comm),
            "encode_ms": round(encode_ms, 6),
            "decode_ms": round(decode_ms, 6),
            "step_ms": round(method_step_ms, 6),
            "steps_to_target": steps_to_target,
            "time_to_target_ms": round(time_to_target_ms, 6) if time_to_target_ms is not None else None,
            "quality_target_reached": reached,
            "quality_gap_pp": quality_gap,
        },
    }


__all__ = [
    "TRACK_IDS",
    "algorithm_cases",
    "algorithm_crossover",
    "compression_comparison",
    "overlap_timeline",
    "overlap_bucket_options",
    "routing_cases",
    "semantic_exchange",
    "topology_comparison",
    "track_profile",
]
