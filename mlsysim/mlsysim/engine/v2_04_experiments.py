"""Traceable storage-pipeline experiments for fleet-scale teaching labs.

The four profiles below are illustrative scenario assumptions.  They describe
real fleet shapes (device-to-gateway pipelines or a regional accelerator pool),
not measurements of branded products.  Every physical calculation is performed
with Pint quantities; callers may format the returned quantities for display.

This module deliberately stops at storage-path behavior.  It does not optimize
checkpoint intervals (a reliability-policy question) or infer model quality
from infrastructure choices.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any

from mlsysim.core.units import Q_


SCENARIO_ASSUMPTION = (
    "Illustrative teaching scenario; rates, prices, and workload sizes are not "
    "measurements of branded hardware or production fleets."
)
_QUANTITY_TYPE = type(Q_(1, "second"))


@dataclass(frozen=True)
class AccessEvent:
    """One accelerator-consumption request in an explicit access trace."""

    arrival: Any
    object_id: str
    payload: Any


def _q(value: Any, unit: str, name: str, *, positive: bool = False):
    try:
        quantity = value.to(unit)
    except AttributeError:
        quantity = Q_(value, unit)
    except Exception as exc:
        raise ValueError(f"{name} must be compatible with {unit}") from exc
    magnitude = quantity.magnitude
    if isinstance(magnitude, bool) or not math.isfinite(float(magnitude)):
        raise ValueError(f"{name} must be finite")
    if magnitude < 0 or (positive and magnitude == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return quantity


def _count(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


# Reusable teaching scenarios. Rates and prices are assumptions, not hardware
# specifications or vendor quotes. Device tracks train on a backend and ingest
# fleet data through gateways; no device is represented as a training worker.
TRACKS: dict[str, dict[str, Any]] = {
    "tinyml": {
        "display": "TinyML",
        "fleet_shape": "sensor fleet -> site gateways -> regional training backend",
        "consumer_unit": "backend accelerator",
        "consumers": 16,
        "samples_per_batch": 256,
        "sample_bytes": Q_(48, "kilobyte"),
        "step_time": Q_(180, "millisecond"),
        "storage_bandwidth": Q_(1.8, "gigabyte/second"),
        "metadata_rate": Q_(900, "count/second"),
        "preprocess_rate": Q_(30_000, "count/second"),
        "accelerator_rate": Q_(1.10, "dollar/hour"),
        "layout_samples": 120_000,
        "checkpoint_bytes": Q_(48, "gigabyte"),
        "local_checkpoint_bandwidth": Q_(3.2, "gigabyte/second"),
        "durable_checkpoint_bandwidth": Q_(0.8, "gigabyte/second"),
    },
    "mobile": {
        "display": "Mobile",
        "fleet_shape": "phone fleet -> regional upload service -> training pool",
        "consumer_unit": "backend accelerator",
        "consumers": 32,
        "samples_per_batch": 192,
        "sample_bytes": Q_(180, "kilobyte"),
        "step_time": Q_(220, "millisecond"),
        "storage_bandwidth": Q_(8, "gigabyte/second"),
        "metadata_rate": Q_(2_000, "count/second"),
        "preprocess_rate": Q_(45_000, "count/second"),
        "accelerator_rate": Q_(1.80, "dollar/hour"),
        "layout_samples": 180_000,
        "checkpoint_bytes": Q_(120, "gigabyte"),
        "local_checkpoint_bandwidth": Q_(5.0, "gigabyte/second"),
        "durable_checkpoint_bandwidth": Q_(1.5, "gigabyte/second"),
    },
    "edge": {
        "display": "Edge",
        "fleet_shape": "camera fleet -> site gateways -> regional replay and training pool",
        "consumer_unit": "backend accelerator",
        "consumers": 64,
        "samples_per_batch": 128,
        "sample_bytes": Q_(420, "kilobyte"),
        "step_time": Q_(240, "millisecond"),
        "storage_bandwidth": Q_(18, "gigabyte/second"),
        "metadata_rate": Q_(4_000, "count/second"),
        "preprocess_rate": Q_(58_000, "count/second"),
        "accelerator_rate": Q_(2.40, "dollar/hour"),
        "layout_samples": 240_000,
        "checkpoint_bytes": Q_(320, "gigabyte"),
        "local_checkpoint_bandwidth": Q_(7.0, "gigabyte/second"),
        "durable_checkpoint_bandwidth": Q_(2.5, "gigabyte/second"),
    },
    "cloud": {
        "display": "Cloud",
        "fleet_shape": "regional object store -> preprocessing service -> accelerator pool",
        "consumer_unit": "training accelerator",
        "consumers": 128,
        "samples_per_batch": 96,
        "sample_bytes": Q_(760, "kilobyte"),
        "step_time": Q_(260, "millisecond"),
        "storage_bandwidth": Q_(42, "gigabyte/second"),
        "metadata_rate": Q_(8_000, "count/second"),
        "preprocess_rate": Q_(82_000, "count/second"),
        "accelerator_rate": Q_(3.20, "dollar/hour"),
        "layout_samples": 320_000,
        "checkpoint_bytes": Q_(800, "gigabyte"),
        "local_checkpoint_bandwidth": Q_(12, "gigabyte/second"),
        "durable_checkpoint_bandwidth": Q_(5, "gigabyte/second"),
    },
}

TIER_PLANS: dict[str, dict[str, Any]] = {
    "economy": {
        "label": "Capacity-first",
        "storage_scale": 0.55,
        "metadata_scale": 0.65,
        "preprocess_scale": 0.80,
        "samples_per_object": 1,
        "request_price": Q_(0.004, "dollar"),
        "movement_price": Q_(0.006, "dollar/gigabyte"),
        "capacity_price": Q_(0.012, "dollar/gigabyte/month"),
    },
    "balanced": {
        "label": "Sharded regional tier",
        "storage_scale": 1.15,
        "metadata_scale": 1.20,
        "preprocess_scale": 1.15,
        "samples_per_object": 64,
        "request_price": Q_(0.006, "dollar"),
        "movement_price": Q_(0.016, "dollar/gigabyte"),
        "capacity_price": Q_(0.026, "dollar/gigabyte/month"),
    },
    "performance": {
        "label": "Performance tier",
        "storage_scale": 2.20,
        "metadata_scale": 1.80,
        "preprocess_scale": 1.55,
        "samples_per_object": 256,
        "request_price": Q_(0.010, "dollar"),
        "movement_price": Q_(0.045, "dollar/gigabyte"),
        "capacity_price": Q_(0.065, "dollar/gigabyte/month"),
    },
}


def track_profile(track_id: str) -> dict[str, Any]:
    """Return a copy of one illustrative fleet profile."""
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}")
    profile = dict(TRACKS[track_id])
    profile["scenario_assumption"] = SCENARIO_ASSUMPTION
    return profile


def serialize_for_evidence(value: Any) -> Any:
    """Convert experiment inputs/results into immutable JSON-compatible data.

    Quantities carry an explicit type tag, magnitude, and unit so a capstone can
    reconstruct evaluator arguments without guessing units from field names.
    """
    if isinstance(value, _QUANTITY_TYPE):
        if not math.isfinite(float(value.magnitude)):
            raise ValueError("cannot serialize a nonfinite quantity")
        return {
            "__type__": "quantity",
            "magnitude": value.magnitude,
            "unit": f"{value.units:~}",
        }
    if isinstance(value, AccessEvent):
        return {
            "__type__": "access_event",
            "arrival": serialize_for_evidence(value.arrival),
            "object_id": value.object_id,
            "payload": serialize_for_evidence(value.payload),
        }
    if isinstance(value, Mapping):
        return {str(key): serialize_for_evidence(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize_for_evidence(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("cannot serialize a nonfinite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize evidence value of type {type(value).__name__}")


def deserialize_evidence_inputs(value: Any) -> Any:
    """Reconstruct quantities and access events from :func:`serialize_for_evidence`."""
    if isinstance(value, list):
        return [deserialize_evidence_inputs(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    value_type = value.get("__type__")
    if value_type == "quantity":
        if set(value) != {"__type__", "magnitude", "unit"}:
            raise ValueError("quantity evidence has unexpected fields")
        return Q_(value["magnitude"], value["unit"])
    if value_type == "access_event":
        if set(value) != {"__type__", "arrival", "object_id", "payload"}:
            raise ValueError("access-event evidence has unexpected fields")
        return AccessEvent(
            deserialize_evidence_inputs(value["arrival"]),
            str(value["object_id"]),
            deserialize_evidence_inputs(value["payload"]),
        )
    if value_type is not None:
        raise ValueError(f"unknown evidence type tag: {value_type}")
    return {str(key): deserialize_evidence_inputs(item) for key, item in value.items()}


def evaluate_delivery(
    track_id: str,
    *,
    consumers: int | None = None,
    target_utilization: float = 1.0,
    files_per_batch: int = 1,
    storage_scale: float = 1.0,
    metadata_scale: float = 1.0,
    preprocess_scale: float = 1.0,
) -> dict[str, Any]:
    """Evaluate demand against storage, metadata, and preprocessing stages.

    Each consumer needs one batch per step. The pipeline's delivered batch rate
    is the minimum service rate across its three modeled stages.
    """
    profile = track_profile(track_id)
    consumer_count = profile["consumers"] if consumers is None else _count(consumers, "consumers", positive=True)
    if (
        isinstance(target_utilization, bool)
        or not isinstance(target_utilization, (int, float))
        or not math.isfinite(target_utilization)
        or not 0 < target_utilization <= 1
    ):
        raise ValueError("target_utilization must be a number greater than 0 and at most 1")
    files_per_batch = _count(files_per_batch, "files_per_batch", positive=True)
    scales = {
        "storage": storage_scale,
        "metadata": metadata_scale,
        "preprocess": preprocess_scale,
    }
    for name, value in scales.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}_scale must be a positive finite number")

    batch_bytes = (profile["sample_bytes"] * profile["samples_per_batch"]).to("byte")
    required_batches = (
        Q_(consumer_count * target_utilization, "count") / profile["step_time"]
    ).to("count/second")
    required_bandwidth = (required_batches * batch_bytes / Q_(1, "count")).to("gigabyte/second")
    required_requests = (required_batches * files_per_batch).to("count/second")
    required_samples = (required_batches * profile["samples_per_batch"]).to("count/second")

    storage_bandwidth = profile["storage_bandwidth"] * storage_scale
    metadata_rate = profile["metadata_rate"] * metadata_scale
    preprocess_rate = profile["preprocess_rate"] * preprocess_scale
    service_rates = {
        "storage": (storage_bandwidth / batch_bytes * Q_(1, "count")).to("count/second"),
        "metadata": (metadata_rate / files_per_batch).to("count/second"),
        "preprocess": (preprocess_rate / profile["samples_per_batch"]).to("count/second"),
    }
    bottleneck = min(service_rates, key=lambda name: service_rates[name].magnitude)
    delivered = service_rates[bottleneck]
    utilization = min(1.0, (delivered / required_batches).to("").magnitude)
    return {
        "inputs": {
            "track_id": track_id,
            "consumers": consumer_count,
            "target_utilization": float(target_utilization),
            "files_per_batch": files_per_batch,
            "storage_scale": float(storage_scale),
            "metadata_scale": float(metadata_scale),
            "preprocess_scale": float(preprocess_scale),
        },
        "track_id": track_id,
        "fleet_shape": profile["fleet_shape"],
        "consumer_unit": profile["consumer_unit"],
        "consumers": consumer_count,
        "target_utilization": float(target_utilization),
        "batch_bytes": batch_bytes,
        "required_batches_per_second": required_batches,
        "required_bandwidth": required_bandwidth,
        "required_requests_per_second": required_requests,
        "required_samples_per_second": required_samples,
        "storage_bandwidth": storage_bandwidth.to("gigabyte/second"),
        "metadata_rate": metadata_rate.to("count/second"),
        "preprocess_rate": preprocess_rate.to("count/second"),
        "service_rates": service_rates,
        "delivered_batches_per_second": delivered,
        "bottleneck": bottleneck,
        "accelerator_utilization": utilization,
        "starved": utilization < 1.0,
    }


def evaluate_track_demand(
    track_id: str,
    *,
    consumer_scale: float,
    target_utilization: float = 1.0,
) -> dict[str, Any]:
    """Scale the active consumers in one fleet without notebook-side arithmetic."""
    if isinstance(consumer_scale, bool) or not isinstance(consumer_scale, (int, float)):
        raise ValueError("consumer_scale must be a positive finite number")
    if not math.isfinite(consumer_scale) or consumer_scale <= 0:
        raise ValueError("consumer_scale must be a positive finite number")
    profile = track_profile(track_id)
    consumers = math.ceil(profile["consumers"] * consumer_scale)
    result = evaluate_delivery(
        track_id,
        consumers=consumers,
        target_utilization=target_utilization,
    )
    result["inputs"] = {
        "track_id": track_id,
        "consumer_scale": float(consumer_scale),
        "target_utilization": float(target_utilization),
    }
    return result


def make_access_trace(
    *,
    accesses: int = 24,
    spacing=Q_(12, "millisecond"),
    payload=Q_(16, "megabyte"),
    burst_start: int = 8,
    burst_length: int = 5,
    burst_spacing=Q_(1, "millisecond"),
    working_set: int = 6,
) -> tuple[AccessEvent, ...]:
    """Build an explicit repeatable trace with a correlated arrival burst."""
    accesses = _count(accesses, "accesses", positive=True)
    burst_start = _count(burst_start, "burst_start")
    burst_length = _count(burst_length, "burst_length")
    working_set = _count(working_set, "working_set", positive=True)
    spacing = _q(spacing, "second", "spacing", positive=True)
    burst_spacing = _q(burst_spacing, "second", "burst_spacing", positive=True)
    payload = _q(payload, "byte", "payload", positive=True)
    if burst_start > accesses or burst_start + burst_length > accesses:
        raise ValueError("burst interval must fit inside the access trace")

    now = Q_(0, "second")
    events = []
    for index in range(accesses):
        events.append(AccessEvent(now, f"object-{index % working_set}", payload))
        now += burst_spacing if burst_start <= index < burst_start + burst_length else spacing
    return tuple(events)


def make_miss_burst_trace(
    *,
    cold_spacing=Q_(1, "millisecond"),
    payload=Q_(8, "megabyte"),
) -> tuple[AccessEvent, ...]:
    """Build a high-hit trace with five cold objects arriving together.

    The same object is requested before and after the cold segment. Changing
    ``cold_spacing`` changes miss correlation without changing object order,
    miss count, payload, or total request count.
    """
    cold_spacing = _q(cold_spacing, "second", "cold_spacing", positive=True)
    payload = _q(payload, "byte", "payload", positive=True)
    events = [AccessEvent(Q_(index * 20, "millisecond"), "hot", payload) for index in range(12)]
    cold_start = Q_(240, "millisecond")
    events.extend(
        AccessEvent(cold_start + offset * cold_spacing, f"cold-{offset}", payload)
        for offset in range(5)
    )
    events.extend(
        AccessEvent(Q_(420 + index * 20, "millisecond"), "hot", payload)
        for index in range(20)
    )
    return tuple(events)


def simulate_cache_prefetch(
    events: Iterable[AccessEvent],
    *,
    cache_capacity=Q_(64, "megabyte"),
    prefetch_depth: int = 2,
    storage_bandwidth=Q_(1.5, "gigabyte/second"),
    miss_latency=Q_(4, "millisecond"),
    hit_latency=Q_(0.15, "millisecond"),
) -> dict[str, Any]:
    """Replay explicit accesses through an LRU cache and finite prefetch queue.

    Storage transfers serialize. Prefetch knows only the next ``prefetch_depth``
    trace entries. A prefetched object that is unfinished at arrival still makes
    the consumer wait, which exposes correlated-miss bursts.
    """
    trace = tuple(events)
    if not trace:
        raise ValueError("events must contain at least one access")
    capacity = _q(cache_capacity, "byte", "cache_capacity", positive=True)
    bandwidth = _q(storage_bandwidth, "byte/second", "storage_bandwidth", positive=True)
    miss_latency = _q(miss_latency, "second", "miss_latency")
    hit_latency = _q(hit_latency, "second", "hit_latency")
    prefetch_depth = _count(prefetch_depth, "prefetch_depth")

    normalized = []
    previous = None
    for event in trace:
        arrival = _q(event.arrival, "second", "event arrival")
        payload = _q(event.payload, "byte", "event payload", positive=True)
        if previous is not None and arrival < previous:
            raise ValueError("event arrivals must be nondecreasing")
        previous = arrival
        normalized.append(AccessEvent(arrival, str(event.object_id), payload))

    cache: OrderedDict[str, Any] = OrderedDict()
    cache_bytes = Q_(0, "byte")
    pending: dict[int, Any] = {}
    storage_available = Q_(0, "second")
    rows = []
    hit_count = 0
    prefetched_count = 0
    total_stall = Q_(0, "second")

    def insert(object_id: str, size):
        nonlocal cache_bytes
        if size > capacity:
            return
        if object_id in cache:
            cache_bytes -= cache.pop(object_id)
        cache[object_id] = size
        cache_bytes += size
        while cache_bytes > capacity:
            _, evicted_size = cache.popitem(last=False)
            cache_bytes -= evicted_size

    def schedule(index: int, known_at):
        nonlocal storage_available
        if index in pending:
            return
        event = normalized[index]
        if event.object_id in cache or any(
            normalized[pending_index].object_id == event.object_id
            for pending_index in pending
        ):
            return
        start = max(storage_available, known_at)
        ready = start + miss_latency + event.payload / bandwidth
        pending[index] = ready.to("second")
        storage_available = ready.to("second")

    for index in range(min(prefetch_depth, len(normalized))):
        schedule(index, Q_(0, "second"))

    for index, event in enumerate(normalized):
        if event.object_id in cache:
            source = "cache_hit"
            hit_count += 1
            cache.move_to_end(event.object_id)
            completion = event.arrival + hit_latency
        elif index in pending:
            source = "prefetch"
            prefetched_count += 1
            completion = max(event.arrival + hit_latency, pending[index])
        else:
            source = "demand_miss"
            start = max(event.arrival, storage_available)
            completion = start + miss_latency + event.payload / bandwidth
            storage_available = completion.to("second")
        pending.pop(index, None)
        stall = max(Q_(0, "second"), completion - event.arrival)
        total_stall += stall
        insert(event.object_id, event.payload)
        rows.append(
            {
                "index": index,
                "arrival": event.arrival.to("millisecond"),
                "object_id": event.object_id,
                "source": source,
                "completion": completion.to("millisecond"),
                "stall": stall.to("millisecond"),
            }
        )
        horizon = index + prefetch_depth
        for future in range(index + 1, min(horizon + 1, len(normalized))):
            schedule(future, completion)

    misses = len(trace) - hit_count
    return {
        "inputs": {
            "events": trace,
            "cache_capacity": capacity,
            "prefetch_depth": prefetch_depth,
            "storage_bandwidth": bandwidth,
            "miss_latency": miss_latency,
            "hit_latency": hit_latency,
        },
        "events": tuple(rows),
        "access_count": len(trace),
        "cache_hits": hit_count,
        "cache_misses": misses,
        "prefetched_accesses": prefetched_count,
        "hit_rate": hit_count / len(trace),
        "total_stall": total_stall.to("millisecond"),
        "mean_stall": (total_stall / len(trace)).to("millisecond"),
        "max_stall": max(row["stall"] for row in rows),
        "prefetch_depth": prefetch_depth,
    }


def evaluate_layout(
    *,
    samples: int,
    sample_bytes,
    samples_per_object: int,
    storage_bandwidth,
    metadata_rate,
    preprocess_rate,
    request_overhead=Q_(4, "kilobyte"),
    minimum_transfer=Q_(64, "kilobyte"),
) -> dict[str, Any]:
    """Evaluate request count, transferred bytes, and stage bottlenecks for a layout."""
    samples = _count(samples, "samples", positive=True)
    samples_per_object = _count(samples_per_object, "samples_per_object", positive=True)
    sample_bytes = _q(sample_bytes, "byte", "sample_bytes", positive=True)
    bandwidth = _q(storage_bandwidth, "byte/second", "storage_bandwidth", positive=True)
    metadata_rate = _q(metadata_rate, "count/second", "metadata_rate", positive=True)
    preprocess_rate = _q(preprocess_rate, "count/second", "preprocess_rate", positive=True)
    request_overhead = _q(request_overhead, "byte", "request_overhead")
    minimum_transfer = _q(minimum_transfer, "byte", "minimum_transfer")

    request_count = math.ceil(samples / samples_per_object)
    useful_bytes = (samples * sample_bytes).to("byte")
    object_payload = samples_per_object * sample_bytes
    full_objects, remainder = divmod(samples, samples_per_object)
    transferred = full_objects * max(object_payload, minimum_transfer)
    if remainder:
        # The final range fetch reads the complete object. Very large shards
        # therefore trade fewer metadata operations for padding past the final
        # useful sample in this bounded dataset slice.
        transferred += max(object_payload, minimum_transfer)
    transferred = (transferred + request_count * request_overhead).to("byte")
    times = {
        "storage": (transferred / bandwidth).to("second"),
        "metadata": (Q_(request_count, "count") / metadata_rate).to("second"),
        "preprocess": (Q_(samples, "count") / preprocess_rate).to("second"),
    }
    bottleneck = max(times, key=lambda name: times[name].magnitude)
    elapsed = times[bottleneck]
    return {
        "inputs": {
            "samples": samples,
            "sample_bytes": sample_bytes,
            "samples_per_object": samples_per_object,
            "storage_bandwidth": bandwidth,
            "metadata_rate": metadata_rate,
            "preprocess_rate": preprocess_rate,
            "request_overhead": request_overhead,
            "minimum_transfer": minimum_transfer,
        },
        "samples": samples,
        "samples_per_object": samples_per_object,
        "request_count": request_count,
        "useful_bytes": useful_bytes,
        "transferred_bytes": transferred,
        "transfer_amplification": (transferred / useful_bytes).to("").magnitude,
        "stage_times": times,
        "elapsed": elapsed,
        "effective_sample_rate": (Q_(samples, "count") / elapsed).to("count/second"),
        "bottleneck": bottleneck,
    }


def simulate_checkpoint_publication(
    *,
    checkpoint_bytes,
    local_bandwidth,
    durable_bandwidth,
    failure_time=None,
) -> dict[str, Any]:
    """Model local checkpoint creation followed by atomic durable publication."""
    size = _q(checkpoint_bytes, "byte", "checkpoint_bytes", positive=True)
    local_bandwidth = _q(local_bandwidth, "byte/second", "local_bandwidth", positive=True)
    durable_bandwidth = _q(durable_bandwidth, "byte/second", "durable_bandwidth", positive=True)
    local_complete = (size / local_bandwidth).to("second")
    durable_complete = (local_complete + size / durable_bandwidth).to("second")
    failure = None if failure_time is None else _q(failure_time, "second", "failure_time")
    recoverable = failure is None or failure >= durable_complete
    if failure is None:
        stage = "complete"
    elif failure < local_complete:
        stage = "local_copy"
    elif failure < durable_complete:
        stage = "durable_publication"
    else:
        stage = "after_publication"
    return {
        "inputs": {
            "checkpoint_bytes": size,
            "local_bandwidth": local_bandwidth,
            "durable_bandwidth": durable_bandwidth,
            "failure_time": failure,
        },
        "checkpoint_bytes": size,
        "local_duration": local_complete,
        "durable_duration": (durable_complete - local_complete).to("second"),
        "local_complete": local_complete,
        "durable_complete": durable_complete,
        "failure_time": failure,
        "failure_stage": stage,
        "recoverable": recoverable,
        "restorable_bytes": size if recoverable else Q_(0, "byte"),
    }


def evaluate_storage_cost(
    *,
    duration,
    accelerator_count: int,
    accelerator_rate,
    accelerator_utilization: float,
    request_count: int,
    request_price_per_thousand,
    moved_bytes,
    movement_price_per_gb,
    stored_bytes=Q_(0, "gigabyte"),
    capacity_price_per_gb_month=Q_(0, "dollar/gigabyte/month"),
) -> dict[str, Any]:
    """Cost accelerator time, idle time, requests, movement, and capacity."""
    duration = _q(duration, "hour", "duration", positive=True)
    accelerator_count = _count(accelerator_count, "accelerator_count", positive=True)
    accelerator_rate = _q(accelerator_rate, "dollar/hour", "accelerator_rate")
    if isinstance(accelerator_utilization, bool) or not isinstance(accelerator_utilization, (int, float)):
        raise ValueError("accelerator_utilization must be a number from 0 to 1")
    if not math.isfinite(accelerator_utilization) or not 0 <= accelerator_utilization <= 1:
        raise ValueError("accelerator_utilization must be a number from 0 to 1")
    request_count = _count(request_count, "request_count")
    request_price = _q(request_price_per_thousand, "dollar", "request_price_per_thousand")
    moved = _q(moved_bytes, "gigabyte", "moved_bytes")
    movement_price = _q(movement_price_per_gb, "dollar/gigabyte", "movement_price_per_gb")
    stored = _q(stored_bytes, "gigabyte", "stored_bytes")
    capacity_price = _q(capacity_price_per_gb_month, "dollar/gigabyte/month", "capacity_price_per_gb_month")

    accelerator_cost = (duration * accelerator_count * accelerator_rate).to("dollar")
    idle_cost = (accelerator_cost * (1.0 - accelerator_utilization)).to("dollar")
    request_cost = (request_count / 1000 * request_price).to("dollar")
    movement_cost = (moved * movement_price).to("dollar")
    capacity_cost = (stored * capacity_price * duration.to("month")).to("dollar")
    total = accelerator_cost + request_cost + movement_cost + capacity_cost
    return {
        "inputs": {
            "duration": duration,
            "accelerator_count": accelerator_count,
            "accelerator_rate": accelerator_rate,
            "accelerator_utilization": float(accelerator_utilization),
            "request_count": request_count,
            "request_price_per_thousand": request_price,
            "moved_bytes": moved,
            "movement_price_per_gb": movement_price,
            "stored_bytes": stored,
            "capacity_price_per_gb_month": capacity_price,
        },
        "accelerator_cost": accelerator_cost,
        "idle_accelerator_cost": idle_cost,
        "request_cost": request_cost,
        "movement_cost": movement_cost,
        "capacity_cost": capacity_cost,
        "total_cost": total.to("dollar"),
        "accelerator_utilization": float(accelerator_utilization),
    }


def evaluate_track_layout(
    track_id: str,
    *,
    samples_per_object: int,
    storage_scale: float = 1.0,
) -> dict[str, Any]:
    """Evaluate a layout using the selected fleet's ingestion assumptions."""
    profile = track_profile(track_id)
    if isinstance(storage_scale, bool) or not isinstance(storage_scale, (int, float)):
        raise ValueError("storage_scale must be a positive finite number")
    if not math.isfinite(storage_scale) or storage_scale <= 0:
        raise ValueError("storage_scale must be a positive finite number")
    result = evaluate_layout(
        samples=profile["layout_samples"],
        sample_bytes=profile["sample_bytes"],
        samples_per_object=samples_per_object,
        storage_bandwidth=profile["storage_bandwidth"] * storage_scale,
        metadata_rate=profile["metadata_rate"],
        preprocess_rate=profile["preprocess_rate"],
    )
    result["inputs"] = {
        "track_id": track_id,
        "samples_per_object": samples_per_object,
        "storage_scale": float(storage_scale),
    }
    result["track_id"] = track_id
    return result


def evaluate_track_cache(
    track_id: str,
    *,
    cold_spacing=Q_(1, "millisecond"),
    prefetch_depth: int = 1,
    cache_capacity: Any = None,
) -> dict[str, Any]:
    """Simulate caching and prefetch under the selected fleet's storage bandwidth and batch demand.

    Illustrative scenario assumptions: cache holds 4 batches by default so cold
    bursts demonstrate cache pollution against active working sets. Hit and miss
    latencies follow shared latency assumptions.
    """
    profile = track_profile(track_id)
    cold_spacing = _q(cold_spacing, "second", "cold_spacing", positive=True)
    prefetch_depth = _count(prefetch_depth, "prefetch_depth")
    batch_bytes = (profile["sample_bytes"] * profile["samples_per_batch"]).to("byte")
    capacity = (
        (batch_bytes * 4).to("byte")
        if cache_capacity is None
        else _q(cache_capacity, "byte", "cache_capacity", positive=True)
    )
    bandwidth = profile["storage_bandwidth"]
    trace = make_miss_burst_trace(cold_spacing=cold_spacing, payload=batch_bytes)
    result = simulate_cache_prefetch(
        trace,
        cache_capacity=capacity,
        prefetch_depth=prefetch_depth,
        storage_bandwidth=bandwidth,
    )
    result["inputs"] = {
        "track_id": track_id,
        "cold_spacing": cold_spacing,
        "prefetch_depth": prefetch_depth,
        "cache_capacity": capacity,
    }
    result["track_id"] = track_id
    result["consumer_unit"] = profile["consumer_unit"]
    result["fleet_shape"] = profile["fleet_shape"]
    result["batch_bytes"] = batch_bytes
    result["storage_bandwidth"] = bandwidth
    return result


def evaluate_track_checkpoint(track_id: str, *, failure_stage: str) -> dict[str, Any]:
    """Inject a failure into one named checkpoint-publication stage."""
    if failure_stage not in {"local_copy", "durable_publication", "after_publication"}:
        raise ValueError(
            "failure_stage must be local_copy, durable_publication, or after_publication"
        )
    profile = track_profile(track_id)
    complete = simulate_checkpoint_publication(
        checkpoint_bytes=profile["checkpoint_bytes"],
        local_bandwidth=profile["local_checkpoint_bandwidth"],
        durable_bandwidth=profile["durable_checkpoint_bandwidth"],
    )
    if failure_stage == "local_copy":
        failure_time = complete["local_complete"] * 0.5
    elif failure_stage == "durable_publication":
        failure_time = (
            complete["local_complete"]
            + (complete["durable_complete"] - complete["local_complete"]) * 0.5
        )
    else:
        failure_time = complete["durable_complete"] + Q_(1, "second")
    result = simulate_checkpoint_publication(
        checkpoint_bytes=profile["checkpoint_bytes"],
        local_bandwidth=profile["local_checkpoint_bandwidth"],
        durable_bandwidth=profile["durable_checkpoint_bandwidth"],
        failure_time=failure_time,
    )
    result["inputs"] = {"track_id": track_id, "failure_stage": failure_stage}
    result["track_id"] = track_id
    return result


def evaluate_tier_plan(track_id: str, *, tier_id: str) -> dict[str, Any]:
    """Evaluate one complete tier placement under a fixed demanding job."""
    if tier_id not in TIER_PLANS:
        raise ValueError(f"tier_id must be one of {', '.join(TIER_PLANS)}")
    profile = track_profile(track_id)
    plan = TIER_PLANS[tier_id]
    consumer_count = math.ceil(profile["consumers"] * 1.8)
    files_per_batch = math.ceil(profile["samples_per_batch"] / plan["samples_per_object"])
    delivery = evaluate_delivery(
        track_id,
        consumers=consumer_count,
        files_per_batch=files_per_batch,
        storage_scale=plan["storage_scale"],
        metadata_scale=plan["metadata_scale"],
        preprocess_scale=plan["preprocess_scale"],
    )
    layout = evaluate_track_layout(
        track_id,
        samples_per_object=plan["samples_per_object"],
        storage_scale=plan["storage_scale"],
    )
    useful_duration = Q_(12, "hour")
    job_duration = (useful_duration / delivery["accelerator_utilization"]).to("hour")
    cost = evaluate_storage_cost(
        duration=job_duration,
        accelerator_count=consumer_count,
        accelerator_rate=profile["accelerator_rate"],
        accelerator_utilization=delivery["accelerator_utilization"],
        request_count=layout["request_count"],
        request_price_per_thousand=plan["request_price"],
        moved_bytes=layout["transferred_bytes"],
        movement_price_per_gb=plan["movement_price"],
        stored_bytes=layout["useful_bytes"],
        capacity_price_per_gb_month=plan["capacity_price"],
    )
    return {
        "inputs": {"track_id": track_id, "tier_id": tier_id},
        "track_id": track_id,
        "tier_id": tier_id,
        "label": plan["label"],
        "delivery": delivery,
        "layout": layout,
        "cost": cost,
        "job_duration": job_duration,
        "accelerator_utilization": delivery["accelerator_utilization"],
        "bottleneck": delivery["bottleneck"],
        "total_cost": cost["total_cost"],
        "idle_accelerator_cost": cost["idle_accelerator_cost"],
        "movement_cost": cost["movement_cost"],
        "request_cost": cost["request_cost"],
        "capacity_cost": cost["capacity_cost"],
    }


__all__ = [
    "AccessEvent",
    "SCENARIO_ASSUMPTION",
    "TIER_PLANS",
    "TRACKS",
    "deserialize_evidence_inputs",
    "evaluate_delivery",
    "evaluate_layout",
    "evaluate_storage_cost",
    "evaluate_tier_plan",
    "evaluate_track_cache",
    "evaluate_track_checkpoint",
    "evaluate_track_demand",
    "evaluate_track_layout",
    "make_access_trace",
    "make_miss_burst_trace",
    "serialize_for_evidence",
    "simulate_cache_prefetch",
    "simulate_checkpoint_publication",
    "track_profile",
]
