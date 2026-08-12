"""Shared model-serving helpers for track-aware tail-latency labs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class ServingTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    model_params_b: float
    memory_capacity_gb: float
    memory_bandwidth_gbs: float
    storage_bandwidth_gbs: float
    interconnect_bandwidth_gbs: float
    tdp_w: float
    battery_wh: float | None
    arrival_qps: float
    service_ms: float
    replicas: int
    slo_ms: float
    service_cv: float
    batch_size: int
    batch_efficiency_gain: float
    context_tokens: int
    state_kind: str
    state_mb_per_request: float | None
    precision_bytes: float
    kv_precision_bytes: float
    default_devices_per_replica: int
    deserialize_gbs: float
    runtime_init_ms: float
    warmup_ms: float
    warm_pool_replicas: int
    scale_out_replicas: int
    validation_tests: tuple[str, ...]
    serving_policy: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class QueueingResult:
    arrival_qps: float
    service_ms: float
    replicas: int
    utilization: float
    stable: bool
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    p999_latency_ms: float
    slo_ms: float
    slo_ok: bool
    queue_wait_probability: float
    queue_amplifier: float


@dataclass(frozen=True)
class BatchingTaxResult:
    batch_size: int
    arrival_qps: float
    formation_delay_ms: float
    batched_service_ms: float
    queue_p99_ms: float
    total_p99_ms: float
    formation_slo_pct: float
    throughput_gain: float
    effective_batch_qps: float
    utilization: float
    slo_ok: bool


@dataclass(frozen=True)
class CacheCapacityResult:
    total_memory_gb: float
    weight_gb: float
    state_per_request_gb: float
    available_gb: float
    max_concurrent: int
    oom: bool
    state_kind: str
    context_tokens: int


@dataclass(frozen=True)
class ColdStartResult:
    model_weight_gb: float
    storage_bandwidth_gbs: float
    interconnect_bandwidth_gbs: float
    effective_bandwidth_gbs: float
    read_ms: float
    deserialize_ms: float
    runtime_init_ms: float
    warmup_ms: float
    cold_start_ms: float
    exposed_first_request_ms: float
    protected_fraction: float
    exceeds_slo: bool


def _quantity_to_float(value: Any, unit: str, default: float) -> float:
    if value is None:
        return default
    if hasattr(value, "m_as"):
        try:
            return float(value.m_as(unit))
        except Exception:
            return default
    if hasattr(value, "to"):
        try:
            return float(value.to(unit).magnitude)
        except Exception:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _model_params_b(model: Any) -> float:
    return _quantity_to_float(getattr(model, "parameters", None), "param", 0.0) / 1e9


def _hardware_quantity(hardware: Any, path: str, unit: str, default: float) -> float:
    current = hardware
    for attr in path.split("."):
        current = getattr(current, attr, None)
        if current is None:
            return default
    return _quantity_to_float(current, unit, default)


def serving_track_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> ServingTrackProfile:
    """Build a source-traced serving profile from MLSysIM refs and variant defaults."""
    defaults = variant.defaults
    storage_default = _hardware_quantity(hardware, "memory.bandwidth", "GB/s", 1.0)
    return ServingTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        model_params_b=_model_params_b(model),
        memory_capacity_gb=_hardware_quantity(hardware, "memory.capacity", "GB", 0.0),
        memory_bandwidth_gbs=_hardware_quantity(hardware, "memory.bandwidth", "GB/s", 0.0),
        storage_bandwidth_gbs=float(
            defaults.get(
                "storage_bandwidth_gbs",
                _hardware_quantity(hardware, "storage.bandwidth", "GB/s", storage_default),
            )
        ),
        interconnect_bandwidth_gbs=float(
            defaults.get(
                "interconnect_bandwidth_gbs",
                _hardware_quantity(hardware, "interconnect.bandwidth", "GB/s", storage_default),
            )
        ),
        tdp_w=_quantity_to_float(getattr(hardware, "tdp", None), "W", 0.0),
        battery_wh=(
            _quantity_to_float(getattr(hardware, "battery_capacity", None), "Wh", 0.0)
            if getattr(hardware, "battery_capacity", None) is not None
            else None
        ),
        arrival_qps=float(defaults.get("arrival_qps", 100.0)),
        service_ms=float(defaults.get("service_ms", 20.0)),
        replicas=int(defaults.get("replicas", 1)),
        slo_ms=float(defaults.get("slo_ms", 100.0)),
        service_cv=float(defaults.get("service_cv", 1.0)),
        batch_size=int(defaults.get("batch_size", 1)),
        batch_efficiency_gain=float(defaults.get("batch_efficiency_gain", 1.0)),
        context_tokens=int(defaults.get("context_tokens", 4096)),
        state_kind=str(defaults.get("state_kind", "state/cache")),
        state_mb_per_request=(
            float(defaults["state_mb_per_request"])
            if "state_mb_per_request" in defaults
            else None
        ),
        precision_bytes=float(defaults.get("precision_bytes", 2.0)),
        kv_precision_bytes=float(defaults.get("kv_precision_bytes", 2.0)),
        default_devices_per_replica=int(defaults.get("devices_per_replica", 1)),
        deserialize_gbs=float(defaults.get("deserialize_gbs", 20.0)),
        runtime_init_ms=float(defaults.get("runtime_init_ms", 800.0)),
        warmup_ms=float(defaults.get("warmup_ms", 500.0)),
        warm_pool_replicas=int(defaults.get("warm_pool_replicas", 0)),
        scale_out_replicas=int(defaults.get("scale_out_replicas", 1)),
        validation_tests=tuple(str(item) for item in defaults.get("validation_tests", ())),
        serving_policy=str(variant.assumptions.get("serving_policy", "track-specific serving policy")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=(variant.hardware_ref, variant.model_ref),
    )


def _erlang_c(arrival_qps: float, service_ms: float, replicas: int) -> tuple[float, float]:
    if service_ms <= 0 or replicas <= 0:
        return math.inf, 1.0
    service_rate = 1000.0 / service_ms
    offered_load = arrival_qps / service_rate
    if offered_load >= replicas:
        return math.inf, 1.0

    terms = [offered_load**n / math.factorial(n) for n in range(replicas)]
    tail = (
        offered_load**replicas
        / math.factorial(replicas)
        * replicas
        / (replicas - offered_load)
    )
    wait_probability = tail / (sum(terms) + tail)
    wait_rate = replicas * service_rate - arrival_qps
    return wait_rate, wait_probability


def queueing_latency(
    *,
    arrival_qps: float,
    service_ms: float,
    replicas: int = 1,
    service_cv: float = 1.0,
    slo_ms: float = 100.0,
) -> QueueingResult:
    """Approximate M/M/c tail latency with a service-variability adjustment."""
    replicas = max(1, int(replicas))
    service_rate = 1000.0 / service_ms if service_ms > 0 else 0.0
    capacity = replicas * service_rate
    utilization = arrival_qps / capacity if capacity > 0 else math.inf
    stable = utilization < 1.0
    variability = max(0.1, (1 + service_cv**2) / 2)

    if not stable or service_rate <= 0:
        return QueueingResult(
            arrival_qps=arrival_qps,
            service_ms=service_ms,
            replicas=replicas,
            utilization=utilization,
            stable=False,
            mean_latency_ms=math.inf,
            p50_latency_ms=math.inf,
            p95_latency_ms=math.inf,
            p99_latency_ms=math.inf,
            p999_latency_ms=math.inf,
            slo_ms=slo_ms,
            slo_ok=False,
            queue_wait_probability=1.0,
            queue_amplifier=math.inf,
        )

    wait_rate, wait_probability = _erlang_c(arrival_qps, service_ms, replicas)
    mean_wait_ms = (wait_probability / wait_rate) * 1000.0 * variability if wait_rate > 0 else math.inf

    def percentile(q: float) -> float:
        no_wait_probability = max(0.0, 1.0 - wait_probability)
        if q <= no_wait_probability or wait_probability <= 0:
            wait_ms = 0.0
        else:
            wait_ms = -math.log((1.0 - q) / wait_probability) / wait_rate * 1000.0
        return service_ms + wait_ms * variability

    mean_latency = service_ms + mean_wait_ms
    p50 = percentile(0.50)
    p95 = percentile(0.95)
    p99 = percentile(0.99)
    p999 = percentile(0.999)
    return QueueingResult(
        arrival_qps=arrival_qps,
        service_ms=service_ms,
        replicas=replicas,
        utilization=utilization,
        stable=True,
        mean_latency_ms=mean_latency,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        p999_latency_ms=p999,
        slo_ms=slo_ms,
        slo_ok=p99 <= slo_ms,
        queue_wait_probability=wait_probability,
        queue_amplifier=p99 / service_ms if service_ms > 0 else math.inf,
    )


def batching_tax(
    *,
    batch_size: int,
    arrival_qps: float,
    service_ms: float,
    slo_ms: float,
    efficiency_gain: float = 1.0,
    replicas: int = 1,
    service_cv: float = 1.0,
) -> BatchingTaxResult:
    """Estimate batch-formation delay and p99 latency for static batching."""
    batch = max(1, int(batch_size))
    arrival = max(0.0001, float(arrival_qps))
    formation_ms = (batch - 1) / (2 * arrival) * 1000.0
    gain = max(1.0, float(efficiency_gain))
    batched_service_ms = service_ms * (1 + 0.08 * math.log2(batch)) / gain
    effective_batch_qps = arrival / batch
    queue = queueing_latency(
        arrival_qps=effective_batch_qps,
        service_ms=batched_service_ms,
        replicas=replicas,
        service_cv=service_cv,
        slo_ms=max(0.001, slo_ms - formation_ms),
    )
    queue_p99_without_service = max(0.0, queue.p99_latency_ms - batched_service_ms)
    total_p99 = formation_ms + batched_service_ms + queue_p99_without_service
    throughput_gain = batch * service_ms / batched_service_ms if batched_service_ms > 0 else math.inf
    return BatchingTaxResult(
        batch_size=batch,
        arrival_qps=arrival,
        formation_delay_ms=formation_ms,
        batched_service_ms=batched_service_ms,
        queue_p99_ms=queue_p99_without_service,
        total_p99_ms=total_p99,
        formation_slo_pct=formation_ms / slo_ms * 100 if slo_ms > 0 else math.inf,
        throughput_gain=throughput_gain,
        effective_batch_qps=effective_batch_qps,
        utilization=queue.utilization,
        slo_ok=total_p99 <= slo_ms,
    )


def _transformer_state_gb(model: Any, context_tokens: int, kv_bytes: float) -> float | None:
    layers = getattr(model, "layers", None)
    hidden_dim = getattr(model, "hidden_dim", None)
    if not layers or not hidden_dim:
        return None
    return 2 * int(layers) * int(hidden_dim) * int(context_tokens) * kv_bytes / 1e9


def cache_capacity(
    profile: ServingTrackProfile,
    model: Any,
    *,
    context_tokens: int,
    precision_bytes: float,
    devices_per_replica: int = 1,
    kv_precision_bytes: float | None = None,
) -> CacheCapacityResult:
    """Estimate how many live requests fit once weights and state/cache are counted."""
    total_memory = max(0.0, int(devices_per_replica) * profile.memory_capacity_gb)
    weight_gb = max(0.0, profile.model_params_b * precision_bytes)
    transformer_state = _transformer_state_gb(
        model,
        context_tokens,
        kv_precision_bytes if kv_precision_bytes is not None else profile.kv_precision_bytes,
    )
    if profile.state_mb_per_request is not None and transformer_state is None:
        state_gb = profile.state_mb_per_request / 1024
    elif transformer_state is not None:
        state_gb = transformer_state
    else:
        state_gb = max(0.001, weight_gb * 0.1)
    available = max(0.0, total_memory - weight_gb)
    max_concurrent = math.floor(available / state_gb) if state_gb > 0 else 0
    return CacheCapacityResult(
        total_memory_gb=total_memory,
        weight_gb=weight_gb,
        state_per_request_gb=state_gb,
        available_gb=available,
        max_concurrent=max_concurrent,
        oom=max_concurrent < 1,
        state_kind=profile.state_kind,
        context_tokens=context_tokens,
    )


def cold_start_latency(
    profile: ServingTrackProfile,
    *,
    precision_bytes: float,
    storage_bandwidth_gbs: float | None = None,
    interconnect_bandwidth_gbs: float | None = None,
    deserialize_gbs: float | None = None,
    runtime_init_ms: float | None = None,
    warmup_ms: float | None = None,
    warm_pool_replicas: int = 0,
    scale_out_replicas: int = 1,
) -> ColdStartResult:
    """Decompose model-serving cold start into data movement and runtime overhead."""
    weight_gb = max(0.0, profile.model_params_b * precision_bytes)
    storage_bw = max(0.001, storage_bandwidth_gbs or profile.storage_bandwidth_gbs)
    interconnect_bw = max(0.001, interconnect_bandwidth_gbs or profile.interconnect_bandwidth_gbs)
    effective_bw = min(storage_bw, interconnect_bw)
    deser_bw = max(0.001, deserialize_gbs or profile.deserialize_gbs)
    init_ms = profile.runtime_init_ms if runtime_init_ms is None else runtime_init_ms
    warm_ms = profile.warmup_ms if warmup_ms is None else warmup_ms
    read_ms = weight_gb / effective_bw * 1000.0
    deserialize_ms = weight_gb / deser_bw * 1000.0
    cold_ms = read_ms + deserialize_ms + init_ms + warm_ms
    scale_out = max(1, int(scale_out_replicas))
    protected = min(1.0, max(0, int(warm_pool_replicas)) / scale_out)
    exposed = profile.service_ms + cold_ms * (1.0 - protected)
    return ColdStartResult(
        model_weight_gb=weight_gb,
        storage_bandwidth_gbs=storage_bw,
        interconnect_bandwidth_gbs=interconnect_bw,
        effective_bandwidth_gbs=effective_bw,
        read_ms=read_ms,
        deserialize_ms=deserialize_ms,
        runtime_init_ms=init_ms,
        warmup_ms=warm_ms,
        cold_start_ms=cold_ms,
        exposed_first_request_ms=exposed,
        protected_fraction=protected,
        exceeds_slo=exposed > profile.slo_ms,
    )


__all__ = [
    "BatchingTaxResult",
    "CacheCapacityResult",
    "ColdStartResult",
    "QueueingResult",
    "ServingTrackProfile",
    "batching_tax",
    "cache_capacity",
    "cold_start_latency",
    "queueing_latency",
    "serving_track_profile",
]
