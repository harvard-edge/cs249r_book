"""Shared inference-economy helpers for track-aware serving labs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .schemas import LabTrackVariant, TrackProfile


SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class InferenceEconomyProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    model_params_b: float
    memory_capacity_gb: float
    memory_bandwidth_gbs: float
    tdp_w: float
    setup_cost: float
    cost_per_event: float
    cost_unit: str
    cost_label: str
    demand_qps: float
    horizon_weeks: int
    context_tokens: int
    state_kind: str
    state_mb_per_request: float | None
    slo_ms: float
    qps_per_slot: float
    hardware_unit_cost_per_hour: float
    default_devices_per_replica: int
    batching_fill_factor: float
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class CostCrossoverResult:
    daily_cost: float
    weekly_cost: float
    crossover_weeks: float
    crossover_days: float
    annual_savings: float


@dataclass(frozen=True)
class StateCapacityResult:
    total_memory_gb: float
    weight_gb: float
    state_per_request_gb: float
    available_gb: float
    max_concurrent: int
    oom: bool
    state_kind: str


@dataclass(frozen=True)
class BatchingResult:
    padding_waste_pct: float
    static_throughput: float
    continuous_throughput: float
    speedup: float


@dataclass(frozen=True)
class ServingPlanResult:
    target_qps: float
    max_batch: int
    per_replica_qps: float
    replicas_needed: int
    total_devices: int
    daily_cost: float
    baseline_daily_cost: float
    savings_pct: float
    oom: bool


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


def inference_economy_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> InferenceEconomyProfile:
    """Build a source-traced track profile for inference-economy labs."""
    defaults = variant.defaults
    return InferenceEconomyProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        model_params_b=_model_params_b(model),
        memory_capacity_gb=_quantity_to_float(getattr(hardware.memory, "capacity", None), "GB", 0.0),
        memory_bandwidth_gbs=_quantity_to_float(getattr(hardware.memory, "bandwidth", None), "GB/s", 0.0),
        tdp_w=_quantity_to_float(getattr(hardware, "tdp", None), "W", 0.0),
        setup_cost=float(defaults.get("setup_cost", 1.0)),
        cost_per_event=float(defaults.get("cost_per_event", 0.01)),
        cost_unit=str(defaults.get("cost_unit", "USD")),
        cost_label=str(defaults.get("cost_label", "serving cost")),
        demand_qps=float(defaults.get("demand_qps", 100)),
        horizon_weeks=int(defaults.get("horizon_weeks", 26)),
        context_tokens=int(defaults.get("context_tokens", 4096)),
        state_kind=str(defaults.get("state_kind", "state/cache")),
        state_mb_per_request=(
            float(defaults["state_mb_per_request"])
            if "state_mb_per_request" in defaults
            else None
        ),
        slo_ms=float(defaults.get("slo_ms", 200)),
        qps_per_slot=float(defaults.get("qps_per_slot", 1.5)),
        hardware_unit_cost_per_hour=float(defaults.get("hardware_unit_cost_per_hour", 0.0)),
        default_devices_per_replica=int(defaults.get("devices_per_replica", 1)),
        batching_fill_factor=float(defaults.get("batching_fill_factor", 0.85)),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=(variant.hardware_ref, variant.model_ref),
    )


def cost_crossover(
    *,
    setup_cost: float,
    demand_qps: float,
    cost_per_event: float,
    optimization_pct: float = 0.0,
) -> CostCrossoverResult:
    """Compute when recurring inference cost exceeds the one-time setup cost."""
    effective_cost = max(0.0, cost_per_event * (1 - optimization_pct / 100))
    daily_cost = demand_qps * SECONDS_PER_DAY * effective_cost
    weekly_cost = daily_cost * 7
    crossover_weeks = setup_cost / weekly_cost if weekly_cost > 0 else math.inf
    annual_savings = demand_qps * SECONDS_PER_DAY * 365 * cost_per_event * (optimization_pct / 100)
    return CostCrossoverResult(
        daily_cost=daily_cost,
        weekly_cost=weekly_cost,
        crossover_weeks=crossover_weeks,
        crossover_days=crossover_weeks * 7 if math.isfinite(crossover_weeks) else math.inf,
        annual_savings=annual_savings,
    )


def _transformer_state_gb(model: Any, context_tokens: int, kv_bytes: float) -> float | None:
    layers = getattr(model, "layers", None)
    hidden_dim = getattr(model, "hidden_dim", None)
    if not layers or not hidden_dim:
        return None
    return 2 * int(layers) * int(hidden_dim) * int(context_tokens) * kv_bytes / 1e9


def state_capacity(
    profile: InferenceEconomyProfile,
    model: Any,
    *,
    context_tokens: int,
    precision_bytes: float,
    devices_per_replica: int = 1,
    kv_precision_bytes: float = 2.0,
) -> StateCapacityResult:
    """Estimate concurrent inference state capacity for the selected track."""
    total_memory = max(0.0, devices_per_replica * profile.memory_capacity_gb)
    weight_gb = max(0.0, profile.model_params_b * precision_bytes)

    transformer_state = _transformer_state_gb(model, context_tokens, kv_precision_bytes)
    if profile.state_mb_per_request is not None:
        state_gb = profile.state_mb_per_request / 1024
    elif transformer_state is not None:
        state_gb = transformer_state
    else:
        state_gb = max(0.001, weight_gb * 0.1)

    available = max(0.0, total_memory - weight_gb)
    max_concurrent = math.floor(available / state_gb) if state_gb > 0 else 0
    return StateCapacityResult(
        total_memory_gb=total_memory,
        weight_gb=weight_gb,
        state_per_request_gb=state_gb,
        available_gb=available,
        max_concurrent=max_concurrent,
        oom=max_concurrent < 1,
        state_kind=profile.state_kind,
    )


def batching_result(
    *,
    avg_len: int,
    max_len: int,
    batch_size: int,
    fill_factor: float = 0.85,
) -> BatchingResult:
    """Compare static padding waste with continuous batching occupancy."""
    if max_len <= 0 or batch_size <= 0:
        return BatchingResult(0.0, 0.0, 0.0, 1.0)
    padding_waste = max(0.0, min(1.0, 1 - avg_len / max_len))
    static = float(batch_size)
    continuous = batch_size * (max_len / max(1, avg_len)) * fill_factor
    speedup = continuous / static if static > 0 else 1.0
    return BatchingResult(
        padding_waste_pct=padding_waste * 100,
        static_throughput=static,
        continuous_throughput=continuous,
        speedup=speedup,
    )


def serving_plan(
    profile: InferenceEconomyProfile,
    model: Any,
    *,
    target_qps: float,
    precision_bytes: float,
    batching_multiplier: float,
    devices_per_replica: int,
    context_tokens: int | None = None,
) -> ServingPlanResult:
    """Size a serving/local-inference plan under memory and throughput limits."""
    state = state_capacity(
        profile,
        model,
        context_tokens=context_tokens or profile.context_tokens,
        precision_bytes=precision_bytes,
        devices_per_replica=devices_per_replica,
    )
    effective_batch = max(1, state.max_concurrent)
    per_replica_qps = effective_batch * profile.qps_per_slot * batching_multiplier
    replicas = math.ceil(target_qps / per_replica_qps) if per_replica_qps > 0 else 9999
    total_devices = replicas * devices_per_replica

    if profile.hardware_unit_cost_per_hour > 0:
        daily_cost = total_devices * profile.hardware_unit_cost_per_hour * 24
        baseline_devices = max(profile.default_devices_per_replica, devices_per_replica)
        baseline_batch = max(1, state.max_concurrent // max(1, devices_per_replica))
        baseline_qps = max(1.0, baseline_batch * profile.qps_per_slot)
        baseline_replicas = math.ceil(target_qps / baseline_qps)
        baseline_daily = baseline_replicas * baseline_devices * profile.hardware_unit_cost_per_hour * 24
    else:
        daily_cost = target_qps * SECONDS_PER_DAY * profile.cost_per_event / max(1.0, batching_multiplier)
        baseline_daily = target_qps * SECONDS_PER_DAY * profile.cost_per_event

    savings = (1 - daily_cost / baseline_daily) * 100 if baseline_daily > 0 else 0.0
    return ServingPlanResult(
        target_qps=target_qps,
        max_batch=state.max_concurrent,
        per_replica_qps=per_replica_qps,
        replicas_needed=replicas,
        total_devices=total_devices,
        daily_cost=daily_cost,
        baseline_daily_cost=baseline_daily,
        savings_pct=savings,
        oom=state.oom,
    )
