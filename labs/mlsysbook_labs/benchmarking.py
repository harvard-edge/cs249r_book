"""Shared benchmarking-trap helpers for track-aware labs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class BenchmarkTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    tdp_w: float
    memory_capacity_gb: float
    benchmark_claim: str
    hidden_failure_metric: str
    component_label: str
    metric_unit: str
    burst_value: float
    sustained_threshold: float
    default_speedup: float
    default_serial_pct: float
    default_duration_s: int
    default_ambient_c: float
    default_cooling: str
    accuracy_min_pct: float
    p99_max_ms: float
    power_max_w: float
    throughput_min: float
    tail_base_ms: float
    tail_sigma: float
    tail_slo_ms: float
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class AmdahlResult:
    component_speedup: float
    serial_fraction: float
    system_speedup: float
    asymptote: float
    wasted_speedup_pct: float
    new_serial_pct: float


@dataclass(frozen=True)
class ThermalSustainResult:
    junction_temp_c: float
    sustained_value: float
    throttled: bool
    loss_pct: float
    cooling: str


@dataclass(frozen=True)
class MetricGateResult:
    accuracy_pct: float
    p99_latency_ms: float
    power_w: float
    throughput: float
    all_pass: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class TailLatencyResult:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    violation_pct: float
    slo_ok: bool


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


def benchmark_track_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> BenchmarkTrackProfile:
    """Build a source-traced benchmarking profile from MLSysIM refs and variant defaults."""
    defaults = variant.defaults
    return BenchmarkTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        tdp_w=_quantity_to_float(getattr(hardware, "tdp", None), "W", 1.0),
        memory_capacity_gb=_quantity_to_float(getattr(hardware.memory, "capacity", None), "GB", 0.0),
        benchmark_claim=str(defaults.get("benchmark_claim", "single benchmark number")),
        hidden_failure_metric=str(defaults.get("hidden_failure_metric", "tail behavior")),
        component_label=str(defaults.get("component_label", "inference")),
        metric_unit=str(defaults.get("metric_unit", "units/s")),
        burst_value=float(defaults.get("burst_value", 30.0)),
        sustained_threshold=float(defaults.get("sustained_threshold", 0.75)),
        default_speedup=float(defaults.get("component_speedup", 10.0)),
        default_serial_pct=float(defaults.get("serial_pct", 45.0)),
        default_duration_s=int(defaults.get("duration_s", 300)),
        default_ambient_c=float(defaults.get("ambient_c", 35.0)),
        default_cooling=str(defaults.get("cooling", "fanless")),
        accuracy_min_pct=float(defaults.get("accuracy_min_pct", 90.0)),
        p99_max_ms=float(defaults.get("p99_max_ms", 100.0)),
        power_max_w=float(defaults.get("power_max_w", max(1.0, _quantity_to_float(getattr(hardware, "tdp", None), "W", 5.0)))),
        throughput_min=float(defaults.get("throughput_min", 1.0)),
        tail_base_ms=float(defaults.get("tail_base_ms", 50.0)),
        tail_sigma=float(defaults.get("tail_sigma", 0.8)),
        tail_slo_ms=float(defaults.get("tail_slo_ms", 200.0)),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=(variant.hardware_ref, variant.model_ref),
    )


def amdahl_speedup(*, component_speedup: float, serial_pct: float) -> AmdahlResult:
    serial = max(0.0, min(0.99, serial_pct / 100))
    speedup = max(1.0, component_speedup)
    system = 1.0 / (serial + (1 - serial) / speedup)
    asymptote = 1.0 / serial if serial > 0 else math.inf
    wasted = (speedup - system) / speedup * 100
    total_after = serial + (1 - serial) / speedup
    new_serial = serial / total_after * 100 if total_after > 0 else 0.0
    return AmdahlResult(
        component_speedup=speedup,
        serial_fraction=serial,
        system_speedup=system,
        asymptote=asymptote,
        wasted_speedup_pct=wasted,
        new_serial_pct=new_serial,
    )


def sustained_benchmark(
    *,
    peak_value: float,
    tdp_w: float,
    duration_s: float,
    ambient_c: float,
    cooling: str,
    throttle_c: float = 85.0,
) -> ThermalSustainResult:
    conductance = {"active": 5.0, "passive": 2.0, "fanless": 0.8}.get(cooling, 1.0)
    tau = {"active": 120.0, "passive": 60.0, "fanless": 30.0}.get(cooling, 60.0)
    junction = ambient_c + (tdp_w / conductance) * (1 - math.exp(-duration_s / tau))
    throttled = junction > throttle_c
    if throttled:
        throttle_factor = max(0.3, 1.0 - (junction - throttle_c) / 50.0)
        sustained = peak_value * throttle_factor
    else:
        sustained = peak_value
    loss_pct = (1 - sustained / peak_value) * 100 if peak_value > 0 else 0.0
    return ThermalSustainResult(
        junction_temp_c=junction,
        sustained_value=sustained,
        throttled=throttled,
        loss_pct=loss_pct,
        cooling=cooling,
    )


def metric_gate(
    *,
    accuracy_pct: float,
    p99_latency_ms: float,
    power_w: float,
    throughput: float,
    thresholds: Mapping[str, float],
) -> MetricGateResult:
    violations: list[str] = []
    if accuracy_pct < thresholds["accuracy_min_pct"]:
        violations.append(f"accuracy {accuracy_pct:.1f}% < {thresholds['accuracy_min_pct']:.1f}%")
    if p99_latency_ms > thresholds["p99_max_ms"]:
        violations.append(f"p99 latency {p99_latency_ms:.1f} ms > {thresholds['p99_max_ms']:.1f} ms")
    if power_w > thresholds["power_max_w"]:
        violations.append(f"power {power_w:.2f} W > {thresholds['power_max_w']:.2f} W")
    if throughput < thresholds["throughput_min"]:
        violations.append(f"throughput {throughput:.1f} < {thresholds['throughput_min']:.1f}")
    return MetricGateResult(
        accuracy_pct=accuracy_pct,
        p99_latency_ms=p99_latency_ms,
        power_w=power_w,
        throughput=throughput,
        all_pass=not violations,
        violations=tuple(violations),
    )


def tail_latency(*, base_ms: float, sigma: float, slo_ms: float) -> TailLatencyResult:
    mean = base_ms * math.exp((sigma**2) / 2)
    p50 = base_ms
    p95 = base_ms * math.exp(1.645 * sigma)
    p99 = base_ms * math.exp(2.326 * sigma)
    p999 = base_ms * math.exp(3.09 * sigma)
    if slo_ms <= 0:
        violation_pct = 100.0
    else:
        z = math.log(slo_ms / base_ms) / sigma if sigma > 0 and base_ms > 0 else math.inf
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        violation_pct = max(0.0, min(100.0, (1 - cdf) * 100))
    return TailLatencyResult(
        mean_ms=mean,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        p999_ms=p999,
        violation_pct=violation_pct,
        slo_ok=p99 <= slo_ms,
    )
