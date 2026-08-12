"""Framework and runtime helpers for track-aware deployment labs."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class RuntimeOption:
    runtime_id: str
    label: str
    execution_mode: str
    footprint_mb: float
    dispatch_overhead_us: float
    kernel_support_pct: float
    compile_cost_s: float
    fusion_factor: float
    latency_multiplier: float
    portability_risk: str
    unsupported_op_consequence: str
    validation_requirement: str
    residual_risk: str


@dataclass(frozen=True)
class FrameworkTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    memory_capacity_mb: float
    memory_bandwidth_gbs: float
    tdp_w: float
    hardware_dispatch_ms: float
    stakeholder: str
    runtime_story: str
    workload_label: str
    op_count: int
    compute_us_per_op: float
    transfer_us: float
    sync_us: float
    memory_us: float
    shape_dynamism_pct: float
    default_reuse_count: int
    reuse_min: int
    reuse_max: int
    reuse_step: int
    latency_budget_ms: float
    memory_budget_mb: float
    kernel_support_floor_pct: float
    runtime_options: tuple[RuntimeOption, ...]
    validation_tests: tuple[str, ...]
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class DispatchStackResult:
    runtime_id: str
    runtime_label: str
    effective_kernels: float
    useful_compute_ms: float
    runtime_dispatch_ms: float
    hardware_dispatch_ms: float
    transfer_ms: float
    sync_ms: float
    memory_ms: float
    total_latency_ms: float
    overhead_pct: float
    footprint_mb: float
    kernel_support_pct: float
    dominant_overhead: str
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class CompileBreakEvenResult:
    runtime_id: str
    runtime_label: str
    baseline_runtime_id: str
    compile_cost_s: float
    baseline_latency_ms: float
    optimized_latency_ms: float
    per_inference_savings_ms: float
    break_even_inferences: int | None
    selected_reuse_count: int
    pays_back: bool


@dataclass(frozen=True)
class RuntimeDecisionResult:
    selected_id: str
    selected_label: str
    feasible: bool
    dominant_overhead: str
    total_latency_ms: float
    break_even_inferences: int | None
    unsupported_op_warning: str
    validation_requirement: str
    residual_risk: str
    rejected_alternatives: tuple[str, ...]
    memo_summary: str


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


def _tuple_str(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if value:
        return (str(value),)
    return ()


def _runtime_options(defaults: Mapping[str, Any]) -> tuple[RuntimeOption, ...]:
    raw_options = defaults.get("runtime_options", {})
    if not isinstance(raw_options, Mapping):
        raw_options = {}
    options: list[RuntimeOption] = []
    for runtime_id, raw in raw_options.items():
        details = raw if isinstance(raw, Mapping) else {}
        options.append(
            RuntimeOption(
                runtime_id=str(runtime_id),
                label=str(details.get("label", runtime_id)),
                execution_mode=str(details.get("execution_mode", "runtime")),
                footprint_mb=float(details.get("footprint_mb", 1.0)),
                dispatch_overhead_us=float(details.get("dispatch_overhead_us", 10.0)),
                kernel_support_pct=float(details.get("kernel_support_pct", 100.0)),
                compile_cost_s=float(details.get("compile_cost_s", 0.0)),
                fusion_factor=max(1.0, float(details.get("fusion_factor", 1.0))),
                latency_multiplier=float(details.get("latency_multiplier", 1.0)),
                portability_risk=str(details.get("portability_risk", "portability risk not specified")),
                unsupported_op_consequence=str(
                    details.get("unsupported_op_consequence", "unsupported-op consequence not specified")
                ),
                validation_requirement=str(details.get("validation_requirement", "validation not specified")),
                residual_risk=str(details.get("residual_risk", "residual runtime risk not specified")),
            )
        )
    if options:
        return tuple(options)
    return (
        RuntimeOption(
            "baseline",
            "Baseline runtime",
            "eager",
            1.0,
            10.0,
            100.0,
            0.0,
            1.0,
            1.0,
            "baseline portability",
            "no unsupported-op policy",
            "baseline validation",
            "no runtime mitigation selected",
        ),
    )


def framework_track_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> FrameworkTrackProfile:
    """Build a source-traced framework/runtime profile from variant defaults."""
    defaults = variant.defaults
    memory = getattr(hardware, "memory", None)
    return FrameworkTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        memory_capacity_mb=_quantity_to_float(getattr(memory, "capacity", None), "MB", 0.0),
        memory_bandwidth_gbs=_quantity_to_float(getattr(memory, "bandwidth", None), "GB/s", 0.0),
        tdp_w=_quantity_to_float(getattr(hardware, "tdp", None), "W", 0.0),
        hardware_dispatch_ms=_quantity_to_float(getattr(hardware, "dispatch_tax", None), "ms", 0.0),
        stakeholder=variant.stakeholder,
        runtime_story=str(defaults.get("runtime_story", variant.workload_summary)),
        workload_label=str(defaults.get("workload_label", variant.workload_summary)),
        op_count=int(defaults.get("op_count", 100)),
        compute_us_per_op=float(defaults.get("compute_us_per_op", 10.0)),
        transfer_us=float(defaults.get("transfer_us", 100.0)),
        sync_us=float(defaults.get("sync_us", 50.0)),
        memory_us=float(defaults.get("memory_us", 500.0)),
        shape_dynamism_pct=float(defaults.get("shape_dynamism_pct", 0.0)),
        default_reuse_count=int(defaults.get("default_reuse_count", 1000)),
        reuse_min=int(defaults.get("reuse_min", 1)),
        reuse_max=int(defaults.get("reuse_max", 100000)),
        reuse_step=max(1, int(defaults.get("reuse_step", 100))),
        latency_budget_ms=float(defaults.get("latency_budget_ms", 50.0)),
        memory_budget_mb=float(defaults.get("memory_budget_mb", 256.0)),
        kernel_support_floor_pct=float(defaults.get("kernel_support_floor_pct", 80.0)),
        runtime_options=_runtime_options(defaults),
        validation_tests=_tuple_str(defaults.get("validation_tests")),
        report_artifact=str(variant.assumptions.get("report_artifact", "runtime deployment recommendation")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=tuple(ref for ref in (variant.hardware_ref, variant.model_ref, variant.system_ref) if ref),
    )


def _runtime(profile: FrameworkTrackProfile, runtime_id: str) -> RuntimeOption:
    return next(
        (option for option in profile.runtime_options if option.runtime_id == runtime_id),
        profile.runtime_options[0],
    )


def dispatch_stack(
    profile: FrameworkTrackProfile,
    *,
    runtime_id: str,
    op_count: int | None = None,
    compute_us_per_op: float | None = None,
) -> DispatchStackResult:
    """Compute dispatch, transfer, synchronization, memory, and feasibility."""
    runtime = _runtime(profile, runtime_id)
    ops = max(1, int(op_count or profile.op_count))
    compute_us = max(0.001, float(compute_us_per_op or profile.compute_us_per_op))
    effective_kernels = max(1.0, ops / runtime.fusion_factor)
    useful_compute_ms = ops * compute_us * runtime.latency_multiplier / 1000.0
    runtime_dispatch_ms = effective_kernels * runtime.dispatch_overhead_us / 1000.0
    hardware_dispatch_ms = profile.hardware_dispatch_ms * (1.0 + effective_kernels / 1000.0)
    transfer_ms = profile.transfer_us * (1.0 + ops / 1000.0) / 1000.0
    sync_ms = profile.sync_us * (1.0 + profile.shape_dynamism_pct / 100.0) / 1000.0
    memory_ms = profile.memory_us / runtime.fusion_factor / 1000.0
    total = useful_compute_ms + runtime_dispatch_ms + hardware_dispatch_ms + transfer_ms + sync_ms + memory_ms
    overhead = total - useful_compute_ms
    overhead_pct = 100.0 * overhead / max(total, 1e-9)
    overheads = {
        "runtime dispatch": runtime_dispatch_ms,
        "hardware dispatch": hardware_dispatch_ms,
        "transfer": transfer_ms,
        "synchronization": sync_ms,
        "memory traffic": memory_ms,
    }
    dominant = max(overheads, key=overheads.get)
    violations = []
    if total > profile.latency_budget_ms:
        violations.append(f"latency {total:.2f} ms > {profile.latency_budget_ms:.2f} ms")
    if runtime.footprint_mb > profile.memory_budget_mb:
        violations.append(f"runtime footprint {runtime.footprint_mb:.3f} MB > {profile.memory_budget_mb:.3f} MB")
    if runtime.kernel_support_pct < profile.kernel_support_floor_pct:
        violations.append(
            f"kernel support {runtime.kernel_support_pct:.1f}% < {profile.kernel_support_floor_pct:.1f}%"
        )
    return DispatchStackResult(
        runtime_id=runtime.runtime_id,
        runtime_label=runtime.label,
        effective_kernels=effective_kernels,
        useful_compute_ms=useful_compute_ms,
        runtime_dispatch_ms=runtime_dispatch_ms,
        hardware_dispatch_ms=hardware_dispatch_ms,
        transfer_ms=transfer_ms,
        sync_ms=sync_ms,
        memory_ms=memory_ms,
        total_latency_ms=total,
        overhead_pct=overhead_pct,
        footprint_mb=runtime.footprint_mb,
        kernel_support_pct=runtime.kernel_support_pct,
        dominant_overhead=dominant,
        feasible=not violations,
        violations=tuple(violations),
    )


def compile_break_even(
    profile: FrameworkTrackProfile,
    *,
    runtime_id: str,
    baseline_runtime_id: str | None = None,
    reuse_count: int | None = None,
    op_count: int | None = None,
) -> CompileBreakEvenResult:
    """Calculate when compile/fusion cost pays back versus a baseline runtime."""
    baseline_id = baseline_runtime_id or profile.runtime_options[-1].runtime_id
    selected_runtime = _runtime(profile, runtime_id)
    baseline = dispatch_stack(profile, runtime_id=baseline_id, op_count=op_count)
    optimized = dispatch_stack(profile, runtime_id=runtime_id, op_count=op_count)
    savings_ms = baseline.total_latency_ms - optimized.total_latency_ms
    if savings_ms <= 0:
        break_even = None
    else:
        break_even = max(1, ceil(selected_runtime.compile_cost_s * 1000.0 / savings_ms))
    selected_reuse = int(reuse_count or profile.default_reuse_count)
    pays_back = break_even is not None and selected_reuse >= break_even
    return CompileBreakEvenResult(
        runtime_id=selected_runtime.runtime_id,
        runtime_label=selected_runtime.label,
        baseline_runtime_id=baseline.runtime_id,
        compile_cost_s=selected_runtime.compile_cost_s,
        baseline_latency_ms=baseline.total_latency_ms,
        optimized_latency_ms=optimized.total_latency_ms,
        per_inference_savings_ms=savings_ms,
        break_even_inferences=break_even,
        selected_reuse_count=selected_reuse,
        pays_back=pays_back,
    )


def runtime_decision(
    profile: FrameworkTrackProfile,
    *,
    runtime_id: str,
    reuse_count: int | None = None,
    op_count: int | None = None,
) -> RuntimeDecisionResult:
    """Return the recommendation memo fields for the selected runtime."""
    runtime = _runtime(profile, runtime_id)
    selected_stack = dispatch_stack(profile, runtime_id=runtime.runtime_id, op_count=op_count)
    selected_break_even = compile_break_even(
        profile,
        runtime_id=runtime.runtime_id,
        reuse_count=reuse_count,
        op_count=op_count,
    )
    rejected_items = []
    for option in profile.runtime_options:
        if option.runtime_id == runtime.runtime_id:
            continue
        result = dispatch_stack(profile, runtime_id=option.runtime_id, op_count=op_count)
        rejected_items.append(
            f"{option.label}: {result.dominant_overhead}; {'feasible' if result.feasible else 'not feasible'}"
        )
    _break_even_text = (
        f"{selected_break_even.break_even_inferences} inferences"
        if selected_break_even.break_even_inferences is not None
        else "no payback"
    )
    summary = (
        f"Choose {runtime.label} for {profile.label}; dominant overhead is "
        f"{selected_stack.dominant_overhead}, compile break-even is {_break_even_text}."
    )
    return RuntimeDecisionResult(
        selected_id=runtime.runtime_id,
        selected_label=runtime.label,
        feasible=selected_stack.feasible,
        dominant_overhead=selected_stack.dominant_overhead,
        total_latency_ms=selected_stack.total_latency_ms,
        break_even_inferences=selected_break_even.break_even_inferences,
        unsupported_op_warning=runtime.unsupported_op_consequence,
        validation_requirement=runtime.validation_requirement,
        residual_risk=runtime.residual_risk,
        rejected_alternatives=tuple(rejected_items),
        memo_summary=summary,
    )


__all__ = [
    "CompileBreakEvenResult",
    "DispatchStackResult",
    "FrameworkTrackProfile",
    "RuntimeDecisionResult",
    "RuntimeOption",
    "compile_break_even",
    "dispatch_stack",
    "framework_track_profile",
    "runtime_decision",
]
