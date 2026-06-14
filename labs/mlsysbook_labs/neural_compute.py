"""Neural-compute helpers for track-aware activation and operator labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class OperatorDesignOption:
    design_id: str
    label: str
    precision_bytes: int
    activation_factor: float
    op_factor: float
    quality_risk: str
    residual_risk: str


@dataclass(frozen=True)
class NeuralComputeTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    peak_tflops: float
    memory_capacity_mb: float
    memory_bandwidth_gbs: float
    tdp_w: float
    stakeholder: str
    operator_story: str
    tensor_label: str
    batch: int
    channels: int
    height: int
    width: int
    sequence: int
    hidden: int
    precision_bytes: int
    activation_budget_mb: float
    bandwidth_budget_gbs: float
    latency_budget_ms: float
    power_budget_w: float
    default_shape_multiplier: float
    shape_min: float
    shape_max: float
    shape_step: float
    activation_multiplier: float
    ops_gmac_at_default: float
    design_options: tuple[OperatorDesignOption, ...]
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class OperationLedgerResult:
    shape_multiplier: float
    precision_bytes: int
    weights_mb: float
    activations_mb: float
    ops_gmac: float
    bytes_moved_mb: float
    arithmetic_intensity: float
    estimated_latency_ms: float
    estimated_bandwidth_gbs: float
    estimated_power_w: float
    dominant_resource: str
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class MemoryCliffResult:
    shape_values: tuple[float, ...]
    activation_mb: tuple[float, ...]
    feasible: tuple[bool, ...]
    threshold_multiplier: float | None
    threshold_activation_mb: float | None


@dataclass(frozen=True)
class OperatorDesignResult:
    design_id: str
    design_label: str
    activation_mb: float
    latency_ms: float
    bandwidth_gbs: float
    feasible: bool
    quality_risk: str
    residual_risk: str
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


def _peak_tflops(hardware: Any, precision: str) -> float:
    peak = None
    precision_flops = getattr(getattr(hardware, "compute", None), "precision_flops", {}) or {}
    if precision:
        peak = precision_flops.get(precision)
    if peak is None:
        peak = getattr(getattr(hardware, "compute", None), "peak_flops", None)
    value = _quantity_to_float(peak, "TFLOPs/s", 0.0)
    if value == 0.0:
        value = _quantity_to_float(peak, "TOPS", 0.0)
    return value


def _design_options(defaults: Mapping[str, Any]) -> tuple[OperatorDesignOption, ...]:
    raw_options = defaults.get("design_options", {})
    if not isinstance(raw_options, Mapping):
        raw_options = {}
    options: list[OperatorDesignOption] = []
    for design_id, raw in raw_options.items():
        details = raw if isinstance(raw, Mapping) else {}
        options.append(
            OperatorDesignOption(
                design_id=str(design_id),
                label=str(details.get("label", design_id)),
                precision_bytes=int(details.get("precision_bytes", defaults.get("precision_bytes", 2))),
                activation_factor=float(details.get("activation_factor", 1.0)),
                op_factor=float(details.get("op_factor", 1.0)),
                quality_risk=str(details.get("quality_risk", "quality risk not specified")),
                residual_risk=str(details.get("residual_risk", "residual operator risk not specified")),
            )
        )
    if options:
        return tuple(options)
    return (
        OperatorDesignOption(
            "baseline",
            "Baseline operator",
            int(defaults.get("precision_bytes", 2)),
            1.0,
            1.0,
            "baseline quality",
            "no operator mitigation selected",
        ),
    )


def neural_compute_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> NeuralComputeTrackProfile:
    """Build a source-traced neural-compute profile from variant defaults."""
    defaults = variant.defaults
    memory = getattr(hardware, "memory", None)
    precision = str(defaults.get("precision", ""))
    return NeuralComputeTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        peak_tflops=_peak_tflops(hardware, precision),
        memory_capacity_mb=_quantity_to_float(getattr(memory, "capacity", None), "MB", 0.0),
        memory_bandwidth_gbs=_quantity_to_float(getattr(memory, "bandwidth", None), "GB/s", 0.0),
        tdp_w=_quantity_to_float(getattr(hardware, "tdp", None), "W", 0.0),
        stakeholder=variant.stakeholder,
        operator_story=str(defaults.get("operator_story", variant.workload_summary)),
        tensor_label=str(defaults.get("tensor_label", "activation tensor")),
        batch=int(defaults.get("batch", 1)),
        channels=int(defaults.get("channels", 32)),
        height=int(defaults.get("height", 1)),
        width=int(defaults.get("width", 1)),
        sequence=int(defaults.get("sequence", 1)),
        hidden=int(defaults.get("hidden", 1)),
        precision_bytes=int(defaults.get("precision_bytes", 2)),
        activation_budget_mb=float(defaults.get("activation_budget_mb", 64.0)),
        bandwidth_budget_gbs=float(defaults.get("bandwidth_budget_gbs", 10.0)),
        latency_budget_ms=float(defaults.get("latency_budget_ms", 50.0)),
        power_budget_w=float(defaults.get("power_budget_w", 10.0)),
        default_shape_multiplier=float(defaults.get("default_shape_multiplier", 1.0)),
        shape_min=float(defaults.get("shape_min", 0.25)),
        shape_max=float(defaults.get("shape_max", 4.0)),
        shape_step=float(defaults.get("shape_step", 0.25)),
        activation_multiplier=float(defaults.get("activation_multiplier", 1.0)),
        ops_gmac_at_default=float(defaults.get("ops_gmac_at_default", 1.0)),
        design_options=_design_options(defaults),
        report_artifact=str(variant.assumptions.get("report_artifact", "operator budget note")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=tuple(ref for ref in (variant.hardware_ref, variant.model_ref, variant.system_ref) if ref),
    )


def operation_ledger(
    profile: NeuralComputeTrackProfile,
    *,
    shape_multiplier: float,
    precision_bytes: int | None = None,
    activation_factor: float = 1.0,
    op_factor: float = 1.0,
) -> OperationLedgerResult:
    """Compute weights, activations, operations, bytes moved, and feasibility."""
    shape = max(profile.shape_min, min(profile.shape_max, float(shape_multiplier)))
    bytes_per_element = int(precision_bytes or profile.precision_bytes)
    spatial_elements = profile.batch * profile.channels * profile.height * profile.width
    sequence_elements = profile.batch * profile.sequence * profile.hidden
    base_elements = max(spatial_elements, sequence_elements)
    activation_elements = base_elements * (shape**2) * profile.activation_multiplier * activation_factor
    activations_mb = activation_elements * bytes_per_element / 1_000_000
    weights_mb = profile.channels * max(1, profile.hidden) * bytes_per_element * max(1.0, shape) / 1_000_000
    ops_gmac = profile.ops_gmac_at_default * (shape**2) * op_factor
    bytes_moved_mb = weights_mb + activations_mb * 2.2
    arithmetic_intensity = (ops_gmac * 1_000_000_000) / max(bytes_moved_mb * 1_000_000, 1e-9)
    latency_ms = ops_gmac / max(profile.peak_tflops * 1000, 1e-9) * 1000
    bandwidth_gbs = bytes_moved_mb / max(latency_ms, 1e-9)
    power = min(max(profile.tdp_w * (ops_gmac / max(profile.ops_gmac_at_default, 1e-9)) * 0.45, 0.001), profile.tdp_w * 2)

    checks = {
        "activation memory": activations_mb / max(profile.activation_budget_mb, 1e-9),
        "bandwidth": bandwidth_gbs / max(profile.bandwidth_budget_gbs, 1e-9),
        "latency": latency_ms / max(profile.latency_budget_ms, 1e-9),
        "power": power / max(profile.power_budget_w, 1e-9),
    }
    dominant = max(checks, key=checks.get)
    violations = []
    if activations_mb > profile.activation_budget_mb:
        violations.append(f"activation memory {activations_mb:.2f} MB > {profile.activation_budget_mb:.2f} MB")
    if bandwidth_gbs > profile.bandwidth_budget_gbs:
        violations.append(f"bandwidth {bandwidth_gbs:.2f} GB/s > {profile.bandwidth_budget_gbs:.2f} GB/s")
    if latency_ms > profile.latency_budget_ms:
        violations.append(f"latency {latency_ms:.2f} ms > {profile.latency_budget_ms:.2f} ms")
    if power > profile.power_budget_w:
        violations.append(f"power {power:.2f} W > {profile.power_budget_w:.2f} W")
    return OperationLedgerResult(
        shape_multiplier=shape,
        precision_bytes=bytes_per_element,
        weights_mb=weights_mb,
        activations_mb=activations_mb,
        ops_gmac=ops_gmac,
        bytes_moved_mb=bytes_moved_mb,
        arithmetic_intensity=arithmetic_intensity,
        estimated_latency_ms=latency_ms,
        estimated_bandwidth_gbs=bandwidth_gbs,
        estimated_power_w=power,
        dominant_resource=dominant,
        feasible=not violations,
        violations=tuple(violations),
    )


def memory_cliff(
    profile: NeuralComputeTrackProfile,
    *,
    samples: int = 40,
) -> MemoryCliffResult:
    """Sweep shape multiplier until activation memory crosses the budget."""
    count = max(2, int(samples))
    span = profile.shape_max - profile.shape_min
    values = tuple(profile.shape_min + span * idx / (count - 1) for idx in range(count))
    results = tuple(operation_ledger(profile, shape_multiplier=value) for value in values)
    crossing = next((result for result in results if result.activations_mb > profile.activation_budget_mb), None)
    return MemoryCliffResult(
        shape_values=tuple(result.shape_multiplier for result in results),
        activation_mb=tuple(result.activations_mb for result in results),
        feasible=tuple(result.feasible for result in results),
        threshold_multiplier=crossing.shape_multiplier if crossing else None,
        threshold_activation_mb=crossing.activations_mb if crossing else None,
    )


def operator_design(
    profile: NeuralComputeTrackProfile,
    *,
    design_id: str,
    shape_multiplier: float,
) -> OperatorDesignResult:
    """Evaluate the selected operator design option."""
    design = next((candidate for candidate in profile.design_options if candidate.design_id == design_id), profile.design_options[0])
    ledger = operation_ledger(
        profile,
        shape_multiplier=shape_multiplier,
        precision_bytes=design.precision_bytes,
        activation_factor=design.activation_factor,
        op_factor=design.op_factor,
    )
    summary = (
        f"Use {design.label}; activation memory {ledger.activations_mb:.2f} MB; "
        f"dominant resource {ledger.dominant_resource}."
    )
    return OperatorDesignResult(
        design_id=design.design_id,
        design_label=design.label,
        activation_mb=ledger.activations_mb,
        latency_ms=ledger.estimated_latency_ms,
        bandwidth_gbs=ledger.estimated_bandwidth_gbs,
        feasible=ledger.feasible,
        quality_risk=design.quality_risk,
        residual_risk=design.residual_risk,
        memo_summary=summary,
    )


__all__ = [
    "MemoryCliffResult",
    "NeuralComputeTrackProfile",
    "OperationLedgerResult",
    "OperatorDesignOption",
    "OperatorDesignResult",
    "memory_cliff",
    "neural_compute_profile",
    "operation_ledger",
    "operator_design",
]
