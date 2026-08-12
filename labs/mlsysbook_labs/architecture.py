"""Architecture-family helpers for track-aware architecture choice labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class ArchitectureCandidate:
    architecture_id: str
    label: str
    family: str
    summary: str
    params_m: float
    ops_gmac: float
    activation_mb: float
    latency_ms: float
    power_w: float
    quality_pct: float
    kernel_support_pct: float
    scaling_law: str
    param_exponent: float
    op_exponent: float
    activation_exponent: float
    latency_exponent: float
    guardrail: str
    validation_requirement: str
    residual_risk: str


@dataclass(frozen=True)
class ArchitectureTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    memory_capacity_mb: float
    memory_bandwidth_gbs: float
    tdp_w: float
    dispatch_ms: float
    stakeholder: str
    architecture_story: str
    workload_label: str
    scaling_variable: str
    scaling_unit: str
    default_scale: float
    scale_min: float
    scale_max: float
    scale_step: float
    memory_budget_mb: float
    latency_budget_ms: float
    power_budget_w: float
    quality_floor_pct: float
    kernel_support_floor_pct: float
    candidates: tuple[ArchitectureCandidate, ...]
    validation_tests: tuple[str, ...]
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class ArchitectureEvaluation:
    architecture_id: str
    label: str
    family: str
    scale_value: float
    params_m: float
    ops_gmac: float
    activation_mb: float
    latency_ms: float
    power_w: float
    quality_pct: float
    kernel_support_pct: float
    dominant_constraint: str
    feasible: bool
    violations: tuple[str, ...]
    score: float
    next_failure: str
    guardrail: str
    validation_requirement: str
    residual_risk: str


@dataclass(frozen=True)
class ArchitectureScalingPoint:
    scale_value: float
    architecture_id: str
    latency_ms: float
    activation_mb: float
    power_w: float
    feasible: bool


@dataclass(frozen=True)
class ArchitectureScalingResult:
    scale_values: tuple[float, ...]
    points_by_candidate: Mapping[str, tuple[ArchitectureScalingPoint, ...]]
    first_failure_by_candidate: Mapping[str, float | None]


@dataclass(frozen=True)
class ArchitectureDecisionResult:
    selected_id: str
    selected_label: str
    feasible: bool
    dominant_constraint: str
    rejected_alternatives: tuple[str, ...]
    next_failure: str
    validation_requirement: str
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


def _tuple_str(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if value:
        return (str(value),)
    return ()


def _candidate_options(defaults: Mapping[str, Any]) -> tuple[ArchitectureCandidate, ...]:
    raw_candidates = defaults.get("candidate_architectures", {})
    if not isinstance(raw_candidates, Mapping):
        raw_candidates = {}
    candidates: list[ArchitectureCandidate] = []
    for architecture_id, raw in raw_candidates.items():
        details = raw if isinstance(raw, Mapping) else {}
        candidates.append(
            ArchitectureCandidate(
                architecture_id=str(architecture_id),
                label=str(details.get("label", architecture_id)),
                family=str(details.get("family", "unspecified")),
                summary=str(details.get("summary", "architecture summary not specified")),
                params_m=float(details.get("params_m", 1.0)),
                ops_gmac=float(details.get("ops_gmac", 1.0)),
                activation_mb=float(details.get("activation_mb", 1.0)),
                latency_ms=float(details.get("latency_ms", 1.0)),
                power_w=float(details.get("power_w", 1.0)),
                quality_pct=float(details.get("quality_pct", 0.0)),
                kernel_support_pct=float(details.get("kernel_support_pct", 100.0)),
                scaling_law=str(details.get("scaling_law", "linear scaling")),
                param_exponent=float(details.get("param_exponent", 0.0)),
                op_exponent=float(details.get("op_exponent", 1.0)),
                activation_exponent=float(details.get("activation_exponent", 1.0)),
                latency_exponent=float(details.get("latency_exponent", 1.0)),
                guardrail=str(details.get("guardrail", "track guardrail not specified")),
                validation_requirement=str(details.get("validation_requirement", "validation not specified")),
                residual_risk=str(details.get("residual_risk", "residual architecture risk not specified")),
            )
        )
    if candidates:
        return tuple(candidates)
    return (
        ArchitectureCandidate(
            "baseline",
            "Baseline architecture",
            "baseline",
            "Fallback architecture option.",
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            80.0,
            100.0,
            "linear scaling",
            0.0,
            1.0,
            1.0,
            1.0,
            "baseline guardrail",
            "baseline validation",
            "no alternative architecture evaluated",
        ),
    )


def architecture_track_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> ArchitectureTrackProfile:
    """Build a source-traced architecture profile from variant defaults."""
    defaults = variant.defaults
    memory = getattr(hardware, "memory", None)
    return ArchitectureTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        memory_capacity_mb=_quantity_to_float(getattr(memory, "capacity", None), "MB", 0.0),
        memory_bandwidth_gbs=_quantity_to_float(getattr(memory, "bandwidth", None), "GB/s", 0.0),
        tdp_w=_quantity_to_float(getattr(hardware, "tdp", None), "W", 0.0),
        dispatch_ms=_quantity_to_float(getattr(hardware, "dispatch_tax", None), "ms", 0.0),
        stakeholder=variant.stakeholder,
        architecture_story=str(defaults.get("architecture_story", variant.workload_summary)),
        workload_label=str(defaults.get("workload_label", variant.workload_summary)),
        scaling_variable=str(defaults.get("scaling_variable", "scale")),
        scaling_unit=str(defaults.get("scaling_unit", "x")),
        default_scale=float(defaults.get("default_scale", 1.0)),
        scale_min=float(defaults.get("scale_min", 0.5)),
        scale_max=float(defaults.get("scale_max", 2.0)),
        scale_step=float(defaults.get("scale_step", 0.1)),
        memory_budget_mb=float(defaults.get("memory_budget_mb", 64.0)),
        latency_budget_ms=float(defaults.get("latency_budget_ms", 50.0)),
        power_budget_w=float(defaults.get("power_budget_w", 10.0)),
        quality_floor_pct=float(defaults.get("quality_floor_pct", 80.0)),
        kernel_support_floor_pct=float(defaults.get("kernel_support_floor_pct", 80.0)),
        candidates=_candidate_options(defaults),
        validation_tests=_tuple_str(defaults.get("validation_tests")),
        report_artifact=str(variant.assumptions.get("report_artifact", "architecture recommendation memo")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=tuple(ref for ref in (variant.hardware_ref, variant.model_ref, variant.system_ref) if ref),
    )


def _scaled(candidate: ArchitectureCandidate, profile: ArchitectureTrackProfile, scale_value: float) -> dict[str, float]:
    scale = max(profile.scale_min, min(profile.scale_max, float(scale_value)))
    ratio = scale / max(profile.default_scale, 1e-9)
    return {
        "scale": scale,
        "params_m": candidate.params_m * (ratio**candidate.param_exponent),
        "ops_gmac": candidate.ops_gmac * (ratio**candidate.op_exponent),
        "activation_mb": candidate.activation_mb * (ratio**candidate.activation_exponent),
        "latency_ms": candidate.latency_ms * (ratio**candidate.latency_exponent),
        "power_w": candidate.power_w * (ratio ** min(candidate.op_exponent, 1.5)),
    }


def _next_failure(ratios: Mapping[str, float]) -> str:
    ordered = sorted(ratios.items(), key=lambda item: item[1], reverse=True)
    name, ratio = ordered[0]
    if ratio >= 1.0:
        return f"{name} already exceeds budget"
    return f"{name} is closest to failure at {ratio:.2f}x budget"


def architecture_signature(
    profile: ArchitectureTrackProfile,
    *,
    scale_value: float | None = None,
) -> tuple[ArchitectureEvaluation, ...]:
    """Evaluate each candidate architecture against the selected track budgets."""
    scale = profile.default_scale if scale_value is None else float(scale_value)
    evaluations: list[ArchitectureEvaluation] = []
    for candidate in profile.candidates:
        values = _scaled(candidate, profile, scale)
        ratios = {
            "activation memory": values["activation_mb"] / max(profile.memory_budget_mb, 1e-9),
            "latency": values["latency_ms"] / max(profile.latency_budget_ms, 1e-9),
            "power": values["power_w"] / max(profile.power_budget_w, 1e-9),
            "quality guardrail": profile.quality_floor_pct / max(candidate.quality_pct, 1e-9),
            "kernel support": profile.kernel_support_floor_pct / max(candidate.kernel_support_pct, 1e-9),
        }
        dominant = max(ratios, key=ratios.get)
        violations = []
        if values["activation_mb"] > profile.memory_budget_mb:
            violations.append(
                f"activation memory {values['activation_mb']:.2f} MB > {profile.memory_budget_mb:.2f} MB"
            )
        if values["latency_ms"] > profile.latency_budget_ms:
            violations.append(f"latency {values['latency_ms']:.2f} ms > {profile.latency_budget_ms:.2f} ms")
        if values["power_w"] > profile.power_budget_w:
            violations.append(f"power {values['power_w']:.3f} W > {profile.power_budget_w:.3f} W")
        if candidate.quality_pct < profile.quality_floor_pct:
            violations.append(f"quality {candidate.quality_pct:.1f}% < {profile.quality_floor_pct:.1f}%")
        if candidate.kernel_support_pct < profile.kernel_support_floor_pct:
            violations.append(
                f"kernel support {candidate.kernel_support_pct:.1f}% < {profile.kernel_support_floor_pct:.1f}%"
            )

        penalty = sum(max(0.0, ratio - 1.0) for ratio in ratios.values()) * 40.0
        score = candidate.quality_pct + 0.08 * candidate.kernel_support_pct - penalty
        evaluations.append(
            ArchitectureEvaluation(
                architecture_id=candidate.architecture_id,
                label=candidate.label,
                family=candidate.family,
                scale_value=values["scale"],
                params_m=values["params_m"],
                ops_gmac=values["ops_gmac"],
                activation_mb=values["activation_mb"],
                latency_ms=values["latency_ms"],
                power_w=values["power_w"],
                quality_pct=candidate.quality_pct,
                kernel_support_pct=candidate.kernel_support_pct,
                dominant_constraint=dominant,
                feasible=not violations,
                violations=tuple(violations),
                score=score,
                next_failure=_next_failure(ratios),
                guardrail=candidate.guardrail,
                validation_requirement=candidate.validation_requirement,
                residual_risk=candidate.residual_risk,
            )
        )
    return tuple(evaluations)


def architecture_scaling_curve(
    profile: ArchitectureTrackProfile,
    *,
    samples: int = 40,
) -> ArchitectureScalingResult:
    """Sweep the track scaling variable and record the first infeasible point."""
    count = max(2, int(samples))
    span = profile.scale_max - profile.scale_min
    values = tuple(profile.scale_min + span * idx / (count - 1) for idx in range(count))
    points_by_candidate: dict[str, tuple[ArchitectureScalingPoint, ...]] = {}
    first_failure_by_candidate: dict[str, float | None] = {}
    for candidate in profile.candidates:
        points = []
        first_failure = None
        for scale in values:
            evaluation = next(
                item for item in architecture_signature(profile, scale_value=scale)
                if item.architecture_id == candidate.architecture_id
            )
            if not evaluation.feasible and first_failure is None:
                first_failure = scale
            points.append(
                ArchitectureScalingPoint(
                    scale_value=scale,
                    architecture_id=evaluation.architecture_id,
                    latency_ms=evaluation.latency_ms,
                    activation_mb=evaluation.activation_mb,
                    power_w=evaluation.power_w,
                    feasible=evaluation.feasible,
                )
            )
        points_by_candidate[candidate.architecture_id] = tuple(points)
        first_failure_by_candidate[candidate.architecture_id] = first_failure
    return ArchitectureScalingResult(
        scale_values=values,
        points_by_candidate=points_by_candidate,
        first_failure_by_candidate=first_failure_by_candidate,
    )


def architecture_decision(
    profile: ArchitectureTrackProfile,
    *,
    architecture_id: str,
    scale_value: float | None = None,
) -> ArchitectureDecisionResult:
    """Return the decision memo fields for a selected architecture."""
    evaluations = architecture_signature(profile, scale_value=scale_value)
    selected = next((item for item in evaluations if item.architecture_id == architecture_id), evaluations[0])
    rejected = tuple(
        f"{item.label}: {item.dominant_constraint}; {'feasible' if item.feasible else 'not feasible'}"
        for item in evaluations
        if item.architecture_id != selected.architecture_id
    )
    summary = (
        f"Choose {selected.label} for {profile.label}; dominant constraint is "
        f"{selected.dominant_constraint}, with {selected.next_failure}."
    )
    return ArchitectureDecisionResult(
        selected_id=selected.architecture_id,
        selected_label=selected.label,
        feasible=selected.feasible,
        dominant_constraint=selected.dominant_constraint,
        rejected_alternatives=rejected,
        next_failure=selected.next_failure,
        validation_requirement=selected.validation_requirement,
        residual_risk=selected.residual_risk,
        memo_summary=summary,
    )


__all__ = [
    "ArchitectureCandidate",
    "ArchitectureDecisionResult",
    "ArchitectureEvaluation",
    "ArchitectureScalingPoint",
    "ArchitectureScalingResult",
    "ArchitectureTrackProfile",
    "architecture_decision",
    "architecture_scaling_curve",
    "architecture_signature",
    "architecture_track_profile",
]
