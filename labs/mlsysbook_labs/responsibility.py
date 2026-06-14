"""Responsible-engineering helpers for track-aware lab scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class ResponsibilityTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    tdp_w: float
    battery_wh: float | None
    stakeholder: str
    harmed_party: str
    obligation: str
    audit_signal: str
    subgroup_a: str
    subgroup_b: str
    primary_metric: str
    guardrail_metric: str
    baseline_quality_pct: float
    baseline_gap_pp: float
    target_gap_pp: float
    fairness_sensitivity: float
    explanation_features: int
    explanation_method: str
    explanation_coverage_pct: float
    base_latency_ms: float
    latency_slo_ms: float
    inference_events_per_day: int
    retrain_frequency_per_year: int
    train_energy_kwh: float
    grid_ci_g_per_kwh: float
    max_energy_factor: float
    max_cost_factor: float
    governance_delay_days: int
    validation_tests: tuple[str, ...]
    report_artifact: str
    residual_harm: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class MetricConflictResult:
    subgroup_a: str
    subgroup_b: str
    base_rate_a_pct: float
    base_rate_b_pct: float
    threshold: float
    accuracy_a_pct: float
    accuracy_b_pct: float
    fpr_a_pct: float
    fpr_b_pct: float
    fnr_a_pct: float
    fnr_b_pct: float
    ppv_a_pct: float
    ppv_b_pct: float
    accuracy_gap_pp: float
    fpr_gap_pp: float
    ppv_gap_pp: float
    harmed_party: str
    conflict_summary: str


@dataclass(frozen=True)
class ResponsibilityBudgetResult:
    privacy_level: float
    explanation_coverage_pct: float
    robustness_level: float
    monitoring_level: float
    latency_ms: float
    latency_slo_ms: float
    energy_factor: float
    cost_factor: float
    quality_delta_pp: float
    estimated_quality_pct: float
    fairness_gap_pp: float
    target_gap_pp: float
    governance_delay_days: float
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class ExplanationOverheadResult:
    method: str
    features: int
    base_latency_ms: float
    multiplier: float
    explanation_latency_ms: float
    total_latency_ms: float
    coverage_pct: float
    p99_added_ms: float
    slo_ms: float
    slo_ok: bool


@dataclass(frozen=True)
class CarbonBudgetResult:
    retrain_frequency_per_year: int
    train_energy_kwh: float
    grid_ci_g_per_kwh: float
    explanation_coverage_pct: float
    inference_events_per_day: int
    base_serving_kwh_per_year: float
    explanation_kwh_per_year: float
    retraining_kwh_per_year: float
    baseline_kwh_per_year: float
    total_kwh_per_year: float
    baseline_kgco2_per_year: float
    total_kgco2_per_year: float
    carbon_multiplier: float


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def responsibility_track_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> ResponsibilityTrackProfile:
    """Build a source-traced responsibility profile from variant defaults."""
    defaults = variant.defaults
    subgroups = tuple(str(item) for item in defaults.get("subgroups", ("Group A", "Group B")))
    subgroup_a = subgroups[0] if subgroups else "Group A"
    subgroup_b = subgroups[1] if len(subgroups) > 1 else "Group B"

    return ResponsibilityTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        tdp_w=_quantity_to_float(getattr(hardware, "tdp", None), "W", 0.0),
        battery_wh=(
            _quantity_to_float(getattr(hardware, "battery_capacity", None), "Wh", 0.0)
            if getattr(hardware, "battery_capacity", None) is not None
            else None
        ),
        stakeholder=variant.stakeholder,
        harmed_party=str(defaults.get("harmed_party", "affected users")),
        obligation=str(defaults.get("obligation", "responsible system behavior")),
        audit_signal=str(defaults.get("audit_signal", "post-deployment audit")),
        subgroup_a=subgroup_a,
        subgroup_b=subgroup_b,
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        baseline_quality_pct=float(defaults.get("baseline_quality_pct", 92.0)),
        baseline_gap_pp=float(defaults.get("baseline_gap_pp", 12.0)),
        target_gap_pp=float(defaults.get("target_gap_pp", 5.0)),
        fairness_sensitivity=float(defaults.get("fairness_sensitivity", 0.35)),
        explanation_features=int(defaults.get("explanation_features", 40)),
        explanation_method=str(defaults.get("explanation_method", "shap")),
        explanation_coverage_pct=float(defaults.get("explanation_coverage_pct", 10.0)),
        base_latency_ms=float(defaults.get("base_latency_ms", 10.0)),
        latency_slo_ms=float(defaults.get("latency_slo_ms", 100.0)),
        inference_events_per_day=int(defaults.get("inference_events_per_day", 100_000)),
        retrain_frequency_per_year=int(defaults.get("retrain_frequency_per_year", 12)),
        train_energy_kwh=float(defaults.get("train_energy_kwh", 100.0)),
        grid_ci_g_per_kwh=float(defaults.get("grid_ci_g_per_kwh", 369.0)),
        max_energy_factor=float(defaults.get("max_energy_factor", 2.0)),
        max_cost_factor=float(defaults.get("max_cost_factor", 2.0)),
        governance_delay_days=int(defaults.get("governance_delay_days", 7)),
        validation_tests=tuple(str(item) for item in defaults.get("validation_tests", ())),
        report_artifact=str(variant.assumptions.get("report_artifact", "responsible engineering decision memo")),
        residual_harm=str(defaults.get("residual_harm", "residual harm remains after mitigation")),
        source_refs=(variant.hardware_ref, variant.model_ref),
    )


def metric_conflict(
    profile: ResponsibilityTrackProfile,
    *,
    base_rate_a_pct: float,
    base_rate_b_pct: float,
    threshold: float,
) -> MetricConflictResult:
    """Estimate subgroup metric conflict under different base rates or contexts."""
    base_a = _clamp(base_rate_a_pct, 1.0, 99.0) / 100.0
    base_b = _clamp(base_rate_b_pct, 1.0, 99.0) / 100.0
    threshold = _clamp(threshold, 0.05, 0.95)

    raw_gap = profile.baseline_gap_pp + (base_a - base_b) * 20.0 * profile.fairness_sensitivity
    raw_gap += (0.55 - threshold) * 5.0
    gap = _clamp(abs(raw_gap), 0.5, 45.0)
    base_fpr = _clamp(4.0 + threshold * 10.0 + (0.25 - base_a) * 8.0, 1.0, 60.0)
    if base_a >= base_b:
        fpr_a = base_fpr
        fpr_b = _clamp(base_fpr + gap, 1.0, 95.0)
    else:
        fpr_b = base_fpr
        fpr_a = _clamp(base_fpr + gap, 1.0, 95.0)

    def _metrics(base: float, fpr_pct: float) -> tuple[float, float, float, float]:
        fpr = fpr_pct / 100.0
        tpr = _clamp(0.84 + (0.50 - threshold) * 0.22 + base * 0.10 - fpr * 0.05, 0.20, 0.995)
        fnr = 1.0 - tpr
        accuracy = base * tpr + (1.0 - base) * (1.0 - fpr)
        positives = base * tpr
        predicted_positive = positives + (1.0 - base) * fpr
        ppv = positives / predicted_positive if predicted_positive > 0 else 0.0
        return accuracy * 100.0, fpr * 100.0, fnr * 100.0, ppv * 100.0

    acc_a, fpr_a, fnr_a, ppv_a = _metrics(base_a, fpr_a)
    acc_b, fpr_b, fnr_b, ppv_b = _metrics(base_b, fpr_b)
    fpr_gap = abs(fpr_a - fpr_b)
    ppv_gap = abs(ppv_a - ppv_b)
    acc_gap = abs(acc_a - acc_b)
    if fpr_gap > profile.target_gap_pp:
        summary = (
            f"{profile.harmed_party} sees a {fpr_gap:.1f} pp error-rate gap; "
            f"the obligation is {profile.obligation}."
        )
    else:
        summary = (
            f"The error-rate gap is within the {profile.target_gap_pp:.1f} pp target, "
            f"but {profile.audit_signal} still needs to confirm it."
        )
    return MetricConflictResult(
        subgroup_a=profile.subgroup_a,
        subgroup_b=profile.subgroup_b,
        base_rate_a_pct=base_a * 100.0,
        base_rate_b_pct=base_b * 100.0,
        threshold=threshold,
        accuracy_a_pct=acc_a,
        accuracy_b_pct=acc_b,
        fpr_a_pct=fpr_a,
        fpr_b_pct=fpr_b,
        fnr_a_pct=fnr_a,
        fnr_b_pct=fnr_b,
        ppv_a_pct=ppv_a,
        ppv_b_pct=ppv_b,
        accuracy_gap_pp=acc_gap,
        fpr_gap_pp=fpr_gap,
        ppv_gap_pp=ppv_gap,
        harmed_party=profile.harmed_party,
        conflict_summary=summary,
    )


def explanation_overhead(
    profile: ResponsibilityTrackProfile,
    *,
    method: str,
    features: int,
    coverage_pct: float,
) -> ExplanationOverheadResult:
    """Compute explanation latency overhead for common teaching methods."""
    method_key = method.strip().lower().replace(" ", "_").replace("-", "_")
    features = max(1, int(features))
    coverage = _clamp(coverage_pct, 0.0, 100.0)
    if method_key in {"none", "off"}:
        multiplier = 0.0
    elif method_key in {"feature_importance", "fi"}:
        multiplier = 1.0
    elif method_key == "lime":
        multiplier = max(5.0, math.sqrt(features) * 2.5)
    elif method_key in {"trace_replay", "replay"}:
        multiplier = max(4.0, features / 2.0)
    elif method_key in {"counterfactual", "counterfactuals"}:
        multiplier = max(8.0, features * 0.75)
    else:
        multiplier = float(features)

    explanation_latency = profile.base_latency_ms * multiplier
    total_latency = profile.base_latency_ms + explanation_latency
    p99_added = explanation_latency * (coverage / 100.0)
    return ExplanationOverheadResult(
        method=method_key,
        features=features,
        base_latency_ms=profile.base_latency_ms,
        multiplier=multiplier,
        explanation_latency_ms=explanation_latency,
        total_latency_ms=total_latency,
        coverage_pct=coverage,
        p99_added_ms=p99_added,
        slo_ms=profile.latency_slo_ms,
        slo_ok=total_latency <= profile.latency_slo_ms,
    )


def responsibility_budget(
    profile: ResponsibilityTrackProfile,
    *,
    privacy_level: float,
    explanation_coverage_pct: float,
    robustness_level: float,
    monitoring_level: float,
) -> ResponsibilityBudgetResult:
    """Score a responsibility budget across quality, latency, energy, and cost."""
    privacy = _clamp(privacy_level, 0.0, 100.0)
    explanation = _clamp(explanation_coverage_pct, 0.0, 100.0)
    robustness = _clamp(robustness_level, 0.0, 100.0)
    monitoring = _clamp(monitoring_level, 0.0, 100.0)

    explanation_result = explanation_overhead(
        profile,
        method=profile.explanation_method,
        features=profile.explanation_features,
        coverage_pct=explanation,
    )
    latency = profile.base_latency_ms + explanation_result.p99_added_ms
    latency *= 1.0 + privacy / 350.0 + robustness / 500.0 + monitoring / 800.0
    energy_factor = 1.0 + privacy / 260.0 + explanation / 180.0 + robustness / 220.0 + monitoring / 420.0
    cost_factor = 1.0 + privacy / 220.0 + explanation / 300.0 + robustness / 160.0 + monitoring / 150.0
    quality_delta = -privacy * 0.018 - explanation * 0.006 + robustness * 0.026 + monitoring * 0.010
    estimated_quality = profile.baseline_quality_pct + quality_delta
    fairness_gap = max(0.0, profile.baseline_gap_pp - robustness * 0.070 - monitoring * 0.030)
    governance_delay = profile.governance_delay_days * (1.0 + monitoring / 300.0 + robustness / 400.0)

    violations: list[str] = []
    if latency > profile.latency_slo_ms:
        violations.append("latency SLO exceeded")
    if energy_factor > profile.max_energy_factor:
        violations.append("energy budget exceeded")
    if cost_factor > profile.max_cost_factor:
        violations.append("cost budget exceeded")
    if fairness_gap > profile.target_gap_pp:
        violations.append("responsibility gap above target")
    if estimated_quality < profile.baseline_quality_pct - 4.0:
        violations.append("quality loss too high")

    return ResponsibilityBudgetResult(
        privacy_level=privacy,
        explanation_coverage_pct=explanation,
        robustness_level=robustness,
        monitoring_level=monitoring,
        latency_ms=latency,
        latency_slo_ms=profile.latency_slo_ms,
        energy_factor=energy_factor,
        cost_factor=cost_factor,
        quality_delta_pp=quality_delta,
        estimated_quality_pct=estimated_quality,
        fairness_gap_pp=fairness_gap,
        target_gap_pp=profile.target_gap_pp,
        governance_delay_days=governance_delay,
        feasible=not violations,
        violations=tuple(violations),
    )


def carbon_budget(
    profile: ResponsibilityTrackProfile,
    *,
    retrain_frequency_per_year: int,
    explanation_coverage_pct: float,
    grid_ci_g_per_kwh: float,
) -> CarbonBudgetResult:
    """Compute annual carbon for retraining plus explanation overhead."""
    retrains = max(1, int(retrain_frequency_per_year))
    coverage = _clamp(explanation_coverage_pct, 0.0, 100.0)
    grid_ci = max(0.0, float(grid_ci_g_per_kwh))
    tdp_w = max(profile.tdp_w, 0.001)
    base_event_kwh = tdp_w * (profile.base_latency_ms / 1000.0) / 3_600_000.0
    events_per_year = max(0, profile.inference_events_per_day) * 365
    base_serving = events_per_year * base_event_kwh
    explanation_result = explanation_overhead(
        profile,
        method=profile.explanation_method,
        features=profile.explanation_features,
        coverage_pct=coverage,
    )
    explanation_kwh = base_serving * explanation_result.multiplier * (coverage / 100.0)
    retraining_kwh = retrains * profile.train_energy_kwh
    baseline_kwh = profile.train_energy_kwh + base_serving
    total_kwh = retraining_kwh + base_serving + explanation_kwh
    baseline_kg = baseline_kwh * grid_ci / 1000.0
    total_kg = total_kwh * grid_ci / 1000.0
    multiplier = total_kg / baseline_kg if baseline_kg > 0 else 0.0
    return CarbonBudgetResult(
        retrain_frequency_per_year=retrains,
        train_energy_kwh=profile.train_energy_kwh,
        grid_ci_g_per_kwh=grid_ci,
        explanation_coverage_pct=coverage,
        inference_events_per_day=profile.inference_events_per_day,
        base_serving_kwh_per_year=base_serving,
        explanation_kwh_per_year=explanation_kwh,
        retraining_kwh_per_year=retraining_kwh,
        baseline_kwh_per_year=baseline_kwh,
        total_kwh_per_year=total_kwh,
        baseline_kgco2_per_year=baseline_kg,
        total_kgco2_per_year=total_kg,
        carbon_multiplier=multiplier,
    )


__all__ = [
    "CarbonBudgetResult",
    "ExplanationOverheadResult",
    "MetricConflictResult",
    "ResponsibilityBudgetResult",
    "ResponsibilityTrackProfile",
    "carbon_budget",
    "explanation_overhead",
    "metric_conflict",
    "responsibility_budget",
    "responsibility_track_profile",
]
