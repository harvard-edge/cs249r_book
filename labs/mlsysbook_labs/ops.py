"""Shared ML-operations helpers for track-aware degradation labs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class OpsTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    tdp_w: float
    battery_wh: float | None
    drift_source: str
    monitoring_signal: str
    rollback_policy: str
    escalation_policy: str
    baseline_quality_pct: float
    quality_floor_pct: float
    drift_rate_psi_per_day: float
    quality_loss_per_psi: float
    alert_threshold_psi: float
    label_delay_days: int
    retrain_cost: float
    drift_cost_per_day: float
    current_cadence_days: int
    monitoring_cost_per_day: float
    downstream_models: int
    base_loss_pp: float
    validation_tests: tuple[str, ...]
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class DriftVisibilityResult:
    days_since_deploy: int
    true_psi: float
    observed_psi: float
    true_quality_pct: float
    observed_quality_pct: float
    alert_day: int
    quality_breach_day: int
    detection_delay_days: int
    accumulated_damage_cost: float
    alert_triggered: bool
    quality_breached: bool


@dataclass(frozen=True)
class RetrainingCadenceResult:
    optimal_days: float
    current_days: int
    retrains_per_year: float
    optimal_annual_cost: float
    current_annual_cost: float
    savings_vs_current: float
    current_too_slow_factor: float


@dataclass(frozen=True)
class OpsPolicyResult:
    threshold_psi: float
    cadence_days: int
    canary_pct: float
    rollback_hours: float
    expected_detection_day: float
    stale_days: float
    annual_monitoring_cost: float
    annual_retrain_cost: float
    annual_risk_cost: float
    total_annual_cost: float
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class DebtCascadeResult:
    missed_cycles: int
    downstream_models: int
    base_loss_pp: float
    compound_loss_pp: float
    cascade_loss_pp: float
    total_loss_pp: float
    linear_loss_pp: float
    debt_multiplier: float


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


def ops_track_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> OpsTrackProfile:
    """Build a source-traced ML-operations profile from variant defaults."""
    defaults = variant.defaults
    return OpsTrackProfile(
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
        drift_source=str(defaults.get("drift_source", "production distribution drift")),
        monitoring_signal=str(defaults.get("monitoring_signal", "delayed labels and proxy metrics")),
        rollback_policy=str(defaults.get("rollback_policy", "canary rollback")),
        escalation_policy=str(defaults.get("escalation_policy", "owner on-call escalation")),
        baseline_quality_pct=float(defaults.get("baseline_quality_pct", 95.0)),
        quality_floor_pct=float(defaults.get("quality_floor_pct", 90.0)),
        drift_rate_psi_per_day=float(defaults.get("drift_rate_psi_per_day", 0.01)),
        quality_loss_per_psi=float(defaults.get("quality_loss_per_psi", 15.0)),
        alert_threshold_psi=float(defaults.get("alert_threshold_psi", 0.2)),
        label_delay_days=int(defaults.get("label_delay_days", 7)),
        retrain_cost=float(defaults.get("retrain_cost", 10_000.0)),
        drift_cost_per_day=float(defaults.get("drift_cost_per_day", 500.0)),
        current_cadence_days=int(defaults.get("current_cadence_days", 30)),
        monitoring_cost_per_day=float(defaults.get("monitoring_cost_per_day", 50.0)),
        downstream_models=int(defaults.get("downstream_models", 2)),
        base_loss_pp=float(defaults.get("base_loss_pp", 2.0)),
        validation_tests=tuple(str(item) for item in defaults.get("validation_tests", ())),
        report_artifact=str(variant.assumptions.get("report_artifact", "operations policy memo")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=(variant.hardware_ref, variant.model_ref),
    )


def drift_visibility(
    profile: OpsTrackProfile,
    *,
    days_since_deploy: int,
    drift_rate_psi_per_day: float | None = None,
    alert_threshold_psi: float | None = None,
) -> DriftVisibilityResult:
    """Compute true drift, observed drift, alert timing, and damage cost."""
    days = max(0, int(days_since_deploy))
    rate = max(0.00001, drift_rate_psi_per_day or profile.drift_rate_psi_per_day)
    threshold = max(0.0001, alert_threshold_psi or profile.alert_threshold_psi)
    label_delay = max(0, profile.label_delay_days)

    true_psi = rate * days
    observed_days = max(0, days - label_delay)
    observed_psi = rate * observed_days
    true_quality = max(0.0, profile.baseline_quality_pct - true_psi * profile.quality_loss_per_psi)
    observed_quality = max(0.0, profile.baseline_quality_pct - observed_psi * profile.quality_loss_per_psi)
    alert_day = math.ceil(threshold / rate + label_delay)
    quality_breach_psi = max(0.0, profile.baseline_quality_pct - profile.quality_floor_pct) / profile.quality_loss_per_psi
    quality_breach_day = math.ceil(quality_breach_psi / rate) if quality_breach_psi > 0 else 0
    detection_delay = max(0, alert_day - quality_breach_day)
    damage_days = max(0, days - quality_breach_day)
    damage_cost = damage_days * profile.drift_cost_per_day
    return DriftVisibilityResult(
        days_since_deploy=days,
        true_psi=true_psi,
        observed_psi=observed_psi,
        true_quality_pct=true_quality,
        observed_quality_pct=observed_quality,
        alert_day=alert_day,
        quality_breach_day=quality_breach_day,
        detection_delay_days=detection_delay,
        accumulated_damage_cost=damage_cost,
        alert_triggered=days >= alert_day,
        quality_breached=days >= quality_breach_day,
    )


def retraining_cadence(
    *,
    retrain_cost: float,
    drift_cost_per_day: float,
    current_days: int,
) -> RetrainingCadenceResult:
    """Compute the EOQ-style optimal retraining cadence."""
    cost = max(0.0, retrain_cost)
    drift_cost = max(0.0001, drift_cost_per_day)
    current = max(1, int(current_days))
    optimal = math.sqrt(2 * cost / drift_cost)

    def annual_cost(interval: float) -> float:
        retrain = 365 / interval * cost
        stale = drift_cost * 365 * interval / 2
        return retrain + stale

    optimal_annual = annual_cost(optimal)
    current_annual = annual_cost(current)
    return RetrainingCadenceResult(
        optimal_days=optimal,
        current_days=current,
        retrains_per_year=365 / optimal,
        optimal_annual_cost=optimal_annual,
        current_annual_cost=current_annual,
        savings_vs_current=current_annual - optimal_annual,
        current_too_slow_factor=current / optimal,
    )


def ops_policy(
    profile: OpsTrackProfile,
    *,
    threshold_psi: float,
    cadence_days: int,
    canary_pct: float,
    rollback_hours: float,
) -> OpsPolicyResult:
    """Score an operations policy under monitoring, retraining, and risk costs."""
    cadence = max(1, int(cadence_days))
    threshold = max(0.0001, threshold_psi)
    canary = max(0.0, min(100.0, canary_pct))
    rollback = max(0.0, rollback_hours)
    cadence_result = retraining_cadence(
        retrain_cost=profile.retrain_cost,
        drift_cost_per_day=profile.drift_cost_per_day,
        current_days=cadence,
    )
    expected_detection = threshold / max(0.00001, profile.drift_rate_psi_per_day) + profile.label_delay_days
    stale_days = max(0.0, cadence - cadence_result.optimal_days)
    annual_monitoring = profile.monitoring_cost_per_day * 365 * (1 + canary / 200)
    annual_retrain = 365 / cadence * profile.retrain_cost
    annual_risk = (expected_detection + stale_days + rollback / 24) * profile.drift_cost_per_day * 12
    violations: list[str] = []
    if threshold > profile.alert_threshold_psi * 1.5:
        violations.append("monitor threshold too loose")
    if cadence > cadence_result.optimal_days * 2:
        violations.append("cadence too slow")
    if canary < 5:
        violations.append("canary coverage too small")
    if rollback > 24:
        violations.append("rollback exposure too long")
    return OpsPolicyResult(
        threshold_psi=threshold,
        cadence_days=cadence,
        canary_pct=canary,
        rollback_hours=rollback,
        expected_detection_day=expected_detection,
        stale_days=stale_days,
        annual_monitoring_cost=annual_monitoring,
        annual_retrain_cost=annual_retrain,
        annual_risk_cost=annual_risk,
        total_annual_cost=annual_monitoring + annual_retrain + annual_risk,
        feasible=not violations,
        violations=tuple(violations),
    )


def debt_cascade(
    *,
    missed_cycles: int,
    downstream_models: int,
    base_loss_pp: float,
    exponent: float = 1.3,
) -> DebtCascadeResult:
    """Estimate compounding ML technical debt from missed retraining cycles."""
    missed = max(1, int(missed_cycles))
    downstream = max(0, int(downstream_models))
    base = max(0.0, base_loss_pp)
    compound = sum(base * (cycle**exponent) for cycle in range(1, missed + 1))
    cascade = missed * downstream * base * 0.3
    total = compound + cascade
    linear = missed * base
    multiplier = total / base if base > 0 else 0.0
    return DebtCascadeResult(
        missed_cycles=missed,
        downstream_models=downstream,
        base_loss_pp=base,
        compound_loss_pp=compound,
        cascade_loss_pp=cascade,
        total_loss_pp=total,
        linear_loss_pp=linear,
        debt_multiplier=multiplier,
    )


__all__ = [
    "DebtCascadeResult",
    "DriftVisibilityResult",
    "OpsPolicyResult",
    "OpsTrackProfile",
    "RetrainingCadenceResult",
    "debt_cascade",
    "drift_visibility",
    "ops_policy",
    "ops_track_profile",
    "retraining_cadence",
]
