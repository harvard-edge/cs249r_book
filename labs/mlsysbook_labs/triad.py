"""Data-Algorithm-Machine triad helpers for track-aware diagnosis labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class TriadTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    stakeholder: str
    failure_story: str
    data_axis: str
    algorithm_axis: str
    machine_axis: str
    data_threshold_pct: float
    algorithm_threshold_pct: float
    machine_threshold_pct: float
    default_data_pct: float
    default_algorithm_pct: float
    default_machine_pct: float
    intervention_options: tuple[str, ...]
    validation_tests: tuple[str, ...]
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class TriadDiagnosisResult:
    data_score_pct: float
    algorithm_score_pct: float
    machine_score_pct: float
    data_threshold_pct: float
    algorithm_threshold_pct: float
    machine_threshold_pct: float
    binding_axis: str
    primary_metric: str
    guardrail_metric: str
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class InterventionFrontierResult:
    data_budget_pct: float
    algorithm_budget_pct: float
    machine_budget_pct: float
    data_score_pct: float
    algorithm_score_pct: float
    machine_score_pct: float
    binding_axis: str
    selected_intervention: str
    best_intervention: str
    selected_score_pct: float
    best_score_pct: float
    feasible: bool
    rejected_alternatives: tuple[str, ...]


def triad_track_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> TriadTrackProfile:
    """Build a source-traced D-A-M profile from variant defaults."""
    defaults = variant.defaults
    return TriadTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        stakeholder=variant.stakeholder,
        failure_story=str(defaults.get("failure_story", variant.workload_summary)),
        data_axis=str(defaults.get("data_axis", "data coverage and freshness")),
        algorithm_axis=str(defaults.get("algorithm_axis", "model architecture and capacity")),
        machine_axis=str(defaults.get("machine_axis", "deployment hardware and runtime")),
        data_threshold_pct=float(defaults.get("data_threshold_pct", 70.0)),
        algorithm_threshold_pct=float(defaults.get("algorithm_threshold_pct", 70.0)),
        machine_threshold_pct=float(defaults.get("machine_threshold_pct", 70.0)),
        default_data_pct=float(defaults.get("default_data_pct", 60.0)),
        default_algorithm_pct=float(defaults.get("default_algorithm_pct", 60.0)),
        default_machine_pct=float(defaults.get("default_machine_pct", 60.0)),
        intervention_options=tuple(str(item) for item in defaults.get("intervention_options", ())),
        validation_tests=tuple(str(item) for item in defaults.get("validation_tests", ())),
        report_artifact=str(variant.assumptions.get("report_artifact", "triad diagnosis memo")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=(variant.hardware_ref, variant.model_ref),
    )


def diagnose_triad(
    profile: TriadTrackProfile,
    *,
    data_score_pct: float,
    algorithm_score_pct: float,
    machine_score_pct: float,
) -> TriadDiagnosisResult:
    """Diagnose the binding D-A-M axis under track-specific thresholds."""
    data = max(0.0, min(100.0, float(data_score_pct)))
    algorithm = max(0.0, min(100.0, float(algorithm_score_pct)))
    machine = max(0.0, min(100.0, float(machine_score_pct)))
    margins = {
        "Data": data - profile.data_threshold_pct,
        "Algorithm": algorithm - profile.algorithm_threshold_pct,
        "Machine": machine - profile.machine_threshold_pct,
    }
    binding = min(margins, key=margins.get)
    violations = tuple(axis for axis, margin in margins.items() if margin < 0)
    return TriadDiagnosisResult(
        data_score_pct=data,
        algorithm_score_pct=algorithm,
        machine_score_pct=machine,
        data_threshold_pct=profile.data_threshold_pct,
        algorithm_threshold_pct=profile.algorithm_threshold_pct,
        machine_threshold_pct=profile.machine_threshold_pct,
        binding_axis=binding,
        primary_metric=profile.primary_metric,
        guardrail_metric=profile.guardrail_metric,
        feasible=not violations,
        violations=violations,
    )


def intervention_frontier(
    profile: TriadTrackProfile,
    *,
    data_budget_pct: float,
    algorithm_budget_pct: float,
    machine_budget_pct: float,
    selected_intervention: str,
) -> InterventionFrontierResult:
    """Score a fixed-budget intervention across the D-A-M axes."""
    data_budget = max(0.0, min(100.0, float(data_budget_pct)))
    algorithm_budget = max(0.0, min(100.0, float(algorithm_budget_pct)))
    machine_budget = max(0.0, min(100.0, float(machine_budget_pct)))
    total = data_budget + algorithm_budget + machine_budget
    if total <= 0:
        data_share = algorithm_share = machine_share = 0.0
    else:
        data_share = data_budget / total
        algorithm_share = algorithm_budget / total
        machine_share = machine_budget / total

    # Teaching model: investment has the largest marginal return on the currently
    # weak axis, but still cannot make an unrelated axis disappear.
    data_score = min(100.0, profile.default_data_pct + data_share * 45.0 + algorithm_share * 5.0)
    algorithm_score = min(100.0, profile.default_algorithm_pct + algorithm_share * 45.0 + machine_share * 5.0)
    machine_score = min(100.0, profile.default_machine_pct + machine_share * 45.0 + data_share * 5.0)
    diagnosis = diagnose_triad(
        profile,
        data_score_pct=data_score,
        algorithm_score_pct=algorithm_score,
        machine_score_pct=machine_score,
    )
    scores = {
        "Data": data_score - profile.data_threshold_pct,
        "Algorithm": algorithm_score - profile.algorithm_threshold_pct,
        "Machine": machine_score - profile.machine_threshold_pct,
    }
    best = max(scores, key=scores.get)
    selected = selected_intervention.strip().title()
    if selected not in scores:
        selected = diagnosis.binding_axis
    rejected = tuple(axis for axis in ("Data", "Algorithm", "Machine") if axis != selected)
    return InterventionFrontierResult(
        data_budget_pct=data_budget,
        algorithm_budget_pct=algorithm_budget,
        machine_budget_pct=machine_budget,
        data_score_pct=data_score,
        algorithm_score_pct=algorithm_score,
        machine_score_pct=machine_score,
        binding_axis=diagnosis.binding_axis,
        selected_intervention=selected,
        best_intervention=best,
        selected_score_pct=scores[selected],
        best_score_pct=scores[best],
        feasible=diagnosis.feasible,
        rejected_alternatives=rejected,
    )


__all__ = [
    "InterventionFrontierResult",
    "TriadDiagnosisResult",
    "TriadTrackProfile",
    "diagnose_triad",
    "intervention_frontier",
    "triad_track_profile",
]
