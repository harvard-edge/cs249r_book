"""Data-selection helpers for track-aware quality, coverage, and cost labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class CoverageSubgroup:
    subgroup_id: str
    label: str
    baseline_coverage_pct: float
    risk_weight: float


@dataclass(frozen=True)
class DataPolicyOption:
    policy_id: str
    label: str
    selection_focus: str
    dataset_fraction_pct: float
    label_quality_pct: float
    noise_pct: float
    coverage_gain_pct: float
    rare_event_multiplier: float
    subgroup_adjustments: Mapping[str, float]
    privacy_risk: str
    system_cost_risk: str
    accepted_blind_spot: str
    next_data: str
    validation_requirement: str
    residual_risk: str


@dataclass(frozen=True)
class DataSelectionTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    stakeholder: str
    selection_story: str
    dataset_unit: str
    dataset_size_k: float
    cost_per_k: float
    compute_cost_per_k: float
    storage_mb_per_k: float
    cost_budget: float
    storage_budget_mb: float
    quality_floor_pct: float
    coverage_floor_pct: float
    rare_event_floor_pct: float
    base_coverage_pct: float
    base_rare_event_pct: float
    subgroups: tuple[CoverageSubgroup, ...]
    policy_options: tuple[DataPolicyOption, ...]
    validation_tests: tuple[str, ...]
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class SelectionUtilityResult:
    policy_id: str
    policy_label: str
    selected_examples_k: float
    quality_score_pct: float
    coverage_score_pct: float
    rare_event_score_pct: float
    utility_score: float
    acquisition_cost: float
    compute_cost: float
    storage_mb: float
    total_cost: float
    dominant_risk: str
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class CoverageCellResult:
    subgroup_id: str
    label: str
    coverage_pct: float
    risk_score: float
    status: str


@dataclass(frozen=True)
class CoverageProfileResult:
    policy_id: str
    policy_label: str
    cells: tuple[CoverageCellResult, ...]
    worst_subgroup: str
    worst_risk_score: float


@dataclass(frozen=True)
class DataPolicyDecisionResult:
    selected_id: str
    selected_label: str
    feasible: bool
    utility_score: float
    dominant_risk: str
    worst_subgroup: str
    accepted_blind_spot: str
    next_data: str
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


def _subgroups(defaults: Mapping[str, Any]) -> tuple[CoverageSubgroup, ...]:
    raw = defaults.get("subgroups", {})
    if not isinstance(raw, Mapping):
        raw = {}
    groups = []
    for subgroup_id, details_raw in raw.items():
        details = details_raw if isinstance(details_raw, Mapping) else {}
        groups.append(
            CoverageSubgroup(
                subgroup_id=str(subgroup_id),
                label=str(details.get("label", subgroup_id)),
                baseline_coverage_pct=float(details.get("baseline_coverage_pct", 50.0)),
                risk_weight=float(details.get("risk_weight", 1.0)),
            )
        )
    return tuple(groups)


def _policy_options(defaults: Mapping[str, Any]) -> tuple[DataPolicyOption, ...]:
    raw = defaults.get("policy_options", {})
    if not isinstance(raw, Mapping):
        raw = {}
    options = []
    for policy_id, details_raw in raw.items():
        details = details_raw if isinstance(details_raw, Mapping) else {}
        adjustments = details.get("subgroup_adjustments", {})
        if not isinstance(adjustments, Mapping):
            adjustments = {}
        options.append(
            DataPolicyOption(
                policy_id=str(policy_id),
                label=str(details.get("label", policy_id)),
                selection_focus=str(details.get("selection_focus", "selection focus not specified")),
                dataset_fraction_pct=float(details.get("dataset_fraction_pct", 50.0)),
                label_quality_pct=float(details.get("label_quality_pct", 80.0)),
                noise_pct=float(details.get("noise_pct", 10.0)),
                coverage_gain_pct=float(details.get("coverage_gain_pct", 10.0)),
                rare_event_multiplier=float(details.get("rare_event_multiplier", 1.0)),
                subgroup_adjustments={str(key): float(value) for key, value in adjustments.items()},
                privacy_risk=str(details.get("privacy_risk", "privacy risk not specified")),
                system_cost_risk=str(details.get("system_cost_risk", "system cost risk not specified")),
                accepted_blind_spot=str(details.get("accepted_blind_spot", "accepted blind spot not specified")),
                next_data=str(details.get("next_data", "next data not specified")),
                validation_requirement=str(details.get("validation_requirement", "validation not specified")),
                residual_risk=str(details.get("residual_risk", "residual data risk not specified")),
            )
        )
    if options:
        return tuple(options)
    return (
        DataPolicyOption(
            "baseline",
            "Baseline data policy",
            "baseline",
            50.0,
            80.0,
            10.0,
            10.0,
            1.0,
            {},
            "baseline privacy risk",
            "baseline cost risk",
            "baseline blind spot",
            "baseline next data",
            "baseline validation",
            "no mitigation selected",
        ),
    )


def data_selection_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> DataSelectionTrackProfile:
    """Build a source-traced data-selection profile from variant defaults."""
    defaults = variant.defaults
    return DataSelectionTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        stakeholder=variant.stakeholder,
        selection_story=str(defaults.get("selection_story", variant.workload_summary)),
        dataset_unit=str(defaults.get("dataset_unit", "examples")),
        dataset_size_k=float(defaults.get("dataset_size_k", 1000.0)),
        cost_per_k=float(defaults.get("cost_per_k", 1.0)),
        compute_cost_per_k=float(defaults.get("compute_cost_per_k", 1.0)),
        storage_mb_per_k=float(defaults.get("storage_mb_per_k", 1.0)),
        cost_budget=float(defaults.get("cost_budget", 1000.0)),
        storage_budget_mb=float(defaults.get("storage_budget_mb", 10000.0)),
        quality_floor_pct=float(defaults.get("quality_floor_pct", 80.0)),
        coverage_floor_pct=float(defaults.get("coverage_floor_pct", 80.0)),
        rare_event_floor_pct=float(defaults.get("rare_event_floor_pct", 50.0)),
        base_coverage_pct=float(defaults.get("base_coverage_pct", 50.0)),
        base_rare_event_pct=float(defaults.get("base_rare_event_pct", 30.0)),
        subgroups=_subgroups(defaults),
        policy_options=_policy_options(defaults),
        validation_tests=_tuple_str(defaults.get("validation_tests")),
        report_artifact=str(variant.assumptions.get("report_artifact", "data selection policy memo")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=tuple(ref for ref in (variant.hardware_ref, variant.model_ref, variant.system_ref) if ref),
    )


def _policy(profile: DataSelectionTrackProfile, policy_id: str) -> DataPolicyOption:
    return next(
        (option for option in profile.policy_options if option.policy_id == policy_id),
        profile.policy_options[0],
    )


def selection_utility(
    profile: DataSelectionTrackProfile,
    *,
    policy_id: str,
    fraction_multiplier: float = 1.0,
) -> SelectionUtilityResult:
    """Evaluate utility, cost, coverage, rare-event score, and feasibility."""
    policy = _policy(profile, policy_id)
    fraction = max(1.0, min(100.0, policy.dataset_fraction_pct * float(fraction_multiplier)))
    selected_k = profile.dataset_size_k * fraction / 100.0
    volume_bonus = min(10.0, fraction / 10.0)
    quality = min(100.0, policy.label_quality_pct - 0.5 * policy.noise_pct + volume_bonus)
    coverage = min(100.0, profile.base_coverage_pct + policy.coverage_gain_pct)
    rare = min(100.0, profile.base_rare_event_pct * policy.rare_event_multiplier)
    utility = 0.45 * quality + 0.35 * coverage + 0.20 * rare
    acquisition = selected_k * profile.cost_per_k
    compute = selected_k * profile.compute_cost_per_k
    storage = selected_k * profile.storage_mb_per_k
    total_cost = acquisition + compute
    ratios = {
        "quality": profile.quality_floor_pct / max(quality, 1e-9),
        "coverage": profile.coverage_floor_pct / max(coverage, 1e-9),
        "rare-event coverage": profile.rare_event_floor_pct / max(rare, 1e-9),
        "cost": total_cost / max(profile.cost_budget, 1e-9),
        "storage": storage / max(profile.storage_budget_mb, 1e-9),
    }
    dominant = max(ratios, key=ratios.get)
    violations = []
    if quality < profile.quality_floor_pct:
        violations.append(f"quality {quality:.1f}% < {profile.quality_floor_pct:.1f}%")
    if coverage < profile.coverage_floor_pct:
        violations.append(f"coverage {coverage:.1f}% < {profile.coverage_floor_pct:.1f}%")
    if rare < profile.rare_event_floor_pct:
        violations.append(f"rare-event coverage {rare:.1f}% < {profile.rare_event_floor_pct:.1f}%")
    if total_cost > profile.cost_budget:
        violations.append(f"cost {total_cost:.1f} > {profile.cost_budget:.1f}")
    if storage > profile.storage_budget_mb:
        violations.append(f"storage {storage:.1f} MB > {profile.storage_budget_mb:.1f} MB")
    return SelectionUtilityResult(
        policy_id=policy.policy_id,
        policy_label=policy.label,
        selected_examples_k=selected_k,
        quality_score_pct=quality,
        coverage_score_pct=coverage,
        rare_event_score_pct=rare,
        utility_score=utility,
        acquisition_cost=acquisition,
        compute_cost=compute,
        storage_mb=storage,
        total_cost=total_cost,
        dominant_risk=dominant,
        feasible=not violations,
        violations=tuple(violations),
    )


def selection_frontier(
    profile: DataSelectionTrackProfile,
    *,
    fraction_multiplier: float = 1.0,
) -> tuple[SelectionUtilityResult, ...]:
    """Evaluate every policy for the selected fraction multiplier."""
    return tuple(
        selection_utility(profile, policy_id=policy.policy_id, fraction_multiplier=fraction_multiplier)
        for policy in profile.policy_options
    )


def coverage_profile(
    profile: DataSelectionTrackProfile,
    *,
    policy_id: str,
) -> CoverageProfileResult:
    """Compute subgroup coverage and risk for a selected policy."""
    policy = _policy(profile, policy_id)
    cells = []
    for subgroup in profile.subgroups:
        adjustment = policy.subgroup_adjustments.get(subgroup.subgroup_id, 0.0)
        coverage = max(0.0, min(100.0, subgroup.baseline_coverage_pct + policy.coverage_gain_pct + adjustment))
        risk = max(0.0, profile.coverage_floor_pct - coverage) * subgroup.risk_weight
        status = "ok" if risk == 0 else "risk"
        cells.append(
            CoverageCellResult(
                subgroup_id=subgroup.subgroup_id,
                label=subgroup.label,
                coverage_pct=coverage,
                risk_score=risk,
                status=status,
            )
        )
    worst = max(cells, key=lambda item: item.risk_score) if cells else None
    return CoverageProfileResult(
        policy_id=policy.policy_id,
        policy_label=policy.label,
        cells=tuple(cells),
        worst_subgroup=worst.label if worst else "not recorded",
        worst_risk_score=worst.risk_score if worst else 0.0,
    )


def data_policy_decision(
    profile: DataSelectionTrackProfile,
    *,
    policy_id: str,
    fraction_multiplier: float = 1.0,
) -> DataPolicyDecisionResult:
    """Return the decision memo fields for a selected data policy."""
    policy = _policy(profile, policy_id)
    selected = selection_utility(profile, policy_id=policy.policy_id, fraction_multiplier=fraction_multiplier)
    coverage = coverage_profile(profile, policy_id=policy.policy_id)
    rejected = tuple(
        f"{item.policy_label}: {item.dominant_risk}; {'feasible' if item.feasible else 'not feasible'}"
        for item in selection_frontier(profile, fraction_multiplier=fraction_multiplier)
        if item.policy_id != policy.policy_id
    )
    summary = (
        f"Use {policy.label} for {profile.label}; dominant risk is {selected.dominant_risk}, "
        f"worst subgroup is {coverage.worst_subgroup}, and next data is {policy.next_data}."
    )
    return DataPolicyDecisionResult(
        selected_id=policy.policy_id,
        selected_label=policy.label,
        feasible=selected.feasible,
        utility_score=selected.utility_score,
        dominant_risk=selected.dominant_risk,
        worst_subgroup=coverage.worst_subgroup,
        accepted_blind_spot=policy.accepted_blind_spot,
        next_data=policy.next_data,
        validation_requirement=policy.validation_requirement,
        residual_risk=policy.residual_risk,
        rejected_alternatives=rejected,
        memo_summary=summary,
    )


__all__ = [
    "CoverageCellResult",
    "CoverageProfileResult",
    "CoverageSubgroup",
    "DataPolicyDecisionResult",
    "DataPolicyOption",
    "DataSelectionTrackProfile",
    "SelectionUtilityResult",
    "coverage_profile",
    "data_policy_decision",
    "data_selection_profile",
    "selection_frontier",
    "selection_utility",
]
