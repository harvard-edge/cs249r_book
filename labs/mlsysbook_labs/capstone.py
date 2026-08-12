"""Capstone helpers for track-aware architecture audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class CapstoneTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    tdp_w: float
    memory_gb: float
    battery_wh: float | None
    architecture_goal: str
    architecture_components: tuple[str, ...]
    prior_decisions: tuple[Mapping[str, Any], ...]
    sensitivity_defaults: Mapping[str, float]
    revision_options: tuple[str, ...]
    top_risks: tuple[str, ...]
    durable_principle: str
    validation_tests: tuple[str, ...]
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class LedgerDecision:
    chapter: int
    label: str
    decision: str
    constraint: str
    source: str
    confidence_pct: float


@dataclass(frozen=True)
class LedgerReplayResult:
    track_id: str
    entries_found: int
    entries_expected: int
    coverage_pct: float
    decisions: tuple[LedgerDecision, ...]
    missing_chapters: tuple[int, ...]
    architecture_summary: str


@dataclass(frozen=True)
class SensitivityAxis:
    name: str
    value: float
    limit: float
    risk_pct: float
    status: str
    mitigation: str


@dataclass(frozen=True)
class SensitivityAuditResult:
    track_id: str
    axes: tuple[SensitivityAxis, ...]
    most_fragile: str
    feasible: bool
    violations: tuple[str, ...]
    fragility_score_pct: float


@dataclass(frozen=True)
class ArchitectureMemoResult:
    track_id: str
    revised_decision: str
    top_risk: str
    mitigation: str
    durable_principle: str
    validation_tests: tuple[str, ...]
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


def _history_get(history: Mapping[Any, Any], chapter: int) -> Mapping[str, Any] | None:
    for key in (chapter, str(chapter)):
        value = history.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def capstone_track_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> CapstoneTrackProfile:
    """Build a source-traced capstone profile from variant defaults."""
    defaults = variant.defaults
    memory = getattr(hardware, "memory", None)
    return CapstoneTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        tdp_w=_quantity_to_float(getattr(hardware, "tdp", None), "W", 0.0),
        memory_gb=_quantity_to_float(getattr(memory, "capacity", None), "GB", 0.0),
        battery_wh=(
            _quantity_to_float(getattr(hardware, "battery_capacity", None), "Wh", 0.0)
            if getattr(hardware, "battery_capacity", None) is not None
            else None
        ),
        architecture_goal=str(defaults.get("architecture_goal", variant.objective)),
        architecture_components=tuple(str(item) for item in defaults.get("architecture_components", ())),
        prior_decisions=tuple(defaults.get("prior_decisions", ())),
        sensitivity_defaults=dict(defaults.get("sensitivity_defaults", {})),
        revision_options=tuple(str(item) for item in defaults.get("revision_options", ())),
        top_risks=tuple(str(item) for item in defaults.get("top_risks", ())),
        durable_principle=str(defaults.get("durable_principle", "Architecture is accumulated constraint management.")),
        validation_tests=tuple(str(item) for item in defaults.get("validation_tests", ())),
        report_artifact=str(variant.assumptions.get("report_artifact", "Volume I architecture memo")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=(variant.hardware_ref, variant.model_ref),
    )


def replay_ledger(
    profile: CapstoneTrackProfile,
    ledger_history: Mapping[Any, Any],
) -> LedgerReplayResult:
    """Replay student ledger entries, falling back to track presets when missing."""
    decisions: list[LedgerDecision] = []
    missing: list[int] = []
    found = 0
    for item in profile.prior_decisions:
        chapter = int(item.get("chapter", 0))
        label = str(item.get("label", f"Lab {chapter}"))
        preset = str(item.get("preset", "Track preset decision"))
        constraint = str(item.get("constraint", profile.guardrail_metric))
        entry = _history_get(ledger_history, chapter)
        if entry:
            found += 1
            decision = str(
                entry.get("decision")
                or entry.get("final_decision")
                or entry.get("completed")
                or entry.get("chapter")
                or preset
            )
            source = "student ledger"
            confidence = 100.0
        else:
            missing.append(chapter)
            decision = preset
            source = "track preset"
            confidence = 65.0
        decisions.append(
            LedgerDecision(
                chapter=chapter,
                label=label,
                decision=decision,
                constraint=constraint,
                source=source,
                confidence_pct=confidence,
            )
        )

    expected = len(profile.prior_decisions)
    coverage = found / expected * 100.0 if expected else 0.0
    summary = (
        f"{profile.label} architecture uses {found}/{expected} student ledger entries; "
        f"missing entries use track presets."
    )
    return LedgerReplayResult(
        track_id=profile.track_id,
        entries_found=found,
        entries_expected=expected,
        coverage_pct=coverage,
        decisions=tuple(decisions),
        missing_chapters=tuple(missing),
        architecture_summary=summary,
    )


def sensitivity_audit(
    profile: CapstoneTrackProfile,
    *,
    workload_multiplier: float,
    model_growth_pct: float,
    guardrail_tightening_pct: float,
    evidence_confidence_pct: float,
) -> SensitivityAuditResult:
    """Score architecture fragility under four capstone perturbations."""
    defaults = profile.sensitivity_defaults
    workload_limit = float(defaults.get("workload_limit_multiplier", 2.0))
    model_growth_limit = float(defaults.get("model_growth_limit_pct", 30.0))
    guardrail_limit = float(defaults.get("guardrail_tightening_limit_pct", 30.0))
    evidence_floor = float(defaults.get("evidence_confidence_floor_pct", 70.0))

    workload = max(0.1, float(workload_multiplier))
    model_growth = max(0.0, float(model_growth_pct))
    guardrail_tightening = max(0.0, float(guardrail_tightening_pct))
    evidence_confidence = max(0.0, min(100.0, float(evidence_confidence_pct)))

    axes = (
        SensitivityAxis(
            name="Workload growth",
            value=workload,
            limit=workload_limit,
            risk_pct=min(100.0, workload / workload_limit * 100.0),
            status="PASS" if workload <= workload_limit else "FAIL",
            mitigation=str(defaults.get("workload_mitigation", "increase headroom or reduce demand")),
        ),
        SensitivityAxis(
            name="Model growth",
            value=model_growth,
            limit=model_growth_limit,
            risk_pct=min(100.0, model_growth / model_growth_limit * 100.0) if model_growth_limit > 0 else 100.0,
            status="PASS" if model_growth <= model_growth_limit else "FAIL",
            mitigation=str(defaults.get("model_mitigation", "compress, distill, or resize model")),
        ),
        SensitivityAxis(
            name="Guardrail tightening",
            value=guardrail_tightening,
            limit=guardrail_limit,
            risk_pct=min(100.0, guardrail_tightening / guardrail_limit * 100.0) if guardrail_limit > 0 else 100.0,
            status="PASS" if guardrail_tightening <= guardrail_limit else "FAIL",
            mitigation=str(defaults.get("guardrail_mitigation", "add margin or simplify path")),
        ),
        SensitivityAxis(
            name="Evidence confidence",
            value=evidence_confidence,
            limit=evidence_floor,
            risk_pct=min(100.0, (100.0 - evidence_confidence) / max(1.0, 100.0 - evidence_floor) * 100.0),
            status="PASS" if evidence_confidence >= evidence_floor else "FAIL",
            mitigation=str(defaults.get("evidence_mitigation", "run validation and fill ledger gaps")),
        ),
    )
    violations = tuple(axis.name for axis in axes if axis.status == "FAIL")
    most_fragile_axis = max(axes, key=lambda axis: axis.risk_pct)
    fragility_score = sum(axis.risk_pct for axis in axes) / len(axes)
    return SensitivityAuditResult(
        track_id=profile.track_id,
        axes=axes,
        most_fragile=most_fragile_axis.name,
        feasible=not violations,
        violations=violations,
        fragility_score_pct=fragility_score,
    )


def architecture_memo(
    profile: CapstoneTrackProfile,
    *,
    revised_decision: str,
    top_risk: str,
    mitigation: str,
) -> ArchitectureMemoResult:
    """Package the final capstone memo decision in a typed result object."""
    summary = (
        f"For {profile.label}, revise '{revised_decision}' because the top risk is "
        f"'{top_risk}'. Mitigation: {mitigation}."
    )
    return ArchitectureMemoResult(
        track_id=profile.track_id,
        revised_decision=revised_decision,
        top_risk=top_risk,
        mitigation=mitigation,
        durable_principle=profile.durable_principle,
        validation_tests=profile.validation_tests,
        memo_summary=summary,
    )


__all__ = [
    "ArchitectureMemoResult",
    "CapstoneTrackProfile",
    "LedgerDecision",
    "LedgerReplayResult",
    "SensitivityAuditResult",
    "SensitivityAxis",
    "architecture_memo",
    "capstone_track_profile",
    "replay_ledger",
    "sensitivity_audit",
]
