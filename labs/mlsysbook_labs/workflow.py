"""Workflow constraint-tax helpers for track-aware ML lifecycle labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class WorkflowGate:
    gate_id: str
    label: str
    stage: int
    validation_focus: str
    residual_risk: str


@dataclass(frozen=True)
class WorkflowTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    stakeholder: str
    constraint_name: str
    failure_story: str
    stage_names: tuple[str, ...]
    default_discovery_stage: int
    recommended_gate_stage: int
    base_rework_days: float
    base_cycle_days: float
    base_residual_risk_pct: float
    min_residual_risk_pct: float
    default_validation_depth_pct: float
    default_automation_pct: float
    default_hardware_realism_pct: float
    default_data_scale_pct: float
    gate_options: tuple[WorkflowGate, ...]
    release_policies: tuple[str, ...]
    rollback_rules: tuple[str, ...]
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class ConstraintTaxResult:
    discovery_stage: int
    discovery_stage_name: str
    recommended_stage: int
    recommended_stage_name: str
    cost_multiplier: float
    rework_days: float
    avoidable_rework_days: float
    artifacts_to_rebuild: tuple[str, ...]
    late_discovery: bool


@dataclass(frozen=True)
class IterationFrontierResult:
    validation_depth_pct: float
    automation_pct: float
    hardware_realism_pct: float
    data_scale_pct: float
    iteration_days: float
    confidence_pct: float
    residual_risk_pct: float
    bottleneck: str


@dataclass(frozen=True)
class WorkflowPolicyResult:
    gate_id: str
    gate_label: str
    release_policy: str
    rollback_rule: str
    gate_stage: int
    gate_stage_name: str
    rework_days_at_gate: float
    residual_risk_pct: float
    blind_spot: str
    policy_summary: str


def _stage_name(profile: WorkflowTrackProfile, stage: int) -> str:
    idx = max(1, min(stage, len(profile.stage_names))) - 1
    return profile.stage_names[idx]


def _workflow_gates(defaults: Mapping[str, Any]) -> tuple[WorkflowGate, ...]:
    raw_gates = defaults.get("gate_options", {})
    if not isinstance(raw_gates, Mapping):
        raw_gates = {}
    gates: list[WorkflowGate] = []
    for gate_id, raw in raw_gates.items():
        details = raw if isinstance(raw, Mapping) else {}
        gates.append(
            WorkflowGate(
                gate_id=str(gate_id),
                label=str(details.get("label", gate_id)),
                stage=int(details.get("stage", 2)),
                validation_focus=str(details.get("validation_focus", "deployment feasibility")),
                residual_risk=str(details.get("residual_risk", "unmodeled deployment behavior")),
            )
        )
    if gates:
        return tuple(gates)
    return (
        WorkflowGate(
            "early_gate",
            "Early deployment gate",
            2,
            "test the dominant deployment constraint before training",
            "prototype assumptions may still miss production tails",
        ),
        WorkflowGate(
            "late_gate",
            "Late deployment gate",
            5,
            "test deployment during release hardening",
            "expensive rework if the physical envelope fails",
        ),
    )


def workflow_track_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> WorkflowTrackProfile:
    """Build a workflow profile from canonical track and V1-03 variant data."""
    defaults = variant.defaults
    return WorkflowTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        stakeholder=variant.stakeholder,
        constraint_name=str(defaults.get("constraint_name", "deployment constraint")),
        failure_story=str(defaults.get("failure_story", variant.workload_summary)),
        stage_names=tuple(str(item) for item in defaults.get(
            "stage_names",
            ("Problem framing", "Data contract", "Model design", "Evaluation", "Release", "Monitoring"),
        )),
        default_discovery_stage=int(defaults.get("default_discovery_stage", 5)),
        recommended_gate_stage=int(defaults.get("recommended_gate_stage", 2)),
        base_rework_days=float(defaults.get("base_rework_days", 5.0)),
        base_cycle_days=float(defaults.get("base_cycle_days", 7.0)),
        base_residual_risk_pct=float(defaults.get("base_residual_risk_pct", 65.0)),
        min_residual_risk_pct=float(defaults.get("min_residual_risk_pct", 8.0)),
        default_validation_depth_pct=float(defaults.get("default_validation_depth_pct", 55.0)),
        default_automation_pct=float(defaults.get("default_automation_pct", 45.0)),
        default_hardware_realism_pct=float(defaults.get("default_hardware_realism_pct", 50.0)),
        default_data_scale_pct=float(defaults.get("default_data_scale_pct", 50.0)),
        gate_options=_workflow_gates(defaults),
        release_policies=tuple(str(item) for item in defaults.get("release_policies", ())),
        rollback_rules=tuple(str(item) for item in defaults.get("rollback_rules", ())),
        report_artifact=str(variant.assumptions.get("report_artifact", "workflow policy memo")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=tuple(ref for ref in (variant.hardware_ref, variant.model_ref, variant.system_ref) if ref),
    )


def constraint_tax(profile: WorkflowTrackProfile, *, discovery_stage: int) -> ConstraintTaxResult:
    """Compute the rework cost of discovering a constraint at a lifecycle stage."""
    stage = max(1, min(int(discovery_stage), len(profile.stage_names)))
    recommended = max(1, min(profile.recommended_gate_stage, len(profile.stage_names)))
    multiplier = 2 ** (stage - 1)
    recommended_multiplier = 2 ** (recommended - 1)
    rework_days = profile.base_rework_days * multiplier
    recommended_days = profile.base_rework_days * recommended_multiplier
    artifacts = tuple(profile.stage_names[:stage])
    return ConstraintTaxResult(
        discovery_stage=stage,
        discovery_stage_name=_stage_name(profile, stage),
        recommended_stage=recommended,
        recommended_stage_name=_stage_name(profile, recommended),
        cost_multiplier=float(multiplier),
        rework_days=rework_days,
        avoidable_rework_days=max(0.0, rework_days - recommended_days),
        artifacts_to_rebuild=artifacts,
        late_discovery=stage > recommended,
    )


def iteration_frontier(
    profile: WorkflowTrackProfile,
    *,
    validation_depth_pct: float,
    automation_pct: float,
    hardware_realism_pct: float,
    data_scale_pct: float,
) -> IterationFrontierResult:
    """Estimate iteration time versus residual deployment risk."""
    depth = max(0.0, min(100.0, float(validation_depth_pct)))
    automation = max(0.0, min(100.0, float(automation_pct)))
    realism = max(0.0, min(100.0, float(hardware_realism_pct)))
    data_scale = max(0.0, min(100.0, float(data_scale_pct)))

    validation_factor = 0.45 + depth / 90 + realism / 110 + data_scale / 140
    automation_factor = max(0.35, 1.0 - automation * 0.006)
    iteration_days = profile.base_cycle_days * validation_factor * automation_factor
    confidence = min(99.0, 18.0 + depth * 0.28 + realism * 0.27 + data_scale * 0.22 + automation * 0.12)
    risk = max(profile.min_residual_risk_pct, profile.base_residual_risk_pct - confidence * 0.62)
    knob_scores = {
        "validation depth": depth,
        "automation": automation,
        "hardware realism": realism,
        "data scale": data_scale,
    }
    bottleneck = min(knob_scores, key=knob_scores.get)
    return IterationFrontierResult(
        validation_depth_pct=depth,
        automation_pct=automation,
        hardware_realism_pct=realism,
        data_scale_pct=data_scale,
        iteration_days=iteration_days,
        confidence_pct=confidence,
        residual_risk_pct=risk,
        bottleneck=bottleneck,
    )


def workflow_policy(
    profile: WorkflowTrackProfile,
    frontier: IterationFrontierResult,
    *,
    gate_id: str,
    release_policy: str,
    rollback_rule: str,
) -> WorkflowPolicyResult:
    """Package the selected workflow policy and residual blind spot."""
    gate = next((candidate for candidate in profile.gate_options if candidate.gate_id == gate_id), profile.gate_options[0])
    gate_tax = constraint_tax(profile, discovery_stage=gate.stage)
    policy = release_policy or (profile.release_policies[0] if profile.release_policies else "release after passing deployment gate")
    rollback = rollback_rule or (profile.rollback_rules[0] if profile.rollback_rules else "rollback on guardrail breach")
    summary = (
        f"Run {gate.label} at {gate_tax.discovery_stage_name}; use {policy}; "
        f"rollback with {rollback}."
    )
    return WorkflowPolicyResult(
        gate_id=gate.gate_id,
        gate_label=gate.label,
        release_policy=policy,
        rollback_rule=rollback,
        gate_stage=gate.stage,
        gate_stage_name=gate_tax.discovery_stage_name,
        rework_days_at_gate=gate_tax.rework_days,
        residual_risk_pct=frontier.residual_risk_pct,
        blind_spot=gate.residual_risk,
        policy_summary=summary,
    )


__all__ = [
    "ConstraintTaxResult",
    "IterationFrontierResult",
    "WorkflowGate",
    "WorkflowPolicyResult",
    "WorkflowTrackProfile",
    "constraint_tax",
    "iteration_frontier",
    "workflow_policy",
    "workflow_track_profile",
]
