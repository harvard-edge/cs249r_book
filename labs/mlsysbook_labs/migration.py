"""Compact helpers for migrating legacy notebooks to the track/report contract."""

from __future__ import annotations

from typing import Any, Mapping

from .reports import build_lab_report, report_export_panel
from .schemas import LabMetadata, LabReport, LabTrackVariant, TrackProfile
from .ui import big_takeaways as big_takeaways_panel
from .ui import learning_objectives as learning_objectives_panel
from .ui import scenario_brief, source_trace, track_context


def baseline_learning_objectives(metadata: LabMetadata, variant: LabTrackVariant) -> tuple[str, ...]:
    """Return conservative measurable objectives for a not-yet-deep-migrated lab."""
    return (
        f"Identify the {metadata.title} system decision for the selected track.",
        f"Compare the primary metric ({variant.primary_metric}) against the guardrail ({variant.guardrail_metric}).",
        "Defend a local engineering decision with source-traced evidence.",
    )


def baseline_big_takeaways(metadata: LabMetadata, variant: LabTrackVariant) -> tuple[str, ...]:
    """Return default takeaways until a lab receives hand-authored synthesis text."""
    return (
        f"{metadata.title} changes when the deployment track changes.",
        f"The primary metric is {variant.primary_metric}, but the guardrail is {variant.guardrail_metric}.",
        "Hardware, model, system, and scenario facts should come from shared registries.",
    )


def variant_source_trace(variant: LabTrackVariant, profile: TrackProfile) -> dict[str, Any]:
    """Return a serializable source trace for a track variant."""
    return {
        "track_id": profile.track_id,
        "scenario_id": variant.scenario_id,
        "hardware_ref": variant.hardware_ref,
        "system_ref": variant.system_ref or "single-device profile",
        "model_ref": variant.model_ref,
        "defaults": dict(variant.defaults),
        "assumptions": dict(variant.assumptions),
        "track_source_policy": profile.source_policy,
    }


def build_migration_report(
    metadata: LabMetadata,
    profile: TrackProfile,
    variant: LabTrackVariant,
    *,
    predictions: Mapping[str, Any] | None = None,
    evidence_summary: Mapping[str, Any] | str | None = None,
    final_decision: Mapping[str, Any] | str | None = None,
    reflections: Mapping[str, Any] | None = None,
    residual_risk: str = "",
    result_snapshot: Mapping[str, Any] | None = None,
    learning_objectives: tuple[str, ...] | list[str] | None = None,
    big_takeaways: tuple[str, ...] | list[str] | None = None,
    incomplete_fields: tuple[str, ...] | list[str] | None = None,
) -> LabReport:
    """Build a local report for a legacy notebook being migrated."""
    objectives = tuple(learning_objectives or baseline_learning_objectives(metadata, variant))
    takeaways = tuple(big_takeaways or baseline_big_takeaways(metadata, variant))
    evidence = evidence_summary or {
        "track": profile.label,
        "primary_metric": variant.primary_metric,
        "guardrail_metric": variant.guardrail_metric,
        "migration_status": variant.defaults.get(
            "implementation_status",
            "legacy_notebook_pending_deep_migration",
        ),
    }
    missing = list(incomplete_fields or ())
    if variant.assumptions.get("fallback_variant"):
        missing.append("Hand-authored track variant")
    missing.append("Deep notebook part checkpoints")
    return build_lab_report(
        metadata,
        track=profile.track_id,
        scenario=variant.scenario_id,
        learning_objectives=objectives,
        predictions=dict(predictions or {}),
        evidence_summary=evidence,
        final_decision=final_decision,
        big_takeaways=takeaways,
        reflections=dict(reflections or {}),
        residual_risk=residual_risk,
        result_snapshot=dict(result_snapshot or {}),
        source_trace=variant_source_trace(variant, profile),
        incomplete_fields=tuple(dict.fromkeys(missing)),
    )


def legacy_migration_panel(
    metadata: LabMetadata,
    profile: TrackProfile,
    variant: LabTrackVariant,
    *,
    learning_objectives: tuple[str, ...] | list[str] | None = None,
    big_takeaways: tuple[str, ...] | list[str] | None = None,
    predictions: Mapping[str, Any] | None = None,
    evidence_summary: Mapping[str, Any] | str | None = None,
    final_decision: Mapping[str, Any] | str | None = None,
    reflections: Mapping[str, Any] | None = None,
    residual_risk: str = "",
    result_snapshot: Mapping[str, Any] | None = None,
):
    """Render a compact track-aware shell plus local report export for a legacy lab."""
    import marimo as mo

    objectives = tuple(learning_objectives or baseline_learning_objectives(metadata, variant))
    takeaways = tuple(big_takeaways or baseline_big_takeaways(metadata, variant))
    trace = variant_source_trace(variant, profile)
    report = build_migration_report(
        metadata,
        profile,
        variant,
        predictions=predictions,
        evidence_summary=evidence_summary,
        final_decision=final_decision,
        reflections=reflections,
        residual_risk=residual_risk,
        result_snapshot=result_snapshot,
        learning_objectives=objectives,
        big_takeaways=takeaways,
    )
    return mo.vstack(
        [
            learning_objectives_panel(objectives),
            track_context(profile),
            scenario_brief(
                "Scenario Brief",
                stakeholder=variant.stakeholder,
                objective=variant.objective,
                constraints={
                    "Workload": variant.workload_summary,
                    "Model source": variant.model_ref,
                    "Hardware source": variant.hardware_ref,
                    "System source": variant.system_ref or "single-device profile",
                    "Primary metric": variant.primary_metric,
                    "Guardrail metric": variant.guardrail_metric,
                },
            ),
            source_trace(trace),
            big_takeaways_panel(takeaways),
            mo.Html(
                """
<div class="mlsysbook-panel">
  <h2>Download Report</h2>
  <p class="mlsysbook-source-summary">
    This report is generated locally from the current track variant and marks
    any remaining migration gaps under Incomplete Fields.
  </p>
</div>
"""
            ),
            report_export_panel(report),
        ]
    )


__all__ = [
    "baseline_big_takeaways",
    "baseline_learning_objectives",
    "build_migration_report",
    "legacy_migration_panel",
    "variant_source_trace",
]
