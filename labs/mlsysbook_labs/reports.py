"""Report builders for submit-ready MLSysBook lab artifacts."""

from __future__ import annotations

import json
import html
from dataclasses import asdict
from typing import Any

from .schemas import ChapterRecap, LabMetadata, LabReport, to_plain


def _bullet_dict(values: dict[str, Any]) -> str:
    if not values:
        return "- Not recorded."
    lines = []
    for key, value in values.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _bullet_list(values: tuple[str, ...] | list[str] | None) -> str:
    if not values:
        return "- Not recorded."
    return "\n".join(f"- {value}" for value in values)


def _section_value(value: Any) -> str:
    if isinstance(value, dict):
        return _bullet_dict(value)
    if isinstance(value, (list, tuple)):
        return _bullet_list(list(value))
    if value:
        return str(value)
    return "Not recorded."


def _missing_fields(
    *,
    track: str,
    scenario: str,
    predictions: dict[str, Any],
    evidence_summary: Any,
    final_decision: Any,
    big_takeaways: tuple[str, ...] | list[str] | None,
    reflections: dict[str, Any],
    residual_risk: str,
    explicit: tuple[str, ...] | list[str] | None,
) -> list[str]:
    missing = list(explicit or ())
    if track == "not recorded":
        missing.append("Track And Scenario: track")
    if scenario == "not recorded":
        missing.append("Track And Scenario: scenario")
    if not predictions:
        missing.append("Predictions")
    if not evidence_summary:
        missing.append("Evidence Summary")
    if not final_decision:
        missing.append("Final Decision")
    if not big_takeaways:
        missing.append("Big Takeaways")
    if not reflections:
        missing.append("Reflections")
    if not residual_risk:
        missing.append("Residual Risk")
    return missing


def build_lab_report(
    metadata: LabMetadata,
    *,
    student_id: str = "",
    track: str = "not recorded",
    scenario: str = "not recorded",
    recap: ChapterRecap | None = None,
    predictions: dict[str, Any] | None = None,
    knob_settings: dict[str, Any] | None = None,
    binding_constraints: dict[str, Any] | None = None,
    decisions: dict[str, Any] | None = None,
    reflections: dict[str, Any] | None = None,
    residual_risk: str = "",
    result_snapshot: Any | None = None,
    learning_objectives: tuple[str, ...] | list[str] | None = None,
    evidence_summary: dict[str, Any] | str | None = None,
    final_decision: dict[str, Any] | str | None = None,
    big_takeaways: tuple[str, ...] | list[str] | None = None,
    source_trace: dict[str, Any] | str | None = None,
    incomplete_fields: tuple[str, ...] | list[str] | None = None,
) -> LabReport:
    """Build a Markdown report and JSON snapshot for a lab submission."""
    predictions = predictions or {}
    knob_settings = knob_settings or {}
    binding_constraints = binding_constraints or {}
    decisions = decisions or {}
    reflections = reflections or {}
    snapshot = to_plain(result_snapshot or {})
    evidence = evidence_summary or {
        "knob_settings": knob_settings or "Not recorded.",
        "binding_constraints": binding_constraints or "Not recorded.",
        "result_snapshot": snapshot or "Not recorded.",
    }
    has_evidence = bool(evidence_summary or knob_settings or binding_constraints or snapshot)
    decision = final_decision or decisions
    trace = source_trace or {
        "mlsysim_version": metadata.mlsysim_version,
        "mlsysbook_labs_version": metadata.mlsysbook_labs_version,
        "report_schema_version": metadata.report_schema_version,
        "book_anchor": metadata.book_anchor,
    }

    recap_lines = ""
    if recap is not None:
        recap_lines = f"""
- Emphasis: {recap.emphasis}
- Key terms: {", ".join(recap.key_terms)}
- ML concept: {recap.ml_concept}
- Systems translation: {recap.systems_translation}
- What to watch: {recap.what_to_watch}
- Common trap: {recap.common_trap}
- Suggested reading: {recap.suggested_reading}
"""

    missing = _missing_fields(
        track=track,
        scenario=scenario,
        predictions=predictions,
        evidence_summary=has_evidence,
        final_decision=decision,
        big_takeaways=big_takeaways,
        reflections=reflections,
        residual_risk=residual_risk,
        explicit=incomplete_fields,
    )
    incomplete_section = ""
    if missing:
        incomplete_section = f"""
## Incomplete Fields

{_bullet_list(missing)}
"""

    markdown = f"""# {metadata.title} Lab Report

## Lab

- Student: {student_id or "Not provided"}
- Lab ID: {metadata.lab_id}
- Lab title: {metadata.title}
- Lab version: {metadata.lab_version}
- Book anchor: {metadata.book_anchor}
- Updated at: {metadata.updated_at}
- MLSysIM version: {metadata.mlsysim_version}
- MLSysBook Labs version: {metadata.mlsysbook_labs_version}
- Report schema version: {metadata.report_schema_version}

## Track And Scenario

- Track: {track}
- Scenario: {scenario}

## Learning Objectives

{_bullet_list(learning_objectives)}

## Predictions

{_bullet_dict(predictions)}

## Evidence Summary

{_section_value(evidence)}

## Final Decision

{_section_value(decision)}

## Big Takeaways

{_bullet_list(big_takeaways)}

## Reflections

{_bullet_dict(reflections)}

## Residual Risk

{residual_risk or "Not recorded."}

## Source Trace

{_section_value(trace)}

{recap_lines}

### MLSysIM Result Snapshot

```json
{json.dumps(snapshot, indent=2, sort_keys=True)}
```
{incomplete_section}
"""

    report_snapshot = {
        "metadata": asdict(metadata),
        "student_id": student_id,
        "track": track,
        "scenario": scenario,
        "learning_objectives": to_plain(learning_objectives or ()),
        "predictions": to_plain(predictions),
        "knob_settings": to_plain(knob_settings),
        "binding_constraints": to_plain(binding_constraints),
        "evidence_summary": to_plain(evidence),
        "final_decision": to_plain(decision),
        "big_takeaways": to_plain(big_takeaways or ()),
        "decisions": to_plain(decisions),
        "reflections": to_plain(reflections),
        "residual_risk": residual_risk,
        "source_trace": to_plain(trace),
        "result_snapshot": snapshot,
        "incomplete_fields": missing,
    }
    return LabReport(
        metadata=metadata,
        student_id=student_id,
        track=track,
        scenario=scenario,
        markdown=markdown,
        snapshot=report_snapshot,
    )


def report_export(report: LabReport, *, include_json: bool = True):
    """Return Marimo download controls for a lab report."""
    import marimo as mo

    safe_id = report.metadata.lab_id.replace("/", "_")
    md_download = mo.download(
        report.markdown.encode("utf-8"),
        filename=f"{safe_id}_report.md",
        mimetype="text/markdown",
        label="Download report",
    )

    if not include_json:
        return md_download

    json_download = mo.download(
        json.dumps(report.snapshot, indent=2, sort_keys=True).encode("utf-8"),
        filename=f"{safe_id}_snapshot.json",
        mimetype="application/json",
        label="Download JSON snapshot",
    )
    return mo.hstack([md_download, json_download], justify="start")


def report_text_fallback(report: LabReport) -> str:
    """Return the local Markdown report text for fallback display or copying."""
    return report.markdown


def report_export_panel(report: LabReport, *, include_json: bool = True, include_fallback: bool = True):
    """Return local download controls plus an optional visible Markdown fallback."""
    import marimo as mo

    incomplete = tuple(str(item) for item in report.snapshot.get("incomplete_fields", ()) if item)
    if incomplete:
        missing_items = "".join(f"<li>{html.escape(item)}</li>" for item in incomplete)
        return mo.Html(
            f"""
<div class="mlsysbook-panel mlsysbook-report-panel" style="border-left:4px solid #B42318;">
  <h2>Report Locked</h2>
  <p class="mlsysbook-action-note">
    Complete the required lab work before downloading the local report.
  </p>
  <ul style="margin:10px 0 0 18px; color:#475467; line-height:1.6;">
    {missing_items}
  </ul>
</div>
"""
        )

    controls = report_export(report, include_json=include_json)
    if not include_fallback:
        return mo.vstack([controls], align="center")

    fallback = mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-report-panel">
  <h2>Report Text Fallback</h2>
  <p class="mlsysbook-action-note">
    This contains the same Markdown artifact as the download button.
  </p>
  <textarea readonly style="width:100%; min-height:260px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size:0.85rem; line-height:1.45; border:1px solid #D9DEE8; border-radius:8px; padding:12px;">{html.escape(report.markdown)}</textarea>
</div>
"""
    )
    return mo.vstack([controls, mo.accordion({"Report text fallback": fallback}, multiple=False)], align="center")
