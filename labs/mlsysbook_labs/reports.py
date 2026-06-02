"""Report builders for submit-ready MLSysBook lab artifacts."""

from __future__ import annotations

import json
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
) -> LabReport:
    """Build a Markdown report and JSON snapshot for a lab submission."""
    predictions = predictions or {}
    knob_settings = knob_settings or {}
    binding_constraints = binding_constraints or {}
    decisions = decisions or {}
    reflections = reflections or {}
    snapshot = to_plain(result_snapshot or {})

    recap_lines = ""
    if recap is not None:
        recap_lines = f"""
## Chapter Recap

- Emphasis: {recap.emphasis}
- Key terms: {", ".join(recap.key_terms)}
- ML concept: {recap.ml_concept}
- Systems translation: {recap.systems_translation}
- What to watch: {recap.what_to_watch}
- Common trap: {recap.common_trap}
- Suggested reading: {recap.suggested_reading}
"""

    markdown = f"""# {metadata.title} Lab Report

## Submission Metadata

- Student: {student_id or "Not provided"}
- Lab ID: {metadata.lab_id}
- Lab version: {metadata.lab_version}
- Book anchor: {metadata.book_anchor}
- Updated at: {metadata.updated_at}
- MLSysIM version: {metadata.mlsysim_version}
- MLSysBook Labs version: {metadata.mlsysbook_labs_version}
- Report schema version: {metadata.report_schema_version}
- Track: {track}
- Scenario: {scenario}
{recap_lines}
## Predictions

{_bullet_dict(predictions)}

## Knob Settings

{_bullet_dict(knob_settings)}

## Binding Constraints

{_bullet_dict(binding_constraints)}

## Engineering Decision

{_bullet_dict(decisions)}

## Reflection

{_bullet_dict(reflections)}

## Residual Risk

{residual_risk or "Not recorded."}

## MLSysIM Result Snapshot

```json
{json.dumps(snapshot, indent=2, sort_keys=True)}
```
"""

    report_snapshot = {
        "metadata": asdict(metadata),
        "student_id": student_id,
        "track": track,
        "scenario": scenario,
        "predictions": to_plain(predictions),
        "knob_settings": to_plain(knob_settings),
        "binding_constraints": to_plain(binding_constraints),
        "decisions": to_plain(decisions),
        "reflections": to_plain(reflections),
        "residual_risk": residual_risk,
        "result_snapshot": snapshot,
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
