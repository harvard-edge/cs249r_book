"""Reusable Marimo UI components for the MLSysBook trade-off labs."""

from __future__ import annotations

import html
from dataclasses import asdict
from typing import Any, Mapping

import marimo as mo

from .schemas import ChapterRecap, InstructorMetadata, LabMetadata, NuggetSpec, TrackProfile
from .tracks import DEFAULT_TRACK_ID, get_track_profile, normalize_track_id, track_display_label, track_options


ACADEMIC_LAB_CSS = mo.Html(
    """
<style>
:root {
  --mlsysbook-crimson: #A51C30;
  --mlsysbook-blue: #1F407A;
  --mlsysbook-ink: #172033;
  --mlsysbook-muted: #667085;
  --mlsysbook-line: #D9DEE8;
  --mlsysbook-panel: #FFFFFF;
  --mlsysbook-soft: #F6F8FB;
  --mlsysbook-intro: #F7FAFF;
  --mlsysbook-launch: #FBFCFE;
  --mlsysbook-ok: #247A4D;
  --mlsysbook-warn: #9A5B00;
  --mlsysbook-danger: #B42318;
  --mlsysbook-readable-width: 720px;
  --mlsysbook-panel-width: 840px;
}
.mlsysbook-lab-shell {
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--mlsysbook-ink);
  width: min(var(--mlsysbook-panel-width), 100%);
  max-width: min(var(--mlsysbook-panel-width), 100%);
  margin-left: auto;
  margin-right: auto;
}
.mlsysbook-lab-header {
  background: var(--mlsysbook-panel);
  border: 1px solid var(--mlsysbook-line);
  border-left: 6px solid var(--mlsysbook-accent, var(--mlsysbook-crimson));
  border-radius: 8px;
  padding: 22px 26px;
  margin: 0 auto 18px auto;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
  width: min(var(--mlsysbook-panel-width), 100%);
  max-width: min(var(--mlsysbook-panel-width), 100%);
}
.mlsysbook-meta {
  color: var(--mlsysbook-muted);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.mlsysbook-lab-header h1 {
  margin: 7px 0 7px 0;
  font-size: 1.9rem;
  line-height: 1.15;
  letter-spacing: 0;
}
.mlsysbook-lab-header p {
  max-width: var(--mlsysbook-readable-width);
  color: #344054;
  line-height: 1.55;
  margin: 0;
}
.mlsysbook-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}
.mlsysbook-chip-row + .mlsysbook-chip-row {
  margin-top: 8px;
}
.mlsysbook-topic-row .mlsysbook-chip {
  background: #FFFFFF;
}
.mlsysbook-chip {
  border: 1px solid var(--mlsysbook-line);
  background: var(--mlsysbook-soft);
  color: #344054;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.78rem;
  font-weight: 650;
}
.mlsysbook-recap,
.mlsysbook-panel {
  background: var(--mlsysbook-panel);
  border: 1px solid var(--mlsysbook-line);
  border-radius: 8px;
  padding: 18px 20px;
  margin: 12px auto;
  width: min(var(--mlsysbook-panel-width), 100%);
  max-width: min(var(--mlsysbook-panel-width), 100%);
}
.mlsysbook-recap h2,
.mlsysbook-panel h2 {
  margin: 0 0 10px 0;
  font-size: 1.05rem;
  letter-spacing: 0;
}
.mlsysbook-section-label {
  color: var(--mlsysbook-blue);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  margin-bottom: 8px;
  text-transform: uppercase;
}
.mlsysbook-intro-panel {
  background: linear-gradient(180deg, #FFFFFF 0%, var(--mlsysbook-intro) 100%);
  border-color: #C9D8EE;
  border-left: 4px solid var(--mlsysbook-blue);
}
.mlsysbook-track-panel {
  background: linear-gradient(135deg, #F8FFFB 0%, #FFFFFF 72%);
  border-color: #B8D8C6;
  border-left: 6px solid var(--mlsysbook-ok);
  box-shadow: 0 6px 16px rgba(31, 64, 122, 0.08);
  margin-top: 22px;
}
.mlsysbook-launch-panel {
  background: linear-gradient(135deg, #F8FFFB 0%, var(--mlsysbook-launch) 76%);
  border-color: #B8D8C6;
  border-left: 4px solid var(--mlsysbook-ok);
}
.mlsysbook-track-panel .mlsysbook-section-label,
.mlsysbook-launch-panel .mlsysbook-section-label,
.mlsysbook-thread-panel .mlsysbook-section-label {
  color: var(--mlsysbook-ok);
}
.mlsysbook-thread-panel {
  background: #F8FFFB;
  border: 1px solid #B8D8C6;
  border-left: 4px solid var(--mlsysbook-ok);
  border-radius: 8px;
  padding: 14px 16px;
  margin: 12px auto;
  width: min(var(--mlsysbook-panel-width), 100%);
  max-width: min(var(--mlsysbook-panel-width), 100%);
}
.mlsysbook-thread-panel h3 {
  margin: 0 0 8px 0;
  font-size: 1rem;
}
.mlsysbook-thread-panel p {
  color: #344054;
  line-height: 1.55;
  margin: 6px 0;
}
.mlsysbook-flow {
  border: 1px solid var(--mlsysbook-line);
  border-radius: 8px;
  padding: 12px;
  margin: 12px auto;
  width: min(var(--mlsysbook-panel-width), 100%);
  max-width: min(var(--mlsysbook-panel-width), 100%);
  background: #FFFFFF;
}
.mlsysbook-flow h3 {
  margin: 0 0 10px 0;
  font-size: 1rem;
}
.mlsysbook-flow-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.mlsysbook-flow-step {
  border: 1px solid #B8D8C6;
  border-radius: 8px;
  background: #F8FFFB;
  color: #123524;
  font-size: 0.86rem;
  font-weight: 650;
  line-height: 1.35;
  min-width: 140px;
  padding: 10px 12px;
}
.mlsysbook-flow-arrow {
  color: var(--mlsysbook-ok);
  font-weight: 800;
}
.mlsysbook-takeaway-panel {
  background: linear-gradient(180deg, #FFFFFF 0%, #F7FAFF 100%);
  border-color: #C9D8EE;
  border-left: 4px solid var(--mlsysbook-blue);
}
.mlsysbook-report-panel {
  background: #FFFBF2;
  border-color: #F4C27A;
  border-left: 4px solid var(--mlsysbook-warn);
}
.mlsysbook-action-box {
  width: min(var(--mlsysbook-panel-width), 100%);
  max-width: min(var(--mlsysbook-panel-width), 100%);
  margin: 14px auto;
  padding: 16px 18px;
  background: linear-gradient(135deg, #F8FFFB 0%, #FFFFFF 84%);
  border: 1px solid #B8D8C6;
  border-left: 4px solid var(--mlsysbook-ok);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(31, 64, 122, 0.06);
}
.mlsysbook-action-box.is-warn {
  background: linear-gradient(135deg, #FFFBF2 0%, #FFFFFF 84%);
  border-color: #F4C27A;
  border-left-color: var(--mlsysbook-warn);
}
.mlsysbook-action-box h3 {
  margin: 0 0 4px 0;
  font-size: 0.98rem;
  letter-spacing: 0;
}
.mlsysbook-action-box p {
  color: #475467;
  line-height: 1.45;
  margin: 0 0 12px 0;
}
.mlsysbook-action-control {
  margin-top: 8px;
}
.mlsysbook-action-note {
  color: #475467;
  font-size: 0.9rem;
  line-height: 1.45;
  margin: 4px 0 0 0;
}
.mlsysbook-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  align-items: stretch;
  gap: 12px;
}
.mlsysbook-field {
  background: var(--mlsysbook-soft);
  border: 1px solid #EDF0F5;
  border-radius: 8px;
  padding: 12px;
  overflow-wrap: anywhere;
  word-break: normal;
}
.mlsysbook-field code {
  white-space: normal;
  overflow-wrap: anywhere;
}
.mlsysbook-field strong {
  display: block;
  color: #475467;
  font-size: 0.74rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.mlsysbook-nugget {
  border-left: 4px solid var(--mlsysbook-accent, var(--mlsysbook-crimson));
}
.mlsysbook-callout {
  background: #F9FAFB;
  border: 1px solid var(--mlsysbook-line);
  border-radius: 8px;
  padding: 12px 14px;
  margin: 10px auto;
  max-width: min(var(--mlsysbook-readable-width), 100%);
}
.mlsysbook-readable {
  max-width: min(var(--mlsysbook-readable-width), 100%);
  margin-left: auto;
  margin-right: auto;
}
.mlsysbook-scenario-narrative {
  max-width: min(var(--mlsysbook-readable-width), 100%);
  margin: 0 auto 14px auto;
  color: #344054;
  line-height: 1.6;
}
.mlsysbook-scenario-narrative p {
  margin: 8px 0;
}
.mlsysbook-compact-fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0;
  border: 1px solid var(--mlsysbook-line);
  border-radius: 8px;
  overflow: hidden;
  background: #FFFFFF;
}
.mlsysbook-compact-field {
  min-height: 86px;
  padding: 12px 14px;
  border-right: 1px solid #EDF0F5;
  border-bottom: 1px solid #EDF0F5;
  overflow-wrap: anywhere;
}
.mlsysbook-compact-field strong {
  display: block;
  color: #475467;
  font-size: 0.72rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.mlsysbook-compact-field:last-child:nth-child(3n + 1) {
  grid-column: 1 / -1;
}
.mlsysbook-compact-field:last-child:nth-child(3n + 2) {
  grid-column: span 2;
}
.mlsysbook-compact-fields.is-brief {
  background: #FFFFFF;
}
.mlsysbook-compact-fields.is-brief .mlsysbook-compact-field {
  background: #FFFFFF;
  min-height: 64px;
}
.mlsysbook-map-list {
  border: 1px solid var(--mlsysbook-line);
  border-radius: 8px;
  overflow: hidden;
  background: #FFFFFF;
}
.mlsysbook-map-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  padding: 12px 14px;
  border-bottom: 1px solid #EDF0F5;
}
.mlsysbook-map-row:last-child {
  border-bottom: 0;
}
.mlsysbook-map-title {
  font-weight: 760;
}
.mlsysbook-map-question {
  color: var(--mlsysbook-muted);
  font-size: 0.88rem;
  line-height: 1.45;
  margin-top: 4px;
}
.mlsysbook-evidence-panel details {
  border-top: 1px solid #EDF0F5;
  margin-top: 10px;
  padding-top: 10px;
}
.mlsysbook-evidence-panel summary {
  cursor: pointer;
  color: var(--mlsysbook-blue);
  font-weight: 750;
}
.mlsysbook-list {
  margin: 8px 0 0 0;
  padding-left: 20px;
  list-style-position: outside;
}
ul.mlsysbook-list {
  list-style-type: disc;
}
ol.mlsysbook-list {
  list-style-type: decimal;
}
.mlsysbook-list li {
  display: list-item;
  margin: 6px 0;
  line-height: 1.5;
}
.mlsysbook-status {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 3px 9px;
  font-size: 0.72rem;
  font-weight: 750;
  background: #EEF2F6;
  color: #344054;
}
.mlsysbook-status.is-progress {
  background: #EAF2FF;
  color: var(--mlsysbook-blue);
}
.mlsysbook-status.is-ok {
  background: #ECFDF3;
  color: var(--mlsysbook-ok);
}
.mlsysbook-status.is-warn {
  background: #FFF7E6;
  color: var(--mlsysbook-warn);
}
.mlsysbook-status.is-danger {
  background: #FEF3F2;
  color: var(--mlsysbook-danger);
}
.mlsysbook-part-title {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: baseline;
}
.mlsysbook-part-title h2 {
  margin-bottom: 0;
}
.mlsysbook-source-summary {
  color: #475467;
  font-size: 0.9rem;
  line-height: 1.45;
  margin: 4px 0 10px 0;
}
.mlsysbook-source-trace details {
  border-top: 1px solid #EDF0F5;
  padding-top: 10px;
}
.mlsysbook-source-trace summary {
  cursor: pointer;
  color: var(--mlsysbook-blue);
  font-weight: 750;
}
.lab-hud {
  width: min(var(--mlsysbook-readable-width), 100%) !important;
  max-width: min(var(--mlsysbook-readable-width), 100%) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  background: #FFFFFF !important;
  color: #344054;
  flex-wrap: wrap;
}
div[class~="fixed"][class~="bottom-0"][class~="right-0"][class~="z-50"],
div[class~="fixed"][class~="right-0"][class~="top-0"][class~="z-50"],
div[class~="fixed"][class~="right-8"][class~="z-10000"],
div[class~="fixed"][class~="top-0"][class~="z-100"][class~="max-h-screen"],
ol[class~="fixed"][class~="top-0"][class~="z-100"][class~="max-h-screen"] {
  display: none !important;
}
.output.block > div:not(:has(svg, canvas, iframe, table, marimo-ui-element, .js-plotly-plot, .plotly)) {
  width: min(var(--mlsysbook-panel-width), 100%) !important;
  max-width: min(var(--mlsysbook-panel-width), 100%) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
.output.block > div:not(:has(svg, canvas, iframe, table, marimo-ui-element, .js-plotly-plot, .plotly)) p,
.output.block > div:not(:has(svg, canvas, iframe, table, marimo-ui-element, .js-plotly-plot, .plotly)) li {
  max-width: min(var(--mlsysbook-readable-width), 100%) !important;
}
.output.block > div:has(marimo-ui-element):not(:has(svg, canvas, iframe, table, marimo-tabs, .js-plotly-plot, .plotly, .mlsysbook-panel, .mlsysbook-lab-header, .mlsysbook-action-box)) {
  width: min(var(--mlsysbook-panel-width), 100%) !important;
  max-width: min(var(--mlsysbook-panel-width), 100%) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  background: linear-gradient(135deg, #F8FFFB 0%, #FFFFFF 82%) !important;
  border: 1px solid #B8D8C6 !important;
  border-left: 4px solid var(--mlsysbook-ok) !important;
  border-radius: 8px !important;
  box-shadow: 0 4px 12px rgba(31, 64, 122, 0.06) !important;
  padding: 14px 18px !important;
}
.output.block > div:has(marimo-ui-element):not(:has(svg, canvas, iframe, table, marimo-tabs, .js-plotly-plot, .plotly, .mlsysbook-panel, .mlsysbook-lab-header, .mlsysbook-action-box)) > div {
  max-width: min(var(--mlsysbook-readable-width), 100%) !important;
}
marimo-tabs,
div[style*="border-left:4px solid"][style*="border-radius:0 10px"],
div[style*="border-left: 4px solid"][style*="border-radius: 0 10px"],
div[style*="border-left:4px solid"][style*="border-radius:0px 10px"],
div[style*="border-left: 4px solid"][style*="border-radius: 0px 10px"] {
  display: block !important;
  width: min(var(--mlsysbook-panel-width), 100%) !important;
  max-width: min(var(--mlsysbook-panel-width), 100%) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
marimo-tabs {
  overflow-x: visible !important;
  overflow-y: visible !important;
}
marimo-callout-output {
  display: block !important;
  width: min(var(--mlsysbook-readable-width), 100%) !important;
  max-width: min(var(--mlsysbook-readable-width), 100%) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
.mlsysbook-version {
  color: var(--mlsysbook-muted);
  font-size: 0.78rem;
  margin-top: 10px;
}
@media (max-width: 760px) {
  .mlsysbook-lab-shell,
  .mlsysbook-lab-header,
  .mlsysbook-recap,
  .mlsysbook-panel,
  .mlsysbook-callout,
  .mlsysbook-readable,
  .lab-hud,
  marimo-callout-output {
    width: 100%;
    max-width: 100%;
  }
  .mlsysbook-compact-fields {
    grid-template-columns: 1fr;
  }
  .output.block > div:has(marimo-ui-element):not(:has(svg, canvas, iframe, table, marimo-tabs, .js-plotly-plot, .plotly)) {
    width: 100% !important;
    max-width: 100% !important;
  }
  .mlsysbook-compact-field:last-child:nth-child(n) {
    grid-column: auto;
  }
}
</style>
"""
)

COMPLETION_STATES = (
    "not_started",
    "prediction_saved",
    "evidence_viewed",
    "checkpoint_saved",
    "decision_complete",
)

_COMPLETION_LABELS = {
    "not_started": "Not Started",
    "prediction_saved": "Prediction Saved",
    "evidence_viewed": "Evidence Viewed",
    "checkpoint_saved": "Checkpoint Saved",
    "decision_complete": "Decision Complete",
}

_STATUS_CLASSES = {
    "not_started": "",
    "prediction_saved": "is-progress",
    "evidence_viewed": "is-progress",
    "checkpoint_saved": "is-ok",
    "decision_complete": "is-ok",
}

_CONSTRAINT_LABELS = {
    "pass": "Pass",
    "ok": "Pass",
    "warn": "Watch",
    "warning": "Watch",
    "fail": "Fails",
    "failure": "Fails",
    "danger": "Fails",
}

_CONSTRAINT_CLASSES = {
    "pass": "is-ok",
    "ok": "is-ok",
    "warn": "is-warn",
    "warning": "is-warn",
    "fail": "is-danger",
    "failure": "is-danger",
    "danger": "is-danger",
}


def _accent(volume: str) -> str:
    return "#1F407A" if "II" in volume or volume.strip() == "2" else "#A51C30"


def _items(values: Any) -> tuple[Any, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(values)


def _render_list(values: Any) -> str:
    return "".join(f"<li>{html.escape(str(value))}</li>" for value in _items(values))


def _render_fields(items: Mapping[str, Any]) -> str:
    rows = []
    for key, value in items.items():
        if isinstance(value, Mapping):
            value = "; ".join(f"{nested_key}: {nested_value}" for nested_key, nested_value in value.items())
        elif isinstance(value, (list, tuple, set)):
            value = ", ".join(str(item) for item in value)
        rows.append(
            f'<div class="mlsysbook-field"><strong>{html.escape(str(key).replace("_", " "))}</strong>{html.escape(str(value))}</div>'
        )
    return "".join(rows)


def _render_compact_fields(items: Mapping[str, Any]) -> str:
    rows = []
    for key, value in items.items():
        if isinstance(value, Mapping):
            value = "; ".join(f"{nested_key}: {nested_value}" for nested_key, nested_value in value.items())
        elif isinstance(value, (list, tuple, set)):
            value = ", ".join(str(item) for item in value)
        rows.append(
            f'<div class="mlsysbook-compact-field"><strong>{html.escape(str(key).replace("_", " "))}</strong>{html.escape(str(value))}</div>'
        )
    return "".join(rows)


def _student_facing_chips(chips: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    hidden = {"track-aware", "track aware", "track_aware"}
    return tuple(chip for chip in chips if chip.strip().lower() not in hidden)


def _part_value(part: Any, *keys: str, default: str = "") -> str:
    for key in keys:
        if isinstance(part, Mapping) and key in part:
            return str(part[key])
        if hasattr(part, key):
            return str(getattr(part, key))
    return default


def _part_heading(part: str, concept: str) -> str:
    label = part.strip()
    if not label.lower().startswith("part "):
        label = f"Part {label}"
    if concept:
        return f"{label} - {concept.strip()}"
    return label


def _completion_status(status: str) -> tuple[str, str]:
    normalized = str(status or "not_started").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in COMPLETION_STATES:
        normalized = "not_started"
    return _COMPLETION_LABELS[normalized], _STATUS_CLASSES[normalized]


def _constraint_status(status: str) -> tuple[str, str]:
    normalized = str(status or "pass").strip().lower().replace("-", "_").replace(" ", "_")
    return _CONSTRAINT_LABELS.get(normalized, str(status).strip().title()), _CONSTRAINT_CLASSES.get(normalized, "")


def lab_header(metadata: LabMetadata, subtitle: str, *, chips: tuple[str, ...] = ()) -> mo.Html:
    """Render the shared professional academic lab header."""
    topic_chips = _student_facing_chips(chips)
    topic_chip_html = "".join(f'<span class="mlsysbook-chip">{html.escape(chip)}</span>' for chip in topic_chips)
    topic_row = ""
    if topic_chip_html:
        topic_row = f'<div class="mlsysbook-chip-row mlsysbook-topic-row">{topic_chip_html}</div>'
    return mo.Html(
        f"""
<div class="mlsysbook-lab-shell">
  <div class="mlsysbook-lab-header" style="--mlsysbook-accent: {_accent(metadata.volume)};">
    <div class="mlsysbook-meta">
      {html.escape(metadata.volume)} | {html.escape(metadata.chapter)} | {html.escape(metadata.lab_id)}
    </div>
    <h1>{html.escape(metadata.title)}</h1>
    <p>{html.escape(subtitle)}</p>
    <div class="mlsysbook-chip-row">
      <span class="mlsysbook-chip">{html.escape(metadata.book_anchor)}</span>
      <span class="mlsysbook-chip">Lab v{html.escape(metadata.lab_version)}</span>
      <span class="mlsysbook-chip">Updated {html.escape(metadata.updated_at)}</span>
    </div>
    {topic_row}
  </div>
</div>
"""
    )


def learning_objectives(objectives: tuple[str, ...] | list[str]) -> mo.Html:
    """Render the required measurable objectives block."""
    return mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-intro-panel">
  <div class="mlsysbook-section-label">Before You Begin</div>
  <h2>Learning Objectives</h2>
  <ul class="mlsysbook-list">
    {_render_list(objectives)}
  </ul>
</div>
"""
    )


def chapter_recap(recap: ChapterRecap) -> mo.Html:
    """Render the required self-contained mini chapter recap."""
    terms = ", ".join(html.escape(term) for term in recap.key_terms)
    return mo.Html(
        f"""
<div class="mlsysbook-recap mlsysbook-intro-panel">
  <h2>Chapter Recap</h2>
  <div class="mlsysbook-grid">
    <div class="mlsysbook-field">
      <strong>Chapter anchor</strong>
      {html.escape(recap.emphasis)}
      <div class="mlsysbook-version">Key terms: {terms}</div>
    </div>
    <div class="mlsysbook-field">
      <strong>ML to systems translation</strong>
      {html.escape(recap.ml_concept)}
      <div class="mlsysbook-version">{html.escape(recap.systems_translation)}</div>
    </div>
    <div class="mlsysbook-field">
      <strong>Engineering lens</strong>
      {html.escape(recap.what_to_watch)}
      <div class="mlsysbook-version">Common trap: {html.escape(recap.common_trap)}</div>
    </div>
  </div>
  <div class="mlsysbook-version">Suggested reading: {html.escape(recap.suggested_reading)}</div>
</div>
"""
    )


def scenario_brief(title: str, stakeholder: str, objective: str, constraints: dict[str, Any]) -> mo.Html:
    rows = _render_compact_fields(constraints)
    return mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-launch-panel">
  <div class="mlsysbook-section-label">Start The Case</div>
  <h2>{html.escape(title)}</h2>
  <div class="mlsysbook-scenario-narrative">
    <p><strong>Stakeholder:</strong> {html.escape(stakeholder)}</p>
    <p><strong>Objective:</strong> {html.escape(objective)}</p>
  </div>
  <div class="mlsysbook-compact-fields is-brief">{rows}</div>
</div>
"""
    )


def scenario_thread(title: str, body: str, *, callout: str = "") -> mo.Html:
    """Render a short track/scenario reminder inside a lab part."""
    callout_html = (
        f'<p style="color:#344054; line-height:1.55; margin:6px 0;">'
        f"<strong>Track consequence:</strong> {html.escape(callout)}</p>"
        if callout
        else ""
    )
    return mo.Html(
        f"""
<div class="mlsysbook-thread-panel" style="background:#F8FFFB; border:1px solid #B8D8C6; border-left:4px solid #247A4D; border-radius:8px; padding:14px 16px; margin:12px auto; width:min(var(--mlsysbook-panel-width), 100%); max-width:min(var(--mlsysbook-panel-width), 100%);">
  <div class="mlsysbook-section-label" style="color:#247A4D; font-size:0.72rem; font-weight:800; letter-spacing:0.12em; margin-bottom:8px; text-transform:uppercase;">Scenario Thread</div>
  <h3 style="margin:0 0 8px 0; font-size:1rem;">{html.escape(title)}</h3>
  <p style="color:#344054; line-height:1.55; margin:6px 0;">{html.escape(body)}</p>
  {callout_html}
</div>
"""
    )


def decision_flow(title: str, steps: tuple[str, ...] | list[str]) -> mo.Html:
    """Render a compact, renderer-independent flow diagram."""
    step_html = []
    for index, step in enumerate(steps):
        if index:
            step_html.append('<span class="mlsysbook-flow-arrow" style="color:#247A4D; font-weight:800;">-></span>')
        step_html.append(
            f'<div class="mlsysbook-flow-step" style="border:1px solid #B8D8C6; border-radius:8px; background:#F8FFFB; color:#123524; font-size:0.86rem; font-weight:650; line-height:1.35; min-width:140px; padding:10px 12px;">{html.escape(step)}</div>'
        )
    return mo.Html(
        f"""
<div class="mlsysbook-flow" style="border:1px solid #D9DEE8; border-radius:8px; padding:12px; margin:12px auto; width:min(var(--mlsysbook-panel-width), 100%); max-width:min(var(--mlsysbook-panel-width), 100%); background:#FFFFFF;">
  <h3 style="margin:0 0 10px 0; font-size:1rem;">{html.escape(title)}</h3>
  <div class="mlsysbook-flow-steps" style="display:flex; flex-wrap:wrap; gap:8px; align-items:center;">{"".join(step_html)}</div>
</div>
"""
    )


def part_workflow(
    title: str,
    parts: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    *,
    scenario: str = "",
    reflection: str = "",
) -> mo.Html:
    """Render a reusable learner workflow for part-level lab activity."""
    scenario_html = (
        f'<p><strong>Scenario:</strong> {html.escape(scenario)}</p>'
        if scenario
        else ""
    )
    reflection_html = (
        f'<div class="mlsysbook-callout"><strong>Reflection:</strong> {html.escape(reflection)}</div>'
        if reflection
        else ""
    )
    rows = []
    for index, part in enumerate(parts, start=1):
        part_label = _part_value(part, "part", "label", default=f"Part {index}")
        concept = _part_value(part, "concept", "title", default="")
        heading = _part_heading(part_label, concept)
        fields = _render_compact_fields(
            {
                "Prediction": _part_value(part, "prediction", default="State what you expect before changing controls."),
                "Controls": _part_value(part, "controls", default="Move the provided knobs or choices."),
                "Evidence": _part_value(part, "evidence", default="Inspect the computed result and plot."),
                "Decision": _part_value(part, "decision", default="Write what you would defend and why."),
            }
        )
        rows.append(
            f"""
    <div class="mlsysbook-map-row">
      <div>
        <div class="mlsysbook-map-title">{html.escape(heading)}</div>
        <div class="mlsysbook-compact-fields is-brief">{fields}</div>
      </div>
    </div>
"""
        )
    return mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-launch-panel">
  <div class="mlsysbook-section-label">How To Work This Lab</div>
  <h2>{html.escape(title)}</h2>
  <div class="mlsysbook-scenario-narrative">
    {scenario_html}
    <p>Use each part as a prediction, controls, evidence, and decision loop. The goal is not just the computed answer; it is the argument you can defend for the selected track.</p>
  </div>
  <div class="mlsysbook-map-list">{"".join(rows)}</div>
  {reflection_html}
</div>
"""
    )


def lab_map(parts: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]], completion: Mapping[str, str] | None = None) -> mo.Html:
    """Render the lab part navigator with contract completion states."""
    completion = completion or {}
    rows = []
    for index, part in enumerate(parts, start=1):
        part_id = _part_value(part, "part_id", "id", "part", default=str(index))
        part_label = _part_value(part, "label", "part", default=part_id)
        concept = _part_value(part, "concept", "title", default="")
        question = _part_value(part, "question", "systems_question", default="")
        status = completion.get(part_id) or completion.get(part_label) or completion.get(concept) or "not_started"
        status_label, status_class = _completion_status(status)
        heading = _part_heading(part_label, concept)
        rows.append(
            f"""
    <div class="mlsysbook-map-row">
      <div>
        <div class="mlsysbook-map-title">{html.escape(heading)}</div>
        <div class="mlsysbook-map-question">{html.escape(question)}</div>
      </div>
      <span class="mlsysbook-status {html.escape(status_class)}">{html.escape(status_label)}</span>
    </div>
"""
        )
    return mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-launch-panel">
  <h2>Lab Map</h2>
  <div class="mlsysbook-map-list">
    {"".join(rows)}
  </div>
</div>
"""
    )


def track_selector(default: str = DEFAULT_TRACK_ID):
    """Return a Marimo radio selector for the canonical student tracks."""
    selected = normalize_track_id(default)
    options = track_options()
    selected_label = next((label for label, track_id in options.items() if track_id == selected), track_display_label(DEFAULT_TRACK_ID))
    return mo.ui.radio(options=options, value=selected_label, label="Choose Your Track", inline=True)


def action_box(
    element,
    *,
    title: str,
    body: str = "",
    name: str = "answer",
    tone: str = "track",
):
    """Render a real UI control inside a reusable student action panel."""
    border = "#F4C27A" if tone == "warn" else "#B8D8C6"
    accent = "var(--mlsysbook-warn)" if tone == "warn" else "var(--mlsysbook-ok)"
    background = "#FFFBF2" if tone == "warn" else "#F8FFFB"
    body_html = f"<p>{html.escape(body)}</p>" if body else ""
    return mo.ui.batch(
        mo.Html(
            f"""
<div class="mlsysbook-action-box" style="width:min(var(--mlsysbook-panel-width),100%);
     max-width:min(var(--mlsysbook-panel-width),100%); margin:14px auto; padding:16px 18px;
     background:linear-gradient(135deg,{background} 0%,#FFFFFF 84%);
     border:1px solid {border}; border-left:4px solid {accent}; border-radius:8px;
     box-shadow:0 4px 12px rgba(31,64,122,0.06);">
  <h3 style="margin:0 0 4px 0; font-size:0.98rem; letter-spacing:0;">{html.escape(title)}</h3>
  <div style="color:#475467; line-height:1.45; margin:0 0 12px 0;">{body_html}</div>
  <div class="mlsysbook-action-control" style="margin-top:8px;">{{{name}}}</div>
</div>
"""
        ),
        {name: element},
    )


def track_context(track: str | TrackProfile) -> mo.Html:
    """Render the selected track profile as a student-facing mission."""
    profile = track if isinstance(track, TrackProfile) else get_track_profile(track)
    metrics = ", ".join(profile.primary_metrics)
    guardrails = ", ".join(profile.guardrail_metrics)
    constraints = ", ".join(profile.dominant_constraints)
    track_delta = (
        f"Watch {profile.primary_metrics[0]} first, protect {profile.guardrail_metrics[0]}, "
        f"and test {profile.dominant_constraints[0]} before treating the design as feasible."
    )
    fields = _render_compact_fields(
        {
            "Track": f"{profile.label} ({profile.category})",
            "Your role": profile.stakeholder,
            "Primary metrics": metrics,
            "Guardrails": guardrails,
            "Dominant constraints": constraints,
            "What changes": track_delta,
        }
    )
    return mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-track-panel">
  <div class="mlsysbook-section-label">Track Mission</div>
  <h2>{html.escape(track_display_label(profile))} Track</h2>
  <div class="mlsysbook-scenario-narrative">
    <p><strong>You are deploying for {html.escape(track_display_label(profile))}.</strong> {html.escape(profile.narrative)}</p>
    <p><strong>Your job in this lab:</strong> {html.escape(track_delta)}</p>
  </div>
  <div class="mlsysbook-compact-fields is-brief">{fields}</div>
</div>
"""
    )


def track_arc_context(track: str | TrackProfile, lab_id: str) -> mo.Html:
    """Render the selected track's cross-lab narrative without registry provenance."""
    profile = track if isinstance(track, TrackProfile) else get_track_profile(track)
    from .track_arcs import track_arc_context_summary

    summary = track_arc_context_summary(profile.track_id, lab_id)
    fields = _render_compact_fields(
        {
            "Journey": summary["Track mission"],
            "This lab's role": summary["This lab's role"],
            "Carry forward": summary["Carry forward"],
            "System goal": summary["System goal"],
        }
    )
    return mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-launch-panel">
  <div class="mlsysbook-section-label">Where This Fits</div>
  <h2>{html.escape(track_display_label(profile))} Journey</h2>
  <div class="mlsysbook-scenario-narrative">
    <p>{html.escape(summary["Volume arc"])}</p>
  </div>
  <div class="mlsysbook-compact-fields is-brief">{fields}</div>
</div>
"""
    )


def part_header(part: str, concept: str, systems_question: str, track_frame: str = "") -> mo.Html:
    """Render the required part anchor and systems question."""
    heading = _part_heading(part, concept)
    frame_html = ""
    if track_frame:
        frame_html = f'<div class="mlsysbook-callout"><strong>Track frame:</strong> {html.escape(track_frame)}</div>'
    return mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-nugget">
  <div class="mlsysbook-part-title">
    <h2>{html.escape(heading)}</h2>
  </div>
  <div class="mlsysbook-callout"><strong>Systems question:</strong> {html.escape(systems_question)}</div>
  {frame_html}
</div>
"""
    )


def what_you_need_to_know(
    bullets: tuple[str, ...] | list[str],
    *,
    equation: str = "",
    track_interpretation: str = "",
    watch_for: str = "",
) -> mo.Html:
    """Render the compact micro-brief for a lab part."""
    extra_fields = {}
    if equation:
        extra_fields["Key relationship"] = equation
    if track_interpretation:
        extra_fields["Track interpretation"] = track_interpretation
    if watch_for:
        extra_fields["Watch for"] = watch_for
    fields = f'<div class="mlsysbook-grid">{_render_fields(extra_fields)}</div>' if extra_fields else ""
    return mo.Html(
        f"""
<div class="mlsysbook-panel">
  <h2>What You Need To Know</h2>
  <ul class="mlsysbook-list">
    {_render_list(bullets)}
  </ul>
  {fields}
</div>
"""
    )


def micro_brief(
    bullets: tuple[str, ...] | list[str],
    *,
    equation: str = "",
    track_interpretation: str = "",
    watch_for: str = "",
) -> mo.Html:
    """Alias for the part-level `What You Need To Know` block."""
    return what_you_need_to_know(
        bullets,
        equation=equation,
        track_interpretation=track_interpretation,
        watch_for=watch_for,
    )


def scenario_slice(
    *,
    stakeholder_pressure: str,
    workload_slice: str,
    active_constraint: str,
    primary_metric: str,
    guardrail_metric: str,
) -> mo.Html:
    """Render the part-specific track situation."""
    return mo.Html(
        f"""
<div class="mlsysbook-panel">
  <h2>Scenario Slice</h2>
  <div class="mlsysbook-grid">
    {_render_fields({
        "Stakeholder pressure": stakeholder_pressure,
        "Workload slice": workload_slice,
        "Active constraint": active_constraint,
        "Primary metric": primary_metric,
        "Guardrail metric": guardrail_metric,
    })}
  </div>
</div>
"""
    )


def nugget_shell(spec: NuggetSpec, body: Any) -> mo.Html:
    return mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-nugget">
  <h2>{html.escape(spec.title)}</h2>
  <div class="mlsysbook-grid">
    <div class="mlsysbook-field"><strong>Chapter idea</strong>{html.escape(spec.chapter_idea)}</div>
    <div class="mlsysbook-field"><strong>Systems question</strong>{html.escape(spec.systems_question)}</div>
    <div class="mlsysbook-field"><strong>Primary knobs</strong>{html.escape(", ".join(spec.primary_knobs))}</div>
    <div class="mlsysbook-field"><strong>Expected constraint</strong>{html.escape(spec.expected_constraint)}</div>
  </div>
  <div class="mlsysbook-callout">{mo.as_html(body).text if hasattr(mo.as_html(body), "text") else body}</div>
</div>
"""
    )


def constraint_check(
    name: str,
    value: Any,
    limit: Any,
    unit: str = "",
    status: str = "pass",
    mitigation: str = "",
) -> mo.Html:
    """Render a feasibility check for the active constraint."""
    status_label, status_class = _constraint_status(status)
    unit_label = f" {unit}" if unit else ""
    mitigation_html = ""
    if mitigation:
        mitigation_html = f'<div class="mlsysbook-callout"><strong>First mitigation:</strong> {html.escape(mitigation)}</div>'
    return mo.Html(
        f"""
<div class="mlsysbook-panel">
  <h2>Constraint Check</h2>
  <div class="mlsysbook-grid">
    <div class="mlsysbook-field"><strong>Constraint</strong>{html.escape(name)}</div>
    <div class="mlsysbook-field"><strong>Value</strong>{html.escape(str(value))}{html.escape(unit_label)}</div>
    <div class="mlsysbook-field"><strong>Limit</strong>{html.escape(str(limit))}{html.escape(unit_label)}</div>
    <div class="mlsysbook-field"><strong>Status</strong><span class="mlsysbook-status {html.escape(status_class)}">{html.escape(status_label)}</span></div>
  </div>
  {mitigation_html}
</div>
"""
    )


def source_trace(sources: Mapping[str, Any] | str, *, collapsed: bool = True, summary: str = "") -> mo.Html:
    """Render MLSysIM APIs, registry refs, equations, and assumptions."""
    if isinstance(sources, Mapping):
        source_items = dict(sources)
        visible_summary = summary or str(
            source_items.get("summary")
            or source_items.get("api")
            or source_items.get("registry")
            or source_items.get("hardware_ref")
            or "MLSysIM-backed source trace"
        )
    else:
        source_items = {"Summary": sources}
        visible_summary = summary or str(sources)

    details_open = "" if collapsed else " open"
    return mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-source-trace">
  <h2>Source Trace</h2>
  <div class="mlsysbook-source-summary">{html.escape(visible_summary)}</div>
  <details{details_open}>
    <summary>Show source details</summary>
    <div class="mlsysbook-grid">
      {_render_fields(source_items)}
    </div>
  </details>
</div>
"""
    )


def evidence_summary(items: Mapping[str, Any], *, caption: str = "") -> mo.Html:
    """Render synthesis evidence in the same shape used by reports."""
    caption_html = f'<div class="mlsysbook-source-summary">{html.escape(caption)}</div>' if caption else ""
    return mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-evidence-panel">
  <h2>Evidence Summary</h2>
  {caption_html}
  <details>
    <summary>Show precomputed evidence</summary>
    <div class="mlsysbook-grid">
      {_render_fields(items)}
    </div>
  </details>
</div>
"""
    )


def checkpoint_card(fields: Mapping[str, Any], *, title: str = "Checkpoint") -> mo.Html:
    """Render saved evidence fields for a part checkpoint or decision."""
    return mo.Html(
        f"""
<div class="mlsysbook-panel">
  <h2>{html.escape(title)}</h2>
  <div class="mlsysbook-grid">
    {_render_fields(fields)}
  </div>
</div>
"""
    )


def big_takeaways(takeaways: tuple[str, ...] | list[str]) -> mo.Html:
    """Render the required end-of-lab carry-forward takeaways."""
    return mo.Html(
        f"""
<div class="mlsysbook-panel mlsysbook-takeaway-panel">
  <h2>Big Takeaways</h2>
  <ul class="mlsysbook-list">
    {_render_list(takeaways)}
  </ul>
</div>
"""
    )


def advanced_knob_drawer(items: dict[str, Any]) -> mo.Html:
    return mo.accordion({"Advanced controls": mo.vstack(list(items.values()))}, multiple=False)


def reflection_card(prompt: str):
    return mo.ui.text_area(label=prompt, placeholder="Explain the trade-off and residual risk.", full_width=True)


def decision_card(prompt: str = "What engineering decision would you defend?"):
    return mo.ui.text_area(label=prompt, placeholder="State the decision, evidence, and residual risk.", full_width=True)


def instructor_adoption_card(metadata: InstructorMetadata) -> mo.Html:
    data = asdict(metadata)
    rows = []
    for key, value in data.items():
        if isinstance(value, tuple):
            value = "; ".join(value)
        rows.append(
            f'<div class="mlsysbook-field"><strong>{html.escape(key.replace("_", " "))}</strong>{html.escape(str(value))}</div>'
        )
    return mo.Html(f'<div class="mlsysbook-panel"><h2>Instructor Adoption</h2><div class="mlsysbook-grid">{"".join(rows)}</div></div>')
