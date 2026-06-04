"""Reusable Marimo UI components for the MLSysBook trade-off labs."""

from __future__ import annotations

import html
from dataclasses import asdict
from typing import Any, Mapping

import marimo as mo

from .schemas import ChapterRecap, InstructorMetadata, LabMetadata, NuggetSpec, TrackProfile
from .tracks import DEFAULT_TRACK_ID, get_track_profile, normalize_track_id, track_options


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
  --mlsysbook-ok: #247A4D;
  --mlsysbook-warn: #9A5B00;
  --mlsysbook-danger: #B42318;
  --mlsysbook-readable-width: 720px;
  --mlsysbook-panel-width: 840px;
}
.mlsysbook-lab-shell {
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--mlsysbook-ink);
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
  max-width: min(var(--mlsysbook-panel-width), 100%);
}
.mlsysbook-recap h2,
.mlsysbook-panel h2 {
  margin: 0 0 10px 0;
  font-size: 1.05rem;
  letter-spacing: 0;
}
.mlsysbook-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
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
.mlsysbook-list {
  margin: 8px 0 0 0;
  padding-left: 20px;
}
.mlsysbook-list li {
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
  overflow-x: auto !important;
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
    max-width: 100%;
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
    chip_html = "".join(f'<span class="mlsysbook-chip">{html.escape(chip)}</span>' for chip in chips)
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
      <span class="mlsysbook-chip">MLSysIM {html.escape(metadata.mlsysim_version)}</span>
      {chip_html}
    </div>
  </div>
</div>
"""
    )


def learning_objectives(objectives: tuple[str, ...] | list[str]) -> mo.Html:
    """Render the required measurable objectives block."""
    return mo.Html(
        f"""
<div class="mlsysbook-panel">
  <h2>Learning Objectives</h2>
  <ol class="mlsysbook-list">
    {_render_list(objectives)}
  </ol>
</div>
"""
    )


def chapter_recap(recap: ChapterRecap) -> mo.Html:
    """Render the required self-contained mini chapter recap."""
    terms = ", ".join(html.escape(term) for term in recap.key_terms)
    return mo.Html(
        f"""
<div class="mlsysbook-recap">
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
    rows = _render_fields(constraints)
    return mo.Html(
        f"""
<div class="mlsysbook-panel">
  <h2>{html.escape(title)}</h2>
  <div class="mlsysbook-callout"><strong>Stakeholder:</strong> {html.escape(stakeholder)}</div>
  <div class="mlsysbook-callout"><strong>Objective:</strong> {html.escape(objective)}</div>
  <div class="mlsysbook-grid">{rows}</div>
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
    <div class="mlsysbook-field">
      <strong>{html.escape(heading)}</strong>
      <span class="mlsysbook-status {html.escape(status_class)}">{html.escape(status_label)}</span>
      <div class="mlsysbook-version">{html.escape(question)}</div>
    </div>
"""
        )
    return mo.Html(
        f"""
<div class="mlsysbook-panel">
  <h2>Lab Map</h2>
  <div class="mlsysbook-grid">
    {"".join(rows)}
  </div>
</div>
"""
    )


def track_selector(default: str = DEFAULT_TRACK_ID):
    """Return a Marimo radio selector for the canonical student tracks."""
    selected = normalize_track_id(default)
    options = track_options()
    selected_label = next((label for label, track_id in options.items() if track_id == selected), "iPhone")
    return mo.ui.radio(options=options, value=selected_label, label="Your Track", inline=True)


def track_context(track: str | TrackProfile) -> mo.Html:
    """Render the selected track profile and its MLSysIM source references."""
    profile = track if isinstance(track, TrackProfile) else get_track_profile(track)
    metrics = ", ".join(html.escape(metric) for metric in profile.primary_metrics)
    guardrails = ", ".join(html.escape(metric) for metric in profile.guardrail_metrics)
    constraints = ", ".join(html.escape(constraint) for constraint in profile.dominant_constraints)
    system_ref = profile.system_ref or "single-device profile"
    track_delta = (
        f"Watch {profile.primary_metrics[0]} first, protect {profile.guardrail_metrics[0]}, "
        f"and test {profile.dominant_constraints[0]} before treating the design as feasible."
    )
    return mo.Html(
        f"""
<div class="mlsysbook-panel">
  <h2>Your Track</h2>
  <div class="mlsysbook-grid">
    <div class="mlsysbook-field"><strong>Track</strong>{html.escape(profile.label)} ({html.escape(profile.category)})</div>
    <div class="mlsysbook-field"><strong>Stakeholder</strong>{html.escape(profile.stakeholder)}</div>
    <div class="mlsysbook-field"><strong>Hardware source</strong><code>{html.escape(profile.hardware_ref)}</code></div>
    <div class="mlsysbook-field"><strong>System source</strong><code>{html.escape(system_ref)}</code></div>
    <div class="mlsysbook-field"><strong>Primary metrics</strong>{metrics}</div>
    <div class="mlsysbook-field"><strong>Guardrails</strong>{guardrails}</div>
    <div class="mlsysbook-field"><strong>Dominant constraints</strong>{constraints}</div>
    <div class="mlsysbook-field"><strong>Narrative</strong>{html.escape(profile.narrative)}</div>
    <div class="mlsysbook-field"><strong>What changed because of your track</strong>{html.escape(track_delta)}</div>
  </div>
  <div class="mlsysbook-version">{html.escape(profile.source_policy)}</div>
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
<div class="mlsysbook-panel">
  <h2>Evidence Summary</h2>
  {caption_html}
  <div class="mlsysbook-grid">
    {_render_fields(items)}
  </div>
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
<div class="mlsysbook-panel">
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
