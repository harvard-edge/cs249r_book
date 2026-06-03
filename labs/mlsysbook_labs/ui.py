"""Reusable Marimo UI components for the MLSysBook trade-off labs."""

from __future__ import annotations

import html
from dataclasses import asdict
from typing import Any

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


def _accent(volume: str) -> str:
    return "#1F407A" if "II" in volume or volume.strip() == "2" else "#A51C30"


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
    rows = "".join(
        f'<div class="mlsysbook-field"><strong>{html.escape(str(k))}</strong>{html.escape(str(v))}</div>'
        for k, v in constraints.items()
    )
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


def track_selector(default: str = DEFAULT_TRACK_ID):
    """Return a Marimo radio selector for the canonical student tracks."""
    selected = normalize_track_id(default)
    return mo.ui.radio(options=track_options(), value=selected, label="Your Track", inline=True)


def track_context(track: str | TrackProfile) -> mo.Html:
    """Render the selected track profile and its MLSysIM source references."""
    profile = track if isinstance(track, TrackProfile) else get_track_profile(track)
    metrics = ", ".join(html.escape(metric) for metric in profile.primary_metrics)
    guardrails = ", ".join(html.escape(metric) for metric in profile.guardrail_metrics)
    constraints = ", ".join(html.escape(constraint) for constraint in profile.dominant_constraints)
    system_ref = profile.system_ref or "single-device profile"
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
  </div>
  <div class="mlsysbook-version">{html.escape(profile.source_policy)}</div>
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
