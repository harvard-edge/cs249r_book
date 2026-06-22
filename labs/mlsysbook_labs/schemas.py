"""Typed metadata contracts for MLSysBook interactive labs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .versions import LEDGER_SCHEMA_VERSION, MLSYSBOOK_LABS_VERSION, REPORT_SCHEMA_VERSION


@dataclass(frozen=True)
class LabMetadata:
    lab_id: str
    title: str
    volume: str
    chapter: str
    book_anchor: str
    lab_version: str = "1.0.0"
    updated_at: str = "2026-06-02"
    release_channel: str = "dev"
    report_schema_version: str = REPORT_SCHEMA_VERSION
    ledger_schema_version: str = LEDGER_SCHEMA_VERSION
    mlsysbook_labs_version: str = MLSYSBOOK_LABS_VERSION
    mlsysim_version: str = "0.1.2"


@dataclass(frozen=True)
class ChapterRecap:
    emphasis: str
    key_terms: tuple[str, ...]
    ml_concept: str
    systems_translation: str
    what_to_watch: str
    common_trap: str
    suggested_reading: str


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    label: str
    context: str
    constraints: tuple[str, ...]


@dataclass(frozen=True)
class TrackProfile:
    track_id: str
    label: str
    category: str
    hardware_ref: str
    stakeholder: str
    primary_metrics: tuple[str, ...]
    guardrail_metrics: tuple[str, ...]
    dominant_constraints: tuple[str, ...]
    narrative: str
    source_policy: str
    system_ref: str | None = None


@dataclass(frozen=True)
class LabTrackVariant:
    lab_id: str
    track_id: str
    scenario_id: str
    stakeholder: str
    workload_summary: str
    objective: str
    primary_metric: str
    guardrail_metric: str
    hardware_ref: str
    model_ref: str = ""
    system_ref: str | None = None
    defaults: Mapping[str, Any] = field(default_factory=dict)
    assumptions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NuggetSpec:
    nugget_id: str
    title: str
    chapter_idea: str
    systems_question: str
    primary_knobs: tuple[str, ...]
    expected_constraint: str
    visual: str
    reflection_prompt: str


@dataclass(frozen=True)
class InstructorMetadata:
    why_assign: str
    where_it_fits: str
    assignment_prompt: str
    expected_report: str
    rubric: tuple[str, ...]
    misconceptions: tuple[str, ...]
    discussion_prompts: tuple[str, ...]
    extensions: tuple[str, ...] = ()
    setup_notes: tuple[str, ...] = ()


@dataclass
class LedgerEntry:
    lab_id: str
    track: str
    scenario: str
    predictions: dict[str, Any] = field(default_factory=dict)
    knob_settings: dict[str, Any] = field(default_factory=dict)
    binding_constraints: dict[str, Any] = field(default_factory=dict)
    decisions: dict[str, Any] = field(default_factory=dict)
    reflections: dict[str, Any] = field(default_factory=dict)
    residual_risk: str = ""
    result_snapshot: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ledger_schema_version: str = LEDGER_SCHEMA_VERSION


@dataclass
class LabReport:
    metadata: LabMetadata
    student_id: str
    track: str
    scenario: str
    markdown: str
    snapshot: dict[str, Any]
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def to_plain(value: Any) -> Any:
    """Convert common result objects into JSON-compatible plain data."""
    if is_dataclass(value):
        return {k: to_plain(v) for k, v in asdict(value).items()}
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump())
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(v) for v in value]
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        return {"value": value.magnitude, "unit": str(value.units)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


DEFAULT_TRACKS = (
    TrackSpec(
        track_id="mobile",
        label="Mobile ML",
        context="Phone or app deployment.",
        constraints=("latency", "battery", "memory", "privacy", "thermal"),
    ),
    TrackSpec(
        track_id="tinyml",
        label="TinyML",
        context="Microcontroller or sensor deployment.",
        constraints=("sram", "flash", "energy", "sampling rate", "connectivity"),
    ),
    TrackSpec(
        track_id="edge",
        label="Edge AI",
        context="Gateway, local server, vehicle, or near-user system.",
        constraints=("network", "local compute", "privacy", "reliability"),
    ),
    TrackSpec(
        track_id="cloud",
        label="Cloud/Fleet",
        context="Datacenter or global service.",
        constraints=("throughput", "p99 latency", "cost", "utilization", "carbon"),
    ),
)
