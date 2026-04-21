"""Pydantic models for vault questions (schema v1.0).

Enum values are imported from the vault's single source of truth at
``interviews/vault/schema/enums.py``. See also the LinkML schema at
``interviews/vault/schema/question_schema.yaml``.

v1.0 (2026-04-21): classification is now encoded in YAML fields rather than
the filesystem path. The path carries only `track` for navigability; all
other axes (level, zone, topic, competency_area, bloom_level, phase) live
in the file itself and are validated on load.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Locate the vault's shared enums module regardless of install layout.
_HERE = Path(__file__).resolve()
_REPO_ROOT_CANDIDATES = [
    _HERE.parents[3] / "vault" / "schema",                # interviews/vault/schema/
    _HERE.parents[4] / "interviews" / "vault" / "schema", # worktree root
]
for _candidate in _REPO_ROOT_CANDIDATES:
    if (_candidate / "enums.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from enums import (  # noqa: E402, I001  # type: ignore[import-not-found]
    VALID_BLOOM_LEVELS,
    VALID_HUMAN_REVIEW_STATUSES,
    VALID_LEVELS,
    VALID_PHASES,
    VALID_PROVENANCES,
    VALID_STATUSES,
    VALID_TRACKS,
    VALID_ZONES,
)


# ─── Enums ──────────────────────────────────────────────────────────────────
# These mirror schema/enums.py values. We build Enum classes dynamically so
# adding a value in one place (enums.py) propagates to both the frozensets
# used by validators and the Enum types used by typed consumers.


def _make_enum(name: str, values: frozenset[str]) -> type[Enum]:
    return Enum(name, {_enum_member_key(v): v for v in sorted(values)})  # type: ignore[return-value]


def _enum_member_key(value: str) -> str:
    """Python Enum member names must be valid identifiers."""
    return (
        value.replace("+", "_plus")
        .replace("-", "_")
        .replace(" ", "_")
        .upper()
        if not value.isidentifier()
        else value
    )


Track = _make_enum("Track", VALID_TRACKS)
Level = _make_enum("Level", VALID_LEVELS)
Zone = _make_enum("Zone", VALID_ZONES)
BloomLevel = _make_enum("BloomLevel", VALID_BLOOM_LEVELS)
Phase = _make_enum("Phase", VALID_PHASES)
Status = _make_enum("Status", VALID_STATUSES)
Provenance = _make_enum("Provenance", VALID_PROVENANCES)
HumanReviewStatus = _make_enum("HumanReviewStatus", VALID_HUMAN_REVIEW_STATUSES)


# ─── Leaf models ────────────────────────────────────────────────────────────


class ChainRef(BaseModel):
    """Structured chain reference with position."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    position: int = Field(ge=0)


class Resource(BaseModel):
    """Author-curated external reference (https only)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    url: str

    @field_validator("url")
    @classmethod
    def _https_only(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError(f"resource url must start with https://; got {v!r}")
        return v


class HumanReview(BaseModel):
    """Human review lineage, independent of LLM validation stamps."""

    model_config = ConfigDict(extra="forbid")

    status: str = "not-reviewed"
    by: str | None = None
    date: date | str | None = None
    notes: str | None = None

    @field_validator("status")
    @classmethod
    def _valid(cls, v: str) -> str:
        if v not in VALID_HUMAN_REVIEW_STATUSES:
            raise ValueError(
                f"invalid human_reviewed.status {v!r}, must be one of "
                f"{sorted(VALID_HUMAN_REVIEW_STATUSES)}"
            )
        return v


class Details(BaseModel):
    # Allow unknown keys: a few imported YAMLs carry legacy fields
    # (e.g. details.question) that we preserve but do not validate strictly.
    model_config = ConfigDict(extra="allow")

    realistic_solution: str
    common_mistake: str | None = None
    napkin_math: str | None = None
    options: list[str] | None = None
    correct_index: int | None = Field(default=None, ge=0, le=3)
    resources: list[Resource] | None = None


# ─── Question ───────────────────────────────────────────────────────────────


class Question(BaseModel):
    """A single StaffML interview question (schema v1.0).

    Every classification axis is a YAML field. The filesystem path encodes
    only `track` for navigability; filename prefix must match `track`.
    """

    model_config = ConfigDict(extra="allow")

    # Identity
    schema_version: str = Field(default="1.0")
    id: str

    # 4-axis classification
    track: str
    level: str
    zone: str
    topic: str
    competency_area: str
    bloom_level: str | None = None
    phase: str | None = None

    # Content
    title: str = Field(max_length=120)
    scenario: str
    details: Details

    # Workflow
    status: str = "draft"
    provenance: str = "imported"
    requires_explanation: bool | None = None
    expected_time_minutes: int | None = Field(default=None, ge=0)
    deletion_reason: str | None = None

    # Chain membership (plural)
    chains: list[ChainRef] | None = None

    # LLM validation
    validated: bool | None = None
    validation_status: str | None = None
    validation_date: date | str | None = None
    validation_model: str | None = None
    validation_issues: list[str] | None = None
    validation_status_pro: str | None = None
    validation_issues_pro: list[str] | None = None

    # Math validation
    math_verified: bool | None = None
    math_status: str | None = None
    math_date: date | str | None = None
    math_model: str | None = None
    math_issues: list[str] | None = None

    # Human review
    human_reviewed: HumanReview | None = None

    # Classification review notes
    classification_review: str | None = None

    # Tags + temporal
    tags: list[str] | None = None
    authors: list[str] | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None
    last_modified: date | str | None = None

    @field_validator("schema_version")
    @classmethod
    def _schema_version_known(cls, v: str) -> str:
        # Accept exact "1.0" or any "1.x" for forward compatibility within major v1.
        if not v.startswith("1."):
            raise ValueError(
                f"unsupported schema_version={v!r}; this loader supports v1.x only."
            )
        return v

    @field_validator("track")
    @classmethod
    def _track(cls, v: str) -> str:
        if v not in VALID_TRACKS:
            raise ValueError(f"invalid track {v!r}; must be one of {sorted(VALID_TRACKS)}")
        return v

    @field_validator("level")
    @classmethod
    def _level(cls, v: str) -> str:
        if v not in VALID_LEVELS:
            raise ValueError(f"invalid level {v!r}; must be one of {sorted(VALID_LEVELS)}")
        return v

    @field_validator("zone")
    @classmethod
    def _zone(cls, v: str) -> str:
        if v not in VALID_ZONES:
            raise ValueError(f"invalid zone {v!r}; must be one of {sorted(VALID_ZONES)}")
        return v

    @field_validator("bloom_level")
    @classmethod
    def _bloom(cls, v: str | None) -> str | None:
        if v and v not in VALID_BLOOM_LEVELS:
            raise ValueError(
                f"invalid bloom_level {v!r}; must be one of {sorted(VALID_BLOOM_LEVELS)}"
            )
        return v

    @field_validator("phase")
    @classmethod
    def _phase(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_PHASES:
            raise ValueError(f"invalid phase {v!r}; must be one of {sorted(VALID_PHASES)}")
        return v

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        if v not in VALID_STATUSES:
            raise ValueError(f"invalid status {v!r}; must be one of {sorted(VALID_STATUSES)}")
        return v

    @field_validator("provenance")
    @classmethod
    def _provenance(cls, v: str) -> str:
        if v not in VALID_PROVENANCES:
            raise ValueError(
                f"invalid provenance {v!r}; must be one of {sorted(VALID_PROVENANCES)}"
            )
        return v

    @field_validator("scenario")
    @classmethod
    def _scenario_plaintext(cls, v: str) -> str:
        # XSS defense: scenario is plaintext; reject HTML and suspect URLs.
        lowered = v.lower()
        forbidden = ("<script", "javascript:", "data:text/html", "onerror=", "onload=")
        for token in forbidden:
            if token in lowered:
                raise ValueError(
                    f"scenario contains forbidden token {token!r}; plaintext only"
                )
        return v


__all__ = [
    "Track",
    "Level",
    "Zone",
    "BloomLevel",
    "Phase",
    "Status",
    "Provenance",
    "HumanReviewStatus",
    "ChainRef",
    "Resource",
    "HumanReview",
    "Details",
    "Question",
]
