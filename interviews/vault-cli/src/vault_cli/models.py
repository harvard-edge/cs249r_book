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

import re
import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_VISUAL_PATH_RE = re.compile(r"^[a-z0-9-]+\.svg$")

# Locate the vault's shared enums module regardless of install layout.
_HERE = Path(__file__).resolve()
_REPO_ROOT_CANDIDATES = [
    _HERE.parents[3] / "vault" / "schema",                # interviews/vault/schema/
    _HERE.parents[4] / "interviews" / "vault" / "schema", # worktree root
]
_VAULT_DIR: Path | None = None
for _candidate in _REPO_ROOT_CANDIDATES:
    if (_candidate / "enums.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        # vault root is the parent of schema/. visuals/ lives next to it.
        _VAULT_DIR = _candidate.parent
        break

# Build-time check: if the vault working tree is reachable, validate that
# every visual.path resolves to an actual SVG file. In production deploys
# (no working tree, e.g. a Cloudflare worker), this resolves to None and
# the file-existence check is skipped — the consumer trusts that the
# build pipeline already validated.
_VISUALS_DIR: Path | None = (
    _VAULT_DIR / "visuals" if _VAULT_DIR and (_VAULT_DIR / "visuals").is_dir() else None
)

from enums import (  # noqa: E402, I001  # type: ignore[import-not-found]
    VALID_BLOOM_LEVELS,
    VALID_COMPETENCY_AREAS,
    VALID_HUMAN_REVIEW_STATUSES,
    VALID_LEVELS,
    VALID_PHASES,
    VALID_PROVENANCES,
    VALID_STATUSES,
    VALID_TRACKS,
    VALID_ZONES,
    ZONE_BLOOM_AFFINITY,
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


class Visual(BaseModel):
    """Static visual attached to a question.

    v0.1.2 hardened: kind is a closed enum (svg only), path must match
    `^[a-z0-9-]+\\.svg$`, alt ≥10 chars, caption required ≥5 chars. The
    `mermaid` value was reserved but never shipped; removed to keep the
    enum honest. Re-add when mermaid actually renders in the practice page.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = "svg"
    path: str
    alt: str
    caption: str

    @field_validator("kind")
    @classmethod
    def _supported_kind(cls, v: str) -> str:
        if v != "svg":
            raise ValueError(
                f"visual.kind must be 'svg' (got {v!r}); "
                "mermaid was reserved but never implemented"
            )
        return v

    @field_validator("path")
    @classmethod
    def _safe_path(cls, v: str) -> str:
        if ".." in v or v.startswith("/") or "\\" in v:
            raise ValueError(
                f"visual.path must be a safe relative filename; got {v!r}"
            )
        if not _VISUAL_PATH_RE.match(v):
            raise ValueError(
                f"visual.path must match ^[a-z0-9-]+\\.svg$ "
                f"(lowercase + dash + dot only, must end in .svg); got {v!r}"
            )
        return v

    @field_validator("alt")
    @classmethod
    def _alt_min_length(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError(
                f"visual.alt must be ≥10 chars (a one-word alt is not "
                f"informative); got {len(v)} chars: {v!r}"
            )
        return v

    @field_validator("caption")
    @classmethod
    def _caption_min_length(cls, v: str) -> str:
        if len(v) < 5:
            raise ValueError(
                f"visual.caption must be ≥5 chars; got {len(v)} chars: {v!r}"
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
    question: str | None = None
    visual: Visual | None = None
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

    @field_validator("competency_area")
    @classmethod
    def _area(cls, v: str) -> str:
        # Closed enum (added 2026-04-25 — see fix_competency_areas.py).
        # Catches Gemini-generated drafts that mistakenly populate the
        # area field with a topic name or zone name instead of one of
        # the 13 canonical competency areas.
        if v not in VALID_COMPETENCY_AREAS:
            raise ValueError(
                f"invalid competency_area {v!r}; "
                f"must be one of {sorted(VALID_COMPETENCY_AREAS)}"
            )
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

    @model_validator(mode="after")
    def _zone_bloom_compatible(self) -> Question:
        """Zone and bloom_level must agree on cognitive level.

        v0.1.2 hard rule: every zone admits a specific Bloom verb set
        (ZONE_BLOOM_AFFINITY). When zone says one thing and bloom_level
        says another, the question's classification literally
        contradicts itself. Fixed at the data boundary so future
        generation runs can never write a self-contradicting item.
        See lint-calibration-2026-04-25 (expert consensus) for the
        rule's pedagogical justification.
        """
        if self.bloom_level is None:
            return self
        admits = ZONE_BLOOM_AFFINITY.get(self.zone)
        if admits is None or self.bloom_level in admits:
            return self
        raise ValueError(
            f"zone={self.zone!r} and bloom_level={self.bloom_level!r} "
            f"are incompatible (zone={self.zone!r} admits "
            f"{sorted(admits)}). Run reclassify_zone_bloom_mismatch.py "
            f"to repair, or correct one of the two fields manually."
        )

    @model_validator(mode="after")
    def _visual_path_resolves(self) -> Question:
        """If a visual is declared, the SVG file MUST exist on disk.

        Prevents the v0.1.1 regression where graphviz/matplotlib renders
        crashed silently and a Question shipped with a `visual:` block
        whose `path` pointed to a nonexistent SVG (e.g. mobile-1962). Skipped
        in production deploys where the working tree is not present.
        """
        if self.visual is None or _VISUALS_DIR is None:
            return self
        svg = _VISUALS_DIR / self.track / self.visual.path
        if not svg.is_file():
            raise ValueError(
                f"visual.path does not resolve to a real file: {svg} "
                f"(question id={self.id}). Either render it via "
                f"`render_visuals.py --id {self.id}` or remove the visual "
                f"block from the YAML."
            )
        return self


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
