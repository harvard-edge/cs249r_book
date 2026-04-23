"""Pydantic schema for the StaffML question corpus (schema v1.0).

Enum values are imported from :mod:`vault.schema.enums` — the single source
of truth. Do not redefine them here. See ``schema/question_schema.yaml`` for
the canonical LinkML schema.

This module validates dict records (e.g. loaded from corpus.json or from
YAMLs via ``vault-cli``'s loader). The authoritative on-disk format is the
per-question YAML described in ``schema/question_schema.yaml``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator

# Import enums from the single source of truth. Because `vault/` is not a
# conventional Python package, we add the schema directory to sys.path.
_THIS_DIR = Path(__file__).resolve().parent
_SCHEMA_DIR = _THIS_DIR / "schema"
if str(_SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(_SCHEMA_DIR))

from enums import (  # noqa: E402  type: ignore[import-not-found]
    VALID_BLOOM_LEVELS,
    VALID_COMPETENCY_AREAS,
    VALID_HUMAN_REVIEW_STATUSES,
    VALID_LEVELS,
    VALID_PHASES,
    VALID_PROVENANCES,
    VALID_STATUSES,
    VALID_TOPICS,
    VALID_TRACKS,
    VALID_ZONES,
)


class Resource(BaseModel):
    """Author-curated external reference attached to a question."""

    name: str
    url: str

    @field_validator("name")
    @classmethod
    def name_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Resource.name must be non-empty")
        if len(v) > 200:
            raise ValueError(f"Resource.name too long ({len(v)} chars, max 200)")
        return v

    @field_validator("url")
    @classmethod
    def url_is_https(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError(f"Resource.url must start with https:// (got: {v[:40]!r})")
        return v


class ChainRef(BaseModel):
    """Structured chain reference with position (plural chains list item)."""

    id: str
    position: int


class HumanReview(BaseModel):
    """Human verification lineage. Distinct from LLM validation stamps."""

    status: str = "not-reviewed"
    by: Optional[str] = None
    date: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: str) -> str:
        if v not in VALID_HUMAN_REVIEW_STATUSES:
            raise ValueError(
                f"invalid human_reviewed.status {v!r}, must be one of "
                f"{sorted(VALID_HUMAN_REVIEW_STATUSES)}"
            )
        return v


class QuestionDetails(BaseModel):
    realistic_solution: str
    common_mistake: str = ""
    napkin_math: str = ""
    resources: list[Resource] = []
    options: Optional[list[str]] = None
    correct_index: Optional[int] = None

    @field_validator("realistic_solution")
    @classmethod
    def realistic_solution_min_length(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError(f"realistic_solution too short ({len(v)} chars, min 5)")
        return v

    @model_validator(mode="after")
    def mcq_consistency(self) -> "QuestionDetails":
        if self.options is not None:
            if len(self.options) != 4:
                raise ValueError(f"MCQ must have exactly 4 options, got {len(self.options)}")
            if self.correct_index is None:
                raise ValueError("MCQ has options but missing correct_index")
            if not (0 <= self.correct_index <= 3):
                raise ValueError(f"correct_index must be 0-3, got {self.correct_index}")
        return self


class Question(BaseModel):
    """A StaffML question (schema v1.0). Every classification axis is a field."""

    # Identity
    schema_version: str = "1.0"
    id: str

    # 4-axis classification
    track: str
    level: str
    zone: str
    topic: str
    competency_area: str
    bloom_level: str = ""
    phase: Optional[str] = None

    # Content
    title: str
    scenario: str
    details: QuestionDetails

    # Workflow
    status: str = "draft"
    provenance: str = "imported"
    requires_explanation: Optional[bool] = None
    expected_time_minutes: Optional[int] = None
    deletion_reason: Optional[str] = None

    # Chain membership (plural)
    chains: list[ChainRef] = []

    # LLM validation
    validated: Optional[bool] = None
    validation_status: Optional[str] = None
    validation_date: Optional[str] = None
    validation_model: Optional[str] = None
    validation_issues: Optional[list[str]] = None
    validation_status_pro: Optional[str] = None
    validation_issues_pro: Optional[list[str]] = None

    # Math validation
    math_verified: Optional[bool] = None
    math_status: Optional[str] = None
    math_date: Optional[str] = None
    math_model: Optional[str] = None
    math_issues: Optional[list[str]] = None

    # Human review (new in v1.0)
    human_reviewed: Optional[HumanReview] = None

    # Pro-model classification review notes
    classification_review: Optional[str] = None

    # Tags + temporal
    tags: list[str] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_modified: Optional[str] = None

    @field_validator("track")
    @classmethod
    def valid_track(cls, v: str) -> str:
        if v not in VALID_TRACKS:
            raise ValueError(f"Invalid track {v!r}, must be one of {sorted(VALID_TRACKS)}")
        return v

    @field_validator("level")
    @classmethod
    def valid_level(cls, v: str) -> str:
        if v not in VALID_LEVELS:
            raise ValueError(f"Invalid level {v!r}, must be one of {sorted(VALID_LEVELS)}")
        return v

    @field_validator("zone")
    @classmethod
    def valid_zone(cls, v: str) -> str:
        if v not in VALID_ZONES:
            raise ValueError(f"Invalid zone {v!r}, must be one of {sorted(VALID_ZONES)}")
        return v

    @field_validator("topic")
    @classmethod
    def valid_topic(cls, v: str) -> str:
        if v not in VALID_TOPICS:
            raise ValueError(f"Invalid topic {v!r} (not in {len(VALID_TOPICS)}-topic curated list)")
        return v

    @field_validator("competency_area")
    @classmethod
    def valid_area(cls, v: str) -> str:
        if v not in VALID_COMPETENCY_AREAS:
            raise ValueError(
                f"Invalid competency_area {v!r}, must be one of {sorted(VALID_COMPETENCY_AREAS)}"
            )
        return v

    @field_validator("bloom_level")
    @classmethod
    def valid_bloom(cls, v: str) -> str:
        if v and v not in VALID_BLOOM_LEVELS:
            raise ValueError(
                f"Invalid bloom_level {v!r}, must be one of {sorted(VALID_BLOOM_LEVELS)}"
            )
        return v

    @field_validator("phase")
    @classmethod
    def valid_phase(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_PHASES:
            raise ValueError(f"Invalid phase {v!r}, must be one of {sorted(VALID_PHASES)}")
        return v

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: str) -> str:
        if v not in VALID_STATUSES:
            raise ValueError(f"Invalid status {v!r}, must be one of {sorted(VALID_STATUSES)}")
        return v

    @field_validator("provenance")
    @classmethod
    def valid_provenance(cls, v: str) -> str:
        if v not in VALID_PROVENANCES:
            raise ValueError(
                f"Invalid provenance {v!r}, must be one of {sorted(VALID_PROVENANCES)}"
            )
        return v

    @field_validator("title")
    @classmethod
    def title_min_length(cls, v: str) -> str:
        if len(v.strip()) < 3:
            raise ValueError(f"title too short ({len(v)} chars, min 3)")
        return v

    @field_validator("scenario")
    @classmethod
    def scenario_quality(cls, v: str) -> str:
        if len(v.strip()) < 30:
            raise ValueError(f"scenario too short ({len(v)} chars, min 30)")
        return v


def validate_corpus(questions: list[dict]) -> tuple[list["Question"], list[str], list[str]]:
    """Validate a list of question dicts against the schema.

    Returns (valid_questions, errors, warnings).
    """
    valid: list[Question] = []
    errors: list[str] = []

    for i, q_dict in enumerate(questions):
        try:
            q = Question(**q_dict)
            valid.append(q)
        except Exception as e:
            qid = q_dict.get("id", f"index-{i}")
            errors.append(f"[{qid}] {e}")

    id_counts: dict[str, int] = {}
    for q in valid:
        id_counts[q.id] = id_counts.get(q.id, 0) + 1
    for qid, count in id_counts.items():
        if count > 1:
            errors.append(f"Duplicate ID: {qid!r} appears {count} times")

    seen_titles: dict[tuple[str, str, str], str] = {}
    warnings: list[str] = []
    for q in valid:
        key = (q.track, q.level, q.title)
        if key in seen_titles:
            warnings.append(
                f"Duplicate title: {q.title!r} in {q.track}/{q.level} "
                f"(IDs: {seen_titles[key]}, {q.id})"
            )
        else:
            seen_titles[key] = q.id

    return valid, errors, warnings
