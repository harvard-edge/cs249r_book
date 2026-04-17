"""Pydantic models for vault questions.

Hand-authored to match ``vault/schema/question_schema.yaml`` line-for-line until
full LinkML codegen is wired in Phase 2. The LinkML file remains the schema
source of truth; this module must be regenerated whenever that file changes.

A CI drift-check in Phase 2 will compare ``linkml-generate-pydantic`` output
against this file and fail on divergence.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Track(str, Enum):
    cloud = "cloud"
    edge = "edge"
    mobile = "mobile"
    tinyml = "tinyml"
    global_ = "global"  # trailing underscore; 'global' is a Python keyword


class Level(str, Enum):
    l1 = "l1"
    l2 = "l2"
    l3 = "l3"
    l4 = "l4"
    l5 = "l5"
    l6 = "l6"


class Zone(str, Enum):
    recall = "recall"
    fluency = "fluency"
    implement = "implement"
    specification = "specification"
    analyze = "analyze"
    diagnosis = "diagnosis"
    design = "design"
    evaluation = "evaluation"


class Status(str, Enum):
    draft = "draft"
    published = "published"
    deprecated = "deprecated"


class Provenance(str, Enum):
    human = "human"
    llm_draft = "llm-draft"
    llm_then_human_edited = "llm-then-human-edited"
    imported = "imported"


class ChainRef(BaseModel):
    """Structured chain reference — fixes REVIEWS.md H-4 stringly-typed footgun."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    position: int = Field(ge=1)


class GenerationMeta(BaseModel):
    """LLM-generation metadata. Required iff provenance != human."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    prompt_hash: str | None = None
    prompt_cost_usd: float | None = Field(default=None, ge=0.0)
    human_reviewed_at: datetime | None = None


class DeepDive(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    url: str

    @field_validator("url")
    @classmethod
    def _https_only(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError(f"deep_dive.url must start with https://; got {v!r}")
        return v


class Details(BaseModel):
    model_config = ConfigDict(extra="forbid")

    common_mistake: str | None = None
    realistic_solution: str
    napkin_math: str | None = None
    deep_dive: DeepDive | None = None


class Question(BaseModel):
    """A single StaffML interview question.

    Classification (track/level/zone) is encoded in the filesystem PATH, not in
    this model. Moving the file reclassifies the question. See ARCHITECTURE.md
    §3.3 and validator.py for the enforcing invariants.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1)
    id: str
    title: str = Field(max_length=120)
    topic: str
    chain: ChainRef | None = None
    status: Status
    created_at: datetime | None = None
    last_modified: datetime | None = None
    provenance: Provenance
    generation_meta: GenerationMeta | None = None
    authors: list[str] | None = None
    scenario: str
    details: Details
    tags: list[str] | None = None

    @field_validator("schema_version")
    @classmethod
    def _schema_version_known(cls, v: int) -> int:
        if v != 1:
            raise ValueError(
                f"unsupported schema_version={v}; this loader supports v1 only. "
                "See schema/EVOLUTION.md for upgrade rules."
            )
        return v

    @field_validator("scenario")
    @classmethod
    def _scenario_plaintext(cls, v: str) -> str:
        # Fast-tier invariant #10: scenario is plaintext; reject HTML and
        # javascript:/data: URLs to prevent XSS in the rendered site.
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
    "Status",
    "Provenance",
    "ChainRef",
    "GenerationMeta",
    "DeepDive",
    "Details",
    "Question",
]
