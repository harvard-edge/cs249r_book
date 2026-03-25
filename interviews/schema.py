"""Pydantic schema for StaffML interview question corpus.

Validates corpus.json against strict rules informed by LeetCode,
Exercism, freeCodeCamp, and IRT best practices.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

VALID_TRACKS = {"cloud", "edge", "mobile", "tinyml", "global"}
VALID_LEVELS = {"L1", "L2", "L3", "L4", "L5", "L6+"}
VALID_AREAS = {
    "compute", "memory", "latency", "precision", "power",
    "architecture", "optimization", "parallelism", "networking",
    "deployment", "reliability", "data", "cross-cutting",
}
VALID_BLOOM = {"remember", "understand", "apply", "analyze", "evaluate", "create", ""}
VALID_STATUS = {"published", "draft", "archived", "flagged"}

# --- v5.3 Taxonomy Controlled Vocabularies ---

VALID_REASONING_COMPETENCIES = {
    "RC-1", "RC-2", "RC-3", "RC-4", "RC-5", "RC-6", "RC-7",
    "RC-8", "RC-9", "RC-10", "RC-11", "RC-12", "RC-13",
}

VALID_KNOWLEDGE_AREAS = {
    # Domain A: ML Foundations
    "A1", "A2", "A3", "A4", "A5", "A6", "A7",
    # Domain B: Hardware & Compute
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
    # Domain C: Systems & Scale
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
    # Domain D: Deployment & Serving
    "D1", "D2", "D3", "D4", "D5", "D6", "D7",
    # Domain E: Efficiency & Governance (merged in v5.3)
    "E1", "E2", "E3",
    # Domain F: Compound AI Systems (merged in v5.3)
    "F1",
}

VALID_REASONING_MODES = {
    "concept-recall",
    "napkin-math",
    "symptom-to-cause",
    "tradeoff-analysis",
    "requirements-to-architecture",
    "optimization-task",
    "failure-to-root-cause",
}


class QuestionDetails(BaseModel):
    common_mistake: str
    realistic_solution: str
    napkin_math: str = ""
    deep_dive_title: str = ""
    deep_dive_url: str = ""
    options: Optional[list[str]] = None
    correct_index: Optional[int] = None

    @field_validator("common_mistake")
    @classmethod
    def common_mistake_min_length(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError(f"common_mistake too short ({len(v)} chars, min 10)")
        return v

    @field_validator("realistic_solution")
    @classmethod
    def realistic_solution_min_length(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError(f"realistic_solution too short ({len(v)} chars, min 5)")
        return v

    @field_validator("napkin_math")
    @classmethod
    def napkin_math_has_substance(cls, v: str) -> str:
        # Only flag if napkin_math is present but suspiciously empty
        # Multi-line extractions may have prose intros without digits — that's OK
        if v and 0 < len(v.strip()) < 5:
            raise ValueError(f"napkin_math too short ({len(v)} chars)")
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
    # Identity
    id: str
    track: str
    scope: str
    level: str
    title: str

    # Classification
    topic: str
    competency_area: str
    canonical_topic: str = ""
    bloom_level: str = ""
    tags: list[str] = []

    # Content
    scenario: str
    details: QuestionDetails

    # Lifecycle
    status: str = "published"
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    # IRT (null until user data)
    difficulty_empirical: Optional[float] = None
    discrimination: Optional[float] = None
    attempt_count: int = 0

    # v5.3 Taxonomy (6-axis classification)
    reasoning_competency: Optional[str] = None   # RC-1 through RC-13
    knowledge_area: Optional[str] = None          # A1 through F1 (35 areas)
    reasoning_mode: Optional[str] = None          # 7 modes (renamed in v5.3)
    concept_tags: Optional[list[str]] = None      # ~150 tags, multi-label
    primary_concept: Optional[str] = None         # preserved taxonomy_concept

    # Validation (stamped by Gemini review)
    validated: Optional[bool] = None              # true if passed Gemini review
    validation_status: Optional[str] = None       # OK, WARN, ERROR
    validation_issues: Optional[list[str]] = None # issues found
    validation_model: Optional[str] = None        # model used for validation
    validation_date: Optional[str] = None         # ISO date of validation

    # Chains
    chain_ids: Optional[list[str]] = None
    chain_positions: Optional[dict[str, int]] = None

    @field_validator("track")
    @classmethod
    def valid_track(cls, v: str) -> str:
        if v not in VALID_TRACKS:
            raise ValueError(f"Invalid track '{v}', must be one of {VALID_TRACKS}")
        return v

    @field_validator("level")
    @classmethod
    def valid_level(cls, v: str) -> str:
        if v not in VALID_LEVELS:
            raise ValueError(f"Invalid level '{v}', must be one of {VALID_LEVELS}")
        return v

    @field_validator("competency_area")
    @classmethod
    def valid_area(cls, v: str) -> str:
        if v not in VALID_AREAS:
            raise ValueError(f"Invalid competency_area '{v}', must be one of {VALID_AREAS}")
        return v

    @field_validator("bloom_level")
    @classmethod
    def valid_bloom(cls, v: str) -> str:
        if v and v not in VALID_BLOOM:
            raise ValueError(f"Invalid bloom_level '{v}', must be one of {VALID_BLOOM}")
        return v

    @field_validator("reasoning_competency")
    @classmethod
    def valid_reasoning_competency(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_REASONING_COMPETENCIES:
            raise ValueError(
                f"Invalid reasoning_competency '{v}', "
                f"must be one of {sorted(VALID_REASONING_COMPETENCIES)}"
            )
        return v

    @field_validator("knowledge_area")
    @classmethod
    def valid_knowledge_area(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_KNOWLEDGE_AREAS:
            raise ValueError(
                f"Invalid knowledge_area '{v}', "
                f"must be one of {sorted(VALID_KNOWLEDGE_AREAS)}"
            )
        return v

    @field_validator("reasoning_mode")
    @classmethod
    def valid_reasoning_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_REASONING_MODES:
            raise ValueError(
                f"Invalid reasoning_mode '{v}', "
                f"must be one of {sorted(VALID_REASONING_MODES)}"
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
        if v.strip().startswith('"') or v.strip().endswith('"'):
            raise ValueError("scenario has stray quotes — should be clean text")
        return v


def validate_corpus(questions: list[dict]) -> tuple[list[Question], list[str]]:
    """Validate a list of question dicts against the schema.

    Returns (valid_questions, errors).
    Errors are strings like "Question 'id': field error message".
    """
    valid = []
    errors = []

    # Parse each question
    for i, q_dict in enumerate(questions):
        try:
            q = Question(**q_dict)
            valid.append(q)
        except Exception as e:
            qid = q_dict.get("id", f"index-{i}")
            errors.append(f"[{qid}] {e}")

    # Cross-question checks
    ids = [q.id for q in valid]
    id_counts = {}
    for qid in ids:
        id_counts[qid] = id_counts.get(qid, 0) + 1
    for qid, count in id_counts.items():
        if count > 1:
            errors.append(f"Duplicate ID: '{qid}' appears {count} times")

    # Duplicate (track, level, title) — tracked as warnings, not blocking errors
    # These are real quality issues but shouldn't block validation
    seen_titles = {}
    warnings = []
    for q in valid:
        if q.status == "published":
            key = (q.track, q.level, q.title)
            if key in seen_titles:
                warnings.append(
                    f"Duplicate title: '{q.title}' in {q.track}/{q.level} "
                    f"(IDs: {seen_titles[key]}, {q.id})"
                )
            else:
                seen_titles[key] = q.id

    return valid, errors, warnings

    return valid, errors
