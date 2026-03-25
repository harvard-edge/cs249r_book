"""
Pydantic schemas for StaffML question generation.

These models enforce the exact structure of the markdown question format,
ensuring LLM output always conforms to the established schema.

Based on: Haladyna, Downing & Rodriguez (2002) — item-writing guidelines
require structured distractor generation with documented misconceptions.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Level(str, Enum):
    """Mastery levels aligned with Bloom's Revised Taxonomy."""
    L1 = "L1"  # Remember
    L2 = "L2"  # Understand
    L3 = "L3"  # Apply
    L4 = "L4"  # Analyze
    L5 = "L5"  # Evaluate
    L6 = "L6+" # Create


class Track(str, Enum):
    """Deployment tracks — each represents a distinct hardware regime."""
    CLOUD = "cloud"
    EDGE = "edge"
    MOBILE = "mobile"
    TINYML = "tinyml"
    FOUNDATIONS = "foundations"


class LevelLabel(str, Enum):
    """Badge labels used in the markdown shield.io badges."""
    L1 = "L1_Foundation"
    L2 = "L2_Analytical"
    L3 = "L3_Junior"
    L4 = "L4_Mid"
    L5 = "L5_Senior"
    L6 = "L6_Staff"


class BadgeColor(str, Enum):
    """Badge colors per level."""
    L1 = "brightgreen"
    L2 = "blue"
    L3 = "brightgreen"
    L4 = "blue"
    L5 = "yellow"
    L6 = "red"


# ---------------------------------------------------------------------------
# Badge / level mapping helpers
# ---------------------------------------------------------------------------

LEVEL_META: dict[str, dict] = {
    "L1": {"label": LevelLabel.L1, "color": BadgeColor.L1, "alt": "Level 1"},
    "L2": {"label": LevelLabel.L2, "color": BadgeColor.L2, "alt": "Level 2"},
    "L3": {"label": LevelLabel.L3, "color": BadgeColor.L3, "alt": "Level 1"},
    "L4": {"label": LevelLabel.L4, "color": BadgeColor.L4, "alt": "Level 2"},
    "L5": {"label": LevelLabel.L5, "color": BadgeColor.L5, "alt": "Level 3"},
    "L6+": {"label": LevelLabel.L6, "color": BadgeColor.L6, "alt": "Level 4"},
}


# ---------------------------------------------------------------------------
# Core question schema
# ---------------------------------------------------------------------------

class MCQOption(BaseModel):
    """A single multiple-choice option."""
    text: str = Field(..., description="The option text")
    is_correct: bool = Field(False, description="Whether this is the correct answer")
    misconception: Optional[str] = Field(
        None,
        description=(
            "The specific misconception this distractor exploits. "
            "Per Haladyna et al. (2002), plausible distractors must come "
            "from documented misconceptions, not random alternatives."
        ),
    )


class NapkinMath(BaseModel):
    """Structured napkin math — the quantitative chain of reasoning.

    This is the crown jewel of each question. It must be a chain of
    3-5 quantitative steps using real hardware constants.
    """
    setup: str = Field(
        ...,
        description="The known values and hardware constants used",
    )
    chain: str = Field(
        ...,
        description=(
            "The full calculation chain, including intermediate results. "
            "Must use real hardware specs, not made-up numbers."
        ),
    )
    conclusion: str = Field(
        ...,
        description="The final quantitative conclusion and its engineering implication",
    )

    def to_markdown(self) -> str:
        """Render as a single napkin math string for the markdown template."""
        return f"{self.setup} {self.chain} {self.conclusion}"


class Question(BaseModel):
    """A complete StaffML interview question.

    This schema maps 1:1 to the markdown <details> template used across
    all track files. Every field is either required or conditionally
    required based on the mastery level.
    """

    # --- Identity ---
    level: str = Field(
        ...,
        description="Mastery level: L1, L2, L3, L4, L5, or L6+",
    )
    title: str = Field(
        ...,
        description=(
            "Concise, evocative title (e.g., 'The HBM vs L1 Latency Gap'). "
            "Should hint at the core insight without giving away the answer."
        ),
    )
    topic: str = Field(
        ...,
        description="Kebab-case topic tag (e.g., 'memory-hierarchy', 'roofline')",
    )
    track: str = Field(
        ...,
        description="Deployment track: cloud, edge, mobile, tinyml, or foundations",
    )

    # --- Content ---
    scenario: str = Field(
        ...,
        description=(
            "The interviewer's question. Must be a realistic scenario, not "
            "'what is X?'. Should set up a specific situation that reveals "
            "a fundamental system constraint."
        ),
    )
    common_mistake: str = Field(
        ...,
        description=(
            "The most common wrong answer and WHY it's wrong. "
            "This is the misconception the question is designed to surface."
        ),
    )
    realistic_solution: str = Field(
        ...,
        description=(
            "The correct answer with full explanation. Must connect back "
            "to a physical or mathematical principle."
        ),
    )
    napkin_math: Optional[str] = Field(
        None,
        description=(
            "Quantitative chain of reasoning using real hardware constants. "
            "Required for L2+. Must include actual numbers, units, and "
            "intermediate calculations. NOT optional for L3+."
        ),
    )
    key_equation: Optional[str] = Field(
        None,
        description="LaTeX equation that captures the core relationship (optional)",
    )

    # --- MCQ options (present in ~30% of questions) ---
    options: Optional[list[MCQOption]] = Field(
        None,
        description=(
            "4 multiple-choice options. Exactly one must be correct. "
            "Each wrong option must exploit a specific misconception."
        ),
    )

    # --- Reference ---
    deep_dive_title: Optional[str] = Field(
        None, description="Title of the textbook section for further reading"
    )
    deep_dive_url: Optional[str] = Field(
        None, description="URL to the textbook section"
    )

    # --- Provenance (per Chip Huyen review: track handwritten vs generated) ---
    source: str = Field(
        "generated",
        description="Origin: 'handwritten' or 'generated'. Enables separate quality tracking.",
    )
    model_used: Optional[str] = Field(
        None, description="Model that generated this question (e.g. gemini-2.5-pro)",
    )
    generation_timestamp: Optional[str] = Field(
        None, description="ISO timestamp of when the question was generated",
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid = {"L1", "L2", "L3", "L4", "L5", "L6+"}
        if v not in valid:
            raise ValueError(f"Level must be one of {valid}, got '{v}'")
        return v

    @field_validator("topic")
    @classmethod
    def validate_topic_kebab(cls, v: str) -> str:
        if " " in v or v != v.lower():
            raise ValueError(
                f"Topic must be kebab-case (lowercase, hyphens), got '{v}'"
            )
        return v

    @field_validator("options")
    @classmethod
    def validate_options(cls, v: list[MCQOption] | None) -> list[MCQOption] | None:
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError(f"Must have exactly 4 options, got {len(v)}")
        correct_count = sum(1 for opt in v if opt.is_correct)
        if correct_count != 1:
            raise ValueError(
                f"Exactly 1 option must be correct, got {correct_count}"
            )
        return v


# ---------------------------------------------------------------------------
# Generation request schema
# ---------------------------------------------------------------------------

class GenerationRequest(BaseModel):
    """Input to the generation pipeline — what to generate."""
    track: str = Field(..., description="Target track")
    concept: str = Field(..., description="The concept to test (from TOPIC_MAP)")
    target_level: str = Field(..., description="Target mastery level")
    competency_area: str = Field(
        ...,
        description="One of the 10 competency areas from TOPIC_MAP",
    )
    chapter_context: Optional[str] = Field(
        None,
        description="Relevant textbook chapter content for grounding",
    )
    hardware_constants: Optional[dict] = Field(
        None,
        description="Relevant hardware constants for napkin math",
    )
    count: int = Field(1, description="Number of questions to generate", ge=1, le=10)


# ---------------------------------------------------------------------------
# Validation result schema
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Output of the validation/solver agent."""
    passed: bool = Field(..., description="Whether the question passed validation")
    solver_answer: Optional[str] = Field(
        None, description="The solver's independent answer"
    )
    solver_agrees: bool = Field(
        False, description="Whether the solver arrived at the same answer"
    )
    arithmetic_correct: bool = Field(
        False, description="Whether the napkin math arithmetic checks out"
    )
    is_duplicate: bool = Field(
        False, description="Whether a semantically similar question exists"
    )
    duplicate_similarity: Optional[float] = Field(
        None, description="Cosine similarity to nearest existing question (0-1)"
    )
    issues: list[str] = Field(
        default_factory=list, description="List of issues found"
    )
