"""Data models for the Volume IV Student Simulation and Pedagogical Audit Pipeline."""

from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Discipline(str, Enum):
    MACHINE_LEARNING = "Machine Learning"
    EMBEDDED_SYSTEMS = "Embedded Systems"
    CONTROL_ROBOTICS = "Control & Robotics"
    PEDAGOGY_FLOW = "Generalist / Pedagogy Flow"


class ConfusionCategory(str, Enum):
    UNANNOUNCED_TERM = "UNANNOUNCED_TERM"  # Jargon or technical acronym used before definition
    MISSING_PHYSICAL_GROUNDING = "MISSING_PHYSICAL_GROUNDING"  # Math or concept detached from physical reality (F=ma, torque, energy)
    UNJUSTIFIED_SYSTEMS_ASSUMPTION = "UNJUSTIFIED_SYSTEMS_ASSUMPTION"  # Hand-wavy latency, memory, or determinism claims
    CONTROL_STABILITY_VOID = "CONTROL_STABILITY_VOID"  # Ignoring feedback stability, actuator saturation, or inertia
    COGNITIVE_LEAP = "COGNITIVE_LEAP"  # Abrupt conceptual leap without scaffolding or stepping stones
    INVERTED_ORDER = "INVERTED_ORDER"  # Solution taught before problem is motivated
    EXCESSIVE_DENSITY = "EXCESSIVE_DENSITY"  # Too many new concepts introduced in a single paragraph


class Priority(str, Enum):
    P0_BLOCKER = "P0_BLOCKER"  # Fundamental obstacle to comprehension
    P1_SIGNIFICANT = "P1_SIGNIFICANT"  # Noticeable friction or stumbling block
    P2_POLISH = "P2_POLISH"  # Minor clarity enhancement or stylistic bridge


class MarginNote(BaseModel):
    """Line-by-line margin annotation written by an individual student reader."""
    line_start: int
    line_end: int
    text_snippet: str
    student_id: str
    student_name: str
    discipline: Discipline
    clarity_score: int = Field(ge=1, le=5, description="1 (baffled) to 5 (crystal clear)")
    category: Optional[ConfusionCategory] = None
    note: str = Field(description="Student's line-by-line margin thought")
    proposed_bridge: Optional[str] = Field(None, description="How the student wishes the text introduced this")


class DiscussionTurn(BaseModel):
    """A single turn in the seminar room collective discussion."""
    speaker: str
    speaker_role: str
    target_speaker: Optional[str] = None
    text: str


class SeminarTopic(BaseModel):
    """A specific passage discussed collectively by the student pool."""
    topic_id: str
    line_range: str
    text_snippet: str
    student_notes: List[MarginNote]
    discussion: List[DiscussionTurn]
    consensus_verdict: str
    priority: Priority
    original_text: str
    proposed_rewrite: str
    rationale: str


class TriDisciplineScorecard(BaseModel):
    """Pedagogical balance scorecard across the three core disciplines."""
    ml_clarity_score: float = Field(ge=1.0, le=5.0)
    systems_clarity_score: float = Field(ge=1.0, le=5.0)
    control_clarity_score: float = Field(ge=1.0, le=5.0)
    progressive_disclosure_index: float = Field(ge=1.0, le=5.0)
    summary: str


class WeeklyReport(BaseModel):
    """Complete synthesized weekly seminar report for the textbook authors."""
    week_number: int
    chapter_num: int
    chapter_title: str
    audited_file: str
    scorecard: TriDisciplineScorecard
    topics: List[SeminarTopic]
    key_takeaways: List[str]
    appendix_referrals_suggested: List[str]
