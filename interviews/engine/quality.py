"""
Question quality test suite.

Provides programmatic quality checks that can run without DeepEval
as a dependency. Each check returns a score (0-1) and a pass/fail verdict.

These checks formalize what the expert reviewers identified:
- Reddi: specificity of hardware references
- Dean: arithmetic correctness
- Huyen: production relevance and misconception plausibility

Usage:
    from engine.quality import run_quality_suite
    results = run_quality_suite(question)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .schemas import Question


@dataclass
class QualityCheck:
    """Result of a single quality check."""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    detail: str


@dataclass
class QualityReport:
    """Full quality report for a question."""
    question_title: str
    checks: list[QualityCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def overall_score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(c.score for c in self.checks) / len(self.checks)

    @property
    def failures(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed]


# ---------------------------------------------------------------------------
# Individual quality checks
# ---------------------------------------------------------------------------

def check_scenario_realism(q: Question) -> QualityCheck:
    """Check that the scenario describes a realistic situation, not a textbook quiz."""
    weak_patterns = [
        r"^what is\b",
        r"^define\b",
        r"^which of the following\b",
        r"^true or false\b",
        r"^list the\b",
    ]
    scenario_lower = q.scenario.lower().strip()

    is_weak = any(re.match(p, scenario_lower) for p in weak_patterns)
    has_role = any(w in scenario_lower for w in ["you are", "your team", "you're", "your company", "your manager"])
    has_observation = any(w in scenario_lower for w in ["observe", "notice", "discover", "profiler shows", "nvidia-smi", "alert"])

    score = 0.3  # Base score for having a scenario at all
    if not is_weak:
        score += 0.3
    if has_role:
        score += 0.2
    if has_observation:
        score += 0.2

    return QualityCheck(
        name="Scenario Realism",
        passed=score >= 0.6,
        score=score,
        detail="Weak pattern" if is_weak else f"Role={has_role}, Observation={has_observation}",
    )


def check_napkin_math_depth(q: Question) -> QualityCheck:
    """Check that napkin math has sufficient calculation steps."""
    if not q.napkin_math:
        return QualityCheck(
            name="Napkin Math Depth",
            passed=q.level == "L1",
            score=0.0 if q.level != "L1" else 0.5,
            detail="No napkin math" + (" (acceptable for L1)" if q.level == "L1" else ""),
        )

    # Count calculation steps (lines with numbers and operators)
    lines = q.napkin_math.split("\n")
    calc_lines = [l for l in lines if re.search(r'\d+.*[×÷/\*=]', l)]
    numbers = re.findall(r'\d+\.?\d*', q.napkin_math)

    # Minimum depth by level
    min_steps = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 4, "L6+": 3}
    required = min_steps.get(q.level, 3)

    step_count = max(len(calc_lines), len(numbers) // 3)
    score = min(1.0, step_count / required)

    return QualityCheck(
        name="Napkin Math Depth",
        passed=step_count >= required,
        score=score,
        detail=f"{step_count} calculation steps (need ≥{required} for {q.level})",
    )


def check_distractor_quality(q: Question) -> QualityCheck:
    """Check that MCQ distractors are plausible, not obviously wrong."""
    if not q.options:
        return QualityCheck(
            name="Distractor Quality",
            passed=True,
            score=0.7,  # No MCQ is fine for L4+
            detail="No MCQ options (acceptable for L4+)",
        )

    wrong_opts = [o for o in q.options if not o.is_correct]

    # Check that distractors are substantial (not just "None of the above")
    trivial_patterns = ["none of the above", "all of the above", "n/a", "not applicable"]
    trivial_count = sum(1 for o in wrong_opts if o.text.lower().strip() in trivial_patterns)

    # Check length — too-short distractors are usually weak
    short_count = sum(1 for o in wrong_opts if len(o.text) < 20)

    score = 1.0
    issues = []
    if trivial_count > 0:
        score -= 0.3 * trivial_count
        issues.append(f"{trivial_count} trivial distractors")
    if short_count > 0:
        score -= 0.15 * short_count
        issues.append(f"{short_count} very short distractors")

    score = max(0, score)
    return QualityCheck(
        name="Distractor Quality",
        passed=score >= 0.6,
        score=score,
        detail="; ".join(issues) if issues else "Distractors are substantive",
    )


def check_common_mistake_specificity(q: Question) -> QualityCheck:
    """Check that the common mistake describes a real, specific misconception."""
    cm = q.common_mistake.lower()

    # Weak patterns — too generic to be useful
    weak = [
        "this is wrong", "this is incorrect", "many people think",
        "a common mistake is", "the wrong answer",
    ]
    is_generic = any(w in cm for w in weak)

    # Strong patterns — references specific technical misconceptions
    strong = [
        "assuming", "confusing", "forgetting", "ignoring", "mixing",
        "treating", "overlooking", "underestimating", "overestimating",
    ]
    has_specific = any(w in cm for w in strong)

    score = 0.5
    if has_specific:
        score += 0.3
    if not is_generic:
        score += 0.2
    if len(q.common_mistake) > 60:
        score = min(score + 0.1, 1.0)

    return QualityCheck(
        name="Misconception Specificity",
        passed=score >= 0.6,
        score=score,
        detail="Generic" if is_generic else "Specific misconception identified",
    )


def check_answer_completeness(q: Question) -> QualityCheck:
    """Check that the realistic solution is comprehensive."""
    sol = q.realistic_solution

    score = 0.0
    detail_parts = []

    # Length check
    if len(sol) > 200:
        score += 0.3
        detail_parts.append("sufficient length")
    elif len(sol) > 100:
        score += 0.15
        detail_parts.append("moderate length")
    else:
        detail_parts.append("too brief")

    # Has quantitative backing
    if re.search(r'\d+', sol):
        score += 0.2
        detail_parts.append("has numbers")

    # Explains WHY, not just WHAT
    why_words = ["because", "since", "due to", "this is why", "the reason", "as a result"]
    if any(w in sol.lower() for w in why_words):
        score += 0.3
        detail_parts.append("explains causation")

    # References specific components
    if re.search(r'[A-Z]{2,}|[a-z]+_[a-z]+', sol):
        score += 0.2
        detail_parts.append("references specific components")

    score = min(score, 1.0)
    return QualityCheck(
        name="Answer Completeness",
        passed=score >= 0.5,
        score=score,
        detail=", ".join(detail_parts),
    )


# ---------------------------------------------------------------------------
# Full quality suite
# ---------------------------------------------------------------------------

def run_quality_suite(question: Question) -> QualityReport:
    """Run all quality checks on a question.

    Returns a QualityReport with individual check results and an overall score.
    """
    report = QualityReport(question_title=question.title)

    report.checks.append(check_scenario_realism(question))
    report.checks.append(check_napkin_math_depth(question))
    report.checks.append(check_distractor_quality(question))
    report.checks.append(check_common_mistake_specificity(question))
    report.checks.append(check_answer_completeness(question))

    return report
