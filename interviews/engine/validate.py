"""
Validation engine — the adversarial quality gate.

Every generated question must pass three checks before being accepted:

1. SOLVER CHECK: An independent LLM call at temperature 0 attempts to
   answer the question using only the hardware constants. If the solver
   arrives at a different answer, the question is flagged.

2. ARITHMETIC CHECK: The napkin math is parsed for numerical claims and
   independently verified. Off-by-more-than-10% fails.

3. DEDUP CHECK: The question is embedded and compared against the existing
   corpus in ChromaDB. Cosine similarity > 0.85 = duplicate.

Based on: Stage 4 of the GENERATION_PIPELINE.md — Adversarial Validation.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional

import subprocess
import tempfile

from .schemas import Question, ValidationResult


# ---------------------------------------------------------------------------
# Solver agent — independent answer verification
# ---------------------------------------------------------------------------

SOLVER_PROMPT_TEMPLATE = """You are a precise ML systems engineer solving an interview question.
You must answer using ONLY the hardware constants and physics principles provided.
Do NOT guess. If you cannot determine the answer from the given information, say "INSUFFICIENT DATA".

## HARDWARE CONSTANTS
{hardware_context}

## QUESTION
{scenario}

## INSTRUCTIONS
1. Work through the problem step by step
2. Show your napkin math with explicit calculations
3. State your final answer clearly
4. Rate your confidence: HIGH, MEDIUM, or LOW

Return your response as JSON:
{{
  "reasoning": "Step-by-step working...",
  "answer": "Your final answer...",
  "napkin_math": "Your independent calculations...",
  "confidence": "HIGH|MEDIUM|LOW"
}}
"""


def _run_solver(
    question: Question,
    hardware_context: str,
    model_name: str = "gemini-2.5-pro",
    api_key: Optional[str] = None,
) -> dict:
    """Run the solver agent to independently answer the question.

    Uses gemini CLI with cached credentials for local execution.
    """
    prompt = SOLVER_PROMPT_TEMPLATE.format(
        hardware_context=hardware_context,
        scenario=question.scenario,
    )

    result = subprocess.run(
        ["gemini", "--model", model_name, "-"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Solver failed: {result.stderr[:300]}")

    raw = result.stdout.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"reasoning": raw, "answer": "PARSE_ERROR", "confidence": "LOW"}


# ---------------------------------------------------------------------------
# Arithmetic checker
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from a text string."""
    # Match integers, decimals, and scientific notation
    pattern = r'(?<![a-zA-Z])(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*(?:×\s*10[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?'
    matches = re.findall(pattern, text or "")
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            continue
    return numbers


# Track-specific arithmetic tolerances.
# Per Reddi review: 25% is dangerously wide for resource-constrained tracks
# where the difference between "fits" and "crashes" is a few KB.
ARITHMETIC_TOLERANCE: dict[str, float] = {
    "cloud": 0.25,       # 25% — cloud numbers are large, estimates are coarser
    "edge": 0.15,        # 15% — tighter thermal/power budgets
    "mobile": 0.10,      # 10% — strict memory budgets (shared RAM)
    "tinyml": 0.05,      # 5%  — 256 KB SRAM, every byte counts
    "foundations": 0.20,  # 20% — general physics, moderate tolerance
}


def _check_arithmetic(
    question: Question,
    solver_result: dict,
) -> tuple[bool, list[str]]:
    """Compare the key numbers in question napkin math vs solver's napkin math.

    Tolerance is track-specific: tighter for resource-constrained tracks.
    Returns (passed, issues).
    """
    if not question.napkin_math:
        return True, []

    tolerance = ARITHMETIC_TOLERANCE.get(question.track, 0.25)

    q_numbers = _extract_numbers(question.napkin_math)
    s_numbers = _extract_numbers(solver_result.get("napkin_math", ""))

    if not q_numbers or not s_numbers:
        return True, ["Could not extract numbers for comparison"]

    issues = []

    # Check if the solver's key numbers appear in the question's math
    # We compare the largest numbers (likely the final answers)
    q_sorted = sorted(q_numbers, reverse=True)[:5]
    s_sorted = sorted(s_numbers, reverse=True)[:5]

    lo = 1 - tolerance
    hi = 1 + tolerance

    for sq in q_sorted[:3]:
        found_match = False
        for ss in s_sorted:
            if sq == 0 and ss == 0:
                found_match = True
                break
            if sq != 0:
                ratio = ss / sq if sq != 0 else float("inf")
                if lo <= ratio <= hi:
                    found_match = True
                    break
        if not found_match and sq > 1:  # Ignore small constants
            issues.append(
                f"Question claims {sq:.4g} but solver's numbers don't match "
                f"within {tolerance:.0%} tolerance ({question.track} track)"
            )

    passed = len(issues) == 0
    return passed, issues


# ---------------------------------------------------------------------------
# Dedup checker (requires ChromaDB — optional)
# ---------------------------------------------------------------------------

def _check_duplicate(
    question: Question,
    chroma_collection=None,
    threshold: float = 0.85,
) -> tuple[bool, float]:
    """Check if a semantically similar question already exists.

    Returns (is_duplicate, similarity_score).
    """
    if chroma_collection is None:
        return False, 0.0

    query_text = f"{question.title} {question.scenario}"
    results = chroma_collection.query(
        query_texts=[query_text],
        n_results=1,
    )

    if not results["distances"] or not results["distances"][0]:
        return False, 0.0

    # ChromaDB with cosine space returns cosine distance (0 = identical, 1 = orthogonal, 2 = opposite)
    # Similarity = 1 - distance (per ChromaDB docs for cosine space)
    distance = results["distances"][0][0]
    similarity = max(0, 1 - distance)

    return similarity >= threshold, similarity


# ---------------------------------------------------------------------------
# Readability check (textstat)
# ---------------------------------------------------------------------------

# Target grade levels by mastery level.
# L1/L2 should be accessible to undergrads; L5/L6+ can be grad-level.
READABILITY_TARGETS: dict[str, tuple[float, float]] = {
    "L1": (10, 16),   # High school to undergrad
    "L2": (10, 16),
    "L3": (12, 18),   # Undergrad to early grad
    "L4": (12, 18),
    "L5": (14, 20),   # Grad level acceptable
    "L6+": (14, 22),  # Professional level acceptable
}


def _check_readability(question: Question) -> tuple[bool, list[str]]:
    """Check if the question's readability matches its target level.

    Uses Flesch-Kincaid Grade Level. Note: FK underestimates difficulty
    for technical text (doesn't account for domain jargon), so we treat
    it as a floor check — flagging only when text is egregiously complex.
    """
    try:
        import textstat
        # Test that textstat works (needs nltk cmudict)
        grade = textstat.flesch_kincaid_grade(question.scenario)
    except (ImportError, LookupError, Exception):
        return True, []  # Skip if textstat not installed or nltk data missing

    issues = []
    lo, hi = READABILITY_TARGETS.get(question.level, (10, 22))

    if grade > hi:
        issues.append(
            f"Readability too complex for {question.level}: grade {grade:.1f} "
            f"(target: {lo}-{hi}). Simplify sentence structure."
        )

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Specificity check (per Reddi review)
# ---------------------------------------------------------------------------

# Hardware-intensive tracks should reference specific components, not just generics.
# Per Reddi: "LLM-generated questions tend toward 'SIMD is faster' without
# referencing specific instructions (SMLAD, VMLADAVA.S8)"
SPECIFICITY_MARKERS: dict[str, list[str]] = {
    "tinyml": [
        "cortex", "m0", "m4", "m7", "m33", "sram", "flash", "dma",
        "cmsis", "smlad", "mve", "helium", "simd", "nrf", "stm32",
        "esp32", "rp2040", "mcu", "watchdog", "fpu", "int8", "int4",
        "arduino", "tflite", "micro", "xip", "swo", "swd",
    ],
    "edge": [
        "jetson", "orin", "hailo", "rockchip", "npu", "dla", "tensorrt",
        "cuda", "nvlink", "lpddr", "mipi", "csi", "tops/w", "dvfs",
        "coral", "tpu", "opencv", "gstreamer", "deepstream",
    ],
    "mobile": [
        "ane", "coreml", "tflite", "nnapi", "hexagon", "snapdragon",
        "a17", "bionic", "npu", "metal", "anr", "ufs", "lpddr5",
        "delegation", "xcode", "mlmodel",
    ],
    "cloud": [
        "h100", "a100", "b200", "hbm", "nvlink", "infiniband",
        "tensor core", "cuda", "nccl", "allreduce", "pcie", "rdma",
        "vllm", "triton", "trt-llm", "kv-cache", "pagedattention",
    ],
}

# Minimum specificity: at least N domain-specific terms in the full answer
MIN_SPECIFICITY: dict[str, int] = {
    "tinyml": 3,  # Must reference specific MCU components
    "edge": 2,    # Must reference specific accelerator features
    "mobile": 2,  # Must reference specific mobile frameworks/chips
    "cloud": 2,   # Must reference specific GPU features
}


def _check_specificity(question: Question) -> tuple[bool, list[str]]:
    """Check if the question uses specific technical terms, not just generics.

    Per expert review: auto-generated questions tend toward vague descriptions
    ('the accelerator is faster') instead of specific references ('the Cortex-M4
    SMLAD instruction performs 2 MAC/cycle').
    """
    markers = SPECIFICITY_MARKERS.get(question.track, [])
    if not markers:
        return True, []

    min_count = MIN_SPECIFICITY.get(question.track, 1)

    # Search across all text fields
    full_text = " ".join([
        question.scenario,
        question.realistic_solution,
        question.napkin_math or "",
    ]).lower()

    found = [m for m in markers if m in full_text]
    issues = []

    if len(found) < min_count:
        issues.append(
            f"Low specificity for {question.track} track: found {len(found)} "
            f"domain terms (need ≥{min_count}). Consider referencing specific "
            f"hardware/tools: {', '.join(markers[:5])}..."
        )

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

def validate_question(
    question: Question,
    hardware_context: str,
    chroma_collection=None,
    model_name: str = "gemini-2.5-pro-preview-06-05",
    api_key: Optional[str] = None,
    skip_solver: bool = False,
) -> ValidationResult:
    """Run the full validation pipeline on a generated question.

    Args:
        question: The question to validate
        hardware_context: Hardware constants text for the solver
        chroma_collection: Optional ChromaDB collection for dedup
        model_name: Gemini model for the solver agent
        api_key: Google AI API key
        skip_solver: Skip the LLM solver check (for testing)

    Returns:
        ValidationResult with pass/fail and detailed issues
    """
    issues: list[str] = []
    solver_answer = None
    solver_agrees = False
    arithmetic_correct = True

    # --- Step 1: Solver check ---
    if not skip_solver:
        try:
            solver_result = _run_solver(
                question, hardware_context, model_name, api_key
            )
            solver_answer = solver_result.get("answer", "")
            confidence = solver_result.get("confidence", "LOW")

            if confidence == "LOW":
                issues.append(
                    "Solver has LOW confidence — question may be ambiguous"
                )

            if solver_result.get("answer") == "INSUFFICIENT DATA":
                issues.append(
                    "Solver could not answer from hardware constants alone "
                    "— question may require domain knowledge not in the constants"
                )

            # Arithmetic comparison
            arith_ok, arith_issues = _check_arithmetic(question, solver_result)
            arithmetic_correct = arith_ok
            issues.extend(arith_issues)

            # Simple agreement check: if solver's confidence is HIGH and
            # the key numbers roughly match, we consider it agreement
            if confidence == "HIGH" and arith_ok:
                solver_agrees = True

        except Exception as e:
            issues.append(f"Solver check failed: {e}")
    else:
        arithmetic_correct = True
        solver_agrees = True

    # --- Step 2: Structural checks ---
    if not question.napkin_math and question.level not in ("L1",):
        issues.append(f"Missing napkin math for {question.level} question")

    if question.options:
        correct_count = sum(1 for o in question.options if o.is_correct)
        if correct_count != 1:
            issues.append(f"Options have {correct_count} correct answers (need exactly 1)")

    if len(question.scenario) < 50:
        issues.append("Scenario is too short — needs more context")

    if len(question.common_mistake) < 20:
        issues.append("Common mistake explanation is too brief")

    # --- Step 3: Dedup check ---
    is_dup, dup_sim = _check_duplicate(question, chroma_collection)
    if is_dup:
        issues.append(
            f"Duplicate detected: {dup_sim:.2%} similarity to existing question"
        )

    # --- Step 4: Readability check ---
    readability_ok, readability_issues = _check_readability(question)
    issues.extend(readability_issues)

    # --- Step 5: Specificity check (hardware tracks) ---
    specificity_ok, specificity_issues = _check_specificity(question)
    issues.extend(specificity_issues)

    # --- Verdict ---
    passed = (
        (solver_agrees or skip_solver)
        and arithmetic_correct
        and not is_dup
        and len(issues) <= 1  # Allow 1 minor issue
    )

    return ValidationResult(
        passed=passed,
        solver_answer=solver_answer,
        solver_agrees=solver_agrees,
        arithmetic_correct=arithmetic_correct,
        is_duplicate=is_dup,
        duplicate_similarity=dup_sim,
        issues=issues,
    )
