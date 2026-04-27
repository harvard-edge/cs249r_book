#!/usr/bin/env python3
"""Validate a quiz JSON file against the canonical spec and the chapter's anchors.

Per ``.claude/docs/shared/quiz-generation.md`` §1 and §11: one quiz per ``##``
section only; material spans the whole section including its ``###``
subsections but ``###`` anchors are never valid ``section_id`` values.

Run this after every generator pass, and again before commit. It
surfaces two classes of issues:

* **errors** — malformed file, section_id missing or pointing at a
  ``###`` anchor, metadata counts disagree with actual entry counts.
  Blocks merge.
* **warnings** — merge-safe but flagged for human review: question
  count outside the 4–6 window (full) or 2–3 (tier 2), MCQ choice
  count outside 3–5, MCQ answer with a letter-reference pattern
  (``Option [A-D]``, ``Choice [A-D]``, ``Answer [A-D]``, ``([A-D])``)
  that the anti-shuffle-bug rule in spec §10 disallows.

Usage
-----
    python3 validate_quiz_json.py <chapter>_quizzes.json <chapter>.qmd

Exits 1 if any errors are found, 0 otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REQUIRED_QUESTION_FIELDS = ("question_type", "question", "answer", "learning_objective")
VALID_QUESTION_TYPES = {"MCQ", "SHORT", "TF", "FILL", "ORDER"}

# Heading match — captures the hash count separately so we can tell ## vs. ### apart.
HEADING_RE = re.compile(r"^(?P<hashes>#{2,6})\s+.+?\{#(?P<anchor>[^\s}]+)[^}]*\}\s*$")

# Anti-shuffle-bug patterns (spec §10 — any of these in an MCQ answer is a warning).
LETTER_REFS = (
    re.compile(r"\bOption [A-D]\b", re.IGNORECASE),
    re.compile(r"\bChoice [A-D]\b", re.IGNORECASE),
    re.compile(r"\bAnswer [A-D]\b", re.IGNORECASE),
    re.compile(r"\([A-D]\)"),
)


def chapter_anchor_levels(qmd_path: Path) -> dict[str, str]:
    """Map each chapter anchor id to its heading level ("section" for ##,
    "subsection" for ###, etc.).
    """
    levels: dict[str, str] = {}
    for line in qmd_path.read_text(encoding="utf-8").splitlines():
        m = HEADING_RE.match(line)
        if not m:
            continue
        hashes = m.group("hashes")
        anchor = f"#{m.group('anchor')}"
        if len(hashes) == 2:
            levels[anchor] = "section"
        elif len(hashes) == 3:
            levels[anchor] = "subsection"
        else:
            levels[anchor] = "deeper"
    return levels


def validate(json_path: Path, qmd_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"JSON parse error in {json_path.name}: {e}"], []

    # Chapter anchors
    levels = chapter_anchor_levels(qmd_path)
    section_anchors = {a for a, lvl in levels.items() if lvl == "section"}
    if not section_anchors:
        errors.append(f"no ## anchors with {{#sec-...}} found in {qmd_path.name}")
        return errors, warnings

    # Metadata sanity
    meta = data.get("metadata", {})
    if meta.get("schema_version") != 2:
        warnings.append("metadata.schema_version != 2")
    if not meta.get("source_file"):
        warnings.append("metadata.source_file is missing")

    # Shape
    sections = data.get("sections")
    if not isinstance(sections, list):
        errors.append("top-level 'sections' must be a list")
        return errors, warnings

    # Declared vs. actual counts
    actual_total = len(sections)
    actual_with = sum(
        1 for s in sections if (s.get("quiz_data") or {}).get("quiz_needed")
    )
    actual_without = actual_total - actual_with

    declared_total = meta.get("total_sections")
    declared_with = meta.get("sections_with_quizzes")
    declared_without = meta.get("sections_without_quizzes")
    if declared_total is not None and declared_total != actual_total:
        errors.append(
            f"metadata.total_sections={declared_total} but {actual_total} entries found"
        )
    if declared_with is not None and declared_with != actual_with:
        errors.append(
            f"metadata.sections_with_quizzes={declared_with} but {actual_with} entries have quiz_needed=true"
        )
    if declared_without is not None and declared_without != actual_without:
        errors.append(
            f"metadata.sections_without_quizzes={declared_without} but {actual_without} entries have quiz_needed=false"
        )

    # Per-entry validation
    seen_ids: set[str] = set()
    for i, sec in enumerate(sections):
        ctx = f"sections[{i}] id={sec.get('section_id', '?')}"
        sid = sec.get("section_id")
        if not sid:
            errors.append(f"{ctx}: missing section_id")
            continue
        if sid in seen_ids:
            errors.append(f"{ctx}: duplicate section_id")
        seen_ids.add(sid)
        anchor_level = levels.get(sid)
        if anchor_level is None:
            errors.append(f"{ctx}: section_id not found in chapter anchors")
            continue
        if anchor_level != "section":
            errors.append(
                f"{ctx}: section_id points to a `{anchor_level}` (###) anchor; "
                "per spec §1 only `##` anchors are quiz candidates"
            )
            continue

        # Questions
        qd = sec.get("quiz_data") or {}
        if not qd.get("quiz_needed"):
            continue
        questions = qd.get("questions") or []
        if not isinstance(questions, list) or not questions:
            errors.append(f"{ctx}: quiz_needed=true but no questions")
            continue

        # Count windows: tier 1 (full) = 4–6; tier 2 (minimal) = 2–3.
        # Validator allows 2–6 as acceptable, flags anything outside that.
        n = len(questions)
        if not 2 <= n <= 6:
            warnings.append(
                f"{ctx}: {n} questions (expected 2–6 — 4–6 full or 2–3 minimal)"
            )

        for j, q in enumerate(questions):
            qctx = f"{ctx} questions[{j}]"
            for f in REQUIRED_QUESTION_FIELDS:
                if f not in q:
                    errors.append(f"{qctx}: missing '{f}'")
            qtype = q.get("question_type")
            if qtype not in VALID_QUESTION_TYPES:
                errors.append(f"{qctx}: invalid question_type '{qtype}'")
            if qtype == "MCQ":
                choices = q.get("choices") or []
                if not 3 <= len(choices) <= 5:
                    warnings.append(
                        f"{qctx}: MCQ has {len(choices)} choices (expected 3–5)"
                    )
                # Anti-shuffle-bug: flag letter-reference patterns in MCQ answers
                ans = q.get("answer", "")
                for pat in LETTER_REFS:
                    hit = pat.search(ans)
                    if hit:
                        warnings.append(
                            f"{qctx}: MCQ answer contains letter-reference '{hit.group(0)}' "
                            "(spec §5/§10 requires content-based distractor references)"
                        )
                        break

    return errors, warnings


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: validate_quiz_json.py <chapter_quizzes.json> <chapter.qmd>",
            file=sys.stderr,
        )
        return 2
    json_path, qmd_path = Path(argv[1]), Path(argv[2])
    if not json_path.is_file():
        print(f"error: {json_path} does not exist", file=sys.stderr)
        return 1
    if not qmd_path.is_file():
        print(f"error: {qmd_path} does not exist", file=sys.stderr)
        return 1

    errors, warnings = validate(json_path, qmd_path)
    for e in errors:
        print(f"ERROR: {e}")
    for w in warnings:
        print(f"WARN:  {w}")
    if not errors and not warnings:
        print(f"OK: {json_path.name} passes schema + anchor validation")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
