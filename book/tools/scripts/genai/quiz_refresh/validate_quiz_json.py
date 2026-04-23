#!/usr/bin/env python3
"""Validate a quiz JSON file against schema v2 and the chapter's anchors.

Run this after every sub-agent completes, and again before merging the
regenerated JSONs into the main build. It surfaces two classes of issues:

* **errors** — the file is malformed or references an anchor that does
  not exist in the chapter, and must not be merged.
* **warnings** — the file is merge-safe but violates a quality guideline
  (question count outside the 4–6 / 2–3 window, MCQ with too few
  choices, etc.). Review but do not block.

Usage
-----
    python3 validate_quiz_json.py <chapter>_quizzes.json <chapter>.qmd

Exits with status 1 if any errors are found, 0 otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REQUIRED_QUESTION_FIELDS = ("question_type", "question", "answer", "learning_objective")
VALID_QUESTION_TYPES = {"MCQ", "SHORT", "TF"}
EXPECTED_COUNTS = {"section": (4, 6), "subsection": (2, 3)}

ANCHOR_RE = re.compile(r"^#{2,3}\s+.+?\{#(?P<anchor>[^\s}]+)[^}]*\}\s*$")


def chapter_anchors(qmd_path: Path) -> set[str]:
    anchors: set[str] = set()
    for line in qmd_path.read_text(encoding="utf-8").splitlines():
        m = ANCHOR_RE.match(line)
        if m:
            anchors.add(f"#{m.group('anchor')}")
    return anchors


def anchor_level(qmd_path: Path, anchor_id: str) -> str | None:
    """Return ``"section"`` or ``"subsection"`` for the given anchor in ``qmd_path``."""
    needle = anchor_id[1:]  # strip leading '#'
    for line in qmd_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(#{2,3})\s+.+?\{#(\S+)", line)
        if m and m.group(2) == needle:
            return "section" if len(m.group(1)) == 2 else "subsection"
    return None


def validate(json_path: Path, qmd_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"JSON parse error in {json_path.name}: {e}"], []

    # Metadata block
    meta = data.get("metadata", {})
    if meta.get("schema_version") != 2:
        warnings.append("metadata.schema_version != 2 (recommended for refreshed files)")
    if not meta.get("source_file"):
        warnings.append("metadata.source_file is missing")

    # Cross-check declared metadata counts against actual entry counts
    sections_list = data.get("sections", []) or []
    actual_sections = sum(1 for s in sections_list if s.get("level") == "section")
    actual_subsections = sum(1 for s in sections_list if s.get("level") == "subsection")
    declared_sections = meta.get("total_sections")
    declared_subsections = meta.get("total_subsections")
    declared_total = meta.get("total_quizzes")
    if declared_sections is not None and declared_sections != actual_sections:
        errors.append(
            f"metadata.total_sections={declared_sections} but {actual_sections} level='section' entries found"
        )
    if declared_subsections is not None and declared_subsections != actual_subsections:
        errors.append(
            f"metadata.total_subsections={declared_subsections} but {actual_subsections} level='subsection' entries found"
        )
    if declared_total is not None and declared_total != len(sections_list):
        errors.append(
            f"metadata.total_quizzes={declared_total} but {len(sections_list)} entries found"
        )

    # Anchors
    anchors = chapter_anchors(qmd_path)
    if not anchors:
        errors.append(f"no ## or ### anchors with {{#sec-…}} found in {qmd_path.name}")
        return errors, warnings

    sections = data.get("sections", [])
    if not isinstance(sections, list):
        errors.append("top-level 'sections' must be a list")
        return errors, warnings

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
        if sid not in anchors:
            errors.append(f"{ctx}: section_id not found in chapter anchors")

        level = sec.get("level")
        if level and level not in EXPECTED_COUNTS:
            errors.append(f"{ctx}: level='{level}' must be 'section' or 'subsection'")
        # Cross-check declared level against actual heading level in the .qmd
        actual_level = anchor_level(qmd_path, sid) if sid in anchors else None
        if level and actual_level and level != actual_level:
            errors.append(
                f"{ctx}: declared level='{level}' but anchor is actually {actual_level}"
            )

        if level == "subsection":
            parent = sec.get("parent_section_id")
            if not parent:
                warnings.append(f"{ctx}: subsection missing parent_section_id")
            elif parent not in anchors:
                errors.append(f"{ctx}: parent_section_id not found in chapter anchors")

        qd = sec.get("quiz_data", {}) or {}
        if qd.get("quiz_needed") is False:
            continue
        questions = qd.get("questions", [])
        if not isinstance(questions, list) or not questions:
            errors.append(f"{ctx}: quiz_needed=true but no questions")
            continue

        # Question count window
        effective_level = level or actual_level or "section"
        lo, hi = EXPECTED_COUNTS.get(effective_level, (1, 99))
        if not lo <= len(questions) <= hi:
            warnings.append(
                f"{ctx}: {len(questions)} questions (expected {lo}–{hi} for {effective_level})"
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

    return errors, warnings


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: validate_quiz_json.py <chapter_quizzes.json> <chapter.qmd>", file=sys.stderr)
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
