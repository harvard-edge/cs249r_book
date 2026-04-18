#!/usr/bin/env python3
"""Phase A1 — classify all 18,439 vol1 annotations into editorial buckets.

Reads ``~/Desktop/MIT_Press_Feedback/00_overview/data/all_annotations.json``
and tags every entry with one of the buckets below plus a
``needs_locator`` flag indicating whether Phase B has to find the change in
the current QMD source (vs. being verified by the audit scanner or handled
by the typesetter).

Output: ``~/Desktop/MIT_Press_Feedback/16_release_audit/ledgers/vol1-annotations-ground-truth.json``
plus a per-chapter summary table.

This script is read-only with respect to the manuscript.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

INPUT = Path.home() / "Desktop/MIT_Press_Feedback/00_overview/data/all_annotations.json"
OUT_DIR = Path.home() / "Desktop/MIT_Press_Feedback/16_release_audit/ledgers"

BUCKETS = [
    "bulk-emdash",
    "bulk-percent",
    "bulk-capitalization",
    "bulk-abbreviation",
    "bulk-bibliography",
    "bulk-spacing",
    "bulk-slash",
    "bulk-microedit",        # silent caret/strike <=4 chars: PDF artifacts covered by bulk passes
    "alt-text",
    "footnote-query",
    "section-number-query",
    "front-matter",
    "layout-typesetter",
    "au-decision",
    "micro-edit",            # multi-word substantive caret/strike >=5 chars
    "highlight-note",
    "other",
]

# Buckets the Phase B locator must touch (i.e. need to be searched in the
# current QMD source). Bulk rules are verified by the scanner. Layout notes
# are typesetter responsibility. Highlight-only annotations without content
# carry no actionable change. ``bulk-microedit`` is the long tail of single-
# character strikes/inserts which the bulk passes already handle and which
# cannot reasonably be disambiguated against the QMD source one by one.
NEEDS_LOCATOR = {
    "front-matter",
    "au-decision",
    "footnote-query",
    "section-number-query",
    "alt-text",
    "micro-edit",
}

# Threshold below which silent caret/strike payloads are treated as PDF
# parsing artifacts / single-character bulk edits and not individually
# located. (See PASS_16_COMPLETION_REPORT §3 for context: em-dash, percent,
# abbreviation, and capitalization passes already covered the long tail.)
MICROEDIT_MIN_CHARS = 5

FRONT_MATTER_CHAPTERS = {
    "About This Book",
    "Acknowledgements",
    "Author's Note",
    "Cover/Title",
    "Notation and Conventions",
    "Part I: Foundations",
    "Part III: Optimize",
    "Part IV: Deploy",
    "Part II: Workflow",
}

LAYOUT_PATTERNS = [
    r"move (this|the) (to|figure|paragraph)",
    r"next page",
    r"avoid widow",
    r"widowed line",
    r"two columns wider",
    r"close to the bottom of the page",
    r"left[- ]aligned",
    r"comp\b",
    r"set (the )?epigraph",
    r"keep this on the same page",
    r"break here",
]

EMDASH_PATTERNS = [
    r"em[- ]?dash",
    r"close up space around the em",
]

PERCENT_PATTERNS = [
    r"percent",
    r"\bspell out\b.*%",
    r"%.*spell",
]

SLASH_PATTERNS = [
    r"close up.*slash",
    r"\bslash\b",
]

SPACING_PATTERNS = [
    r"add (a )?space (between|after|before)",
    r"close up (the )?space",
    r"insert (a )?space",
    r"space between the (number|date|initials)",
    r"remove (a )?space",
]

ABBREV_PATTERNS = [
    r"spell out",
    r"first use",
    r"abbrev",
    r"\bab\b",
]

CAPITALIZATION_PATTERNS = [
    r"lowercase",
    r"lower-case",
    r"capital(ize|ization)",
    r"sentence (style|case)",
    r"title case",
    r"headline style",
]

BIBLIOGRAPHY_PATTERNS = [
    r"publisher (name|location)",
    r"page number",
    r"please provide (the )?(publisher|page|doi|year|location|volume|issue)",
    r"please verify",
    r"\bdoi\b",
    r"deleted the publisher location",
]

ALT_TEXT_PATTERNS = [
    r"alt text",
    r"alt-text",
    r"alternative text",
    r"figure description",
]

FOOTNOTE_PATTERNS = [
    r"footnote",
    r"sidenote",
    r"\bfn\b",
]

SECTION_NUM_PATTERNS = [
    r"section number",
    r"insert (figure|table|section|chapter) (number|name)",
    r"add the correct section number",
    r"\bcross[- ]ref",
    r"\?\?",
]


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify(entry: dict[str, Any]) -> tuple[str, bool, str]:
    """Return (bucket, needs_locator, rationale)."""
    chapter = entry.get("chapter") or ""
    content = (entry.get("content") or "").strip()
    struck = (entry.get("struck_text") or "").strip()
    insert = (entry.get("insert_text") or "").strip()
    highlighted = (entry.get("highlighted_text") or "").strip()
    atype = entry.get("type") or ""
    content_l = content.lower()

    # 1. Front matter chapters always go to front-matter bucket.
    if chapter in FRONT_MATTER_CHAPTERS:
        return ("front-matter", True, f"chapter={chapter}")

    # Bibliography chapter → bulk-bibliography.
    if "Bibliography" in chapter:
        return ("bulk-bibliography", False, f"chapter={chapter}")

    # 2. Layout / typesetter notes (don't need source edits).
    if matches_any(content_l, LAYOUT_PATTERNS):
        return ("layout-typesetter", False, "layout-note")

    # 3. Specific content-bearing notes — order matters (most specific first).
    if content:
        if matches_any(content_l, ALT_TEXT_PATTERNS):
            return ("alt-text", True, "content mentions alt text")

        if matches_any(content_l, FOOTNOTE_PATTERNS):
            return ("footnote-query", True, "content mentions footnote/sidenote")

        if matches_any(content_l, SECTION_NUM_PATTERNS):
            return ("section-number-query", True, "content mentions section/figure number")

        if matches_any(content_l, BIBLIOGRAPHY_PATTERNS):
            return ("bulk-bibliography", False, "bibliography query")

        if matches_any(content_l, EMDASH_PATTERNS):
            return ("bulk-emdash", False, "em dash rule")

        if matches_any(content_l, PERCENT_PATTERNS):
            return ("bulk-percent", False, "percent rule")

        if matches_any(content_l, SLASH_PATTERNS):
            return ("bulk-slash", False, "slash spacing rule")

        if matches_any(content_l, ABBREV_PATTERNS):
            return ("bulk-abbreviation", False, "abbreviation rule")

        if matches_any(content_l, CAPITALIZATION_PATTERNS):
            return ("bulk-capitalization", False, "capitalization rule")

        if matches_any(content_l, SPACING_PATTERNS):
            return ("bulk-spacing", False, "spacing rule")

        # AU/Comp: catch-all for queries with content but no rule match.
        if re.match(r"^(au|comp)\b", content_l):
            return ("au-decision", True, "AU/Comp query without bucket match")

        # Otherwise treat content-bearing strikeouts/highlights without a known
        # rule as highlight-note (something the editor commented but no bulk
        # rule applies).
        if atype == "Highlight":
            return ("highlight-note", False, "Highlight with comment, no rule match")

    # 4. Silent micro-edits: caret-insertion or strikeout without comment.
    # Most of these are 1-3 character payloads where the editor's annotation
    # box bounded a single letter or punctuation mark; they are de-facto
    # covered by the global bulk passes (em dash, percent, hyphen, abbrev,
    # capitalization) and cannot be uniquely located in the QMD source. We
    # reserve ``micro-edit`` (needs_locator=True) for substantive multi-word
    # edits only.
    if atype == "Caret" and insert:
        payload = insert.strip()
        if len(payload) < MICROEDIT_MIN_CHARS:
            return ("bulk-microedit", False, f"silent caret insert<{MICROEDIT_MIN_CHARS}c: {insert!r}")
        return ("micro-edit", True, f"silent caret insert={insert!r}")

    if atype == "StrikeOut" and struck:
        payload = struck.strip()
        if len(payload) < MICROEDIT_MIN_CHARS:
            return ("bulk-microedit", False, f"silent strike<{MICROEDIT_MIN_CHARS}c: {struck!r}")
        return ("micro-edit", True, f"silent strike struck={struck!r}")

    if atype == "Highlight" and highlighted and not content:
        # Highlight with no content is likely a position marker the editor used
        # for visual orientation; not actionable on its own.
        return ("highlight-note", False, "Highlight without comment")

    # Catch-all
    return ("other", False, f"type={atype} unclassified")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT.open() as f:
        annotations = json.load(f)

    results = []
    bucket_counts: Counter[str] = Counter()
    per_chapter: dict[str, Counter[str]] = defaultdict(Counter)
    needs_locator_count = 0

    for idx, entry in enumerate(annotations):
        bucket, needs_locator, rationale = classify(entry)
        record = {
            "annotation_id": f"v1-{idx:05d}",
            **entry,
            "bucket": bucket,
            "needs_locator": needs_locator,
            "classification_rationale": rationale,
        }
        results.append(record)
        bucket_counts[bucket] += 1
        per_chapter[entry.get("chapter") or "unknown"][bucket] += 1
        if needs_locator:
            needs_locator_count += 1

    out_path = OUT_DIR / "vol1-annotations-ground-truth.json"
    out_path.write_text(json.dumps(results, indent=2))

    summary_path = OUT_DIR / "vol1-annotations-ground-truth-SUMMARY.json"
    summary = {
        "source": str(INPUT),
        "total_annotations": len(annotations),
        "needs_locator_total": needs_locator_count,
        "bucket_totals": dict(sorted(bucket_counts.items(), key=lambda kv: -kv[1])),
        "per_chapter": {
            ch: dict(sorted(c.items(), key=lambda kv: -kv[1]))
            for ch, c in sorted(per_chapter.items())
        },
        "buckets": BUCKETS,
        "needs_locator_buckets": sorted(NEEDS_LOCATOR),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    # Pretty print to stdout for the operator log.
    print(f"Wrote {out_path} ({len(results)} entries)")
    print(f"Wrote {summary_path}")
    print()
    print("Bucket totals:")
    width = max(len(b) for b in bucket_counts)
    for bucket, n in summary["bucket_totals"].items():
        print(f"  {bucket:<{width}} {n:>6}")
    print()
    print(f"needs_locator_total: {needs_locator_count}")


if __name__ == "__main__":
    main()
