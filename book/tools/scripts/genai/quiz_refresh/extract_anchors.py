#!/usr/bin/env python3
"""Extract ``## section`` and ``### subsection`` anchors from a Quarto ``.qmd`` chapter.

Used by the quiz-refresh pipeline to give each sub-agent an authoritative
map of the exact anchors it may target as ``section_id`` / ``parent_section_id``
values in the generated quiz JSON.

Usage
-----
    python3 extract_anchors.py path/to/chapter.qmd

Emits a single JSON object on stdout:

    {
      "source": "path/to/chapter.qmd",
      "anchors": [
        {"level": "section",    "id": "#sec-…", "title": "…", "line": 42},
        {"level": "subsection", "id": "#sec-…", "title": "…", "line": 73,
         "parent_id": "#sec-…"},
        …
      ]
    }

Headings without an explicit ``{#sec-…}`` identifier are skipped: the
Lua filter matches by identifier only, so anonymous headings cannot
receive injected quizzes anyway.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterator

# Matches ``## Title {#sec-...}`` or ``### Title {#sec-...}``. Any trailing
# attribute tokens after the identifier are tolerated but not captured.
ANCHOR_RE = re.compile(
    r"^(?P<hashes>#{2,3})\s+(?P<title>.+?)(?:\s+\{#(?P<anchor>[^\s}]+)[^}]*\})?\s*$"
)


def iter_anchors(qmd_text: str) -> Iterator[dict]:
    """Yield anchor entries in document order.

    ``parent_id`` is attached to subsection entries and points at the most
    recent section-level anchor.
    """
    current_section: str | None = None
    for lineno, line in enumerate(qmd_text.splitlines(), start=1):
        m = ANCHOR_RE.match(line)
        if not m:
            continue
        hashes = m.group("hashes")
        title = m.group("title").strip()
        anchor = m.group("anchor")
        if not anchor:
            # anonymous heading: Quarto will auto-anchor, but the quiz
            # filter cannot target it — skip.
            continue
        level = "section" if len(hashes) == 2 else "subsection"
        entry: dict = {
            "level": level,
            "id": f"#{anchor}",
            "title": title,
            "line": lineno,
        }
        if level == "section":
            current_section = f"#{anchor}"
        else:
            entry["parent_id"] = current_section
        yield entry


def extract(qmd_path: Path) -> dict:
    anchors = list(iter_anchors(qmd_path.read_text(encoding="utf-8")))
    return {"source": str(qmd_path), "anchors": anchors}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: extract_anchors.py <chapter.qmd>", file=sys.stderr)
        return 2
    qmd_path = Path(argv[1])
    if not qmd_path.is_file():
        print(f"error: {qmd_path} does not exist", file=sys.stderr)
        return 1
    print(json.dumps(extract(qmd_path), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
