#!/usr/bin/env python3
"""Detect project(s) from a merged PR's changed-file paths.

Used by ``all-contributors-auto-credit.yml``. Project detection is
deterministic (top-level dir → ``projects.json`` key) — type detection
is delegated to the LLM step in the workflow because PR titles aren't
a reliable signal on their own.

Reads ``{"files": [...]}`` from argv[1] (or stdin), writes JSON to stdout:

    {
      "projects":   ["kits", ...],   # canonical project keys; may be empty
      "confidence": "high" | "low"   # low when no project detected
    }

Multi-project results are still high confidence — the existing
``all-contributors-add.yml`` workflow handles "in kits, book" natively.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Reuse the single source of truth for projects
sys.path.insert(0, str(Path(__file__).parent))
from projects import projects as _projects_list  # noqa: E402


def _project_dir_to_key() -> dict[str, str]:
    """Map every project's on-disk dir to its canonical key."""
    return {p["dir"]: p["key"] for p in _projects_list()}


def detect_projects(files: list[str]) -> list[str]:
    """Return list of project keys touched by the PR's changed files.

    Order is the order projects were first encountered in the file list,
    so the most-changed area tends to come first.
    """
    dir_to_key = _project_dir_to_key()
    found: list[str] = []
    seen: set[str] = set()
    for f in files:
        top = f.split("/", 1)[0]
        key = dir_to_key.get(top)
        if key and key not in seen:
            found.append(key)
            seen.add(key)
    return found


def main() -> None:
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    files = data.get("files", [])
    projects_found = detect_projects(files)

    json.dump({
        "projects": projects_found,
        "confidence": "high" if projects_found else "low",
    }, sys.stdout)


if __name__ == "__main__":
    main()
