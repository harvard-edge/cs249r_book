#!/usr/bin/env python3
"""Detect project(s) and contribution type(s) from a merged PR's metadata.

Used by ``all-contributors-auto-credit.yml`` to auto-post the
``@all-contributors please add @author for <types> in <projects>`` comment
when a PR is merged, so maintainers don't have to type it manually.

Project detection: top-level dir of changed files → project key (via
``projects.json``). Multi-project PRs return all detected projects.

Type detection: conventional-commit prefix in PR title (``fix:``/``feat:``/
``docs:``/...) gives the primary type, then the file mix layers on
secondary types (``bug`` PRs touching code → ``bug, code``; touching only
qmd/md → ``bug, doc``; touching only scss/css → ``bug, design``).

Output is written as JSON to stdout:

    {
      "projects":   ["kits", ...],          # may be empty
      "types":      ["bug", "code", ...],   # may be empty
      "confidence": "high" | "low"          # low → workflow asks, doesn't auto-post
    }

Confidence is low when project or type detection failed entirely.
Multi-project / multi-type results are still high confidence — the existing
``all-contributors-add.yml`` workflow handles those natively.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Reuse the single source of truth for projects
sys.path.insert(0, str(Path(__file__).parent))
from projects import projects as _projects_list  # noqa: E402

# Conventional-commit prefix → primary contribution type(s)
TITLE_PREFIX_TYPES = {
    "fix":      ["bug"],
    "feat":     ["code"],
    "docs":     ["doc"],
    "doc":      ["doc"],
    "test":     ["test"],
    "style":    ["design"],
    "chore":    ["code"],
    "refactor": ["code"],
    "perf":     ["code"],
}

# File extension → contribution type
EXT_TO_TYPE = {
    # code
    ".py":   "code",
    ".ts":   "code",
    ".tsx":  "code",
    ".js":   "code",
    ".jsx":  "code",
    ".sh":   "code",
    ".yml":  "code",
    ".yaml": "code",
    ".toml": "code",
    ".json": "code",
    # docs (textbook content lives in .qmd)
    ".qmd": "doc",
    ".md":  "doc",
    ".txt": "doc",
    ".rst": "doc",
    ".ipynb": "doc",
    # visual / styling
    ".scss": "design",
    ".sass": "design",
    ".css":  "design",
    ".svg":  "design",
}


def _project_dir_to_key() -> dict[str, str]:
    """Map every project's on-disk dir to its canonical key."""
    return {p["dir"]: p["key"] for p in _projects_list()}


def detect_projects(files: list[str]) -> list[str]:
    """Return list of project keys touched by the PR's changed files."""
    dir_to_key = _project_dir_to_key()
    found = []
    seen = set()
    for f in files:
        top = f.split("/", 1)[0]
        key = dir_to_key.get(top)
        if key and key not in seen:
            found.append(key)
            seen.add(key)
    return found


def _title_prefix_types(title: str) -> list[str]:
    """Extract types from a conventional-commit title prefix."""
    m = re.match(r"^\s*(\w+)(?:\([^)]*\))?\s*[:!]", title.lower())
    if not m:
        return []
    return TITLE_PREFIX_TYPES.get(m.group(1), [])


def _file_types(files: list[str]) -> set[str]:
    """Infer contribution types from the file mix."""
    types: set[str] = set()
    for f in files:
        lower = f.lower()
        if "/test" in lower or lower.startswith("test") or "/tests/" in lower:
            types.add("test")
            continue
        suffix = Path(f).suffix.lower()
        t = EXT_TO_TYPE.get(suffix)
        if t:
            types.add(t)
    return types


def detect_types(title: str, files: list[str]) -> list[str]:
    """Return contribution types from title prefix + file mix.

    A ``fix:`` PR is always at least ``bug``; what *kind* of fix (code, doc,
    design, test) comes from the file mix. A ``feat:`` PR touching only
    .qmd files becomes ``code, doc`` because new feature + docs.
    """
    types: set[str] = set(_title_prefix_types(title))
    file_types = _file_types(files)

    if "bug" in types:
        # bug fix — layer on what kind of fix it was
        types |= file_types
    elif types:
        # feat:/docs:/etc. — augment with file-type evidence
        types |= file_types
    else:
        # no recognized title prefix — fall back to file types alone
        types |= file_types

    return sorted(types)


def main() -> None:
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    title = data.get("title", "")
    files = data.get("files", [])

    projects_found = detect_projects(files)
    types_found = detect_types(title, files)

    confidence = "high"
    if not projects_found or not types_found:
        confidence = "low"

    json.dump({
        "projects": projects_found,
        "types": types_found,
        "confidence": confidence,
    }, sys.stdout)


if __name__ == "__main__":
    main()
