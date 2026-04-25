"""Persistent accept-list for editorially-verified scanner false positives.

Pass 16 — Item A (the highest-leverage Pass 16 change).

Context
-------
The Pass 15 h3_titlecase scanner's multi-cap-token heuristic over-flags any
heading that contains two or more capitalized tokens, even when the capitals
are proper nouns, acronyms, named principles (Amdahl's Law), legislation
(EU AI Act), D·A·M / C³ taxonomy axis labels, or the first word after a
colon (CMS 8.158). Pass 15 manually walked all 626 vol1 + 747 vol2 hits,
fixed the real violations, and left behind 22 vol1 + 53 vol2 entries that
are editorially correct per book-prose-merged.md §10.9. Without a durable
mechanism, those 75 known-good hits re-appear as "open" in every future
scan, drowning the signal from any new category the scanner gains.

This module provides that durable mechanism. A JSON file next to the
scanner holds the accepted-FP entries; `apply_accept_list` is called at
the end of `scan()` and flips matching ledger issues from `open` to
`accepted`, recording the rule tag in `Issue.protected_context`.

Matching contract
-----------------
The match key is (category, file_relative_path, before_exact). `before` is
the raw source line including the `###` prefix and any trailing
`{#sec-...}` or `{.unnumbered}` attributes — so if the heading text OR its
slug is edited, the accept-list entry stops matching and the issue
correctly returns to `open` for re-review. `line` is stored as a hint for
humans but is not part of the match key, which makes the accept-list
immune to line drift from edits elsewhere in the file.

Stale detection
---------------
Any accept-list entry with zero matches in the current scan is reported
as `stale` — a warning, not an error. Stale entries are the signal that a
previously-accepted heading has been edited, so the entry can be removed.

Schema
------
See `accepted_fps.json`. Top-level is a JSON object:

    {
      "schema_version": 1,
      "description": "...",
      "entries": [
        {
          "category": "h3-titlecase",
          "file": "book/quarto/contents/vol1/.../foo.qmd",  # repo-relative
          "line": 478,                                       # hint only
          "before": "### AMD Instinct MI300X {.unnumbered}",
          "rule": "§10.9-proper-noun-hardware",
          "accepted_in_pass": 15,
          "accepted_date": "2026-04-08"
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from audit.ledger import Issue, Ledger, STATUS_ACCEPTED

SCHEMA_VERSION = 1

# Default accept-list path — next to scan.py in book/tools/audit/.
DEFAULT_ACCEPT_LIST = Path(__file__).resolve().parent / "accepted_fps.json"
# Notation-specific accept-list (kept separate to avoid mixing categories).
DEFAULT_NOTATION_ACCEPT_LIST = (
    Path(__file__).resolve().parent / "accepted_fps_notation.json"
)


@dataclass(frozen=True)
class MatchKey:
    """Exact-match key for accept-list lookup.

    Three fields fully identify an accepted FP: the rule category it came
    from, the repo-relative path of the source file, and the exact raw
    source line (`before`). Line number is intentionally NOT part of the
    key — lines drift when edits happen above, but `before` does not.
    """

    category: str
    file: str
    before: str


@dataclass
class AcceptEntry:
    """A single accept-list entry loaded from disk."""

    category: str
    file: str          # repo-relative, forward slashes
    line: int          # hint only, not part of match key
    before: str
    rule: str          # e.g. "§10.9-named-principle"
    accepted_in_pass: int = 0
    accepted_date: str = ""

    @property
    def key(self) -> MatchKey:
        return MatchKey(self.category, self.file, self.before)


@dataclass
class ApplyResult:
    """Outcome of applying the accept-list to a ledger."""

    total_entries: int
    matched: int
    stale: list[AcceptEntry]


# ── Load ────────────────────────────────────────────────────────────────────


def load_accept_list(path: Path) -> list[AcceptEntry]:
    """Load and validate an accept-list JSON file.

    Missing file is NOT an error — an empty list is returned and the
    scanner behaves exactly as it did before Pass 16. A corrupt file IS
    an error and raises; we do not silently silence-failed editorial
    policy.
    """
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"Accept-list file is not valid JSON: {path}\n  {e}"
        )

    if not isinstance(data, dict) or "entries" not in data:
        raise SystemExit(
            f"Accept-list file missing required 'entries' key: {path}"
        )

    version = data.get("schema_version", 0)
    if version != SCHEMA_VERSION:
        raise SystemExit(
            f"Accept-list schema_version={version} but this scanner "
            f"expects {SCHEMA_VERSION}: {path}"
        )

    entries: list[AcceptEntry] = []
    for i, raw in enumerate(data["entries"]):
        try:
            entries.append(
                AcceptEntry(
                    category=raw["category"],
                    file=raw["file"],
                    line=int(raw.get("line", 0)),
                    before=raw["before"],
                    rule=raw["rule"],
                    accepted_in_pass=int(raw.get("accepted_in_pass", 0)),
                    accepted_date=raw.get("accepted_date", ""),
                )
            )
        except (KeyError, ValueError, TypeError) as e:
            raise SystemExit(
                f"Accept-list entry {i} is malformed in {path}: {e}"
            )

    return entries


# ── Apply ───────────────────────────────────────────────────────────────────


def _repo_relative(absolute_path: str, repo_root: Path) -> str:
    """Convert an absolute ledger file path to a repo-relative POSIX string.

    Scanner-produced issues carry absolute paths. Accept-list entries use
    repo-relative paths so the file can travel with the repo. Matching
    happens in repo-relative space.
    """
    try:
        rel = Path(absolute_path).resolve().relative_to(repo_root.resolve())
    except ValueError:
        # Path is outside the repo — use the basename-anchored fallback so
        # we never raise from a normal code path.
        return absolute_path
    return rel.as_posix()


def apply_accept_list(
    ledger: Ledger,
    entries: list[AcceptEntry],
    repo_root: Path,
    scanned_files: Optional[set[str]] = None,
) -> ApplyResult:
    """Flip accept-list-matched issues from `open` to `accepted` in place.

    Every issue in the ledger is checked against the accept-list's match
    index. When an issue's (category, rel_path, before) matches an entry,
    the issue's status becomes `accepted` and `protected_context` is set
    to the rule tag from the entry (e.g. '§10.9-named-principle').

    Non-matching issues are untouched. Previously-accepted issues (from,
    e.g., inline-span protection) are also untouched — this pass only
    flips `open` → `accepted`.

    `scanned_files` is an optional set of repo-relative file paths that
    were actually walked by the scanner. When provided, stale-entry
    detection is scope-aware: an accept-list entry is only counted stale
    if its file WAS in the scanned set but no issue matched. Entries
    pointing to files outside the scope (e.g. vol2 entries during a vol1
    scan) are silently excluded from stale reporting. When omitted, all
    unmatched entries are considered stale (the original behavior).

    Returns an ApplyResult with counts and a stale-entry list.
    """
    if not entries:
        return ApplyResult(total_entries=0, matched=0, stale=[])

    # Build the match index once: MatchKey -> AcceptEntry
    index: dict[MatchKey, AcceptEntry] = {}
    for entry in entries:
        index[entry.key] = entry

    # Track which entries got matched, so we can report stale ones at the end.
    matched_keys: set[MatchKey] = set()
    matched_count = 0

    for issue in ledger.issues:
        # Don't re-flip things that are already out of the open bucket.
        if issue.status != "open":
            continue

        key = MatchKey(
            category=issue.category,
            file=_repo_relative(issue.file, repo_root),
            before=issue.before,
        )
        entry = index.get(key)
        if entry is None:
            continue

        issue.status = STATUS_ACCEPTED
        issue.protected_context = entry.rule
        matched_keys.add(key)
        matched_count += 1

    stale: list[AcceptEntry] = []
    for entry in entries:
        if entry.key in matched_keys:
            continue
        # Scope-aware stale filter: if a scanned-file set was provided and
        # this entry points outside it, the entry isn't stale — it's just
        # out of scope for this scan. Only entries whose file WAS walked
        # but produced no matching issue are genuinely stale.
        if scanned_files is not None and entry.file not in scanned_files:
            continue
        stale.append(entry)

    return ApplyResult(
        total_entries=len(entries),
        matched=matched_count,
        stale=stale,
    )


# ── Reporting helper (called by scan.py) ────────────────────────────────────


def format_report(result: ApplyResult) -> str:
    """One-line summary line for the scan output."""
    return (
        f"Accept-list: {result.total_entries} entries, "
        f"{result.matched} matched, {len(result.stale)} stale"
    )


def format_stale_warnings(result: ApplyResult, max_show: int = 10) -> list[str]:
    """Produce human-readable stale-entry warnings.

    A stale entry is an accepted FP that no longer matches any issue in
    the current scan — usually because the heading was edited. This is
    not an error: the accept-list file just needs the entry removed.
    """
    if not result.stale:
        return []
    lines = [
        f"WARNING: {len(result.stale)} accept-list entries did not match "
        f"any scanned issue (likely edited headings):"
    ]
    for entry in result.stale[:max_show]:
        lines.append(
            f"  - [{entry.category}] {entry.file}:{entry.line} "
            f"{entry.before[:70]!r}"
        )
    if len(result.stale) > max_show:
        lines.append(f"  ... and {len(result.stale) - max_show} more")
    lines.append(
        "Remove stale entries from book/tools/audit/accepted_fps.json "
        "if the corresponding headings have been intentionally edited."
    )
    return lines
