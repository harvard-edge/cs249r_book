#!/usr/bin/env python3
"""Backfill the explicit ``provenance: imported`` line on YAMLs that lack it.

Pydantic was already filling ``provenance="imported"`` as a default at
load time, so this is a clarity-only fix: 407 published YAMLs in the
corpus have no explicit ``provenance:`` line, and we want every YAML to
carry the field on disk so it shows up in diffs, in the corpus
manifest, and in `vault edit` output.

CORPUS_HARDENING_PLAN.md Phase 1.

Idempotent — re-running is a no-op once the field is present.

Algorithm:

  1. Walk ``interviews/vault/questions/**/*.yaml``.
  2. Skip files where any top-level line starts with ``provenance:``.
  3. Locate the top-level ``status:`` line. Skip if not exactly
     ``status: published`` — drafts default to ``llm-draft`` and other
     statuses (flagged / deleted / archived) carry their own semantics
     that this mechanical pass should not overwrite.
  4. Insert ``provenance: imported`` on the line immediately below the
     ``status: published`` line, preserving the file's existing
     indentation discipline (top-level keys are at column 0).

Usage:

    # Dry run — print what would change, don't write:
    python3 interviews/vault-cli/scripts/backfill_provenance.py --dry-run

    # Apply:
    python3 interviews/vault-cli/scripts/backfill_provenance.py

    # Apply but cap to N files (smoke testing):
    python3 interviews/vault-cli/scripts/backfill_provenance.py --limit 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
QUESTIONS_DIR = REPO_ROOT / "interviews" / "vault" / "questions"


def has_top_level_provenance(lines: list[str]) -> bool:
    """True iff any line starts with ``provenance:`` at column 0."""
    return any(line.startswith("provenance:") for line in lines)


def find_status_published_line(lines: list[str]) -> int | None:
    """Return the index of the top-level ``status: published`` line, or None.

    Returns None for any other status (draft / flagged / deleted /
    archived) or if no top-level status line exists. The mechanical
    pass only touches published questions; the others have their own
    semantics that should not be overwritten.
    """
    for i, line in enumerate(lines):
        if line.startswith("status:"):
            # Only published — match exact value modulo whitespace.
            stripped = line[len("status:"):].strip()
            if stripped == "published":
                return i
            return None  # found status but not published; bail
    return None


def backfill_one(path: Path, *, dry_run: bool) -> str:
    """Return one of: 'inserted', 'already-present', 'skipped-not-published'."""
    text = path.read_text(encoding="utf-8")
    # Preserve trailing newline behavior across writes.
    had_trailing_newline = text.endswith("\n")
    lines = text.split("\n")
    if had_trailing_newline:
        # split("\n") produces a trailing empty string we'll restore later.
        assert lines[-1] == ""
        lines = lines[:-1]

    if has_top_level_provenance(lines):
        return "already-present"

    status_idx = find_status_published_line(lines)
    if status_idx is None:
        return "skipped-not-published"

    new_lines = (
        lines[: status_idx + 1]
        + ["provenance: imported"]
        + lines[status_idx + 1 :]
    )
    if dry_run:
        return "inserted"

    out = "\n".join(new_lines)
    if had_trailing_newline:
        out += "\n"
    path.write_text(out, encoding="utf-8")
    return "inserted"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of files modified this run")
    args = ap.parse_args()

    targets = sorted(QUESTIONS_DIR.rglob("*.yaml"))
    inserted = already = skipped = 0
    inserted_paths: list[Path] = []

    for path in targets:
        verdict = backfill_one(path, dry_run=args.dry_run)
        if verdict == "inserted":
            inserted += 1
            inserted_paths.append(path)
            if args.limit is not None and inserted >= args.limit:
                # Stop AFTER counting the limit-th insertion.
                break
        elif verdict == "already-present":
            already += 1
        else:  # skipped-not-published
            skipped += 1

    label = "would insert" if args.dry_run else "inserted"
    print(f"\n{label}: {inserted}")
    print(f"already had provenance: {already}")
    print(f"skipped (not status: published): {skipped}")
    print(f"total scanned: {inserted + already + skipped}")

    if args.dry_run and inserted_paths[:5]:
        print("\nfirst 5 candidates:")
        for p in inserted_paths[:5]:
            try:
                rel = p.relative_to(REPO_ROOT)
            except ValueError:
                rel = p
            print(f"  {rel}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
