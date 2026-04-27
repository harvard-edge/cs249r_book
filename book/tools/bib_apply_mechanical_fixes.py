#!/usr/bin/env python3
"""Apply safe §5 mechanical fixes to .bib files (in-place, idempotent where possible).

Fixes (no web lookups, no title-case guessing):
  - `doi` — strip https://doi.org/ and similar prefixes
  - `title` — remove a trailing `.` (skip common abbrev: Inc., U.S., etc. & ``...``)
  - `pages` — remove leading p./pp.; use ``--`` for a single page-range hyphen
  - `journal` — expand known JOURNAL_ABBREV_PATTERNS from ``bib_lint`` (no Phys-Rev series)

Re-emits with ``bib_lint.format_entry`` so address/organization stay dropped per §5.
After running, use pre-commit's bibtex-tidy, then ``python3 book/tools/bib_lint.py --all --check``."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# `book/` on path for `from tools.bib_lint import ...`
_REPO = Path(__file__).resolve().parents[2]
_BOOK = _REPO / "book"
if str(_BOOK) not in sys.path:
    sys.path.insert(0, str(_BOOK))

from tools.bib_lint import (  # noqa: E402
    JOURNAL_ABBREV_PATTERNS,
    Field,
    format_entry,
    parse_bib,
)

# Do not auto-replace series journals (letter matters)
def _abbrev_pairs() -> list[tuple[re.Pattern[str], str]]:
    out: list[tuple[re.Pattern[str], str]] = []
    for pat, rep in JOURNAL_ABBREV_PATTERNS:
        if "Phys" in pat and r"[A-Z]" in pat:
            continue
        out.append((re.compile(pat), rep))
    return out

_ABBREV = _abbrev_pairs()

_TITLE_TRAIL_ABBREV = re.compile(
    r"(?i)(?:^|\s)(inc|ltd|corp|co|etc|jr|sr|e\.g|i\.e|u\.s\.a?|u\.k|al|st|ave|"
    r"fig|vol|no|ed|ph\.?d|dr|m\.d|b\.?v|n\.?v|s\.a)\s*\.\s*$"
)
_TITLE_U_S = re.compile(r"\.[A-Z]\.$")  # e.g. "U.S." at end


def _fix_doi(v: str) -> str:
    s = (v or "").strip()
    for p in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if s[: len(p)].lower() == p.lower():
            s = s[len(p) :]
            break
    return s


def _fix_title(v: str) -> str:
    t = (v or "").rstrip()
    if not t.endswith(".") or t.endswith("..."):
        return v
    if _TITLE_TRAIL_ABBREV.search(t) or _TITLE_U_S.search(t):
        return v
    return t[:-1]


def _fix_pages(v: str) -> str:
    s = (v or "").strip()
    s = re.sub(r"^(?:p|pp)\.\s*", "", s, flags=re.IGNORECASE)
    if "--" not in s and "-" in s and s.count("-") == 1 and not s.startswith("-"):
        s = s.replace("-", "--", 1)
    return s


def _fix_journal(v: str) -> str:
    s = v or ""
    for rx, rep in _ABBREV:
        if rx.search(s):
            s = rx.sub(rep, s)
    return s


def _fix_year_field(f: Field) -> None:
    if f.name.lower() != "year":
        return
    t = f.value.strip()
    if re.match(r"^\d{4}\s*$", t):
        # Keep braces (`year = {2024}`) to match the downstream
        # `bibtex-tidy --curly` pre-commit hook output and avoid a
        # format-cycle conflict between the two hooks.
        f.value = t.strip()
        f.quote_style = "{"


def _apply_to_entry_fields(entry) -> int:
    n = 0
    for f in entry.fields:
        on = f.name.lower()
        old = f.value
        if on == "doi":
            f.value = _fix_doi(f.value)
        elif on == "title":
            f.value = _fix_title(f.value)
        elif on == "pages":
            f.value = _fix_pages(f.value)
        elif on == "journal":
            f.value = _fix_journal(f.value)
        if on == "year":
            _fix_year_field(f)
        if f.value != old:
            n += 1
    return n


def _emit(text: str) -> str:
    entries, pre = parse_bib(text)
    for e in entries:
        _apply_to_entry_fields(e)
    out: list[str] = []
    if pre and pre[0].strip():
        out.append(pre[0].rstrip() + "\n\n")
    for e in entries:
        # align_col=0 produces 'name = {value}' (single space) which matches
        # bibtex-tidy's output and avoids the format-cycle conflict between
        # this hook and the downstream `bibtex-tidy` pre-commit hook.
        out.append(format_entry(e, align_col=0))
        out.append("\n\n")
    return "".join(out).rstrip() + "\n"


def _find_bib_files(explicit: list[Path] | None) -> list[Path]:
    if explicit:
        return [p.resolve() for p in explicit if p.suffix == ".bib"]
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "*.bib"], cwd=_REPO, text=False
    )
    paths: list[Path] = []
    for chunk in raw.split(b"\0"):
        if not chunk:
            continue
        paths.append(_REPO / chunk.decode("utf-8", errors="replace"))
    return sorted({p for p in paths if p.is_file()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "files",
        nargs="*",
        type=Path,
        help=".bib files (default: all tracked *.bib in repo root)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write",
    )
    ap.add_argument(
        "--pre-commit",
        action="store_true",
        help="For .pre-commit-config: exit 1 if any file was modified (re-stage and retry)",
    )
    args = ap.parse_args()
    if args.pre_commit and not args.files:
        # Invoked as `... --pre-commit` with no paths — nothing to do (avoid full-repo run).
        return 0
    targets = _find_bib_files(
        [Path(f) for f in args.files] if args.files else None
    )
    if not targets:
        print("No .bib files to process", file=sys.stderr)
        return 1

    changed = 0
    for path in targets:
        try:
            old = path.read_text(encoding="utf-8")
        except OSError as e:
            print(f"SKIP {path}: {e}", file=sys.stderr)
            continue
        new = _emit(old)
        try:
            rel = path.relative_to(_REPO)
        except ValueError:
            rel = path
        if new != old:
            if not args.dry_run:
                path.write_text(new, encoding="utf-8")
            print(f"{'(dry-run) ' if args.dry_run else ''}{rel}  (updated)")
            changed += 1
        else:
            print(f"{rel}  (no changes)")

    print(f"Done: {changed} / {len(targets)} files with edits")
    if args.pre_commit and changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
