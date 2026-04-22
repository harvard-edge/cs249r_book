#!/usr/bin/env python3
"""Check that every footnote definition starts with a capital letter.

MIT Press style requires that terms like "computer engineering" remain
lowercase in running prose (e.g., "computer engineering is a discipline").
A global search/replace to enforce that rule can accidentally lowercase the
*first letter of a footnote*, which must still be a sentence-case capital.

This script scans every `.qmd` file under `book/quarto/contents/` for
footnote *definitions* of the form:

    [^fn-id]: **Term**: Body of the footnote.
    [^fn-id]: Body of the footnote.

It flags definitions where the first letter character (after skipping an
optional leading `**` bold opener and whitespace) is lowercase.

Footnotes that begin with a non-letter (citations, numbers, math,
URLs) are not flagged — they have no first-letter to capitalize.

Exit status: 1 if any violations are found, 0 otherwise. Intended for use
as a pre-commit / CI binder check.

Usage:
    python3 book/tools/scripts/mit_press/check_footnote_caps.py
    python3 book/tools/scripts/mit_press/check_footnote_caps.py --fix
    python3 book/tools/scripts/mit_press/check_footnote_caps.py <path> [<path> ...]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

# A footnote definition must start at column 0 and match `[^id]: `.
# The id may contain letters, digits, hyphens, underscores, colons, or dots.
FOOTNOTE_DEF = re.compile(r"^\[\^([A-Za-z0-9_:.\-]+)\]:\s+(.*)$")
FOOTNOTE_DEF_PREFIX = re.compile(r"^\[\^[A-Za-z0-9_:.\-]+\]:\s+")

# After the `]: `, peel off an optional leading `**` (bold term opener) and
# any whitespace; the next character decides the check.
BOLD_OPENER = re.compile(r"^\*\*\s*")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_SCAN_ROOT = REPO_ROOT / "book" / "quarto" / "contents"
DEFAULT_ALLOWLIST = pathlib.Path(__file__).with_name("footnote_caps_allowlist.txt")


def load_allowlist(path: pathlib.Path) -> set[str]:
    """Parse an allowlist file; return the set of footnote ids to skip.

    One id per line (without the surrounding `[^...]`). Content after `#` is
    a comment and is stripped. Blank lines are ignored.
    """
    if not path.exists():
        return set()
    ids: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


@dataclass
class Violation:
    path: pathlib.Path
    line_no: int
    raw_line: str
    prefix: str   # the `[^id]: ` part, reproduced verbatim for --fix
    body: str     # the remainder of the line after the prefix
    first_char: str


def find_qmd_files(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """Return every `.qmd` file reachable from the given paths."""
    out: list[pathlib.Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".qmd":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.rglob("*.qmd")))
    return out


def first_meaningful_char(body: str) -> tuple[str, int]:
    """Return (char, offset-into-body) of the first letter to audit.

    Strips an optional leading `**` bold opener. Returns ("", -1) if no
    letter is present in the body at all.
    """
    stripped = body
    offset = 0
    m = BOLD_OPENER.match(stripped)
    if m:
        stripped = stripped[m.end():]
        offset += m.end()

    for i, ch in enumerate(stripped):
        if ch.isalpha():
            return ch, offset + i
        # Allow whitespace drift; anything non-letter, non-space means the
        # footnote doesn't start with a word and we don't flag it.
        if not ch.isspace():
            return "", -1
    return "", -1


def scan_file(path: pathlib.Path, allowlist: set[str]) -> list[Violation]:
    """Return every lowercase-first-letter footnote in `path`.

    Footnote ids listed in `allowlist` are skipped.
    """
    violations: list[Violation] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return violations

    for line_no, line in enumerate(text.splitlines(), start=1):
        m = FOOTNOTE_DEF.match(line)
        if not m:
            continue
        fn_id, body = m.group(1), m.group(2)
        if fn_id in allowlist:
            continue
        prefix_match = FOOTNOTE_DEF_PREFIX.match(line)
        assert prefix_match is not None  # same line, same regex shape
        prefix = prefix_match.group(0)
        ch, _ = first_meaningful_char(body)
        if ch and ch.islower():
            violations.append(
                Violation(
                    path=path,
                    line_no=line_no,
                    raw_line=line,
                    prefix=prefix,
                    body=body,
                    first_char=ch,
                )
            )
    return violations


def apply_fix(v: Violation) -> str:
    """Return the corrected line for a single violation.

    Uppercases the first letter while preserving everything else verbatim.
    """
    body = v.body
    m = BOLD_OPENER.match(body)
    head = ""
    rest = body
    if m:
        head = body[: m.end()]
        rest = body[m.end():]
    # Find the first alpha character in `rest` and uppercase it in place.
    for i, ch in enumerate(rest):
        if ch.isalpha():
            rest = rest[:i] + ch.upper() + rest[i + 1:]
            break
    return v.prefix + head + rest


def fix_files(violations: list[Violation]) -> None:
    """Rewrite each affected file, applying all fixes within it."""
    by_file: dict[pathlib.Path, list[Violation]] = {}
    for v in violations:
        by_file.setdefault(v.path, []).append(v)

    for path, vs in by_file.items():
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for v in vs:
            original = lines[v.line_no - 1]
            # Preserve any trailing newline the file had on that line.
            newline = ""
            stripped = original
            if stripped.endswith("\r\n"):
                newline = "\r\n"
                stripped = stripped[:-2]
            elif stripped.endswith("\n"):
                newline = "\n"
                stripped = stripped[:-1]
            assert stripped == v.raw_line, (
                f"File changed under us at {path}:{v.line_no}"
            )
            lines[v.line_no - 1] = apply_fix(v) + newline
        path.write_text("".join(lines), encoding="utf-8")


def format_report(violations: list[Violation]) -> str:
    if not violations:
        return "OK: every footnote starts with a capital letter."
    out = [f"Found {len(violations)} footnote(s) starting with a lowercase letter:", ""]
    for v in violations:
        try:
            rel = v.path.relative_to(REPO_ROOT)
        except ValueError:
            rel = v.path
        snippet = v.raw_line if len(v.raw_line) <= 160 else v.raw_line[:157] + "..."
        out.append(f"  {rel}:{v.line_no}  (first letter: {v.first_char!r})")
        out.append(f"    {snippet}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        help="Files or directories to scan (default: book/quarto/contents).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Uppercase the first letter of each flagged footnote in place.",
    )
    parser.add_argument(
        "--allowlist",
        type=pathlib.Path,
        default=DEFAULT_ALLOWLIST,
        help=(
            "Path to an allowlist of footnote ids whose first letter is "
            "intentionally lowercase (brand names, math notation, SI units)."
        ),
    )
    args = parser.parse_args()

    scan_paths = args.paths or [DEFAULT_SCAN_ROOT]
    qmd_files = find_qmd_files(scan_paths)
    if not qmd_files:
        print("No .qmd files found in the given paths.", file=sys.stderr)
        return 2

    allowlist = load_allowlist(args.allowlist)
    violations: list[Violation] = []
    for f in qmd_files:
        violations.extend(scan_file(f, allowlist))

    print(format_report(violations))

    if args.fix and violations:
        fix_files(violations)
        print(f"\nApplied {len(violations)} fix(es). Re-run without --fix to verify.")
        return 0

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
