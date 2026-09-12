"""Mechanical bibliography fixes used by Binder and related book tools."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO

BOOK_DIR = Path(__file__).resolve().parents[2]
if str(BOOK_DIR) not in sys.path:
    sys.path.insert(0, str(BOOK_DIR))

from cli.checks.bib_lint import (  # noqa: E402
    JOURNAL_ABBREV_PATTERNS,
    Field,
    format_entry,
    parse_bib,
)


@dataclass(frozen=True)
class MechanicalFileResult:
    """Outcome for one .bib file; ``error`` is set when the file could not be read."""

    path: Path
    changed: bool
    error: str | None = None


def _project_root(start: Path) -> Path:
    """Resolve the repository root from the repository, the toolchain, or books/."""
    path = Path(start).resolve()
    if (path / "binder" / "binder").is_file():
        return path
    if (path / "binder").is_file() and path.name == "binder":
        return path.parent
    if (path / "config").is_dir() and path.name == "books":
        return path.parent
    try:
        raw = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return Path(raw.strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        return path


def _abbrev_pairs() -> list[tuple[re.Pattern[str], str]]:
    """Compile the lint's journal-abbreviation patterns, except ``Phys`` ones using ``[A-Z]``."""
    out: list[tuple[re.Pattern[str], str]] = []
    for pattern, replacement in JOURNAL_ABBREV_PATTERNS:
        if "Phys" in pattern and r"[A-Z]" in pattern:
            continue
        out.append((re.compile(pattern), replacement))
    return out


_ABBREV = _abbrev_pairs()

_TITLE_TRAIL_ABBREV = re.compile(
    r"(?i)(?:^|\s)(inc|ltd|corp|co|etc|jr|sr|e\.g|i\.e|u\.s\.a?|u\.k|al|st|ave|"
    r"fig|vol|no|ed|ph\.?d|dr|m\.d|b\.?v|n\.?v|s\.a)\s*\.\s*$"
)
_TITLE_U_S = re.compile(r"\.[A-Z]\.$")


def fix_doi(value: str) -> str:
    """Strip whitespace and a leading ``doi.org`` or ``dx.doi.org`` URL prefix from a DOI."""
    text = (value or "").strip()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if text[: len(prefix)].lower() == prefix.lower():
            text = text[len(prefix) :]
            break
    return text


def fix_title(value: str) -> str:
    """Drop one trailing period from a title.

    Returns the original value unchanged when there is no trailing period, the
    title ends in an ellipsis, or the period belongs to a known abbreviation
    (``Inc.``, ``e.g.``, ``U.S.``, and similar).
    """
    title = (value or "").rstrip()
    if not title.endswith(".") or title.endswith("..."):
        return value
    if _TITLE_TRAIL_ABBREV.search(title) or _TITLE_U_S.search(title):
        return value
    return title[:-1]


def fix_pages(value: str) -> str:
    """Remove a ``p.``/``pp.`` prefix and turn a single-hyphen range into a ``--`` range."""
    text = (value or "").strip()
    text = re.sub(r"^(?:p|pp)\.\s*", "", text, flags=re.IGNORECASE)
    if (
        "--" not in text
        and "-" in text
        and text.count("-") == 1
        and not text.startswith("-")
    ):
        text = text.replace("-", "--", 1)
    return text


def fix_journal(value: str) -> str:
    """Apply every matching journal-abbreviation replacement to a journal name."""
    text = value or ""
    for pattern, replacement in _ABBREV:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
    return text


def _fix_year_field(field: Field) -> None:
    """Strip a four-digit ``year`` value in place and switch its quote style to braces."""
    if field.name.lower() != "year":
        return
    text = field.value.strip()
    if re.match(r"^\d{4}\s*$", text):
        field.value = text
        field.quote_style = "{"


def _apply_to_entry_fields(entry) -> int:
    """Apply the DOI, title, pages, journal, and year fixes in place.

    Returns the number of fields whose value changed; a year quote-style change
    alone is not counted.
    """
    changed = 0
    for field in entry.fields:
        name = field.name.lower()
        old_value = field.value
        if name == "doi":
            field.value = fix_doi(field.value)
        elif name == "title":
            field.value = fix_title(field.value)
        elif name == "pages":
            field.value = fix_pages(field.value)
        elif name == "journal":
            field.value = fix_journal(field.value)
        if name == "year":
            _fix_year_field(field)
        if field.value != old_value:
            changed += 1
    return changed


def apply_mechanical_fixes_to_text(text: str) -> str:
    """Return bibliography text with deterministic section 5 fixes applied."""
    entries, preamble = parse_bib(text)
    for entry in entries:
        _apply_to_entry_fields(entry)

    out: list[str] = []
    if preamble and preamble[0].strip():
        # One newline, not two: bibtex-tidy strips a blank line between a
        # leading comment block and the first entry, so emitting one here made
        # the two bib hooks undo each other on every run and no commit touching
        # a bibliography with a header could ever pass.
        out.append(preamble[0].rstrip() + "\n")
    for entry in entries:
        out.append(format_entry(entry, align_col=0))
        out.append("\n\n")
    return "".join(out).rstrip() + "\n"


def apply_mechanical_fixes_to_file(
    path: Path,
    *,
    dry_run: bool = False,
) -> MechanicalFileResult:
    """Apply mechanical bibliography fixes to one file."""
    try:
        old = path.read_text(encoding="utf-8")
    except OSError as exc:
        return MechanicalFileResult(path=path, changed=False, error=str(exc))

    new = apply_mechanical_fixes_to_text(old)
    if new == old:
        return MechanicalFileResult(path=path, changed=False)

    if not dry_run:
        path.write_text(new, encoding="utf-8")
    return MechanicalFileResult(path=path, changed=True)


def _resolve_explicit_file(path: Path, repo: Path) -> Path:
    """Resolve a user-supplied path: absolute as-is, else cwd if it exists, else repo-relative."""
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repo / path).resolve()


def find_bib_files(repo: Path, explicit: Sequence[str | Path] | None = None) -> list[Path]:
    """Return explicit .bib files or every git-tracked .bib in the repo."""
    repo = _project_root(repo)
    if explicit:
        return [
            _resolve_explicit_file(Path(path), repo)
            for path in explicit
            if Path(path).suffix == ".bib"
        ]

    try:
        raw = subprocess.check_output(
            ["git", "ls-files", "-z", "*.bib"],
            cwd=repo,
            text=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(path for path in repo.rglob("*.bib") if path.is_file())

    paths: list[Path] = []
    for chunk in raw.split(b"\0"):
        if chunk:
            paths.append(repo / chunk.decode("utf-8", errors="replace"))
    return sorted({path for path in paths if path.is_file()})


def run_mechanical_fixes(
    repo: Path,
    files: Sequence[str | Path] | None = None,
    *,
    dry_run: bool = False,
    pre_commit: bool = False,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the Binder bibliography mechanical-fix workflow."""
    if pre_commit and not files:
        return 0

    repo = _project_root(repo)
    targets = find_bib_files(repo, files)
    if not targets:
        print("No .bib files to process", file=stderr)
        return 1

    changed = 0
    for target in targets:
        result = apply_mechanical_fixes_to_file(target, dry_run=dry_run)
        try:
            label = result.path.relative_to(repo)
        except ValueError:
            label = result.path
        if result.error:
            print(f"SKIP {label}: {result.error}", file=stderr)
            continue
        if result.changed:
            prefix = "(dry-run) " if dry_run else ""
            print(f"{prefix}{label}  (updated)", file=stdout)
            changed += 1
        else:
            print(f"{label}  (no changes)", file=stdout)

    print(f"Done: {changed} / {len(targets)} files with edits", file=stdout)
    if pre_commit and changed:
        return 1
    return 0
