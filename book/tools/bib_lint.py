#!/usr/bin/env python3
"""BibTeX linter, validator, and formatter for the MLSysBook project.

Enforces the canonical schema and formatting rules documented in
`.claude/rules/book-prose-merged.md` §5 Bibliography Hygiene.

Usage:
    python3 book/tools/bib_lint.py <file.bib> [--check|--fix|--report]
    python3 book/tools/bib_lint.py --all [--check|--fix|--report]

Modes:
    --check   Exit 1 if any violations found; no output rewrites.
              Default when called from pre-commit.
    --fix     Rewrite the file(s) to canonical form: fix field order,
              indentation, quoting, spacing, trailing commas.
    --report  Print a detailed violation report without rewriting.
              Default when called directly.

What it does:
    1. Parses .bib files with a proper state machine (brace counting +
       quote tracking). Handles nested braces in titles correctly.
    2. Validates each entry against §5 schema: required fields per
       entry type, canonical field order, quoting style, author list
       rules, journal spell-out, publisher canonical forms, etc.
    3. Auto-fixes safely-fixable violations: field reordering,
       indentation, quote style, trailing commas, en-dash in pages.
    4. Reports unfixable violations (missing required fields,
       abbreviated journal names, initial-only authors) as warnings
       so the Pass 16+ sweep can address them via agent verification.

What it does NOT do:
    - Invent missing field values. If `publisher` is absent, it is
      reported, not filled.
    - Verify metadata against external sources. That's the job of the
      parallel-agent sweep + (future) Crossref refresh tool.
    - Rewrite or paraphrase content. All field VALUES are preserved
      byte-exact across --fix runs; only FORMATTING changes.

Integration points:
    - Pre-commit hook: called with `--check <modified-bib-files>`.
      Exits non-zero if any .bib file has violations that the hook
      should block.
    - CLI: called directly by engineer or agent for ad-hoc checks.
    - Apply pipeline: parallel-agent sweep calls `apply_fields()` on
      each .bib file after agents return verified metadata. This
      function is the SAFE way to insert new fields into an entry —
      no regex, no brace-counting bugs.

Canonical rule source:
    .claude/rules/book-prose-merged.md §5 "Bibliography Hygiene"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Optional


# ─── Allow-list for pre-existing issues ──────────────────────────────────────

# Location of the baseline file that grandfathers known pre-existing
# violations. The pre-commit hook checks new violations against this
# baseline — violations in the baseline are allowed; anything else
# blocks the commit. The baseline is regenerated via --baseline mode
# and committed to the repo so it's auditable.
ALLOW_LIST_PATH = Path(__file__).parent / "bib_lint_baseline.json"


def load_baseline() -> set[tuple[str, str, str]]:
    """Load the baseline allow-list as a set of (file, key, rule) tuples."""
    if not ALLOW_LIST_PATH.exists():
        return set()
    data = json.loads(ALLOW_LIST_PATH.read_text())
    return {(e["file"], e["key"], e["rule"]) for e in data.get("allowed", [])}


def save_baseline(violations: list[tuple[str, "Violation"]]) -> None:
    """Write the baseline allow-list from a list of (file, violation) pairs."""
    entries = sorted(
        [
            {"file": fp, "key": v.entry_key, "rule": v.rule,
             "message": v.message}
            for fp, v in violations
            if v.severity == "error"
        ],
        key=lambda e: (e["file"], e["key"], e["rule"]),
    )
    ALLOW_LIST_PATH.write_text(
        json.dumps(
            {
                "description": (
                    "bib_lint allow-list: pre-existing violations grandfathered "
                    "at the time of the baseline. New violations NOT in this file "
                    "will block commits. Regenerate via: "
                    "python3 book/tools/bib_lint.py --all --baseline"
                ),
                "generated": "2026-04-08",
                "allowed": entries,
            },
            indent=2,
        )
        + "\n"
    )


# ─── Canonical schema (from §5) ──────────────────────────────────────────────

REQUIRED_FIELDS: dict[str, list[str]] = {
    "inproceedings": ["author", "title", "booktitle", "publisher", "year"],
    "article": ["author", "title", "journal", "year"],
    "book": ["title", "publisher", "year"],  # author OR editor
    "incollection": ["author", "title", "booktitle", "publisher", "year"],
    "techreport": ["author", "title", "institution", "year"],
    "phdthesis": ["author", "title", "school", "year"],
    "mastersthesis": ["author", "title", "school", "year"],
    "misc": ["title"],
}

# Canonical field order per entry type (from §5 template)
CANONICAL_ORDER: dict[str, list[str]] = {
    "inproceedings": [
        "author", "editor", "title", "booktitle", "publisher",
        "year", "pages", "volume", "series", "address", "doi", "url",
        "x-verified", "x-verified-by", "x-verified-source", "x-no-doi",
    ],
    "article": [
        "author", "title", "journal", "volume", "number", "pages",
        "year", "month", "doi", "url",
        "x-verified", "x-verified-by", "x-verified-source", "x-no-doi",
    ],
    "book": [
        "author", "editor", "title", "publisher", "year", "edition",
        "address", "isbn", "doi", "url",
        "x-verified", "x-verified-by", "x-verified-source", "x-no-doi",
    ],
    "incollection": [
        "author", "title", "booktitle", "editor", "publisher",
        "year", "pages", "doi", "url",
        "x-verified", "x-verified-by", "x-verified-source", "x-no-doi",
    ],
    "techreport": [
        "author", "title", "institution", "year", "number", "type",
        "url", "doi",
        "x-verified", "x-verified-by", "x-verified-source", "x-no-doi",
    ],
    "phdthesis": [
        "author", "title", "school", "year", "type", "url", "doi",
        "x-verified", "x-verified-by", "x-verified-source", "x-no-doi",
    ],
    "mastersthesis": [
        "author", "title", "school", "year", "type", "url", "doi",
        "x-verified", "x-verified-by", "x-verified-source", "x-no-doi",
    ],
    "misc": [
        "author", "title", "year", "howpublished", "publisher",
        "institution", "note", "url", "doi",
        "x-verified", "x-verified-by", "x-verified-source", "x-no-doi",
    ],
}

# Fields forbidden by §5 (MIT Press round 1 cleanup)
FORBIDDEN_FIELDS: set[str] = {
    "address",       # publisher locations removed in round 1
    "organization",  # replaced with publisher per §5
}

# Common abbreviated-journal patterns that must be spelled out
JOURNAL_ABBREV_PATTERNS: list[tuple[str, str]] = [
    (r"\bJ\. Mach\. Learn\. Res\.", "Journal of Machine Learning Research"),
    (r"\bCommun\. ACM\b", "Communications of the ACM"),
    (r"\bIEEE Trans\. Pattern Anal\. Mach\. Intell\.",
     "IEEE Transactions on Pattern Analysis and Machine Intelligence"),
    (r"\bNat\. Mach\. Intell\.", "Nature Machine Intelligence"),
    (r"\bNeural Comput\.", "Neural Computation"),
    (r"\bProc\. IEEE\b", "Proceedings of the IEEE"),
    (r"\bACM Comput\. Surv\.", "ACM Computing Surveys"),
]

# Canonical publishers (warn if not in this set — may be OK but flag for review)
CANONICAL_PUBLISHERS: set[str] = {
    "IEEE", "ACM", "ACM/IEEE", "PMLR", "OpenReview.net",
    "Curran Associates Inc.", "Neural Information Processing Systems Foundation",
    "Association for Computational Linguistics", "USENIX Association",
    "mlsys.org", "CIDR", "AAAI Press", "Springer", "Elsevier",
    "Nature Publishing Group", "AAAS", "arXiv", "Schloss Dagstuhl",
    "MIT Press", "Cambridge University Press", "Oxford University Press",
    "Morgan Kaufmann", "Wiley", "O'Reilly Media", "SIAM",
    "The MIT Press", "Dartmouth College", "Microsoft Research",
    "Carnegie Mellon University", "University of Toronto", "Stanford University",
    "Google Research", "Uber Engineering Blog", "VentureBeat",
}


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Field:
    """One field inside a BibTeX entry."""
    name: str
    value: str
    quote_style: str  # '"', '{', or 'raw' (for integers)

    def formatted(self, indent: str = "  ", align_col: int = 16) -> str:
        """Render this field as a canonical line.

        Uses brace-quoted form `{...}` to match the repo convention
        (vol1/vol2 references.bib are brace-quoted) and the existing
        bibtex-tidy pre-commit hook (--curly flag). Integer values
        (raw quote_style) are emitted without any quoting.
        """
        name_padded = self.name.ljust(align_col)
        if self.quote_style == "raw":
            return f"{indent}{name_padded} = {self.value},"
        # Brace-quoted form for all string values
        return f"{indent}{name_padded} = {{{self.value}}},"


@dataclass
class Entry:
    """One BibTeX entry (@inproceedings{...}, @article{...}, etc.)."""
    entry_type: str  # lowercase: "inproceedings", "article", ...
    key: str
    fields: list[Field]
    raw: str  # original text span for diff/debug
    start_line: int  # 1-indexed

    def get(self, name: str) -> Optional[Field]:
        for f in self.fields:
            if f.name.lower() == name.lower():
                return f
        return None

    def has(self, name: str) -> bool:
        return self.get(name) is not None


@dataclass
class Violation:
    """A lint violation against the §5 rules."""
    entry_key: str
    entry_line: int
    severity: str  # "error", "warning", "info"
    rule: str
    message: str
    fixable: bool = False


# ─── Parser (stateful, brace-counting, handles nested braces) ───────────────

def parse_bib(text: str) -> tuple[list[Entry], list[str]]:
    """Parse a BibTeX file into entries and preamble passthrough chunks.

    Returns (entries, preamble_chunks). Preamble chunks are anything
    between entries that is not an @entry — comments, @string defs,
    @preamble defs, blank lines. We preserve them verbatim so --fix
    never loses content.
    """
    entries: list[Entry] = []
    preamble_chunks: list[str] = []

    # State: track cursor position and line number
    i = 0
    n = len(text)
    last_chunk_start = 0

    while i < n:
        # Skip to next @
        while i < n and text[i] != "@":
            i += 1
        if i >= n:
            # Trailing non-entry content
            if last_chunk_start < n:
                preamble_chunks.append(text[last_chunk_start:n])
            break

        # Capture any pre-@ content as a preamble chunk
        if i > last_chunk_start:
            preamble_chunks.append(text[last_chunk_start:i])

        # Match @type{
        m = re.match(r"@(\w+)\s*\{", text[i:])
        if not m:
            # Lone @, treat as content
            i += 1
            last_chunk_start = i - 1
            continue

        entry_type = m.group(1).lower()
        entry_start = i
        body_brace_pos = i + m.end() - 1  # position of opening '{'

        # Find matching close brace via brace counting.
        # At the entry-boundary level, we only care about { and }. Double
        # quotes inside field values don't affect the outer structure —
        # particularly because LaTeX accent macros like \"{o} (for ö) and
        # \'{e} (for é) contain characters that look like quote delimiters
        # but are just characters inside brace-quoted values. Ignoring
        # quotes at this level is strictly correct because BibTeX entry
        # structure is defined by balanced braces only.
        depth = 1
        j = body_brace_pos + 1
        while j < n and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1

        if depth != 0:
            raise ValueError(
                f"Unbalanced braces starting at line "
                f"{text[:entry_start].count(chr(10)) + 1}"
            )

        entry_end = j + 1  # include closing '}'
        raw_entry = text[entry_start:entry_end]
        start_line = text[:entry_start].count("\n") + 1

        if entry_type in ("string", "preamble", "comment"):
            # Pass through verbatim as a preamble chunk
            preamble_chunks.append(raw_entry)
        else:
            entry = _parse_entry_body(
                raw_entry, entry_type, body_brace_pos - entry_start, start_line
            )
            entries.append(entry)

        i = entry_end
        last_chunk_start = i

    return entries, preamble_chunks


def _parse_entry_body(
    raw: str, entry_type: str, body_brace_offset: int, start_line: int
) -> Entry:
    """Parse the body of a single entry (key, field=value, ...)."""
    # raw spans from '@' through final '}'
    # body_brace_offset is the position of '{' within raw
    body = raw[body_brace_offset + 1 : -1]  # strip outer {}

    parts = _split_at_top_level_commas(body)
    if not parts:
        raise ValueError(f"Empty entry body at line {start_line}")

    key = parts[0].strip()
    if not key:
        raise ValueError(f"Missing citekey at line {start_line}")

    fields: list[Field] = []
    for part in parts[1:]:
        if "=" not in part:
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        value = value.strip()
        if not name:
            continue

        # Determine quote style and extract inner value
        quote_style: str
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            inner = value[1:-1]
            quote_style = '"'
        elif value.startswith("{") and value.endswith("}") and len(value) >= 2:
            # Must verify braces are balanced (not "{...}{...}")
            depth = 0
            balanced = True
            for k, c in enumerate(value):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0 and k < len(value) - 1:
                        balanced = False
                        break
            if balanced:
                inner = value[1:-1]
                quote_style = "{"
            else:
                inner = value
                quote_style = "raw"
        else:
            inner = value
            quote_style = "raw"

        fields.append(Field(name=name, value=inner, quote_style=quote_style))

    return Entry(
        entry_type=entry_type,
        key=key,
        fields=fields,
        raw=raw,
        start_line=start_line,
    )


def _split_at_top_level_commas(body: str) -> list[str]:
    """Split a comma-separated list while respecting braces and quotes.

    Tracks both brace depth AND "..."-quoted string state. Recognizes
    backslash-escaped quotes (`\\"`) as part of the content, NOT as
    string delimiters — this is critical because LaTeX accent macros
    like `\\"{o}` (for ö) contain a double quote that would otherwise
    flip the state incorrectly inside brace-quoted author fields.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_dquote = False
    i = 0
    while i < len(body):
        c = body[i]
        if c == '"' and (i == 0 or body[i - 1] != "\\"):
            # Only flip in_dquote for non-escaped quotes. Handles both
            # entry and exit of string literals.
            in_dquote = not in_dquote
            current.append(c)
        elif in_dquote:
            current.append(c)
        elif c == "{":
            depth += 1
            current.append(c)
        elif c == "}":
            depth -= 1
            current.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(c)
        i += 1
    if current:
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
    return parts


# ─── Validator ───────────────────────────────────────────────────────────────

def validate_entry(entry: Entry) -> list[Violation]:
    """Run §5 validation rules against one entry."""
    v: list[Violation] = []

    # Rule 1: required fields per entry type.
    # Field equivalencies (biblatex conventions accepted as
    # satisfying classic bibtex field names):
    #   - `date` satisfies `year` (biblatex uses ISO-8601 dates)
    #   - `editor` satisfies `author` for @book (editor-only volumes)
    req = REQUIRED_FIELDS.get(entry.entry_type, [])
    for rf in req:
        if entry.has(rf):
            continue
        # biblatex `date` satisfies `year`
        if rf == "year" and entry.has("date"):
            continue
        # @book accepts editor instead of author
        if rf == "author" and entry.entry_type == "book":
            if entry.has("editor"):
                continue
        v.append(Violation(
            entry.key, entry.start_line, "error",
            "missing-required-field",
            f"@{entry.entry_type} missing required field `{rf}`",
        ))

    # Rule 2: forbidden fields
    for f in entry.fields:
        if f.name.lower() in FORBIDDEN_FIELDS:
            v.append(Violation(
                entry.key, entry.start_line, "warning",
                "forbidden-field",
                f"field `{f.name}` is forbidden by §5 "
                f"(MIT Press round 1 cleanup); should be removed "
                f"or converted (organization → publisher, address → delete)",
                fixable=(f.name.lower() in ("organization", "address")),
            ))

    # Rule 3: journal name must not be abbreviated
    j = entry.get("journal")
    if j:
        for pat, expanded in JOURNAL_ABBREV_PATTERNS:
            if re.search(pat, j.value):
                v.append(Violation(
                    entry.key, entry.start_line, "warning",
                    "journal-abbreviated",
                    f"journal `{j.value}` looks abbreviated; "
                    f"should be `{expanded}` per §5",
                ))

    # Rule 4: publisher should be in canonical set (warning only)
    p = entry.get("publisher")
    if p and p.value not in CANONICAL_PUBLISHERS:
        v.append(Violation(
            entry.key, entry.start_line, "info",
            "publisher-not-canonical",
            f"publisher `{p.value}` is not in the canonical set; "
            f"verify it matches §5's publisher mapping table",
        ))

    # Rule 5: author field rules
    a = entry.get("author")
    if a:
        # Rule 5a: no "et al." in bib entries
        if re.search(r"\bet al\b", a.value, re.IGNORECASE):
            v.append(Violation(
                entry.key, entry.start_line, "error",
                "author-et-al",
                "author list contains `et al.`; §5 requires complete "
                "author lists (no truncation in the .bib entry)",
            ))
        # Rule 5b: no em-dash repeat-author shorthand
        if "---" in a.value or "—" in a.value:
            v.append(Violation(
                entry.key, entry.start_line, "error",
                "author-em-dash",
                "author list uses em-dash repeat shorthand; §5 requires "
                "the full author list on every entry",
            ))
        # Rule 5c: initial-only first names (heuristic: single cap + period)
        # Matches "Shallue, C. J." or "Shallue, C." pattern
        if re.search(r",\s+[A-Z]\.(?:\s+[A-Z]\.)*(?:\s+and\b|$|\s*,)", a.value):
            v.append(Violation(
                entry.key, entry.start_line, "warning",
                "author-initials-only",
                "author list uses initials for some first names; §5 "
                "prefers full first names when discoverable",
            ))

    # Rule 6: pages format (en-dash via --)
    pg = entry.get("pages")
    if pg and pg.value:
        if "-" in pg.value and "--" not in pg.value:
            # Single hyphen — should be double
            v.append(Violation(
                entry.key, entry.start_line, "warning",
                "pages-single-hyphen",
                f"pages `{pg.value}` uses single hyphen; "
                f"§5 requires `--` (en-dash) between start and end",
                fixable=True,
            ))

    # Rule 7: DOI must not include https:// prefix
    d = entry.get("doi")
    if d and d.value:
        if d.value.startswith("http://") or d.value.startswith("https://"):
            v.append(Violation(
                entry.key, entry.start_line, "warning",
                "doi-with-prefix",
                f"doi `{d.value}` includes URL prefix; "
                f"§5 requires bare DOI (no https://doi.org/)",
                fixable=True,
            ))

    # Rule 8 (field order) is intentionally not enforced by bib_lint.
    # The bibtex-tidy pre-commit hook uses --sort-fields to canonicalize
    # field order automatically. Duplicating that logic here would
    # produce double the noise and risk fighting bibtex-tidy's output.
    # bib_lint owns semantic validation; bibtex-tidy owns format.

    # Rule 9: x-verified must be ISO-8601 YYYY-MM-DD when present
    xv = entry.get("x-verified")
    if xv and not re.match(r"^\d{4}-\d{2}-\d{2}$", xv.value.strip()):
        v.append(Violation(
            entry.key, entry.start_line, "error",
            "bad-x-verified-date",
            f"x-verified `{xv.value}` is not ISO-8601 YYYY-MM-DD",
        ))

    return v


# ─── Formatter (auto-fix) ────────────────────────────────────────────────────

def format_entry(
    entry: Entry, indent: str = "  ", align_col: int = 16
) -> str:
    """Render an entry in canonical §5 form.

    - Preserves the insertion order of fields (bibtex-tidy will
      canonicalize field order at pre-commit time; bib_lint's job
      is semantic validation, not field reordering)
    - Two-space indent, aligned '=' column
    - Brace-quoted form `{...}` to match repo convention
    - Trailing comma on every field
    - Fields that are FORBIDDEN (organization, address) are DROPPED
      per §5 MIT Press round 1 cleanup rules

    Content is preserved byte-exact; only formatting changes.
    """
    fields_out: list[Field] = []
    for f in entry.fields:
        if f.name.lower() in FORBIDDEN_FIELDS:
            continue  # drop per §5
        fields_out.append(f)

    lines = [f"@{entry.entry_type}{{{entry.key},"]
    for f in fields_out:
        lines.append(f.formatted(indent=indent, align_col=align_col))
    lines.append("}")
    return "\n".join(lines)


def format_file(text: str) -> tuple[str, list[Violation]]:
    """Parse, validate, and re-emit a whole .bib file.

    Returns (formatted_text, violations). Violations are reported
    against the PRE-format file so line numbers match the source.
    """
    entries, preamble_chunks = parse_bib(text)
    all_violations: list[Violation] = []
    for e in entries:
        all_violations.extend(validate_entry(e))

    # Re-emit: preserve any leading preamble, then entries with blank
    # lines between them. We emit in original order.
    out_parts: list[str] = []
    if preamble_chunks and preamble_chunks[0].strip():
        out_parts.append(preamble_chunks[0].rstrip() + "\n\n")
    for e in entries:
        out_parts.append(format_entry(e))
        out_parts.append("\n\n")
    # Strip trailing double blank
    result = "".join(out_parts).rstrip() + "\n"
    return result, all_violations


# ─── Apply helper (used by the parallel-agent sweep apply step) ──────────────

def apply_fields(
    file_text: str,
    entry_key: str,
    entry_type: str,
    new_fields: list[tuple[str, str]],
    replace_existing: bool = False,
) -> str:
    """Safely insert new fields into an entry.

    This is the SAFE alternative to regex-based field insertion. It
    parses the file, locates the entry, adds the new fields, and
    re-emits the whole file in canonical form.

    Arguments:
        file_text: full .bib file contents
        entry_key: citekey of the target entry
        entry_type: entry type (for disambiguation; case-insensitive)
        new_fields: list of (name, value) tuples to add
        replace_existing: if True, overwrite existing fields with same
            name; if False, skip fields that already exist

    Returns the updated file text (in canonical format).
    """
    entries, preamble_chunks = parse_bib(file_text)
    found = False
    for e in entries:
        if e.key == entry_key and e.entry_type == entry_type.lower():
            found = True
            existing_names = {f.name.lower() for f in e.fields}
            for name, value in new_fields:
                nm = name.lower()
                if nm in existing_names:
                    if replace_existing:
                        # Replace in-place
                        for idx, f in enumerate(e.fields):
                            if f.name.lower() == nm:
                                e.fields[idx] = Field(
                                    name=name,
                                    value=str(value),
                                    quote_style='"' if not isinstance(value, int) else "raw",
                                )
                                break
                    continue
                e.fields.append(Field(
                    name=name,
                    value=str(value),
                    quote_style='"' if not isinstance(value, int) else "raw",
                ))
            break
    if not found:
        raise KeyError(f"Entry `{entry_key}` not found in file")

    # Re-emit (preamble + entries in order)
    out_parts: list[str] = []
    if preamble_chunks and preamble_chunks[0].strip():
        out_parts.append(preamble_chunks[0].rstrip() + "\n\n")
    for e in entries:
        out_parts.append(format_entry(e))
        out_parts.append("\n\n")
    return "".join(out_parts).rstrip() + "\n"


# ─── CLI ─────────────────────────────────────────────────────────────────────

def report_violations(path: Path, violations: list[Violation]) -> None:
    """Print violations grouped by severity."""
    if not violations:
        print(f"  {path}: OK")
        return
    errs = [v for v in violations if v.severity == "error"]
    warns = [v for v in violations if v.severity == "warning"]
    infos = [v for v in violations if v.severity == "info"]
    print(f"  {path}: {len(errs)} errors, {len(warns)} warnings, {len(infos)} info")
    for v in errs:
        print(f"    ERROR {v.entry_key}:{v.entry_line} [{v.rule}] {v.message}")
    for v in warns:
        print(f"    WARN  {v.entry_key}:{v.entry_line} [{v.rule}] {v.message}")
    for v in infos[:10]:  # cap info to avoid flooding
        print(f"    INFO  {v.entry_key}:{v.entry_line} [{v.rule}] {v.message}")
    if len(infos) > 10:
        print(f"    ... and {len(infos) - 10} more info items")


def find_all_bib_files() -> list[Path]:
    """Walk the repo and return all tracked .bib files."""
    import subprocess
    out = subprocess.check_output(
        ["git", "ls-files", "*.bib"], cwd=Path.cwd()
    ).decode()
    return [Path(p.strip()) for p in out.splitlines() if p.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help=".bib files to process")
    parser.add_argument("--all", action="store_true",
                        help="process all tracked .bib files in the repo")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="validate only; exit 1 on any NEW error "
                      "(errors in the baseline allow-list are grandfathered)")
    mode.add_argument("--fix", action="store_true",
                      help="rewrite files in canonical form")
    mode.add_argument("--report", action="store_true",
                      help="print detailed violation report (default)")
    mode.add_argument("--baseline", action="store_true",
                      help="regenerate the baseline allow-list from "
                      "current violations; grandfathers all current errors")
    args = parser.parse_args()

    if args.all:
        targets = find_all_bib_files()
    else:
        targets = [Path(f) for f in args.files]

    if not targets:
        parser.print_help()
        return 2

    mode_str = (
        "check" if args.check else "fix" if args.fix else "report"
    )
    print(f"bib_lint: {mode_str} mode, {len(targets)} file(s)")

    total_errors = 0
    total_new_errors = 0  # errors NOT in baseline
    total_warnings = 0
    total_fixed = 0
    all_violations_for_baseline: list[tuple[str, Violation]] = []
    baseline = load_baseline()

    for path in targets:
        if not path.exists():
            print(f"  {path}: MISSING")
            total_errors += 1
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"  {path}: READ ERROR {exc}")
            total_errors += 1
            continue

        try:
            entries, _ = parse_bib(text)
        except ValueError as exc:
            print(f"  {path}: PARSE ERROR {exc}")
            total_errors += 1
            continue

        violations: list[Violation] = []
        for e in entries:
            violations.extend(validate_entry(e))

        errs = [v for v in violations if v.severity == "error"]
        warns = [v for v in violations if v.severity == "warning"]
        total_errors += len(errs)
        total_warnings += len(warns)

        # For --baseline mode, collect all violations keyed by file
        fp_str = str(path).replace("/Users/VJ/GitHub/MLSysBook/", "")
        if args.baseline:
            for v in violations:
                all_violations_for_baseline.append((fp_str, v))

        # For --check mode, filter out baseline-allowed violations
        new_errs = errs
        if args.check:
            new_errs = [
                v for v in errs
                if (fp_str, v.entry_key, v.rule) not in baseline
            ]
            total_new_errors += len(new_errs)

        if args.fix:
            new_text, _ = format_file(text)
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                total_fixed += 1
                print(f"  {path}: FIXED ({len(entries)} entries)")
            else:
                print(f"  {path}: already canonical ({len(entries)} entries)")
        elif args.check:
            if new_errs:
                report_violations(path, new_errs)
        elif args.baseline:
            pass  # handled after loop
        else:
            report_violations(path, violations)

    if args.baseline:
        save_baseline(all_violations_for_baseline)
        n_err = sum(
            1 for _, v in all_violations_for_baseline
            if v.severity == "error"
        )
        print(
            f"\nBaseline written to {ALLOW_LIST_PATH.name}: "
            f"{n_err} errors grandfathered"
        )
        return 0

    if args.check:
        grandfathered = total_errors - total_new_errors
        print(
            f"\nTotal: {total_new_errors} NEW errors ({grandfathered} grandfathered), "
            f"{total_warnings} warnings"
        )
        if total_new_errors > 0:
            return 1
        return 0

    print(f"\nTotal: {total_errors} errors, {total_warnings} warnings, {total_fixed} fixed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
