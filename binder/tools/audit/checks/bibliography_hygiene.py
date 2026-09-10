"""Check: Bibliography hygiene (§10.13 / §5).

Rule: book-prose-merged.md section 10.13 and section 5

    "Every `@inproceedings` must have a `publisher` field.
     Every `@article` must have a `journal` field.
     Include `pages` and `doi` when available.
     No em dashes for repeat author names.
     Letter-by-letter alphabetical order.
     Publisher locations (cities) removed for consistency.
     Confirm all URLs are live before submission."

This check is the first non-.qmd check in the audit scanner. It
runs on `.bib` files (declared via the module attribute
`FILE_EXTENSIONS = (".bib",)`) and is skipped by all .qmd-only
existing checks via the same scan.py filter.

First-pass implementation targets the two highest-leverage,
mechanically-decidable rules:

  1. Every `@inproceedings` entry has a `publisher` field.
  2. Every `@article` entry has a `journal` field.

Other §10.13 rules (pages, doi, repeat-author em dashes,
letter-by-letter order, URL liveness, publisher city removal)
are deferred to follow-up items. Each requires different parsing
logic and several of them require network I/O (URL checks) or
careful ordering logic that is best handled in a dedicated pass.

Parser strategy: simple, regex-based, one-pass extraction. Not a
full BibTeX parser — .bib syntax permits nested braces and
@string substitution that a regex cannot fully handle. But the
two rules we check (publisher on inproceedings, journal on
article) only require knowing an entry's type and whether its
body contains the field name as a top-level key.

auto_fixable=False. Adding a `publisher` or `journal` field to
a bibliography entry requires looking up the correct value for
that specific publication — editorial/subagent work.
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id

CATEGORY = "bibliography-hygiene"
RULE = "book-prose-merged.md section 10.13"
RULE_TEXT = "Bibliography entries must have required fields per §10.13"

# Declare that this check runs on .bib files, not .qmd.
FILE_EXTENSIONS = (".bib",)


# ── BibTeX entry parser (conservative regex) ───────────────────────────────
#
# BibTeX entries start with `@type{key,` on their own line. The entry
# body runs until the matching closing `}` at the top level. Entries
# may contain nested braces inside field values, so a simple regex
# cannot match an entry atomically — we walk the text and balance
# braces manually.

_ENTRY_START_RE = re.compile(r"^\s*@(\w+)\s*\{\s*([\w:-]+)\s*,", re.MULTILINE)


def _entry_body_end(text: str, open_brace_pos: int) -> int:
    """Return the index of the closing `}` that balances the open brace
    at `open_brace_pos`. Returns -1 if unbalanced (malformed entry)."""
    depth = 0
    i = open_brace_pos
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            # Skip escaped characters (e.g. `\{`, `\}` inside field values)
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


# Detects a top-level field key inside an entry body. BibTeX is forgiving
# about whitespace and case, so we accept `Publisher = "..."` as well as
# `publisher = "..."`. Matches at line start (or after a comma) to avoid
# matching the same word inside a quoted title.
_FIELD_KEY_RE_CACHE: dict[str, re.Pattern] = {}


def _has_field(body: str, field: str) -> bool:
    """Return True if the entry body has a top-level `field = ...` key.

    Matching is case-insensitive. We use a regex anchored to a line
    boundary or a comma to avoid matching the word inside a quoted or
    braced title. Whitespace around `=` is flexible.
    """
    pattern = _FIELD_KEY_RE_CACHE.get(field)
    if pattern is None:
        pattern = re.compile(
            r"(?:^|[,\n])\s*" + re.escape(field) + r"\s*=",
            re.IGNORECASE,
        )
        _FIELD_KEY_RE_CACHE[field] = pattern
    return pattern.search(body) is not None


# ── Rule: every @inproceedings needs `publisher`, every @article needs `journal` ──

_REQUIRED_FIELDS = {
    "inproceedings": "publisher",
    "article":       "journal",
}


def _line_number_at(text: str, pos: int) -> int:
    """Return 1-indexed line number at byte offset `pos`."""
    return text.count("\n", 0, pos) + 1


def _entry_first_line(text: str, entry_start_pos: int) -> str:
    """Return the first line of the entry (for the Issue's `before` field)."""
    end = text.find("\n", entry_start_pos)
    if end == -1:
        end = len(text)
    return text[entry_start_pos:end]


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan a .bib file for §10.13 bibliography-hygiene violations.

    For each top-level entry, check whether it has the required field
    for its entry type. Emit one Issue per missing field.
    """
    issues: list[Issue] = []
    counter = start_counter

    # Safety: only process .bib files.
    if file_path.suffix != ".bib":
        return issues, counter

    for entry_match in _ENTRY_START_RE.finditer(text):
        entry_type = entry_match.group(1).lower()
        entry_key = entry_match.group(2)
        required = _REQUIRED_FIELDS.get(entry_type)
        if required is None:
            continue  # entry type has no required field in this check

        # Find the opening brace position for the body
        open_brace = text.find("{", entry_match.start())
        if open_brace == -1:
            continue
        end = _entry_body_end(text, open_brace)
        if end == -1:
            continue
        body = text[open_brace + 1:end]

        if _has_field(body, required):
            continue

        # Missing required field. Emit issue at the entry start line.
        entry_start = entry_match.start()
        line_num = _line_number_at(text, entry_start)
        issues.append(
            Issue(
                id=make_issue_id(scope, CATEGORY, counter),
                category=CATEGORY,
                rule=RULE,
                rule_text=RULE_TEXT,
                file=str(file_path),
                line=line_num,
                col=0,
                before=_entry_first_line(text, entry_start),
                suggested_after="",
                auto_fixable=False,
                needs_subagent=True,
                reason=(
                    f"@{entry_type}{{{entry_key}}} missing required "
                    f"field `{required}`"
                ),
            )
        )
        counter += 1

    return issues, counter


# ── Adversarial self-test ──────────────────────────────────────────────────

_TESTS = [
    # Positive: @inproceedings missing publisher
    (
        "inproceedings missing publisher",
        """@inproceedings{foo2020bar,
  author = "Foo, Jane",
  title = "A paper",
  booktitle = "NeurIPS",
  year = 2020
}
""",
        1,
    ),
    # Negative: @inproceedings with publisher
    (
        "inproceedings with publisher",
        """@inproceedings{foo2020bar,
  author = "Foo, Jane",
  title = "A paper",
  booktitle = "NeurIPS",
  publisher = "Curran Associates",
  year = 2020
}
""",
        0,
    ),
    # Positive: @article missing journal
    (
        "article missing journal",
        """@article{bar2021baz,
  author = "Bar, John",
  title = "A journal paper",
  year = 2021,
  volume = 42,
  pages = "1-10"
}
""",
        1,
    ),
    # Negative: @article with journal
    (
        "article with journal",
        """@article{bar2021baz,
  author = "Bar, John",
  title = "A journal paper",
  journal = "Nature",
  year = 2021
}
""",
        0,
    ),
    # Negative: @book is not checked (no required field in this pass)
    (
        "book entry skipped",
        """@book{clrs2009,
  author = "Cormen, Thomas H.",
  title = "Introduction to Algorithms",
  year = 2009
}
""",
        0,
    ),
    # Mixed: one missing, one present
    (
        "mixed inproceedings: one missing publisher, one has it",
        """@inproceedings{good2020,
  author = "Good",
  title = "Fine",
  booktitle = "X",
  publisher = "Springer",
  year = 2020
}

@inproceedings{bad2021,
  author = "Bad",
  title = "Missing",
  booktitle = "Y",
  year = 2021
}
""",
        1,
    ),
    # Nested braces in title (edge case for body parser)
    (
        "nested braces don't confuse body parser",
        """@inproceedings{nest2020,
  author = "Nest",
  title = "A paper with {nested} braces",
  booktitle = "CoRR",
  year = 2020
}
""",
        1,  # still missing publisher
    ),
    # Case-insensitive field key
    (
        "capitalized Publisher field still counts",
        """@inproceedings{cap2020,
  author = "Cap",
  title = "P",
  booktitle = "X",
  Publisher = "ACM",
  year = 2020
}
""",
        0,
    ),
]


def _self_test() -> int:
    passed = 0
    failed = 0
    failures: list[str] = []

    class _FakePath:
        """Pretend to be a Path so `check` accepts us; pose as .bib."""

        def __init__(self, name: str):
            self.suffix = ".bib"
            self.name = name

        def __str__(self) -> str:
            return f"<test:{self.name}>"

    for name, text, expected in _TESTS:
        issues, _ = check(_FakePath(name), text, "test", 0)
        got = len(issues)
        if got == expected:
            passed += 1
        else:
            failed += 1
            failures.append(f"{name}: expected {expected}, got {got}")

    total = passed + failed
    print(f"bibliography_hygiene self-test: {passed}/{total} passed")
    for f in failures:
        print(f"  - {f}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_self_test())
