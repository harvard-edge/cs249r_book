r"""Pre-commit guardrail: EPUB hygiene at source.

Purpose
-------
Prevent regression of the four source-level invariants that broke the
vol1/vol2 EPUB builds in early 2026 (see `.claude/_epub_audit/` and
the fix/epub-issues branch). Each invariant was previously detectable
only at epubcheck time (heavy Java tool, full EPUB build required).
This script checks them fast (<5s) directly against source files so a
bad commit never reaches CI.

Invariants checked
------------------

1. SVG aria-label C0 control characters.
   The matplotlib -> SVG path sometimes leaks raw-bytes representations
   into plot titles, producing `aria-label="\x03X"` or
   `aria-label="\x1dValidation"` style attributes. XML 1.0 forbids
   C0 control characters (U+0000-U+001F) in attribute values, except
   TAB (U+0009), LF (U+000A), CR (U+000D). Epubcheck raises FATAL
   RSC-016 and the EPUB fails to load in ClearView and Kindle.

2. Duplicate <marker id="X"/> inside a single SVG's <defs>.
   Copy-paste mistakes produce SVGs with two `<marker id="arrow">`
   entries (and similar for arrow-red, arrow-green). Epubcheck raises
   RSC-005 "Duplicate id" for every duplicate, and some EPUB readers
   fail to render the figure at all.

3. Backslash escapes (\\_ and \\%) in BibTeX URL / DOI fields.
   BibTeX requires `\_` to emit an underscore in LaTeX and `\%` to
   emit a percent. When citeproc hands the URL to an HTML href, the
   backslash leaks through. Epubcheck raises RSC-020 "Backslash used
   as path-segment delimiter."

4. Raw angle brackets (<, >) in BibTeX URL / DOI fields that will
   render to a href. Wiley's SICI-style DOI format legitimately
   contains `<` and `>`, but strict URI syntax forbids them. Epubcheck
   raises RSC-020 for these too.

Usage
-----
    python3 book/tools/audit/checks/epub_hygiene.py

Exit codes:
    0  - all source files clean
    1  - one or more hygiene violations; issues printed to stdout

Scope
-----
The script walks `book/quarto/contents/**/*.svg` for SVG files and
`book/quarto/**/*.bib` for BibTeX files, relative to the repository
root. It deliberately avoids parsing QMD files: QMD-level issues
(bare `<br>`, `--` inside TikZ comment blocks) are handled by the
`epub_postprocess.py` sanitizer as a defense-in-depth layer, and
scanning them at source would produce false positives from valid
TikZ edge syntax and valid HTML5 `<br>` tags in tables.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable


# -- paths --------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]  # .../book/tools/audit/checks -> repo
BOOK_DIR = REPO_ROOT / "book"
QUARTO_DIR = BOOK_DIR / "quarto"
CONTENTS_DIR = QUARTO_DIR / "contents"


# -- invariant 1: C0 control chars in SVG aria-label -------------------------

# C0 controls minus TAB/LF/CR (the three XML 1.0 permits in attribute values).
_C0_CONTROL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

# aria-label="..." with a single-line (non-quote) value.
_SVG_ARIA_LABEL_RE = re.compile(r'aria-label="([^"]*)"')


def check_svg_aria_label_c0(svg_file: Path) -> list[str]:
    """Return violation messages for C0 chars in aria-label values."""
    issues: list[str] = []
    try:
        text = svg_file.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError):
        return issues  # skip unreadable binary-ish files

    for match in _SVG_ARIA_LABEL_RE.finditer(text):
        value = match.group(1)
        bad = _C0_CONTROL_RE.findall(value)
        if bad:
            # Report the first few codepoints for diagnostic use.
            codepoints = ", ".join(f"U+{ord(c):04X}" for c in bad[:3])
            line_no = text.count('\n', 0, match.start()) + 1
            issues.append(
                f"{svg_file.relative_to(REPO_ROOT)}:{line_no}: "
                f"aria-label contains C0 control char(s) {codepoints} "
                f"(XML 1.0 forbids these in attribute values; "
                f"epubcheck FATAL RSC-016)"
            )
    return issues


# -- invariant 2: duplicate <marker id=...> in a single SVG ------------------

# Match any `id="X"` attribute on a <marker> element, tolerating other
# attributes before or after.
_MARKER_ID_RE = re.compile(r'<marker\b[^>]*\bid="([^"]+)"')


def check_svg_duplicate_markers(svg_file: Path) -> list[str]:
    """Return violation messages for duplicate <marker id=...> in one file."""
    issues: list[str] = []
    try:
        text = svg_file.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError):
        return issues

    seen: dict[str, int] = {}
    for match in _MARKER_ID_RE.finditer(text):
        mid = match.group(1)
        line_no = text.count('\n', 0, match.start()) + 1
        if mid in seen:
            issues.append(
                f"{svg_file.relative_to(REPO_ROOT)}:{line_no}: "
                f"duplicate <marker id=\"{mid}\"/> "
                f"(first defined at line {seen[mid]}; "
                f"epubcheck RSC-005 Duplicate id)"
            )
        else:
            seen[mid] = line_no
    return issues


# -- invariant 3+4: BibTeX URL / DOI field hygiene ---------------------------

# Lines that begin a `url = { ... }` or `doi = { ... }` block. We walk
# field contents across continuation lines until the closing brace.
_BIB_URL_FIELD_RE = re.compile(
    r'^\s*(url|doi)\s*=\s*\{([^{}]*)\}',
    flags=re.MULTILINE,
)


def check_bibtex_url_escapes(bib_file: Path) -> list[str]:
    """Return violation messages for backslash escapes / raw <>  in
    url= or doi= field values.
    """
    issues: list[str] = []
    try:
        text = bib_file.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError):
        return issues

    rel = bib_file.relative_to(REPO_ROOT)
    for match in _BIB_URL_FIELD_RE.finditer(text):
        field = match.group(1)
        value = match.group(2)
        line_no = text.count('\n', 0, match.start()) + 1

        # Skip http(s)-less values (e.g. a bare DOI like "10.1145/...")
        # which are not URL-validated by epubcheck directly.
        is_url = 'http' in value.lower() or field == 'url'
        if not is_url:
            continue

        if r'\_' in value:
            issues.append(
                f"{rel}:{line_no}: "
                r"{field}= contains '\_' escape — ".replace('{field}', field)
                + "citeproc leaks this as literal backslash into href; "
                "epubcheck RSC-020 'backslash used as path-segment delimiter'"
            )
        if r'\%' in value:
            issues.append(
                f"{rel}:{line_no}: "
                r"{field}= contains '\%' escape — ".replace('{field}', field)
                + "citeproc leaks this as literal backslash into href; "
                "epubcheck RSC-020"
            )
        if '<' in value or '>' in value:
            # Only raw angle brackets — not `&lt;` / `&gt;` which are valid
            # as XML attribute content but still need percent-encoding for
            # strict URI syntax (epubcheck decodes entities before checking).
            issues.append(
                f"{rel}:{line_no}: "
                f"{field}= contains raw '<' or '>' — "
                "strict URI syntax forbids these in path segments; "
                "percent-encode as %3C / %3E; epubcheck RSC-020"
            )

    return issues


# -- driver -------------------------------------------------------------------

def _iter_svgs() -> Iterable[Path]:
    if not CONTENTS_DIR.is_dir():
        return
    yield from CONTENTS_DIR.rglob('*.svg')


def _iter_bibs() -> Iterable[Path]:
    if not QUARTO_DIR.is_dir():
        return
    yield from QUARTO_DIR.rglob('*.bib')


def main() -> int:
    all_issues: list[str] = []

    for svg in _iter_svgs():
        all_issues.extend(check_svg_aria_label_c0(svg))
        all_issues.extend(check_svg_duplicate_markers(svg))

    for bib in _iter_bibs():
        all_issues.extend(check_bibtex_url_escapes(bib))

    if all_issues:
        print("EPUB hygiene violations detected:", file=sys.stderr)
        print("", file=sys.stderr)
        for issue in all_issues:
            print(f"  {issue}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            f"Found {len(all_issues)} issue(s). See "
            "book/tools/audit/checks/epub_hygiene.py for rule context.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
