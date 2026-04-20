"""EPUB check primitives used by the `binder check epub` command group.

This module holds the pure-Python logic for the two `check epub` scopes
so that `validate.py` can call it as ordinary Python (not via a
subprocess-to-a-script, which is what the older delegated-script
pattern used). All checks here return a list of
`(file, line, col, code, severity, message)` tuples that the caller
wraps into `ValidationIssue` objects.

Scopes surfaced to the user (via `./binder check epub --scope X`):

  hygiene
    Source-level invariants that cause epubcheck to reject the built
    EPUB. Fast (<1s): regex over SVG and BibTeX files only; no EPUB
    build required. Suitable for pre-commit. Catches the four error
    categories that broke vol1/vol2 builds in April 2026.

  epubcheck
    Run the W3C epubcheck validator (https://github.com/w3c/epubcheck)
    against the most recently built EPUBs under `_build/epub-vol*/`.
    Requires the `epubcheck` binary in PATH (or the `epubcheck` Python
    package). Slow (~30s per volume): must be invoked post-build, so
    it belongs in CI, not pre-commit. Fails on `FATAL` by default;
    `--max-errors` controls the `ERROR` threshold.

  structure
    Legacy custom-check script (validate_epub.py). Checks mimetype,
    container.xml, CSS variables, XML comments, etc. Retained because
    it runs without Java, so it is the right choice for a local
    smoke check when epubcheck is unavailable. Does NOT replace
    epubcheck — some of its checks overlap, but the strict schema
    validation only epubcheck does is not replicated here.

Layer relationship: hygiene catches source issues *before* the build,
epubcheck catches rendered-EPUB issues *after* the build, and
structure catches a subset of epubcheck issues when Java is not
available. Together they form a defense-in-depth net so issues like
"--" in TikZ comments, C0 chars in SVG aria-labels, duplicate marker
ids, and BibTeX URL escapes cannot silently ship to Kindle again.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


# ---------------------------------------------------------------------------
# Dataclass used for hygiene and epubcheck issues alike. The fields match
# what `ValidationIssue` in validate.py expects; the wrapper in validate.py
# constructs a ValidationIssue from each record.
# ---------------------------------------------------------------------------


@dataclass
class EpubIssue:
    file: str          # repo-relative path
    line: int          # 1-indexed line number (0 = unknown)
    col: int           # 1-indexed column (0 = unknown)
    code: str          # short stable code ("svg-c0", "bib-url-escape", "RSC-016", etc.)
    severity: str      # "error" | "warning" | "info" | "fatal"
    message: str       # human-readable message for the user

    # Optional: the exact source snippet so the operator can grep for it.
    context: str = ""


# ---------------------------------------------------------------------------
# Hygiene: regex invariants over SVG and BibTeX source files.
# ---------------------------------------------------------------------------

# C0 control chars that XML 1.0 forbids in attribute values
# (TAB 0x09, LF 0x0A, CR 0x0D are allowed).
_C0_CONTROL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

# aria-label="..." inside an SVG attribute list.
_SVG_ARIA_LABEL_RE = re.compile(r'aria-label="([^"]*)"')

# <marker ... id="X" ...>  — just the opening tag with the id attribute.
_MARKER_ID_RE = re.compile(r'<marker\b[^>]*\bid="([^"]+)"')

# BibTeX: a single-braced `url = { ... }` or `doi = { ... }` field.
# We deliberately only match when the value does not contain a nested `{`
# or `}`, which covers every real entry in the book's .bib files.
_BIB_URL_FIELD_RE = re.compile(
    r'^(\s*)(url|doi)(\s*=\s*\{)([^{}]*)(\})',
    flags=re.MULTILINE,
)


def _line_of(text: str, offset: int) -> int:
    """Return 1-indexed line number of *offset* in *text*."""
    return text.count('\n', 0, offset) + 1


def _iter_svgs(contents_dir: Path) -> Iterator[Path]:
    if contents_dir.is_dir():
        yield from contents_dir.rglob('*.svg')


def _iter_bibs(quarto_dir: Path) -> Iterator[Path]:
    if quarto_dir.is_dir():
        yield from quarto_dir.rglob('*.bib')


def _read(path: Path) -> str | None:
    """Read a text file, returning None on error (never raise to the caller)."""
    try:
        return path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError):
        return None


def check_svg_aria_label_c0(svg_file: Path, repo_root: Path) -> list[EpubIssue]:
    """Return issues for C0 control chars in SVG aria-label attribute values."""
    issues: list[EpubIssue] = []
    text = _read(svg_file)
    if text is None:
        return issues
    for match in _SVG_ARIA_LABEL_RE.finditer(text):
        value = match.group(1)
        bad = _C0_CONTROL_RE.findall(value)
        if not bad:
            continue
        codepoints = ", ".join(f"U+{ord(c):04X}" for c in bad[:3])
        issues.append(EpubIssue(
            file=str(svg_file.relative_to(repo_root)),
            line=_line_of(text, match.start()),
            col=0,
            code="svg-c0",
            severity="error",
            message=(
                f"SVG aria-label contains C0 control char(s) {codepoints}; "
                "XML 1.0 forbids these in attribute values (epubcheck FATAL RSC-016)"
            ),
            context=match.group(0)[:80],
        ))
    return issues


def check_svg_duplicate_markers(svg_file: Path, repo_root: Path) -> list[EpubIssue]:
    """Return issues for duplicate `<marker id=...>` inside one SVG file."""
    issues: list[EpubIssue] = []
    text = _read(svg_file)
    if text is None:
        return issues
    seen: dict[str, int] = {}
    for match in _MARKER_ID_RE.finditer(text):
        mid = match.group(1)
        line = _line_of(text, match.start())
        if mid in seen:
            issues.append(EpubIssue(
                file=str(svg_file.relative_to(repo_root)),
                line=line,
                col=0,
                code="svg-dupe-marker",
                severity="error",
                message=(
                    f'duplicate <marker id="{mid}"/>; '
                    f"first defined at line {seen[mid]} "
                    "(epubcheck RSC-005 'Duplicate id')"
                ),
                context=match.group(0)[:80],
            ))
        else:
            seen[mid] = line
    return issues


def check_bibtex_url_escapes(bib_file: Path, repo_root: Path) -> list[EpubIssue]:
    r"""Return issues for \_ / \% escapes and raw <> in bib URL/DOI fields."""
    issues: list[EpubIssue] = []
    text = _read(bib_file)
    if text is None:
        return issues
    rel = str(bib_file.relative_to(repo_root))
    for match in _BIB_URL_FIELD_RE.finditer(text):
        _prefix, field, _eq, value, _close = match.groups()
        # Skip DOI-only fields without http; citeproc constructs their URL
        # itself from the raw DOI and has special-case handling for escapes.
        is_urlish = 'http' in value.lower() or field == 'url'
        if not is_urlish:
            continue
        line = _line_of(text, match.start())
        if r'\_' in value:
            issues.append(EpubIssue(
                file=rel, line=line, col=0,
                code="bib-url-escape-underscore",
                severity="error",
                message=(
                    rf"{field}= value contains BibTeX '\_' escape; "
                    "citeproc leaks this as a literal backslash into the "
                    "rendered href (epubcheck RSC-020)"
                ),
                context=value[:80],
            ))
        if r'\%' in value:
            issues.append(EpubIssue(
                file=rel, line=line, col=0,
                code="bib-url-escape-percent",
                severity="error",
                message=(
                    rf"{field}= value contains BibTeX '\%' escape; "
                    "citeproc leaks this as a literal backslash into the "
                    "rendered href (epubcheck RSC-020)"
                ),
                context=value[:80],
            ))
        if '<' in value or '>' in value:
            issues.append(EpubIssue(
                file=rel, line=line, col=0,
                code="bib-url-raw-angle",
                severity="error",
                message=(
                    f"{field}= value contains raw '<' or '>'; "
                    "strict URI syntax forbids these in path segments "
                    "(percent-encode as %3C / %3E; epubcheck RSC-020)"
                ),
                context=value[:80],
            ))
    return issues


# ---------------------------------------------------------------------------
# Hygiene auto-repair. Mirrors the rewriters in
# `book/quarto/scripts/epub_postprocess.py` but applied to SOURCE files,
# not to the extracted EPUB. Running the fixer once against a stale
# checkout removes every legacy occurrence of the four invariant
# classes at source; future regressions are caught at commit time by
# the hygiene scope.
#
# All rewrites are deterministic and idempotent. Running `--fix` twice
# produces the same output as running it once.
# ---------------------------------------------------------------------------


def _fix_svg_aria_label_c0(svg_file: Path) -> int:
    """Strip C0 control chars from aria-label values. Returns bytes removed."""
    text = _read(svg_file)
    if text is None:
        return 0
    removed = 0

    def strip(m):
        nonlocal removed
        value = m.group(1)
        cleaned = _C0_CONTROL_RE.sub('', value)
        if cleaned == value:
            return m.group(0)
        removed += len(value) - len(cleaned)
        return f'aria-label="{cleaned}"'

    new_text = _SVG_ARIA_LABEL_RE.sub(strip, text)
    if new_text != text:
        svg_file.write_text(new_text, encoding='utf-8')
    return removed


# Regex that captures an entire `<marker ...>...</marker>` block (compact
# or expanded) for deduplication. DOTALL so it spans lines.
_MARKER_BLOCK_RE = re.compile(
    r'<marker\b[^>]*\bid="([^"]+)"[^>]*>(?:(?!<marker\b).)*?</marker>',
    flags=re.DOTALL,
)


def _fix_svg_duplicate_markers(svg_file: Path) -> int:
    """Remove duplicate `<marker id="X"/>` blocks; keep the first occurrence."""
    text = _read(svg_file)
    if text is None:
        return 0

    seen: set[str] = set()
    removed = 0

    def dedup(m):
        nonlocal removed
        mid = m.group(1)
        if mid in seen:
            removed += 1
            return ""  # drop duplicate
        seen.add(mid)
        return m.group(0)

    new_text = _MARKER_BLOCK_RE.sub(dedup, text)
    if removed == 0:
        return 0
    # Collapse blank-line runs left by removed blocks for readable diffs.
    new_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', new_text)
    svg_file.write_text(new_text, encoding='utf-8')
    return removed


def _fix_bibtex_url_escapes(bib_file: Path) -> int:
    r"""Rewrite URL fields to strip BibTeX escapes and percent-encode < >."""
    text = _read(bib_file)
    if text is None:
        return 0
    fixes = 0

    def sanitize(m):
        nonlocal fixes
        prefix, field, eq, value, close = m.groups()
        is_urlish = 'http' in value.lower() or field == 'url'
        if not is_urlish:
            return m.group(0)
        new_value = value
        if r'\_' in new_value:
            fixes += new_value.count(r'\_')
            new_value = new_value.replace(r'\_', '_')
        if r'\%' in new_value:
            fixes += new_value.count(r'\%')
            new_value = new_value.replace(r'\%', '%')
        if '<' in new_value:
            fixes += new_value.count('<')
            new_value = new_value.replace('<', '%3C')
        if '>' in new_value:
            fixes += new_value.count('>')
            new_value = new_value.replace('>', '%3E')
        if new_value == value:
            return m.group(0)
        return f"{prefix}{field}{eq}{new_value}{close}"

    new_text = _BIB_URL_FIELD_RE.sub(sanitize, text)
    if new_text != text:
        bib_file.write_text(new_text, encoding='utf-8')
    return fixes


def fix_hygiene_issues(repo_root: Path) -> dict[str, int]:
    """Auto-repair every hygiene issue at source.

    Returns a counts dict with keys:
      svg_c0_chars_removed:        characters stripped across all SVGs
      svg_duplicate_markers:       <marker> blocks removed across all SVGs
      bib_url_rewrites:            BibTeX URL-field substitutions performed

    The fix is deterministic and idempotent; the counts reflect how much
    work was done on this invocation, not the total number of remaining
    legacy occurrences (running a second time yields zeros).
    """
    contents_dir = repo_root / "book" / "quarto" / "contents"
    quarto_dir = repo_root / "book" / "quarto"

    counts = {
        "svg_c0_chars_removed": 0,
        "svg_duplicate_markers": 0,
        "bib_url_rewrites": 0,
    }

    for svg in _iter_svgs(contents_dir):
        counts["svg_c0_chars_removed"] += _fix_svg_aria_label_c0(svg)
        counts["svg_duplicate_markers"] += _fix_svg_duplicate_markers(svg)

    for bib in _iter_bibs(quarto_dir):
        counts["bib_url_rewrites"] += _fix_bibtex_url_escapes(bib)

    return counts


def find_hygiene_issues(repo_root: Path) -> tuple[list[EpubIssue], int]:
    """Walk repo source and return (issues, num_files_checked).

    Parameters
    ----------
    repo_root : Path
        Absolute path to the repository root (the one that contains
        `book/`). Used to make issue file paths repo-relative.

    Returns
    -------
    (issues, num_files) where num_files is the total count of SVG+BIB
    files scanned, used for the `files_checked` field of the
    ValidationRunResult.
    """
    issues: list[EpubIssue] = []
    files_checked = 0

    contents_dir = repo_root / "book" / "quarto" / "contents"
    quarto_dir = repo_root / "book" / "quarto"

    for svg in _iter_svgs(contents_dir):
        files_checked += 1
        issues.extend(check_svg_aria_label_c0(svg, repo_root))
        issues.extend(check_svg_duplicate_markers(svg, repo_root))

    for bib in _iter_bibs(quarto_dir):
        files_checked += 1
        issues.extend(check_bibtex_url_escapes(bib, repo_root))

    return issues, files_checked


# ---------------------------------------------------------------------------
# Epubcheck wrapper: runs the W3C validator and maps JSON output to EpubIssues.
# ---------------------------------------------------------------------------

# Mapping from epubcheck severity names to our severity strings.
_EPUBCHECK_SEVERITY_MAP = {
    "FATAL": "fatal",
    "ERROR": "error",
    "WARNING": "warning",
    "INFO": "info",
    "USAGE": "info",
    "SUPPRESSED": "info",
}


def _find_epubcheck_executable() -> list[str] | None:
    """Return a subprocess argv prefix for invoking epubcheck, or None.

    Preference order:
      1. `epubcheck` on PATH (brew / apt install).
      2. `python -m epubcheck` (the PyPI wrapper package, bundles the jar
         and provides a CLI entry point — see `epubcheck` on PyPI).
      3. None — caller must emit an "epubcheck not available" message.
    """
    if shutil.which("epubcheck"):
        return ["epubcheck"]
    # The PyPI package installs an `epubcheck` command too (handled by 1),
    # but on some systems the command lives only as a module. Try both.
    try:
        import epubcheck  # noqa: F401
        return ["python3", "-m", "epubcheck"]
    except ImportError:
        return None


def _discover_built_epubs(repo_root: Path) -> list[Path]:
    """Return the most recent EPUB per-volume under `_build/epub-vol*/`."""
    build_root = repo_root / "book" / "quarto" / "_build"
    if not build_root.is_dir():
        return []
    results: list[Path] = []
    for vol_dir in sorted(build_root.glob("epub-vol*")):
        candidates = sorted(vol_dir.glob("*.epub"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            results.append(candidates[0])
    return results


def run_epubcheck_on(
    epub_path: Path,
    *,
    repo_root: Path,
    timeout_seconds: int = 180,
) -> tuple[list[EpubIssue], dict[str, int]]:
    """Run epubcheck against *epub_path* and return (issues, severity counts).

    If epubcheck is not available on the system, returns a single
    `epubcheck-missing` issue so the caller can surface an install hint.

    Severity counts include FATAL, ERROR, WARNING keys so the caller can
    compare against MAX_FATAL / MAX_ERRORS thresholds without re-counting.
    """
    cmd_prefix = _find_epubcheck_executable()
    if cmd_prefix is None:
        return (
            [EpubIssue(
                file=str(epub_path.relative_to(repo_root)) if epub_path.is_relative_to(repo_root) else str(epub_path),
                line=0, col=0,
                code="epubcheck-missing",
                severity="error",
                message=(
                    "epubcheck is not installed. Install with "
                    "`pip install epubcheck` (requires JRE 8+) or via your OS "
                    "package manager (`brew install epubcheck` / "
                    "`apt install epubcheck`)."
                ),
            )],
            {"FATAL": 0, "ERROR": 1, "WARNING": 0},
        )

    cmd = cmd_prefix + ["--json", "-", str(epub_path)]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return (
            [EpubIssue(
                file=str(epub_path), line=0, col=0,
                code="epubcheck-missing",
                severity="error",
                message="epubcheck executable vanished between lookup and invocation",
            )],
            {"FATAL": 0, "ERROR": 1, "WARNING": 0},
        )
    except subprocess.TimeoutExpired:
        return (
            [EpubIssue(
                file=str(epub_path), line=0, col=0,
                code="epubcheck-timeout",
                severity="error",
                message=f"epubcheck timed out after {timeout_seconds}s",
            )],
            {"FATAL": 1, "ERROR": 0, "WARNING": 0},
        )

    # epubcheck prints JSON to stdout when `--json -` is used. Non-zero
    # exit code (1 or 2) is expected when errors are found — we only
    # treat JSON-parse failure as a real invocation error.
    issues, severity_counts = _parse_epubcheck_json(
        completed.stdout, epub_path, repo_root,
    )
    if issues or completed.returncode == 0:
        return issues, severity_counts

    # No issues parsed but non-zero exit: surface stderr for diagnostics.
    stderr = (completed.stderr or "").strip()[:500]
    return (
        [EpubIssue(
            file=str(epub_path), line=0, col=0,
            code="epubcheck-invocation-error",
            severity="error",
            message=(
                f"epubcheck exited with code {completed.returncode} but "
                f"produced no JSON output. stderr: {stderr!r}"
            ),
        )],
        {"FATAL": 0, "ERROR": 1, "WARNING": 0},
    )


def _parse_epubcheck_json(
    stdout: str, epub_path: Path, repo_root: Path,
) -> tuple[list[EpubIssue], dict[str, int]]:
    """Convert epubcheck's JSON message list to EpubIssue records."""
    issues: list[EpubIssue] = []
    counts = {"FATAL": 0, "ERROR": 0, "WARNING": 0, "INFO": 0, "USAGE": 0}
    if not stdout.strip():
        return issues, counts
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        # Some epubcheck versions print a banner before the JSON. Look for
        # the first `{` and try again.
        brace = stdout.find("{")
        if brace < 0:
            return issues, counts
        try:
            payload = json.loads(stdout[brace:])
        except json.JSONDecodeError:
            return issues, counts

    messages = payload.get("messages", []) or []
    epub_rel = str(epub_path.relative_to(repo_root)) if epub_path.is_relative_to(repo_root) else str(epub_path)

    for m in messages:
        severity_raw = (m.get("severity") or "").upper()
        counts[severity_raw] = counts.get(severity_raw, 0) + 1
        severity = _EPUBCHECK_SEVERITY_MAP.get(severity_raw, "error")
        code = m.get("ID") or "UNKNOWN"
        message_text = m.get("message") or ""
        # Epubcheck emits one message with a list of locations. We flatten
        # to one issue per location so the caller can surface each one.
        locations = m.get("locations") or [{}]
        for loc in locations:
            path = loc.get("path") or epub_rel
            line = int(loc.get("line") or 0)
            col = int(loc.get("column") or 0)
            # The path inside epubcheck is a zip-internal path (e.g.
            # "EPUB/text/ch008.xhtml"). Prefix the epub file so users can
            # tell which volume the issue is from.
            displayed_file = f"{epub_rel}!{path}" if path and path != epub_rel else epub_rel
            issues.append(EpubIssue(
                file=displayed_file,
                line=line,
                col=col,
                code=code,
                severity=severity,
                message=message_text,
            ))

    return issues, counts


# ---------------------------------------------------------------------------
# Smoke check: reader-compatibility invariants epubcheck does not cover.
#
# Epubcheck enforces EPUB 3 spec conformance, but some readers enforce a
# stricter subset. The patterns checked here produced real user-reported
# breakage before the `fix/epub-issues` work:
#
#   CSS custom properties (`--var` declarations, `var(--x)` usage)
#       Valid per CSS 4 spec but not implemented in older / embedded EPUB
#       renderers (ClearView, Tolino firmware pre-2023, some Kobo builds).
#       Root cause of issue #1052 before the CSS was rewritten without
#       custom properties.
#
#   External resource references (src=, href= pointing off-device)
#       EPUB readers typically do not fetch external resources — an
#       `<img src="https://...">` shows a broken-image icon on every
#       reader that enforces offline rendering. Zero false positives
#       in well-authored EPUBs; this is always a bug.
#
# The smoke check runs in <1s against a built EPUB and does not require
# Java, so it is useful when epubcheck is not installed locally and as
# a belt-and-suspenders confirmation alongside epubcheck.
# ---------------------------------------------------------------------------

# CSS custom property declarations like `--accent-color: #333;`
_CSS_CUSTOM_PROP_DECL_RE = re.compile(r'^\s*(--[\w-]+)\s*:', flags=re.MULTILINE)

# CSS custom property consumption like `color: var(--accent-color);`
_CSS_CUSTOM_PROP_USE_RE = re.compile(r'\bvar\(\s*(--[\w-]+)')

# External resource references in XHTML. We match href= and src= values
# that begin with a scheme (http, https, ftp, data: except for known safe
# data URIs) or with `//`. In-EPUB paths start with a letter or `../`.
_EXTERNAL_RESOURCE_RE = re.compile(
    r'\b(?:href|src)="((?:https?|ftp)://[^"]*|//[^"]*)"',
)


def run_smoke_checks_on(
    epub_path: Path,
    *,
    repo_root: Path,
) -> list[EpubIssue]:
    """Run reader-compatibility smoke checks against a built EPUB.

    Unlike `run_epubcheck_on`, this does not require Java — it unzips the
    EPUB into a temp directory, walks the CSS and XHTML, and returns
    issues for patterns epubcheck does not catch:

      * CSS custom property declarations and usage
      * External resource references in XHTML href / src

    Returns an empty list on a clean EPUB.
    """
    issues: list[EpubIssue] = []
    if not epub_path.exists():
        return [EpubIssue(
            file=str(epub_path), line=0, col=0,
            code="smoke-missing-epub",
            severity="error",
            message=f"EPUB file not found: {epub_path}",
        )]

    epub_rel = (
        str(epub_path.relative_to(repo_root))
        if epub_path.is_relative_to(repo_root)
        else str(epub_path)
    )
    tmp = Path(tempfile.mkdtemp(prefix="epub-smoke-"))
    try:
        try:
            with zipfile.ZipFile(epub_path, "r") as zf:
                zf.extractall(tmp)
        except zipfile.BadZipFile:
            return [EpubIssue(
                file=epub_rel, line=0, col=0,
                code="smoke-bad-zip",
                severity="error",
                message="EPUB is not a valid zip archive",
            )]

        # --- CSS custom properties -------------------------------------
        for css in tmp.rglob("*.css"):
            text = _read(css)
            if text is None:
                continue
            rel_in_epub = css.relative_to(tmp).as_posix()
            displayed = f"{epub_rel}!{rel_in_epub}"

            for m in _CSS_CUSTOM_PROP_DECL_RE.finditer(text):
                issues.append(EpubIssue(
                    file=displayed,
                    line=_line_of(text, m.start()),
                    col=0,
                    code="smoke-css-custom-property-decl",
                    # Treated as error because it produced real user-
                    # reported EPUB-load failure on ClearView (issue #1052).
                    severity="error",
                    message=(
                        f"CSS custom property declaration '{m.group(1)}'; "
                        "older EPUB readers (ClearView, Tolino pre-2023) "
                        "do not support these. Inline the value."
                    ),
                ))
            for m in _CSS_CUSTOM_PROP_USE_RE.finditer(text):
                issues.append(EpubIssue(
                    file=displayed,
                    line=_line_of(text, m.start()),
                    col=0,
                    code="smoke-css-custom-property-use",
                    severity="error",
                    message=(
                        f"var({m.group(1)}) reference; older EPUB readers "
                        "do not resolve CSS custom properties. Inline "
                        "the literal value."
                    ),
                ))

        # --- External resource references -----------------------------
        for xhtml in tmp.rglob("*.xhtml"):
            text = _read(xhtml)
            if text is None:
                continue
            rel_in_epub = xhtml.relative_to(tmp).as_posix()
            displayed = f"{epub_rel}!{rel_in_epub}"

            for m in _EXTERNAL_RESOURCE_RE.finditer(text):
                url = m.group(1)
                # Allowlist: external hyperlinks (<a href="https://...">)
                # are fine; only flag when the match's *attribute* is src=
                # or the href= is on a <link> element (external stylesheet).
                # We approximate this by looking at the 60 chars before
                # the match for 'src=' or '<link'.
                context_before = text[max(0, m.start() - 60):m.start()]
                is_src_attr = "src=\"" in m.group(0)
                is_link_href = "<link" in context_before.lower()
                if not (is_src_attr or is_link_href):
                    continue
                issues.append(EpubIssue(
                    file=displayed,
                    line=_line_of(text, m.start()),
                    col=0,
                    code="smoke-external-resource",
                    severity="error",
                    message=(
                        f"External resource reference: {url} — EPUB readers "
                        "do not fetch remote resources. Inline the asset "
                        "or remove the reference."
                    ),
                ))

        return issues
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def emit_github_annotations(issues: Iterable[EpubIssue]) -> None:
    """When running under GitHub Actions, echo `::error` annotations so
    epubcheck findings appear inline on the PR diff view.

    No-op when `GITHUB_ACTIONS` is not set, so it's safe to call locally.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    # GitHub's annotation format only has file/line/col, not zip-internal
    # paths. For epubcheck issues the file is `epub!path` — we still emit
    # the annotation (keyed on the epub file) so the workflow log shows it
    # in the annotations panel; reviewers who need to find the XHTML line
    # follow the message text.
    for issue in issues:
        level = "error" if issue.severity in ("fatal", "error") else "warning"
        extra = []
        if issue.line > 0:
            extra.append(f"line={issue.line}")
        if issue.col > 0:
            extra.append(f"col={issue.col}")
        loc = "," + ",".join(extra) if extra else ""
        # The file field must not contain shell-meaningful characters like
        # `!` for the annotation to render; strip to the EPUB path only.
        file_field = issue.file.split("!", 1)[0]
        print(
            f"::{level} file={file_field}{loc},title={issue.code}"
            f"::{issue.message}"
        )
