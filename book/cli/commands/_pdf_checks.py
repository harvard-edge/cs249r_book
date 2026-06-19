"""PDF post-build verification used by `binder build pdf` and `binder check pdf`.

Scans rendered PDF text (via ``pdftotext``) for defects Quarto/LuaLaTeX can
emit without failing the render: unresolved cross-refs (``?@sec-foo``), literal
unrendered cross-refs (``@sec-foo``), undefined LaTeX references
(``Figure ??``), leaked Python tracebacks, and matplotlib/Python
``UserWarning`` text that escaped into the rendered output.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PDF_BY_VOLUME = {
    "vol1": "Machine-Learning-Systems-Vol1.pdf",
    "vol2": "Machine-Learning-Systems-Vol2.pdf",
}

XREF_KINDS = r"(?:sec|fig|tbl|eq|lst|alg)"
RESIDUAL_XREF = re.compile(rf"\?@({XREF_KINDS}-[\w.-]+)")
BARE_XREF = re.compile(rf"(?<![?\w])@({XREF_KINDS}-[\w.-]+)")
LATEX_UNDEF = re.compile(r"\b(?:Figure|Table|Section|Equation|Listing)\s+\?\?+")
MALFORMED_OBJECT_NUMBER = re.compile(
    r"\b(?:Table|Figure|Listing|Algorithm|Lighthouse|Example|War Story|"
    r"Systems Perspective)\s+(\d+\.\d+\.\d+(?:\.\d+)*)\b"
)
TABLE_CAPTION = re.compile(r"\bTable\s+(\d+\.\d+):\s+([^\n]+)")
NUMBERING_ISSUE_CODES = {
    "malformed-object-number",
    "duplicate-table-number",
}
TABLE_PROSE_MIN_GAP_PT = 6.0
PYTHON_LEAK = re.compile(
    r"(?:Traceback \(most recent call last\)|"
    r"NameError:|AttributeError:|ModuleNotFoundError:|Error rendering|"
    r"UserWarning:)",
    re.I,
)
QUARTO_XREF_WARN = re.compile(
    r"Unable to resolve crossref (@(?:sec|fig|tbl|eq|lst)-[\w.-]+)"
)
OVERFULL_HBOX = re.compile(
    r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\)"
)
# Vertical overflow: content (tall table/callout, or an overrunning margin note)
# exceeded the available vertical space. LuaLaTeX emits these during \output, so
# they carry a shipped-out page number "[N]" rather than a source "lines N--M".
OVERFULL_VBOX = re.compile(
    r"Overfull \\vbox \((\d+(?:\.\d+)?)pt too high\)[^\n\[]*(?:\[(\d+)\])?"
)


@dataclass(frozen=True)
class PdfIssue:
    code: str
    message: str
    count: int = 1
    # "error" issues fail the build; "warning" issues print but do not (layout
    # polish — e.g. vertical/margin overflow — should surface without blocking).
    severity: str = "error"


@dataclass
class PdfCheckItem:
    check_id: str
    label: str
    passed: bool
    skipped: bool = False
    detail: str = ""
    # A warning-level check shows ⚠ when it fails but does NOT flip result.ok.
    is_warning: bool = False


@dataclass
class PdfValidationResult:
    volume: str
    pdf_path: Path
    issues: list[PdfIssue] = field(default_factory=list)
    checks: list[PdfCheckItem] = field(default_factory=list)
    margin_geometry: object | None = None

    @property
    def ok(self) -> bool:
        no_errors = not any(i.severity == "error" for i in self.issues)
        checks_ok = all(c.passed or c.skipped or c.is_warning for c in self.checks)
        return no_errors and checks_ok


def default_pdf_path(quarto_dir: Path, volume: str) -> Path:
    return quarto_dir / "_build" / f"pdf-{volume}" / PDF_BY_VOLUME[volume]


def _pdftotext(pdf_path: Path) -> str:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pdftotext failed on {pdf_path}: {(proc.stderr or proc.stdout or '').strip()}"
        )
    return proc.stdout or ""


def _x_overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))


def _pdf_line_text(chars: list[dict]) -> str:
    """Reconstruct a line of PDF text with approximate word gaps."""
    if not chars:
        return ""
    parts: list[str] = []
    prev: dict | None = None
    for char in chars:
        if prev is not None:
            gap = float(char.get("x0", 0.0) or 0.0) - float(prev.get("x1", 0.0) or 0.0)
            size = max(
                float(char.get("size", 0.0) or 0.0),
                float(prev.get("size", 0.0) or 0.0),
                6.0,
            )
            if gap > max(1.2, size * 0.25):
                parts.append(" ")
        parts.append(str(char.get("text", "")))
        prev = char
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _pdf_main_text_lines(page) -> list[dict]:
    """Cluster main-flow PDF chars into line boxes."""
    page_h = float(getattr(page, "height", 0.0) or 0.0)
    page_w = float(getattr(page, "width", 0.0) or 0.0)
    if page_h <= 0 or page_w <= 0:
        return []
    header_y = page_h * 0.06
    footer_y = page_h * 0.94
    main_x = page_w * 0.88
    groups: list[list[dict]] = []
    for char in sorted(getattr(page, "chars", []) or [], key=lambda c: (c["top"], c["x0"])):
        if not (header_y < char.get("top", 0) < footer_y):
            continue
        if char.get("x0", 0) > main_x:
            continue
        for group in groups:
            if abs(group[0]["top"] - char["top"]) < 2.0:
                group.append(char)
                break
        else:
            groups.append([char])

    lines: list[dict] = []
    for group in groups:
        group = sorted(group, key=lambda c: c["x0"])
        text = _pdf_line_text(group)
        if not text:
            continue
        lines.append(
            {
                "text": text,
                "x0": min(c["x0"] for c in group),
                "x1": max(c["x1"] for c in group),
                "top": min(c["top"] for c in group),
                "bottom": max(c["bottom"] for c in group),
                "size": sum(float(c.get("size", 0.0) or 0.0) for c in group) / len(group),
            }
        )
    return lines


def _pdf_horizontal_rules(page) -> list[dict]:
    """Return rendered horizontal rules in the main text area."""
    page_h = float(getattr(page, "height", 0.0) or 0.0)
    page_w = float(getattr(page, "width", 0.0) or 0.0)
    if page_h <= 0 or page_w <= 0:
        return []
    header_y = page_h * 0.06
    footer_y = page_h * 0.94
    main_x = page_w * 0.88
    rules: list[dict] = []
    for line in getattr(page, "lines", []) or []:
        x0 = float(line.get("x0", 0.0) or 0.0)
        x1 = float(line.get("x1", 0.0) or 0.0)
        top = float(line.get("top", 0.0) or 0.0)
        height = abs(float(line.get("height", 0.0) or 0.0))
        width = x1 - x0
        if width < 120 or height > 1.0:
            continue
        if not (header_y < top < footer_y):
            continue
        if x0 > main_x:
            continue
        rules.append({"x0": x0, "x1": x1, "top": top, "bottom": top, "width": width})
    return rules


def _pdf_table_rule_clusters(page) -> list[list[dict]]:
    """Return horizontal-rule clusters that look like booktabs tables."""
    rules = _pdf_horizontal_rules(page)
    groups: list[list[dict]] = []
    for rule in sorted(rules, key=lambda r: (r["x0"], r["x1"], r["top"])):
        for group in groups:
            if abs(group[0]["x0"] - rule["x0"]) <= 6.0 and abs(group[0]["x1"] - rule["x1"]) <= 6.0:
                group.append(rule)
                break
        else:
            groups.append([rule])

    return [
        sorted(group, key=lambda r: r["top"])
        for group in groups
        if len(group) >= 3
    ]


def _line_is_between_table_rules(
    line: dict,
    bottom_rule: dict,
    rules: list[dict],
) -> bool:
    """Return true when the next line is probably a table row, not prose."""
    for later in rules:
        if later["top"] <= bottom_rule["top"] + 1.0:
            continue
        overlap = _x_overlap(later, bottom_rule)
        min_width = min(float(later.get("width", 0.0) or 0.0), float(bottom_rule.get("width", 0.0) or 0.0))
        if overlap < max(80.0, min_width * 0.55):
            continue
        # Header/body rules are often separated by one table row. If the next
        # text line sits in that band, the rule is internal to the table.
        if line["top"] < later["top"] and later["top"] - line["top"] <= 90.0:
            return True
    return False


def _looks_like_post_table_prose(text: str) -> bool:
    """Filter out tick labels, legends, and table rows from spacing warnings."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return False
    if re.match(r"^(?:Table|Figure|Listing|Algorithm)\s+\d", normalized):
        return False
    if re.match(r"^[\d.,%~$×+−–—<>=/()\\s-]+$", normalized):
        return False
    if re.match(r"^\d", normalized):
        return False
    label_prefixes = (
        "Challenge:",
        "Consequence:",
        "Context:",
        "Failure mode:",
        "Given:",
        "Insight:",
        "Lesson:",
        "Problem:",
        "Resolution:",
        "Systems insight:",
        "Systemsinsight:",
        "Systems lesson:",
        "Systemslesson:",
        "Takeaway:",
        "Why it matters:",
    )
    if normalized.startswith(label_prefixes):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", normalized)
    if len(words) < 7:
        return False
    digit_chars = sum(ch.isdigit() for ch in normalized)
    if digit_chars > len(normalized) * 0.35:
        return False
    return bool(re.search(r"[a-z]", normalized))


def scan_pdf_text(pdf_path: Path) -> list[PdfIssue]:
    """Return issues found in extracted PDF text."""
    body = _pdftotext(pdf_path)
    issues: list[PdfIssue] = []

    xref_counts = Counter(m.group(1) for m in RESIDUAL_XREF.finditer(body))
    for slug, count in sorted(xref_counts.items()):
        issues.append(
            PdfIssue(
                code="unresolved-crossref",
                message=f"?@{slug} appears in PDF text (Quarto cross-ref did not resolve)",
                count=count,
            )
        )

    bare_xref_counts = Counter(m.group(1) for m in BARE_XREF.finditer(body))
    for slug, count in sorted(bare_xref_counts.items()):
        issues.append(
            PdfIssue(
                code="literal-crossref",
                message=f"@{slug} appears in PDF text (Quarto cross-ref rendered literally)",
                count=count,
            )
        )

    undef_counts = Counter(m.group(0) for m in LATEX_UNDEF.finditer(body))
    for text, count in sorted(undef_counts.items()):
        issues.append(
            PdfIssue(
                code="undefined-latex-ref",
                message=f'"{text}" appears in PDF text (LaTeX reference undefined)',
                count=count,
            )
        )

    malformed_counts = Counter(m.group(0) for m in MALFORMED_OBJECT_NUMBER.finditer(body))
    for text, count in sorted(malformed_counts.items()):
        issues.append(
            PdfIssue(
                code="malformed-object-number",
                message=(
                    f'"{text}" appears in PDF text. Numbered objects should use '
                    "chapter-local numbering such as Table 8.2, not subsection-shaped "
                    "numbering such as Table 8.2.1."
                ),
                count=count,
            )
        )

    # A renderer/counter regression can make several distinct in-callout tables
    # reuse one number. Repeated occurrences of the same number are fine when the
    # caption text is identical (for example a longtable continuation or repeated
    # list entry). The failure is one table number attached to multiple captions.
    table_captions: dict[str, Counter[str]] = {}
    for match in TABLE_CAPTION.finditer(body):
        number = match.group(1)
        caption = re.sub(r"\s+", " ", match.group(2)).strip()
        if not caption:
            continue
        table_captions.setdefault(number, Counter())[caption] += 1
    duplicate_numbers = {
        number: captions
        for number, captions in table_captions.items()
        if len(captions) > 1
    }
    for number, captions in sorted(duplicate_numbers.items()):
        examples = "; ".join(list(captions.keys())[:3])
        issues.append(
            PdfIssue(
                code="duplicate-table-number",
                message=(
                    f"Table {number} is attached to multiple distinct captions in "
                    f"the PDF text: {examples}"
                ),
                count=sum(captions.values()),
            )
        )

    if PYTHON_LEAK.search(body):
        issues.append(
            PdfIssue(
                code="python-traceback",
                message="Python error/traceback text leaked into PDF output",
            )
        )

    return issues


def scan_build_log(log_path: Path | None) -> list[PdfIssue]:
    """Return cross-ref warnings and overfull hbox alerts from a render log."""
    if log_path is None or not log_path.exists():
        return []
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [PdfIssue(code="log-read-error", message=str(exc))]

    issues: list[PdfIssue] = []
    warn_counts = Counter(m.group(1) for m in QUARTO_XREF_WARN.finditer(text))
    for ref, count in sorted(warn_counts.items()):
        issues.append(
            PdfIssue(
                code="quarto-crossref-warning",
                message=f"{ref} flagged in build log (Unable to resolve crossref)",
                count=count,
            )
        )

    # Overfull hbox: content overflowed the text block (tables, figures, code).
    # Map each warning to a chapter via the companion .tex file.
    SEVERE_THRESHOLD_PT = 20.0
    overfull_line_re = re.compile(
        r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\).*?lines (\d+)"
    )

    # Build chapter map from the .tex file (cover images mark chapter starts)
    chapter_map: list[tuple[int, str]] = []
    tex_path = None
    if log_path:
        # Try the common names for the generated .tex alongside the log
        for candidate in (
            log_path.with_suffix(".tex"),
            log_path.parent / "Machine-Learning-Systems-Vol1.tex",
            log_path.parent / "Machine-Learning-Systems-Vol2.tex",
        ):
            if candidate.is_file():
                tex_path = candidate
                break
    if tex_path and tex_path.is_file():
        cover_re = re.compile(r"contents/vol[12]/([^/]+)/images/png/cover_")
        try:
            for i, line in enumerate(tex_path.read_text(errors="replace").splitlines(), 1):
                m = cover_re.search(line)
                if m:
                    chapter_map.append((i, m.group(1)))
        except OSError:
            pass

    def _chapter_for_tex_line(line_num: int) -> str:
        if not chapter_map:
            return "unknown"
        result = chapter_map[0][1]
        for start, name in chapter_map:
            if line_num >= start:
                result = name
            else:
                break
        return result

    severe_items: list[tuple[float, str]] = []
    for m in overfull_line_re.finditer(text):
        pts = float(m.group(1))
        if pts >= SEVERE_THRESHOLD_PT:
            tex_line = int(m.group(2))
            chapter = _chapter_for_tex_line(tex_line)
            severe_items.append((pts, chapter))

    if severe_items:
        by_chapter: dict[str, list[float]] = {}
        for pts, ch in severe_items:
            by_chapter.setdefault(ch, []).append(pts)
        details = "; ".join(
            f"{ch}: {len(pts_list)} overflow(s), worst {max(pts_list):.0f}pt"
            for ch, pts_list in sorted(by_chapter.items(), key=lambda x: -max(x[1]))
        )
        worst = max(pts for pts, _ in severe_items)
        issues.append(
            PdfIssue(
                code="overfull-hbox",
                message=(
                    f"{len(severe_items)} overfull \\hbox >= {SEVERE_THRESHOLD_PT:.0f}pt "
                    f"(worst: {worst:.0f}pt). By chapter: {details}"
                ),
                count=len(severe_items),
            )
        )

    # Overfull vbox: vertical overflow — a tall table/callout, or (the case this
    # was added for) a margin note/figure that overran its column and pushed off
    # the page. Severe rows are blocking because they correspond to rendered
    # content exceeding the available page geometry.
    vbox_items: list[tuple[float, str]] = []
    for m in OVERFULL_VBOX.finditer(text):
        pts = float(m.group(1))
        if pts >= SEVERE_THRESHOLD_PT:
            vbox_items.append((pts, m.group(2) or "?"))

    if vbox_items:
        worst = max(pts for pts, _ in vbox_items)
        pages = sorted({p for _, p in vbox_items if p != "?"}, key=int)
        page_note = f" pages: {', '.join(pages)}." if pages else ""
        issues.append(
            PdfIssue(
                code="overfull-vbox",
                message=(
                    f"{len(vbox_items)} overfull \\vbox >= {SEVERE_THRESHOLD_PT:.0f}pt "
                    f"(worst: {worst:.0f}pt).{page_note} "
                    f"Run `binder layout margins <pdf>` to localize margin overflow."
                ),
                count=len(vbox_items),
            )
        )

    return issues


def scan_table_prose_spacing(
    pdf_path: Path,
    *,
    min_gap_pt: float = TABLE_PROSE_MIN_GAP_PT,
) -> list[PdfIssue]:
    """Warn when rendered table bottom rules are jammed against following text."""
    if not pdf_path.is_file():
        return []
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return []

    findings: list[tuple[int, float, str]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                text_lines = _pdf_main_text_lines(page)
                if not text_lines:
                    continue
                rules = _pdf_horizontal_rules(page)
                for cluster in _pdf_table_rule_clusters(page):
                    bottom_rule = cluster[-1]
                    after = [
                        line for line in text_lines
                        if line["top"] > bottom_rule["top"] + 0.5
                        and _x_overlap(line, bottom_rule) >= 20.0
                    ]
                    if not after:
                        continue
                    next_line = min(after, key=lambda line: line["top"])
                    # Plot gridlines can look like table rules to PDF geometry;
                    # their next "line" is often numeric tick labels. This
                    # check is for prose/semantic labels jammed after tables.
                    if float(next_line.get("size", 0.0) or 0.0) < 8.5:
                        continue
                    if not _looks_like_post_table_prose(next_line["text"]):
                        continue
                    if _line_is_between_table_rules(next_line, bottom_rule, rules):
                        continue
                    gap = next_line["top"] - bottom_rule["top"]
                    if 0 < gap < min_gap_pt:
                        findings.append((page_idx, gap, next_line["text"][:80]))
    except Exception as exc:
        return [
            PdfIssue(
                code="table-prose-spacing-scan-error",
                message=f"Could not scan rendered table/prose spacing: {exc}",
                severity="warning",
            )
        ]

    if not findings:
        return []

    examples = "; ".join(
        f"sheet {page}: {gap:.1f}pt before \"{snippet}\""
        for page, gap, snippet in findings[:3]
    )
    return [
        PdfIssue(
            code="table-prose-crowding",
            message=(
                f"{len(findings)} rendered table bottom rule(s) are followed "
                f"by text with less than {min_gap_pt:.1f}pt of spacing. "
                f"Examples: {examples}"
            ),
            count=len(findings),
            severity="warning",
        )
    ]


def verify_pdf(
    pdf_path: Path,
    *,
    log_path: Path | None = None,
) -> list[PdfIssue]:
    """Scan ``pdf_path`` (and optional ``log_path``) and return all issues."""
    if not pdf_path.is_file():
        return [PdfIssue(code="missing-pdf", message=f"PDF not found: {pdf_path}")]

    if shutil.which("pdftotext") is None:
        return [
            PdfIssue(
                code="pdftotext-missing",
                message="pdftotext not installed; install poppler (e.g. brew install poppler)",
            )
        ]

    issues = scan_pdf_text(pdf_path)
    issues.extend(scan_table_prose_spacing(pdf_path))
    issues.extend(scan_build_log(log_path))
    return issues


def scan_pdf_numbering(pdf_path: Path) -> list[PdfIssue]:
    """Return rendered-PDF object-numbering regressions only.

    This narrower scan is useful after chapter-only PDF builds, where ordinary
    cross-reference checks can fail because other chapters are intentionally
    absent from the temporary render.
    """
    if not pdf_path.is_file():
        return [PdfIssue(code="missing-pdf", message=f"PDF not found: {pdf_path}")]

    if shutil.which("pdftotext") is None:
        return [
            PdfIssue(
                code="pdftotext-missing",
                message="pdftotext not installed; install poppler (e.g. brew install poppler)",
            )
        ]

    return [
        issue
        for issue in scan_pdf_text(pdf_path)
        if issue.code in NUMBERING_ISSUE_CODES
    ]


def scan_margin_geometry_summary(pdf_path: Path):
    """Return the Binder-native margin geometry summary, or ``None``.

    ``None`` means PyMuPDF is unavailable or the scan failed; callers should
    treat that as skipped rather than as a clean geometry result.
    """
    try:
        try:
            from cli.checks.margin_geometry import scan_pdf
        except ImportError:
            from book.cli.checks.margin_geometry import scan_pdf
    except ImportError:
        return None
    try:
        return scan_pdf(pdf_path)
    except Exception:
        return None


def scan_margin_geometry(pdf_path: Path):
    """Return ``(overlaps, overflows)`` from rendered margin geometry."""
    summary = scan_margin_geometry_summary(pdf_path)
    if summary is None:
        return None
    return summary.overlaps, summary.overflows


def verify_volume_pdf(
    quarto_dir: Path,
    volume: str,
    *,
    log_path: Path | None = None,
) -> PdfValidationResult:
    """Verify the default built PDF for ``volume`` (``vol1`` or ``vol2``)."""
    pdf_path = default_pdf_path(quarto_dir, volume)
    issues = verify_pdf(pdf_path, log_path=log_path)
    geom_summary = scan_margin_geometry_summary(pdf_path) if pdf_path.is_file() else None
    geom = (
        (geom_summary.overlaps, geom_summary.overflows)
        if geom_summary is not None
        else None
    )

    checks = [
        PdfCheckItem("artifact", "PDF artifact exists", pdf_path.is_file()),
        PdfCheckItem(
            "pdftotext",
            "pdftotext available",
            shutil.which("pdftotext") is not None,
            skipped=shutil.which("pdftotext") is None,
        ),
        PdfCheckItem(
            "unresolved-crossref",
            "No ?@sec/fig/tbl/eq/lst/alg literals in PDF text",
            not any(i.code == "unresolved-crossref" for i in issues),
        ),
        PdfCheckItem(
            "literal-crossref",
            "No bare @sec/fig/tbl/eq/lst/alg literals in PDF text",
            not any(i.code == "literal-crossref" for i in issues),
        ),
        PdfCheckItem(
            "undefined-latex-ref",
            "No Figure/Table/Section ?? in PDF text",
            not any(i.code == "undefined-latex-ref" for i in issues),
        ),
        PdfCheckItem(
            "malformed-object-number",
            "No subsection-shaped object numbers such as Table 8.2.1",
            not any(i.code == "malformed-object-number" for i in issues),
        ),
        PdfCheckItem(
            "duplicate-table-number",
            "No duplicate table numbers with different captions",
            not any(i.code == "duplicate-table-number" for i in issues),
        ),
        PdfCheckItem(
            "table-prose-crowding",
            "No cramped text immediately after rendered table rules",
            not any(i.code == "table-prose-crowding" for i in issues),
            is_warning=True,
        ),
        PdfCheckItem(
            "python-traceback",
            "No Python tracebacks or UserWarnings in PDF text",
            not any(i.code == "python-traceback" for i in issues),
        ),
        PdfCheckItem(
            "quarto-crossref-warning",
            "No Quarto crossref warnings in build log",
            not any(i.code == "quarto-crossref-warning" for i in issues),
            skipped=log_path is None,
        ),
        PdfCheckItem(
            "overfull-hbox",
            "No severe layout overflow (Overfull hbox >= 20pt)",
            not any(i.code == "overfull-hbox" for i in issues),
            skipped=log_path is None,
        ),
        PdfCheckItem(
            "overfull-vbox",
            "No vertical/margin overflow (Overfull vbox >= 20pt)",
            not any(i.code == "overfull-vbox" for i in issues),
            skipped=log_path is None,
        ),
        PdfCheckItem(
            "margin-geometry",
            (
                f"Margin geometry: {geom[0]} overlap(s), {geom[1]} overflow "
                "(PyMuPDF; run `binder layout overlaps` to localize)"
                if geom is not None
                else "Margin geometry scan (PyMuPDF unavailable)"
            ),
            geom is None or geom[0] == 0,
            skipped=geom is None,
            is_warning=True,
        ),
    ]

    return PdfValidationResult(
        volume=volume,
        pdf_path=pdf_path,
        issues=issues,
        checks=checks,
        margin_geometry=geom_summary,
    )


def format_checklist(result: PdfValidationResult) -> str:
    """Human-readable checklist for console output."""
    vol_label = "Volume I" if result.volume == "vol1" else "Volume II"
    lines = [f"PDF validation ({vol_label}): {result.pdf_path.name}", ""]
    for item in result.checks:
        if item.skipped:
            mark, status = "ⓘ", "skipped"
        elif item.passed:
            mark, status = "✓", "pass"
        elif item.is_warning:
            mark, status = "⚠", "warn"
        else:
            mark, status = "✗", "fail"
        detail = f" — {item.detail}" if item.detail else ""
        lines.append(f"  [{mark}] {item.label} ({status}){detail}")

    errors = [i for i in result.issues if i.severity != "warning"]
    warnings = [i for i in result.issues if i.severity == "warning"]

    if errors:
        lines.extend(["", "Issues:"])
        for idx, issue in enumerate(errors, start=1):
            suffix = f" ({issue.count} occurrence(s))" if issue.count > 1 else ""
            lines.append(f"  {idx}. [{issue.code}] {issue.message}{suffix}")

    if warnings:
        lines.extend(["", "Warnings (non-blocking):"])
        for idx, issue in enumerate(warnings, start=1):
            suffix = f" ({issue.count} occurrence(s))" if issue.count > 1 else ""
            lines.append(f"  {idx}. ⚠ [{issue.code}] {issue.message}{suffix}")
    return "\n".join(lines)


def format_failure_report(label: str, pdf_path: Path, issues: list[PdfIssue]) -> str:
    lines = [
        f"PDF validation failed for {label}:",
        f"  artifact: {pdf_path}",
        "",
    ]
    for idx, issue in enumerate(issues, start=1):
        suffix = f" ({issue.count} occurrence(s))" if issue.count > 1 else ""
        lines.append(f"  {idx}. [{issue.code}] {issue.message}{suffix}")
    lines.extend(
        [
            "",
            "Fix the source QMD (missing {#tbl-...} label, broken @ref, blank lines",
            "inside pipe tables, etc.) and rebuild. Bypass with --skip-validate only",
            "when you intentionally need a broken artifact.",
        ]
    )
    return "\n".join(lines)
