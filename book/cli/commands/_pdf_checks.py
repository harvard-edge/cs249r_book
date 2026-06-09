"""PDF post-build verification used by `binder build pdf` and `binder check pdf`.

Scans rendered PDF text (via ``pdftotext``) for defects Quarto/LuaLaTeX can
emit without failing the render: unresolved cross-refs (``?@sec-foo``),
undefined LaTeX references (``Figure ??``), leaked Python tracebacks, and
matplotlib/Python ``UserWarning`` text that escaped into the rendered output.
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

RESIDUAL_XREF = re.compile(r"\?@((?:sec|fig|tbl|eq|lst)-[\w.-]+)")
LATEX_UNDEF = re.compile(r"\b(?:Figure|Table|Section|Equation|Listing)\s+\?\?+")
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

    undef_counts = Counter(m.group(0) for m in LATEX_UNDEF.finditer(body))
    for text, count in sorted(undef_counts.items()):
        issues.append(
            PdfIssue(
                code="undefined-latex-ref",
                message=f'"{text}" appears in PDF text (LaTeX reference undefined)',
                count=count,
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
    # the page. Warning-level: surfaced in the build but non-blocking. Keyed by
    # shipped-out page number rather than source line.
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
                severity="warning",
            )
        )

    return issues


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
    issues.extend(scan_build_log(log_path))
    return issues


def scan_margin_geometry(pdf_path: Path):
    """True-geometry margin scan via the PyMuPDF oracle (book/tools/audit/
    margin_geometry.py). Returns ``(overlaps, overflows)`` or ``None`` when
    PyMuPDF is unavailable. Mirrors ``binder layout overlaps``; used as a
    non-blocking build-time warning so every validated build reports margin
    geometry (element-on-element overlap + footer/header overflow) that the
    LaTeX-log overfull checks cannot localize.
    """
    try:
        import fitz  # PyMuPDF
        import importlib.util
        import sys
    except ImportError:
        return None
    try:
        oracle_path = (
            Path(__file__).resolve().parents[2]
            / "tools" / "audit" / "margin_geometry.py"
        )
        spec = importlib.util.spec_from_file_location(
            "margin_geometry", oracle_path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        doc = fitz.open(str(pdf_path))
        findings = []
        for i in range(doc.page_count):
            findings.extend(mod.scan_page(doc[i], i + 1))
        overlaps = sum(1 for f in findings if f.issue == "overlap")
        return overlaps, len(findings) - overlaps
    except Exception:
        return None


def verify_volume_pdf(
    quarto_dir: Path,
    volume: str,
    *,
    log_path: Path | None = None,
) -> PdfValidationResult:
    """Verify the default built PDF for ``volume`` (``vol1`` or ``vol2``)."""
    pdf_path = default_pdf_path(quarto_dir, volume)
    issues = verify_pdf(pdf_path, log_path=log_path)
    geom = scan_margin_geometry(pdf_path) if pdf_path.is_file() else None

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
            "No ?@sec/fig/tbl/eq/lst literals in PDF text",
            not any(i.code == "unresolved-crossref" for i in issues),
        ),
        PdfCheckItem(
            "undefined-latex-ref",
            "No Figure/Table/Section ?? in PDF text",
            not any(i.code == "undefined-latex-ref" for i in issues),
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
            is_warning=True,
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
